import pandas as pd
import aiohttp
from typing import List
from pydantic import BaseModel
import json, re
from ast import literal_eval
import argparse, asyncio, math, time
from pathlib import Path
from tqdm import tqdm
import os
import sys

sys.path.append(str(Path(__file__).parent.parent))
from profiles import ArtistProfile

# ──────────────────────────────── RATE-LIMIT SETTINGS ─────────────────────────
RATE_LIMIT_RPM   = 500                  # <-- edit this if your quota changes
SAFETY_FACTOR    = 0.75                 # 25 % head-room
RPS              = RATE_LIMIT_RPM / 60  # requests per second

# Better concurrency calculation that works with lower rate limits
# Allow reasonable concurrency even with low RPM, but cap it to avoid overwhelming
if RATE_LIMIT_RPM >= 100:
    MAX_CONCURRENCY = max(1, int(RPS * SAFETY_FACTOR))
else:
    # For lower rate limits, use a more generous formula
    MAX_CONCURRENCY = 4 # max(5, min(20, int(RATE_LIMIT_RPM / 10)))

MAX_RETRIES      = 3
RETRY_BACKOFF_SEC = max(1, int(60 / RATE_LIMIT_RPM * 2))  # Reduce from *10 to *2

print(f"Rate limit settings: {RATE_LIMIT_RPM} RPM, {MAX_CONCURRENCY} concurrent requests")

FENCE_RE = re.compile(r"^```(?:\w+)?\s*([\s\S]*?)\s*```$", re.DOTALL)

# ──────────────────────────────── ARGPARSE ────────────────────────────────
parser = argparse.ArgumentParser(
    description="Batch-generate artist profiles using Perplexity Sonar API."
)
parser.add_argument("-i", "--input",  required=True,
                    help="Path to input JSON file of artists")
parser.add_argument("-o", "--output", required=True,
                    help="Path to output JSONL file")
parser.add_argument("-n", "--num_entries", type=int, default=None,
                    help="Process only the first N entries (testing)")
parser.add_argument("-l", "--log", default=None,
                    help="Path to raw API response log file (defaults to <output>.raw.jsonl)")
parser.add_argument("-p", "--prompt", required=True,
                        help="Path to prompt template txt file (variable {{artist}} will be replaced)")
parser.add_argument("--perplexity-api-key", default=None,
                        help="Override the PERPLEXITY_API_KEY environment variable")

cli_args = parser.parse_args()

API_KEY = cli_args.perplexity_api_key or os.getenv("PERPLEXITY_API_KEY")
if not API_KEY:
    raise Exception("PERPLEXITY_API_KEY environment variable not set or empty")

# ─────────────────────────────── PROMPT TEMPLATE ──────────────────────────────
PROMPT_TEMPLATE = Path(cli_args.prompt).read_text(encoding="utf-8").strip()



def get_formatted_prompt(artist):
    prompt = PROMPT_TEMPLATE.replace("{{artist}}", artist)
    return prompt


def parse_llm_payload(raw: str) -> dict:
    """
    Convert Perplexity Sonar's JSON-ish string to a Python dict.
    Raises ValueError if we still can't decode.
    """
    # 1) Strip surrounding quotes if we already got a Python str repr
    if (raw.startswith("'") and raw.endswith("'")) or \
       (raw.startswith('"') and raw.endswith('"')):
        raw = raw[1:-1]

    # 2) Remove ``` fences (keep inner group if Regex matches)
    m = FENCE_RE.match(raw.strip())
    cleaned = m.group(1) if m else raw

    # 3) Try strict JSON first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 4) Try once more after literal_eval unescaping
        try:
            return json.loads(literal_eval(cleaned))
        except Exception as e:
            raise ValueError(f"Cannot decode payload: {e}")


async def get_response(session: aiohttp.ClientSession, prompt: str):
    """Async version of get_response using aiohttp"""
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "response_format": { 
            "type": "json_schema", 
            "json_schema": {"schema": ArtistProfile.model_json_schema()} }
    }

    async with session.post(url, headers=headers, json=payload) as response:
        if response.status != 200:
            # Try to get response text for better error reporting
            try:
                error_text = await response.text()
                raise Exception(f"API request failed with status {response.status}: {error_text[:500]}")
            except:
                raise Exception(f"API request failed with status {response.status}")
        
        response_json = await response.json()
        raw_response = response_json["choices"][0]['message']['content']
        parsed_response = parse_llm_payload(raw_response)
        return parsed_response, response_json


# ──────────────────────────────── ASYNC WORKER ────────────────────────────────
sem = asyncio.Semaphore(MAX_CONCURRENCY)

async def generate_artist_profile(session: aiohttp.ClientSession, artist: str) -> tuple:
    """Generate profile for a single artist with error handling and retries"""
    prompt = get_formatted_prompt(artist)
    
    for attempt in range(1, MAX_RETRIES + 1):
        async with sem:  # limits concurrent requests
            try:
                # Debug: Show concurrent request activity (only first few to avoid spam)
                if not hasattr(generate_artist_profile, '_debug_count'):
                    generate_artist_profile._debug_count = 0
                if generate_artist_profile._debug_count < MAX_CONCURRENCY:
                    print(f"[DEBUG] Starting request #{generate_artist_profile._debug_count + 1} for '{artist}'")
                    generate_artist_profile._debug_count += 1
                
                parsed_response, raw_response = await get_response(session, prompt)
                return "success", parsed_response, raw_response, None, artist, prompt
                
            except Exception as err:
                if attempt == MAX_RETRIES:
                    error_msg = f"Failed after {MAX_RETRIES} attempts: {str(err)}"
                    print(f"[ERROR] Artist '{artist}': {error_msg}")
                    return "error", None, None, error_msg, artist, prompt
                
                wait = RETRY_BACKOFF_SEC * attempt
                print(f"[retry {attempt}/{MAX_RETRIES}] {artist}: {err} → sleeping {wait}s")
                await asyncio.sleep(wait)


async def main(args) -> None:
    # Load the data from JSON file
    with open(args.input, 'r') as f:
        artists = json.load(f)
    
    
    if args.num_entries:
        artists = artists[:args.num_entries]
    
    print(f"Loaded {len(artists)} artists from input file...")
    
    # Check for existing results and filter out already-processed songs
    output_path = Path(args.output)
    already_processed = set()
    
    if output_path.exists():
        print(f"Found existing output file: {output_path}")
        print("Reading existing results to resume from where we left off...")
        
        try:
            with output_path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        result = json.loads(line.strip())
                        if 'original_artist' in result:
                            already_processed.add(result['original_artist'])
                        else:
                            # Fallback for older format without original_ fields
                            if 'artist' in result:
                                already_processed.add(result['artist'])
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed line {line_num} in existing output: {e}")
                        continue
            
            print(f"Found {len(already_processed)} already-processed artists")
            
            # Filter out already-processed artists
            original_count = len(artists)
            artists_before = artists.copy()  # Keep original for debugging
            artists = [artist for artist in artists if artist not in already_processed]
            skipped_count = original_count - len(artists)
            
            print(f"Skipping {skipped_count} already-processed artists")
            
            # Show a few examples of skipped artists for verification
            if skipped_count > 0:
                skipped_artists = [artist for artist in artists_before if artist in already_processed]
                examples = skipped_artists[:3]
                print("  Examples of skipped artists:")
                for artist in examples:
                    print(f"    - '{artist}'")
                if skipped_count > 3:
                    print(f"    ... and {skipped_count - 3} more")
            
            print(f"Remaining artists to process: {len(artists)}")
            
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")
            print("Starting fresh (will overwrite existing file)")
            already_processed = set()
    
    if len(artists) == 0:
        print("No artists to process - all artists have already been completed!")
        return

    # If directory of output path doesn't exist, create it
    print(f"Creating directory for output path: {output_path.parent}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sanity check: print first formatted prompt
    first_artist = artists[0]
    sample_prompt = get_formatted_prompt(first_artist)
    print(f"\n{'='*60}")
    print(f"SAMPLE FORMATTED PROMPT (Artist: '{first_artist}'):")
    print(f"{'='*60}")
    print(sample_prompt)
    print(f"{'='*60}\n")
    
    # Determine log file path
    log_path = Path(args.log) if args.log else Path(f"{args.output}.raw.jsonl")
    
    # Create aiohttp session
    timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY)
    
    success_count = 0
    error_count = 0
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Open output file in append mode to preserve existing results
        file_mode = "a" if output_path.exists() and len(already_processed) > 0 else "w"
        log_mode = "a" if log_path.exists() and len(already_processed) > 0 else "w"
        
        with output_path.open(file_mode, encoding="utf-8") as sink, \
             log_path.open(log_mode, encoding="utf-8") as raw_sink:
            
            # Create tasks for all songs
            tasks = [
                asyncio.create_task(
                    generate_artist_profile(session, artist)
                ) 
                for artist in artists
            ]
            
            # Process tasks with progress bar
            with tqdm(total=len(tasks), unit="artist") as pbar:
                for coro in asyncio.as_completed(tasks):
                    status, parsed_response, raw_response, error_msg, artist, prompt = await coro
                    
                    if status == "success":
                        parsed_response['original_artist'] = artist
                        raw_response['original_artist'] = artist
                        raw_response['original_prompt'] = prompt
                        sink.write(json.dumps(parsed_response, ensure_ascii=False) + "\n")
                        raw_sink.write(json.dumps(raw_response, ensure_ascii=False) + "\n")
                        success_count += 1
                    else:
                        error_count += 1
                        # Log error to raw file for debugging
                        error_entry = {
                            "error": error_msg,
                            "artist": artist
                        }
                        raw_sink.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
                    
                    pbar.update(1)
    
    # Report final statistics
    total_processed_this_run = success_count + error_count
    success_rate_this_run = (success_count / total_processed_this_run * 100) if total_processed_this_run > 0 else 0
    total_already_completed = len(already_processed)
    
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Artists processed this run: {total_processed_this_run}")
    print(f"  ├─ Successful: {success_count}")
    print(f"  └─ Failed: {error_count}")
    print(f"Success rate this run: {success_rate_this_run:.1f}%")
    if total_already_completed > 0:
        print(f"Artists already completed (skipped): {total_already_completed}")
        total_overall = total_processed_this_run + total_already_completed
        print(f"Total artists in dataset: {total_overall}")
    print(f"Results saved to: {args.output}")
    print(f"Raw logs saved to: {log_path}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    asyncio.run(main(cli_args))
    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed:.1f}s with "
          f"{MAX_CONCURRENCY}-way concurrency (rate ≤ {RATE_LIMIT_RPM} RPM)")