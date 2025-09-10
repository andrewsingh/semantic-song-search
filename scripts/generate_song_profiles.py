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
from profiles import SongProfile

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

API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not API_KEY:
    raise Exception("PERPLEXITY_API_KEY environment variable not set")

FENCE_RE = re.compile(r"^```(?:\w+)?\s*([\s\S]*?)\s*```$", re.DOTALL)

# ──────────────────────────────── ARGPARSE ────────────────────────────────
parser = argparse.ArgumentParser(
    description="Batch-generate song profiles using Perplexity Sonar API."
)
parser.add_argument("-i", "--input",  required=True,
                    help="Path to input JSON file with array of track objects")
parser.add_argument("-o", "--output", required=True,
                    help="Path to output JSONL file")
parser.add_argument("-n", "--num_entries", type=int, default=None,
                    help="Process only the first N entries (testing)")
parser.add_argument("-l", "--log", default=None,
                    help="Path to raw API response log file (defaults to <output>.raw.jsonl)")
parser.add_argument("-p", "--prompt", required=True,
                        help="Path to prompt template txt file (variables {{song}}, {{artists}}, and {{main_artist}} will be replaced)")

cli_args = parser.parse_args()

# ─────────────────────────────── PROMPT TEMPLATE ──────────────────────────────
PROMPT_TEMPLATE = Path(cli_args.prompt).read_text(encoding="utf-8").strip()



def get_formatted_prompt(song, artists, main_artist):
    prompt = PROMPT_TEMPLATE.replace("{{song}}", song).replace("{{artists}}", artists).replace("{{main_artist}}", main_artist)
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
            "json_schema": {"schema": SongProfile.model_json_schema()} }
    }

    async with session.post(url, headers=headers, json=payload) as response:
        response_json = await response.json()
        
        if response.status != 200:
            raise Exception(f"API request failed with status {response.status}: {response_json}")
        
        raw_response = response_json["choices"][0]['message']['content']
        parsed_response = parse_llm_payload(raw_response)
        return parsed_response, response_json


# ──────────────────────────────── ASYNC WORKER ────────────────────────────────
sem = asyncio.Semaphore(MAX_CONCURRENCY)

async def generate_song_profile(session: aiohttp.ClientSession, song: str, artists: str, main_artist: str, uri: str) -> tuple:
    """Generate profile for a single song with error handling and retries"""
    prompt = get_formatted_prompt(song, artists, main_artist)
    
    for attempt in range(1, MAX_RETRIES + 1):
        async with sem:  # limits concurrent requests
            try:
                # Debug: Show concurrent request activity (only first few to avoid spam)
                current_time = time.time()
                if not hasattr(generate_song_profile, '_debug_count'):
                    generate_song_profile._debug_count = 0
                if generate_song_profile._debug_count < MAX_CONCURRENCY:
                    print(f"[DEBUG] Starting request #{generate_song_profile._debug_count + 1} for '{song}' by '{main_artist}'")
                    generate_song_profile._debug_count += 1
                
                parsed_response, raw_response = await get_response(session, prompt)
                return "success", parsed_response, raw_response, None, song, artists, main_artist, uri, prompt
                
            except Exception as err:
                if attempt == MAX_RETRIES:
                    error_msg = f"Failed after {MAX_RETRIES} attempts: {str(err)}"
                    print(f"[ERROR] Song '{song}' by '{main_artist}': {error_msg}")
                    return "error", None, None, error_msg, song, artists, main_artist, uri, prompt
                
                wait = RETRY_BACKOFF_SEC * attempt
                print(f"[retry {attempt}/{MAX_RETRIES}] {song} by {main_artist}: {err} → sleeping {wait}s")
                await asyncio.sleep(wait)


def validate_track(track):
    """Validate that a track object has all required fields"""
    try:
        # Check basic structure
        if not isinstance(track, dict):
            return False, "Track is not a dictionary"
        
        # Check required fields exist
        if 'name' not in track:
            return False, "Missing 'name' field"
        if 'artists' not in track or not isinstance(track['artists'], list) or len(track['artists']) == 0:
            return False, "Missing or empty 'artists' field"
        if 'uri' not in track:
            return False, "Missing 'uri' field"
        
        # Check that we can extract the required strings
        song = track['name']
        if not isinstance(song, str) or not song.strip():
            return False, "Invalid song name"
        
        artists_list = [artist['name'] for artist in track['artists'] if isinstance(artist, dict) and 'name' in artist]
        if len(artists_list) == 0:
            return False, "No valid artist names found"
        
        if not isinstance(track['artists'][0], dict) or 'name' not in track['artists'][0]:
            return False, "First artist object missing 'name' field"
        
        main_artist = track['artists'][0]['name']
        if not isinstance(main_artist, str) or not main_artist.strip():
            return False, "Invalid main artist name"
        
        uri = track['uri']
        if not isinstance(uri, str) or not uri.strip():
            return False, "Invalid URI"
        
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


async def main(args) -> None:
    # Load the JSON data
    with open(args.input, 'r', encoding='utf-8') as f:
        tracks_data = json.load(f)
    
    if not isinstance(tracks_data, list):
        raise ValueError("Input JSON must be an array of track objects")
    
    print(f"Loaded {len(tracks_data)} tracks from input file...")
    
    # Validate tracks and filter out invalid ones
    valid_tracks = []
    invalid_count = 0
    
    for i, track in enumerate(tracks_data):
        is_valid, error_msg = validate_track(track)
        if is_valid:
            valid_tracks.append(track)
        else:
            invalid_count += 1
            if invalid_count <= 5:  # Show first 5 invalid tracks
                print(f"  Invalid track #{i+1}: {error_msg}")
    
    if invalid_count > 5:
        print(f"  ... and {invalid_count - 5} more invalid tracks")
    
    print(f"Valid tracks: {len(valid_tracks)}")
    print(f"Invalid tracks: {invalid_count}")
    
    if len(valid_tracks) == 0:
        print("No valid tracks found in input file!")
        return
    
    # Apply num_entries limit if specified
    if args.num_entries:
        valid_tracks = valid_tracks[:args.num_entries]
        print(f"Limited to first {len(valid_tracks)} tracks for processing")
    
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
                        # Check for new format with uri
                        if 'uri' in result:
                            already_processed.add(result['uri'])
                        # Fallback for older format
                        elif 'prompt_vars' in result and 'song' in result['prompt_vars'] and 'main_artist' in result['prompt_vars']:
                            # Create a pseudo-key for old format
                            song = result['prompt_vars']['song']
                            artist = result['prompt_vars']['main_artist']
                            already_processed.add(f"legacy:{song}::{artist}")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed line {line_num} in existing output: {e}")
                        continue
            
            print(f"Found {len(already_processed)} already-processed songs")
            
            # Filter out already-processed tracks
            original_count = len(valid_tracks)
            unprocessed_tracks = []
            skipped_count = 0
            
            for track in valid_tracks:
                uri = track['uri']
                # Also check legacy format for backward compatibility
                song = track['name']
                main_artist = track['artists'][0]['name']
                legacy_key = f"legacy:{song}::{main_artist}"
                
                if uri not in already_processed and legacy_key not in already_processed:
                    unprocessed_tracks.append(track)
                else:
                    skipped_count += 1
            
            valid_tracks = unprocessed_tracks
            print(f"Skipping {skipped_count} already-processed songs")
            
            # Show a few examples of skipped songs for verification
            if skipped_count > 0:
                examples_shown = 0
                # Look through original valid_tracks to find skipped ones
                original_valid_tracks = [track for track in tracks_data if validate_track(track)[0]]
                
                for track in original_valid_tracks:
                    if examples_shown >= 3:
                        break
                    try:
                        uri = track['uri']
                        song = track['name']
                        main_artist = track['artists'][0]['name']
                        legacy_key = f"legacy:{song}::{main_artist}"
                        
                        if uri in already_processed or legacy_key in already_processed:
                            if examples_shown == 0:
                                print("  Examples of skipped songs:")
                            print(f"    - '{song}' by '{main_artist}'")
                            examples_shown += 1
                    except:
                        continue
                        
                if skipped_count > 3:
                    print(f"    ... and {skipped_count - 3} more")
            
            print(f"Remaining songs to process: {len(valid_tracks)}")
            
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")
            print("Starting fresh (will overwrite existing file)")
            already_processed = set()
    
    if len(valid_tracks) == 0:
        print("No songs to process - all songs have already been completed!")
        return
    
    # Sanity check: print first formatted prompt
    first_track = valid_tracks[0]
    first_song = first_track['name']
    first_artists = ", ".join([artist['name'] for artist in first_track['artists']])
    first_main_artist = first_track['artists'][0]['name']
    sample_prompt = get_formatted_prompt(first_song, first_artists, first_main_artist)
    print(f"\n{'='*60}")
    print(f"SAMPLE FORMATTED PROMPT (Song: '{first_song}' by {first_artists}):")
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
            tasks = []
            for track in valid_tracks:
                song = track['name']
                artists = ", ".join([artist['name'] for artist in track['artists'] if isinstance(artist, dict) and 'name' in artist])
                main_artist = track['artists'][0]['name']
                uri = track['uri']
                
                task = asyncio.create_task(
                    generate_song_profile(session, song, artists, main_artist, uri)
                )
                tasks.append(task)
            
            # Process tasks with progress bar
            with tqdm(total=len(tasks), unit="song") as pbar:
                for coro in asyncio.as_completed(tasks):
                    status, parsed_response, raw_response, error_msg, song, artists, main_artist, uri, prompt = await coro
                    
                    if status == "success":
                        # Add new metadata structure
                        parsed_response['prompt_vars'] = {
                            "song": song,
                            "artists": artists,
                            "main_artist": main_artist
                        }
                        parsed_response['uri'] = uri
                        
                        raw_response['prompt_vars'] = {
                            "song": song,
                            "artists": artists,
                            "main_artist": main_artist
                        }
                        raw_response['uri'] = uri
                        raw_response['original_prompt'] = prompt
                        
                        sink.write(json.dumps(parsed_response, ensure_ascii=False) + "\n")
                        raw_sink.write(json.dumps(raw_response, ensure_ascii=False) + "\n")
                        success_count += 1
                    else:
                        error_count += 1
                        # Log error to raw file for debugging
                        error_entry = {
                            "error": error_msg,
                            "prompt_vars": {
                                "song": song,
                                "artists": artists,
                                "main_artist": main_artist
                            },
                            "uri": uri
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
    print(f"Total tracks in input file: {len(tracks_data)}")
    print(f"Valid tracks: {len(tracks_data) - invalid_count}")
    print(f"Invalid tracks: {invalid_count}")
    print(f"Songs processed this run: {total_processed_this_run}")
    print(f"  ├─ Successful: {success_count}")
    print(f"  └─ Failed: {error_count}")
    print(f"Success rate this run: {success_rate_this_run:.1f}%")
    if total_already_completed > 0:
        print(f"Songs already completed (skipped): {total_already_completed}")
    print(f"Results saved to: {args.output}")
    print(f"Raw logs saved to: {log_path}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    asyncio.run(main(cli_args))
    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed:.1f}s with "
          f"{MAX_CONCURRENCY}-way concurrency (rate ≤ {RATE_LIMIT_RPM} RPM)")