#!/usr/bin/env python3
"""
Anonymize song profile descriptions by replacing proper nouns with generic terms
using the OpenAI API. This script processes the 'sound', 'meaning', and 'mood' 
fields from song profiles to make them suitable for embedding-based similarity search.

Usage:
  python anonymize_song_profiles.py -i song_profiles.jsonl \
                                   -o anonymized_profiles.jsonl \
                                   -p prompts/anonymize_desc_v0.txt \
                                   -n 100  # optional for testing
"""
import argparse, asyncio, json, time, re
from pathlib import Path
from typing import List, Dict, Any

from openai import AsyncOpenAI, RateLimitError, APIError
from tqdm import tqdm

# ──────────────────────────────── RATE-LIMIT SETTINGS ─────────────────────────
RATE_LIMIT_RPM   = 2000                 # <-- edit this if your quota changes
SAFETY_FACTOR    = 0.80                 # 20% head-room
RPS              = RATE_LIMIT_RPM / 60  # requests per second
MAX_CONCURRENCY  = max(1, int(RPS * SAFETY_FACTOR))
MAX_RETRIES      = 3
RETRY_BACKOFF_SEC = max(1, int(60 / RATE_LIMIT_RPM * 10))

print(f"Rate limit settings: {RATE_LIMIT_RPM} RPM, {MAX_CONCURRENCY} concurrent requests")

# ──────────────────────────────── ARGPARSE ────────────────────────────────
parser = argparse.ArgumentParser(
    description="Anonymize song profile descriptions using OpenAI API."
)
parser.add_argument("-i", "--input", required=True,
                    help="Path to input JSONL file with song profiles")
parser.add_argument("-o", "--output", required=True,
                    help="Path to output JSONL file for anonymized profiles")
parser.add_argument("-p", "--prompt", required=True,
                    help="Path to anonymization prompt template ({{desc}} will be replaced)")
parser.add_argument("-n", "--num_entries", type=int, default=None,
                    help="Process only the first N entries (testing)")
parser.add_argument("-l", "--log", default=None,
                    help="Path to raw API response log file (defaults to <output>.raw.jsonl)")

cli_args = parser.parse_args()

# ─────────────────────────────── PROMPT TEMPLATE ──────────────────────────────
PROMPT_TEMPLATE = Path(cli_args.prompt).read_text(encoding="utf-8").strip()

# ────────────────────────────── HELPER FUNCTIONS ──────────────────────────────
def clean_citations(text: str) -> str:
    """Remove Perplexity citations like [1], [2], [1][3] from text"""
    if not text:
        return text
    return re.sub(r"\[\d+\]", "", text)

def build_prompt(description: str) -> str:
    # Clean citations before building prompt
    cleaned_description = clean_citations(description)
    return PROMPT_TEMPLATE.replace("{{desc}}", cleaned_description)

def load_song_profiles(input_path: Path, num_entries: int = None) -> List[Dict[str, Any]]:
    """Load song profiles from JSONL file"""
    profiles = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                profiles.append(json.loads(line.strip()))
    
    if num_entries:
        profiles = profiles[:num_entries]
    
    return profiles

def get_profile_key(profile: Dict[str, Any]) -> tuple:
    """Generate a unique key for a profile to track processing"""
    return (profile.get('original_song', profile.get('song')), 
            profile.get('original_artist', profile.get('artist')))

# ──────────────────────────────── ASYNC WORKER ────────────────────────────────
client = AsyncOpenAI()
sem = asyncio.Semaphore(MAX_CONCURRENCY)

async def anonymize_description(description: str, field_name: str, song_key: tuple) -> str:
    """Anonymize a single description using OpenAI API"""
    if not description or description == "Not applicable - instrumental track":
        return description
    
    prompt = build_prompt(description)
    
    for attempt in range(1, MAX_RETRIES + 1):
        async with sem:
            try:
                response = await client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1  # Low temperature for consistent anonymization
                )
                
                anonymized = response.choices[0].message.content.strip()
                return anonymized
                
            except (RateLimitError, APIError) as err:
                if attempt == MAX_RETRIES:
                    print(f"[ERROR] Failed to anonymize {field_name} for {song_key}: {err}")
                    return description  # Return original on failure
                
                wait = RETRY_BACKOFF_SEC * attempt
                print(f"[retry {attempt}/{MAX_RETRIES}] {song_key} {field_name}: {err} → sleeping {wait}s")
                await asyncio.sleep(wait)

async def anonymize_profile(profile: Dict[str, Any]) -> tuple:
    """Anonymize sound, meaning, and mood descriptions for a single profile"""
    song_key = get_profile_key(profile)
    
    # Fields to anonymize
    fields_to_anonymize = ['sound', 'meaning', 'mood']
    
    # Create tasks for each field that needs anonymization
    tasks = []
    for field in fields_to_anonymize:
        if field in profile and profile[field]:
            task = asyncio.create_task(
                anonymize_description(profile[field], field, song_key)
            )
            tasks.append((field, task))
    
    # Wait for all anonymization tasks to complete
    anonymized_profile = profile.copy()
    raw_responses = {}
    
    for field, task in tasks:
        try:
            anonymized_desc = await task
            # Keep original field and add anonymized version
            anonymized_profile[f"{field}_anonymized"] = anonymized_desc
            raw_responses[f"{field}_anonymized"] = anonymized_desc
        except Exception as e:
            print(f"[ERROR] Failed to anonymize {field} for {song_key}: {e}")
            # Keep original description, don't add anonymized version on error
    
    return "success", anonymized_profile, raw_responses, song_key

# ──────────────────────────────── MAIN DRIVER ────────────────────────────────
async def main(args) -> None:
    # Load profiles
    input_path = Path(args.input)
    profiles = load_song_profiles(input_path, args.num_entries)
    print(f"Loaded {len(profiles)} song profiles")
    
    # Check for existing results and filter out already-processed profiles
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
                        key = get_profile_key(result)
                        already_processed.add(key)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed line {line_num} in existing output: {e}")
                        continue
            
            print(f"Found {len(already_processed)} already-processed profiles")
            
            # Filter out already-processed profiles
            original_count = len(profiles)
            profiles = [p for p in profiles if get_profile_key(p) not in already_processed]
            skipped_count = original_count - len(profiles)
            
            print(f"Skipping {skipped_count} already-processed profiles")
            print(f"Remaining profiles to process: {len(profiles)}")
            
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")
            print("Starting fresh (will overwrite existing file)")
            already_processed = set()
    
    if len(profiles) == 0:
        print("No profiles to process - all profiles have already been completed!")
        return
    
    # Show sample of what we'll be processing
    if profiles:
        sample = profiles[0]
        sample_key = get_profile_key(sample)
        print(f"\n{'='*60}")
        print(f"SAMPLE PROFILE: '{sample_key[0]}' by '{sample_key[1]}'")
        print(f"Fields to anonymize: sound, meaning, mood")
        print(f"{'='*60}")
        
        # Show sample prompts for each field
        for field in ['sound', 'meaning', 'mood']:
            if sample.get(field):
                original_desc = sample[field]
                sample_prompt = build_prompt(original_desc)
                
                print(f"\nSAMPLE {field.upper()} PROMPT:")
                print(f"Original description: {original_desc[:200]}{'...' if len(original_desc) > 200 else ''}")
                print(f"Cleaned description: {clean_citations(original_desc)[:200]}{'...' if len(clean_citations(original_desc)) > 200 else ''}")
                print(f"Full prompt sent to API:")
                print("-" * 40)
                print(sample_prompt)
                print("-" * 40)
                break  # Just show one sample to avoid too much output
        
        print(f"{'='*60}\n")
    
    # Determine log file path
    log_path = Path(args.log) if args.log else Path(f"{args.output}.raw.jsonl")
    
    success_count = 0
    error_count = 0
    
    # Open output files in append mode to preserve existing results
    file_mode = "a" if output_path.exists() and len(already_processed) > 0 else "w"
    log_mode = "a" if log_path.exists() and len(already_processed) > 0 else "w"
    
    with output_path.open(file_mode, encoding="utf-8") as sink, \
         log_path.open(log_mode, encoding="utf-8") as raw_sink:
        
        # Create tasks for all profiles
        tasks = [
            asyncio.create_task(anonymize_profile(profile))
            for profile in profiles
        ]
        
        # Process tasks with progress bar
        with tqdm(total=len(tasks), unit="profile") as pbar:
            for coro in asyncio.as_completed(tasks):
                status, anonymized_profile, raw_responses, song_key = await coro
                
                if status == "success":
                    # Write anonymized profile
                    sink.write(json.dumps(anonymized_profile, ensure_ascii=False) + "\n")
                    sink.flush()  # Ensure immediate write
                    
                    # Log raw responses for debugging
                    log_entry = {
                        "song_key": song_key,
                        "raw_responses": raw_responses,
                        "timestamp": time.time()
                    }
                    raw_sink.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    raw_sink.flush()
                    
                    success_count += 1
                else:
                    error_count += 1
                
                pbar.update(1)
    
    # Report final statistics
    total_processed_this_run = success_count + error_count
    success_rate_this_run = (success_count / total_processed_this_run * 100) if total_processed_this_run > 0 else 0
    total_already_completed = len(already_processed)
    
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Profiles processed this run: {total_processed_this_run}")
    print(f"  ├─ Successful: {success_count}")
    print(f"  └─ Failed: {error_count}")
    print(f"Success rate this run: {success_rate_this_run:.1f}%")
    if total_already_completed > 0:
        print(f"Profiles already completed (skipped): {total_already_completed}")
        total_overall = total_processed_this_run + total_already_completed
        print(f"Total profiles in dataset: {total_overall}")
    print(f"Results saved to: {args.output}")
    print(f"Raw logs saved to: {log_path}")

if __name__ == "__main__":
    t0 = time.perf_counter()
    asyncio.run(main(cli_args))
    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed:.1f}s with "
          f"{MAX_CONCURRENCY}-way concurrency (rate ≤ {RATE_LIMIT_RPM} RPM)") 