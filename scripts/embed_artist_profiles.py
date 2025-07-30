#!/usr/bin/env python3
"""
Generate text embeddings for artist profiles using OpenAI's text-embedding-3-large model.
Creates 4 embeddings per artist: musical_style, lyrical_themes, mood, and full_profile.

Usage:
  python embed_artist_profiles.py -i selected_artists_v0_profiles.jsonl \
                                  -o artist_embeddings.npz \
                                  -n 10  # optional, for testing
  
  # Or save as JSON (less efficient but human-readable)
  python embed_artist_profiles.py -i input.jsonl -o output.json --format json
"""
import argparse, asyncio, json, time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from openai import AsyncOpenAI, RateLimitError, APIError
from tqdm import tqdm

# ──────────────────────────────── RATE-LIMIT SETTINGS ─────────────────────────
RATE_LIMIT_RPM = 500
SAFETY_FACTOR = 0.80
RPS = RATE_LIMIT_RPM / 60
MAX_CONCURRENCY = max(1, int(RPS * SAFETY_FACTOR))
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = max(1, int(60 / RATE_LIMIT_RPM * 10))

# ──────────────────────────────── OPENAI SETTINGS ─────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-large"

def format_individual_sections(profile_data):
    artist = profile_data['artist']
    
    musical_style_text = f"Musical style of {artist}: {profile_data['musical_style']}"
    lyrical_themes_text = f"Lyrical themes of {artist}: {profile_data['lyrical_themes']}"
    mood_text = f"Mood of {artist}'s music: {profile_data['mood']}"
    
    return musical_style_text, lyrical_themes_text, mood_text


def format_full_profile(profile_data):
    artist = profile_data['artist']
    
    full_profile_text = f"""Artist: {artist}

Musical Style: {profile_data['musical_style']}

Lyrical Themes: {profile_data['lyrical_themes']}

Mood: {profile_data['mood']}"""
    
    return full_profile_text

def log_text_examples(profiles: List[Dict], num_examples: int = 3):
    """Log examples of formatted text for sanity checking."""
    print(f"\n{'='*60}")
    print("TEXT FORMATTING EXAMPLES (for sanity checking):")
    print(f"{'='*60}")
    
    for i, profile in enumerate(profiles[:num_examples]):
        artist = profile['artist']
        print(f"\n--- Example {i+1}: {artist} ---")
        
        # Get formatted texts
        musical_style_text, lyrical_themes_text, mood_text = format_individual_sections(profile)
        full_profile_text = format_full_profile(profile)
        
        print(f"\n1. MUSICAL STYLE TEXT:")
        print(f"   {musical_style_text}")
        
        print(f"\n2. LYRICAL THEMES TEXT:")
        print(f"   {lyrical_themes_text}")
        
        print(f"\n3. MOOD TEXT:")
        print(f"   {mood_text}")
        
        print(f"\n4. FULL PROFILE TEXT:")
        print(f"   {full_profile_text}")
        
        print(f"\n{'-'*50}")
    
    print(f"\n{'='*60}")
    print("END OF TEXT EXAMPLES")
    print(f"{'='*60}\n")

# ──────────────────────────────── ASYNC WORKER ────────────────────────────────
client = AsyncOpenAI()
sem = asyncio.Semaphore(MAX_CONCURRENCY)

async def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text string with retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        async with sem:
            try:
                response = await client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=text,
                    encoding_format="float"
                )
                return response.data[0].embedding
            except (RateLimitError, APIError) as err:
                if attempt == MAX_RETRIES:
                    raise
                wait = RETRY_BACKOFF_SEC * attempt
                print(f"[retry {attempt}/{MAX_RETRIES}] {err} → sleeping {wait}s")
                await asyncio.sleep(wait)

async def process_artist_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single artist profile and generate all 4 embeddings."""
    artist = profile_data['artist']
    
    # Get the formatted text strings using existing functions
    musical_style_text, lyrical_themes_text, mood_text = format_individual_sections(profile_data)
    full_profile_text = format_full_profile(profile_data)
    
    # Generate all 4 embeddings concurrently
    musical_style_embedding, lyrical_themes_embedding, mood_embedding, full_profile_embedding = await asyncio.gather(
        get_embedding(musical_style_text),
        get_embedding(lyrical_themes_text), 
        get_embedding(mood_text),
        get_embedding(full_profile_text)
    )
    
    return {
        "artist": artist,
        "embeddings": {
            "musical_style": musical_style_embedding,
            "lyrical_themes": lyrical_themes_embedding,
            "mood": mood_embedding,
            "full_profile": full_profile_embedding
        }
    }

def save_embeddings_numpy(results: List[Dict], output_file: str):
    """Save embeddings in NumPy format for better precision and efficiency."""
    artists = []
    musical_style_embeddings = []
    lyrical_themes_embeddings = []
    mood_embeddings = []
    full_profile_embeddings = []
    
    for result in results:
        artists.append(result['artist'])
        embeddings = result['embeddings']
        musical_style_embeddings.append(embeddings['musical_style'])
        lyrical_themes_embeddings.append(embeddings['lyrical_themes'])
        mood_embeddings.append(embeddings['mood'])
        full_profile_embeddings.append(embeddings['full_profile'])
    
    # Convert to numpy arrays
    musical_style_array = np.array(musical_style_embeddings)
    lyrical_themes_array = np.array(lyrical_themes_embeddings)
    mood_array = np.array(mood_embeddings)
    full_profile_array = np.array(full_profile_embeddings)
    
    # Save as compressed numpy file
    np.savez_compressed(
        output_file,
        artists=artists,
        musical_style=musical_style_array,
        lyrical_themes=lyrical_themes_array,
        mood=mood_array,
        full_profile=full_profile_array
    )
    
    print(f"Embeddings saved to {output_file}")
    print(f"Shape: {musical_style_array.shape} per embedding type")
    print(f"To load: data = np.load('{output_file}'); artists = data['artists']; musical_style = data['musical_style']")

def save_embeddings_json(results: List[Dict], output_file: str):
    """Save embeddings in JSON format (human-readable but less efficient)."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Embeddings saved to {output_file} (JSON format)")

# ──────────────────────────────── MAIN DRIVER ────────────────────────────────
async def main(input_file: str, output_file: str, output_format: str = "numpy", num_entries: int = None):
    """Main function to process all artist profiles."""
    # Load profiles from JSONL file
    profiles = []
    num_skipped = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                profile = json.loads(line)
                if profile["familiar"]:
                    profiles.append(profile)
                else:
                    num_skipped += 1

    
    if num_entries:
        profiles = profiles[:num_entries]
    
    print(f"Processing {len(profiles)} artist profiles...")
    print(f"Generating 4 embeddings per artist (total: {len(profiles) * 4} embeddings)")
    
    # Log text examples for sanity checking
    log_text_examples(profiles, num_examples=min(3, len(profiles)))

    print(f"Skipped {num_skipped} artist profiles")
    
    # Process all profiles concurrently
    results = []
    tasks = [asyncio.create_task(process_artist_profile(profile)) for profile in profiles]
    
    with tqdm(total=len(tasks), unit="artist") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)
    
    # Save embeddings in requested format
    if output_format.lower() == "numpy":
        save_embeddings_numpy(results, output_file)
    else:
        save_embeddings_json(results, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for artist profiles")
    parser.add_argument("-i", "--input", required=True, 
                        help="Input JSONL file with artist profiles")
    parser.add_argument("-o", "--output", required=True, 
                        help="Output file for embeddings (.npz for numpy, .jsonl for JSON)")
    parser.add_argument("-n", "--num_entries", type=int, default=None, 
                        help="Process only first N entries (for testing)")
    parser.add_argument("--format", choices=["numpy", "json"], default="numpy",
                        help="Output format: numpy (default, .npz) or json (.jsonl)")
    
    args = parser.parse_args()
    
    t0 = time.perf_counter()
    asyncio.run(main(args.input, args.output, args.format, args.num_entries))
    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed:.1f}s with {MAX_CONCURRENCY}-way concurrency")