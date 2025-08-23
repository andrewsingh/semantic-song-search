#!/usr/bin/env python3
"""
Generate text embeddings for artist profiles using OpenAI's text-embedding-3-large model.
Creates 4 embeddings per artist: musical_style, lyrical_themes, mood, and full_profile.
Saves embeddings in separate files by type following the new embedding structure.

Usage:
  # Save to directory (recommended)
  python embed_artist_profiles.py -i selected_artists_v0_profiles.jsonl \
                                  -o artist_embeddings/ \
                                  -n 10  # optional, for testing
  
  # Or save with base name (creates subdirectory)
  python embed_artist_profiles.py -i input.jsonl -o artist_embeddings
  
  # This creates:
  # - musical_style_embeddings.npz
  # - lyrical_themes_embeddings.npz  
  # - mood_embeddings.npz
  # - full_profile_embeddings.npz
"""
import argparse, asyncio, json, time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from openai import AsyncOpenAI, RateLimitError, APIError
from tqdm import tqdm

# ──────────────────────────────── RATE-LIMIT SETTINGS ─────────────────────────
RATE_LIMIT_RPM = 3000
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
        },
        "original_texts": {
            "musical_style": musical_style_text,
            "lyrical_themes": lyrical_themes_text,
            "mood": mood_text,
            "full_profile": full_profile_text
        }
    }

def save_embeddings_numpy(results: List[Dict], output_path: str):
    """Save embeddings in separate files by type for the new structure."""
    # Determine if output_path is a directory or a base filename
    output_path_obj = Path(output_path)
    
    if output_path.endswith('/') or output_path_obj.is_dir():
        # Directory mode - save separate files in directory
        output_dir = output_path_obj
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Base filename mode - use directory of the file and create subdirectory
        output_dir = output_path_obj.parent / f"{output_path_obj.stem}_artist_embeddings"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for each embedding type
    embedding_types = ['musical_style', 'lyrical_themes', 'mood', 'full_profile']
    
    for embedding_type in embedding_types:
        artists = []
        embeddings = []
        artist_indices = []
        field_values = []
        
        for i, result in enumerate(results):
            artists.append(result['artist'])
            embeddings.append(result['embeddings'][embedding_type])
            artist_indices.append(i)
            
            # Get the original text that was embedded for this type
            field_values.append(result['original_texts'][embedding_type])
        
        # Convert to numpy arrays
        embeddings_array = np.array(embeddings)
        artist_indices_array = np.array(artist_indices)
        
        # Save individual embedding type file
        output_file = output_dir / f"{embedding_type}_embeddings.npz"
        np.savez_compressed(
            output_file,
            artists=np.array(artists),
            embeddings=embeddings_array,
            artist_indices=artist_indices_array,
            field_values=np.array(field_values)
        )
        
        print(f"Saved {embedding_type} embeddings: {output_file}")
        print(f"  Shape: {embeddings_array.shape}")
    
    print(f"\nAll embedding files saved to: {output_dir}")
    print(f"Generated {len(embedding_types)} embedding files for {len(results)} artists")

def save_embeddings_json(results: List[Dict], output_file: str):
    """Save embeddings in JSON format (human-readable but less efficient)."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Embeddings saved to {output_file} (JSON format)")

# ──────────────────────────────── MAIN DRIVER ────────────────────────────────
async def main(input_file: str, output_path: str, output_format: str = "numpy", num_entries: int = None):
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
        save_embeddings_numpy(results, output_path)
    else:
        save_embeddings_json(results, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for artist profiles")
    parser.add_argument("-i", "--input", required=True, 
                        help="Input JSONL file with artist profiles")
    parser.add_argument("-o", "--output", required=True, 
                        help="Output directory or base name for embeddings (creates separate files by type)")
    parser.add_argument("-n", "--num_entries", type=int, default=None, 
                        help="Process only first N entries (for testing)")
    parser.add_argument("--format", choices=["numpy", "json"], default="numpy",
                        help="Output format: numpy (default, .npz) or json (.jsonl)")
    
    args = parser.parse_args()
    
    t0 = time.perf_counter()
    asyncio.run(main(args.input, args.output, args.format, args.num_entries))
    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed:.1f}s with {MAX_CONCURRENCY}-way concurrency")