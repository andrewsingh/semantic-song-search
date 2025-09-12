#!/usr/bin/env python3
"""
Generate text embeddings for song profiles using OpenAI's text-embedding-3-large model.
Creates 5 embeddings per song using the new SongProfile V6 schema.

EMBEDDING TYPES (V6 Schema):
  1. genres: Genre descriptors (comma-separated string)
  2. vocal_style: Vocal characteristic descriptors (comma-separated string)
  3. production_sound_design: Production and sonic descriptors (comma-separated string)
  4. lyrical_meaning: Lyrical meaning descriptors (comma-separated string)
  5. mood_atmosphere: Mood and atmospheric descriptors (comma-separated string)

Each embedding type is stored in a separate .npz file for independent processing.

Usage:
  # Generate all embedding types:
  python embed_song_profiles.py -i song_profiles_v6.jsonl \
                                -o data/embeddings/
                                
  # Generate specific embedding types:
  python embed_song_profiles.py -i song_profiles_v6.jsonl \
                                -o data/embeddings/ \
                                --embed_types genres vocal_style
                                
  # Optional flags:
  python embed_song_profiles.py -i song_profiles_v6.jsonl \
                                -o data/embeddings/ \
                                -n 100  # for testing first N songs
  
  # The output will be separate .npz files for each embedding type containing:
  # - songs: song names (from prompt_vars)
  # - artists: artist names (from prompt_vars)
  # - main_artists: main artist names (from prompt_vars)
  # - uris: track URIs (from top-level profile)
  # - embeddings: embedding vectors for this type
  # - song_indices: which song each embedding belongs to
  # - field_values: original descriptor text that was embedded
"""
import argparse, asyncio, json, time, os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from openai import AsyncOpenAI, RateLimitError, APIError
from tqdm import tqdm

# ──────────────────────────────── RATE-LIMIT SETTINGS ─────────────────────────
RATE_LIMIT_RPM = 5000
SAFETY_FACTOR = 0.80
RPS = RATE_LIMIT_RPM / 60
MAX_CONCURRENCY = max(1, int(RPS * SAFETY_FACTOR))
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = max(1, int(60 / RATE_LIMIT_RPM * 10))

# ──────────────────────────────── OPENAI SETTINGS ─────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_TYPES = ['genres', 'vocal_style', 'production_sound_design', 'lyrical_meaning', 'mood_atmosphere']

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required but not set")

print(f"Rate limit settings: {RATE_LIMIT_RPM} RPM, {MAX_CONCURRENCY} concurrent requests")

def format_individual_sections(profile_data):
    """Format individual sections as comma-separated strings with prefixes (matching artist embedding format)."""
    # Convert arrays to comma-separated strings with safe field access and validation
    def safe_join(field_name):
        field_data = profile_data.get(field_name)
        if field_data and isinstance(field_data, list):
            # Filter out empty strings and None values
            clean_data = [str(item).strip() for item in field_data if item and str(item).strip()]
            return ", ".join(clean_data) if clean_data else ""
        return ""
    
    genres_text = safe_join('genres')
    vocal_style_text = safe_join('vocal_style')
    production_sound_design_text = safe_join('production_sound_design')
    lyrical_meaning_text = safe_join('lyrical_meaning')
    mood_atmosphere_text = safe_join('mood_atmosphere')
    
    # Add prefixes to match artist embedding format
    genres_text = f"genres: {genres_text}"
    vocal_style_text = f"vocal style: {vocal_style_text}"
    production_sound_design_text = f"production & sound design: {production_sound_design_text}"
    lyrical_meaning_text = f"lyrical meaning: {lyrical_meaning_text}"
    mood_atmosphere_text = f"mood & atmosphere: {mood_atmosphere_text}"
    
    return genres_text, vocal_style_text, production_sound_design_text, lyrical_meaning_text, mood_atmosphere_text


def log_text_examples(profiles: List[Dict], num_examples: int = 3):
    """Log examples of formatted text for sanity checking."""
    print(f"\n{'='*60}")
    print("TEXT FORMATTING EXAMPLES (for sanity checking):")
    print(f"{'='*60}")
    
    for i, profile in enumerate(profiles[:num_examples]):
        prompt_vars = profile.get('prompt_vars', {})
        song = prompt_vars.get('song', 'Unknown Song')
        artists = prompt_vars.get('artists', 'Unknown Artist')
        main_artist = prompt_vars.get('main_artist', 'Unknown Main Artist')
        uri = profile.get('uri', 'Unknown URI')
        print(f"\n--- Example {i+1}: {song} by {artists} (Main: {main_artist}) [{uri}] ---")
        
        # Get formatted texts
        genres_text, vocal_style_text, production_sound_design_text, lyrical_meaning_text, mood_atmosphere_text = format_individual_sections(profile)
        
        print(f"\n1. GENRES TEXT:")
        print(f"   {genres_text}")
        
        print(f"\n2. VOCAL STYLE TEXT:")
        print(f"   {vocal_style_text}")
        
        print(f"\n3. PRODUCTION & SOUND DESIGN TEXT:")
        print(f"   {production_sound_design_text}")
        
        print(f"\n4. LYRICAL MEANING TEXT:")
        print(f"   {lyrical_meaning_text}")
        
        print(f"\n5. MOOD & ATMOSPHERE TEXT:")
        print(f"   {mood_atmosphere_text}")
        
        print(f"\n{'-'*50}")
    
    print(f"\n{'='*60}")
    print("END OF TEXT EXAMPLES")
    print(f"{'='*60}\n")

# ──────────────────────────────── ASYNC WORKER ────────────────────────────────
client = AsyncOpenAI()
sem = asyncio.Semaphore(MAX_CONCURRENCY)

async def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text string with retry logic."""
    # Handle empty strings - use a minimal placeholder to avoid API issues
    if not text or text.strip() == "":
        text = "unknown"
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with sem:  # Move semaphore inside try block to ensure proper release
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

async def process_song_profile(profile_data: Dict[str, Any], embed_types: List[str]) -> Dict[str, Any]:
    """Process a single song profile and generate embeddings for requested types only."""
    # Extract metadata from prompt_vars and top-level profile
    prompt_vars = profile_data.get('prompt_vars', {})
    song = prompt_vars.get('song', 'Unknown Song')
    artists = prompt_vars.get('artists', 'Unknown Artist')
    main_artist = prompt_vars.get('main_artist', 'Unknown Main Artist')
    uri = profile_data.get('uri', 'Unknown URI')
    
    # Get the formatted text strings using existing functions
    genres_text, vocal_style_text, production_sound_design_text, lyrical_meaning_text, mood_atmosphere_text = format_individual_sections(profile_data)
    
    # Map embedding types to their texts
    text_map = {
        "genres": genres_text,
        "vocal_style": vocal_style_text,
        "production_sound_design": production_sound_design_text,
        "lyrical_meaning": lyrical_meaning_text,
        "mood_atmosphere": mood_atmosphere_text
    }
    
    # Generate only requested embeddings concurrently
    embedding_tasks = {}
    for embed_type in embed_types:
        if embed_type in text_map:
            embedding_tasks[embed_type] = get_embedding(text_map[embed_type])
    
    # Wait for all requested embeddings
    completed_embeddings = await asyncio.gather(*embedding_tasks.values(), return_exceptions=True)
    
    # Build results dictionary
    embeddings = {}
    original_texts = {}
    
    for i, embed_type in enumerate(embedding_tasks.keys()):
        result = completed_embeddings[i]
        if not isinstance(result, Exception):
            embeddings[embed_type] = result
            original_texts[embed_type] = text_map[embed_type]
        else:
            print(f"Warning: Failed to generate {embed_type} embedding for {song}: {result}")
    
    return {
        "song": song,
        "artists": artists,
        "main_artist": main_artist,
        "uri": uri,
        "embeddings": embeddings,
        "original_texts": original_texts
    }

def save_embeddings_numpy(results: List[Dict], output_path: str, embed_types: List[str]):
    """Save embeddings in separate files by type for the new structure."""
    # Determine if output_path is a directory or a base filename
    output_path_obj = Path(output_path)
    
    output_dir = output_path_obj
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for each embedding type
    for embedding_type in embed_types:
        songs = []
        artists = []
        main_artists = []
        uris = []
        embeddings = []
        song_indices = []
        field_values = []
        
        for i, result in enumerate(results):
            if embedding_type in result['embeddings']:  # Only process requested types
                songs.append(result['song'])
                artists.append(result['artists'])
                main_artists.append(result['main_artist'])
                uris.append(result['uri'])
                embeddings.append(result['embeddings'][embedding_type])
                song_indices.append(i)
                
                # Get the original text that was embedded for this type
                field_values.append(result['original_texts'][embedding_type])
        
        if not embeddings:  # Skip if no embeddings for this type
            continue
            
        # Convert to numpy arrays
        embeddings_array = np.array(embeddings)
        song_indices_array = np.array(song_indices)
        
        # Save individual embedding type file
        output_file = output_dir / f"{embedding_type}_embeddings.npz"
        np.savez_compressed(
            output_file,
            songs=np.array(songs),
            artists=np.array(artists),
            main_artists=np.array(main_artists),
            uris=np.array(uris),
            embeddings=embeddings_array,
            song_indices=song_indices_array,
            field_values=np.array(field_values)
        )
        
        print(f"Saved {embedding_type} embeddings: {output_file}")
        print(f"  Shape: {embeddings_array.shape}")
    
    print(f"\nAll embedding files saved to: {output_dir}")
    print(f"Generated {len(embed_types)} embedding files for {len(results)} songs")

# ──────────────────────────────── MAIN DRIVER ────────────────────────────────
async def main(input_file: str, output_path: str, embed_types: List[str], num_entries: int = None):
    """Main function to process all song profiles."""
    print(f"Loading profiles from {input_file}")
    print(f"Generating embedding types: {', '.join(embed_types)}")
    
    # Load profiles from JSONL file
    profiles = []
    num_skipped = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                profile = json.loads(line.strip())
                if profile.get("familiar", True):  # Process if familiar is True or missing
                    profiles.append(profile)
                else:
                    num_skipped += 1

    if num_entries:
        profiles = profiles[:num_entries]
    
    print(f"Processing {len(profiles)} song profiles...")
    print(f"Generating {len(embed_types)} embeddings per song (total: {len(profiles) * len(embed_types)} embeddings)")
    
    # Log text examples for sanity checking
    log_text_examples(profiles, num_examples=min(3, len(profiles)))

    print(f"Skipped {num_skipped} song profiles (not familiar)")
    
    # Process all profiles concurrently
    results = []
    tasks = [asyncio.create_task(process_song_profile(profile, embed_types)) for profile in profiles]
    
    with tqdm(total=len(tasks), unit="song") as pbar:
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                # Validate result structure before adding
                if (result and isinstance(result, dict) and 
                    'song' in result and 'embeddings' in result and 
                    'original_texts' in result):
                    results.append(result)
                else:
                    print(f"Warning: Skipping malformed result for song")
            except Exception as e:
                print(f"Error processing song profile: {e}")
                # Continue processing other songs instead of failing completely
            pbar.update(1)
    
    # Save embeddings
    save_embeddings_numpy(results, output_path, embed_types)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for song profiles")
    parser.add_argument("-i", "--input", required=True, 
                        help="Input JSONL file with song profiles")
    parser.add_argument("-o", "--output", required=True, 
                        help="Output directory for separate embedding files")
    parser.add_argument("--embed_types", nargs='+', default=['all'],
                        choices=(EMBEDDING_TYPES + ['all']),
                        help="Embedding types to generate (space-separated, default: all)")
    parser.add_argument("-n", "--num_entries", type=int, default=None, 
                        help="Process only first N entries (for testing)")
    
    args = parser.parse_args()
    
    t0 = time.perf_counter()

    if 'all' in args.embed_types:
        embed_types = EMBEDDING_TYPES
    else:
        embed_types = args.embed_types
    
    # Validate embed_types
    if not embed_types:
        raise ValueError("No embedding types specified")
    
    invalid_types = [et for et in embed_types if et not in EMBEDDING_TYPES]
    if invalid_types:
        raise ValueError(f"Invalid embedding types: {invalid_types}. Valid types: {EMBEDDING_TYPES}")
    
    print("Embed types: ", embed_types)
    asyncio.run(main(args.input, args.output, embed_types, args.num_entries))
    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed:.1f}s with {MAX_CONCURRENCY}-way concurrency") 