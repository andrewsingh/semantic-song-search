#!/usr/bin/env python3
"""
Generate text embeddings for artist profiles using OpenAI's text-embedding-3-large model.
Creates 7 embeddings per artist: genres, vocal_style, production_sound_design, lyrical_themes, 
mood_atmosphere, cultural_context_scene, and full_profile.
Saves embeddings in separate files by type following the V6 schema embedding structure.

Usage:
  # Save to directory (recommended)
  python embed_artist_profiles.py -i artist_profiles_v6.jsonl \
                                  -o artist_embeddings/ \
                                  -n 10  # optional, for testing
  
  # Or save with base name (creates subdirectory)
  python embed_artist_profiles.py -i input.jsonl -o artist_embeddings
  
  # This creates:
  # - genres_artist_embeddings.npz
  # - vocal_style_artist_embeddings.npz
  # - production_sound_design_artist_embeddings.npz
  # - lyrical_themes_artist_embeddings.npz  
  # - mood_atmosphere_artist_embeddings.npz
  # - cultural_context_scene_artist_embeddings.npz
  # - full_profile_artist_embeddings.npz
"""
import argparse, asyncio, json, time, os
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

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required but not set")

def format_individual_sections(profile_data):
    """Format individual sections as comma-separated strings of descriptors only."""
    # Convert arrays to comma-separated strings (no prefixes) with safe field access
    genres_text = ", ".join(profile_data.get('genres') or []) if profile_data.get('genres') else ""
    vocal_style_text = ", ".join(profile_data.get('vocal_style') or []) if profile_data.get('vocal_style') else ""
    production_sound_design_text = ", ".join(profile_data.get('production_sound_design') or []) if profile_data.get('production_sound_design') else ""
    lyrical_themes_text = ", ".join(profile_data.get('lyrical_themes') or []) if profile_data.get('lyrical_themes') else ""
    mood_atmosphere_text = ", ".join(profile_data.get('mood_atmosphere') or []) if profile_data.get('mood_atmosphere') else ""
    cultural_context_scene_text = ", ".join(profile_data.get('cultural_context_scene') or []) if profile_data.get('cultural_context_scene') else ""
    
    return genres_text, vocal_style_text, production_sound_design_text, lyrical_themes_text, mood_atmosphere_text, cultural_context_scene_text


def format_full_profile(profile_data):
    """Format full profile as comma-separated concatenation of all descriptors."""
    # Collect all non-empty descriptor arrays with safe field access
    all_descriptors = []
    
    if profile_data.get('genres'):
        all_descriptors.extend(profile_data['genres'])
    if profile_data.get('vocal_style'):
        all_descriptors.extend(profile_data['vocal_style'])
    if profile_data.get('production_sound_design'):
        all_descriptors.extend(profile_data['production_sound_design'])
    if profile_data.get('lyrical_themes'):
        all_descriptors.extend(profile_data['lyrical_themes'])
    if profile_data.get('mood_atmosphere'):
        all_descriptors.extend(profile_data['mood_atmosphere'])
    if profile_data.get('cultural_context_scene'):
        all_descriptors.extend(profile_data['cultural_context_scene'])
    
    # Return comma-separated string of all descriptors
    return ", ".join(all_descriptors)

def log_text_examples(profiles: List[Dict], num_examples: int = 3):
    """Log examples of formatted text for sanity checking."""
    print(f"\n{'='*60}")
    print("TEXT FORMATTING EXAMPLES (for sanity checking):")
    print(f"{'='*60}")
    
    for i, profile in enumerate(profiles[:num_examples]):
        artist = profile['artist']
        print(f"\n--- Example {i+1}: {artist} ---")
        
        # Get formatted texts
        genres_text, vocal_style_text, production_sound_design_text, lyrical_themes_text, mood_atmosphere_text, cultural_context_scene_text = format_individual_sections(profile)
        full_profile_text = format_full_profile(profile)
        
        print(f"\n1. GENRES TEXT:")
        print(f"   {genres_text}")
        
        print(f"\n2. VOCAL STYLE TEXT:")
        print(f"   {vocal_style_text}")
        
        print(f"\n3. PRODUCTION & SOUND DESIGN TEXT:")
        print(f"   {production_sound_design_text}")
        
        print(f"\n4. LYRICAL THEMES TEXT:")
        print(f"   {lyrical_themes_text}")
        
        print(f"\n5. MOOD & ATMOSPHERE TEXT:")
        print(f"   {mood_atmosphere_text}")
        
        print(f"\n6. CULTURAL CONTEXT & SCENE TEXT:")
        print(f"   {cultural_context_scene_text}")
        
        print(f"\n7. FULL PROFILE TEXT:")
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

async def process_artist_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single artist profile and generate all 7 embeddings."""
    artist = profile_data.get('artist', 'Unknown Artist')
    
    # Get the formatted text strings using existing functions
    genres_text, vocal_style_text, production_sound_design_text, lyrical_themes_text, mood_atmosphere_text, cultural_context_scene_text = format_individual_sections(profile_data)
    full_profile_text = format_full_profile(profile_data)
    
    # Generate all 7 embeddings concurrently
    genres_embedding, vocal_style_embedding, production_sound_design_embedding, lyrical_themes_embedding, mood_atmosphere_embedding, cultural_context_scene_embedding, full_profile_embedding = await asyncio.gather(
        get_embedding(genres_text),
        get_embedding(vocal_style_text),
        get_embedding(production_sound_design_text),
        get_embedding(lyrical_themes_text), 
        get_embedding(mood_atmosphere_text),
        get_embedding(cultural_context_scene_text),
        get_embedding(full_profile_text)
    )
    
    return {
        "artist": artist,
        "lead_vocalist_gender": profile_data.get('lead_vocalist_gender', 'N/A'),  # Include lead_vocalist_gender with safe access
        "embeddings": {
            "genres": genres_embedding,
            "vocal_style": vocal_style_embedding,
            "production_sound_design": production_sound_design_embedding,
            "lyrical_themes": lyrical_themes_embedding,
            "mood_atmosphere": mood_atmosphere_embedding,
            "cultural_context_scene": cultural_context_scene_embedding,
            "full_profile": full_profile_embedding
        },
        "original_texts": {
            "genres": genres_text,
            "vocal_style": vocal_style_text,
            "production_sound_design": production_sound_design_text,
            "lyrical_themes": lyrical_themes_text,
            "mood_atmosphere": mood_atmosphere_text,
            "cultural_context_scene": cultural_context_scene_text,
            "full_profile": full_profile_text
        }
    }

def save_embeddings_numpy(results: List[Dict], output_path: str):
    """Save embeddings in separate files by type for the new structure."""
    # Determine if output_path is a directory or a base filename
    output_path_obj = Path(output_path)
    
    output_dir = output_path_obj
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for each embedding type
    embedding_types = ['genres', 'vocal_style', 'production_sound_design', 'lyrical_themes', 'mood_atmosphere', 'cultural_context_scene', 'full_profile']
    
    for embedding_type in embedding_types:
        artists = []
        embeddings = []
        artist_indices = []
        field_values = []
        lead_vocalist_genders = []  # Add lead_vocalist_gender metadata
        
        for i, result in enumerate(results):
            artists.append(result['artist'])
            embeddings.append(result['embeddings'][embedding_type])
            artist_indices.append(i)
            lead_vocalist_genders.append(result['lead_vocalist_gender'])
            
            # Get the original text that was embedded for this type
            field_values.append(result['original_texts'][embedding_type])
        
        # Convert to numpy arrays
        embeddings_array = np.array(embeddings)
        artist_indices_array = np.array(artist_indices)
        
        # Save individual embedding type file
        output_file = output_dir / f"{embedding_type}_artist_embeddings.npz"
        np.savez_compressed(
            output_file,
            artists=np.array(artists),
            embeddings=embeddings_array,
            artist_indices=artist_indices_array,
            field_values=np.array(field_values),
            lead_vocalist_genders=np.array(lead_vocalist_genders)  # Include lead_vocalist_gender
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
    print(f"Generating 7 embeddings per artist (total: {len(profiles) * 7} embeddings)")
    
    # Log text examples for sanity checking
    log_text_examples(profiles, num_examples=min(3, len(profiles)))

    print(f"Skipped {num_skipped} artist profiles")
    
    # Process all profiles concurrently
    results = []
    tasks = [asyncio.create_task(process_artist_profile(profile)) for profile in profiles]
    
    with tqdm(total=len(tasks), unit="artist") as pbar:
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                # Validate result structure before adding
                if (result and isinstance(result, dict) and 
                    'artist' in result and 'embeddings' in result and 
                    'original_texts' in result and 'lead_vocalist_gender' in result):
                    results.append(result)
                else:
                    print(f"Warning: Skipping malformed result for artist")
            except Exception as e:
                print(f"Error processing artist profile: {e}")
                # Continue processing other artists instead of failing completely
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