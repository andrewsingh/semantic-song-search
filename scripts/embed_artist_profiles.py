#!/usr/bin/env python3
"""
Generate text embeddings for artist profiles using OpenAI's text-embedding-3-large model.
Creates embeddings per artist: individual genre embeddings (V7 schema with prominence scores), 
vocal_style, production_sound_design, lyrical_themes, mood_atmosphere, and cultural_context_scene.
Saves embeddings in separate files by type following the V7 schema embedding structure.

Usage:
  # Save to directory (recommended)
  python embed_artist_profiles.py -i artist_profiles_v7.jsonl \
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

def load_genre_embedding_store(store_path: str) -> Dict[str, List[float]]:
    """Load the shared genre embedding store from a .npz file."""
    # If no path provided, use default repo-relative path
    if not store_path:
        store_path = get_default_genre_store_path()
        print(f"No genre store path provided, checking default: {store_path}")
    
    if not Path(store_path).exists():
        print(f"No existing genre embedding store found at: {store_path}")
        print("Will create new store at this location.")
        return {}
    
    try:
        with np.load(store_path, allow_pickle=True) as data:
            genre_keys = data['genre_keys']
            embeddings = data['embeddings']
            
            # Convert back to dictionary
            store = {}
            for i, key in enumerate(genre_keys):
                store[key] = embeddings[i].tolist()
            
            print(f"Loaded genre embedding store with {len(store)} genres from: {store_path}")
            return store
    except Exception as e:
        print(f"Warning: Could not load genre store from {store_path}: {e}")
        print("Creating new store.")
        return {}

def get_default_genre_store_path() -> str:
    """Get the default genre store path relative to the repository root."""
    # Get the repository root (parent of the scripts directory)
    repo_root = Path(__file__).parent.parent
    return str(repo_root / "data" / "genre_embedding_store.npz")

def save_genre_embedding_store(store: Dict[str, List[float]], store_path: str) -> str:
    """Save the shared genre embedding store to a .npz file. Returns the actual path used."""
    if not store:
        print("Warning: Empty genre store, not saving.")
        return None
    
    # If no path provided, use default repo-relative path
    if not store_path:
        store_path = get_default_genre_store_path()
        print(f"No genre store path provided, using default: {store_path}")
    
    # Convert dictionary to arrays
    genre_keys = list(store.keys())
    embeddings = [store[key] for key in genre_keys]
    
    # Ensure output directory exists
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        store_path,
        genre_keys=np.array(genre_keys),
        embeddings=np.array(embeddings)
    )
    
    print(f"Saved genre embedding store with {len(store)} genres to: {store_path}")
    return store_path

def format_individual_sections(profile_data):
    """Format individual sections. For genres, returns individual genre objects. For others, comma-separated strings."""
    # Handle genres specially - return list of genre objects for individual embedding
    genres_data = []
    if profile_data.get('genres'):
        for genre_obj in profile_data['genres']:
            if isinstance(genre_obj, dict):
                genre_name = genre_obj.get('name', '')
                prominence = genre_obj.get('prominence', 1)
                genres_data.append({
                    'text': f"genre: {genre_name}",
                    'name': genre_name,
                    'prominence': prominence
                })
    
    # Convert other arrays to comma-separated strings (no prefixes) with safe field access
    vocal_style_text = ", ".join(profile_data.get('vocal_style') or []) if profile_data.get('vocal_style') else ""
    production_sound_design_text = ", ".join(profile_data.get('production_sound_design') or []) if profile_data.get('production_sound_design') else ""
    lyrical_themes_text = ", ".join(profile_data.get('lyrical_themes') or []) if profile_data.get('lyrical_themes') else ""
    mood_atmosphere_text = ", ".join(profile_data.get('mood_atmosphere') or []) if profile_data.get('mood_atmosphere') else ""
    cultural_context_scene_text = ", ".join(profile_data.get('cultural_context_scene') or []) if profile_data.get('cultural_context_scene') else ""

    vocal_style_text = f"vocal style: {vocal_style_text}"
    production_sound_design_text = f"production & sound design: {production_sound_design_text}"
    lyrical_themes_text = f"lyrical themes: {lyrical_themes_text}"
    mood_atmosphere_text = f"mood & atmosphere: {mood_atmosphere_text}"
    cultural_context_scene_text = f"cultural context & scene: {cultural_context_scene_text}"
    
    return genres_data, vocal_style_text, production_sound_design_text, lyrical_themes_text, mood_atmosphere_text, cultural_context_scene_text



def log_text_examples(profiles: List[Dict], num_examples: int = 3):
    """Log examples of formatted text for sanity checking."""
    print(f"\n{'='*60}")
    print("TEXT FORMATTING EXAMPLES (for sanity checking):")
    print(f"{'='*60}")
    
    for i, profile in enumerate(profiles[:num_examples]):
        artist = profile['artist']
        print(f"\n--- Example {i+1}: {artist} ---")
        
        # Get formatted texts
        genres_data, vocal_style_text, production_sound_design_text, lyrical_themes_text, mood_atmosphere_text, cultural_context_scene_text = format_individual_sections(profile)
        
        print(f"\n1. GENRES DATA:")
        for j, genre_data in enumerate(genres_data):
            print(f"   {j+1}. {genre_data['text']} (prominence: {genre_data['prominence']})")
        
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
        
        print(f"\n{'-'*50}")
    
    print(f"\n{'='*60}")
    print("END OF TEXT EXAMPLES")
    print(f"{'='*60}\n")

# ──────────────────────────────── ASYNC WORKER ────────────────────────────────
client = AsyncOpenAI()
sem = asyncio.Semaphore(MAX_CONCURRENCY)
genre_store_lock = asyncio.Lock()  # Protect genre store from race conditions

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

async def process_artist_profile(profile_data: Dict[str, Any], genre_store: Dict[str, List[float]]) -> Dict[str, Any]:
    """Process a single artist profile and generate embeddings."""
    artist = profile_data.get('artist', 'Unknown Artist')
    
    # Get the formatted text strings using existing functions
    genres_data, vocal_style_text, production_sound_design_text, lyrical_themes_text, mood_atmosphere_text, cultural_context_scene_text = format_individual_sections(profile_data)
    
    # Handle genre embeddings with cache-first approach
    genres_with_keys = []
    new_genres_to_embed = []
    
    # Check cache and collect genres that need embedding (thread-safe read)
    async with genre_store_lock:
        for genre_data in genres_data:
            genre_key = genre_data['text']  # This is "genre: {genre_name}"
            
            if genre_key in genre_store:
                # Already have this genre embedding, just store the key and prominence
                genres_with_keys.append({
                    'key': genre_key,
                    'name': genre_data['name'],
                    'prominence': genre_data['prominence']
                })
            else:
                # New genre, need to embed it
                genres_with_keys.append({
                    'key': genre_key,
                    'name': genre_data['name'],
                    'prominence': genre_data['prominence']
                })
                new_genres_to_embed.append(genre_key)
    
    # Embed only the new genres (outside lock to avoid blocking)
    if new_genres_to_embed:
        new_genre_embeddings = await asyncio.gather(*[get_embedding(genre_key) for genre_key in new_genres_to_embed])
        
        # Add new embeddings to the store (thread-safe write)
        async with genre_store_lock:
            for i, genre_key in enumerate(new_genres_to_embed):
                # Double-check in case another task added it while we were embedding
                if genre_key not in genre_store:
                    genre_store[genre_key] = new_genre_embeddings[i]
    
    # Generate other embeddings concurrently
    vocal_style_embedding, production_sound_design_embedding, lyrical_themes_embedding, mood_atmosphere_embedding, cultural_context_scene_embedding = await asyncio.gather(
        get_embedding(vocal_style_text),
        get_embedding(production_sound_design_text),
        get_embedding(lyrical_themes_text), 
        get_embedding(mood_atmosphere_text),
        get_embedding(cultural_context_scene_text)
    )
    
    return {
        "artist": artist,
        "lead_vocalist_gender": profile_data.get('lead_vocalist_gender', 'N/A'),
        "embeddings": {
            "genres": genres_with_keys,  # Now contains genre keys and prominence scores only
            "vocal_style": vocal_style_embedding,
            "production_sound_design": production_sound_design_embedding,
            "lyrical_themes": lyrical_themes_embedding,
            "mood_atmosphere": mood_atmosphere_embedding,
            "cultural_context_scene": cultural_context_scene_embedding
        },
        "original_texts": {
            "vocal_style": vocal_style_text,
            "production_sound_design": production_sound_design_text,
            "lyrical_themes": lyrical_themes_text,
            "mood_atmosphere": mood_atmosphere_text,
            "cultural_context_scene": cultural_context_scene_text
        }
    }

def save_embeddings_numpy(results: List[Dict], output_path: str):
    """Save embeddings in separate files by type for the new structure."""
    # Determine if output_path is a directory or a base filename
    output_path_obj = Path(output_path)
    
    output_dir = output_path_obj
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle genres separately due to individual embedding structure
    save_genre_embeddings(results, output_dir)
    
    # Prepare data for other embedding types (unchanged structure)
    embedding_types = ['vocal_style', 'production_sound_design', 'lyrical_themes', 'mood_atmosphere', 'cultural_context_scene']
    
    for embedding_type in embedding_types:
        artists = []
        embeddings = []
        artist_indices = []
        field_values = []
        lead_vocalist_genders = []
        
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
            lead_vocalist_genders=np.array(lead_vocalist_genders)
        )
        
        print(f"Saved {embedding_type} embeddings: {output_file}")
        print(f"  Shape: {embeddings_array.shape}")
    
    print(f"\nAll embedding files saved to: {output_dir}")
    print(f"Generated {len(embedding_types) + 1} embedding files for {len(results)} artists")

def save_genre_embeddings(results: List[Dict], output_dir: Path):
    """Save genre data with keys and prominence scores only (embeddings stored in shared store)."""
    # Collect all individual genre data across all artists
    all_artists = []
    all_artist_indices = []
    all_genre_keys = []
    all_genre_names = []
    all_prominence_scores = []
    all_lead_vocalist_genders = []
    
    for i, result in enumerate(results):
        artist = result['artist']
        lead_vocalist_gender = result['lead_vocalist_gender']
        genres_data = result['embeddings']['genres']
        
        for genre_data in genres_data:
            all_artists.append(artist)
            all_artist_indices.append(i)
            all_genre_keys.append(genre_data['key'])  # Store the full key (e.g., "genre: pop")
            all_genre_names.append(genre_data['name'])  # Store just the genre name (e.g., "pop")
            all_prominence_scores.append(genre_data['prominence'])
            all_lead_vocalist_genders.append(lead_vocalist_gender)
    
    # Convert to numpy arrays
    artist_indices_array = np.array(all_artist_indices)
    prominence_scores_array = np.array(all_prominence_scores)
    
    # Save genre data file (no embeddings, just references to shared store)
    output_file = output_dir / "genres_artist_embeddings.npz"
    np.savez_compressed(
        output_file,
        artists=np.array(all_artists),
        artist_indices=artist_indices_array,
        genre_keys=np.array(all_genre_keys),  # Full keys for looking up in shared store
        genre_names=np.array(all_genre_names),  # Just names for convenience
        prominence_scores=prominence_scores_array,
        lead_vocalist_genders=np.array(all_lead_vocalist_genders)
    )
    
    print(f"Saved genres data: {output_file}")
    print(f"  Total individual genres: {len(all_genre_keys)}")
    print(f"  Data includes keys and prominence scores (embeddings in shared store)")

def save_embeddings_json(results: List[Dict], output_file: str):
    """Save embeddings in JSON format (human-readable but less efficient)."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Embeddings saved to {output_file} (JSON format)")

# ──────────────────────────────── MAIN DRIVER ────────────────────────────────
async def main(input_file: str, output_path: str, output_format: str = "numpy", num_entries: int = None, genre_store_path: str = None):
    """Main function to process all artist profiles."""
    # Load shared genre embedding store
    genre_store = load_genre_embedding_store(genre_store_path)
    initial_store_size = len(genre_store)
    
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
    # Count total embeddings: individual genres + 5 other types per artist
    total_genre_count = sum(len(profile.get('genres', [])) for profile in profiles)
    total_other_embeddings = len(profiles) * 5
    print(f"Generating individual genre embeddings + 5 other embeddings per artist")
    print(f"  - {total_genre_count} individual genres across all artists")
    print(f"  - {total_other_embeddings} other embeddings (5 per artist)")
    
    # Log text examples for sanity checking
    log_text_examples(profiles, num_examples=min(3, len(profiles)))

    print(f"Skipped {num_skipped} artist profiles")
    
    # Process all profiles concurrently
    results = []
    tasks = [asyncio.create_task(process_artist_profile(profile, genre_store)) for profile in profiles]
    
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
    
    # Save updated genre store
    actual_save_path = save_genre_embedding_store(genre_store, genre_store_path)
    
    # Report genre store statistics
    final_store_size = len(genre_store)
    new_genres_added = final_store_size - initial_store_size
    
    print(f"\n{'='*60}")
    print(f"GENRE EMBEDDING STORE STATISTICS")
    print(f"{'='*60}")
    print(f"Genre store size at start: {initial_store_size}")
    print(f"Genre store size at end: {final_store_size}")
    print(f"New genres added: {new_genres_added}")
    if actual_save_path:
        print(f"Genre store saved to: {actual_save_path}")
    else:
        print("Note: Genre store was not saved (empty store)")
    print(f"{'='*60}")

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
    parser.add_argument("--genre-store", type=str, default=None,
                        help="Path to shared genre embedding store (.npz file). Creates new store if not provided.")
    
    args = parser.parse_args()
    
    t0 = time.perf_counter()
    asyncio.run(main(args.input, args.output, args.format, args.num_entries, args.genre_store))
    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed:.1f}s with {MAX_CONCURRENCY}-way concurrency")