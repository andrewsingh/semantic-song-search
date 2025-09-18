#!/usr/bin/env python3
"""
Generate text embeddings for song profiles using OpenAI's text-embedding-3-large model.
Creates 6 embeddings per song using the new SongProfile V6 schema with batch processing
and resume capability for large-scale processing.

EMBEDDING_TYPES (V6 Schema):
  1. genres: Genre descriptors (comma-separated string)
  2. vocal_style: Vocal characteristic descriptors (comma-separated string)
  3. production_sound_design: Production and sonic descriptors (comma-separated string)
  4. lyrical_meaning: Lyrical meaning descriptors (comma-separated string)
  5. mood_atmosphere: Mood and atmospheric descriptors (comma-separated string)
  6. tags: Single-word tags (comma-separated string)

Features:
  - Batch processing: Process songs in configurable batches with intermediate saving
  - Resume capability: Automatically detects and skips completed batches on restart
  - Fault tolerance: If the script crashes, progress is preserved and can be resumed
  - Progress tracking: Shows batch-by-batch progress with detailed logging

Usage:
  # Generate all embedding types with default batch size:
  python embed_song_profiles.py -i song_profiles_v6.jsonl -o data/embeddings/

  # Generate with custom batch size (larger batches = fewer saves, more risk):
  python embed_song_profiles.py -i song_profiles_v6.jsonl -o data/embeddings/ -b 50

  # Generate specific embedding types:
  python embed_song_profiles.py -i song_profiles_v6.jsonl -o data/embeddings/ \
                                --embed_types genres vocal_style tags

  # Disable resume mode (start from beginning):
  python embed_song_profiles.py -i song_profiles_v6.jsonl -o data/embeddings/ --no_resume

  # Resume after a crash (default behavior):
  python embed_song_profiles.py -i song_profiles_v6.jsonl -o data/embeddings/

  # Test with first N songs:
  python embed_song_profiles.py -i song_profiles_v6.jsonl -o data/embeddings/ -n 100

Batch Processing:
  - Each batch saves intermediate files: {type}_embeddings_batch_{num:04d}.npz
  - After all batches complete, batch files are merged into final files
  - Batch files are automatically cleaned up after successful merge
  - Resume mode checks for existing batch files and skips completed batches

Output Files:
  # Final output files (after batch merge):
  # - {type}_embeddings.npz for each embedding type containing:
  #   - songs: song names (from prompt_vars)
  #   - artists: artist names (from prompt_vars)
  #   - main_artists: main artist names (from prompt_vars)
  #   - uris: track URIs (from top-level profile)
  #   - embeddings: embedding vectors for this type
  #   - song_indices: which song each embedding belongs to
  #   - field_values: original descriptor text that was embedded
"""
import argparse, asyncio, json, time, os, glob
from pathlib import Path
from typing import List, Dict, Any, Set
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
EMBEDDING_TYPES = ['genres', 'vocal_style', 'production_sound_design', 'lyrical_meaning', 'mood_atmosphere', 'tags']

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
    
    genres_text = safe_join('genres').lower()
    vocal_style_text = safe_join('vocal_style').lower()
    production_sound_design_text = safe_join('production_sound_design').lower()
    lyrical_meaning_text = safe_join('lyrical_meaning').lower()
    mood_atmosphere_text = safe_join('mood_atmosphere').lower()
    tags_text = safe_join('tags').lower()

    # Add prefixes to match artist embedding format
    genres_text = f"genres: {genres_text}"
    vocal_style_text = f"vocal style: {vocal_style_text}"
    production_sound_design_text = f"production & sound design: {production_sound_design_text}"
    lyrical_meaning_text = f"lyrical meaning: {lyrical_meaning_text}"
    mood_atmosphere_text = f"mood & atmosphere: {mood_atmosphere_text}"

    return genres_text, vocal_style_text, production_sound_design_text, lyrical_meaning_text, mood_atmosphere_text, tags_text


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
        genres_text, vocal_style_text, production_sound_design_text, lyrical_meaning_text, mood_atmosphere_text, tags_text = format_individual_sections(profile)

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

        print(f"\n6. TAGS TEXT:")
        print(f"   {tags_text}")

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
    genres_text, vocal_style_text, production_sound_design_text, lyrical_meaning_text, mood_atmosphere_text, tags_text = format_individual_sections(profile_data)

    # Map embedding types to their texts
    text_map = {
        "genres": genres_text,
        "vocal_style": vocal_style_text,
        "production_sound_design": production_sound_design_text,
        "lyrical_meaning": lyrical_meaning_text,
        "mood_atmosphere": mood_atmosphere_text,
        "tags": tags_text
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

def save_embeddings_numpy(results: List[Dict], output_path: str, embed_types: List[str], batch_num: int = None, batch_start_idx: int = 0):
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
                # Use global song index, not batch-local index
                song_indices.append(batch_start_idx + i)

                # Get the original text that was embedded for this type
                field_values.append(result['original_texts'][embedding_type])
        
        if not embeddings:  # Skip if no embeddings for this type
            continue
            
        # Convert to numpy arrays
        embeddings_array = np.array(embeddings)
        song_indices_array = np.array(song_indices)
        
        # Save individual embedding type file with optional batch number
        if batch_num is not None:
            output_file = output_dir / f"{embedding_type}_embeddings_batch_{batch_num:04d}.npz"
        else:
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
    
    if batch_num is not None:
        print(f"\nBatch {batch_num} embedding files saved to: {output_dir}")
    else:
        print(f"\nAll embedding files saved to: {output_dir}")
    print(f"Generated {len(embed_types)} embedding files for {len(results)} songs")

def get_completed_batches(output_path: str, embed_types: List[str]) -> Set[int]:
    """Identify which batches have already been completed by checking for output files."""
    output_dir = Path(output_path)
    if not output_dir.exists():
        return set()

    completed_batches = set()

    # Check for batch files - a batch is complete if ALL embedding types exist
    for embed_type in embed_types:
        pattern = str(output_dir / f"{embed_type}_embeddings_batch_*.npz")
        batch_files = glob.glob(pattern)

        type_batches = set()
        for file_path in batch_files:
            # Extract batch number from filename
            filename = Path(file_path).name
            if '_batch_' in filename:
                try:
                    batch_str = filename.split('_batch_')[1].split('.npz')[0]
                    batch_num = int(batch_str)
                    type_batches.add(batch_num)
                except (ValueError, IndexError):
                    continue

        if not completed_batches:
            completed_batches = type_batches
        else:
            # Only keep batches that exist for ALL embedding types
            completed_batches = completed_batches.intersection(type_batches)

    return completed_batches

def merge_batch_files(output_path: str, embed_types: List[str], keep_batch_files: bool = False):
    """Merge all batch files into final combined files and clean up batch files."""
    output_dir = Path(output_path)

    print("\nMerging batch files into final embeddings...")

    merged_any = False
    for embed_type in embed_types:
        # Find all batch files for this embedding type
        pattern = str(output_dir / f"{embed_type}_embeddings_batch_*.npz")
        batch_files = sorted(glob.glob(pattern))

        if not batch_files:
            print(f"Warning: No batch files found for {embed_type}")
            continue

        # Load and combine all batch data
        all_songs = []
        all_artists = []
        all_main_artists = []
        all_uris = []
        all_embeddings = []
        all_song_indices = []
        all_field_values = []

        for batch_file in batch_files:
            try:
                data = np.load(batch_file)
                all_songs.extend(data['songs'])
                all_artists.extend(data['artists'])
                all_main_artists.extend(data['main_artists'])
                all_uris.extend(data['uris'])
                all_embeddings.extend(data['embeddings'])
                all_song_indices.extend(data['song_indices'])
                all_field_values.extend(data['field_values'])
            except Exception as e:
                print(f"Error loading batch file {batch_file}: {e}")
                continue

        if not all_embeddings:
            print(f"Warning: No valid data found for {embed_type}")
            continue

        # Save combined file
        final_file = output_dir / f"{embed_type}_embeddings.npz"
        try:
            np.savez_compressed(
                final_file,
                songs=np.array(all_songs),
                artists=np.array(all_artists),
                main_artists=np.array(all_main_artists),
                uris=np.array(all_uris),
                embeddings=np.array(all_embeddings),
                song_indices=np.array(all_song_indices),
                field_values=np.array(all_field_values)
            )

            print(f"Merged {len(batch_files)} batch files into {final_file}")
            print(f"  Total songs: {len(all_songs)}, Embeddings shape: {np.array(all_embeddings).shape}")
            merged_any = True

            # Clean up batch files only after successful merge (unless keeping them)
            if not keep_batch_files:
                for batch_file in batch_files:
                    try:
                        os.remove(batch_file)
                    except Exception as e:
                        print(f"Warning: Could not remove batch file {batch_file}: {e}")
            else:
                print(f"Batch files preserved for debugging")

        except Exception as e:
            print(f"Error saving merged file {final_file}: {e}")
            continue

    if merged_any:
        if keep_batch_files:
            print("Batch file merge completed. Batch files preserved for debugging.")
        else:
            print("Batch file merge completed and temporary files cleaned up.")
    else:
        print("No batch files were successfully merged.")

# ──────────────────────────────── MAIN DRIVER ────────────────────────────────
async def main(input_file: str, output_path: str, embed_types: List[str], num_entries: int = None, batch_size: int = 100, resume: bool = True, keep_batch_files: bool = False):
    """Main function to process all song profiles with batch processing and resume capability."""
    print(f"Loading profiles from {input_file}")
    print(f"Generating embedding types: {', '.join(embed_types)}")
    print(f"Batch size: {batch_size}")

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

    # Validate inputs
    if len(profiles) == 0:
        print("No profiles to process. Exiting.")
        return

    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0")

    print(f"Total profiles to process: {len(profiles)}")
    print(f"Generating {len(embed_types)} embeddings per song (total: {len(profiles) * len(embed_types)} embeddings)")
    print(f"Skipped {num_skipped} song profiles (not familiar)")

    # Calculate number of batches
    num_batches = (len(profiles) + batch_size - 1) // batch_size
    print(f"Processing in {num_batches} batches of {batch_size} songs each")

    # Check for existing completed batches if resume is enabled
    completed_batches = set()
    if resume:
        completed_batches = get_completed_batches(output_path, embed_types)
        if completed_batches:
            print(f"Resume mode: Found {len(completed_batches)} completed batches: {sorted(completed_batches)}")
        else:
            print("Resume mode: No completed batches found, starting from beginning")

    # Log text examples for sanity checking (only for first batch)
    if not completed_batches or 0 not in completed_batches:
        log_text_examples(profiles, num_examples=min(3, len(profiles)))

    # Process profiles in batches
    batches_processed = 0
    total_songs_processed = 0
    total_songs_failed = 0
    failed_songs = []

    for batch_num in range(num_batches):
        if batch_num in completed_batches:
            print(f"\nSkipping batch {batch_num} (already completed)")
            continue

        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(profiles))
        batch_profiles = profiles[start_idx:end_idx]

        print(f"\nProcessing batch {batch_num}/{num_batches-1} ({len(batch_profiles)} songs)...")

        # Process batch concurrently
        batch_results = []
        tasks = [asyncio.create_task(process_song_profile(profile, embed_types)) for profile in batch_profiles]

        batch_failed_count = 0

        with tqdm(total=len(tasks), unit="song", desc=f"Batch {batch_num}") as pbar:
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    # Validate result structure before adding
                    if (result and isinstance(result, dict) and
                        'song' in result and 'embeddings' in result and
                        'original_texts' in result):
                        batch_results.append(result)
                        total_songs_processed += 1
                    else:
                        song_name = result.get('song', 'Unknown Song') if result else 'Unknown Song'
                        print(f"Warning: Skipping malformed result for song: {song_name}")
                        failed_songs.append(song_name)
                        batch_failed_count += 1
                        total_songs_failed += 1
                except Exception as e:
                    print(f"Error processing song profile: {e}")
                    failed_songs.append("Unknown Song")
                    batch_failed_count += 1
                    total_songs_failed += 1
                    # Continue processing other songs instead of failing completely
                pbar.update(1)

        # Save batch results immediately
        if batch_results:
            save_embeddings_numpy(batch_results, output_path, embed_types, batch_num, start_idx)
            print(f"Batch {batch_num} saved with {len(batch_results)} songs (failed: {batch_failed_count})")
            batches_processed += 1
        else:
            print(f"Warning: Batch {batch_num} produced no valid results (failed: {batch_failed_count})")

    # Print final statistics
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY:")
    print(f"{'='*60}")
    print(f"Total songs loaded: {len(profiles)}")
    print(f"Songs successfully processed: {total_songs_processed}")
    print(f"Songs failed: {total_songs_failed}")
    print(f"Success rate: {total_songs_processed/(total_songs_processed+total_songs_failed)*100:.1f}%")
    print(f"Batches processed: {batches_processed}")

    if failed_songs:
        print(f"\nFailed songs ({len(failed_songs)} total):")
        for song in failed_songs[:10]:  # Show first 10
            print(f"  - {song}")
        if len(failed_songs) > 10:
            print(f"  ... and {len(failed_songs)-10} more")
    print(f"{'='*60}\n")

    # Only merge if we actually processed some batches or if there are existing batch files
    output_dir = Path(output_path)
    has_batch_files = any(glob.glob(str(output_dir / f"{et}_embeddings_batch_*.npz")) for et in embed_types)

    if batches_processed > 0 or has_batch_files:
        # Merge all batch files into final files
        merge_batch_files(output_path, embed_types, keep_batch_files)
    else:
        print("No batch files to merge.")


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
    parser.add_argument("-b", "--batch_size", type=int, default=500,
                        help="Number of songs to process per batch (default: 500)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Disable resume mode and start from beginning")
    parser.add_argument("--keep_batch_files", action="store_true",
                        help="Keep batch files after merging (useful for debugging)")
    
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
    print(f"Batch size: {args.batch_size}")
    print(f"Resume mode: {not args.no_resume}")

    asyncio.run(main(args.input, args.output, embed_types, args.num_entries, args.batch_size, not args.no_resume, args.keep_batch_files))
    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed:.1f}s with {MAX_CONCURRENCY}-way concurrency") 