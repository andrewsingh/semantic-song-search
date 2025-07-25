#!/usr/bin/env python3
"""
Generate text embeddings for song profiles using OpenAI's text-embedding-3-large model.
Creates embeddings for individual strings in genres/tags and full strings for sound/meaning/mood.
Supports batch processing with automatic saving and crash recovery.

Usage:
  python embed_song_profiles.py -i song_profiles.jsonl \
                                -o song_embeddings.npz \
                                -b 50   # optional: batch size (default: 50 songs)
                                -n 100  # optional: for testing first N songs
  
  # The output will be a .npz file containing:
  # - songs/artists: song names and artists
  # - embeddings: all embedding vectors
  # - song_indices/field_types/field_values: metadata for each embedding
  
  # Resumable: if the script crashes, just run it again with the same output file
"""
import argparse, asyncio, json, time
from pathlib import Path
from typing import List, Dict, Any, Tuple
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

print(f"Rate limit settings: {RATE_LIMIT_RPM} RPM, {MAX_CONCURRENCY} concurrent requests")

def extract_embedding_tasks(profiles: List[Dict]) -> Tuple[List[Dict], List[Tuple]]:
    """
    Extract all individual strings that need embeddings from song profiles.
    
    Returns:
        songs_metadata: List of dicts with song info
        embedding_tasks: List of tuples (song_idx, field_type, text_value)
    """
    songs_metadata = []
    embedding_tasks = []
    
    for song_idx, profile in enumerate(profiles):
        # Store song metadata
        songs_metadata.append({
            'song': profile['original_song'],
            'artist': profile['original_artist']
        })
        
        # Extract individual genre strings
        if profile.get('genres'):
            for genre in profile['genres']:
                embedding_tasks.append((song_idx, 'genre', genre))
        
        # Extract individual tag strings  
        if profile.get('tags'):
            for tag in profile['tags']:
                embedding_tasks.append((song_idx, 'tag', tag))
        
        # Extract full field strings
        for field in ['sound', 'meaning', 'mood']:
            if profile.get(field):
                embedding_tasks.append((song_idx, field, profile[field]))
    
    return songs_metadata, embedding_tasks

def log_embedding_examples(songs_metadata: List[Dict], embedding_tasks: List[Tuple], num_examples: int = 3):
    """Log examples of texts that will be embedded for sanity checking."""
    print(f"\n{'='*70}")
    print("EMBEDDING TASKS EXAMPLES (for sanity checking):")
    print(f"{'='*70}")
    
    # Group tasks by song for better readability
    tasks_by_song = {}
    for song_idx, field_type, text_value in embedding_tasks:
        if song_idx not in tasks_by_song:
            tasks_by_song[song_idx] = []
        tasks_by_song[song_idx].append((field_type, text_value))
    
    # Show first few songs
    for i, (song_idx, tasks) in enumerate(list(tasks_by_song.items())[:num_examples]):
        song_info = songs_metadata[song_idx]
        print(f"\n--- Example {i+1}: '{song_info['song']}' by '{song_info['artist']}' ---")
        
        # Group by field type
        by_field = {}
        for field_type, text_value in tasks:
            if field_type not in by_field:
                by_field[field_type] = []
            by_field[field_type].append(text_value)
        
        for field_type, values in by_field.items():
            print(f"\n  {field_type.upper()}:")
            for value in values:
                # Truncate long values for display
                display_value = value if len(value) <= 100 else value[:97] + "..."
                print(f"    - \"{display_value}\"")
    
    print(f"\n{'='*70}")
    print(f"TOTAL EMBEDDING TASKS: {len(embedding_tasks)}")
    
    # Show breakdown by field type
    field_counts = {}
    for _, field_type, _ in embedding_tasks:
        field_counts[field_type] = field_counts.get(field_type, 0) + 1
    
    print("BREAKDOWN BY FIELD TYPE:")
    for field_type, count in sorted(field_counts.items()):
        print(f"  {field_type}: {count} embeddings")
    
    print(f"{'='*70}\n")

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

async def process_embedding_task(task_idx: int, song_idx: int, field_type: str, text_value: str) -> Tuple[int, List[float]]:
    """Process a single embedding task and return the result."""
    embedding = await get_embedding(text_value)
    return task_idx, embedding

def load_existing_embeddings(output_file: str) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load existing embeddings from file, keeping data as numpy arrays."""
    if not Path(output_file).exists():
        empty_embeddings = np.empty((0, 3072))
        empty_indices = np.array([], dtype=int)
        empty_types = np.array([], dtype=object)
        empty_values = np.array([], dtype=object)
        return [], empty_embeddings, empty_indices, empty_types, empty_values
    
    try:
        # Check file size first
        file_size = Path(output_file).stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        print(f"Loading existing embeddings file ({file_size_mb:.1f}MB)...")
        
        data = np.load(output_file, allow_pickle=True)
        
        print(f"Found {len(data['songs'])} songs with {len(data['embeddings'])} embeddings")
        
        # Reconstruct songs_metadata
        songs_metadata = [
            {'song': data['songs'][i], 'artist': data['artists'][i]} 
            for i in range(len(data['songs']))
        ]
        
        # Keep data as numpy arrays - much more efficient!
        embeddings = data['embeddings']
        song_indices = data['song_indices']
        field_types = data['field_types']
        field_values = data['field_values']
        
        print(f"✅ Loading complete! Ready to process remaining songs.")
        return songs_metadata, embeddings, song_indices, field_types, field_values
        
    except Exception as e:
        print(f"Warning: Could not load existing embeddings: {e}")
        print("Starting fresh...")
        empty_embeddings = np.empty((0, 3072))
        empty_indices = np.array([], dtype=int)
        empty_types = np.array([], dtype=object)
        empty_values = np.array([], dtype=object)
        return [], empty_embeddings, empty_indices, empty_types, empty_values

def save_song_embeddings(songs_metadata: List[Dict], 
                        embeddings: np.ndarray,
                        song_indices: np.ndarray,
                        field_types: np.ndarray,
                        field_values: np.ndarray,
                        output_file: str):
    """Save all song embeddings and metadata in organized numpy format."""
    
    # Create songs metadata arrays
    song_names = np.array([meta['song'] for meta in songs_metadata])
    artist_names = np.array([meta['artist'] for meta in songs_metadata])
    
    # Save everything in one organized .npz file
    # Use a temporary file and atomic rename to prevent corruption
    temp_output_file = f"{output_file}.tmp"
    np.savez_compressed(
        temp_output_file,
        # Song metadata
        songs=song_names,
        artists=artist_names,
        # Embedding data
        embeddings=embeddings,
        song_indices=song_indices,
        field_types=field_types,
        field_values=field_values
    )
    # Atomic rename to prevent corruption if the script is killed during save
    Path(temp_output_file).rename(output_file)

def print_save_summary(songs_metadata: List[Dict], 
                      embeddings: np.ndarray,
                      song_indices: np.ndarray,
                      field_types: np.ndarray,
                      field_values: np.ndarray,
                      output_file: str):
    """Print summary of saved embeddings."""
    print(f"\n{'='*50}")
    print("EMBEDDINGS SAVED SUCCESSFULLY")
    print(f"{'='*50}")
    print(f"Output file: {output_file}")
    print(f"Total songs: {len(songs_metadata)}")
    print(f"Total embeddings: {len(embeddings)}")
    if len(embeddings) > 0:
        print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Show field type breakdown
    unique_types, counts = np.unique(field_types, return_counts=True)
    
    print(f"\nEmbeddings by field type:")
    for field_type, count in zip(unique_types, counts):
        print(f"  {field_type}: {count}")
    
    print(f"\nTo load embeddings:")
    print(f"  data = np.load('{output_file}')")
    print(f"  songs = data['songs']")
    print(f"  embeddings = data['embeddings']")
    print(f"  song_indices = data['song_indices']")
    print(f"  field_types = data['field_types']")
    print(f"  field_values = data['field_values']")

# ──────────────────────────────── MAIN DRIVER ────────────────────────────────
async def main(input_file: str, output_file: str, num_entries: int = None, batch_size: int = 50):
    """Main function to process all song profiles and generate embeddings in batches."""
    print(f"Loading profiles from {input_file}")
    
    # Load existing embeddings if any
    existing_songs_metadata, existing_embeddings, existing_song_indices, existing_field_types, existing_field_values = load_existing_embeddings(output_file)
    
    # Create set of already-processed songs for deduplication
    processed_songs = set()
    if existing_songs_metadata:
        for meta in existing_songs_metadata:
            processed_songs.add((meta['song'], meta['artist']))
        print(f"Found {len(processed_songs)} already-processed songs")
    
    # Load profiles from JSONL file
    profiles_to_process = []
    unfamiliar_count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                profile = json.loads(line)
                if profile['familiar']:
                    song_key = (profile['original_song'], profile['original_artist'])
                    if song_key not in processed_songs:
                        profiles_to_process.append(profile)
                else:
                    unfamiliar_count += 1
    
    if num_entries:
        profiles_to_process = profiles_to_process[:num_entries]
    
    print(f"Loaded {len(profiles_to_process)} new song profiles to process.")
    print(f"Filtered out {unfamiliar_count} unfamiliar profiles.")
    
    if not profiles_to_process:
        print("No new songs to process - all songs have already been embedded!")
        return
    
    # Extract embedding tasks for new songs only
    new_songs_metadata, new_embedding_tasks = extract_embedding_tasks(profiles_to_process)
    
    print(f"Extracted {len(new_embedding_tasks)} new embedding tasks.")
    
    if new_embedding_tasks:
        log_embedding_examples(new_songs_metadata, new_embedding_tasks, num_examples=min(3, len(new_songs_metadata)))
    
    # Group tasks by song for batch processing
    tasks_by_song = {}
    for task_idx, (song_idx, field_type, text_value) in enumerate(new_embedding_tasks):
        if song_idx not in tasks_by_song:
            tasks_by_song[song_idx] = []
        tasks_by_song[song_idx].append({'task_idx': task_idx, 'field_type': field_type, 'text_value': text_value})
        
    # Process songs in batches
    song_indices_to_process = list(tasks_by_song.keys())
    total_batches = (len(song_indices_to_process) + batch_size - 1) // batch_size
    
    all_songs_metadata = existing_songs_metadata.copy()
    
    # These will be built up batch by batch
    final_embeddings = existing_embeddings
    final_song_indices = existing_song_indices
    final_field_types = existing_field_types
    final_field_values = existing_field_values
    
    print(f"\nProcessing {len(song_indices_to_process)} songs in {total_batches} batches of {batch_size}")
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(song_indices_to_process))
        batch_song_indices = song_indices_to_process[start_idx:end_idx]
        
        print(f"\n--- Batch {batch_num + 1}/{total_batches}: Processing songs {start_idx + 1}-{end_idx} ---")
        
        batch_tasks_flat = []
        for song_idx in batch_song_indices:
            all_songs_metadata.append(new_songs_metadata[song_idx])
            for task in tasks_by_song[song_idx]:
                adjusted_song_idx = len(existing_songs_metadata) + song_idx
                batch_tasks_flat.append((task['task_idx'], adjusted_song_idx, task['field_type'], task['text_value']))

        async_tasks = [
            asyncio.create_task(process_embedding_task(i, *task_info))
            for i, task_info in enumerate(batch_tasks_flat)
        ]
        
        batch_results = [None] * len(async_tasks)
        with tqdm(total=len(async_tasks), unit="embedding", desc=f"Batch {batch_num + 1}") as pbar:
            for coro in asyncio.as_completed(async_tasks):
                task_idx, embedding = await coro
                batch_results[task_idx] = embedding
                pbar.update(1)
        
        # Create numpy arrays for the new batch data
        new_embeddings = np.array(batch_results)
        new_song_indices = np.array([task[1] for task in batch_tasks_flat])
        new_field_types = np.array([task[2] for task in batch_tasks_flat])
        new_field_values = np.array([task[3] for task in batch_tasks_flat])

        # Concatenate with existing data
        final_embeddings = np.concatenate([final_embeddings, new_embeddings])
        final_song_indices = np.concatenate([final_song_indices, new_song_indices])
        final_field_types = np.concatenate([final_field_types, new_field_types])
        final_field_values = np.concatenate([final_field_values, new_field_values])
        
        save_song_embeddings(
            all_songs_metadata, final_embeddings, final_song_indices,
            final_field_types, final_field_values, output_file
        )
        print(f"Saved batch {batch_num + 1}/{total_batches} - Total: {len(all_songs_metadata)} songs, {len(final_embeddings)} embeddings")

    print_save_summary(
        all_songs_metadata, final_embeddings, final_song_indices,
        final_field_types, final_field_values, output_file
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for song profiles")
    parser.add_argument("-i", "--input", required=True, 
                        help="Input JSONL file with song profiles")
    parser.add_argument("-o", "--output", required=True, 
                        help="Output .npz file for embeddings")
    parser.add_argument("-n", "--num_entries", type=int, default=None, 
                        help="Process only first N entries (for testing)")
    parser.add_argument("-b", "--batch_size", type=int, default=50,
                        help="Number of songs to process before saving (default: 50)")
    
    args = parser.parse_args()
    
    t0 = time.perf_counter()
    asyncio.run(main(args.input, args.output, args.num_entries, args.batch_size))
    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed:.1f}s with {MAX_CONCURRENCY}-way concurrency") 