#!/usr/bin/env python3
"""
Generate text embeddings for song profiles using OpenAI's text-embedding-3-large model.
Creates 5 embeddings per song optimized for intelligent music search:

SONG-TO-SONG SEARCH (4 embeddings):
  1. Full profile: Complete song information formatted as structured text
  2. Sound aspect: "Sound of [song] by [artist]: [description]"  
  3. Meaning aspect: "Meaning of [song] by [artist]: [description]"
  4. Mood aspect: "Mood of [song] by [artist]: [description]"

TEXT-TO-SONG SEARCH (1 embedding):
  5. Tags + genres: Comma-separated list of tags and genres for query matching

Supports batch processing with automatic saving and crash recovery.

Usage:
  python embed_song_profiles.py -i song_profiles.jsonl \
                                -o song_embeddings.npz \
                                -b 50   # optional: batch size (default: 50 songs)
                                -n 100  # optional: for testing first N songs
  
  # The output will be a .npz file containing:
  # - songs/artists: song names and artists
  # - embeddings: all embedding vectors  
  # - song_indices: which song each embedding belongs to
  # - field_types: embedding type (full_profile, sound_aspect, etc.)
  # - field_values: original text that was embedded
  
  # Resumable: if the script crashes, just run it again with the same output file
"""
import argparse, asyncio, json, time
from pathlib import Path
from typing import List, Dict, Any, Tuple
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

print(f"Rate limit settings: {RATE_LIMIT_RPM} RPM, {MAX_CONCURRENCY} concurrent requests")

def format_full_profile(profile: Dict) -> str:
    """Format a complete song profile for song-to-song search."""
    # Format genres and tags as comma-separated strings
    genres_str = ', '.join(profile.get('genres', []))
    tags_str = ', '.join(profile.get('tags', []))
    
    # Handle missing fields gracefully
    sound = profile.get('sound', 'N/A')
    meaning = profile.get('meaning', 'N/A')
    mood = profile.get('mood', 'N/A')
    
    return f"""Genres: {genres_str}
Tags: {tags_str}
Sound:
{sound}
Meaning:
{meaning}
Mood:
{mood}"""


def format_aspect_embedding(profile: Dict, aspect: str) -> str:
    """Format individual aspect (sound/meaning/mood) for song-to-song search."""
    aspect_value = profile.get(aspect, 'N/A')
    return aspect_value


def format_tags_genres(profile: Dict) -> str:
    """Format tags + genres for text-to-song search."""
    tags = profile.get('tags', [])
    genres = profile.get('genres', [])
    
    # Combine tags first, then genres, as comma-separated string
    all_labels = tags + genres
    return ', '.join(all_labels)


def extract_embedding_tasks(profiles: List[Dict]) -> Tuple[List[Dict], List[Tuple]]:
    """
    Extract the 5 embedding tasks per song profile for music search.
    
    Returns:
        songs_metadata: List of dicts with song info
        embedding_tasks: List of tuples (song_idx, embedding_type, text_value)
    """
    songs_metadata = []
    embedding_tasks = []
    
    for song_idx, profile in enumerate(profiles):
        # Store song metadata
        songs_metadata.append({
            'song': profile['original_song'],
            'artist': profile['original_artist']
        })
        
        # 1. Full profile embedding (song-to-song search)
        full_profile_text = format_full_profile(profile)
        embedding_tasks.append((song_idx, 'full_profile', full_profile_text))
        
        # 2-4. Individual aspect embeddings (song-to-song search)
        for aspect in ['sound', 'meaning', 'mood']:
            if profile.get(aspect):  # Only create embedding if aspect exists
                aspect_text = format_aspect_embedding(profile, aspect)
                embedding_tasks.append((song_idx, f'{aspect}_aspect', aspect_text))
        
        # 5. Tags + genres embedding (text-to-song search)
        if profile.get('tags') or profile.get('genres'):  # Only if we have tags or genres
            tags_genres_text = format_tags_genres(profile)
            if tags_genres_text.strip():  # Only if non-empty
                embedding_tasks.append((song_idx, 'tags_genres', tags_genres_text))
    
    return songs_metadata, embedding_tasks

def log_embedding_examples(songs_metadata: List[Dict], embedding_tasks: List[Tuple], num_examples: int = 2):
    """Log examples of the 5 embedding types for sanity checking."""
    print(f"\n{'='*80}")
    print("EMBEDDING EXAMPLES (for sanity checking):")
    print(f"{'='*80}")
    
    # Group tasks by song for better readability
    tasks_by_song = {}
    for song_idx, embedding_type, text_value in embedding_tasks:
        if song_idx not in tasks_by_song:
            tasks_by_song[song_idx] = []
        tasks_by_song[song_idx].append((embedding_type, text_value))
    
    # Show first few songs with all their embeddings
    for i, (song_idx, tasks) in enumerate(list(tasks_by_song.items())[:num_examples]):
        song_info = songs_metadata[song_idx]
        print(f"\n--- Example {i+1}: '{song_info['song']}' by '{song_info['artist']}' ---")
        
        # Show embeddings in a logical order
        embedding_order = ['full_profile', 'sound_aspect', 'meaning_aspect', 'mood_aspect', 'tags_genres']
        tasks_dict = {embedding_type: text_value for embedding_type, text_value in tasks}
        
        for embedding_type in embedding_order:
            if embedding_type in tasks_dict:
                text_value = tasks_dict[embedding_type]
                print(f"\n  {embedding_type.upper().replace('_', ' ')}:")
                
                # Special handling for different embedding types
                if embedding_type == 'full_profile':
                    # Show first few lines for full profile
                    lines = text_value.split('\n')[:4]
                    for line in lines:
                        print(f"    {line}")
                    if len(text_value.split('\n')) > 4:
                        print(f"    ... ({len(text_value.split('\n')) - 4} more lines)")
                elif embedding_type == 'tags_genres':
                    # Show tags/genres nicely formatted
                    print(f"    {text_value}")
                else:
                    # Show individual aspects with truncation
                    display_value = text_value if len(text_value) <= 120 else text_value[:117] + "..."
                    print(f"    {display_value}")
    
    print(f"\n{'='*80}")
    print(f"TOTAL EMBEDDING TASKS: {len(embedding_tasks)}")
    
    # Show breakdown by embedding type
    type_counts = {}
    for _, embedding_type, _ in embedding_tasks:
        type_counts[embedding_type] = type_counts.get(embedding_type, 0) + 1
    
    print("BREAKDOWN BY EMBEDDING TYPE:")
    for embedding_type, count in sorted(type_counts.items()):
        print(f"  {embedding_type}: {count} embeddings")
    
    expected_per_song = len(type_counts)
    actual_per_song = len(embedding_tasks) / len(tasks_by_song) if tasks_by_song else 0
    print(f"\nAverage embeddings per song: {actual_per_song:.1f} (expected: ~{expected_per_song})")
    print(f"{'='*80}\n")

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
        empty_embeddings = np.empty((0, 3072), dtype=np.float32)
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
        embeddings = data['embeddings'].astype(np.float32)  # Ensure consistent dtype
        song_indices = data['song_indices'].astype(int)
        field_types = data['field_types']  # Keep as loaded (should be object)
        field_values = data['field_values']  # Keep as loaded (should be object)
        
        print(f"✅ Loading complete! Ready to process remaining songs.")
        return songs_metadata, embeddings, song_indices, field_types, field_values
        
    except Exception as e:
        print(f"Warning: Could not load existing embeddings: {e}")
        print("Starting fresh...")
        empty_embeddings = np.empty((0, 3072), dtype=np.float32)
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
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save everything in one organized .npz file
    np.savez_compressed(
        output_file,
        # Song metadata
        songs=song_names,
        artists=artist_names,
        # Embedding data
        embeddings=embeddings,
        song_indices=song_indices,
        field_types=field_types,
        field_values=field_values
    )

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
            adjusted_song_idx = len(existing_songs_metadata) + song_idx
            for task in tasks_by_song[song_idx]:
                # Store the correct adjusted song index for each task
                batch_tasks_flat.append((adjusted_song_idx, task['field_type'], task['text_value']))

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
        
        # Create numpy arrays for the new batch data with proper dtypes
        new_embeddings = np.array(batch_results, dtype=np.float32)
        new_song_indices = np.array([task[0] for task in batch_tasks_flat], dtype=int)
        new_field_types = np.array([task[1] for task in batch_tasks_flat], dtype=object)
        new_field_values = np.array([task[2] for task in batch_tasks_flat], dtype=object)

        # Concatenate with existing data - handle empty case properly
        if len(final_embeddings) == 0:
            final_embeddings = new_embeddings
            final_song_indices = new_song_indices
            final_field_types = new_field_types
            final_field_values = new_field_values
        else:
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