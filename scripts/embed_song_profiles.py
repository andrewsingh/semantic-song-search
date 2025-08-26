#!/usr/bin/env python3
"""
Generate text embeddings for song profiles using OpenAI's text-embedding-3-large model.
Creates embeddings per song optimized for semantic music search.

AVAILABLE EMBEDDING TYPES:
  1. full_profile: Complete song information formatted as structured text
  2. sound_aspect: Sound description only  
  3. meaning_aspect: Meaning description only
  4. mood_aspect: Mood description only
  5. tags_genres: Comma-separated list of tags and genres combined
  6. tags: Comma-separated list of tags only (NEW)
  7. genres: Comma-separated list of genres only (NEW)

Supports batch processing with automatic saving and crash recovery.
Each embedding type is stored in a separate .npz file for independent processing.

Usage:
  # Generate specific embedding types:
  python embed_song_profiles.py -i song_profiles.jsonl \
                                -o data/embeddings/ \
                                --embed_types tags genres
                                
  # Generate all embedding types:
  python embed_song_profiles.py -i song_profiles.jsonl \
                                -o data/embeddings/ \
                                --embed_types full_profile sound_aspect meaning_aspect mood_aspect tags_genres tags genres
                                
  # Optional flags:
  python embed_song_profiles.py -i song_profiles.jsonl \
                                -o data/embeddings/ \
                                --embed_types tags genres \
                                -b 50   # batch size (default: 50 songs)
                                -n 100  # for testing first N songs
  
  # The output will be separate .npz files for each embedding type containing:
  # - songs/artists: song names and artists
  # - embeddings: embedding vectors for this type
  # - song_indices: which song each embedding belongs to
  # - field_values: original text that was embedded
  
  # Resumable: if the script crashes, just run it again with the same arguments
"""
import argparse, asyncio, json, time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from openai import AsyncOpenAI, RateLimitError, APIError
from tqdm import tqdm
import pdb

# ──────────────────────────────── RATE-LIMIT SETTINGS ─────────────────────────
RATE_LIMIT_RPM = 5000
SAFETY_FACTOR = 0.80
RPS = RATE_LIMIT_RPM / 60
MAX_CONCURRENCY = max(1, int(RPS * SAFETY_FACTOR))
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = max(1, int(60 / RATE_LIMIT_RPM * 10))

# ──────────────────────────────── OPENAI SETTINGS ─────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-large"
EMBED_TYPES = ['full_profile', 'sound_aspect', 'meaning_aspect', 'mood_aspect', 'tags_genres', 'tags', 'genres']

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
    tags = profile.get('tags', []) or []
    genres = profile.get('genres', []) or []
    
    # Ensure we have lists (handle None values)
    if not isinstance(tags, list):
        tags = []
    if not isinstance(genres, list):
        genres = []
    
    # Combine tags first, then genres, as comma-separated string
    all_labels = tags + genres
    return ', '.join(all_labels)


def format_tags_only(profile: Dict) -> str:
    """Format tags only for text-to-song search."""
    tags = profile.get('tags', []) or []
    
    # Ensure we have a list (handle None values)
    if not isinstance(tags, list):
        tags = []
    
    return ', '.join(tags)


def format_genres_only(profile: Dict) -> str:
    """Format genres only for text-to-song search."""
    genres = profile.get('genres', []) or []
    
    # Ensure we have a list (handle None values)
    if not isinstance(genres, list):
        genres = []
    
    return ', '.join(genres)


def extract_embedding_tasks(profiles: List[Dict], embed_types: List[str]) -> Tuple[List[Dict], List[Tuple]]:
    """
    Extract embedding tasks per song profile for specified embedding types.
    
    Args:
        profiles: List of song profile dictionaries
        embed_types: List of embedding types to generate
    
    Returns:
        songs_metadata: List of dicts with song info
        embedding_tasks: List of tuples (song_idx, embedding_type, text_value)
    """
    # Define available embedding type functions
    embedding_formatters = {
        'full_profile': format_full_profile,
        'sound_aspect': lambda p: format_aspect_embedding(p, 'sound'),
        'meaning_aspect': lambda p: format_aspect_embedding(p, 'meaning'),
        'mood_aspect': lambda p: format_aspect_embedding(p, 'mood'),
        'tags_genres': format_tags_genres,
        'tags': format_tags_only,
        'genres': format_genres_only
    }
    
    # Validate requested embedding types
    valid_types = set(embedding_formatters.keys())
    invalid_types = set(embed_types) - valid_types
    if invalid_types:
        raise ValueError(f"Invalid embedding types: {invalid_types}. Valid types: {valid_types}")
    
    songs_metadata = []
    embedding_tasks = []
    
    for song_idx, profile in enumerate(profiles):
        # Store song metadata
        songs_metadata.append({
            'song': profile['original_song'],
            'artist': profile['original_artist'],
            'track_id': profile['track_id']
        })
        
        # Generate embeddings for requested types
        for embed_type in embed_types:
            formatter = embedding_formatters[embed_type]
            
            # Special validation for aspect-based embeddings
            if embed_type.endswith('_aspect'):
                aspect_name = embed_type.replace('_aspect', '')
                if not profile.get(aspect_name):
                    continue  # Skip if aspect doesn't exist
            
            # Special validation for tag/genre-based embeddings
            if embed_type in ['tags_genres', 'tags', 'genres']:
                if embed_type == 'tags_genres':
                    if not (profile.get('tags') or profile.get('genres')):
                        continue
                elif embed_type == 'tags':
                    if not profile.get('tags'):
                        continue
                elif embed_type == 'genres':
                    if not profile.get('genres'):
                        continue
            
            # Generate the text for this embedding type
            text_value = formatter(profile)
            
            # Only add task if we have non-empty text
            if text_value and text_value.strip():
                embedding_tasks.append((song_idx, embed_type, text_value))
    
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
        
        # Show embeddings in a logical order (only those that exist)
        all_possible_types = ['full_profile', 'sound_aspect', 'meaning_aspect', 'mood_aspect', 'tags_genres', 'tags', 'genres']
        tasks_dict = {embedding_type: text_value for embedding_type, text_value in tasks}
        
        # Only show embedding types that actually exist in the tasks
        relevant_types = [etype for etype in all_possible_types if etype in tasks_dict]
        
        for embedding_type in relevant_types:
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
                elif embedding_type in ['tags_genres', 'tags', 'genres']:
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

def load_existing_embeddings(output_base: str, embed_types: List[str]) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load existing embeddings from separate files by embedding type."""
    embedding_types = embed_types  # Only load the requested embedding types
    
    # Check if we're using old combined format or new separate format
    output_path = Path(output_base)
    if output_path.is_file() and output_path.suffix == '.npz':
        # Old combined format - load as before for backwards compatibility
        print(f"Detected old combined format, loading from: {output_base}")
        songs_metadata, embeddings, song_indices, field_types, field_values = load_existing_embeddings_combined(output_base)
        
        # Filter to only requested embedding types for consistency
        if len(embeddings) > 0:
            mask = np.isin(field_types, embed_types)
            embeddings = embeddings[mask]
            song_indices = song_indices[mask]
            field_types = field_types[mask]
            field_values = field_values[mask]
            print(f"Filtered to {len(embeddings)} embeddings of requested types: {', '.join(embed_types)}")
        
        return songs_metadata, embeddings, song_indices, field_types, field_values
    
    # New separate format - output_base should be a directory or base name
    if output_path.is_dir():
        base_dir = output_path
        base_name = ""
    else:
        base_dir = output_path.parent
        base_name = output_path.stem + "_"
    
    print(f"Loading existing embeddings from separate files in: {base_dir}")
    
    all_embeddings = []
    all_song_indices = []
    all_field_types = []
    all_field_values = []
    songs_metadata = None
    
    total_loaded = 0
    
    for embed_type in embedding_types:
        embed_file = base_dir / f"{base_name}{embed_type}_embeddings.npz"
        
        if not embed_file.exists():
            print(f"  {embed_type}: No existing file found ({embed_file.name})")
            continue
            
        try:
            # Check file size
            file_size = embed_file.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            print(f"  {embed_type}: Loading {file_size_mb:.1f}MB...")
            
            data = np.load(embed_file, allow_pickle=True)
            
            # Load songs metadata from first file (should be consistent across all files)
            if songs_metadata is None:
                # Check if track_ids exist in the file (backwards compatibility)
                if 'track_ids' in data:
                    songs_metadata = [
                        {'song': data['songs'][i], 'artist': data['artists'][i], 'track_id': data['track_ids'][i]} 
                        for i in range(len(data['songs']))
                    ]
                else:
                    # Fallback for older files without track_ids
                    songs_metadata = [
                        {'song': data['songs'][i], 'artist': data['artists'][i]} 
                        for i in range(len(data['songs']))
                    ]
                print(f"  Loaded metadata for {len(songs_metadata)} songs")
            
            # Load embeddings for this type
            embeddings = data['embeddings'].astype(np.float32)
            song_indices = data['song_indices'].astype(int)
            field_values = data['field_values']
            
            # Create field_types array for this embedding type
            field_types = np.array([embed_type] * len(embeddings), dtype=object)
            
            all_embeddings.append(embeddings)
            all_song_indices.append(song_indices)
            all_field_types.append(field_types)
            all_field_values.append(field_values)
            
            total_loaded += len(embeddings)
            print(f"  {embed_type}: Loaded {len(embeddings)} embeddings")
            
        except Exception as e:
            print(f"  {embed_type}: Warning - Could not load ({e})")
            continue
    
    if total_loaded == 0:
        print("No existing embeddings found, starting fresh...")
        empty_embeddings = np.empty((0, 3072), dtype=np.float32)
        empty_indices = np.array([], dtype=int)
        empty_types = np.array([], dtype=object)
        empty_values = np.array([], dtype=object)
        return [], empty_embeddings, empty_indices, empty_types, empty_values
    
    # Combine all loaded embeddings
    combined_embeddings = np.concatenate(all_embeddings, axis=0)
    combined_song_indices = np.concatenate(all_song_indices, axis=0)
    combined_field_types = np.concatenate(all_field_types, axis=0)
    combined_field_values = np.concatenate(all_field_values, axis=0)
    
    print(f"✅ Loading complete! {total_loaded} total embeddings from {len([t for t in embedding_types if (base_dir / f'{base_name}{t}_embeddings.npz').exists()])} embedding files")
    
    return songs_metadata, combined_embeddings, combined_song_indices, combined_field_types, combined_field_values

def load_existing_embeddings_combined(output_file: str) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load existing embeddings from single combined file (backwards compatibility)."""
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
        print(f"Loading combined embeddings file ({file_size_mb:.1f}MB)...")
        
        data = np.load(output_file, allow_pickle=True)
        
        print(f"Found {len(data['songs'])} songs with {len(data['embeddings'])} embeddings")
        
        # Reconstruct songs_metadata
        # Check if track_ids exist in the file (backwards compatibility)
        if 'track_ids' in data:
            songs_metadata = [
                {'song': data['songs'][i], 'artist': data['artists'][i], 'track_id': data['track_ids'][i]} 
                for i in range(len(data['songs']))
            ]
        else:
            # Fallback for older files without track_ids
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
                        output_base: str,
                        embed_types: List[str]):
    """Save song embeddings to separate files by embedding type."""
    
    # Create songs metadata arrays
    song_names = np.array([meta['song'] for meta in songs_metadata])
    artist_names = np.array([meta['artist'] for meta in songs_metadata])
    track_ids = np.array([meta['track_id'] for meta in songs_metadata])
    
    # Determine output directory and base name
    output_path = Path(output_base)
    if output_path.suffix == '.npz':
        # If given a .npz file path, use the directory and stem as base
        base_dir = output_path.parent
        base_name = output_path.stem + "_"
    else:
        base_dir = output_path
        base_name = ""
    
    # Ensure output directory exists
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Group embeddings by type and save to separate files (only for requested types)
    saved_files = []
    
    for embed_type in embed_types:
        # Find embeddings of this type
        mask = field_types == embed_type
        
        if not mask.any():
            continue  # Skip if no embeddings of this type
        
        # Extract embeddings for this type
        type_embeddings = embeddings[mask]
        type_song_indices = song_indices[mask]
        type_field_values = field_values[mask]
        
        # Create output filename
        output_file = base_dir / f"{base_name}{embed_type}_embeddings.npz"
        
        # Save to separate file
        np.savez_compressed(
            output_file,
            # Song metadata (same for all files)
            songs=song_names,
            artists=artist_names,
            track_ids=track_ids,
            # Embedding data (filtered for this type)
            embeddings=type_embeddings,
            song_indices=type_song_indices,
            field_values=type_field_values
        )
        
        saved_files.append((embed_type, output_file, len(type_embeddings)))
    
    return saved_files

def print_save_summary(songs_metadata: List[Dict], 
                      embeddings: np.ndarray,
                      song_indices: np.ndarray,
                      field_types: np.ndarray,
                      field_values: np.ndarray,
                      saved_files: List[Tuple]):
    """Print summary of saved embeddings to separate files."""
    print(f"\n{'='*60}")
    print("EMBEDDINGS SAVED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Total songs: {len(songs_metadata)}")
    print(f"Total embeddings: {len(embeddings)}")
    if len(embeddings) > 0:
        print(f"Embedding dimension: {embeddings.shape[1]}")
    
    print(f"\nSaved to {len(saved_files)} separate files:")
    total_size_mb = 0
    for embed_type, file_path, count in saved_files:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        total_size_mb += file_size_mb
        print(f"  {embed_type}: {count} embeddings → {file_path.name} ({file_size_mb:.1f}MB)")
    
    print(f"\nTotal file size: {total_size_mb:.1f}MB")
    
    # Show breakdown by field type for verification
    unique_types, counts = np.unique(field_types, return_counts=True)
    print(f"\nEmbedding verification:")
    for field_type, count in zip(unique_types, counts):
        print(f"  {field_type}: {count} embeddings")
    
    if saved_files:
        base_dir = saved_files[0][1].parent
        print(f"\nFiles saved to: {base_dir}")
        print(f"\nTo load embeddings:")
        print(f"  # Load specific embedding type:")
        print(f"  data = np.load('{saved_files[0][1]}')")
        print(f"  # Or use the updated embed_song_profiles.py with directory/base path")
        print(f"  python embed_song_profiles.py -i input.jsonl -o {base_dir}/")
    else:
        print(f"\nNo files were saved (no embeddings to save)")
    
    print(f"{'='*60}")

# ──────────────────────────────── MAIN DRIVER ────────────────────────────────
async def main(input_file: str, output_file: str, embed_types: List[str], num_entries: int = None, batch_size: int = 50):
    """Main function to process all song profiles and generate embeddings in batches."""
    print(f"Loading profiles from {input_file}")
    print(f"Generating embedding types: {', '.join(embed_types)}")
    
    # Load existing embeddings if any
    existing_songs_metadata, existing_embeddings, existing_song_indices, existing_field_types, existing_field_values = load_existing_embeddings(output_file, embed_types)
    
    # Create set of already-processed songs for deduplication
    processed_songs = set()
    if existing_songs_metadata:
        for meta in existing_songs_metadata:
            # Use track_id as primary key if available, fallback to (song, artist) for backwards compatibility
            if 'track_id' in meta:
                processed_songs.add(meta['track_id'])
            else:
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
                    # Use track_id as primary key for deduplication
                    if profile['track_id'] not in processed_songs:
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
    new_songs_metadata, new_embedding_tasks = extract_embedding_tasks(profiles_to_process, embed_types)
    
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
        
        saved_files = save_song_embeddings(
            all_songs_metadata, final_embeddings, final_song_indices,
            final_field_types, final_field_values, output_file, embed_types
        )
        print(f"Saved batch {batch_num + 1}/{total_batches} - Total: {len(all_songs_metadata)} songs, {len(final_embeddings)} embeddings to {len(saved_files)} files")

    # Use the last saved_files for summary (no need to save again)
    print_save_summary(
        all_songs_metadata, final_embeddings, final_song_indices,
        final_field_types, final_field_values, saved_files
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for song profiles")
    parser.add_argument("-i", "--input", required=True, 
                        help="Input JSONL file with song profiles")
    parser.add_argument("-o", "--output", required=True, 
                        help="Output directory or base path for separate embedding files")
    parser.add_argument("--embed_types", nargs='+', required=True,
                        choices=(EMBED_TYPES + ['all']),
                        help="Embedding types to generate (space-separated)")
    parser.add_argument("-n", "--num_entries", type=int, default=None, 
                        help="Process only first N entries (for testing)")
    parser.add_argument("-b", "--batch_size", type=int, default=500,
                        help="Number of songs to process before saving (default: 500)")
    
    args = parser.parse_args()
    
    t0 = time.perf_counter()

    if 'all' in args.embed_types:
        embed_types = EMBED_TYPES
    else:
        embed_types = args.embed_types
    print("Embed types: ", embed_types)
    asyncio.run(main(args.input, args.output, embed_types, args.num_entries, args.batch_size))
    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed:.1f}s with {MAX_CONCURRENCY}-way concurrency") 