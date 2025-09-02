"""
Data utilities for semantic song search.

This module contains shared utilities for loading and processing song data,
embeddings, and other data structures used across ranking algorithms and search engines.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


def load_song_metadata(songs_file: str) -> List[Dict]:
    """
    Load song metadata from JSON file.
    
    Args:
        songs_file: Path to songs JSON file (array of song objects with Spotify metadata)
        
    Returns:
        List of song metadata dictionaries with track_id fields added
    """
    songs_path = Path(songs_file)
    if not songs_path.exists():
        raise FileNotFoundError(f"Songs file not found: {songs_path}")
    
    logger.info(f"Loading song metadata from {songs_path}")
    
    with open(songs_path, 'r', encoding='utf-8') as f:
        songs_array = json.load(f)
    
    # Validate that we got an array
    if not isinstance(songs_array, list):
        raise ValueError(f"Expected JSON array of song objects, got {type(songs_array)}")
    
    # Process each song object in the array
    songs = []
    for i, song_metadata in enumerate(songs_array):
        if not isinstance(song_metadata, dict):
            logger.warning(f"Invalid song metadata at index {i}: {song_metadata}")
            continue
            
        if 'id' not in song_metadata:
            logger.warning(f"Song at index {i} missing 'id' field, skipping")
            continue
            
        # Create a song object with all Spotify metadata
        song = dict(song_metadata)  # Copy all Spotify metadata
        song['track_id'] = song['id']  # Ensure track_id is available as 'track_id' for clarity
        
        # For backwards compatibility during transition, derive original_song and original_artist
        song['original_song'] = song_metadata.get('name', '')
        if song_metadata.get('artists') and len(song_metadata['artists']) > 0:
            song['original_artist'] = song_metadata['artists'][0]['name']
        else:
            song['original_artist'] = ''
            
        songs.append(song)
    
    logger.info(f"Loaded {len(songs)} songs from array")
    return songs


def load_embeddings_data(embeddings_path: str) -> Dict[str, Dict]:
    """
    Load embeddings data from npz files.
    
    Args:
        embeddings_path: Path to embeddings directory (new format with track_ids)
        
    Returns:
        Dictionary mapping embedding types to their data
    """
    embeddings_path = Path(embeddings_path)
    embedding_indices = {}
    
    if embeddings_path.is_dir():
        # New separate format with track_ids
        logger.info(f"Loading embeddings from directory: {embeddings_path}")
        
        try:
            from . import constants
        except ImportError:
            import constants
        for embed_type in constants.EMBEDDING_TYPES:
            embed_file = embeddings_path / f"{embed_type}_embeddings.npz"
            if embed_file.exists():
                data = np.load(embed_file, allow_pickle=True)
                embedding_indices[embed_type] = {
                    'embeddings': data['embeddings'],
                    'song_indices': data['song_indices'],
                    'field_values': data['field_values'],
                    'track_ids': data['track_ids'],  # New field for track IDs
                    'songs': data.get('songs', np.array([])),
                    'artists': data.get('artists', np.array([]))
                }
                logger.info(f"Loaded {embed_type}: {len(data['embeddings'])} embeddings with track IDs")
    else:
        raise FileNotFoundError(f"Embeddings path not found: {embeddings_path}. New format requires directory with separate embedding files.")
    
    return embedding_indices


def load_artist_embeddings_data(artist_embeddings_path: str) -> Dict[str, Dict]:
    """
    Load artist embeddings data from npz files.
    
    Args:
        artist_embeddings_path: Path to artist embeddings directory
        
    Returns:
        Dictionary mapping embedding types to their artist data
    """
    artist_embeddings_path = Path(artist_embeddings_path)
    artist_embedding_indices = {}
    
    if not artist_embeddings_path.is_dir():
        raise FileNotFoundError(f"Artist embeddings path not found: {artist_embeddings_path}")
    
    logger.info(f"Loading artist embeddings from directory: {artist_embeddings_path}")
    
    try:
        from . import constants
    except ImportError:
        import constants
    
    for embed_type in constants.ARTIST_EMBEDDING_TYPES:
        embed_file = artist_embeddings_path / f"{embed_type}_artist_embeddings.npz"
        if embed_file.exists():
            data = np.load(embed_file, allow_pickle=True)
            artist_embedding_indices[embed_type] = {
                'artists': data['artists'],
                'embeddings': data['embeddings']
            }
            
            # Include additional fields if available (for consistency with song embeddings)
            if 'artist_indices' in data:
                artist_embedding_indices[embed_type]['artist_indices'] = data['artist_indices']
            if 'field_values' in data:
                artist_embedding_indices[embed_type]['field_values'] = data['field_values']
                logger.info(f"Loaded artist {embed_type}: {len(data['artists'])} artist embeddings with indices")
            else:
                logger.info(f"Loaded artist {embed_type}: {len(data['artists'])} artist embeddings")
    
    if not artist_embedding_indices:
        logger.warning(f"No artist embedding files found in {artist_embeddings_path}")
    
    return artist_embedding_indices


def build_embedding_lookup(embedding_indices: Dict, songs_metadata: List[Dict], 
                         embed_type: str = 'full_profile') -> Dict[str, np.ndarray]:
    """
    Build a lookup dictionary mapping track_id keys to embeddings, including linked_from IDs.
    
    Args:
        embedding_indices: Output from load_embeddings_data
        songs_metadata: Song metadata list (used for track_id mapping)
        embed_type: Type of embeddings to use
        
    Returns:
        Dictionary mapping track_id strings to normalized embeddings
    """
    if not isinstance(songs_metadata, list):
        raise ValueError("songs_metadata must be a list")
    if not isinstance(embedding_indices, dict):
        raise ValueError("embedding_indices must be a dictionary")
    if embed_type not in embedding_indices:
        raise ValueError(f"Embedding type {embed_type} not found in data")
    
    indices = embedding_indices[embed_type]
    embeddings = indices['embeddings']
    track_ids = indices['track_ids']
    
    # Validate that embeddings and track_ids arrays have same length
    if len(embeddings) != len(track_ids):
        raise ValueError(f"Mismatch between embeddings ({len(embeddings)}) and track_ids ({len(track_ids)}) arrays")
    
    # Create mapping from embedding track_ids to song metadata for linked_from support
    embedding_to_song = {}
    for song in songs_metadata:
        canonical_id = song.get('track_id') or song.get('id')
        if canonical_id:
            # Map canonical ID
            embedding_to_song[canonical_id] = song
            
            # Map linked_from ID if it exists
            linked_from = song.get('linked_from')
            if linked_from and 'id' in linked_from:
                linked_from_id = linked_from['id']
                embedding_to_song[linked_from_id] = song
    
    # Build lookup dictionary from track_ids, using canonical IDs as keys to avoid duplication
    embedding_lookup = {}
    processed_canonical_ids = set()
    
    for i, embedding_track_id in enumerate(track_ids):
        embedding = embeddings[i]
        
        # Validate embedding
        if not isinstance(embedding, np.ndarray):
            logger.warning(f"Invalid embedding type for track_id {embedding_track_id}: {type(embedding)}")
            continue
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:  # Avoid division by very small numbers
            embedding = embedding / norm
        else:
            logger.warning(f"Zero or near-zero norm embedding for track_id {embedding_track_id}")
            continue
        
        # Find the canonical ID for this embedding
        if embedding_track_id in embedding_to_song:
            song = embedding_to_song[embedding_track_id]
            canonical_id = song.get('track_id') or song.get('id')
            
            # Only add each unique song once using its canonical ID as the key
            if canonical_id not in processed_canonical_ids:
                embedding_lookup[canonical_id] = embedding
                processed_canonical_ids.add(canonical_id)
        else:
            # Skip embeddings that don't correspond to any song in the reconciled metadata
            # This ensures the embedding lookup only contains songs that exist in filtered_songs
            logger.debug(f"Skipping orphaned embedding track_id {embedding_track_id} (no corresponding song metadata)")
    
    logger.info(f"Built embedding lookup for {len(embedding_lookup)} songs using {embed_type}")
    return embedding_lookup


def reconcile_song_indices(embedding_indices: Dict, songs_metadata: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Reconcile song track IDs between embeddings and metadata, handling missing songs and linked_from IDs.
    
    Args:
        embedding_indices: Dictionary of embedding data with track_ids
        songs_metadata: List of song metadata with track_id fields
        
    Returns:
        Tuple of (filtered_songs, embedding_indices_updated)
    """
    # Collect all track IDs that have embeddings
    all_embedding_track_ids = set()
    
    for embed_type, indices in embedding_indices.items():
        track_ids = indices.get('track_ids', np.array([]))
        all_embedding_track_ids.update(track_ids)
    
    # Create comprehensive track_id to song mapping for metadata
    # This includes both canonical track IDs and linked_from track IDs
    track_id_to_song = {}
    for song in songs_metadata:
        track_id = song.get('track_id') or song.get('id')
        if track_id:
            # Add canonical track ID mapping
            track_id_to_song[track_id] = song
            
            # Add linked_from track ID mapping if it exists
            linked_from = song.get('linked_from')
            if linked_from and 'id' in linked_from:
                linked_from_id = linked_from['id']
                track_id_to_song[linked_from_id] = song
                logger.debug(f"Added linked_from mapping: {linked_from_id} -> {track_id}")
    
    # Filter songs to only include those with embeddings
    filtered_songs = []
    track_id_to_new_index = {}  # track_id -> new_index (includes both canonical and linked_from)
    canonical_track_ids = set()  # Track unique songs by canonical ID
    
    for embedding_track_id in all_embedding_track_ids:
        if embedding_track_id in track_id_to_song:
            song = track_id_to_song[embedding_track_id]
            canonical_track_id = song.get('track_id') or song.get('id')
            
            # Only add each unique song once (by canonical track ID)
            if canonical_track_id not in canonical_track_ids:
                new_idx = len(filtered_songs)
                canonical_track_ids.add(canonical_track_id)
                filtered_songs.append(song)
                
                # Map both the embedding track ID and canonical track ID to the same index
                track_id_to_new_index[embedding_track_id] = new_idx
                if embedding_track_id != canonical_track_id:
                    track_id_to_new_index[canonical_track_id] = new_idx
            else:
                # Song already added, just add the mapping for this embedding track ID
                existing_song_idx = next(
                    i for i, s in enumerate(filtered_songs) 
                    if (s.get('track_id') or s.get('id')) == canonical_track_id
                )
                track_id_to_new_index[embedding_track_id] = existing_song_idx
    
    logger.info(f"Reconciled track IDs: {len(filtered_songs)}/{len(songs_metadata)} songs have embeddings")
    linked_count = sum(1 for song in songs_metadata if song.get('linked_from'))
    logger.info(f"Found {linked_count} songs with linked_from metadata")
    
    # Update embedding indices (no need to change indices since track_ids directly reference songs)
    updated_indices = {}
    for embed_type, indices in embedding_indices.items():
        track_ids = indices.get('track_ids', np.array([]))
        
        # Create mask for track IDs that exist in metadata
        valid_mask = np.array([track_id in track_id_to_song for track_id in track_ids])
        
        if valid_mask.any():
            # Create new song_indices that map to position in filtered_songs
            new_song_indices = np.array([
                track_id_to_new_index[track_id] 
                for track_id, is_valid in zip(track_ids, valid_mask) 
                if is_valid
            ])
            
            updated_indices[embed_type] = {
                'embeddings': indices['embeddings'][valid_mask],
                'song_indices': new_song_indices,
                'track_ids': track_ids[valid_mask],  # Keep filtered track_ids
                'field_values': indices['field_values'][valid_mask] if 'field_values' in indices else np.array([]),
                'songs': indices.get('songs', np.array([]))[valid_mask] if len(indices.get('songs', [])) > 0 else np.array([]),
                'artists': indices.get('artists', np.array([]))[valid_mask] if len(indices.get('artists', [])) > 0 else np.array([])
            }
        else:
            logger.warning(f"No valid embeddings found for {embed_type}")
            updated_indices[embed_type] = {
                'embeddings': np.array([]),
                'song_indices': np.array([]),
                'track_ids': np.array([]),
                'field_values': np.array([]),
                'songs': np.array([]),
                'artists': np.array([])
            }
    
    return filtered_songs, updated_indices




def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalize embeddings for cosine similarity computation.
    
    Args:
        embeddings: Array of embeddings to normalize
        
    Returns:
        L2-normalized embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return embeddings / norms


def filter_history_to_known_songs(history_df: pd.DataFrame, songs_metadata: List[Dict]) -> pd.DataFrame:
    """
    Filter history DataFrame to only include songs in metadata.
    Supports both track_id-based matching (preferred) and fallback to (song, artist) matching.
    
    Args:
        history_df: Listening history DataFrame
        songs_metadata: List of song metadata with track_id fields
        
    Returns:
        Filtered history DataFrame
    """
    if not isinstance(history_df, pd.DataFrame):
        raise ValueError("history_df must be a pandas DataFrame")
    if not isinstance(songs_metadata, list):
        raise ValueError("songs_metadata must be a list")
    
    # Create set of known track IDs and fallback (song, artist) keys
    metadata_track_ids = set()
    metadata_keys = set()
    
    for song in songs_metadata:
        if isinstance(song, dict):
            # Preferred: track_id matching
            track_id = song.get('track_id') or song.get('id')
            if track_id:
                metadata_track_ids.add(track_id)
            
            # Fallback: (song, artist) matching
            if 'original_song' in song and 'original_artist' in song:
                metadata_keys.add((song['original_song'], song['original_artist']))
    
    original_count = len(history_df)
    mask = None
    
    # Try track_id matching first if available in history
    if 'track_id' in history_df.columns or 'spotify_track_uri' in history_df.columns:
        if 'track_id' in history_df.columns:
            # Direct track_id matching
            track_ids = history_df['track_id'].fillna('')
            mask = [track_id in metadata_track_ids for track_id in track_ids]
        elif 'spotify_track_uri' in history_df.columns:
            # Extract track_id from Spotify URI (format: spotify:track:TRACK_ID)
            track_ids = history_df['spotify_track_uri'].fillna('').str.extract(r'spotify:track:(.+)')[0].fillna('')
            mask = [track_id in metadata_track_ids for track_id in track_ids]
        
        filtered_df = history_df[mask].reset_index(drop=True) if any(mask) else history_df.iloc[0:0].copy()
        matched_count = len(filtered_df)
        
        if matched_count > 0:
            logger.info(f"Filtered history using track IDs: {matched_count}/{original_count} entries match known songs")
            return filtered_df
        else:
            logger.info("No matches found with track_id matching, falling back to (song, artist) matching")
    
    # Fallback to (song, artist) matching
    if 'original_song' not in history_df.columns or 'original_artist' not in history_df.columns:
        logger.warning("Required columns for song matching missing from history_df")
        return history_df.iloc[0:0].copy()  # Return empty DataFrame with same structure
    
    # Create boolean mask using vectorized operation
    song_keys = list(zip(history_df['original_song'], history_df['original_artist']))
    mask = [key in metadata_keys for key in song_keys]
    filtered_df = history_df[mask].reset_index(drop=True)
    
    filtered_count = len(filtered_df)
    logger.info(f"Filtered history using (song, artist) keys: {filtered_count}/{original_count} entries match known songs")
    
    return filtered_df


def build_text_search_index(songs: List[Dict]) -> Dict:
    """
    Build a RapidFuzz search index for song/artist/album matching.
    
    Args:
        songs: List of song metadata dictionaries
        
    Returns:
        Dictionary mapping song keys to searchable fields
    """
    if not isinstance(songs, list):
        raise ValueError("songs must be a list")
    
    logger.info(f"Building RapidFuzz search index for {len(songs)} songs...")
    
    search_index = {}
    for i, song in enumerate(songs):
        if not isinstance(song, dict):
            logger.warning(f"Invalid song metadata: {song}")
            continue
            
        song_name = song.get('original_song', '') or ''
        artist_name = song.get('original_artist', '') or ''
        # metadata = song.get('metadata') or {}  # Handle None metadata
        # album_name = metadata.get('album_name', '') or ''

        # Use index as unique key to avoid collisions with duplicate (song, artist) pairs
        # This ensures all songs remain searchable even if they have identical names/artists
        unique_key = i
        
        # Store individual fields for weighted fuzzy search
        search_index[unique_key] = {
            'track_name': song_name,
            'artist_name': artist_name,
            # 'album_name': album_name,
            'combined': f"TRACK:{song_name} ARTIST:{artist_name}",
            'song_index': i,  # Store the original index for compatibility
            'song_key': (song_name, artist_name)  # Store original song key for reference
        }
    
    logger.info(f"Built RapidFuzz search index for {len(search_index)} songs")
    return search_index


def get_openai_embedding(text: str, model: str = None, normalize: bool = True) -> np.ndarray:
    """
    Get OpenAI embedding for text query.
    
    Args:
        text: Input text to embed
        model: OpenAI embedding model name
        normalize: Whether to L2-normalize the embedding
        
    Returns:
        Normalized or raw embedding vector
    """
    try:
        from openai import OpenAI
        import os
        
        try:
            from . import constants
        except ImportError:
            import constants
        
        if model is None:
            model = constants.OPENAI_EMBEDDING_MODEL
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        client = OpenAI(api_key=api_key)
        
        response = client.embeddings.create(
            model=model,
            input=text,
            encoding_format="float"
        )
        
        embedding = np.array(response.data[0].embedding)
        
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm
            else:
                logger.warning(f"Near-zero norm embedding for text: '{text[:50]}...'")
        
        return embedding
        
    except ImportError:
        logger.error("OpenAI library not available")
        raise
    except Exception as e:
        logger.error(f"Error getting text embedding: {e}")
        raise


def search_songs_by_text(query: str, search_index: Dict, songs: List[Dict], 
                        limit: int = None, min_score: float = None) -> List[Tuple[int, float, str]]:
    """
    Search for songs using RapidFuzz fuzzy string matching.
    
    Args:
        query: Search query text
        search_index: Pre-built search index with individual fields
        songs: List of song metadata (for compatibility)
        limit: Maximum number of results
        min_score: Minimum similarity threshold
        
    Returns:
        List of (song_index, similarity_score, display_label) tuples
    """
    try:
        try:
            from . import constants
        except ImportError:
            import constants
    except ImportError:
        import constants
    
    # Use defaults from constants if not provided
    if limit is None:
        limit = constants.DEFAULT_SUGGESTION_LIMIT
    if min_score is None:
        min_score = constants.FUZZY_SEARCH_CONSTANTS['SEARCH_MIN_SCORE']
    
    if not query or not query.strip() or not search_index:
        return []
    
    results = []
    query_lower = query.lower().strip()
    
    for unique_key, fields in search_index.items():
        if not fields:
            continue
            
        try:
            # Safely get field values with robust None/empty string handling
            track_name = (fields.get('track_name') or '').strip()
            artist_name = (fields.get('artist_name') or '').strip() 
            # album_name = (fields.get('album_name') or '').strip()
            combined = (fields.get('combined') or '').strip()
            
            # Calculate weighted scores for each field using constants
            track_score = fuzz.WRatio(query_lower, track_name.lower()) * constants.FUZZY_SEARCH_CONSTANTS['TRACK_WEIGHT']
            artist_score = fuzz.WRatio(query_lower, artist_name.lower()) * constants.FUZZY_SEARCH_CONSTANTS['ARTIST_WEIGHT'] 
            # album_score = fuzz.WRatio(query_lower, album_name.lower()) * constants.FUZZY_SEARCH_CONSTANTS['ALBUM_WEIGHT']
            
            # Also try combined search for cross-field matches
            combined_score = fuzz.WRatio(query_lower, combined.lower()) * constants.FUZZY_SEARCH_CONSTANTS['COMBINED_WEIGHT']
            
            # Take the best score across all fields
            best_score = max(track_score, artist_score, combined_score)
            
            # Only include results above threshold
            if best_score >= min_score:
                song_index = fields.get('song_index', unique_key)  # Get original song index
                song_key = fields.get('song_key', (track_name, artist_name))  # Get original song key
                song_name, artist_name = song_key
                label = f"{song_name} - {artist_name}"
                results.append((song_index, best_score, label))
                
        except Exception as e:
            # Skip problematic entries but don't crash the search
            logger.warning(f"Error processing song {unique_key}: {e}")
            continue
    
    # Sort by score (descending) and limit results
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]


# Spotify History Processing Functions

def smart_convert_to_datetime(timestamp: int) -> Optional[datetime]:
    """Convert timestamp to datetime, handling both seconds and milliseconds."""
    if timestamp is None:
        return None
    # timestamp may be in milliseconds or seconds
    if timestamp < 10000000000:
        return datetime.fromtimestamp(timestamp)
    else:
        return datetime.fromtimestamp(timestamp / 1000)


def load_spotify_history_entries(json_dir: Path) -> List[Dict]:
    """Load entries from Spotify Extended Streaming History JSON files."""
    entries = []
    json_files = list(json_dir.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {json_dir}")
        return []
    
    logger.info(f"Loading history from {len(json_files)} JSON files...")
    
    for json_file in sorted(json_files):
        # Only include audio history files
        if "Audio" not in json_file.name:
            logger.debug(f"Skipping non-audio history file: {json_file.name}")
            continue
            
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                entries.extend(data)
                logger.info(f"Loaded {len(data)} entries from {json_file.name}")
            else:
                logger.warning(f"Expected list in {json_file.name}, got {type(data)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {json_file}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            continue
    
    logger.info(f"Total entries loaded: {len(entries)}")
    return entries


def clean_history_entries(entries: List[Dict]) -> List[Dict]:
    """Clean and standardize history entry data."""
    cleaned_entries = []
    skipped_count = 0
    
    for entry in entries:
        # Skip entries without track URI
        if not entry.get("spotify_track_uri"):
            skipped_count += 1
            continue
        
        # Skip entries without required metadata
        if not entry.get("master_metadata_track_name") or not entry.get("master_metadata_album_artist_name"):
            skipped_count += 1
            continue
        
        # Handle timestamp conversion with comprehensive error handling
        try:
            if entry.get("offline"):
                timestamp = smart_convert_to_datetime(entry.get("offline_timestamp"))
                if timestamp is None:
                    skipped_count += 1
                    continue
            else:
                timestamp_str = entry.get("ts", "")
                if not timestamp_str:
                    skipped_count += 1
                    continue
                
                # Handle various timestamp formats
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except ValueError:
                    # Try alternative parsing methods
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
                    except ValueError:
                        logger.warning(f"Could not parse timestamp: {timestamp_str}")
                        skipped_count += 1
                        continue
            
            # Strip timezone info to work with naive datetime objects
            if timestamp:
                timestamp = timestamp.replace(tzinfo=None)
                
                # Validate timestamp is reasonable (not too far in past/future)
                now = datetime.now()
                if timestamp.year < 2006 or timestamp > now:  # Spotify founded in 2006
                    logger.warning(f"Timestamp out of reasonable range: {timestamp}")
                    skipped_count += 1
                    continue
            else:
                skipped_count += 1
                continue
                
        except Exception as e:
            logger.warning(f"Error processing timestamp for entry, skipping: {e}")
            skipped_count += 1
            continue
        
        # Build clean entry
        clean_entry = {
            'original_song': entry["master_metadata_track_name"],
            'original_artist': entry["master_metadata_album_artist_name"],
            'ms_played': int(entry.get("ms_played", 0)),
            'track_id': entry["spotify_track_uri"].split(":")[-1],
            'timestamp': timestamp,
            'skipped': entry.get("skipped", False),
            'offline': entry.get("offline", False)
        }
        
        cleaned_entries.append(clean_entry)
    
    logger.info(f"Cleaned {len(cleaned_entries)} entries, skipped {skipped_count}")
    return cleaned_entries


def load_and_process_spotify_history(history_path: Path) -> Tuple[pd.DataFrame, bool]:
    """
    Load and process Spotify Extended Streaming History.
    
    Args:
        history_path: Path to directory containing JSON files
        
    Returns:
        Tuple of (DataFrame, success_flag)
    """
    try:
        # Load raw entries
        entries = load_spotify_history_entries(history_path)
        if not entries:
            logger.warning("No history entries found")
            return pd.DataFrame(), False
        
        # Clean entries
        cleaned_entries = clean_history_entries(entries)
        if not cleaned_entries:
            logger.warning("No valid entries after cleaning")
            return pd.DataFrame(), False
        
        # Convert to DataFrame
        df = pd.DataFrame(cleaned_entries)
        logger.info(f"Successfully loaded history with {len(df)} entries")
        return df, True
        
    except Exception as e:
        logger.error(f"Error loading Spotify history: {e}")
        return pd.DataFrame(), False