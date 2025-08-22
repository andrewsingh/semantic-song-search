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
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def load_song_metadata(songs_file: str) -> List[Dict]:
    """
    Load song metadata from JSON file.
    
    Args:
        songs_file: Path to songs JSON file
        
    Returns:
        List of song metadata dictionaries
    """
    songs_path = Path(songs_file)
    if not songs_path.exists():
        raise FileNotFoundError(f"Songs file not found: {songs_path}")
    
    logger.info(f"Loading song metadata from {songs_path}")
    
    with open(songs_path, 'r', encoding='utf-8') as f:
        songs = json.load(f)
    
    logger.info(f"Loaded {len(songs)} songs")
    return songs


def load_embeddings_data(embeddings_path: str) -> Dict[str, Dict]:
    """
    Load embeddings data from npz files (old or new format).
    
    Args:
        embeddings_path: Path to embeddings file or directory
        
    Returns:
        Dictionary mapping embedding types to their data
    """
    embeddings_path = Path(embeddings_path)
    embedding_indices = {}
    
    if embeddings_path.is_file() and embeddings_path.suffix == '.npz':
        # Old combined format
        logger.info(f"Loading embeddings from combined file: {embeddings_path}")
        data = np.load(embeddings_path, allow_pickle=True)
        
        embedding_indices['combined'] = {
            'embeddings': data['embeddings'],
            'song_indices': data['song_indices'], 
            'field_values': data['field_values'],
            'field_types': data.get('field_types', np.array(['full_profile'] * len(data['embeddings']))),
            'songs': data.get('songs', np.array([])),
            'artists': data.get('artists', np.array([]))
        }
        
    elif embeddings_path.is_dir():
        # New separate format
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
                    'songs': data.get('songs', np.array([])),
                    'artists': data.get('artists', np.array([]))
                }
                logger.info(f"Loaded {embed_type}: {len(data['embeddings'])} embeddings")
    else:
        raise FileNotFoundError(f"Embeddings path not found: {embeddings_path}")
    
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
    
    for embed_type in constants.EMBEDDING_TYPES:
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
                         embed_type: str = 'full_profile') -> Dict[Tuple[str, str], np.ndarray]:
    """
    Build a lookup dictionary mapping (song, artist) keys to embeddings.
    
    Args:
        embedding_indices: Output from load_embeddings_data
        songs_metadata: Song metadata list (used for song key mapping)
        embed_type: Type of embeddings to use
        
    Returns:
        Dictionary mapping (song, artist) tuples to normalized embeddings
    """
    if not isinstance(songs_metadata, list):
        raise ValueError("songs_metadata must be a list")
    if not isinstance(embedding_indices, dict):
        raise ValueError("embedding_indices must be a dictionary")
    if embed_type not in embedding_indices:
        # Try combined format or fallback
        if 'combined' in embedding_indices:
            indices = embedding_indices['combined']
            # Filter by type if available
            if 'field_types' in indices:
                mask = indices['field_types'] == embed_type
                if mask.any():
                    embeddings = indices['embeddings'][mask]
                    song_indices = indices['song_indices'][mask]
                else:
                    logger.warning(f"No embeddings found for type {embed_type}, using all")
                    embeddings = indices['embeddings']
                    song_indices = indices['song_indices']
            else:
                embeddings = indices['embeddings'] 
                song_indices = indices['song_indices']
        else:
            raise ValueError(f"Embedding type {embed_type} not found in data")
    else:
        indices = embedding_indices[embed_type]
        embeddings = indices['embeddings']
        song_indices = indices['song_indices']
    
    # Create song index to metadata mapping using the reconciled songs_metadata
    # The songs_metadata list should already be filtered to match embedding indices
    song_idx_to_key = {}
    for i, song in enumerate(songs_metadata):
        if not isinstance(song, dict) or 'original_song' not in song or 'original_artist' not in song:
            logger.warning(f"Invalid song metadata at index {i}: {song}")
            continue
        song_key = (song['original_song'], song['original_artist'])
        song_idx_to_key[i] = song_key
    
    # Build lookup dictionary
    embedding_lookup = {}
    
    for i, song_idx in enumerate(song_indices):
        if song_idx in song_idx_to_key:
            song_key = song_idx_to_key[song_idx]
            embedding = embeddings[i]
            
            # Validate embedding
            if not isinstance(embedding, np.ndarray):
                logger.warning(f"Invalid embedding type for {song_key}: {type(embedding)}")
                continue
            
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:  # Avoid division by very small numbers
                embedding = embedding / norm
            else:
                logger.warning(f"Zero or near-zero norm embedding for {song_key}")
                continue
            
            embedding_lookup[song_key] = embedding
        else:
            logger.debug(f"Song index {song_idx} not found in metadata mapping")
    
    logger.info(f"Built embedding lookup for {len(embedding_lookup)} songs using {embed_type}")
    return embedding_lookup


def reconcile_song_indices(embedding_indices: Dict, songs_metadata: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Reconcile song indices between embeddings and metadata, handling missing songs.
    
    Args:
        embedding_indices: Dictionary of embedding data
        songs_metadata: List of song metadata
        
    Returns:
        Tuple of (filtered_songs, embedding_indices_updated)
    """
    # Collect all song indices that have embeddings
    all_embedding_indices = set()
    
    for embed_type, indices in embedding_indices.items():
        song_indices = indices.get('song_indices', np.array([]))
        all_embedding_indices.update(song_indices)
    
    # Filter songs to only include those with embeddings
    filtered_songs = []
    index_mapping = {}  # old_index -> new_index
    
    for old_idx, song in enumerate(songs_metadata):
        if old_idx in all_embedding_indices:
            new_idx = len(filtered_songs)
            index_mapping[old_idx] = new_idx
            filtered_songs.append(song)
    
    logger.info(f"Reconciled indices: {len(filtered_songs)}/{len(songs_metadata)} songs have embeddings")
    
    # Update embedding indices with new mapping
    updated_indices = {}
    for embed_type, indices in embedding_indices.items():
        old_song_indices = indices['song_indices']
        
        # Map old indices to new indices, keeping only valid ones
        valid_mask = np.array([idx in index_mapping for idx in old_song_indices])
        
        if valid_mask.any():
            new_song_indices = np.array([index_mapping[idx] for idx in old_song_indices[valid_mask]])
            
            updated_indices[embed_type] = {
                'embeddings': indices['embeddings'][valid_mask],
                'song_indices': new_song_indices,
                'field_values': indices['field_values'][valid_mask] if 'field_values' in indices else np.array([]),
                'songs': indices.get('songs', np.array([]))[valid_mask] if len(indices.get('songs', [])) > 0 else np.array([]),
                'artists': indices.get('artists', np.array([]))[valid_mask] if len(indices.get('artists', [])) > 0 else np.array([])
            }
        else:
            logger.warning(f"No valid embeddings found for {embed_type}")
            updated_indices[embed_type] = {
                'embeddings': np.array([]),
                'song_indices': np.array([]),
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
    
    Args:
        history_df: Listening history DataFrame
        songs_metadata: List of song metadata
        
    Returns:
        Filtered history DataFrame
    """
    if not isinstance(history_df, pd.DataFrame):
        raise ValueError("history_df must be a pandas DataFrame")
    if not isinstance(songs_metadata, list):
        raise ValueError("songs_metadata must be a list")
    # Create set of known song keys with validation
    metadata_keys = set()
    for song in songs_metadata:
        if isinstance(song, dict) and 'original_song' in song and 'original_artist' in song:
            metadata_keys.add((song['original_song'], song['original_artist']))
    
    # Filter dataframe using vectorized operations for better performance
    if 'original_song' not in history_df.columns or 'original_artist' not in history_df.columns:
        logger.warning("Required columns 'original_song' or 'original_artist' missing from history_df")
        return history_df.iloc[0:0].copy()  # Return empty DataFrame with same structure
    
    original_count = len(history_df)
    # Create boolean mask using vectorized operation instead of apply
    song_keys = list(zip(history_df['original_song'], history_df['original_artist']))
    mask = [key in metadata_keys for key in song_keys]
    filtered_df = history_df[mask].reset_index(drop=True)
    
    filtered_count = len(filtered_df)
    logger.info(f"Filtered history: {filtered_count}/{original_count} entries match known songs")
    
    return filtered_df


def build_text_search_index(songs: List[Dict], ngram_range: Tuple[int, int] = None, 
                          max_features: int = None) -> Tuple[TfidfVectorizer, Any]:
    """
    Build a TF-IDF text search index for song/artist/album matching.
    
    Args:
        songs: List of song metadata dictionaries
        ngram_range: N-gram range for TF-IDF
        max_features: Maximum number of features for TF-IDF
        
    Returns:
        Tuple of (TfidfVectorizer, fitted_matrix)
    """
    if not isinstance(songs, list):
        raise ValueError("songs must be a list")
    
    try:
        try:
            from . import constants
        except ImportError:
            import constants
    except ImportError:
        import constants
    
    # Use defaults from constants if not provided
    if ngram_range is None:
        ngram_range = constants.TFIDF_NGRAM_RANGE
    if max_features is None:
        max_features = constants.TFIDF_MAX_FEATURES
    
    logger.info(f"Building text search index for {len(songs)} songs...")
    
    search_texts = []
    for song in songs:
        if not isinstance(song, dict):
            logger.warning(f"Invalid song metadata: {song}")
            continue
            
        metadata = song.get('metadata', {})
        searchable_text = f"{song.get('original_song', '')} {song.get('original_artist', '')} {metadata.get('album_name', '')}"
        search_texts.append(searchable_text.lower())
    
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(search_texts)
    
    logger.info(f"Built text search index with {len(vectorizer.vocabulary_)} features")
    return vectorizer, tfidf_matrix


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


def search_songs_by_text(query: str, vectorizer: TfidfVectorizer, tfidf_matrix: Any, 
                        songs: List[Dict], limit: int = None, min_score: float = None) -> List[Tuple[int, float, str]]:
    """
    Search for songs using text similarity.
    
    Args:
        query: Search query text
        vectorizer: Fitted TfidfVectorizer
        tfidf_matrix: Fitted TF-IDF matrix
        songs: List of song metadata
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
        min_score = constants.TEXT_SEARCH_MIN_SCORE
    
    if not query or not query.strip():
        return []
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    query_lower = query.lower().strip()
    query_vec = vectorizer.transform([query_lower])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    top_indices = similarities.argsort()[::-1][:limit]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > min_score:
            if idx < len(songs):
                song = songs[idx]
                label = f"{song.get('original_song', 'Unknown')} - {song.get('original_artist', 'Unknown')}"
                results.append((int(idx), float(similarities[idx]), label))
    
    return results


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
            'track_uri': entry["spotify_track_uri"],
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