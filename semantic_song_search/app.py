#!/usr/bin/env python3
"""
Semantic Song Search App
Supports both text-to-song and song-to-song search with multiple embedding types.
"""
import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib
from openai import OpenAI
from typing import List, Dict, Tuple, Optional
import logging
import argparse
from pathlib import Path
try:
    from . import ranking_utils
except ImportError:
    import ranking_utils
import uuid
from datetime import datetime, timedelta
import mixpanel
import pandas as pd
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command line arguments
def parse_arguments():
    """Parse command line arguments for data file paths."""
    parser = argparse.ArgumentParser(
        description="Semantic Song Search and Playlist Creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py
  python app.py --songs custom_songs.json --embeddings custom_embeddings.npz
  python app.py -s /path/to/songs.json -e /path/to/embeddings.npz
        """
    )
    
    # Default paths (relative to the script location)
    default_songs_file = Path(__file__).parent.parent / 'data' / 'eval_set_v2' / 'eval_set_v2_metadata_ready.json'
    default_embeddings_file = Path(__file__).parent.parent / 'data' / 'eval_set_v2' / 'eval_set_v2_embeddings'
    
    parser.add_argument(
        '-s', '--songs',
        type=str,
        default=str(default_songs_file),
        help=f'Path to song profiles JSON file (default: {default_songs_file})'
    )
    
    parser.add_argument(
        '-e', '--embeddings',
        type=str,
        default=str(default_embeddings_file),
        help=f'Path to embeddings file/directory (supports combined .npz file or directory with separate embedding files) (default: {default_embeddings_file})'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to run the server on (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to run the server on (default: 5000)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )
    
    parser.add_argument(
        '--history',
        type=str,
        default=None,
        help='Path to Spotify Extended Streaming History directory (optional - enables personalized ranking)'
    )
    
    return parser.parse_args()

# Global variable to store arguments (will be set in main)
args = None

app = Flask(__name__)
# Use persistent secret key or generate one (sessions reset on app restart with random key)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', os.urandom(24))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_FILE_THRESHOLD'] = 500
Session(app)

# Spotify configuration
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET') 
SPOTIFY_SCOPES = "streaming user-read-email user-read-private user-read-playback-state user-modify-playback-state playlist-modify-public playlist-modify-private user-top-read"

# Validate required environment variables
if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    logger.error("Missing required Spotify credentials. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
    exit(1)

# OpenAI configuration
openai_client = OpenAI()

# Validate OpenAI API key exists
if not os.getenv('OPENAI_API_KEY'):
    logger.warning("OPENAI_API_KEY not set. Text-to-song search will not work.")

# Mixpanel configuration
MIXPANEL_TOKEN = os.getenv('MIXPANEL_TOKEN')
if not MIXPANEL_TOKEN:
    logger.warning("MIXPANEL_TOKEN not set. Analytics tracking will be disabled.")
    mp = None
else:
    mp = mixpanel.Mixpanel(MIXPANEL_TOKEN)

# Helper functions for tracking
def get_or_create_user_id():
    """Get existing user ID or create a new one."""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']

def get_user_properties():
    """Get common user properties for tracking."""
    # Ensure session_start is set if not already present
    if 'session_start' not in session:
        session['session_start'] = datetime.now().isoformat()[:19]
    
    # Get additional header information for location context
    headers = {
        'accept_language': request.headers.get('Accept-Language', ''),
        'x_forwarded_for': request.headers.get('X-Forwarded-For', ''),  # For proxied requests
        'cf_ipcountry': request.headers.get('CF-IPCountry', ''),  # Cloudflare country header
        'x_real_ip': request.headers.get('X-Real-IP', ''),  # Real IP from proxy
    }
    
    return {
        'is_authenticated': bool(session.get('token_info')),
        'session_start': session['session_start'],
        'user_agent': request.headers.get('User-Agent', '')[:100],  # Limit length
        'ip': request.remote_addr,
        'real_ip': headers['x_real_ip'] or headers['x_forwarded_for'] or request.remote_addr,
        'referrer': request.headers.get('Referer', ''),
        'accept_language': headers['accept_language'][:50],  # Browser language preferences
        'cf_country': headers['cf_ipcountry'],  # Will be empty unless using Cloudflare
        'timestamp': datetime.now().isoformat()[:19]
    }

def track_event(event_name, properties=None):
    """Helper function to track events with error handling."""
    if mp is None:
        logger.debug(f"Mixpanel not configured, skipping event: {event_name}")
        return  # Skip if Mixpanel is not configured
    
    try:
        user_id = get_or_create_user_id()
        event_properties = get_user_properties()
        if properties:
            event_properties.update(properties)
        logger.info(f"Tracking event: {event_name} for user: {user_id}")
        mp.track(user_id, event_name, event_properties)
    except Exception as e:
        logger.error(f"Error tracking event {event_name}: {e}")

class MusicSearchEngine:
    """Core search engine for semantic music search with personalized ranking."""
    
    def __init__(self, songs_file: str, embeddings_file: str, history_path: str = None):
        self.songs_file = songs_file
        self.embeddings_file = embeddings_file
        self.history_path = history_path
        self.songs = []
        self.embeddings_data = None
        self.song_lookup = {}
        self.embedding_indices = {}
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Personalized ranking data structures
        self.history_df = None
        self.user_song_stats = {}  # Aggregated stats per (original_song, original_artist)
        self.has_history = False
        
        # Ranking algorithm configuration
        self.ranking_config = ranking_utils.RankingConfig()
        
        self._load_data()
        self._build_indices()
        self._build_text_search_index()
        
        if self.history_path:
            self._load_and_process_history()
    
    def _load_data(self):
        """Load song profiles and embeddings."""
        logger.info("Loading song profiles...")
        with open(self.songs_file, 'r') as f:
            self.songs = json.load(f)
        
        # Create lookup for quick access
        for i, song in enumerate(self.songs):
            song_key = (song['original_song'], song['original_artist'])
            self.song_lookup[song_key] = i
        
        logger.info(f"Loaded {len(self.songs)} songs")
        
        # Load embeddings (support both old combined and new separate formats)
        self.embeddings_data = self._load_embeddings_data()
    
    def _load_embeddings_data(self):
        """Load embeddings data from either combined file or separate files by type."""
        embeddings_path = Path(self.embeddings_file)
        
        # Check if it's a single combined file (old format)
        if embeddings_path.is_file() and embeddings_path.suffix == '.npz':
            logger.info(f"Loading embeddings from combined file: {embeddings_path}")
            data = np.load(self.embeddings_file, allow_pickle=True)
            logger.info(f"Loaded {len(data['embeddings'])} embeddings")
            
            # Check if field_values exists (for backwards compatibility)
            if 'field_values' not in data:
                logger.warning("Embeddings file does not contain field_values - accordion functionality will be disabled")
                # Create placeholder field_values array
                data = dict(data)  # Convert to regular dict for modification
                data['field_values'] = np.array(['N/A'] * len(data['embeddings']), dtype=object)
            
            # CRITICAL: Also reconcile indices for old combined format
            logger.info("Reconciling song indices between combined embeddings and JSON song list...")
            data = dict(data)  # Ensure we can modify
            
            (reconciled_indices, valid_mask) = self._reconcile_song_indices(
                data['song_indices'], data['songs'], data['artists']
            )
            
            # Filter all arrays to keep only valid embeddings
            if valid_mask is not None:
                data['embeddings'] = data['embeddings'][valid_mask]
                data['field_types'] = data['field_types'][valid_mask]
                data['field_values'] = data['field_values'][valid_mask]
            
            data['song_indices'] = reconciled_indices
            
            return data
        
        # New separate files format
        logger.info(f"Loading embeddings from separate files in: {embeddings_path}")
        
        # Determine directory and base name
        if embeddings_path.is_dir():
            base_dir = embeddings_path
            base_name = ""
        else:
            base_dir = embeddings_path.parent
            # Try to infer base name from the provided path
            if embeddings_path.stem.endswith('_embeddings'):
                # Handle cases like "pop_eval_set_v0_embeddings" -> "pop_eval_set_v0_"
                base_name = embeddings_path.stem.replace('_embeddings', '') + "_"
            else:
                base_name = embeddings_path.stem + "_"
        
        embedding_types = ['full_profile', 'sound_aspect', 'meaning_aspect', 'mood_aspect', 'tags_genres']
        
        all_embeddings = []
        all_song_indices = []
        all_field_types = []
        all_field_values = []
        songs_data = None
        artists_data = None
        
        total_loaded = 0
        
        for embed_type in embedding_types:
            embed_file = base_dir / f"{base_name}{embed_type}_embeddings.npz"
            
            if not embed_file.exists():
                logger.warning(f"Embedding file not found: {embed_file}")
                continue
                
            try:
                data = np.load(embed_file, allow_pickle=True)
                
                # Load songs metadata from first file (should be consistent across all files)
                if songs_data is None:
                    songs_data = data['songs']
                    artists_data = data['artists']
                    logger.info(f"Loaded metadata for {len(songs_data)} songs")
                
                # Load embeddings for this type
                embeddings = data['embeddings']
                song_indices = data['song_indices']
                field_values = data['field_values']
                
                # Create field_types array for this embedding type
                field_types = np.array([embed_type] * len(embeddings), dtype=object)
                
                all_embeddings.append(embeddings)
                all_song_indices.append(song_indices)
                all_field_types.append(field_types)
                all_field_values.append(field_values)
                
                total_loaded += len(embeddings)
                logger.info(f"Loaded {len(embeddings)} {embed_type} embeddings")
                
            except Exception as e:
                logger.error(f"Error loading {embed_type} embeddings: {e}")
                continue
        
        if total_loaded == 0:
            raise FileNotFoundError(f"No embedding files found in {base_dir}")
        
        # Combine all loaded embeddings
        combined_embeddings = np.concatenate(all_embeddings, axis=0)
        combined_song_indices = np.concatenate(all_song_indices, axis=0)
        combined_field_types = np.concatenate(all_field_types, axis=0)
        combined_field_values = np.concatenate(all_field_values, axis=0)
        
        # CRITICAL: Reconcile song indices between embeddings and app's song list
        # The embeddings' song_indices refer to positions in songs_data, but we need them
        # to refer to positions in self.songs (loaded from JSON)
        logger.info("Reconciling song indices between embeddings and JSON song list...")
        
        (reconciled_song_indices, valid_mask) = self._reconcile_song_indices(
            combined_song_indices, songs_data, artists_data
        )
        
        # Filter all arrays to keep only valid embeddings
        if valid_mask is not None:
            combined_embeddings = combined_embeddings[valid_mask]
            combined_field_types = combined_field_types[valid_mask]
            combined_field_values = combined_field_values[valid_mask]
        
        combined_data = {
            'songs': songs_data,
            'artists': artists_data,
            'embeddings': combined_embeddings,
            'song_indices': reconciled_song_indices,  # Use reconciled indices
            'field_types': combined_field_types,
            'field_values': combined_field_values
        }
        
        logger.info(f"Successfully loaded {total_loaded} total embeddings from {len(all_embeddings)} embedding files")
        
        return combined_data
    
    def _reconcile_song_indices(self, embedding_song_indices: np.ndarray, 
                               embedding_songs: np.ndarray, embedding_artists: np.ndarray) -> tuple:
        """
        Reconcile song indices between embeddings data and app's song list.
        
        The embeddings contain song_indices that refer to positions in the embeddings' own
        songs/artists arrays. We need to map these to positions in self.songs (from JSON).
        
        Args:
            embedding_song_indices: Original indices from embeddings (refer to embedding_songs positions)
            embedding_songs: Song names from embeddings file
            embedding_artists: Artist names from embeddings file
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (reconciled_indices, valid_mask)
            - reconciled_indices: New indices that refer to positions in self.songs
            - valid_mask: Boolean mask indicating which embeddings to keep
        """
        logger.info(f"Reconciling {len(embedding_song_indices)} embedding indices...")
        
        # Create mapping from embedding song index to app song index
        embedding_to_app_index = {}
        missing_songs = []
        
        for emb_idx in range(len(embedding_songs)):
            song_name = embedding_songs[emb_idx]
            artist_name = embedding_artists[emb_idx]
            song_key = (song_name, artist_name)
            
            if song_key in self.song_lookup:
                app_idx = self.song_lookup[song_key]
                embedding_to_app_index[emb_idx] = app_idx
            else:
                missing_songs.append(f"'{song_name}' by '{artist_name}'")
                embedding_to_app_index[emb_idx] = -1  # Mark as missing
        
        if missing_songs:
            logger.warning(f"Found {len(missing_songs)} songs in embeddings that are not in JSON file:")
            for i, song in enumerate(missing_songs[:5]):  # Show first 5
                logger.warning(f"  - {song}")
            if len(missing_songs) > 5:
                logger.warning(f"  ... and {len(missing_songs) - 5} more")
        
        # Create valid mask first to identify which embeddings to keep
        valid_mask = []
        skipped_embeddings = 0
        
        for i, orig_idx in enumerate(embedding_song_indices):
            if orig_idx in embedding_to_app_index:
                app_idx = embedding_to_app_index[orig_idx]
                if app_idx >= 0:  # Valid mapping
                    valid_mask.append(True)
                else:  # Missing song
                    valid_mask.append(False)
                    skipped_embeddings += 1
            else:
                logger.error(f"Invalid embedding song index: {orig_idx}")
                valid_mask.append(False)
                skipped_embeddings += 1
        
        if skipped_embeddings > 0:
            logger.warning(f"Skipped {skipped_embeddings} embeddings due to missing songs in JSON file")
        
        valid_mask_array = np.array(valid_mask, dtype=bool)
        
        # Now create reconciled indices only for valid embeddings
        reconciled_indices = []
        for i, orig_idx in enumerate(embedding_song_indices):
            if valid_mask_array[i]:  # Only process valid embeddings
                app_idx = embedding_to_app_index[orig_idx]
                reconciled_indices.append(app_idx)
        
        reconciled_array = np.array(reconciled_indices, dtype=int)
        
        logger.info(f"Successfully reconciled {len(reconciled_array)} embedding indices (filtered {skipped_embeddings} invalid)")
        
        # Return None for valid_mask if no embeddings were skipped
        return reconciled_array, valid_mask_array if skipped_embeddings > 0 else None
    
    def _build_indices(self):
        """Build indices for each embedding type."""
        logger.info("Building embedding indices...")
        
        # Group embeddings by type
        for embed_type in ['full_profile', 'sound_aspect', 'meaning_aspect', 'mood_aspect', 'tags_genres']:
            mask = self.embeddings_data['field_types'] == embed_type
            if mask.any():
                self.embedding_indices[embed_type] = {
                    'embeddings': self.embeddings_data['embeddings'][mask],
                    'song_indices': self.embeddings_data['song_indices'][mask],
                    'field_values': self.embeddings_data['field_values'][mask]
                }
                logger.info(f"Built index for {embed_type}: {len(self.embedding_indices[embed_type]['embeddings'])} embeddings")
    
    def _build_text_search_index(self):
        """Build text search index for song/artist/album matching."""
        logger.info("Building text search index...")
        
        # Create searchable text for each song
        search_texts = []
        for song in self.songs:
            metadata = song.get('metadata', {})
            searchable_text = f"{song['original_song']} {song['original_artist']} {metadata.get('album_name', '')}"
            search_texts.append(searchable_text.lower())
        
        # Build TF-IDF index
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(search_texts)
    
    def _filter_history_to_known_songs(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """Filter history to only include songs that exist in our metadata."""
        # Create set of known song keys for fast lookup
        metadata_keys = set((song['original_song'], song['original_artist']) for song in self.songs)
        
        # Filter dataframe to only include known songs
        original_count = len(history_df)
        filtered_df = history_df[
            history_df.apply(lambda x: (x['original_song'], x['original_artist']) in metadata_keys, axis=1)
        ].reset_index(drop=True)
        
        filtered_count = len(filtered_df)
        logger.info(f"Filtered history: {filtered_count}/{original_count} entries match known songs")
        
        return filtered_df
    
    def _load_and_process_history(self):
        """Load and process Spotify Extended Streaming History."""
        try:
            history_path = Path(self.history_path)
            
            if not history_path.exists():
                logger.error(f"History path does not exist: {history_path}")
                return
            
            if not history_path.is_dir():
                logger.error(f"History path is not a directory: {history_path}")
                return
            
            logger.info(f"Loading Spotify Extended Streaming History from: {history_path}")
            
            # Use the new ranking_utils functions
            self.history_df, success = ranking_utils.load_and_process_spotify_history(history_path)
            if not success or self.history_df.empty:
                logger.warning("Failed to load history data")
                return
            
            # Filter to songs we know about
            self.history_df = self._filter_history_to_known_songs(self.history_df)
            
            if len(self.history_df) == 0:
                logger.warning("No history entries match songs in metadata")
                return
            
            # Compute per-song statistics using the new ranking_utils function
            self.user_song_stats = ranking_utils.compute_user_song_stats(self.history_df, self.ranking_config)
            
            self.has_history = True
            logger.info(f"Successfully loaded history with {len(self.history_df)} plays across {len(self.user_song_stats)} songs")
            
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            self.has_history = False
            self.history_df = None
            self.user_song_stats = {}
    
    def search_songs_by_text(self, query: str, limit: int = 10) -> List[Tuple[int, float, str]]:
        """Search for songs using text similarity (for song-to-song search suggestions)."""
        if not query or not self.tfidf_vectorizer:
            return []
        
        query_lower = query.lower().strip()
        
        # Transform query for TF-IDF
        query_vec = self.tfidf_vectorizer.transform([query_lower])
        
        # Compute similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top matches with original threshold
        top_indices = similarities.argsort()[::-1][:limit]
        
        results = []
        tfidf_results_count = 0
        
        # First, try TF-IDF results with original threshold
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Original threshold
                song = self.songs[idx]
                label = f"{song['original_song']} - {song['original_artist']}"
                results.append((int(idx), float(similarities[idx]), label))
                tfidf_results_count += 1
        
        # If TF-IDF didn't find good matches, fall back to exact string matching
        if tfidf_results_count < 3:  # If we have fewer than 3 good TF-IDF matches
            logger.info(f"TF-IDF found only {tfidf_results_count} matches for '{query}', using fallback search")
            
            # Build a list of candidates with their match scores
            fallback_candidates = []
            
            for idx, song in enumerate(self.songs):
                metadata = song.get('metadata', {})
                song_title = song['original_song'].lower()
                artist_name = song['original_artist'].lower()
                album_name = metadata.get('album_name', '').lower()
                
                # Calculate simple string similarity scores
                match_score = 0.0
                
                # Exact matches get highest priority
                if query_lower == song_title:
                    match_score = 1.0
                elif query_lower == artist_name:
                    match_score = 0.9
                elif query_lower == album_name:
                    match_score = 0.8
                # Substring matches
                elif query_lower in song_title:
                    match_score = 0.7
                elif query_lower in artist_name:
                    match_score = 0.6
                elif query_lower in album_name:
                    match_score = 0.5
                # Word boundary matches (for multi-word titles)
                elif any(word.startswith(query_lower) for word in song_title.split()):
                    match_score = 0.4
                elif any(word.startswith(query_lower) for word in artist_name.split()):
                    match_score = 0.3
                
                if match_score > 0:
                    fallback_candidates.append((idx, match_score, song))
            
            # Sort by match score and add to results (avoid duplicates)
            existing_indices = {result[0] for result in results}
            fallback_candidates.sort(key=lambda x: x[1], reverse=True)
            
            for idx, score, song in fallback_candidates:
                if idx not in existing_indices and len(results) < limit:
                    label = f"{song['original_song']} - {song['original_artist']}"
                    results.append((int(idx), float(score), label))
        
        # Sort results by score (descending) and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def get_song_embedding(self, song_idx: int, embed_type: str) -> Optional[np.ndarray]:
        """Get embedding for a specific song and embedding type."""
        if embed_type not in self.embedding_indices:
            return None
        
        indices = self.embedding_indices[embed_type]
        mask = indices['song_indices'] == song_idx
        
        if mask.any():
            return indices['embeddings'][mask][0]
        return None
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding for text query."""
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text,
                encoding_format="float"
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error getting text embedding: {e}")
            raise
    
    def similarity_search(self, query_embedding: np.ndarray, embed_type: str, k: int = 20, offset: int = 0) -> Tuple[List[Dict], int]:
        """Perform similarity search with enhanced personalized ranking.
        
        Returns:
            Tuple[List[Dict], int]: (paginated_results_with_detailed_scores, total_count)
            Each result dict contains: song_idx, similarity, field_value, and component scores
        """
        if embed_type not in self.embedding_indices:
            return [], 0
        
        indices = self.embedding_indices[embed_type]
        embeddings = indices['embeddings']
        song_indices = indices['song_indices']
        field_values = indices['field_values']
        
        # Normalize embeddings for cosine similarity
        query_norm_value = np.linalg.norm(query_embedding)
        if query_norm_value == 0:
            logger.warning("Query embedding is zero vector, returning empty results")
            return [], 0
        
        query_norm = query_embedding / query_norm_value
        embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embedding_norms = np.where(embedding_norms == 0, 1, embedding_norms)
        embeddings_norm = embeddings / embedding_norms
        
        # Compute semantic similarities
        similarities = np.dot(embeddings_norm, query_norm)
        
        # Compute enhanced ranking scores for all candidate songs
        candidate_scores = []
        config = self.ranking_config
        
        for idx in range(len(similarities)):
            song_idx = song_indices[idx]
            semantic_sim = similarities[idx]
            field_value = field_values[idx]
            
            # Get song metadata for popularity
            song = self.songs[song_idx]
            song_key = (song['original_song'], song['original_artist'])
            popularity = song.get('metadata', {}).get('popularity', 50) / 100.0  # Normalize to 0-1
            
            # Compute component scores
            s_t = float(semantic_sim)  # Semantic similarity (already 0-1)
            
            # Personal interest score (I_t) with bounds checking
            if self.has_history and song_key in self.user_song_stats:
                raw_interest = self.user_song_stats[song_key]['interest']
                i_t = np.clip(raw_interest / 100.0, 0, 1)  # Normalize to 0-1 and clip
                ucb = np.clip(self.user_song_stats[song_key]['ucb'], 0, 1)  # Clip UCB too
            else:
                # No history data - use neutral values
                i_t = 0.0  # No personal interest 
                ucb = min(config.beta, 1.0)  # Cap exploration bonus at 1.0
            
            # Damped popularity score (P_t^gamma) with bounds checking
            popularity_normalized = np.clip(popularity, 0, 1)
            p_t = popularity_normalized ** config.gamma
            
            # Composite score with safety checks
            final_score = (
                config.w_sem * s_t +
                config.w_int * i_t +
                config.w_ucb * ucb +
                config.w_pop * p_t
            )
            
            # Ensure final score is finite and reasonable
            if not np.isfinite(final_score):
                logger.warning(f"Non-finite final score for song {song_idx}, using semantic similarity only")
                final_score = s_t
            
            final_score = float(np.clip(final_score, 0, 1))  # Ensure 0-1 range
            
            candidate_scores.append({
                'idx': idx,
                'song_idx': int(song_idx),
                'field_value': str(field_value),
                'final_score': float(final_score),
                # Component scores (both raw and weighted)
                'semantic_similarity': float(s_t),
                'semantic_weighted': float(config.w_sem * s_t),
                'personal_interest': float(i_t),
                'interest_weighted': float(config.w_int * i_t),
                'exploration_bonus': float(ucb),
                'exploration_weighted': float(config.w_ucb * ucb),
                'popularity_score': float(p_t),
                'popularity_weighted': float(config.w_pop * p_t),
                'has_history': bool(self.has_history and song_key in self.user_song_stats)
            })
        
        # Sort by final score (descending)
        candidate_scores.sort(key=lambda x: x['final_score'], reverse=True)
        total_count = len(candidate_scores)
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + k
        paginated_results = candidate_scores[start_idx:end_idx]
        
        # Convert to the expected format (but include detailed scores)
        results = []
        for result in paginated_results:
            results.append(result)
        
        return results, total_count
    
    def get_ranking_weights(self) -> Dict:
        """Get the current ranking algorithm weights for display."""
        return {
            'w_sem': self.ranking_config.w_sem,
            'w_int': self.ranking_config.w_int,
            'w_ucb': self.ranking_config.w_ucb, 
            'w_pop': self.ranking_config.w_pop,
            'has_history': self.has_history,
            'history_songs_count': len(self.user_song_stats) if self.has_history else 0
        }

# Initialize search engine
search_engine = None

def init_search_engine(songs_file: str = None, embeddings_file: str = None, history_path: str = None):
    """Initialize the search engine with data files."""
    global search_engine
    if search_engine is None:
        # Use provided file paths or fall back to parsed arguments or defaults
        if songs_file and embeddings_file:
            songs_path = songs_file
            embeddings_path = embeddings_file
            history_path_arg = history_path
        elif args:
            songs_path = songs_file or args.songs
            embeddings_path = embeddings_file or args.embeddings
            history_path_arg = history_path or args.history
        else:
            # Fallback defaults when no args available (e.g., when imported)
            default_songs = Path(__file__).parent.parent / 'data' / 'eval_set_v2' / 'eval_set_v2_metadata_ready.json'
            default_embeddings = Path(__file__).parent.parent / 'data' / 'eval_set_v2' / 'eval_set_v2_embeddings'
            songs_path = songs_file or str(default_songs)
            embeddings_path = embeddings_file or str(default_embeddings)
            history_path_arg = history_path
        
        # Validate that files exist
        if not Path(songs_path).exists():
            logger.error(f"Songs file not found: {songs_path}")
            raise FileNotFoundError(f"Songs file not found: {songs_path}")
        
        # For embeddings, validate based on format (file vs directory/base path)
        embeddings_path_obj = Path(embeddings_path)
        if embeddings_path_obj.is_file():
            # Old combined format - file must exist
            if not embeddings_path_obj.exists():
                logger.error(f"Embeddings file not found: {embeddings_path}")
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        else:
            # New separate format - directory or parent directory should exist
            if embeddings_path_obj.is_dir():
                # Path is a directory - it exists, validation will happen in _load_embeddings_data
                pass
            elif embeddings_path_obj.parent.exists():
                # Path is a base name - parent directory exists, validation will happen in _load_embeddings_data
                pass
            else:
                logger.error(f"Embeddings path not found: {embeddings_path}")
                raise FileNotFoundError(f"Embeddings path not found: {embeddings_path}")
        
        # Validate history path if provided
        if history_path_arg:
            history_path_obj = Path(history_path_arg)
            if not history_path_obj.exists():
                logger.warning(f"History path does not exist: {history_path_arg}")
                history_path_arg = None
            elif not history_path_obj.is_dir():
                logger.warning(f"History path is not a directory: {history_path_arg}")
                history_path_arg = None
        
        logger.info(f"Initializing search engine with:")
        logger.info(f"  Songs file: {songs_path}")
        logger.info(f"  Embeddings file: {embeddings_path}")
        if history_path_arg:
            logger.info(f"  History path: {history_path_arg} (personalized ranking enabled)")
        else:
            logger.info(f"  History path: None (using semantic similarity only)")
        
        search_engine = MusicSearchEngine(songs_path, embeddings_path, history_path_arg)

# Spotify OAuth setup
def get_spotify_oauth():
    # Construct redirect URI dynamically based on environment
    # Check if we're in production (Railway/deployed) or local development
    if os.getenv('RAILWAY_ENVIRONMENT') or request.host not in ['127.0.0.1:5000', 'localhost:5000']:
        # Production: use the current request host with HTTPS
        redirect_uri = f"https://{request.host}/callback"
    else:
        # Local development: use localhost with HTTP
        host = args.host if args else '127.0.0.1'
        port = args.port if args else 5000
        redirect_uri = f"http://{host}:{port}/callback"
    
    logger.info(f"Using OAuth redirect URI: {redirect_uri}")
    
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=redirect_uri,
        scope=SPOTIFY_SCOPES,
        cache_path=None
    )

@app.route('/')
def index():
    """Main page."""
    init_search_engine()
    
    # Set session start time if not already set
    if 'session_start' not in session:
        session['session_start'] = datetime.now().isoformat()
        
        # Track page load
        track_event('Page Loaded', {
            'page_title': 'Semantic Song Search',
            'is_new_session': True
        })
    
    # Pass debug flag to template
    debug_mode = getattr(args, 'debug', False) if args else False
    return render_template('index.html', debug_mode=debug_mode)

@app.route('/login')
def login():
    """Spotify login."""
    # Track login attempt
    track_event('Login Initiated', {
        'auth_provider': 'spotify'
    })
    
    sp_oauth = get_spotify_oauth()
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    """Spotify OAuth callback."""
    try:
        sp_oauth = get_spotify_oauth()
        code = request.args.get('code')
        token_info = sp_oauth.get_access_token(code)
        session['token_info'] = token_info
        
        # Track successful authentication
        track_event('Authentication Successful', {
            'auth_provider': 'spotify'
        })
        
        return redirect(url_for('index'))
        
    except Exception as e:
        # Track authentication failure
        track_event('Authentication Failed', {
            'auth_provider': 'spotify',
            'error_message': str(e)[:200]  # Limit error message length
        })
        logger.error(f"Authentication failed: {e}")
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    """Clear Spotify session."""
    # Track logout
    track_event('Logout', {
        'auth_provider': 'spotify'
    })
    
    session.pop('token_info', None)
    logger.info("User logged out, session cleared")
    return redirect(url_for('index'))

@app.route('/api/search_suggestions')
def search_suggestions():
    """Get song suggestions for song-to-song search."""
    if search_engine is None:
        return jsonify({'error': 'Search engine not initialized'}), 500
        
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify([])
    
    start_time = datetime.now()
    results = search_engine.search_songs_by_text(query, limit=100)
    search_duration = (datetime.now() - start_time).total_seconds()
    
    # Track suggestion search
    track_event('Search Suggestions Requested', {
        'query_length': len(query),
        'results_count': len(results),
        'search_duration_seconds': round(search_duration, 3)
    })
    
    suggestions = []
    for song_idx, score, label in results:
        song = search_engine.songs[song_idx]
        metadata = song.get('metadata', {})
        suggestions.append({
            'song_idx': int(song_idx),  # Convert numpy.int64 to native Python int
            'label': label,
            'song': song['original_song'],
            'artist': song['original_artist'],
            'album': metadata.get('album_name', ''),
            'cover_url': metadata.get('cover_url', ''),
            'spotify_id': metadata.get('song_id', ''),
            'score': float(score)  # Convert numpy.float64 to native Python float
        })
    
    return jsonify(suggestions)

@app.route('/api/search', methods=['POST'])
def search():
    """Main search endpoint."""
    if search_engine is None:
        return jsonify({'error': 'Search engine not initialized'}), 500
        
    data = request.json
    search_type = data.get('search_type')  # 'text' or 'song'
    embed_type = data.get('embed_type')
    query = data.get('query', '').strip()
    song_idx = data.get('song_idx')  # For song-to-song search
    k = data.get('k', 20)
    offset = data.get('offset', 0)  # For pagination
    # Note: Top artists filtering is now handled client-side for better performance
    
    start_time = datetime.now()
    
    # Validate pagination parameters
    if k <= 0 or k > 100:  # Reasonable limits
        return jsonify({'error': 'k must be between 1 and 100'}), 400
    if offset < 0:
        return jsonify({'error': 'offset must be non-negative'}), 400
    
    if not query and song_idx is None:
        return jsonify({'error': 'Query or song_idx required'}), 400
    
    try:
        if search_type == 'text':
            # Text-to-song search
            query_embedding = search_engine.get_text_embedding(query)
            results, total_count = search_engine.similarity_search(query_embedding, embed_type, k, offset)
            
        elif search_type == 'song':
            # Song-to-song search
            if song_idx is None:
                return jsonify({'error': 'song_idx required for song-to-song search'}), 400
            
            query_embedding = search_engine.get_song_embedding(song_idx, embed_type)
            if query_embedding is None:
                return jsonify({'error': 'Song embedding not found'}), 404
            
            results, total_count = search_engine.similarity_search(query_embedding, embed_type, k, offset)
            
        else:
            return jsonify({'error': 'Invalid search_type'}), 400
        
        # Format results with detailed scoring information
        formatted_results = []
        
        for result in results:
            try:
                # New format includes detailed scoring components
                if isinstance(result, dict):
                    result_song_idx = result['song_idx']
                    field_value = result['field_value']
                    
                    song = search_engine.songs[result_song_idx]
                    metadata = song.get('metadata', {})
                    
                    formatted_result = {
                        'song_idx': result_song_idx,
                        'song': song['original_song'],
                        'artist': song['original_artist'],
                        'album': metadata.get('album_name', ''),
                        'cover_url': metadata.get('cover_url', ''),
                        'spotify_id': metadata.get('song_id', ''),
                        'field_value': field_value,
                        'genres': song.get('genres', []),
                        'tags': song.get('tags', []),
                        # Enhanced scoring information
                        'final_score': result['final_score'],
                        'scoring_components': {
                            'semantic_similarity': result['semantic_similarity'],
                            'semantic_weighted': result['semantic_weighted'],
                            'personal_interest': result['personal_interest'],
                            'interest_weighted': result['interest_weighted'],
                            'exploration_bonus': result['exploration_bonus'],
                            'exploration_weighted': result['exploration_weighted'],
                            'popularity_score': result['popularity_score'],
                            'popularity_weighted': result['popularity_weighted'],
                            'has_history': result['has_history']
                        },
                        # Backwards compatibility
                        'similarity': result['semantic_similarity']
                    }
                    formatted_results.append(formatted_result)
                else:
                    # Legacy format fallback
                    if len(result) == 3:
                        result_song_idx, similarity, field_value = result
                    elif len(result) == 2:
                        result_song_idx, similarity = result
                        field_value = "N/A"
                    else:
                        logger.error(f"Unexpected result format: {result}")
                        continue
                        
                    song = search_engine.songs[result_song_idx]
                    metadata = song.get('metadata', {})
                    
                    formatted_results.append({
                        'song_idx': int(result_song_idx),
                        'song': song['original_song'],
                        'artist': song['original_artist'],
                        'album': metadata.get('album_name', ''),
                        'cover_url': metadata.get('cover_url', ''),
                        'spotify_id': metadata.get('song_id', ''),
                        'similarity': float(similarity),
                        'field_value': field_value,
                        'genres': song.get('genres', []),
                        'tags': song.get('tags', []),
                        # Default values for missing components
                        'final_score': float(similarity),
                        'scoring_components': {
                            'semantic_similarity': float(similarity),
                            'semantic_weighted': float(similarity),
                            'personal_interest': 0.0,
                            'interest_weighted': 0.0,
                            'exploration_bonus': 0.0,
                            'exploration_weighted': 0.0,
                            'popularity_score': 0.0,
                            'popularity_weighted': 0.0,
                            'has_history': False
                        }
                    })
            except (ValueError, IndexError, KeyError) as e:
                logger.error(f"Error processing search result {result}: {e}")
                continue
        
        # Calculate search performance
        search_duration = (datetime.now() - start_time).total_seconds()
        
        # Track successful search with enhanced context
        search_properties = {
            'search_type': search_type,
            'embed_type': embed_type,
            'results_returned': total_count,
            'results_requested': k,
            'search_offset': offset,
            'search_duration_seconds': round(search_duration, 3),
            'is_paginated_search': offset > 0,
            'returned_count': len(formatted_results)
        }
        
        # Add query-specific information
        if search_type == 'text' and query:
            search_properties.update({
                'query': query[:200],  # Limit query length for storage
                'query_length': len(query)
            })
        elif search_type == 'song' and song_idx is not None:
            # Get song details for better tracking
            try:
                song = search_engine.songs[song_idx]
                search_properties.update({
                    'query_song_idx': song_idx,
                    'query_song_name': song['original_song'][:100],  # Limit length
                    'query_artist_name': song['original_artist'][:100]  # Limit length
                })
            except (IndexError, KeyError):
                search_properties['query_song_idx'] = song_idx
        
        track_event('Search Performed', search_properties)
        
        # Get ranking weights for display
        ranking_weights = search_engine.get_ranking_weights()
        
        # Return results with ranking information
        return jsonify({
            'results': formatted_results,
            'search_type': search_type,
            'embed_type': embed_type,
            'query': query,
            'ranking_weights': ranking_weights,
            'pagination': {
                'offset': offset,
                'limit': k,
                'total_count': total_count,
                'has_more': offset + k < total_count,
                'returned_count': len(formatted_results)
            }
        })
        
    except Exception as e:
        # Track search errors
        search_duration = (datetime.now() - start_time).total_seconds()
        track_event('Search Error', {
            'search_type': search_type,
            'embed_type': embed_type,
            'query_length': len(query) if query else 0,
            'error_message': str(e)[:200],
            'search_duration_seconds': round(search_duration, 3)
        })
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_song')
def get_song():
    """Get detailed information about a specific song."""
    if search_engine is None:
        return jsonify({'error': 'Search engine not initialized'}), 500
        
    song_idx = request.args.get('song_idx', type=int)
    if song_idx is None or song_idx < 0 or song_idx >= len(search_engine.songs):
        return jsonify({'error': 'Invalid song_idx'}), 400
    
    song = search_engine.songs[song_idx]
    metadata = song.get('metadata', {})
    
    return jsonify({
        'song_idx': song_idx,
        'song': song['original_song'],
        'artist': song['original_artist'],
        'album': metadata.get('album_name', ''),
        'cover_url': metadata.get('cover_url', ''),
        'spotify_id': metadata.get('song_id', ''),
        'genres': song.get('genres', []),
        'tags': song.get('tags', []),
        'sound': song.get('sound', ''),
        'meaning': song.get('meaning', ''),
        'mood': song.get('mood', '')
    })

@app.route('/api/create_playlist', methods=['POST'])
def create_playlist():
    """Create a Spotify playlist with selected songs."""
    token_info = session.get('token_info')
    if not token_info:
        return jsonify({'error': 'Not authenticated'}), 401
    
    start_time = datetime.now()
    
    # Check if token is expired and refresh if needed
    sp_oauth = get_spotify_oauth()
    if sp_oauth.is_token_expired(token_info):
        try:
            token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
            session['token_info'] = token_info
        except Exception as e:
            logger.error(f"Failed to refresh Spotify token: {e}")
            return jsonify({'error': 'Token refresh failed, please re-authenticate'}), 401
    
    try:
        # Get request data
        data = request.json
        playlist_name = data.get('playlist_name', 'Semantic Song Search Playlist')
        song_count = data.get('song_count', 20)
        song_spotify_ids = data.get('song_spotify_ids', [])
        search_context = data.get('search_context', {})
        
        # Validate inputs
        if not playlist_name.strip():
            return jsonify({'error': 'Playlist name cannot be empty'}), 400
        
        if song_count <= 0 or song_count > 100:
            return jsonify({'error': 'Song count must be between 1 and 100'}), 400
        
        if not song_spotify_ids:
            return jsonify({'error': 'No songs provided'}), 400
        
        # Limit songs to the requested count
        songs_to_add = song_spotify_ids[:song_count]
        
        # Filter out invalid Spotify IDs
        valid_songs = [song_id for song_id in songs_to_add if song_id and song_id.strip()]
        
        if not valid_songs:
            return jsonify({'error': 'No valid Spotify track IDs provided'}), 400
        
        # Create Spotify client
        sp = spotipy.Spotify(auth=token_info['access_token'])
        
        # Get current user info
        user_info = sp.current_user()
        user_id = user_info['id']
        
        # Create playlist
        playlist = sp.user_playlist_create(
            user=user_id,
            name=playlist_name.strip(),
            public=False,  # Create as private by default
            description=f"Created by Semantic Song Search - {len(valid_songs)} tracks"
        )
        
        # Add tracks to playlist
        # Convert track IDs to Spotify URIs
        track_uris = [f"spotify:track:{track_id}" for track_id in valid_songs]
        
        # Add tracks in batches (Spotify API limit is 100 tracks per request)
        batch_size = 100
        for i in range(0, len(track_uris), batch_size):
            batch = track_uris[i:i + batch_size]
            sp.playlist_add_items(playlist['id'], batch)
        
        # Track successful playlist creation with full context
        creation_duration = (datetime.now() - start_time).total_seconds()
        
        # Start with playlist-specific properties
        playlist_properties = {
            'playlist_name': playlist_name.strip()[:100],  # Limit length for storage
            'playlist_name_length': len(playlist_name.strip()),
            'songs_requested': song_count,
            'songs_added': len(valid_songs),
            'creation_duration_seconds': round(creation_duration, 3)
        }
        
        # Add search context if provided
        if search_context:
            # Add search type and embedding type
            if 'search_type' in search_context:
                playlist_properties['search_type'] = search_context['search_type']
            if 'embed_type' in search_context:
                playlist_properties['embed_type'] = search_context['embed_type']
            
            # Add query information
            if search_context.get('search_type') == 'text':
                if 'query' in search_context:
                    playlist_properties['query'] = search_context['query'][:200]  # Limit length
                if 'query_length' in search_context:
                    playlist_properties['query_length'] = search_context['query_length']
            elif search_context.get('search_type') == 'song':
                if 'query_song_idx' in search_context:
                    playlist_properties['query_song_idx'] = search_context['query_song_idx']
                if 'query_song_name' in search_context:
                    playlist_properties['query_song_name'] = search_context['query_song_name'][:100]
                if 'query_artist_name' in search_context:
                    playlist_properties['query_artist_name'] = search_context['query_artist_name'][:100]
            
            # Add filter and selection information
            if 'is_filtered' in search_context:
                playlist_properties['is_filtered'] = search_context['is_filtered']
            if 'is_manual_selection' in search_context:
                playlist_properties['is_manual_selection'] = search_context['is_manual_selection']
            if 'selected_songs_count' in search_context:
                playlist_properties['selected_songs_count'] = search_context['selected_songs_count']
        
        track_event('Playlist Created', playlist_properties)
        
        logger.info(f"Created playlist '{playlist_name}' with {len(valid_songs)} tracks")
        
        return jsonify({
            'success': True,
            'playlist_id': playlist['id'],
            'playlist_url': playlist['external_urls']['spotify'],
            'playlist_name': playlist['name'],
            'track_count': len(valid_songs)
        })
        
    except spotipy.exceptions.SpotifyException as e:
        # Track playlist creation errors
        creation_duration = (datetime.now() - start_time).total_seconds()
        # Build error properties with search context
        error_properties = {
            'error_type': 'spotify_api_error',
            'error_code': e.http_status,
            'error_message': str(e)[:200],
            'songs_requested': song_count,
            'creation_duration_seconds': round(creation_duration, 3)
        }
        
        # Add search context to error tracking
        if search_context:
            if 'search_type' in search_context:
                error_properties['search_type'] = search_context['search_type']
            if 'embed_type' in search_context:
                error_properties['embed_type'] = search_context['embed_type']
        
        track_event('Playlist Creation Failed', error_properties)
        
        logger.error(f"Spotify API error: {e}")
        if e.http_status == 401:
            return jsonify({'error': 'Spotify authentication expired, please login again'}), 401
        elif e.http_status == 403:
            # Check if it's a scope/permission issue
            error_msg = str(e).lower()
            if 'scope' in error_msg or 'insufficient' in error_msg:
                logger.info("Clearing session due to insufficient permissions")
                session.pop('token_info', None)  # Clear the invalid session
                return jsonify({'error': 'Insufficient permissions to create playlists. Please re-authenticate to grant playlist creation access.'}), 403
            else:
                return jsonify({'error': f'Access denied: {e.msg}'}), 403
        else:
            return jsonify({'error': f'Spotify API error: {e.msg}'}), 500
    except Exception as e:
        # Track general playlist creation errors
        creation_duration = (datetime.now() - start_time).total_seconds()
        
        # Build error properties with search context
        error_properties = {
            'error_type': 'general_error',
            'error_message': str(e)[:200],
            'songs_requested': song_count,
            'creation_duration_seconds': round(creation_duration, 3)
        }
        
        # Add search context to error tracking
        if search_context:
            if 'search_type' in search_context:
                error_properties['search_type'] = search_context['search_type']
            if 'embed_type' in search_context:
                error_properties['embed_type'] = search_context['embed_type']
        
        track_event('Playlist Creation Failed', error_properties)
        logger.error(f"Failed to create playlist: {e}")
        return jsonify({'error': 'Failed to create playlist'}), 500

@app.route('/api/token')
def get_token():
    """Get Spotify access token for client-side player."""
    token_info = session.get('token_info')
    if not token_info:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Check if token is expired and refresh if needed
    sp_oauth = get_spotify_oauth()
    if sp_oauth.is_token_expired(token_info):
        try:
            token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
            session['token_info'] = token_info
        except Exception as e:
            logger.error(f"Failed to refresh Spotify token: {e}")
            return jsonify({'error': 'Token refresh failed, please re-authenticate'}), 401
    
    return jsonify({'access_token': token_info['access_token']})

@app.route('/api/test_tracking')
def test_tracking():
    """Test endpoint to verify Mixpanel tracking is working."""
    try:
        # Check if Mixpanel is configured
        if mp is None:
            return jsonify({
                'success': False, 
                'error': 'Mixpanel not configured - MIXPANEL_TOKEN environment variable not set',
                'token_set': bool(MIXPANEL_TOKEN)
            }), 500
        
        track_event('Test Event', {
            'test_property': 'test_value',
            'endpoint': 'api/test_tracking'
        })
        return jsonify({
            'success': True, 
            'message': 'Test event tracked successfully',
            'token_set': bool(MIXPANEL_TOKEN)
        })
    except Exception as e:
        logger.error(f"Test tracking failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/top_artists')
def get_top_artists():
    """Get user's top artists across all time ranges."""
    token_info = session.get('token_info')
    if not token_info:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Check if token is expired and refresh if needed
    sp_oauth = get_spotify_oauth()
    if sp_oauth.is_token_expired(token_info):
        try:
            token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
            session['token_info'] = token_info
        except Exception as e:
            logger.error(f"Failed to refresh Spotify token: {e}")
            return jsonify({'error': 'Token refresh failed, please re-authenticate'}), 401
    
    try:
        # Create Spotify client
        sp = spotipy.Spotify(auth=token_info['access_token'])
        
        # Get top artists for all time ranges
        time_ranges = ['short_term', 'medium_term', 'long_term']
        all_top_artists = set()  # Use set to automatically deduplicate
        
        for time_range in time_ranges:
            try:
                results = sp.current_user_top_artists(limit=50, time_range=time_range)
                if results and 'items' in results:
                    for artist in results['items']:
                        # Ensure artist has a name field before processing
                        if artist and 'name' in artist and artist['name']:
                            # Store artist name in lowercase for case-insensitive matching
                            all_top_artists.add(artist['name'].strip().lower())
                    logger.info(f"Retrieved {len(results['items'])} top artists for {time_range}")
                else:
                    logger.warning(f"Unexpected response format for {time_range}: {results}")
            except Exception as e:
                logger.warning(f"Failed to get top artists for {time_range}: {e}")
                continue
        
        # Convert back to list for JSON serialization
        top_artists_list = list(all_top_artists)
        
        logger.info(f"Retrieved {len(top_artists_list)} unique top artists across all time ranges")
        
        # Handle case where user has no top artists
        if len(top_artists_list) == 0:
            logger.warning("User has no top artists - they may have a new account or insufficient listening history")
        
        return jsonify({
            'top_artists': top_artists_list,
            'count': len(top_artists_list)
        })
        
    except spotipy.exceptions.SpotifyException as e:
        logger.error(f"Spotify API error: {e}")
        if e.http_status == 401:
            return jsonify({'error': 'Spotify authentication expired, please login again'}), 401
        elif e.http_status == 403:
            # Check if it's a scope/permission issue
            error_msg = str(e).lower()
            if 'scope' in error_msg or 'insufficient' in error_msg:
                logger.info("Clearing session due to insufficient permissions for top artists")
                session.pop('token_info', None)  # Clear the invalid session
                return jsonify({
                    'error': 'Insufficient permissions to access top artists. Please logout and login again to grant the required permissions.',
                    'requires_reauth': True
                }), 403
            else:
                return jsonify({'error': f'Access denied: {e.msg}'}), 403
        else:
            return jsonify({'error': f'Spotify API error: {e.msg}'}), 500
    except Exception as e:
        logger.error(f"Failed to get top artists: {e}")
        return jsonify({'error': 'Failed to retrieve top artists'}), 500

@app.route('/api/ranking_weights', methods=['GET'])
def get_ranking_weights():
    """Get current ranking algorithm weights and configuration."""
    if search_engine is None:
        return jsonify({'error': 'Search engine not initialized'}), 500
    
    try:
        weights = search_engine.get_ranking_weights()
        return jsonify(weights)
    except Exception as e:
        logger.error(f"Failed to get ranking weights: {e}")
        return jsonify({'error': 'Failed to retrieve ranking weights'}), 500

@app.route('/api/ranking_weights', methods=['PUT'])
def update_ranking_weights():
    """Update ranking algorithm weights."""
    if search_engine is None:
        return jsonify({'error': 'Search engine not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required weight keys
        required_keys = ['w_sem', 'w_int', 'w_ucb', 'w_pop']
        for key in required_keys:
            if key not in data:
                return jsonify({'error': f'Missing required weight: {key}'}), 400
            if not isinstance(data[key], (int, float)) or data[key] < 0:
                return jsonify({'error': f'Invalid weight value for {key}: must be non-negative number'}), 400
        
        # Validate weights sum to approximately 1.0
        total_weight = sum(data[key] for key in required_keys)
        if abs(total_weight - 1.0) > 0.1:
            return jsonify({'error': f'Weights must sum to approximately 1.0, got {total_weight:.3f}'}), 400
        
        # Update the search engine weights
        search_engine.ranking_config.update_weights(data)
        
        logger.info(f"Updated ranking weights: {data}")
        
        # Return the updated weights
        weights = search_engine.get_ranking_weights()
        return jsonify(weights)
    
    except Exception as e:
        logger.error(f"Failed to update ranking weights: {e}")
        return jsonify({'error': 'Failed to update ranking weights'}), 500

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    
    # Print startup information
    logger.info("Starting Semantic Song Search application...")
    logger.info(f"Server will run on: http://{args.host}:{args.port}")
    logger.info(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    
    # Initialize search engine early to catch any data file issues
    try:
        init_search_engine()
        logger.info("Search engine initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        logger.error("Please check your data file paths and try again.")
        exit(1)
    
    # Start the Flask application
    app.run(debug=args.debug, host=args.host, port=args.port) 