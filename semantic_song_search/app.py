#!/usr/bin/env python3
"""
Semantic Song Search App
Supports text-to-song and song-to-song search with personalized ranking algorithm.
"""
import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from openai import OpenAI
from typing import List, Dict, Tuple
import logging
import argparse
from pathlib import Path
try:
    from . import ranking, data_utils, constants
except ImportError:
    import ranking, data_utils, constants
import uuid
from datetime import datetime, timedelta, timezone
import mixpanel
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("OPENAI_API_KEY not set. Text search will be disabled.")
    openai_client = None
else:
    openai_client = OpenAI(api_key=openai_api_key)

# Initialize Mixpanel (optional)
mp = None
if os.environ.get("MIXPANEL_TOKEN"):
    mp = mixpanel.Mixpanel(os.environ["MIXPANEL_TOKEN"])

# Initialize Flask app
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_FILE_THRESHOLD'] = 500
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', str(uuid.uuid4()))
Session(app)

# Spotify OAuth configuration
SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET')

if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    logger.warning("Spotify credentials not provided. Spotify features will be disabled.")


class MusicSearchEngine:
    """Music search engine with personalized ranking."""
    
    def __init__(self, songs_file: str, embeddings_file: str, history_path: str = None, artist_embeddings_file: str = None):
        self.songs_file = songs_file
        self.embeddings_file = embeddings_file
        self.history_path = history_path
        self.artist_embeddings_file = artist_embeddings_file
        
        # Data structures
        self.songs = []
        self.embeddings_data = None
        self.embedding_indices = {}
        self.embedding_lookup = {}  # (song, artist) -> embedding
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Artist embeddings
        self.artist_embeddings_data = None
        self.artist_embedding_lookup = {}  # artist -> embedding
        
        # V2 Ranking engine
        self.ranking_engine = None
        self.has_history = False
        self.history_df = None
        
        # Similarity matrices for optimized search
        self.user_similarity_matrix = None  # For current user-selected embedding type
        self.artist_similarity_matrix = None  # For artist similarity computations
        self.user_matrix_embed_type = None  # Track which embedding type user matrix uses
        self.artist_matrix_embed_type = constants.DEFAULT_ARTIST_MATRIX_EMBED_TYPE  # Default embedding type for artist similarity
        
        # Performance optimization mappings
        self.song_key_to_index = {}  # (song, artist) -> index mapping for O(1) lookups
        self.artist_to_songs = {}  # artist -> [(song_key, index, song_metadata)] mapping
        
        # Load all data
        self._load_all_data()
    
    def _load_all_data(self):
        """Load songs, embeddings, and initialize ranking engine."""
        # Load song metadata
        self.songs = data_utils.load_song_metadata(self.songs_file)
        
        # Load embeddings
        raw_embedding_indices = data_utils.load_embeddings_data(self.embeddings_file)
        
        # Reconcile indices
        self.songs, self.embedding_indices = data_utils.reconcile_song_indices(
            raw_embedding_indices, self.songs
        )
        
        # Load artist embeddings if provided
        if self.artist_embeddings_file:
            self._load_artist_embeddings()
        
        # Build embedding lookups for all available embedding types
        self.embedding_lookups = {}
        try:
            from . import constants
        except ImportError:
            import constants
            
        for embed_type in constants.EMBEDDING_TYPES:
            try:
                embedding_lookup = data_utils.build_embedding_lookup(
                    self.embedding_indices, self.songs, embed_type
                )
                if embedding_lookup:  # Only store if non-empty
                    self.embedding_lookups[embed_type] = embedding_lookup
                    logger.info(f"Built {embed_type} lookup with {len(embedding_lookup)} songs")
            except Exception as e:
                logger.warning(f"Failed to build {embed_type} lookup: {e}")
        
        # Fallback to full_profile for compatibility 
        self.embedding_lookup = self.embedding_lookups.get('full_profile', {})
        
        # Build performance optimization mappings
        self._build_performance_mappings()
        
        # Build text search index
        self._build_text_search_index()
        
        # Initialize ranking engine
        self._initialize_ranking_engine()
        
        logger.info(f"Loaded {len(self.songs)} songs with embeddings")
    
    def _load_artist_embeddings(self):
        """Load artist embeddings data."""
        logger.info(f"Loading artist embeddings from: {self.artist_embeddings_file}")
        try:
            self.artist_embeddings_data = data_utils.load_artist_embeddings_data(self.artist_embeddings_file)
            
            # Build artist embedding lookup for the default embed type
            embed_type = constants.DEFAULT_ARTIST_EMBED_TYPE
            if embed_type in self.artist_embeddings_data:
                artists = self.artist_embeddings_data[embed_type]['artists']
                embeddings = self.artist_embeddings_data[embed_type]['embeddings']
                
                # Validate artist_indices if present
                if 'artist_indices' in self.artist_embeddings_data[embed_type]:
                    artist_indices = self.artist_embeddings_data[embed_type]['artist_indices']
                    if len(artist_indices) != len(embeddings):
                        logger.warning(f"Artist indices length ({len(artist_indices)}) doesn't match embeddings length ({len(embeddings)})")
                    if len(artist_indices) != len(artists):
                        logger.warning(f"Artist indices length ({len(artist_indices)}) doesn't match artists length ({len(artists)})")
                
                # Normalize embeddings
                from sklearn.preprocessing import normalize
                normalized_embeddings = normalize(embeddings, axis=1)
                
                self.artist_embedding_lookup = {
                    artist: normalized_embeddings[i] 
                    for i, artist in enumerate(artists)
                }
                logger.info(f"Built artist embedding lookup for {len(self.artist_embedding_lookup)} artists using {embed_type}")
            else:
                logger.warning(f"Artist embedding type {embed_type} not found in loaded data")
        except Exception as e:
            logger.error(f"Failed to load artist embeddings: {e}")
            self.artist_embeddings_data = None
            self.artist_embedding_lookup = {}
    
    def _build_performance_mappings(self):
        """Build optimization mappings for fast lookups."""
        logger.info("Building performance optimization mappings...")
        
        # Build song_key to index mapping
        self.song_key_to_index = {}
        self.artist_to_songs = {}
        
        for i, song in enumerate(self.songs):
            song_key = (song['original_song'], song['original_artist'])
            artist = song['original_artist']
            
            # Build song key to index mapping
            self.song_key_to_index[song_key] = i
            
            # Build artist to songs mapping
            if artist not in self.artist_to_songs:
                self.artist_to_songs[artist] = []
            self.artist_to_songs[artist].append((song_key, i, song))
        
        logger.info(f"Built mappings for {len(self.songs)} songs and {len(self.artist_to_songs)} artists")
    
    def _build_text_search_index(self):
        """Build text search index for song/artist/album matching."""
        self.tfidf_vectorizer, self.tfidf_matrix = data_utils.build_text_search_index(self.songs)
    
    def _initialize_ranking_engine(self):
        """Initialize ranking engine with history if available."""
        config = ranking.RankingConfig()
        
        if self.history_path:
            # Load and process history
            self.history_df, success = data_utils.load_and_process_spotify_history(Path(self.history_path))
            
            if success and not self.history_df.empty:
                # Filter to known songs
                self.history_df = data_utils.filter_history_to_known_songs(self.history_df, self.songs)
                
                if len(self.history_df) > 0:
                    # Initialize ranking engine with history
                    self.ranking_engine = ranking.initialize_ranking_engine(
                        self.history_df, self.songs, self.embedding_lookups, config
                    )
                    self.has_history = True
                    logger.info(f"Initialized ranking with {len(self.history_df)} history entries")
                else:
                    logger.warning("No history entries match known songs")
            else:
                logger.warning("Failed to load history, using prior-only ranking")
        
        # Initialize without history if needed
        if self.ranking_engine is None:
            self.ranking_engine = ranking.RankingEngine(config)
            # Even without history, we need track priors for scoring
            self.ranking_engine.compute_track_priors(self.songs)
            logger.info("Initialized ranking engine without history")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding for text query."""
        return data_utils.get_openai_embedding(text, normalize=True)
    
    def get_artist_for_song(self, song_key: Tuple[str, str]) -> str:
        """Extract artist name from song_key tuple."""
        return song_key[1]  # song_key is (song_name, artist_name)
    
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension by checking available embeddings or use default."""        
        # Try to get dimension from song embeddings
        if self.embedding_lookups:
            for _, lookup in self.embedding_lookups.items():
                if lookup:
                    first_embedding = next(iter(lookup.values()))
                    return first_embedding.shape[0]
        
        # Default to 3072 for text-embedding-3-large
        return 3072
    
    def _safe_normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Safely normalize an embedding vector, handling NaN and zero vectors."""
        # Replace any NaN or inf values with zeros
        embedding_clean = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute norm
        norm = np.linalg.norm(embedding_clean)
        
        # Handle zero vector case
        if norm < 1e-12:
            return embedding_clean  # Return zero vector as-is
        
        return embedding_clean / norm
    
    def compute_similarity_matrix(self, embed_type: str) -> np.ndarray:
        """
        Compute N×N similarity matrix for all songs using specified embedding type.
        
        Args:
            embed_type: Type of embeddings to use ('full_profile', 'tags', etc.)
            
        Returns:
            N×N numpy array where element [i,j] is similarity between song i and song j
        """
        if embed_type not in self.embedding_lookups:
            logger.warning(f"Embedding type '{embed_type}' not available for similarity matrix")
            return None
            
        embedding_lookup = self.embedding_lookups[embed_type]
        if not embedding_lookup:
            logger.warning(f"No embeddings found for type '{embed_type}'")
            return None
        
        logger.info(f"Computing {len(self.songs)}×{len(self.songs)} similarity matrix for '{embed_type}' embeddings...")
        
        # Create ordered list of embeddings matching song order
        embeddings_list = []
        valid_indices = []
        
        for i, song in enumerate(self.songs):
            song_key = (song['original_song'], song['original_artist'])
            if song_key in embedding_lookup:
                embeddings_list.append(embedding_lookup[song_key])
                valid_indices.append(i)
            else:
                # Add zero vector for missing embeddings
                embedding_dim = self._get_embedding_dimension()
                embeddings_list.append(np.zeros(embedding_dim))
                valid_indices.append(i)
        
        # Convert to matrix: shape (N, embedding_dim)
        embeddings_matrix = np.array(embeddings_list)
        
        # Normalize all embeddings for cosine similarity
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms < 1e-12, 1.0, norms)
        embeddings_matrix_norm = embeddings_matrix / norms
        
        # Compute similarity matrix: (N, embedding_dim) @ (embedding_dim, N) = (N, N)
        similarity_matrix = embeddings_matrix_norm @ embeddings_matrix_norm.T
        
        logger.info(f"Completed similarity matrix computation for '{embed_type}' ({similarity_matrix.shape})")
        return similarity_matrix
    
    def get_similarity_matrix(self, embed_type: str, matrix_type: str = 'user') -> np.ndarray:
        """
        Get or compute similarity matrix for specified embedding type.
        
        Args:
            embed_type: Embedding type to use
            matrix_type: 'user' for user matrix, 'artist' for artist matrix
            
        Returns:
            Similarity matrix or None if computation fails
        """
        if matrix_type == 'user':
            # Check if we already have the right user matrix
            if (self.user_similarity_matrix is not None and 
                self.user_matrix_embed_type == embed_type):
                return self.user_similarity_matrix
            
            # Compute new user matrix
            logger.info(f"Computing user similarity matrix for embedding type: {embed_type}")
            self.user_similarity_matrix = self.compute_similarity_matrix(embed_type)
            self.user_matrix_embed_type = embed_type
            return self.user_similarity_matrix
            
        elif matrix_type == 'artist':
            # Check if we already have the right artist matrix
            if (self.artist_similarity_matrix is not None and 
                self.artist_matrix_embed_type == embed_type):
                return self.artist_similarity_matrix
            
            # Compute new artist matrix
            logger.info(f"Computing artist similarity matrix for embedding type: {embed_type}")
            self.artist_similarity_matrix = self.compute_similarity_matrix(embed_type)
            self.artist_matrix_embed_type = embed_type
            return self.artist_similarity_matrix
        
        else:
            raise ValueError(f"Invalid matrix_type: '{matrix_type}'. Must be 'user' or 'artist'")
    
    def set_artist_similarity_embed_type(self, embed_type: str) -> bool:
        """
        Set the embedding type to use for artist similarity calculations.
        
        Args:
            embed_type: Embedding type to use ('tags', 'tags_genres', 'full_profile', etc.)
            
        Returns:
            True if successfully set, False if embedding type not available
        """
        if embed_type not in self.embedding_lookups:
            logger.warning(f"Embedding type '{embed_type}' not available for artist similarity")
            return False
        
        if embed_type != self.artist_matrix_embed_type:
            logger.info(f"Changing artist similarity embedding type from '{self.artist_matrix_embed_type}' to '{embed_type}'")
            self.artist_matrix_embed_type = embed_type
            # Clear existing artist matrix to force recomputation
            self.artist_similarity_matrix = None
        
        return True
    
    def ensure_artist_similarity_matrix(self) -> bool:
        """
        Ensure artist similarity matrix is computed for current embed type.
        
        Returns:
            True if matrix is available, False if computation failed
        """
        if self.artist_similarity_matrix is None:
            logger.info(f"Initializing artist similarity matrix with '{self.artist_matrix_embed_type}' embeddings")
            self.artist_similarity_matrix = self.compute_similarity_matrix(self.artist_matrix_embed_type)
            return self.artist_similarity_matrix is not None
        return True
    
    def compute_artist_similarity_v2(self, query_song_key: Tuple[str, str], candidate_song_key: Tuple[str, str], 
                                   query_embedding: np.ndarray = None) -> Tuple[float, float]:
        """
        Compute artist similarity using score-then-average approach (V2) - OPTIMIZED.
        
        Args:
            query_song_key: (song, artist) tuple for query song (for song-to-song search)
            candidate_song_key: (song, artist) tuple for candidate song
            query_embedding: Normalized query embedding (for text-to-song search)
            
        Returns:
            Tuple of (popularity_similarity, personal_similarity)
        """
        # Ensure artist similarity matrix is available
        if not self.ensure_artist_similarity_matrix():
            logger.warning("Artist similarity matrix not available, falling back to zero similarities")
            return 0.0, 0.0
        
        candidate_artist = candidate_song_key[1]
        
        # Use optimized artist-to-songs mapping for O(1) lookup
        if candidate_artist not in self.artist_to_songs:
            return 0.0, 0.0
        
        artist_songs_data = self.artist_to_songs[candidate_artist]
        
        # Get query song index for song-to-song searches (O(1) lookup)
        query_idx = None
        if query_song_key is not None:
            query_idx = self.song_key_to_index.get(query_song_key)
            if query_idx is None:
                logger.warning(f"Query song {query_song_key} not found in index")
                return 0.0, 0.0
        
        # Compute individual similarities for each song by the candidate artist
        similarities = []
        popularity_weights = []
        personal_weights = []
        
        # Pre-fetch embedding lookup for text-to-song searches
        embedding_lookup = None
        if query_song_key is None and self.artist_matrix_embed_type in self.embedding_lookups:
            embedding_lookup = self.embedding_lookups[self.artist_matrix_embed_type]
        
        for song_key, song_idx, song_metadata in artist_songs_data:
            # Get similarity score
            if query_song_key is not None:
                # Song-to-song search: use precomputed matrix (O(1) lookup)
                similarity = self.artist_similarity_matrix[query_idx, song_idx]
            else:
                # Text-to-song search: compute against query embedding
                if embedding_lookup and song_key in embedding_lookup:
                    candidate_embedding = embedding_lookup[song_key]
                    candidate_norm = self._safe_normalize(candidate_embedding)
                    similarity = np.dot(query_embedding, candidate_norm)
                else:
                    similarity = 0.0
            
            # normalize simlarity to [0, 1]
            similarity = np.clip((similarity + 1) / 2, 0, 1)
            similarities.append(similarity)
            
            # Get popularity weight from cached metadata
            popularity_weight = 0.0
            if ('metadata' in song_metadata and song_metadata['metadata'] and 
                'popularity' in song_metadata['metadata']):
                popularity_weight = song_metadata['metadata']['popularity'] / 100.0
            
            popularity_weights.append(popularity_weight)
            
            # Get personal weight (R_t_s)
            personal_weight = 0.0
            if (hasattr(self.ranking_engine, 'track_stats') and 
                self.ranking_engine.track_stats and 
                song_key in self.ranking_engine.track_stats):
                track_data = self.ranking_engine.track_stats[song_key]
                if 'R_t' in track_data:
                    personal_weight = np.clip(track_data['R_t'], 0, 1)
            
            personal_weights.append(personal_weight)
        
        # Handle empty similarities list
        if not similarities:
            return 0.0, 0.0
        
        # Compute weighted averages of similarity scores
        similarities = np.array(similarities)
        popularity_weights = np.array(popularity_weights)
        personal_weights = np.array(personal_weights)
        
        # Popularity-weighted average
        pop_total_weight = np.sum(popularity_weights)
        if pop_total_weight > 1e-12:  # More robust zero check
            artist_pop_similarity = np.average(similarities, weights=popularity_weights)
        else:
            artist_pop_similarity = np.mean(similarities)
        
        # Personal-weighted average
        personal_total_weight = np.sum(personal_weights)
        if personal_total_weight > 1e-12:  # More robust zero check
            artist_personal_similarity = np.average(similarities, weights=personal_weights)
        else:
            # Fallback to unweighted average if no personal listening data
            artist_personal_similarity = np.mean(similarities)
        
        return float(artist_pop_similarity), float(artist_personal_similarity)
    
    def get_song_embedding(self, song_idx: int, embed_type: str = 'full_profile') -> np.ndarray:
        """
        Get embedding for a specific song by index.
        
        Args:
            song_idx: Index of the song in self.songs
            embed_type: Type of embedding to retrieve
            
        Returns:
            Song embedding as numpy array
            
        Raises:
            IndexError: If song_idx is out of range
            ValueError: If embed_type is not available
            KeyError: If song has no embedding of the specified type
        """
        if song_idx >= len(self.songs) or song_idx < 0:
            raise IndexError(f"Song index {song_idx} out of range (0-{len(self.songs)-1})")
        
        if embed_type not in self.embedding_lookups:
            available_types = list(self.embedding_lookups.keys())
            raise ValueError(f"Embedding type '{embed_type}' not available. Available types: {available_types}")
        
        song = self.songs[song_idx]
        song_key = (song['original_song'], song['original_artist'])
        
        embedding_lookup = self.embedding_lookups[embed_type]
        if song_key not in embedding_lookup:
            raise KeyError(f"No '{embed_type}' embedding for song: {song_key}")
        
        return embedding_lookup[song_key]
    
    def compute_artist_similarity(self, query_artist_or_text: str, candidate_artist: str, is_text_query: bool = False) -> float:
        """
        Compute artist-artist similarity.
        
        Args:
            query_artist_or_text: Artist name (song-to-song) or text query (text-to-song search)
            candidate_artist: Candidate song's artist
            is_text_query: True for text-to-song search, False for song-to-song search
            
        Returns:
            Cosine similarity score [0, 1] or 0.0 if embeddings not available
        """
        if not self.artist_embedding_lookup:
            return 0.0
        
        # Get candidate artist embedding
        if candidate_artist not in self.artist_embedding_lookup:
            return 0.0
        
        candidate_embedding = self.artist_embedding_lookup[candidate_artist]
        
        if is_text_query:
            # Text-to-song: embed the text query and compare with artist
            if not openai_client:
                return 0.0
            try:
                response = openai_client.embeddings.create(
                    model=constants.OPENAI_EMBEDDING_MODEL,
                    input=query_artist_or_text
                )
                query_embedding = np.array(response.data[0].embedding)
                # Normalize query embedding
                from sklearn.preprocessing import normalize
                query_embedding = normalize(query_embedding.reshape(1, -1), axis=1)[0]
            except Exception as e:
                logger.warning(f"Failed to create embedding for text query: {e}")
                return 0.0
        else:
            # Song-to-song: get query artist embedding
            if query_artist_or_text not in self.artist_embedding_lookup:
                return 0.0
            query_embedding = self.artist_embedding_lookup[query_artist_or_text]
        
        # Compute cosine similarity
        similarity = np.dot(query_embedding, candidate_embedding)
        return float(np.clip(similarity, 0, 1))
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 20, offset: int = 0, 
                         embed_type: str = 'full_profile', lambda_val: float = 0.5,
                         familiarity_min: float = 0.0, familiarity_max: float = 1.0,
                         query_song_key: Tuple[str, str] = None, 
                         genre_query_embedding: np.ndarray = None,
                         query_text: str = None,
                         **advanced_params) -> Tuple[List[Dict], int]:
        """
        Perform V2.6 similarity search with personalized ranking and multi-dimensional similarity.
        
        Args:
            query_embedding: Normalized query embedding
            k: Number of results to return
            offset: Pagination offset
            embed_type: Type of embeddings to use for similarity ('full_profile', 'sound_aspect', etc.)
            lambda_val: Weight for semantic vs personal utility (0=personal, 1=semantic)
            familiarity_min: Minimum familiarity score to include (0.0-1.0)
            familiarity_max: Maximum familiarity score to include (0.0-1.0)
            query_song_key: If provided, enables song-to-song search with artist similarities (song_name, artist_name)
            genre_query_embedding: Optional genre query embedding for dual similarity scoring
            query_text: Original text query for text-to-song artist similarity (optional)
        
        Returns:
            Tuple of (results, total_count)
        """
        # Update lambda_val in config for scoring
        self.ranking_engine.config.lambda_val = lambda_val
        logger.info(f"🔧 Updated lambda_val in ranking config: {lambda_val}")
        
        # Update advanced parameters if provided
        if advanced_params:
            logger.info(f"🔧 Updating ranking config with advanced params: {advanced_params}")
            
            # Update configuration
            self.ranking_engine.config.update_weights(advanced_params)
            logger.info(f"🔧 Updated H_c in config: {self.ranking_engine.config.H_c}")
            
            # Check if any critical parameters that require re-initialization have changed
            critical_params = ['H_c', 'H_E', 'knn_embed_type', 'gamma_s', 'gamma_f', 'kappa', 'alpha_0', 'beta_0', 
                             'K_s', 'K_E', 'gamma_A', 'eta', 'tau', 'beta_f', 'K_life', 'K_recent', 
                             'psi', 'k_neighbors', 'sigma', 'theta_c', 'tau_c', 
                             'K_c', 'tau_K', 'M_A', 'K_fam', 'R_min', 'C_fam', 'min_plays', 'beta_genre', 'beta_pop']
            needs_reinit = any(param in advanced_params for param in critical_params)
            
            if needs_reinit:
                logger.info("🔧 Parameters requiring re-initialization changed, rebuilding ranking engine...")
                self.ranking_engine.reinitialize_with_new_config(
                    self.history_df,
                    self.songs, 
                    self.embedding_lookups
                )
            else:
                logger.info("🔧 Only minor parameters changed, no re-initialization needed")
        else:
            logger.info("🔧 No advanced parameters provided")
        
        # Get the appropriate embedding lookup for the specified type
        if embed_type not in self.embedding_lookups:
            logger.warning(f"Embedding type {embed_type} not available, falling back to full_profile")
            embed_type = 'full_profile'
        
        embedding_lookup = self.embedding_lookups.get(embed_type, {})
        if not embedding_lookup:
            logger.error(f"No embeddings available for type {embed_type}")
            return [], 0
        
        # V2.6: Compute candidates and their priors for quantile normalization
        candidates_data = []
        
        # Try to use precomputed similarity matrix for song-to-song searches
        use_precomputed_matrix = False
        query_song_idx = None
        user_matrix = None
        
        if query_song_key is not None:
            # Song-to-song search: try to use precomputed matrix
            user_matrix = self.get_similarity_matrix(embed_type, 'user')
            if user_matrix is not None:
                # Find query song index
                for i, song in enumerate(self.songs):
                    if (song['original_song'], song['original_artist']) == query_song_key:
                        query_song_idx = i
                        break
                
                if query_song_idx is not None:
                    use_precomputed_matrix = True
                    logger.debug(f"Using precomputed similarity matrix for song-to-song search")
        
        for i, song in enumerate(self.songs):
            song_key = (song['original_song'], song['original_artist'])
            
            # Compute semantic similarity
            if use_precomputed_matrix:
                # Use precomputed matrix for song-to-song search
                semantic_similarity = user_matrix[query_song_idx, i]
            else:
                # Use direct embedding computation
                if song_key in embedding_lookup:
                    song_embedding = embedding_lookup[song_key]
                    semantic_similarity = np.dot(query_embedding, song_embedding)
                else:
                    # No embedding available for this type, skip
                    continue
            
            # normalize semantic similarity to [0, 1]
            semantic_similarity = np.clip((semantic_similarity + 1) / 2, 0, 1)
            
            candidates_data.append({
                'song_idx': i,
                'song_key': song_key,
                'song': song,
                'semantic_similarity': semantic_similarity,
            })
        
        if not candidates_data:
            return [], 0
        
        
        # Compute V2.6 final scores with familiarity filtering
        candidate_scores = []
        
        for candidate in candidates_data:
            # Compute artist similarities for multi-dimensional search
            artist_pop_similarity = 0.0
            artist_personal_similarity = 0.0
            
            try:
                # Use V2 score-then-average artist similarity method
                artist_pop_similarity, artist_personal_similarity = self.compute_artist_similarity_v2(
                    query_song_key, candidate['song_key'], query_embedding
                )
                
            except Exception as e:
                logger.warning(f"Error computing V2 artist similarities for {candidate['song_key']}: {e}")
                # Keep default values of 0.0 for both similarities
            
            # Compute genre similarity if genre query embedding is provided
            genre_similarity = None
            if genre_query_embedding is not None:
                try:
                    # Get genre embedding for the candidate song
                    if 'genres' in self.embedding_lookups and candidate['song_key'] in self.embedding_lookups['genres']:
                        candidate_genre_embedding = self.embedding_lookups['genres'][candidate['song_key']]
                        candidate_genre_norm = self._safe_normalize(candidate_genre_embedding)
                        genre_query_norm = self._safe_normalize(genre_query_embedding)
                        genre_similarity = np.dot(genre_query_norm, candidate_genre_norm)
                        # Normalize to [0, 1] range
                        genre_similarity = np.clip((genre_similarity + 1) / 2, 0, 1)
                    else:
                        # No genre embedding available for this song
                        genre_similarity = 0.0
                except Exception as e:
                    logger.warning(f"Error computing genre similarity for {candidate['song_key']}: {e}")
                    genre_similarity = 0.0
            
            # Compute artist-artist similarity
            artist_similarity = 0.0
            if query_song_key is not None:
                # Song-to-song search: compare query artist with candidate artist
                query_artist = query_song_key[1]
                candidate_artist = candidate['song_key'][1]
                artist_similarity = self.compute_artist_similarity(query_artist, candidate_artist, is_text_query=False)
            elif query_text is not None:
                # Text-to-song search: compare text query with candidate artist
                candidate_artist = candidate['song_key'][1]
                artist_similarity = self.compute_artist_similarity(query_text, candidate_artist, is_text_query=True)
            
            final_score, components = self.ranking_engine.compute_v25_final_score(
                candidate['semantic_similarity'], 
                candidate['song_key'],
                artist_pop_similarity,
                artist_personal_similarity,
                genre_similarity,
                artist_similarity
            )
            
            # Get familiarity score for filtering
            if candidate['song_key'] in self.ranking_engine.track_priors:
                fam_t = self.ranking_engine.track_priors[candidate['song_key']]['Fam_t']
            else:
                # If no history, use a default low familiarity
                fam_t = 0.0
            
            # Apply familiarity filtering
            if familiarity_min <= fam_t <= familiarity_max:
                candidate_scores.append({
                    'song_idx': candidate['song_idx'],
                    'song_key': candidate['song_key'],
                    'final_score': final_score,
                    'semantic_similarity': candidate['semantic_similarity'],
                    'familiarity': fam_t,
                    **components  # Include all component scores for debugging
                })
        
        # Sort by final score
        candidate_scores.sort(key=lambda x: x['final_score'], reverse=True)
        total_count = len(candidate_scores)
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + k
        paginated_results = candidate_scores[start_idx:end_idx]
        
        # Convert to V2.6 result format
        results = []
        for result in paginated_results:
            song_idx = result['song_idx']
            song = self.songs[song_idx]
            song_key = result['song_key']
            
            # Build result dict with V2.6 structure
            result_dict = {
                'song_idx': song_idx,
                'song': song['original_song'],
                'artist': song['original_artist'],
                'cover_url': song.get('metadata', {}).get('cover_url'),
                'album': song.get('metadata', {}).get('album_name', 'Unknown Album'),
                'spotify_id': song.get('metadata', {}).get('song_id', ''),
                'field_value': self._get_field_value(song_key, embed_type),
                'genres': song.get('genres', []),
                'tags': song.get('tags', []),
                'final_score': result['final_score'],
                'scoring_components': result  # Include all V2.6 components
            }
            
            # Add has_history flag for compatibility
            result_dict['scoring_components']['has_history'] = song_key in (
                self.ranking_engine.track_stats if self.ranking_engine.track_stats else {}
            )
            
            results.append(result_dict)
        
        return results, total_count
    
    def _get_field_value(self, song_key: Tuple[str, str], embed_type: str) -> str:
        """Get field value for a song and embedding type."""
        try:
            if embed_type in self.embedding_indices:
                indices = self.embedding_indices[embed_type]
                
                # Find the song in the embedding data
                for i, song_idx in enumerate(indices['song_indices']):
                    if song_idx < len(self.songs):
                        song = self.songs[song_idx]
                        if (song['original_song'], song['original_artist']) == song_key:
                            if 'field_values' in indices and i < len(indices['field_values']):
                                return str(indices['field_values'][i])
                            break
            
            # Fallback based on embedding type
            if embed_type == 'full_profile':
                return 'Full profile'
            elif embed_type in ['sound_aspect', 'meaning_aspect', 'mood_aspect', 'tags_genres']:
                return embed_type.replace('_', ' ').title()
            else:
                return 'N/A'
        except Exception as e:
            logger.debug(f"Error getting field value for {song_key}, {embed_type}: {e}")
            return 'N/A'
    
    def search_songs_by_text(self, query: str, limit: int = 10) -> List[Tuple[int, float, str]]:
        """Search for songs using text similarity (for song-to-song search suggestions)."""
        if not query or not self.tfidf_vectorizer:
            return []
        
        return data_utils.search_songs_by_text(
            query, self.tfidf_vectorizer, self.tfidf_matrix, self.songs, limit
        )
    
    def get_ranking_weights(self) -> Dict:
        """Get current V2.6 ranking weights for display."""
        config = self.ranking_engine.config
        weights = config.to_dict()
        weights.update({
            'version': '2.6',
            'has_history': self.has_history,
            'history_songs_count': len(self.ranking_engine.track_stats) if self.ranking_engine.track_stats else 0,
            'artist_similarity_embed_type': self.artist_matrix_embed_type,
            'user_matrix_embed_type': self.user_matrix_embed_type,
            'has_artist_matrix': self.artist_similarity_matrix is not None,
            'has_user_matrix': self.user_similarity_matrix is not None
        })
        return weights


# Initialize search engine
search_engine = None

def init_search_engine(songs_file: str = None, embeddings_file: str = None, history_path: str = None, artist_embeddings_file: str = None):
    """Initialize the search engine with data files."""
    global search_engine
    if search_engine is None:
        # Use provided file paths or fall back to parsed arguments or defaults
        if songs_file and embeddings_file:
            songs_path = songs_file
            embeddings_path = embeddings_file
            history_path_arg = history_path
            artist_embeddings_path = artist_embeddings_file
        elif 'args' in globals() and args:
            songs_path = songs_file or args.songs
            embeddings_path = embeddings_file or args.embeddings
            history_path_arg = history_path or args.history
            artist_embeddings_path = artist_embeddings_file or getattr(args, 'artist_embeddings', constants.DEFAULT_ARTIST_EMBEDDINGS_PATH)
        else:
            # Fallback defaults
            default_songs = Path(__file__).parent.parent / constants.DEFAULT_SONGS_FILE
            default_embeddings = Path(__file__).parent.parent / constants.DEFAULT_EMBEDDINGS_PATH
            default_artist_embeddings = Path(__file__).parent.parent / constants.DEFAULT_ARTIST_EMBEDDINGS_PATH
            songs_path = songs_file or str(default_songs)
            embeddings_path = embeddings_file or str(default_embeddings)
            history_path_arg = history_path
            artist_embeddings_path = artist_embeddings_file or str(default_artist_embeddings)
        
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
        if artist_embeddings_path and Path(artist_embeddings_path).exists():
            logger.info(f"  Artist embeddings: {artist_embeddings_path}")
        else:
            logger.info(f"  Artist embeddings: None (artist similarity disabled)")
            artist_embeddings_path = None  # Don't pass invalid path to search engine
        
        # Create search engine
        search_engine = MusicSearchEngine(songs_path, embeddings_path, history_path_arg, artist_embeddings_path)


# Flask routes (keep most of the existing routes, update search endpoint)
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
    debug_mode = getattr(args, 'debug', False) if 'args' in globals() and args else False
    return render_template('index.html', debug_mode=debug_mode)

@app.route('/api/search', methods=['POST'])
def search():
    """Main search endpoint with personalized ranking."""
    if search_engine is None:
        return jsonify({'error': 'Search engine not initialized'}), 500
    
    start_time = datetime.now()
    
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        genre_query = data.get('genre_query', '').strip()  # Optional genre query for dual similarity
        search_type = data.get('search_type', data.get('type', 'text'))  # Accept both for compatibility
        embed_type = data.get('embed_type', 'full_profile')  # Add embedding type parameter
        limit = int(data.get('k', data.get('limit', 20)))  # Accept both 'k' and 'limit'
        offset = int(data.get('offset', 0))
        song_idx = data.get('song_idx')
        
        # Personalization parameters
        lambda_val = float(data.get('lambda_val', 0.5))  # Semantic vs personal weight
        lambda_val = np.clip(lambda_val, 0.0, 1.0)  # Ensure valid range
        familiarity_min = float(data.get('familiarity_min', 0.0))  # Min familiarity
        familiarity_max = float(data.get('familiarity_max', 1.0))  # Max familiarity
        familiarity_min = np.clip(familiarity_min, 0.0, 1.0)
        familiarity_max = np.clip(familiarity_max, 0.0, 1.0)
        
        # Ensure min <= max
        if familiarity_min > familiarity_max:
            familiarity_min, familiarity_max = familiarity_max, familiarity_min
        
        # Extract advanced ranking parameters
        advanced_params = {}
        # List of valid advanced parameter names (matching RankingConfig)
        valid_advanced_params = {
            'H_c', 'H_E', 'gamma_s', 'gamma_f', 'kappa', 'alpha_0', 'beta_0', 'K_s',
            'K_E', 'gamma_A', 'eta', 'tau', 'beta_f', 'K_life', 'K_recent', 'psi',
            'k_neighbors', 'sigma', 'knn_embed_type', 'beta_p', 'beta_s', 'beta_a',
            'kappa_E', 'theta_c', 'tau_c', 'K_c', 'tau_K', 'M_A', 'K_fam', 'R_min',
            'C_fam', 'min_plays', 'beta_track', 'beta_artist_pop', 'beta_artist_personal', 'beta_genre', 'beta_pop'
        }
        
        for param_name in valid_advanced_params:
            if param_name in data:
                advanced_params[param_name] = data[param_name]
        
        logger.info(f"🔧 Advanced parameters received: {advanced_params}")
        logger.info(f"🔧 Lambda value: {lambda_val}")
        logger.info(f"🔧 Familiarity range: [{familiarity_min}, {familiarity_max}]")
        
        # Validate pagination parameters
        if limit <= 0 or limit > 100:  # Reasonable limits
            return jsonify({'error': 'limit must be between 1 and 100'}), 400
        if offset < 0:
            return jsonify({'error': 'offset must be non-negative'}), 400
        
        if not query_text and search_type == 'text':
            return jsonify({'error': 'Query text is required'}), 400
        
        if search_type == 'text':
            # Text-to-song search
            query_embedding = search_engine.get_text_embedding(query_text)
            
            # Get genre query embedding if genre query is provided
            genre_query_embedding = None
            if genre_query:
                genre_query_embedding = search_engine.get_text_embedding(genre_query)
                
            results, total_count = search_engine.similarity_search(
                query_embedding, k=limit, offset=offset, embed_type=embed_type,
                lambda_val=lambda_val, familiarity_min=familiarity_min, familiarity_max=familiarity_max,
                genre_query_embedding=genre_query_embedding,
                query_text=query_text,
                **advanced_params
            )
        
        elif search_type == 'song':
            # Song-to-song search
            if song_idx is None:
                return jsonify({'error': 'song_idx is required for song search'}), 400
            
            if song_idx >= len(search_engine.songs):
                return jsonify({'error': 'Invalid song_idx'}), 400
            
            reference_song = search_engine.songs[song_idx]
            song_key = (reference_song['original_song'], reference_song['original_artist'])
            
            # Get embedding for the specified type
            embedding_lookup = search_engine.embedding_lookups.get(embed_type, {})
            if song_key in embedding_lookup:
                query_embedding = embedding_lookup[song_key]
                
                # Get genre embedding for the reference song (for genre similarity)
                genre_query_embedding = None
                if 'genres' in search_engine.embedding_lookups and song_key in search_engine.embedding_lookups['genres']:
                    genre_query_embedding = search_engine.embedding_lookups['genres'][song_key]
                
                results, total_count = search_engine.similarity_search(
                    query_embedding, k=limit, offset=offset, embed_type=embed_type,
                    lambda_val=lambda_val, familiarity_min=familiarity_min, familiarity_max=familiarity_max,
                    query_song_key=song_key,  # Enable song-to-song artist similarities
                    genre_query_embedding=genre_query_embedding,
                    **advanced_params
                )
            else:
                return jsonify({'error': f'No {embed_type} embedding available for reference song'}), 400
        
        else:
            return jsonify({'error': 'Invalid search type'}), 400
        
        # Calculate search performance
        search_duration = (datetime.now() - start_time).total_seconds()
        
        # Track successful search with enhanced context
        search_properties = {
            'search_type': search_type,
            'embed_type': embed_type,
            'results_returned': total_count,
            'results_requested': limit,
            'search_offset': offset,
            'search_duration_seconds': round(search_duration, 3),
            'is_paginated_search': offset > 0,
            'returned_count': len(results),
            'lambda_val': lambda_val,
            'familiarity_min': familiarity_min,
            'familiarity_max': familiarity_max
        }
        
        # Add query-specific information
        if search_type == 'text' and query_text:
            search_properties.update({
                'query': query_text[:200],  # Limit query length for storage
                'query_length': len(query_text)
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
        
        return jsonify({
            'results': results,
            'search_type': search_type,
            'embed_type': embed_type,
            'query': query_text,
            'ranking_weights': search_engine.get_ranking_weights(),
            'pagination': {
                'offset': offset,
                'limit': limit,
                'total_count': total_count,
                'has_more': offset + limit < total_count,
                'returned_count': len(results)
            }
        })
    
    except Exception as e:
        # Track search errors
        search_duration = (datetime.now() - start_time).total_seconds()
        track_event('Search Error', {
            'search_type': search_type if 'search_type' in locals() else 'unknown',
            'embed_type': embed_type if 'embed_type' in locals() else 'unknown',
            'query_length': len(query_text) if 'query_text' in locals() and query_text else 0,
            'error_message': str(e)[:200],
            'search_duration_seconds': round(search_duration, 3)
        })
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

@app.route('/api/ranking_weights', methods=['GET'])
def get_ranking_weights():
    """Get current ranking weights and configuration."""
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
    """Update ranking weights."""
    if search_engine is None:
        return jsonify({'error': 'Search engine not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Update the ranking engine config
        search_engine.ranking_engine.config.update_weights(data)
        
        logger.info(f"Updated ranking weights: {data}")
        
        # Return the updated weights
        weights = search_engine.get_ranking_weights()
        return jsonify(weights)
    
    except Exception as e:
        logger.error(f"Failed to update ranking weights: {e}")
        return jsonify({'error': 'Failed to update ranking weights'}), 500

# Helper functions for routes

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

def get_spotify_oauth():
    """Get Spotify OAuth object with dynamic redirect URI."""
    # Check if we're in production or local development
    if os.getenv('RAILWAY_ENVIRONMENT') or request.host not in ['127.0.0.1:5000', 'localhost:5000']:
        # Production: use the current request host with HTTPS
        redirect_uri = f"https://{request.host}/callback"
    else:
        # Local development: use localhost with HTTP
        redirect_uri = f"http://{constants.DEFAULT_HOST}:{constants.DEFAULT_PORT}/callback"
    
    logger.info(f"Using OAuth redirect URI: {redirect_uri}")
    
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=redirect_uri,
        scope=constants.SPOTIFY_SCOPES,
        cache_path=None
    )

# Additional API routes

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
        user = sp.current_user()
        user_id = user['id']
        
        # Create playlist
        playlist = sp.user_playlist_create(
            user_id, 
            playlist_name, 
            public=False, 
            description=f"Created by Semantic Song Search - {len(valid_songs)} songs"
        )
        
        # Convert Spotify IDs to URIs
        track_uris = [f"spotify:track:{song_id}" for song_id in valid_songs]
        
        # Add songs to playlist (Spotify API limits to 100 tracks per request)
        batch_size = 100
        added_tracks = 0
        for i in range(0, len(track_uris), batch_size):
            batch = track_uris[i:i + batch_size]
            try:
                sp.playlist_add_items(playlist['id'], batch)
                added_tracks += len(batch)
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                break
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Track playlist creation
        track_event('Playlist Created', {
            'playlist_id': playlist['id'],
            'playlist_name': playlist_name,
            'song_count_requested': song_count,
            'songs_added': added_tracks,
            'creation_duration_seconds': round(duration, 3),
            'search_context': search_context
        })
        
        return jsonify({
            'success': True,
            'playlist_url': playlist['external_urls']['spotify'],
            'playlist_id': playlist['id'],
            'playlist_name': playlist['name'],
            'songs_added': added_tracks,
            'total_requested': song_count
        })
        
    except spotipy.exceptions.SpotifyException as e:
        logger.error(f"Spotify API error: {e}")
        return jsonify({'error': f'Spotify error: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Playlist creation error: {e}")
        return jsonify({'error': 'Failed to create playlist'}), 500

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
        session['user_id'] = str(uuid.uuid4())  # Generate session user ID
        
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
    session.pop('user_id', None)
    logger.info("User logged out, session cleared")
    return redirect(url_for('index'))

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

@app.route('/api/test_tracking')
def test_tracking():
    """Test endpoint to verify Mixpanel tracking is working."""
    try:
        # Check if Mixpanel is configured
        if mp is None:
            return jsonify({
                'success': False, 
                'error': 'Mixpanel not configured - MIXPANEL_TOKEN environment variable not set',
                'token_set': bool(os.environ.get('MIXPANEL_TOKEN'))
            }), 500
        
        track_event('Test Event', {
            'test_property': 'test_value',
            'endpoint': 'api/test_tracking'
        })
        return jsonify({
            'success': True, 
            'message': 'Test event tracked successfully',
            'token_set': bool(os.environ.get('MIXPANEL_TOKEN'))
        })
    except Exception as e:
        logger.error(f"Test tracking failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/top_artists')
def top_artists():
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
        sp = spotipy.Spotify(auth=token_info['access_token'])
        
        # Get top artists from different time ranges
        all_artists = {}
        time_ranges = ['short_term', 'medium_term', 'long_term']
        
        for time_range in time_ranges:
            try:
                results = sp.current_user_top_artists(limit=50, time_range=time_range)
                for i, artist in enumerate(results['items']):
                    artist_id = artist['id']
                    if artist_id not in all_artists:
                        all_artists[artist_id] = {
                            'id': artist_id,
                            'name': artist['name'],
                            'genres': artist.get('genres', []),
                            'popularity': artist.get('popularity', 0),
                            'image_url': artist['images'][0]['url'] if artist.get('images') else '',
                            'external_url': artist['external_urls']['spotify'],
                            'rankings': {}
                        }
                    all_artists[artist_id]['rankings'][time_range] = i + 1
            except Exception as e:
                logger.warning(f"Failed to get {time_range} top artists: {e}")
        
        # Convert to list and sort by overall popularity/ranking
        artists_list = list(all_artists.values())
        artists_list.sort(key=lambda x: (
            len(x['rankings']),  # Prefer artists who appear in multiple time ranges
            -x['popularity']      # Then by popularity
        ), reverse=True)
        
        # Track request
        track_event('Top Artists Requested', {
            'artists_count': len(artists_list),
            'time_ranges_successful': len([tr for tr in time_ranges if any(artist['rankings'].get(tr) for artist in artists_list)])
        })
        
        return jsonify({
            'artists': artists_list,  # Return all deduplicated artists (no artificial limit)
            'time_ranges': time_ranges
        })
        
    except Exception as e:
        logger.error(f"Error getting top artists: {e}")
        return jsonify({'error': 'Failed to retrieve top artists'}), 500

@app.route('/api/default_ranking_config')
def get_default_ranking_config():
    """Get default ranking configuration parameters."""
    try:
        # Create a default RankingConfig instance and return its parameters
        default_config = ranking.RankingConfig()
        return jsonify(default_config.to_dict())
    except Exception as e:
        logger.error(f"Error getting default ranking config: {e}")
        return jsonify({'error': 'Failed to retrieve default ranking configuration'}), 500

@app.route('/api/artist_similarity_config', methods=['GET'])
def get_artist_similarity_config():
    """Get current artist similarity configuration."""
    try:
        if search_engine is None:
            return jsonify({'error': 'Search engine not initialized'}), 500
        
        config = {
            'artist_embed_type': search_engine.artist_matrix_embed_type,
            'available_types': list(search_engine.embedding_lookups.keys()),
            'has_matrix': search_engine.artist_similarity_matrix is not None
        }
        return jsonify(config)
    except Exception as e:
        logger.error(f"Error getting artist similarity config: {e}")
        return jsonify({'error': 'Failed to retrieve artist similarity configuration'}), 500

@app.route('/api/artist_similarity_config', methods=['PUT'])
def set_artist_similarity_config():
    """Set artist similarity embedding type."""
    try:
        if search_engine is None:
            return jsonify({'error': 'Search engine not initialized'}), 500
        
        data = request.get_json()
        if not data or 'artist_embed_type' not in data:
            return jsonify({'error': 'Missing artist_embed_type parameter'}), 400
        
        embed_type = data['artist_embed_type']
        
        # Validate and set the embedding type
        success = search_engine.set_artist_similarity_embed_type(embed_type)
        if not success:
            return jsonify({'error': f'Invalid embedding type: {embed_type}'}), 400
        
        # Return updated configuration
        config = {
            'artist_embed_type': search_engine.artist_matrix_embed_type,
            'available_types': list(search_engine.embedding_lookups.keys()),
            'has_matrix': search_engine.artist_similarity_matrix is not None
        }
        return jsonify(config)
        
    except Exception as e:
        logger.error(f"Error setting artist similarity config: {e}")
        return jsonify({'error': 'Failed to set artist similarity configuration'}), 500

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Semantic Song Search and Playlist Creation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py
  python app.py --songs custom_songs.json --embeddings custom_embeddings.npz
  python app.py -s /path/to/songs.json -e /path/to/embeddings.npz
        """
    )
    parser.add_argument('-s', '--songs', type=str, default=constants.DEFAULT_SONGS_FILE, 
                       help=f'Path to songs JSON file (default: {constants.DEFAULT_SONGS_FILE})')
    parser.add_argument('-e', '--embeddings', type=str, default=constants.DEFAULT_EMBEDDINGS_PATH,
                       help=f'Path to embeddings file/directory (supports combined .npz file or directory with separate embedding files) (default: {constants.DEFAULT_EMBEDDINGS_PATH})')  
    parser.add_argument('--history', type=str, default=None, help='Path to Spotify Extended Streaming History directory (optional - enables personalized ranking)')
    parser.add_argument('--artist-embeddings', type=str, default=constants.DEFAULT_ARTIST_EMBEDDINGS_PATH, help=f'Path to artist embeddings directory (default: {constants.DEFAULT_ARTIST_EMBEDDINGS_PATH})')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--host', type=str, default=constants.DEFAULT_HOST, help='Host to run the server on (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=constants.DEFAULT_PORT, help='Port to run the server on (default: 5000)')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    
    # Print startup information
    logger.info("Starting Semantic Song Search application...")
    logger.info(f"Server will run on: http://{args.host}:{args.port}")
    logger.info(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    
    # Initialize search engine early to catch any data file issues
    try:
        init_search_engine(args.songs, args.embeddings, args.history, getattr(args, 'artist_embeddings', None))
        logger.info("Search engine initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        logger.error("Please check your data file paths and try again.")
        exit(1)
    
    # Start the Flask application
    app.run(debug=args.debug, host=args.host, port=args.port)