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
import time

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
    
    def __init__(self, songs_file: str, embeddings_file: str, history_path: str = None, artist_embeddings_file: str = None, shared_genre_store_path: str = None, profiles_file: str = None):
        self.songs_file = songs_file
        self.embeddings_file = embeddings_file
        self.history_path = history_path
        self.artist_embeddings_file = artist_embeddings_file
        self.shared_genre_store_path = shared_genre_store_path
        self.profiles_file = profiles_file
        
        # Data structures
        self.songs = []
        self.embeddings_data = None
        self.embedding_indices = {}
        self.embedding_lookup = {}  # track_id -> embedding
        self.fuzzy_search_index = None  # RapidFuzz search index

        # Tags and genres lookups
        self.tags_lookup = {}  # track_id -> tags list
        self.genres_lookup = {}  # track_id -> genres list
        
        # Artist embeddings
        self.artist_embeddings_data = None
        self.artist_embedding_lookup = {}  # artist -> embedding
        
        # Ranking engine
        self.ranking_engine = None
        self.has_history = False
        self.history_df = None
        
        # Descriptor-based artist data
        self.artist_data = None  # Will contain embeddings, metadata, and genre store
        
        # Legacy lookups removed - using descriptor-based system
        
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
            self._load_artist_data()

        # Load song profiles for tags and genres
        self._load_song_profiles()

        # Build embedding lookups for all available song descriptor types
        self.embedding_lookups = {}
        try:
            from . import constants
        except ImportError:
            import constants
            
        for embed_type in constants.SONG_EMBEDDING_TYPES:
            try:
                embedding_lookup = data_utils.build_embedding_lookup(
                    self.embedding_indices, self.songs, embed_type
                )
                if embedding_lookup:  # Only store if non-empty
                    self.embedding_lookups[embed_type] = embedding_lookup
                    logger.info(f"Built song {embed_type} lookup with {len(embedding_lookup)} songs")
            except Exception as e:
                logger.warning(f"Failed to build song {embed_type} lookup: {e}")
        
        # Use genres as default for compatibility
        self.embedding_lookup = self.embedding_lookups.get('genres', {})
        
        # Build text search index
        self._build_text_search_index()
        
        # Initialize ranking engine
        self._initialize_ranking_engine()
        
        logger.info(f"Loaded {len(self.songs)} songs with embeddings")
    
    def _load_artist_data(self):
        """Load artist embeddings, metadata, and genre store using new descriptor format."""
        logger.info(f"Loading artist data from: {self.artist_embeddings_file}")
        try:
            self.artist_data = data_utils.load_artist_embeddings_data(self.artist_embeddings_file, self.shared_genre_store_path)
            logger.info(f"Successfully loaded artist descriptor data")
        except Exception as e:
            logger.error(f"Failed to load artist data: {e}")
            self.artist_data = None

    def _load_song_profiles(self):
        """Load song profiles JSONL and create tags/genres lookups."""
        if not self.profiles_file:
            logger.info("No profiles file specified, skipping tags/genres loading")
            return

        if not os.path.exists(self.profiles_file):
            logger.warning(f"Profiles file not found: {self.profiles_file}")
            return

        logger.info(f"Loading song profiles from: {self.profiles_file}")

        profile_count = 0
        matched_count = 0

        try:
            with open(self.profiles_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        profile = json.loads(line)
                        profile_count += 1

                        uri = profile.get('uri', '')
                        if uri and uri.startswith('spotify:track:'):
                            # Extract track_id from spotify:track:abc123
                            track_id = uri.split(":")[-1]

                            # Store tags and genres
                            self.tags_lookup[track_id] = profile.get('tags', [])
                            self.genres_lookup[track_id] = profile.get('genres', [])
                            matched_count += 1

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse profile line: {e}")
                        continue

            logger.info(f"Loaded {profile_count} profiles, matched {matched_count} with track IDs")
            logger.info(f"Tags lookup contains {len(self.tags_lookup)} entries")
            logger.info(f"Genres lookup contains {len(self.genres_lookup)} entries")

        except Exception as e:
            logger.error(f"Failed to load song profiles: {e}")

    
    def _build_text_search_index(self):
        """Build RapidFuzz search index for song/artist/album matching."""
        self.fuzzy_search_index = data_utils.build_text_search_index(self.songs)
    
    def compute_song_descriptor_similarity(self, query_embedding: np.ndarray, candidate_track_id: str, 
                                         song_weights: Dict[str, float], search_type: str = 'text') -> float:
        """
        Compute song descriptor similarity using weighted combination of descriptor similarities.
        
        Args:
            query_embedding: Query embedding (single embedding for text search, or dict for song-to-song)
            candidate_track_id: Track ID of candidate song
            song_weights: Dictionary mapping descriptor types to weights (b0-b4)
            search_type: 'text' for text-to-song, 'song' for song-to-song
            
        Returns:
            Weighted song similarity score
        """
        try:
            from . import constants
        except ImportError:
            import constants
        
        total_similarity = 0.0
        total_weight = 0.0
        
        # For each song descriptor type, compute similarity
        for embed_type in constants.SONG_EMBEDDING_TYPES:
            if embed_type not in song_weights:
                continue
                
            weight = song_weights[embed_type]
            if weight == 0:
                continue
                
            # Get candidate embedding for this descriptor type
            if embed_type not in self.embedding_lookups:
                continue
                
            candidate_embedding = self.embedding_lookups[embed_type].get(candidate_track_id)
            if candidate_embedding is None:
                continue
            
            # Compute similarity based on search type
            if search_type == 'text':
                # Text-to-song: single query embedding vs each descriptor type
                similarity = np.dot(query_embedding, candidate_embedding)
            elif search_type == 'song':
                # Song-to-song: query should be dict with descriptor embeddings
                if not isinstance(query_embedding, dict) or embed_type not in query_embedding:
                    continue
                query_desc_embedding = query_embedding[embed_type]
                similarity = np.dot(query_desc_embedding, candidate_embedding)
            else:
                logger.warning(f"Unknown search type: {search_type}")
                continue
            
            # Accumulate weighted similarity
            total_similarity += weight * similarity
            total_weight += weight
        
        # Normalize by total weight if any similarities were computed
        if total_weight > 0:
            return total_similarity / total_weight
        else:
            return 0.0
    
    def compute_pairwise_weighted_sum(self, genres_a: List[Dict], genres_b: List[Dict]) -> float:
        """
        Compute weighted pairwise sum using optimized matrix multiplication (from notebook).
        
        Args:
            genres_a: List of genre info dicts for artist A
            genres_b: List of genre info dicts for artist B  
            
        Returns:
            Weighted sum of pairwise similarities
        """
        if len(genres_a) == 0 or len(genres_b) == 0:
            return 0.0
        
        if not self.artist_data or 'genre_store' not in self.artist_data:
            return 0.0
        
        genre_store = self.artist_data['genre_store']
        
        # Get embeddings and prominence scores
        embeddings_a = []
        prominences_a = []
        for genre_info in genres_a:
            genre_key = genre_info['key']
            if genre_key in genre_store:
                embeddings_a.append(genre_store[genre_key])
                prominences_a.append(genre_info['prominence'])
        
        embeddings_b = []
        prominences_b = []
        for genre_info in genres_b:
            genre_key = genre_info['key']
            if genre_key in genre_store:
                embeddings_b.append(genre_store[genre_key])
                prominences_b.append(genre_info['prominence'])
        
        if len(embeddings_a) == 0 or len(embeddings_b) == 0:
            return 0.0
        
        # Convert to numpy arrays
        embeddings_a = np.array(embeddings_a)
        embeddings_b = np.array(embeddings_b)
        prominences_a = np.array(prominences_a)
        prominences_b = np.array(prominences_b)
        
        # Compute pairwise similarities using matrix multiplication
        similarity_matrix = embeddings_a @ embeddings_b.T
        
        # Weight by prominence products (normalized to [0,1] by dividing by 100)
        weight_matrix = (prominences_a[:, None] * prominences_b[None, :]) / 100.0
        
        # Compute weighted sum
        weighted_sum = np.sum(similarity_matrix * weight_matrix)
        
        return float(weighted_sum)
    
    def compute_normalized_genre_similarity_clean(self, artist_a: str, artist_b: str) -> float:
        """
        Compute self-similarity normalized genre similarity using CLEAN V7 format (from notebook).
        
        This ensures that identical artists get a perfect similarity score of 1.0,
        analogous to cosine similarity normalization.
        
        Formula: cross_sim / sqrt(self_sim_a * self_sim_b)
        
        Args:
            artist_a: First artist name
            artist_b: Second artist name  
            
        Returns:
            Normalized genre similarity score (0-1), with identical artists = 1.0
        """
        # Handle same-artist case when metadata is missing or incomplete
        if artist_a == artist_b:
            # Same artist should always have perfect genre similarity
            if not self.artist_data or 'metadata' not in self.artist_data:
                return 1.0  # Perfect similarity for same artist when metadata unavailable

            artist_metadata = self.artist_data['metadata']
            if artist_a not in artist_metadata:
                return 1.0  # Perfect similarity for same artist when not found in metadata

        # Different artists - check metadata availability
        if not self.artist_data or 'metadata' not in self.artist_data:
            return 0.0

        artist_metadata = self.artist_data['metadata']

        if artist_a not in artist_metadata or artist_b not in artist_metadata:
            return 0.0
        
        # Get genre data for both artists
        genres_a = artist_metadata[artist_a]['genres']
        genres_b = artist_metadata[artist_b]['genres']
        
        if len(genres_a) == 0 or len(genres_b) == 0:
            return 0.0
        
        # Compute cross-similarity
        cross_sim = self.compute_pairwise_weighted_sum(genres_a, genres_b)
        
        # Compute self-similarities
        self_sim_a = self.compute_pairwise_weighted_sum(genres_a, genres_a)
        self_sim_b = self.compute_pairwise_weighted_sum(genres_b, genres_b)
        
        # Normalize by geometric mean of self-similarities (like cosine similarity)
        if self_sim_a <= 0 or self_sim_b <= 0:
            return 0.0
        
        normalized_similarity = cross_sim / np.sqrt(self_sim_a * self_sim_b)
        
        # Clamp to [0, 1] range to handle any numerical issues
        return float(np.clip(normalized_similarity, 0.0, 1.0))
    
    def get_artist_gender(self, artist_name: str) -> str:
        """
        Get the lead vocalist gender for an artist.

        Args:
            artist_name: Name of the artist

        Returns:
            Gender string ('male', 'female', etc.) or empty string if not found
        """
        if not self.artist_data or 'metadata' not in self.artist_data:
            return ""

        if artist_name in self.artist_data['metadata']:
            return self.artist_data['metadata'][artist_name].get('lead_vocalist_gender', '')

        return ""

    def compute_prominence_weighted_similarity(self, query_embedding: np.ndarray, artist_genres: List[Dict]) -> float:
        """
        Compute prominence-weighted average similarity for text-to-song artist genre matching.
        
        Args:
            query_embedding: Single text embedding
            artist_genres: List of genre info dicts with keys and prominence scores
            
        Returns:
            Prominence-weighted average similarity score
        """
        if len(artist_genres) == 0:
            return 0.0
        
        if not self.artist_data or 'genre_store' not in self.artist_data:
            return 0.0
        
        genre_store = self.artist_data['genre_store']
        
        similarities = []
        prominences = []
        
        for genre_info in artist_genres:
            genre_key = genre_info['key']
            if genre_key in genre_store:
                genre_embedding = genre_store[genre_key]
                similarity = np.dot(query_embedding, genre_embedding)
                similarities.append(similarity)
                prominences.append(genre_info['prominence'])
        
        if len(similarities) == 0:
            return 0.0
        
        # Compute prominence-weighted average
        similarities = np.array(similarities)
        prominences = np.array(prominences)
        
        total_prominence = np.sum(prominences)
        if total_prominence > 0:
            weighted_similarity = np.sum(similarities * prominences) / total_prominence
            return float(weighted_similarity)
        else:
            return 0.0
    
    def compute_artist_descriptor_similarity(self, query_data, candidate_artist: str, 
                                           artist_weights: Dict[str, float], search_type: str = 'text') -> float:
        """
        Compute artist descriptor similarity using weighted combination of descriptor similarities.
        
        Args:
            query_data: Query data (single embedding for text search, or artist name for song-to-song)
            candidate_artist: Name of candidate artist
            artist_weights: Dictionary mapping descriptor types to weights (c0-c5)
            search_type: 'text' for text-to-song, 'song' for song-to-song
            
        Returns:
            Weighted artist similarity score
        """
        if not self.artist_data:
            return 0.0
        
        try:
            from . import constants
        except ImportError:
            import constants
        
        total_similarity = 0.0
        total_weight = 0.0
        
        # For each artist descriptor type, compute similarity
        for embed_type in constants.ARTIST_EMBEDDING_TYPES:
            if embed_type not in artist_weights:
                continue
                
            weight = artist_weights[embed_type]
            if weight == 0:
                continue
            
            # Special handling for genres
            if embed_type == 'genres':
                if search_type == 'text':
                    # Text-to-song: prominence-weighted similarity
                    if 'metadata' in self.artist_data and candidate_artist in self.artist_data['metadata']:
                        candidate_genres = self.artist_data['metadata'][candidate_artist]['genres']
                        similarity = self.compute_prominence_weighted_similarity(query_data, candidate_genres)
                    else:
                        similarity = 0.0
                elif search_type == 'song':
                    # Song-to-song: normalized genre similarity
                    query_artist = query_data  # Should be artist name
                    similarity = self.compute_normalized_genre_similarity_clean(query_artist, candidate_artist)
                else:
                    similarity = 0.0
            else:
                # Regular descriptor similarity
                if 'embeddings' not in self.artist_data or embed_type not in self.artist_data['embeddings']:
                    continue
                
                embed_data = self.artist_data['embeddings'][embed_type]
                if 'artists' not in embed_data or candidate_artist not in embed_data['artists']:
                    continue
                
                # Find candidate artist index
                candidate_idx = None
                for i, artist in enumerate(embed_data['artists']):
                    if artist == candidate_artist:
                        candidate_idx = i
                        break
                
                if candidate_idx is None:
                    continue
                
                candidate_embedding = embed_data['embeddings'][candidate_idx]
                
                # Compute similarity based on search type
                if search_type == 'text':
                    # Text-to-song: single query embedding vs artist descriptor
                    similarity = np.dot(query_data, candidate_embedding)
                elif search_type == 'song':
                    # Song-to-song: query artist vs candidate artist
                    query_artist = query_data  # Should be artist name
                    query_idx = None
                    for i, artist in enumerate(embed_data['artists']):
                        if artist == query_artist:
                            query_idx = i
                            break
                    
                    if query_idx is None:
                        continue
                    
                    query_embedding = embed_data['embeddings'][query_idx]
                    similarity = np.dot(query_embedding, candidate_embedding)
                else:
                    similarity = 0.0

            # Accumulate weighted similarity
            total_similarity += weight * similarity
            total_weight += weight

        # Normalize by total weight if any similarities were computed
        if total_weight > 0:
            base_similarity = total_similarity / total_weight

            # Apply gender similarity bonus if applicable
            if search_type == 'song':
                # For song-to-song search, apply gender bonus
                query_artist = query_data  # Should be artist name
                query_gender = self.get_artist_gender(query_artist)
                candidate_gender = self.get_artist_gender(candidate_artist)

                if (query_gender and candidate_gender and
                    query_gender.strip() != '' and candidate_gender.strip() != '' and
                    query_gender.lower().strip() == candidate_gender.lower().strip()):
                    # Same gender - apply multiplicative bonus
                    gender_bonus = self.ranking_engine.config.gender_similarity_bonus
                    base_similarity *= gender_bonus

            # Ensure similarity stays in [0, 1] range
            return min(base_similarity, 1.0)
        else:
            return 0.0

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
            logger.info("Initialized ranking engine without history, still computed track priors")
        
        # Artist auxiliary structures not needed for new descriptor-based system
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding for text query."""
        return data_utils.get_openai_embedding(text, normalize=True)
    

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
        track_id = song.get('track_id') or song.get('id')
        if not track_id:
            raise KeyError(f"Song at index {song_idx} missing track_id")
        
        embedding_lookup = self.embedding_lookups[embed_type]
        if track_id not in embedding_lookup:
            raise KeyError(f"No '{embed_type}' embedding for track_id: {track_id}")
        
        return embedding_lookup[track_id]
    

    def similarity_search(self, query_embedding: np.ndarray, k: int = 20, offset: int = 0, 
                         lambda_val: float = 0.5, familiarity_min: float = 0.0, familiarity_max: float = 1.0,
                         query_track_id: str = None,
                         **advanced_params) -> Tuple[List[Dict], int]:
        """
        Perform descriptor-based similarity search with new 4-component scoring system.
        
        Args:
            query_embedding: Normalized query embedding (or dict of embeddings for song-to-song)
            k: Number of results to return
            offset: Pagination offset
            lambda_val: Weight for semantic vs personal utility (0=personal, 1=semantic)
            familiarity_min: Minimum familiarity score to include (0.0-1.0)
            familiarity_max: Maximum familiarity score to include (0.0-1.0)
            query_track_id: If provided, enables song-to-song search (artist extracted automatically)
            **advanced_params: Additional parameters for ranking configuration
        
        Returns:
            Tuple of (results, total_count)
        """
        t_similarity_search_start = time.time()
        
        # Update lambda_val in config for scoring
        self.ranking_engine.config.lambda_val = lambda_val
        logger.info(f"🔧 Updated lambda_val in ranking config: {lambda_val}")
        
        # Update advanced parameters if provided
        if advanced_params:
            logger.info(f"🔧 Updating ranking config with advanced params")
            self.ranking_engine.config.update_weights(advanced_params)
        
        # Determine search type
        search_type = 'song' if query_track_id is not None else 'text'
        logger.info(f"Performing {search_type}-to-song search")
        
        # Get weight configurations
        song_weights = self.ranking_engine.config.get_song_weights()
        artist_weights = self.ranking_engine.config.get_artist_weights()
        
        # For song-to-song search, build query embeddings dict and get query artist
        query_song_embeddings = None
        query_artist = None
        if search_type == 'song' and query_track_id:
            query_song_embeddings = {}
            for embed_type in song_weights.keys():
                if embed_type in self.embedding_lookups:
                    embedding = self.embedding_lookups[embed_type].get(query_track_id)
                    if embedding is not None:
                        query_song_embeddings[embed_type] = embedding
            
            # Get query artist from song data if not provided
            if not query_artist and query_track_id:
                for song in self.songs:
                    if song.get('track_id') == query_track_id or song.get('id') == query_track_id:
                        query_artist = song.get('artists', [{}])[0].get('name', '') if song.get('artists') else ''
                        break
        
        # Compute similarity scores for all candidate songs with optimized artist similarity caching
        candidates_data = []

        t_semantic_start = time.time()

        # Phase 1: Collect unique artists from all candidate songs
        unique_artists = set()
        for song in self.songs:
            track_id = song.get('track_id') or song.get('id')
            if not track_id:
                continue
            candidate_artist = song.get('artists', [{}])[0].get('name', '') if song.get('artists') else ''
            if candidate_artist:
                unique_artists.add(candidate_artist)

        logger.info(f"Found {len(unique_artists)} unique artists among {len(self.songs)} songs")

        # Phase 2: Pre-compute artist similarities once per unique artist
        artist_similarities = {}
        t_artist_start = time.time()

        for candidate_artist in unique_artists:
            try:
                if search_type == 'text':
                    artist_similarity = self.compute_artist_descriptor_similarity(
                        query_embedding, candidate_artist, artist_weights, 'text'
                    )
                else:
                    artist_similarity = self.compute_artist_descriptor_similarity(
                        query_artist, candidate_artist, artist_weights, 'song'
                    )
                artist_similarities[candidate_artist] = artist_similarity
            except Exception as e:
                logger.warning(f"Error computing artist similarity for {candidate_artist}: {e}")
                artist_similarities[candidate_artist] = 0.0

        t_artist_end = time.time()
        logger.info(f"Computed artist similarities for {len(unique_artists)} unique artists in {t_artist_end - t_artist_start:.2f}s")

        # Phase 3: Compute 95th percentile from unique artist similarities
        unique_artist_similarities = list(artist_similarities.values())
        # Filter out NaN values before percentile calculation
        clean_similarities = [x for x in unique_artist_similarities if not np.isnan(x)]
        artist_similarity_p95 = np.percentile(clean_similarities, 95) if clean_similarities else 0.0

        if len(clean_similarities) < len(unique_artist_similarities):
            nan_count = len(unique_artist_similarities) - len(clean_similarities)
            logger.warning(f"Filtered out {nan_count} NaN artist similarities before percentile calculation")

        logger.info(f"Artist similarity 95th percentile (from {len(clean_similarities)} valid unique artists): {artist_similarity_p95:.4f}")

        # Phase 4: Process songs using cached artist similarities
        for i, song in enumerate(self.songs):
            track_id = song.get('track_id') or song.get('id')
            if not track_id:
                continue

            try:
                # Compute song descriptor similarity
                if search_type == 'text':
                    song_similarity = self.compute_song_descriptor_similarity(
                        query_embedding, track_id, song_weights, 'text'
                    )
                else:
                    song_similarity = self.compute_song_descriptor_similarity(
                        query_song_embeddings, track_id, song_weights, 'song'
                    )

                # Get artist and lookup pre-computed similarity
                candidate_artist = song.get('artists', [{}])[0].get('name', '') if song.get('artists') else ''
                artist_similarity = artist_similarities.get(candidate_artist, 0.0)

                # Store intermediate data for final score computation
                candidates_data.append({
                    'song': song,
                    'song_similarity': song_similarity,
                    'artist_similarity': artist_similarity,
                    'track_id': track_id,
                    'candidate_artist': candidate_artist,
                    'index': i
                })

            except Exception as e:
                logger.warning(f"Error processing song {track_id}: {e}")
                continue

        # Second pass: compute final scores with percentile value
        final_candidates_data = []
        for candidate in candidates_data:
            try:
                # Apply familiarity filtering (placeholder for now)
                familiarity = 1.0  # Default to familiar for no-history mode
                if familiarity < familiarity_min or familiarity > familiarity_max:
                    continue

                # Compute final score using new 4-component system with percentile
                final_score, components = self.ranking_engine.compute_v25_final_score(
                    candidate['song_similarity'], candidate['artist_similarity'],
                    candidate['track_id'], query_artist, candidate['candidate_artist'],
                    artist_similarity_p95=artist_similarity_p95
                )

                final_candidates_data.append({
                    'song': candidate['song'],
                    'song_similarity': candidate['song_similarity'],
                    'artist_similarity': candidate['artist_similarity'],
                    'final_score': final_score,
                    'components': components,
                    'index': candidate['index']
                })

            except Exception as e:
                logger.warning(f"Error computing final score for song {candidate['track_id']}: {e}")
                continue

        # Use final candidates data for sorting
        candidates_data = final_candidates_data
        
        t_semantic_end = time.time()
        logger.info(f"Computed similarities for {len(candidates_data)} candidates in {t_semantic_end - t_semantic_start:.2f}s")
        
        # Sort by final score
        candidates_data.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Apply pagination
        total_count = len(candidates_data)
        paginated_candidates = candidates_data[offset:offset + k]
        
        # Build results
        results = []
        for candidate in paginated_candidates:
            song = candidate['song']
            
            # Extract artists info
            artists_list = song.get('artists', [])
            primary_artist = artists_list[0]['name'] if artists_list else ''
            all_artists = [artist['name'] for artist in artists_list] if artists_list else []
            
            # Extract cover art from album images
            album_images = song.get('album', {}).get('images', [])
            cover_url = album_images[-1]['url'] if album_images else ''  # Use smallest image (last in list)
            
            # Get track_id for tags/genres lookup
            track_id = song.get('track_id') or song.get('id')

            result = {
                'index': candidate['index'],
                'song_idx': candidate['index'],  # For consistency with suggestions
                'song': song.get('name', ''),  # Frontend expects 'song' not 'song_name'
                'artist': primary_artist,  # Primary artist
                'artists': all_artists,  # All artists array
                'all_artists': all_artists,  # For frontend compatibility
                'album': song.get('album', {}).get('name', '') if song.get('album') else '',
                'cover_url': cover_url,  # Cover art URL
                'track_id': track_id,
                'spotify_id': track_id,  # For Spotify playback
                'uri': song.get('uri', ''),
                'popularity': song.get('popularity', 0),
                'song_similarity': candidate['song_similarity'],
                'artist_similarity': candidate['artist_similarity'],
                'final_score': candidate['final_score'],
                'components': candidate['components'],
                'scoring_components': candidate['components'],  # For frontend compatibility
                'tags': self.tags_lookup.get(track_id, []),  # Add tags from profiles
                'genres': self.genres_lookup.get(track_id, [])  # Add genres from profiles
            }
            results.append(result)
        
        t_similarity_search_end = time.time()
        logger.info(f"Similarity search completed in {t_similarity_search_end - t_similarity_search_start:.2f}s")
        
        return results, total_count
    
    def search_songs_by_text(self, query: str, limit: int = 10) -> List[Tuple[int, float, str]]:
        """Search for songs using RapidFuzz fuzzy matching (for song-to-song search suggestions)."""
        if not query or not self.fuzzy_search_index:
            return []
        
        return data_utils.search_songs_by_text(
            query, self.fuzzy_search_index, self.songs, limit
        )
    
    def _convert_numpy_to_python(self, obj):
        """
        Recursively convert numpy types to native Python types for JSON serialization.
        
        Args:
            obj: Object that may contain numpy values
            
        Returns:
            Object with numpy values converted to native Python types
        """
        if isinstance(obj, dict):
            return {key: self._convert_numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_python(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def get_ranking_weights(self) -> Dict:
        """Get current ranking weights for display."""
        config = self.ranking_engine.config
        weights = config.to_dict()
        weights.update({
            'version': '2.7',  # Updated to reflect new descriptor-based system
            'has_history': self.has_history,
            'history_songs_count': len(self.ranking_engine.track_stats) if self.ranking_engine.track_stats else 0
        })
        return weights


# Initialize search engine
search_engine = None

def init_search_engine(songs_file: str = None, embeddings_file: str = None, history_path: str = None, artist_embeddings_file: str = None, shared_genre_store_file: str = None, profiles_file: str = None):
    """Initialize the search engine with data files."""
    global search_engine
    if search_engine is None:
        # Use provided file paths or fall back to parsed arguments or defaults
        if songs_file and embeddings_file:
            songs_path = songs_file
            embeddings_path = embeddings_file
            history_path_arg = history_path
            artist_embeddings_path = artist_embeddings_file
            shared_genre_store_path = shared_genre_store_file
        elif 'args' in globals() and args:
            songs_path = songs_file or args.songs
            embeddings_path = embeddings_file or args.embeddings
            history_path_arg = history_path or args.history
            artist_embeddings_path = artist_embeddings_file or getattr(args, 'artist_embeddings', constants.DEFAULT_ARTIST_EMBEDDINGS_PATH)
            shared_genre_store_path = shared_genre_store_file or getattr(args, 'shared_genre_store', constants.DEFAULT_SHARED_GENRE_STORE_PATH)
        else:
            # Fallback defaults
            default_songs = Path(__file__).parent.parent / constants.DEFAULT_SONGS_FILE
            default_embeddings = Path(__file__).parent.parent / constants.DEFAULT_EMBEDDINGS_PATH
            default_artist_embeddings = (Path(__file__).parent.parent / constants.DEFAULT_ARTIST_EMBEDDINGS_PATH) if constants.DEFAULT_ARTIST_EMBEDDINGS_PATH else None
            default_shared_genre_store = (Path(__file__).parent.parent / constants.DEFAULT_SHARED_GENRE_STORE_PATH) if constants.DEFAULT_SHARED_GENRE_STORE_PATH else None
            songs_path = songs_file or str(default_songs)
            embeddings_path = embeddings_file or str(default_embeddings)
            history_path_arg = history_path
            artist_embeddings_path = artist_embeddings_file or (str(default_artist_embeddings) if default_artist_embeddings else None)
            shared_genre_store_path = shared_genre_store_file or (str(default_shared_genre_store) if default_shared_genre_store else None)
        
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
        
        if shared_genre_store_path and Path(shared_genre_store_path).exists():
            logger.info(f"  Shared genre store: {shared_genre_store_path}")
        else:
            logger.info(f"  Shared genre store: None (using fallback genre loading)")
            shared_genre_store_path = None  # Will use fallback loading
        
        # Create search engine
        profiles_path = profiles_file or getattr(constants, 'DEFAULT_PROFILES_FILE', None)
        search_engine = MusicSearchEngine(songs_path, embeddings_path, history_path_arg, artist_embeddings_path, shared_genre_store_path, profiles_path)


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
            'page_title': 'SongMatch',
            'is_new_session': True
        })
    
    # Pass debug flag and default config values to template  
    debug_mode = getattr(args, 'debug', False) if 'args' in globals() and args else False
    default_config = ranking.RankingConfig()
    return render_template('index.html', debug_mode=debug_mode, default_config=default_config)

@app.route('/api/search', methods=['POST'])
def search():
    """Main search endpoint with personalized ranking."""
    if search_engine is None:
        return jsonify({'error': 'Search engine not initialized'}), 500
    
    start_time = datetime.now()
    
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        search_type = data.get('search_type', data.get('type', 'text'))  # Accept both for compatibility
        # embed_type removed - using all descriptors simultaneously
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
            'C_fam', 'min_plays',
            # New 9-weight system parameters
            'a0_song_sim', 'a1_artist_sim', 'a2_total_streams', 'a3_daily_streams',
            'b0_genres', 'b1_vocal_style', 'b2_production_sound_design', 'b3_lyrical_meaning', 'b4_mood_atmosphere'
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
            
            results, total_count = search_engine.similarity_search(
                query_embedding, k=limit, offset=offset,
                lambda_val=lambda_val, familiarity_min=familiarity_min, familiarity_max=familiarity_max,
                query_text=query_text,
                **advanced_params
            )
        
        elif search_type == 'song':
            # Song-to-song search
            if song_idx is None:
                return jsonify({'error': 'song_idx is required for song search'}), 400
            
            try:
                if song_idx >= len(search_engine.songs):
                    return jsonify({'error': 'Invalid song_idx'}), 400
                
                reference_song = search_engine.songs[song_idx]
                track_id = reference_song.get('track_id') or reference_song.get('id')
                
                if not track_id:
                    return jsonify({'error': 'Reference song missing track_id'}), 400
                
                logger.info(f"Song-to-song search for track_id: {track_id} using all descriptors")
                
                # For song-to-song search, we pass any single embedding as query_embedding
                # The similarity_search method will internally build the complete descriptor dict
                query_embedding = None
                for embed_type in constants.SONG_EMBEDDING_TYPES:
                    embedding_lookup = search_engine.embedding_lookups.get(embed_type, {})
                    if embedding_lookup and track_id in embedding_lookup:
                        query_embedding = embedding_lookup[track_id]
                        logger.info(f"Found query embedding for track_id: {track_id}, using {embed_type}, shape: {query_embedding.shape}")
                        break
                
                if query_embedding is None:
                    logger.error(f"No embeddings found for track_id: {track_id}")
                    return jsonify({'error': 'No embeddings available for reference song'}), 400
                
                logger.info(f"Starting similarity search with {len(search_engine.songs)} total songs")
                results, total_count = search_engine.similarity_search(
                    query_embedding, k=limit, offset=offset,
                    lambda_val=lambda_val, familiarity_min=familiarity_min, familiarity_max=familiarity_max,
                    query_track_id=track_id,  # Enable song-to-song artist similarities
                    **advanced_params
                )
                logger.info(f"Similarity search completed, found {total_count} results")
            
            except Exception as e:
                logger.error(f"Error in song-to-song search: {str(e)}", exc_info=True)
                return jsonify({'error': f'Song search failed: {str(e)}'}), 500
        
        else:
            return jsonify({'error': 'Invalid search type'}), 400
        
        # Calculate search performance
        search_duration = (datetime.now() - start_time).total_seconds()
        
        # Track successful search with enhanced context
        search_properties = {
            'search_type': search_type,
            'descriptors': 'all',  # Using all descriptors simultaneously
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
            'descriptors': 'all',  # Using all descriptors simultaneously
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
            'descriptors': 'all',  # Using all descriptors simultaneously
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
    # Local development: host is localhost or 127.0.0.1 (any port)
    host_without_port = request.host.split(':')[0]
    is_local = host_without_port in ['127.0.0.1', 'localhost']
    is_production = os.getenv('RAILWAY_ENVIRONMENT') or not is_local
    
    if is_production:
        # Production: use the current request host with HTTPS
        redirect_uri = f"https://{request.host}/callback"
    else:
        # Local development: use current host with HTTP (preserves custom port)
        redirect_uri = f"http://{request.host}/callback"
    
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
        
        # Extract data from new Spotify metadata structure
        track_id = song.get('track_id') or song.get('id', '')
        song_name = song.get('name', song.get('original_song', ''))
        
        # Handle multiple artists
        artists_list = song.get('artists', [])
        primary_artist = artists_list[0]['name'] if artists_list else song.get('original_artist', '')
        
        # Get album info and cover art
        album_name = song.get('album', {}).get('name', '')
        album_images = song.get('album', {}).get('images', [])
        cover_url = album_images[-1]['url'] if album_images else ''
        
        suggestions.append({
            'song_idx': int(song_idx),  # Convert numpy.int64 to native Python int
            'label': label,
            'song': song_name,
            'artist': primary_artist,
            'album': album_name,
            'cover_url': cover_url,
            'spotify_id': track_id,
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
        playlist_name = data.get('playlist_name', 'SongMatch Playlist')
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
            description=f"Created by SongMatch - {len(valid_songs)} songs"
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
    
    # Extract data from new Spotify metadata structure
    track_id = song.get('track_id') or song.get('id', '')
    song_name = song.get('name', song.get('original_song', ''))
    
    # Handle multiple artists
    artists_list = song.get('artists', [])
    primary_artist = artists_list[0]['name'] if artists_list else song.get('original_artist', '')
    all_artists = [artist['name'] for artist in artists_list] if artists_list else [primary_artist]
    
    # Get album info and cover art
    album_name = song.get('album', {}).get('name', '')
    album_images = song.get('album', {}).get('images', [])
    cover_url = album_images[-1]['url'] if album_images else ''
    
    # Get duration
    duration_ms = song.get('duration_ms', 0)
    
    return jsonify({
        'song_idx': song_idx,
        'song': song_name,
        'artist': primary_artist,
        'all_artists': all_artists,  # New: support multiple artists
        'album': album_name,
        'cover_url': cover_url,
        'spotify_id': track_id,
        'duration_ms': duration_ms,  # New: duration in milliseconds
        'genres': search_engine.genres_lookup.get(track_id, []),
        'tags': search_engine.tags_lookup.get(track_id, []),
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
    parser.add_argument('--shared-genre-store', type=str, default=constants.DEFAULT_SHARED_GENRE_STORE_PATH, help=f'Path to shared genre embedding store file (default: {constants.DEFAULT_SHARED_GENRE_STORE_PATH})')
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
        init_search_engine(args.songs, args.embeddings, args.history, getattr(args, 'artist_embeddings', None), getattr(args, 'shared_genre_store', None))
        logger.info("Search engine initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        logger.error("Please check your data file paths and try again.")
        exit(1)
    
    # Start the Flask application
    app.run(debug=args.debug, host=args.host, port=args.port)