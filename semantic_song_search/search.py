
"""
Search engine implementation with descriptor-based similarity search.
"""

import logging
import numpy as np
import json
import os
import time
import math
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    from . import data_utils, constants
except ImportError:
    import data_utils, constants

logger = logging.getLogger(__name__)


class SearchConfig:
    """Configuration class for search engine hyperparameters."""

    def __init__(self,
        # Top-level component weights (a_i) - should sum to 1.0 - USER CONFIGURABLE
        a0_song_sim: float,        # Weight for song descriptor similarity
        a1_artist_sim: float,      # Weight for artist descriptor similarity
        a2_total_streams: float,   # Weight for total streams score
        a3_daily_streams: float,   # Weight for daily streams score
        a4_release_date: float,    # Weight for release date similarity score

        # Song descriptor weights (b_i) - should sum to 1.0 - USER CONFIGURABLE
        b0_genres: float,                   # Weight for song genres similarity
        b1_vocal_style: float,             # Weight for vocal style similarity
        b2_production_sound_design: float, # Weight for production & sound design similarity
        b3_lyrical_meaning: float,         # Weight for lyrical meaning similarity
        b4_mood_atmosphere: float,         # Weight for mood & atmosphere similarity
        b5_tags: float,                    # Weight for tags similarity

        # Core semantic vs utility balance - OPTIONAL (has default)
        lambda_val: float = 0.5,    # Relevance vs utility balance (λ in formula)

        # Familiarity filtering - OPTIONAL (has defaults)
        familiarity_min: float = 0.0,  # Minimum familiarity threshold
        familiarity_max: float = 1.0,  # Maximum familiarity threshold

        # Artist descriptor weights (c_i) - should sum to 1.0 - INTERNAL (has defaults)
        c0_artist_genres: float = 0.375,                     # Weight for artist genres similarity
        c1_artist_vocal_style: float = 0.15,               # Weight for artist vocal style similarity
        c2_artist_production_sound_design: float = 0.15,   # Weight for artist production similarity
        c3_artist_lyrical_themes: float = 0.075,            # Weight for artist lyrical themes similarity
        c4_artist_mood_atmosphere: float = 0.125,           # Weight for artist mood similarity
        c5_artist_cultural_context_scene: float = 0.125,    # Weight for artist cultural context similarity

        # Stream-based popularity priors - INTERNAL (has defaults)
        K_total: float = 1e7,           # Total streams normalization constant
        K_daily: float = 1e4,           # Daily streams normalization constant

        # Artist gender similarity bonus - INTERNAL (has defaults)
        gender_similarity_bonus: float = 1.05,              # Multiplicative bonus for same-gender artists (1.0 = no bonus)

        # Release date similarity configuration - INTERNAL (has defaults)
        release_date_decay_constant: float = 10950,        # Decay constant for release date similarity (days)
        ):
            """Initialize search configuration with provided values."""
            self.lambda_val = lambda_val
            self.familiarity_min = familiarity_min
            self.familiarity_max = familiarity_max

            # Top-level weights
            self.a0_song_sim = a0_song_sim
            self.a1_artist_sim = a1_artist_sim
            self.a2_total_streams = a2_total_streams
            self.a3_daily_streams = a3_daily_streams
            self.a4_release_date = a4_release_date

            # Song descriptor weights
            self.b0_genres = b0_genres
            self.b1_vocal_style = b1_vocal_style
            self.b2_production_sound_design = b2_production_sound_design
            self.b3_lyrical_meaning = b3_lyrical_meaning
            self.b4_mood_atmosphere = b4_mood_atmosphere
            self.b5_tags = b5_tags

            # Artist descriptor weights
            self.c0_artist_genres = c0_artist_genres
            self.c1_artist_vocal_style = c1_artist_vocal_style
            self.c2_artist_production_sound_design = c2_artist_production_sound_design
            self.c3_artist_lyrical_themes = c3_artist_lyrical_themes
            self.c4_artist_mood_atmosphere = c4_artist_mood_atmosphere
            self.c5_artist_cultural_context_scene = c5_artist_cultural_context_scene

            # Stream priors
            self.K_total = K_total
            self.K_daily = K_daily

            # Artist gender bonus
            self.gender_similarity_bonus = gender_similarity_bonus

            # Release date similarity configuration
            self.release_date_decay_constant = release_date_decay_constant

    def get_song_weights(self) -> Dict[str, float]:
        """Get song descriptor weights as a dictionary mapping descriptor types to weights."""
        return {
            constants.SONG_EMBEDDING_TYPES[0]: self.b0_genres,  # 'genres'
            constants.SONG_EMBEDDING_TYPES[1]: self.b1_vocal_style,  # 'vocal_style'
            constants.SONG_EMBEDDING_TYPES[2]: self.b2_production_sound_design,  # 'production_sound_design'
            constants.SONG_EMBEDDING_TYPES[3]: self.b3_lyrical_meaning,  # 'lyrical_meaning'
            constants.SONG_EMBEDDING_TYPES[4]: self.b4_mood_atmosphere,  # 'mood_atmosphere'
            constants.SONG_EMBEDDING_TYPES[5]: self.b5_tags,  # 'tags'
        }

    def get_artist_weights(self) -> Dict[str, float]:
        """Get artist descriptor weights as a dictionary mapping descriptor types to weights."""
        return {
            constants.ARTIST_EMBEDDING_TYPES[0]: self.c0_artist_genres,  # 'genres'
            constants.ARTIST_EMBEDDING_TYPES[1]: self.c1_artist_vocal_style,  # 'vocal_style'
            constants.ARTIST_EMBEDDING_TYPES[2]: self.c2_artist_production_sound_design,  # 'production_sound_design'
            constants.ARTIST_EMBEDDING_TYPES[3]: self.c3_artist_lyrical_themes,  # 'lyrical_themes'
            constants.ARTIST_EMBEDDING_TYPES[4]: self.c4_artist_mood_atmosphere,  # 'mood_atmosphere'
            constants.ARTIST_EMBEDDING_TYPES[5]: self.c5_artist_cultural_context_scene,  # 'cultural_context_scene'
        }

    def to_dict(self) -> Dict:
        """Convert config to dictionary format."""
        return {
            'lambda': self.lambda_val,
            'familiarity_min': self.familiarity_min,
            'familiarity_max': self.familiarity_max,
            'a0_song_sim': self.a0_song_sim,
            'a1_artist_sim': self.a1_artist_sim,
            'a2_total_streams': self.a2_total_streams,
            'a3_daily_streams': self.a3_daily_streams,
            'a4_release_date': self.a4_release_date,
            'b0_genres': self.b0_genres,
            'b1_vocal_style': self.b1_vocal_style,
            'b2_production_sound_design': self.b2_production_sound_design,
            'b3_lyrical_meaning': self.b3_lyrical_meaning,
            'b4_mood_atmosphere': self.b4_mood_atmosphere,
            'b5_tags': self.b5_tags,
            'c0_artist_genres': self.c0_artist_genres,
            'c1_artist_vocal_style': self.c1_artist_vocal_style,
            'c2_artist_production_sound_design': self.c2_artist_production_sound_design,
            'c3_artist_lyrical_themes': self.c3_artist_lyrical_themes,
            'c4_artist_mood_atmosphere': self.c4_artist_mood_atmosphere,
            'c5_artist_cultural_context_scene': self.c5_artist_cultural_context_scene,
            'K_total': self.K_total,
            'K_daily': self.K_daily,
            'gender_similarity_bonus': self.gender_similarity_bonus,
            'release_date_decay_constant': self.release_date_decay_constant,
        }

    def update_weights(self, weights: Dict[str, float]):
        """Update weights from dictionary with validation."""
        for key, value in sorted(weights.items()):
            try:
                float_value = float(value)

                if key == 'lambda':  # Handle the special case
                    if 0.0 <= float_value <= 1.0:
                        self.lambda_val = float_value
                    else:
                        logger.warning(f"Lambda must be in [0,1], got {float_value}")
                elif hasattr(self, key):
                    # Additional validation for specific parameters
                    if key.startswith(('a0', 'a1', 'a2', 'a3', 'a4', 'b', 'c')) and not 0.0 <= float_value <= 1.0:
                        logger.warning(f"{key} should typically be in [0,1], got {float_value}")
                    elif key in ['K_total', 'K_daily', 'release_date_decay_constant'] and float_value < 0.0:
                        logger.warning(f"{key} should be non-negative, got {float_value}")
                    setattr(self, key, float_value)
                else:
                    logger.warning(f"Unknown search parameter: {key}")
            except (TypeError, ValueError):
                logger.warning(f"Invalid value for {key}: {value}")


class MusicSearchEngine:
    """Music search engine with descriptor-based similarity search."""

    def __init__(self, songs_file: str, embeddings_file: str, artist_embeddings_file: str = None,
                 shared_genre_store_path: str = None, profiles_file: str = None, config: SearchConfig = None):
        self.songs_file = songs_file
        self.embeddings_file = embeddings_file
        self.artist_embeddings_file = artist_embeddings_file
        self.shared_genre_store_path = shared_genre_store_path
        self.profiles_file = profiles_file

        # Search configuration (optional - can be set later)
        self.config = config

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

        # Descriptor-based artist data
        self.artist_data = None  # Will contain embeddings, metadata, and genre store

        # Track streams for popularity scoring
        self.track_streams = {}  # track_id -> {S_total, S_daily}

        # Load all data
        self._load_all_data()
    
    def _load_all_data(self):
        """Load songs, embeddings, and artist data."""
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

        # Compute track streams for popularity scoring using default normalization constants
        self._compute_track_streams()

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
                    gender_bonus = self.config.gender_similarity_bonus
                    base_similarity *= gender_bonus

            # Ensure similarity stays in [0, 1] range
            return min(base_similarity, 1.0)
        else:
            return 0.0

    def compute_release_date_similarity(self, query_song: Dict, candidate_song: Dict) -> Optional[float]:
        """
        Compute release date similarity using exponential decay function.

        Args:
            query_song: Query song metadata dictionary
            candidate_song: Candidate song metadata dictionary

        Returns:
            Release date similarity score (0.0 to 1.0), or None if computation failed
        """
        try:
            from . import data_utils
        except ImportError:
            import data_utils

        # Extract release date information from both songs
        query_album = query_song.get('album', {})
        candidate_album = candidate_song.get('album', {})

        query_release_date = query_album.get('release_date')
        query_precision = query_album.get('release_date_precision')
        candidate_release_date = candidate_album.get('release_date')
        candidate_precision = candidate_album.get('release_date_precision')

        # Parse release dates
        query_date = data_utils.parse_release_date(query_release_date, query_precision)
        candidate_date = data_utils.parse_release_date(candidate_release_date, candidate_precision)

        # If either date parsing failed, return None to indicate computation failure
        if query_date is None or candidate_date is None:
            return None

        # Calculate days difference
        days_diff = abs((candidate_date - query_date).days)

        # Apply exponential decay: score = exp(-|days_diff| / decay_constant)
        similarity_score = math.exp(-days_diff / self.config.release_date_decay_constant)

        return float(similarity_score)

    def _compute_track_streams(self):
        """Compute track stream scores for popularity scoring using default normalization constants."""
        # Use default normalization constants (these are internal and rarely need to change)
        K_total_default = 1e7
        K_daily_default = 1e4

        track_streams = {}

        for song in self.songs:
            track_id = song.get('track_id') or song.get('id')
            if not track_id:
                continue

            # Compute stream-based scores
            streams_data = song.get('streams', {})
            streams_total = streams_data.get('streams_total', 0)
            S_total = streams_total / (streams_total + K_total_default)

            streams_daily = streams_data.get('streams_daily', 0)
            S_daily = streams_daily / (streams_daily + K_daily_default)

            track_streams[track_id] = {
                'S_total': S_total,
                'S_daily': S_daily,
            }

        self.track_streams = track_streams
        logger.info(f"Computed track streams for {len(track_streams)} tracks")

    def compute_final_score(self, song_similarity: float, artist_similarity: float, track_id: str,
                           query_artist: str = None, candidate_artist: str = None,
                           artist_similarity_p95: float = None, ranking_engine=None,
                           release_date_similarity: float = None, query_song: Dict = None,
                           candidate_song: Dict = None, search_type: str = 'text') -> Tuple[float, Dict]:
        """
        Compute final score using descriptor-based system.

        Args:
            song_similarity: Weighted song descriptor similarity [0, 1]
            artist_similarity: Weighted artist descriptor similarity [0, 1]
            track_id: Spotify track ID
            query_artist: Query artist name (for same-artist handling in song-to-song search)
            candidate_artist: Candidate artist name (for same-artist handling in song-to-song search)
            artist_similarity_p95: 95th percentile of all artist similarities for this query
            ranking_engine: Optional ranking engine for personalization
            release_date_similarity: Precomputed release date similarity (or None for auto-computation)
            query_song: Query song metadata (for release date similarity computation)
            candidate_song: Candidate song metadata (for release date similarity computation)
            search_type: 'text' for text-to-song, 'song' for song-to-song

        Returns:
            Tuple of (final_score, component_breakdown)
        """
        # Get the 5 component scores
        # Handle NaN values by replacing with zero
        S_song = 0.0 if np.isnan(song_similarity) else song_similarity
        S_artist = 0.0 if np.isnan(artist_similarity) else artist_similarity
        S_streams_total = self.track_streams.get(track_id, {}).get('S_total', 0.0)
        S_streams_daily = self.track_streams.get(track_id, {}).get('S_daily', 0.0)

        # Compute release date similarity
        S_release_date = 0.0
        use_release_date_component = False

        if search_type == 'song':
            # Only use release date similarity for song-to-song search
            if release_date_similarity is not None:
                S_release_date = release_date_similarity
                use_release_date_component = True
            elif query_song is not None and candidate_song is not None:
                try:
                    S_release_date_result = self.compute_release_date_similarity(query_song, candidate_song)
                    if S_release_date_result is not None:
                        S_release_date = S_release_date_result
                        use_release_date_component = True
                    else:
                        # Date parsing failed, exclude release date component
                        S_release_date = 0.0
                        use_release_date_component = False
                except Exception as e:
                    logger.warning(f"Failed to compute release date similarity for track {track_id}: {e}")
                    S_release_date = 0.0
                    use_release_date_component = False

        # Check for same-artist candidates in song-to-song search
        is_same_artist = (query_artist is not None and
                         candidate_artist is not None and
                         query_artist.strip() != '' and
                         candidate_artist.strip() != '' and
                         query_artist.lower().strip() == candidate_artist.lower().strip())

        if is_same_artist and artist_similarity_p95 is not None:
            # For same-artist candidates, use 95th percentile as placeholder
            # Handle NaN values in percentile (fallback to 0.0)
            S_artist_score = 0.0 if np.isnan(artist_similarity_p95) else artist_similarity_p95
            S_artist_display = S_artist_score  # Display the value actually used for scoring

            if np.isnan(artist_similarity_p95):
                logger.warning(f"Artist similarity 95th percentile is NaN, using 0.0 for same-artist candidate")
            else:
                logger.debug(f"Same-artist candidate: using 95th percentile {artist_similarity_p95:.4f} instead of {S_artist:.4f}")
        else:
            # Use actual artist similarity
            S_artist_score = S_artist
            S_artist_display = S_artist

        # Dynamic weight adjustment based on whether release date component is used
        if use_release_date_component:
            # Use all 5 components including release date
            total_weight = (self.config.a0_song_sim + self.config.a1_artist_sim +
                           self.config.a2_total_streams + self.config.a3_daily_streams +
                           self.config.a4_release_date)

            if total_weight > 1e-8:  # Use small epsilon to handle floating point precision
                # Normalize weights to ensure they sum to 1
                norm_a0 = self.config.a0_song_sim / total_weight
                norm_a1 = self.config.a1_artist_sim / total_weight
                norm_a2 = self.config.a2_total_streams / total_weight
                norm_a3 = self.config.a3_daily_streams / total_weight
                norm_a4 = self.config.a4_release_date / total_weight

                S_semantic = (norm_a0 * S_song +
                             norm_a1 * S_artist_score +
                             norm_a2 * S_streams_total +
                             norm_a3 * S_streams_daily +
                             norm_a4 * S_release_date)

                logger.debug(f"Using 5-component scoring with release date similarity: {S_release_date:.4f}")
            else:
                # Fallback to song similarity only if all weights are zero
                logger.warning("All similarity weights are zero, falling back to song similarity only")
                S_semantic = S_song
        else:
            # Exclude release date component and renormalize remaining 4 weights
            total_weight = (self.config.a0_song_sim + self.config.a1_artist_sim +
                           self.config.a2_total_streams + self.config.a3_daily_streams)

            if total_weight > 1e-8:  # Use small epsilon to handle floating point precision
                # Normalize weights to ensure they sum to 1
                norm_a0 = self.config.a0_song_sim / total_weight
                norm_a1 = self.config.a1_artist_sim / total_weight
                norm_a2 = self.config.a2_total_streams / total_weight
                norm_a3 = self.config.a3_daily_streams / total_weight

                S_semantic = (norm_a0 * S_song +
                             norm_a1 * S_artist_score +
                             norm_a2 * S_streams_total +
                             norm_a3 * S_streams_daily)

                logger.debug(f"Using 4-component scoring (release date excluded)")
            else:
                # Fallback to song similarity only if all weights are zero
                logger.warning("All similarity weights are zero, falling back to song similarity only")
                S_semantic = S_song

        S_semantic = np.clip(S_semantic, 0, 1)

        # Create basic components dictionary
        components = {
            'semantic_similarity': S_semantic,
            'S_song': S_song,
            'S_artist': S_artist_display,
            'S_streams_total': S_streams_total,
            'S_streams_daily': S_streams_daily,
            'S_release_date': S_release_date,
            'lambda': self.config.lambda_val,
            'a0_song_sim': self.config.a0_song_sim,
            'a1_artist_sim': self.config.a1_artist_sim,
            'a2_total_streams': self.config.a2_total_streams,
            'a3_daily_streams': self.config.a3_daily_streams,
            'a4_release_date': self.config.a4_release_date,
            # Song descriptor weights (for debugging)
            'b0_genres': self.config.b0_genres,
            'b1_vocal_style': self.config.b1_vocal_style,
            'b2_production_sound_design': self.config.b2_production_sound_design,
            'b3_lyrical_meaning': self.config.b3_lyrical_meaning,
            'b4_mood_atmosphere': self.config.b4_mood_atmosphere,
            'b5_tags': self.config.b5_tags,
            # Additional debug info
            'is_same_artist': is_same_artist,
            'use_release_date_component': use_release_date_component,
            'search_type': search_type,
        }

        # No-history case: return pure semantic similarity
        if ranking_engine is None or not getattr(ranking_engine, 'has_history', False):
            components['final_score'] = S_semantic
            return S_semantic, components

        # History case: add personalization via ranking engine
        return ranking_engine.add_personalization(
            S_semantic, S_streams_total, S_streams_daily, components, track_id
        )
    
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
            available_types = sorted(list(self.embedding_lookups.keys()))
            raise ValueError(f"Embedding type '{embed_type}' not available. Available types: {available_types}")
        
        song = self.songs[song_idx]
        track_id = song.get('track_id') or song.get('id')
        if not track_id:
            raise KeyError(f"Song at index {song_idx} missing track_id")
        
        embedding_lookup = self.embedding_lookups[embed_type]
        if track_id not in embedding_lookup:
            raise KeyError(f"No '{embed_type}' embedding for track_id: {track_id}")
        
        return embedding_lookup[track_id]
    

    def similarity_search(self, query_embedding: np.ndarray, k: int, offset: int = 0,
                         query_track_id: str = None, ranking_engine=None,
                         **advanced_params) -> Tuple[List[Dict], int]:
        """
        Perform descriptor-based similarity search with new 4-component scoring system.
        
        Args:
            query_embedding: Normalized query embedding (or dict of embeddings for song-to-song)
            k: Number of results to return
            offset: Pagination offset
            query_track_id: If provided, enables song-to-song search (artist extracted automatically)
            ranking_engine: Optional ranking engine for personalization
            **advanced_params: Additional parameters including lambda_val, familiarity bounds, and other weights
        
        Returns:
            Tuple of (results, total_count)
        """
        t_similarity_search_start = time.time()
        
        # Create search config from advanced parameters
        if not advanced_params:
            raise ValueError("Advanced parameters are required and must include all weight values")

        logger.info(f"🔧 Creating search config from advanced params")
        self.config = SearchConfig(**advanced_params)

        # Determine search type
        search_type = 'song' if query_track_id is not None else 'text'
        logger.info(f"Performing {search_type}-to-song search")

        # Get weight configurations
        song_weights = self.config.get_song_weights()
        artist_weights = self.config.get_artist_weights()
        
        # For song-to-song search, build query embeddings dict and get query artist and song
        query_song_embeddings = None
        query_artist = None
        query_song = None
        if search_type == 'song' and query_track_id:
            query_song_embeddings = {}
            for embed_type in sorted(song_weights.keys()):
                if embed_type in self.embedding_lookups:
                    embedding = self.embedding_lookups[embed_type].get(query_track_id)
                    if embedding is not None:
                        query_song_embeddings[embed_type] = embedding

            # Get query artist and song from song data if not provided
            if not query_artist and query_track_id:
                for song in self.songs:
                    if song.get('track_id') == query_track_id or song.get('id') == query_track_id:
                        query_artist = song.get('artists', [{}])[0].get('name', '') if song.get('artists') else ''
                        query_song = song  # Store the full song object for release date similarity
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

        for candidate_artist in sorted(unique_artists):
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
        unique_artist_similarities = sorted(artist_similarities.values())
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
                if familiarity < self.config.familiarity_min or familiarity > self.config.familiarity_max:
                    continue

                # Compute final score using new 5-component system with percentile
                final_score, components = self.compute_final_score(
                    candidate['song_similarity'], candidate['artist_similarity'],
                    candidate['track_id'], query_artist, candidate['candidate_artist'],
                    artist_similarity_p95=artist_similarity_p95, ranking_engine=ranking_engine,
                    query_song=query_song, candidate_song=candidate['song'], search_type=search_type
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
        
        # Sort by final score with stable tie-breaking using index
        candidates_data.sort(key=lambda x: (-x['final_score'], x['index']))
        
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
            return {key: self._convert_numpy_to_python(value) for key, value in sorted(obj.items())}
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

