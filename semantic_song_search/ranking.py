"""
Ranking Algorithm Implementation

This module implements the personalized ranking algorithm that combines
semantic relevance with predicted utility using embeddings and listening history.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RankingConfig:
    """Configuration class for ranking algorithm hyperparameters."""
    
    def __init__(self):
        """Initialize with default hyperparameters from design doc."""
        # Core V2 parameters
        self.H_c = 30.0          # Affinity half-life in days
        self.K = 5.0             # Confidence scale parameter
        self.beta_p = 0.4        # Prior weight for popularity
        self.beta_s = 0.4        # Prior weight for semantic similarity to top tracks
        self.beta_a = 0.2        # Prior weight for artist affinity
        self.lambda_val = 0.5    # Relevance vs utility balance (λ in formula)
        
        # Discovery slider (user-controlled, default neutral)
        self.d = 0.5             # Discovery slider 0=familiar, 1=new
        
        # Reference track selection
        self.top_m_tracks = 100  # Number of top tracks for reference centroid
        self.min_confidence_threshold = 0.3  # Minimum w_hist for reference tracks
        self.min_reference_tracks = 10  # Minimum reference tracks needed
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary format."""
        return {
            'H_c': self.H_c,
            'K': self.K,
            'beta_p': self.beta_p,
            'beta_s': self.beta_s,
            'beta_a': self.beta_a,
            'lambda': self.lambda_val,
            'd': self.d,
            'top_m_tracks': self.top_m_tracks,
            'min_confidence_threshold': self.min_confidence_threshold,
            'min_reference_tracks': self.min_reference_tracks
        }
    
    def update_weights(self, weights: Dict[str, float]):
        """Update weights from dictionary with validation."""
        for key, value in weights.items():
            try:
                float_value = float(value)
                
                if key == 'lambda':  # Handle the special case
                    if 0.0 <= float_value <= 1.0:
                        self.lambda_val = float_value
                    else:
                        logger.warning(f"Lambda must be in [0,1], got {float_value}")
                elif key == 'd':  # Discovery slider
                    if 0.0 <= float_value <= 1.0:
                        self.d = float_value
                    else:
                        logger.warning(f"Discovery slider must be in [0,1], got {float_value}")
                elif hasattr(self, key):
                    # Additional validation for other parameters
                    if key in ['beta_p', 'beta_s', 'beta_a'] and not 0.0 <= float_value <= 1.0:
                        logger.warning(f"{key} should typically be in [0,1], got {float_value}")
                    setattr(self, key, float_value)
                else:
                    logger.warning(f"Unknown weight parameter: {key}")
            except (TypeError, ValueError):
                logger.warning(f"Invalid value for {key}: {value}")


class RankingEngine:
    """Ranking engine that combines semantic similarity with predicted utility."""
    
    def __init__(self, config: RankingConfig = None):
        """Initialize the ranking engine."""
        self.config = config or RankingConfig()
        
        # Computed user profile data
        self.track_stats = {}           # Per-track statistics
        self.artist_affinities = {}     # Per-artist affinities  
        self.reference_centroid = None  # Centroid of top-affinity tracks
        self.has_history = False
    
    def compute_track_statistics(self, history_df: pd.DataFrame, songs_metadata: List[Dict]) -> Dict:
        """
        Compute per-track statistics from listening history.
        
        Args:
            history_df: DataFrame with cleaned listening history
            songs_metadata: List of song metadata dicts for duration lookup
            
        Returns:
            Dictionary with per-track statistics
        """
        if history_df.empty:
            logger.warning("Empty history DataFrame provided")
            return {}
        
        logger.info(f"Computing track statistics for {len(history_df)} history entries...")
        
        # Create duration lookup
        duration_lookup = {}
        for song in songs_metadata:
            song_key = (song['original_song'], song['original_artist'])
            duration_ms = song.get('metadata', {}).get('duration', 180) * 1000  # Default 3 minutes
            duration_lookup[song_key] = duration_ms
        
        # Work with a copy
        df = history_df.copy()
        
        # Calculate completion ratios
        def calculate_completion(row):
            song_key = (row['original_song'], row['original_artist'])
            duration_ms = duration_lookup.get(song_key, 180000)  # Default 3 minutes
            return min(1.0, row['ms_played'] / max(30000, duration_ms))  # Minimum 30 seconds
        
        df['completion'] = df.apply(calculate_completion, axis=1)
        
        # Calculate recency weights
        now_utc = datetime.now(timezone.utc)
        # Ensure timestamps are timezone-aware
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        days_ago = (now_utc - df['timestamp']).dt.total_seconds() / (24 * 3600)
        # Add numerical stability to prevent underflow
        df['recency_weight'] = np.maximum(np.exp(-np.log(2) * days_ago / self.config.H_c), 1e-10)
        
        # Group by song key
        song_key_col = df[['original_song', 'original_artist']].apply(tuple, axis=1)
        df['song_key'] = song_key_col
        
        # Aggregate per track
        track_stats = {}
        
        for song_key, group in df.groupby('song_key'):
            # Recency-weighted play count (n_t)
            n_t = group['recency_weight'].sum()
            
            # Affinity estimate (A_t) - weighted average completion
            if n_t > 0:
                A_t = (group['recency_weight'] * group['completion']).sum() / n_t
            else:
                A_t = 0.0
            
            # Confidence gate (g_t)
            g_t = n_t / (n_t + self.config.K)
            
            track_stats[song_key] = {
                'n_t': float(n_t),
                'A_t': float(np.clip(A_t, 0, 1)),
                'g_t': float(np.clip(g_t, 0, 1)),
                'play_count': len(group),
                'last_play': group['timestamp'].max()
            }
        
        logger.info(f"Computed statistics for {len(track_stats)} unique tracks")
        self.track_stats = track_stats
        self.has_history = True
        return track_stats
    
    def compute_artist_affinities(self) -> Dict[str, float]:
        """
        Compute per-artist affinities (B_a) from track statistics.
        
        Returns:
            Dictionary mapping artist names to affinity scores
        """
        if not self.track_stats:
            logger.warning("No track statistics available for artist affinity computation")
            return {}
        
        artist_data = {}
        
        # Aggregate by artist
        for (_, artist), stats in self.track_stats.items():
            if artist not in artist_data:
                artist_data[artist] = {'weighted_affinity': 0.0, 'total_weight': 0.0}
            
            n_t = stats['n_t']
            A_t = stats['A_t']
            
            artist_data[artist]['weighted_affinity'] += n_t * A_t
            artist_data[artist]['total_weight'] += n_t
        
        # Compute artist affinities (B_a)
        artist_affinities = {}
        for artist, data in artist_data.items():
            if data['total_weight'] > 0:
                B_a = data['weighted_affinity'] / data['total_weight']
                artist_affinities[artist] = float(np.clip(B_a, 0, 1))
            else:
                artist_affinities[artist] = 0.0
        
        logger.info(f"Computed artist affinities for {len(artist_affinities)} artists")
        self.artist_affinities = artist_affinities
        return artist_affinities
    
    def compute_reference_centroid(self, embeddings_data: Dict) -> np.ndarray:
        """
        Compute reference centroid (μ_top) from top-affinity tracks.
        
        Args:
            embeddings_data: Dictionary with track embeddings
            
        Returns:
            Normalized reference centroid or None if insufficient data
        """
        if not self.track_stats:
            logger.warning("No track statistics for reference centroid computation")
            return None
        
        # Compute confidence-weighted affinity (w_hist)
        reference_candidates = []
        
        for song_key, stats in self.track_stats.items():
            g_t = stats['g_t']
            A_t = stats['A_t']
            w_hist = g_t * A_t
            
            # Check if we have embeddings for this track
            if song_key in embeddings_data:
                reference_candidates.append({
                    'song_key': song_key,
                    'w_hist': w_hist,
                    'embedding': embeddings_data[song_key]
                })
        
        if not reference_candidates:
            logger.warning("No reference candidates with embeddings found")
            return None
        
        # Sort by w_hist and select top candidates
        reference_candidates.sort(key=lambda x: x['w_hist'], reverse=True)
        
        # Try top-M approach first
        top_m = reference_candidates[:self.config.top_m_tracks]
        
        # Filter by minimum confidence threshold if needed
        filtered_refs = [r for r in top_m if r['w_hist'] >= self.config.min_confidence_threshold]
        
        # Use the better option
        if len(filtered_refs) >= self.config.min_reference_tracks:
            selected_refs = filtered_refs
        elif len(top_m) >= self.config.min_reference_tracks:
            selected_refs = top_m
        else:
            logger.warning(f"Only {len(reference_candidates)} reference tracks available, using all")
            selected_refs = reference_candidates
        
        if len(selected_refs) < 3:  # Too few for meaningful centroid
            logger.warning("Too few reference tracks for meaningful centroid")
            return None
        
        # Compute weighted centroid
        embeddings = np.array([r['embedding'] for r in selected_refs])
        weights = np.array([r['w_hist'] for r in selected_refs])
        
        if np.sum(weights) < 1e-6:
            logger.warning("Reference weights too small")
            return None
        
        # Weighted average
        centroid = np.average(embeddings, axis=0, weights=weights)
        
        # Normalize
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-6:
            logger.warning("Reference centroid has zero norm")
            return None
        
        centroid = centroid / centroid_norm
        
        logger.info(f"Computed reference centroid from {len(selected_refs)} tracks")
        self.reference_centroid = centroid
        return centroid
    
    def compute_prior_utility(self, song_key: Tuple[str, str], song_metadata: Dict, 
                            song_embedding: np.ndarray) -> float:
        """
        Compute prior utility (F_t) for a track.
        
        Args:
            song_key: (song, artist) tuple
            song_metadata: Song metadata dict
            song_embedding: Normalized track embedding
            
        Returns:
            Prior utility score [0, 1]
        """
        # P_t: Popularity component
        metadata = song_metadata.get('metadata', {})
        popularity = metadata.get('popularity', 50)
        try:
            P_t = float(popularity) / 100.0
            P_t = np.clip(P_t, 0.0, 1.0)  # Ensure valid range
        except (TypeError, ValueError):
            logger.warning(f"Invalid popularity value: {popularity}, using default")
            P_t = 0.5
        
        # C_t: Cosine similarity to reference centroid
        # Ensure embeddings are normalized
        if np.abs(np.linalg.norm(song_embedding) - 1.0) > 1e-6:
            logger.warning(f"Song embedding not normalized: {np.linalg.norm(song_embedding)}")
            song_embedding = song_embedding / np.linalg.norm(song_embedding)
        
        if self.reference_centroid is not None:
            if np.abs(np.linalg.norm(self.reference_centroid) - 1.0) > 1e-6:
                logger.warning("Reference centroid not normalized")
            # Compute cosine similarity
            cos_sim = np.dot(song_embedding, self.reference_centroid)
            # Rescale from [-1, 1] to [0, 1]
            C_t = (cos_sim + 1) / 2
        else:
            # Fallback: uninformative mid-prior
            C_t = 0.5
        
        # B_t: Artist affinity
        _, artist = song_key
        B_t = self.artist_affinities.get(artist, 0.0)
        
        # Combine components
        F_t = (self.config.beta_p * P_t + 
               self.config.beta_s * C_t + 
               self.config.beta_a * B_t)
        
        return float(np.clip(F_t, 0, 1))
    
    def compute_predicted_utility(self, song_key: Tuple[str, str], song_metadata: Dict,
                                song_embedding: np.ndarray) -> float:
        """
        Compute predicted utility U_t(d) for a track.
        
        Args:
            song_key: (song, artist) tuple  
            song_metadata: Song metadata dict
            song_embedding: Normalized track embedding
            
        Returns:
            Predicted utility score [0, 1]
        """
        d = self.config.d  # Discovery slider
        
        if song_key in self.track_stats:
            # Track with listening history
            stats = self.track_stats[song_key]
            g_t = stats['g_t']
            A_t = stats['A_t']
            
            # Exploitation term: (1-d) * g_t * A_t
            exploit_term = (1 - d) * g_t * A_t
            
            # Exploration term: d * (1-g_t) * F_t
            F_t = self.compute_prior_utility(song_key, song_metadata, song_embedding)
            explore_term = d * (1 - g_t) * F_t
            
        else:
            # Track with no listening history
            g_t = 0.0  # No confidence for unplayed tracks
            A_t = 0.0  # No affinity data
            
            # Exploitation term: (1-d) * g_t * A_t = 0
            exploit_term = (1 - d) * g_t * A_t
            
            # Exploration term: d * (1-g_t) * F_t = d * 1 * F_t
            F_t = self.compute_prior_utility(song_key, song_metadata, song_embedding)
            explore_term = d * (1 - g_t) * F_t
        
        U_t = exploit_term + explore_term
        return float(np.clip(U_t, 0, 1))
    
    def compute_final_score(self, semantic_similarity: float, song_key: Tuple[str, str],
                          song_metadata: Dict, song_embedding: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute final V2 score for a track.
        
        Args:
            semantic_similarity: Query-track semantic similarity [0, 1]
            song_key: (song, artist) tuple
            song_metadata: Song metadata dict  
            song_embedding: Normalized track embedding
            
        Returns:
            Tuple of (final_score, component_breakdown)
        """
        # Semantic relevance (S_t)
        S_t = float(np.clip(semantic_similarity, 0, 1))
        
        # Predicted utility (U_t)
        U_t = self.compute_predicted_utility(song_key, song_metadata, song_embedding)
        
        # Final score: λ * S_t + (1-λ) * U_t
        lambda_val = self.config.lambda_val
        final_score = lambda_val * S_t + (1 - lambda_val) * U_t
        
        # Component breakdown for analysis
        components = {
            'semantic_similarity': S_t,
            'semantic_weighted': lambda_val * S_t,
            'predicted_utility': U_t,
            'utility_weighted': (1 - lambda_val) * U_t,
            'final_score': final_score,
            'lambda': lambda_val,
            'discovery_slider': self.config.d
        }
        
        # Add detailed utility breakdown if we have history
        if song_key in self.track_stats:
            stats = self.track_stats[song_key]
            F_t = self.compute_prior_utility(song_key, song_metadata, song_embedding)
            
            components.update({
                'confidence_gate': stats['g_t'],
                'track_affinity': stats['A_t'],
                'prior_utility': F_t,
                'exploit_term': (1 - self.config.d) * stats['g_t'] * stats['A_t'],
                'explore_term': self.config.d * (1 - stats['g_t']) * F_t
            })
        
        return float(np.clip(final_score, 0, 1)), components


def initialize_ranking_engine(history_df: pd.DataFrame, songs_metadata: List[Dict], 
                       embeddings_data: Dict, config: RankingConfig = None) -> RankingEngine:
    """
    Initialize a complete ranking engine with data.
    
    Args:
        history_df: Listening history DataFrame
        songs_metadata: List of song metadata dicts
        embeddings_data: Dictionary mapping song_keys to embeddings
        config: Optional custom configuration
        
    Returns:
        Initialized RankingEngine instance
    """
    engine = RankingEngine(config)
    
    if not history_df.empty:
        # Compute track statistics
        engine.compute_track_statistics(history_df, songs_metadata)
        
        # Compute artist affinities
        engine.compute_artist_affinities()
        
        # Compute reference centroid
        engine.compute_reference_centroid(embeddings_data)
    else:
        logger.warning("No history data provided - engine will use prior-only scoring")
    
    return engine