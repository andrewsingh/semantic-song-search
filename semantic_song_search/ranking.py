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
    
    def __init__(self, 
                 # Core V2.5 parameters - same as V2
                 H_c: float = 30.0,          # History half-life in days
                 K: float = 5.0,             # Confidence ramp parameter
                 lambda_val: float = 0.5,    # Relevance vs utility balance (λ in formula)
                 
                 # Discovery slider (user-controlled, default neutral)
                 d: float = 0.5,             # Discovery slider 0=familiar, 1=new
                 rho: float = 1.0,           # Slider curve parameter (d' = d^rho)
                 
                 # V2.5: Curved signed evidence parameters
                 gamma_s: float = 1.2,       # Completion curvature for success
                 gamma_f: float = 1.4,       # Skip curvature for failures
                 kappa: float = 2.0,         # Skip amplification factor
                 
                 # V2.5: Beta prior parameters
                 alpha_0: float = 1.0,       # Beta prior alpha
                 beta_0: float = 1.0,        # Beta prior beta
                 
                 # V2.5: Freshness parameter
                 eta: float = 1.2,           # Freshness sharpness
                 
                 # V2.5: kNN similarity prior parameters
                 k_neighbors: int = 50,      # Number of kNN neighbors
                 sigma: float = 10.0,        # Softmax temperature for kNN weights
                 
                 # V2.5: Artist affinity prior parameters
                 phi: float = 4.0,           # Artist style focus exponent
                 epsilon: float = 1e-6,      # Denominator guard
                 
                 # V2.5: Prior combination weights
                 beta_p: float = 0.4,        # Prior weight for popularity
                 beta_s: float = 0.4,        # Prior weight for kNN similarity
                 beta_a: float = 0.2):       # Prior weight for artist affinity
        """Initialize with V2.5 hyperparameters (all configurable via keyword arguments)."""
        self.H_c = H_c
        self.K = K
        self.lambda_val = lambda_val
        self.d = d
        self.rho = rho
        self.gamma_s = gamma_s
        self.gamma_f = gamma_f
        self.kappa = kappa
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.eta = eta
        self.k_neighbors = k_neighbors
        self.sigma = sigma
        self.phi = phi
        self.epsilon = epsilon
        self.beta_p = beta_p
        self.beta_s = beta_s
        self.beta_a = beta_a
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary format."""
        return {
            'H_c': self.H_c,
            'K': self.K,
            'lambda': self.lambda_val,
            'd': self.d,
            'rho': self.rho,
            'gamma_s': self.gamma_s,
            'gamma_f': self.gamma_f,
            'kappa': self.kappa,
            'alpha_0': self.alpha_0,
            'beta_0': self.beta_0,
            'eta': self.eta,
            'k_neighbors': self.k_neighbors,
            'sigma': self.sigma,
            'phi': self.phi,
            'epsilon': self.epsilon,
            'beta_p': self.beta_p,
            'beta_s': self.beta_s,
            'beta_a': self.beta_a
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
        self.has_history = False
    
    def compute_track_statistics(self, history_df: pd.DataFrame, songs_metadata: List[Dict]) -> Dict:
        """
        Compute per-track statistics using V2.5 curved signed evidence approach.
        
        Args:
            history_df: DataFrame with cleaned listening history
            songs_metadata: List of song metadata dicts for duration lookup
            
        Returns:
            Dictionary with per-track statistics including A_t, g_t, f_t
        """
        if history_df.empty:
            logger.warning("Empty history DataFrame provided")
            return {}
        
        logger.info(f"Computing V2.5 track statistics for {len(history_df)} history entries...")
        
        # Create duration lookup
        duration_lookup = {}
        for song in songs_metadata:
            song_key = (song['original_song'], song['original_artist'])
            # V2.5 expects track_duration_ms in metadata
            duration_ms = song.get('metadata', {}).get('duration', 180) * 1000  # Default 3 minutes
            duration_lookup[song_key] = duration_ms
        
        # Work with a copy
        df = history_df.copy()
        
        # Calculate per-play primitives (V2.5 §2 & §3.1)
        now_utc = datetime.now(timezone.utc)
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        
        # Days since each play
        days_ago = (now_utc - df['timestamp']).dt.total_seconds() / (24 * 3600)
        
        # Recency weights: w_{t,i} = exp(-ln(2) * (tau - t_{t,i}) / H_c)
        df['w_ti'] = np.maximum(np.exp(-np.log(2) * days_ago / self.config.H_c), 1e-10)
        
        # Completion ratios: c_{t,i} = min(ms_played / D_t, 1)
        def calculate_completion(row):
            song_key = (row['original_song'], row['original_artist'])
            duration_ms = duration_lookup.get(song_key, 180000)  # Default 3 minutes
            # Use actual duration without artificial minimum (as per V2.5 spec)
            return min(1.0, row['ms_played'] / max(1, duration_ms))  # Guard against zero duration
        
        df['c_ti'] = df.apply(calculate_completion, axis=1)
        
        # Curved signed evidence (V2.5 §3.1)
        # s_{t,i} = w_{t,i} * c_{t,i}^{gamma_s}  (success weight)
        # f_{t,i} = w_{t,i} * kappa * (1-c_{t,i})^{gamma_f}  (failure weight)
        df['s_ti'] = df['w_ti'] * (df['c_ti'] ** self.config.gamma_s)
        df['f_ti'] = df['w_ti'] * self.config.kappa * ((1 - df['c_ti']) ** self.config.gamma_f)

        # consider a play a completion if the completion ratio is >= 0.95
        df['num_completions'] = (df['c_ti'] >= 0.95).astype(int)
        # consider a play a skip if the completion ratio is < 0.25
        df['num_skips'] = (df['c_ti'] < 0.25).astype(int)
        
        # Group by song key
        song_key_col = df[['original_song', 'original_artist']].apply(tuple, axis=1)
        df['song_key'] = song_key_col
        
        # Aggregate per track (V2.5 §3.2-3.4)
        track_stats = {}
        
        for song_key, group in df.groupby('song_key'):
            # Aggregate signed evidence
            S_t = group['s_ti'].sum()  # Total success evidence
            F_t = group['f_ti'].sum()  # Total failure evidence  
            n_t = S_t + F_t           # Total evidence
            R_t = group['c_ti'].sum()
            
            # Bayesian affinity (V2.5 §3.2)
            alpha = self.config.alpha_0 + S_t
            beta = self.config.beta_0 + F_t
            A_t = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            
            # Variance-aware confidence (V2.5 §3.3)
            if (alpha + beta) > 0:
                variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
                variance_factor = 1 - (variance / (1/12))  # Normalize by max variance
                g_t = (n_t / (n_t + self.config.K)) * max(0, variance_factor)
            else:
                g_t = 0.0
            
            # Freshness from all plays (V2.5 §3.4)
            # f_t = (sum(w_{t,i}) / num_plays)^eta
            num_plays = len(group)
            num_completions = group['num_completions'].sum().astype(int)
            num_skips = group['num_skips'].sum().astype(int)
            mean_recency = group['w_ti'].sum() / num_plays if num_plays > 0 else 0
            f_t = mean_recency ** self.config.eta
            
            track_stats[song_key] = {
                'S_t': float(S_t),
                'F_t': float(F_t),
                'n_t': float(n_t),
                'R_t': float(R_t),
                'A_t': float(np.clip(A_t, 0, 1)),
                'g_t': float(np.clip(g_t, 0, 1)),
                'f_t': float(np.clip(f_t, 0, 1)),
                'play_count': num_plays,
                'num_completions': num_completions,
                'num_skips': num_skips,
                # Keep for debugging
                'alpha': float(alpha),
                'beta': float(beta),
                'variance': float(variance) if (alpha + beta) > 0 else 0.0
            }
        
        logger.info(f"Computed V2.5 statistics for {len(track_stats)} unique tracks")
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
    
    def compute_knn_similarity_prior(self, candidate_embedding: np.ndarray, 
                                   embeddings_data: Dict) -> float:
        """
        Compute kNN similarity prior C_t for a candidate track (V2.5 §4.1).
        
        Args:
            candidate_embedding: Unit-norm embedding for the candidate track
            embeddings_data: Dictionary mapping (song, artist) -> embedding for historical tracks
            
        Returns:
            C_t: kNN similarity prior score [0, 1]
        """
        if not self.track_stats or not embeddings_data:
            return 0.5  # Fallback for missing data
        
        # Get embeddings for tracks with history
        historical_embeddings = []
        track_keys = []
        trust_scores = []  # g_t * A_t values
        
        for song_key, stats in self.track_stats.items():
            if song_key in embeddings_data:
                historical_embeddings.append(embeddings_data[song_key])
                track_keys.append(song_key)
                trust_scores.append(stats['g_t'] * stats['A_t'])
        
        # Use a more reasonable minimum threshold (spec doesn't define k_min)
        k_min = max(3, min(10, self.config.k_neighbors // 5))  # Dynamic minimum
        if len(historical_embeddings) < k_min:
            # Not enough neighbors for meaningful kNN
            return 0.5
        
        historical_embeddings = np.array(historical_embeddings)
        trust_scores = np.array(trust_scores)
        
        # Compute cosine similarities (embeddings are unit-norm)
        similarities = np.dot(historical_embeddings, candidate_embedding)
        
        # Get top-k neighbors
        k = min(self.config.k_neighbors, len(similarities))
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        
        # Compute softmax weights (V2.5 §4.1)
        top_k_similarities = similarities[top_k_indices]
        exp_scores = np.exp(self.config.sigma * top_k_similarities)
        softmax_weights = exp_scores / np.sum(exp_scores)
        
        # Trust-weighted average
        top_k_trust = trust_scores[top_k_indices]
        C_t = np.dot(softmax_weights, top_k_trust)
        
        return float(np.clip(C_t, 0, 1))
    
    def compute_artist_affinity_prior(self, candidate_song_key: Tuple[str, str],
                                    candidate_embedding: np.ndarray,
                                    embeddings_data: Dict) -> float:
        """
        Compute similarity-conditioned artist affinity B_t (V2.5 §4.2).
        
        Args:
            candidate_song_key: (song, artist) for candidate
            candidate_embedding: Unit-norm embedding for candidate
            embeddings_data: Dictionary mapping (song, artist) -> embedding
            
        Returns:
            B_t: Artist affinity prior score [0, 1]
        """
        if not self.track_stats or not embeddings_data:
            return 0.5
        
        candidate_artist = candidate_song_key[1]
        
        # Find same-artist tracks in history
        same_artist_tracks = []
        for song_key, stats in self.track_stats.items():
            if song_key[1] == candidate_artist and song_key in embeddings_data:
                same_artist_tracks.append({
                    'embedding': embeddings_data[song_key],
                    'n_t': stats['n_t'],
                    'A_t': stats['A_t']
                })
        
        if not same_artist_tracks:
            # No same-artist history
            return 0.5
        
        # Compute similarity-weighted affinity (V2.5 §4.2)
        numerator = 0.0
        denominator = self.config.epsilon
        
        for track in same_artist_tracks:
            # Cosine similarity (both embeddings are unit-norm)
            similarity = np.dot(candidate_embedding, track['embedding'])
            # Only positive similarities, raised to power phi
            similarity_weight = max(0, similarity) ** self.config.phi
            
            numerator += similarity_weight * track['n_t'] * track['A_t']
            denominator += similarity_weight * track['n_t']
        
        B_t = numerator / denominator if denominator > self.config.epsilon else 0.5
        return float(np.clip(B_t, 0, 1))
    
    @staticmethod
    def compute_quantile_normalization(values: List[float]) -> Dict[float, float]:
        """
        Compute empirical CDF for quantile normalization (V2.5 §5).
        
        Uses all values (including duplicates) to compute true empirical CDF:
        F(x) = #{v <= x} / N
        
        Args:
            values: List of raw values to normalize
            
        Returns:
            Dictionary mapping raw values to quantile-normalized [0, 1] values
        """
        if not values:
            return {}
        
        # Sort ALL values (not just unique ones) for proper ECDF
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Build quantile map using empirical CDF with multiplicities
        quantile_map = {}
        
        for value in sorted_values:
            if value not in quantile_map:
                # Count how many values are <= this value
                count_leq = sum(1 for v in sorted_values if v <= value)
                # Empirical CDF: F(x) = #{v <= x} / N
                quantile = count_leq / n
                quantile_map[value] = quantile
        
        return quantile_map
    
    def compute_v25_final_score(self, semantic_similarity: float, song_key: Tuple[str, str],
                               song_metadata: Dict, song_embedding: np.ndarray,
                               embeddings_data: Dict, quantile_maps: Dict[str, Dict]) -> Tuple[float, Dict]:
        """
        Compute final V2.5 score for a track.
        
        Args:
            semantic_similarity: Query-track semantic similarity [0, 1]
            song_key: (song, artist) tuple
            song_metadata: Song metadata dict  
            song_embedding: Normalized track embedding
            embeddings_data: Dictionary mapping song keys to embeddings
            quantile_maps: Pre-computed quantile normalization maps for P, C, B
            
        Returns:
            Tuple of (final_score, component_breakdown)
        """
        # Semantic relevance (S_t): rescale cosine [-1,1] to [0,1] 
        S_t = float(np.clip((semantic_similarity + 1) / 2, 0, 1))
        
        # Compute three priors (V2.5 §4)
        # P_t: Popularity prior
        popularity = song_metadata.get('metadata', {}).get('popularity', 50)
        P_t = np.clip(popularity / 100.0, 0.0, 1.0)
        
        # C_t: kNN similarity prior
        C_t = self.compute_knn_similarity_prior(song_embedding, embeddings_data)
        
        # B_t: Artist affinity prior
        B_t = self.compute_artist_affinity_prior(song_key, song_embedding, embeddings_data)
        
        # Quantile normalize priors (V2.5 §5)
        P_t_hat = quantile_maps['P'].get(P_t, P_t) if quantile_maps.get('P') else P_t
        C_t_hat = quantile_maps['C'].get(C_t, C_t) if quantile_maps.get('C') else C_t
        B_t_hat = quantile_maps['B'].get(B_t, B_t) if quantile_maps.get('B') else B_t
        
        # Combined prior utility F_t
        F_t = (self.config.beta_p * P_t_hat + 
               self.config.beta_s * C_t_hat + 
               self.config.beta_a * B_t_hat)
        
        # Get track statistics if available
        if song_key in self.track_stats:
            stats = self.track_stats[song_key]
            A_t = stats['A_t']
            g_t = stats['g_t']  
            f_t = stats['f_t']
            
            # Exploitation utility: g_t * A_t * f_t
            U_exp_t = g_t * A_t * f_t
        else:
            # No history for this track
            A_t = g_t = f_t = U_exp_t = 0.0
        
        # Discovery control (V2.5 §6.1)
        d_prime = self.config.d ** self.config.rho
        
        # Predicted utility U_t(d) (V2.5 §6.2)
        U_t = (1 - d_prime) * U_exp_t + d_prime * (1 - g_t) * F_t
        
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
            'discovery_slider': self.config.d,
            'd_prime': d_prime,
            # Prior components
            'P_t': P_t,
            'C_t': C_t,
            'B_t': B_t,
            'P_t_hat': P_t_hat,
            'C_t_hat': C_t_hat,
            'B_t_hat': B_t_hat,
            'F_t': F_t,
            # History components
            'A_t': A_t,
            'g_t': g_t,
            'f_t': f_t,
            'U_exp_t': U_exp_t,
            'exploit_term': (1 - d_prime) * U_exp_t,
            'explore_term': d_prime * (1 - g_t) * F_t
        }
        
        return float(np.clip(final_score, 0, 1)), components


def initialize_ranking_engine(history_df: pd.DataFrame, songs_metadata: List[Dict], 
                       embeddings_data: Dict = None, config: RankingConfig = None) -> RankingEngine:
    """
    Initialize a complete ranking engine with data.
    
    Args:
        history_df: Listening history DataFrame
        songs_metadata: List of song metadata dicts
        embeddings_data: Dictionary mapping song_keys to embeddings (unused in V2.5)
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
        
        # V2.5: Reference centroid not needed (uses kNN and artist priors instead)
    else:
        logger.warning("No history data provided - engine will use prior-only scoring")
    
    return engine