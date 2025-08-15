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
import time

logger = logging.getLogger(__name__)


class RankingConfig:
    """Configuration class for ranking algorithm hyperparameters."""
    
    def __init__(self, 
        # Core V2.5 parameters - same as V2
        H_c: float = 30.0,          # History half-life in days
        H_E: float = 270.0,         # Artist history half-life in days
        K: float = 5.0,             # Confidence ramp parameter
        lambda_val: float = 0.5,    # Relevance vs utility balance (λ in formula)
        
        # Discovery slider (user-controlled, default neutral)
        d: float = 0.5,             # Discovery slider 0=familiar, 1=new
        H_d: float = 360.0,         # Discovery familiarity half-life in days
        K_d: float = 8.0,           # Discovery familiarity scaling parameter
        kappa_d: float = 1.386,     # Discovery strength parameter (ln(4))
        
        # V2.5: Curved signed evidence parameters
        gamma_s: float = 1.2,       # Completion curvature for success
        gamma_f: float = 1.4,       # Skip curvature for failures
        kappa: float = 2.0,         # Skip amplification factor
        
        # V2.5: Beta prior and stability parameters
        alpha_0: float = 1.0,       # Beta prior alpha
        beta_0: float = 1.0,        # Beta prior beta
        K_s: float = 3.0,           # Stability parameter

        # V2.5: Artist affinity prior parameters
        K_E: float = 10.0,           # Artist affinity prior parameter
        gamma_A: float = 0.7,
        
        # V2.5: Freshness parameters
        eta: float = 1.2,           # Freshness sharpness
        tau: float = 0.8,           # Freshness threshold
        beta_f: float = 1.5,          # Freshness exponent

        # V2.5: Familiarity parameters
        K_life: float = 10.0,       # Familiarity half-life
        K_recent: float = 5.0,      # Recent familiarity half-life
        psi: float = 1.4,           # Familiarity exponent
        
        # V2.5: kNN similarity prior parameters
        k_neighbors: int = 50,      # Number of kNN neighbors
        sigma: float = 10.0,        # Softmax temperature for kNN weights        
        
        # V2.5: Prior combination weights
        beta_p: float = 0.4,        # Prior weight for popularity
        beta_s: float = 0.4,        # Prior weight for kNN similarity
        beta_a: float = 0.2,        # Prior weight for artist affinity
        kappa_E: float = 0.25       # Scaling factor for E_t
        ):       
            """Initialize with V2.5 hyperparameters (all configurable via keyword arguments)."""
            self.H_c = H_c
            self.H_E = H_E
            self.K = K
            self.lambda_val = lambda_val
            self.d = d
            self.H_d = H_d
            self.K_d = K_d
            self.kappa_d = kappa_d
            self.gamma_s = gamma_s
            self.gamma_f = gamma_f
            self.kappa = kappa
            self.alpha_0 = alpha_0
            self.beta_0 = beta_0
            self.K_s = K_s
            self.K_E = K_E
            self.gamma_A = gamma_A
            self.eta = eta
            self.tau = tau
            self.beta_f = beta_f
            self.K_life = K_life
            self.K_recent = K_recent
            self.psi = psi
            self.k_neighbors = k_neighbors
            self.sigma = sigma
            self.beta_p = beta_p
            self.beta_s = beta_s
            self.beta_a = beta_a
            self.kappa_E = kappa_E
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary format."""
        return {
            'H_c': self.H_c,
            'H_E': self.H_E,
            'K': self.K,
            'lambda': self.lambda_val,
            'd': self.d,
            'H_d': self.H_d,
            'K_d': self.K_d,
            'kappa_d': self.kappa_d,
            'gamma_s': self.gamma_s,
            'gamma_f': self.gamma_f,
            'kappa': self.kappa,
            'alpha_0': self.alpha_0,
            'beta_0': self.beta_0,
            'K_s': self.K_s,
            'K_E': self.K_E,
            'gamma_A': self.gamma_A,
            'eta': self.eta,
            'tau': self.tau,
            'beta_f': self.beta_f,
            'K_life': self.K_life,
            'K_recent': self.K_recent,
            'psi': self.psi,
            'k_neighbors': self.k_neighbors,
            'sigma': self.sigma,
            'beta_p': self.beta_p,
            'beta_s': self.beta_s,
            'beta_a': self.beta_a,
            'kappa_E': self.kappa_E
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
                elif key == 'kappa_d':  # Discovery strength parameter
                    if float_value > 0:
                        self.kappa_d = float_value
                    else:
                        logger.warning(f"Discovery strength must be positive, got {float_value}")
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
        self.knn_similarities = {}      # Per-track kNN similarities
        self.track_priors = {}          # Per-track priors
        self.p05_C_t = None
        self.p95_C_t = None
        self.s_base = None
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
        df['w_ti_a'] = np.maximum(np.exp(-np.log(2) * days_ago / self.config.H_E), 1e-10)
        df['w_ti_d'] = np.maximum(np.exp(-np.log(2) * days_ago / self.config.H_d), 1e-10)
        
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
        df['s_ti_a'] = df['w_ti_a'] * (df['c_ti'] ** self.config.gamma_s)
        df['s_ti_d'] = df['w_ti_d'] * (df['c_ti'] ** self.config.gamma_s)

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
            S_t_a = group['s_ti_a'].sum()  # Total success evidence (artist affinity)
            S_t_d = group['s_ti_d'].sum()  # Total success evidence (discovery familiarity)
            R_t = group['c_ti'].sum()
            
            # Bayesian affinity (V2.5 §3.2)
            alpha = self.config.alpha_0 + S_t
            beta = self.config.beta_0 + F_t
            A_t = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            
            # Variance-aware confidence (V2.5 §3.3)
            if (alpha + beta) > 0:
                variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
                s_t = (1 - (variance * 12)) * (n_t / (n_t + self.config.K_s))
            else:
                s_t = (n_t / (n_t + self.config.K_s))
            
            # Freshness from all plays (V2.5 §3.4)
            # f_t = (sum(w_{t,i}) / num_plays)^eta
            num_plays = len(group)
            num_completions = group['num_completions'].sum().astype(int)
            num_skips = group['num_skips'].sum().astype(int)
            mean_recency = group['w_ti'].sum() / num_plays if num_plays > 0 else 0
            f_t = mean_recency ** self.config.eta
            m_f_t = self.config.tau + (1 - self.config.tau) * (f_t ** self.config.beta_f)
            
            # Familiarity from all plays (V2.5 §3.4)
            L_t = (R_t / (R_t + self.config.K_life))
            R_t_rec = (n_t / (n_t + self.config.K_recent))
            h_t = 1 - ((1 - L_t) * ((1 - R_t_rec) ** self.config.psi))
            
            # Discovery familiarity (long horizon)
            phi_t = S_t_d / (S_t_d + self.config.K_d)

            # Final expoitation quality score
            Q_t = s_t * A_t * m_f_t
            
            track_stats[song_key] = {
                'S_t': float(S_t),
                'F_t': float(F_t),
                'n_t': float(n_t),
                'S_t_a': float(S_t_a),
                'S_t_d': float(S_t_d),
                'R_t': float(R_t),
                'A_t': float(np.clip(A_t, 0, 1)),
                's_t': float(np.clip(s_t, 0, 1)),
                'f_t': float(np.clip(f_t, 0, 1)),
                'm_f_t': float(np.clip(m_f_t, 0, 1)),
                'h_t': float(np.clip(h_t, 0, 1)),
                'phi_t': float(np.clip(phi_t, 0, 1)),
                'Q_t': float(np.clip(Q_t, 0, 1)),
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
    

    def compute_track_priors(self, songs_metadata: Dict, embeddings_data: Dict) -> None:
        """
        Compute per-track priors using V2.5 §4.
        
        Args:
            song_metadata: Dictionary mapping (song, artist) -> song metadata
        """ 
        if not embeddings_data:
            return {}
        
        track_priors = {}
        
        for song in songs_metadata:
            song_key = (song['original_song'], song['original_artist'])
            # Compute three priors (V2.5 §4)
            # P_t: Popularity prior
            popularity = song.get('metadata', {}).get('popularity', 50)
            P_t = np.clip(popularity / 100.0, 0.0, 1.0)
            
            # C_t: kNN similarity prior
            C_t = self.knn_similarities.get(song_key)
            C_t_hat = np.clip((C_t - self.p05_C_t) * (1 / (self.p95_C_t - self.p05_C_t)), 0, 1)
            
            # B_t: Artist affinity prior
            B_t = self.artist_affinities.get(song_key[1], 0.0)

            # Combined prior utility E_t
            E_t = (
                self.config.beta_p * P_t + 
                self.config.beta_s * C_t_hat + 
                self.config.beta_a * B_t)

            track_priors[song_key] = {
                'P_t': P_t,
                'C_t': C_t,
                'B_t': B_t,
                'C_t_hat': C_t_hat,
                'E_t': E_t
            }
        
        # self.median_Q_t = np.median([stats['Q_t'] for stats in self.track_stats.values()])
        # self.median_E_t = np.median([priors['E_t'] for priors in track_priors.values()])
        # self.s_base = self.median_Q_t / (self.median_E_t + 1e-9)
        self.track_priors = track_priors
        return track_priors


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
                artist_data[artist] = {'E_A': 0}

            artist_data[artist]['E_A'] += (stats['S_t_a'] ** self.config.gamma_A)
                    
        artist_affinities = {}
        for artist, data in artist_data.items():
            artist_affinities[artist] = data['E_A'] / (data['E_A'] + (self.config.K_E ** self.config.gamma_A))

        logger.info(f"Computed artist affinities for {len(artist_affinities)} artists")
        self.artist_affinities = artist_affinities
        return artist_affinities
    
    
    def compute_knn_similarities(self, embeddings_data: Dict) -> Dict[Tuple[str, str], float]:
        """
        Compute kNN similarity prior C_t for all tracks in embeddings_data (V2.5 §4.1).
        
        For each track in embeddings_data, find the k most similar tracks from those with history,
        then compute a trust-weighted average of their Q_t scores.
        
        Args:
            embeddings_data: Dictionary mapping (song, artist) -> embedding for all tracks
            
        Returns:
            Tuple of (knn_similarities_dict, max_terms_dict) for debugging
        """
        if not embeddings_data:
            return {}
        
        # Build lists: all tracks vs historical tracks
        all_embeddings = []
        all_track_keys = []
        historical_embeddings = []
        historical_track_keys = []
        trust_scores = []
        
        t0 = time.time()
        for song_key in embeddings_data:
            all_embeddings.append(embeddings_data[song_key])
            all_track_keys.append(song_key)
            
            # Only tracks with history can be neighbors
            if song_key in self.track_stats:
                historical_embeddings.append(embeddings_data[song_key])
                historical_track_keys.append(song_key)
                trust_scores.append(self.track_stats[song_key]['Q_t'])
        
        t1 = time.time()
        print(f"Time taken to get embeddings: {(t1 - t0) * 1000} milliseconds")
        print(f"All tracks: {len(all_embeddings)}, Historical tracks: {len(historical_embeddings)}")
        
        # Use a more reasonable minimum threshold
        k_min = max(3, min(10, self.config.k_neighbors // 5))
        if len(historical_embeddings) < k_min:
            print(f"Not enough historical tracks ({len(historical_embeddings)}) for meaningful kNN")
            return {}, {}
        
        t2 = time.time()
        all_embeddings = np.array(all_embeddings)  # shape: (n_all, embed_dim)
        historical_embeddings = np.array(historical_embeddings)  # shape: (n_hist, embed_dim)
        trust_scores = np.array(trust_scores)  # shape: (n_hist,)
        
        n_all_tracks = len(all_embeddings)
        n_historical_tracks = len(historical_embeddings)
        
        # Compute similarity matrix: all_tracks x historical_tracks
        # similarity_matrix[i, j] = cosine similarity between all_track[i] and historical_track[j]
        similarity_matrix = np.matmul(all_embeddings, historical_embeddings.T)  # shape: (n_all, n_hist)
        
        # Handle self-similarities: for tracks that are also historical, 
        # we need to exclude their similarity to themselves
        # Create efficient lookup: historical_key -> historical_index
        hist_key_to_idx = {key: idx for idx, key in enumerate(historical_track_keys)}
        
        # Find matches and exclude self-similarities
        all_to_hist_mapping = {}
        for i, all_key in enumerate(all_track_keys):
            if all_key in hist_key_to_idx:
                j = hist_key_to_idx[all_key]
                all_to_hist_mapping[i] = j
                similarity_matrix[i, j] = -np.inf  # Exclude self-similarity
        
        # Get the indices corresponding to the top-k similarities for each track
        # If any tracks are both in all_tracks and historical_tracks, we excluded self-similarities,
        # so we need to account for having one fewer neighbor available for those tracks
        if all_to_hist_mapping:
            # Some tracks have self-similarities excluded
            k = min(self.config.k_neighbors, n_historical_tracks - 1)
        else:
            # No self-similarities to exclude
            k = min(self.config.k_neighbors, n_historical_tracks)
        
        top_k_indices = np.argpartition(similarity_matrix, -k, axis=1)[:, -k:]  # shape: (n_all, k)
        
        # Get the top-k similarities and trust scores for each track using advanced indexing
        row_indices = np.arange(n_all_tracks)[:, np.newaxis]  # shape: (n_all, 1)
        top_k_similarities = similarity_matrix[row_indices, top_k_indices]  # shape: (n_all, k)
        top_k_trust = trust_scores[top_k_indices]  # shape: (n_all, k)
        
        t3 = time.time()
        print(f"Time taken to compute similarities: {(t3 - t2) * 1000} milliseconds")
        
        # Compute softmax weights (V2.5 §4.1)
        exp_scores = np.exp(self.config.sigma * top_k_similarities)  # shape: (n_all, k)
        softmax_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # shape: (n_all, k)
        
        # Trust-weighted average
        C_t_terms = softmax_weights * top_k_trust  # shape: (n_all, k)
        C_t_values = np.clip(np.sum(C_t_terms, axis=1), 0, 1)  # shape: (n_all,)
        # Get max term for debugging purposes
        # max_terms = np.max(C_t_terms, axis=1)  # shape: (n_all,)
        
        t4 = time.time()
        print(f"Time taken to compute C_t: {(t4 - t3) * 1000} milliseconds")
        
        # Build dictionaries mapping (song, artist) -> values
        knn_similarities = {song_key: float(C_t_val) for song_key, C_t_val in zip(all_track_keys, C_t_values)}
        # max_terms_dict = {song_key: float(max_term) for song_key, max_term in zip(all_track_keys, max_terms)}
        
        self.p05_C_t = np.percentile(C_t_values, 5)
        self.p95_C_t = np.percentile(C_t_values, 95)
        self.knn_similarities = knn_similarities
        return knn_similarities
    
    
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
                # round key to 4 decimal places to avoid floating point precision issues
                quantile_map[round(value, 4)] = quantile
        
        return quantile_map
    

    def compute_v25_final_score(self, semantic_similarity: float, song_key: Tuple[str, str]) -> Tuple[float, Dict]:
        """
        Compute final V2.5 score for a track.
        
        Args:
            semantic_similarity: Query-track semantic similarity [0, 1]
            song_key: (song, artist) tuple
            
        Returns:
            Tuple of (final_score, component_breakdown)
        """
        # Semantic relevance (S_t): rescale cosine [-1,1] to [0,1] 
        S_t = float(np.clip((semantic_similarity + 1) / 2, 0, 1))
        
        # Get track statistics if available
        if song_key in self.track_stats:
            stats = self.track_stats[song_key]
            Q_t = stats['Q_t']
            h_t = stats['h_t']
            phi_t = stats['phi_t']
        else:
            # No history for this track - low familiarity
            Q_t = 0.0
            h_t = 0.0
            phi_t = 0.0

        priors = self.track_priors[song_key]
        P_t = priors['P_t']
        C_t = priors['C_t']
        B_t = priors['B_t']
        C_t_hat = priors['C_t_hat']
        E_t = priors['E_t']
        E_t_hat = self.config.kappa_E * E_t

        # Core utility (V2.5 revised): keep principled Q_t vs E_t balance
        U_t = h_t * Q_t + (1 - h_t) * E_t_hat
        
        # Discovery tilt (exponential familiarity-based scaling)
        # s_d = exp(kappa_d * (d - 0.5) * (1 - 2*phi_t))
        discovery_bias = (self.config.d - 0.5) * (1 - 2 * phi_t)
        s_d = np.exp(self.config.kappa_d * discovery_bias)
        
        # Discovery-adjusted utility
        U_t_discovery = s_d * U_t
        
        # Final score: λ * S_t + (1-λ) * U_t^(d)
        lambda_val = self.config.lambda_val
        final_score = lambda_val * S_t + (1 - lambda_val) * U_t_discovery
        
        # Component breakdown for analysis
        components = {
            'semantic_similarity': S_t,
            'semantic_weighted': lambda_val * S_t,
            'core_utility': U_t,
            'discovery_multiplier': s_d,
            'discovery_adjusted_utility': U_t_discovery,
            'utility_weighted': (1 - lambda_val) * U_t_discovery,
            'final_score': final_score,
            'lambda': lambda_val,
            'discovery_slider': self.config.d,
            'discovery_bias': discovery_bias,
            # Prior components
            'P_t': P_t,
            'C_t': C_t,
            'B_t': B_t,
            'C_t_hat': C_t_hat,
            'E_t': E_t,
            'E_t_hat': E_t_hat,
            # History components
            'Q_t': Q_t,
            'h_t': h_t,
            'phi_t': phi_t,
            'exploit_term': h_t * Q_t,
            'explore_term': (1 - h_t) * E_t_hat
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

        # Compute kNN similarities
        engine.compute_knn_similarities(embeddings_data)

        # Compute track priors
        engine.compute_track_priors(songs_metadata, embeddings_data)

    else:
        logger.warning("No history data provided - engine will use prior-only scoring")
    
    return engine
