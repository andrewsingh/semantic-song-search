"""
Ranking utilities for semantic song search.

This module contains the personalized ranking algorithm implementation,
including Spotify history processing and score computation functions.
Designed to be reusable for analysis in Jupyter notebooks.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RankingConfig:
    """Configuration class for ranking algorithm hyperparameters."""
    
    def __init__(self):
        """Initialize with default hyperparameters from design doc v1."""
        # Temporal dynamics parameters
        self.H = 30          # Recency half-life in days
        self.kappa = 0.46    # Affinity saturation rate
        self.lambda_val = 0.5    # Short-skip penalty strength (renamed to avoid keyword conflict)
        self.alpha = 2       # Prior weight for affinity smoothing
        self.A0 = 5          # Prior affinity value
        self.S = 10          # Satiation time scale in days
        self.beta = 0.5      # UCB exploration coefficient
        self.gamma = 0.6     # Popularity damping exponent
        
        # Component weights
        self.w_sem = 0.50    # Weight for semantic similarity
        self.w_int = 0.30    # Weight for personal interest
        self.w_ucb = 0.15    # Weight for exploration
        self.w_pop = 0.05    # Weight for popularity
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary format."""
        return {
            'H': self.H,
            'kappa': self.kappa,
            'lambda': self.lambda_val,  # Note: using lambda_val internally
            'alpha': self.alpha,
            'A0': self.A0,
            'S': self.S,
            'beta': self.beta,
            'gamma': self.gamma,
            'w_sem': self.w_sem,
            'w_int': self.w_int,
            'w_ucb': self.w_ucb,
            'w_pop': self.w_pop
        }
    
    def update_weights(self, weights: Dict[str, float]):
        """Update component weights from dictionary."""
        for key, value in weights.items():
            if hasattr(self, key):
                setattr(self, key, float(value))
            else:
                logger.warning(f"Unknown weight parameter: {key}")


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
            'completion': min(1.0, int(entry.get("ms_played", 0)) / max(1, int(entry.get("track_end_timestamp", 1)) - int(entry.get("track_start_timestamp", 0)))),
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


def compute_user_song_stats(history_df: pd.DataFrame, config: RankingConfig) -> Dict[Tuple[str, str], Dict]:
    """
    Compute personalized statistics for each song based on listening history.
    
    Args:
        history_df: DataFrame with cleaned listening history
        config: RankingConfig with hyperparameters
        
    Returns:
        Dictionary mapping (song, artist) tuples to statistics
    """
    if history_df.empty:
        logger.warning("Empty history DataFrame provided")
        return {}
    
    # Extract parameters from config
    H = config.H
    kappa = config.kappa
    lambda_val = config.lambda_val
    alpha = config.alpha
    A0 = config.A0
    S = config.S
    beta = config.beta
    
    logger.info(f"Computing user song statistics for {len(history_df)} history entries...")
    
    # Work with a copy to avoid modifying original
    df = history_df.copy()
    
    # Calculate recency weights
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    days_ago = (now_utc - df['timestamp']).dt.total_seconds() / (24 * 3600)
    df['recency_weight'] = np.exp(-np.log(2) * days_ago / H)
    
    # Calculate short-skip penalty
    short_skip_mask = df['ms_played'] < 15000
    df['skip_penalty'] = lambda_val * short_skip_mask * (1 - df['completion']) * df['recency_weight']
    
    # Calculate equivalent full listens
    df['efl'] = df['recency_weight'] * df['completion']
    
    # Group by song key and aggregate
    song_key_col = df[['original_song', 'original_artist']].apply(tuple, axis=1)
    df['song_key'] = song_key_col
    
    grouped = df.groupby('song_key').agg({
        'efl': 'sum',
        'skip_penalty': 'sum', 
        'ms_played': 'size',  # Count of plays
        'timestamp': 'max'    # Most recent play
    }).rename(columns={'ms_played': 'play_count', 'timestamp': 'last_play'})
    
    # Calculate net equivalent full listens (E_net)
    grouped['E_net'] = np.clip(grouped['efl'] - grouped['skip_penalty'], 0, None)
    
    # Calculate raw affinity with bounds checking
    # Clip E_net to prevent overflow in exp calculation
    e_net_clipped = np.clip(grouped['E_net'], 0, 20)  # exp(-20*0.46) is effectively 0
    grouped['A_raw'] = 100 * (1 - np.exp(-kappa * e_net_clipped))
    
    # Smooth affinity with prior - add safety check for division
    denominator = alpha + grouped['play_count']
    grouped['Affinity'] = np.where(
        denominator > 0,
        (alpha * A0 + grouped['play_count'] * grouped['A_raw']) / denominator,
        A0  # Fallback to prior if denominator is somehow 0
    )
    
    # Calculate satiation and interest with safety checks
    if S is not None and S > 0:
        try:
            days_since_last = (now_utc - grouped['last_play']).dt.total_seconds() / (24 * 3600)
            # Clip days_since_last to prevent extreme values
            days_since_last = np.clip(days_since_last, 0, 365 * 5)  # Max 5 years
            satiation = np.exp(-days_since_last / S)
            grouped['Interest'] = grouped['Affinity'] * (1 - satiation)
        except Exception as e:
            logger.warning(f"Error calculating satiation, using raw affinity: {e}")
            grouped['Interest'] = grouped['Affinity']
    else:
        grouped['Interest'] = grouped['Affinity']
    
    # Calculate UCB exploration bonus with bounds checking
    grouped['UCB'] = beta / np.sqrt(np.maximum(1 + grouped['play_count'], 1))  # Ensure >= 1
    
    # Convert to dictionary format with validation
    user_stats = {}
    invalid_count = 0
    
    for song_key, stats in grouped.iterrows():
        # Validate all numeric values are reasonable
        try:
            efl = float(stats['efl'])
            skip_penalty = float(stats['skip_penalty'])
            E_net = float(stats['E_net'])
            play_count = int(stats['play_count'])
            affinity = float(stats['Affinity'])
            interest = float(stats['Interest'])
            ucb = float(stats['UCB'])
            
            # Check for invalid values
            if not all(np.isfinite([efl, skip_penalty, E_net, affinity, interest, ucb])):
                logger.warning(f"Non-finite values for song {song_key}, skipping")
                invalid_count += 1
                continue
            
            if play_count < 0:
                logger.warning(f"Negative play count for song {song_key}, skipping")
                invalid_count += 1
                continue
            
            user_stats[song_key] = {
                'efl': efl,
                'skip_penalty': skip_penalty,
                'E_net': E_net,
                'play_count': play_count,
                'last_play': stats['last_play'],
                'affinity': np.clip(affinity, 0, 100),  # Ensure reasonable bounds
                'interest': np.clip(interest, 0, 100),  # Ensure reasonable bounds
                'ucb': np.clip(ucb, 0, 1)  # UCB should be 0-1
            }
        except Exception as e:
            logger.warning(f"Error processing stats for song {song_key}: {e}")
            invalid_count += 1
            continue
    
    if invalid_count > 0:
        logger.warning(f"Skipped {invalid_count} songs with invalid statistics")
    
    logger.info(f"Computed statistics for {len(user_stats)} unique songs")
    return user_stats


def compute_composite_score(
    semantic_similarity: float,
    personal_interest: float,
    exploration_bonus: float,
    popularity_score: float,
    config: RankingConfig
) -> Tuple[float, Dict[str, float]]:
    """
    Compute composite ranking score from individual components.
    
    Args:
        semantic_similarity: Cosine similarity score (0-1)
        personal_interest: Personal interest score (0-1)
        exploration_bonus: UCB exploration bonus (0-1)
        popularity_score: Popularity score (0-1)
        config: RankingConfig with component weights
        
    Returns:
        Tuple of (final_score, component_breakdown)
    """
    # Compute weighted components
    semantic_weighted = config.w_sem * semantic_similarity
    interest_weighted = config.w_int * personal_interest
    exploration_weighted = config.w_ucb * exploration_bonus
    popularity_weighted = config.w_pop * popularity_score
    
    # Final composite score
    final_score = semantic_weighted + interest_weighted + exploration_weighted + popularity_weighted
    
    # Component breakdown for analysis
    components = {
        'semantic_similarity': semantic_similarity,
        'semantic_weighted': semantic_weighted,
        'personal_interest': personal_interest,
        'interest_weighted': interest_weighted,
        'exploration_bonus': exploration_bonus,
        'exploration_weighted': exploration_weighted,
        'popularity_score': popularity_score,
        'popularity_weighted': popularity_weighted,
        'final_score': final_score
    }
    
    return final_score, components


# Convenience function for notebook analysis
def analyze_personal_scores(user_stats: Dict[Tuple[str, str], Dict]) -> pd.DataFrame:
    """
    Convert user statistics to DataFrame for easy analysis in notebooks.
    
    Args:
        user_stats: Output from compute_user_song_stats
        
    Returns:
        DataFrame with one row per song and columns for all statistics
    """
    if not user_stats:
        return pd.DataFrame()
    
    # Convert to list of records
    records = []
    for (song, artist), stats in user_stats.items():
        record = {
            'song': song,
            'artist': artist,
            **stats
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Sort by interest score descending for easy analysis
    df = df.sort_values('interest', ascending=False).reset_index(drop=True)
    
    return df