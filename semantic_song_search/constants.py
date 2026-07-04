"""
Constants and configuration values shared across the semantic song search application.
"""

# OpenAI Configuration
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"


# RapidFuzz Search Configuration
FUZZY_SEARCH_CONSTANTS = {
    'TRACK_WEIGHT': 1.0,
    'ARTIST_WEIGHT': 0.8, 
    'ALBUM_WEIGHT': 0.6,
    'COMBINED_WEIGHT': 0.7,
    'DEFAULT_MIN_SCORE': 75,
    'SEARCH_MIN_SCORE': 70,
    'DEFAULT_SEARCH_LIMIT': 20,
    'TOP_SONGS_LIMIT': 50,
    'MAX_PLAY_BONUS': 5,
    'PLAY_BONUS_DIVISOR': 1000,
    'FLOAT_EPSILON': 0.001
}

# Default File Paths
DEFAULT_SONGS_FILE = "/Users/andrew/dev/semantic-song-search/data/library_v4/library_v4_metadata_with_streams.json"
DEFAULT_EMBEDDINGS_PATH = "/Users/andrew/dev/semantic-song-search/data/library_v4/library_v4_embeddings"
DEFAULT_ARTIST_EMBEDDINGS_PATH = "/Users/andrew/dev/semantic-song-search/data/library_v4/library_v4_artist_embeddings"
DEFAULT_SHARED_GENRE_STORE_PATH = "/Users/andrew/dev/semantic-song-search/data/genre_embedding_store.npz"
DEFAULT_PROFILES_FILE = "/Users/andrew/dev/semantic-song-search/data/library_v4/library_v4_profiles_v4.2.jsonl"

# New Descriptor-Based Embedding Types  
SONG_EMBEDDING_TYPES = ['genres', 'vocal_style', 'production_sound_design', 'lyrical_meaning', 'mood_atmosphere', 'tags']
ARTIST_EMBEDDING_TYPES = ['genres', 'vocal_style', 'production_sound_design', 'lyrical_themes', 'mood_atmosphere', 'cultural_context_scene']

# Legacy embedding types (for backward compatibility during transition)
LEGACY_EMBEDDING_TYPES = ['full_profile', 'sound_aspect', 'meaning_aspect', 'mood_aspect', 'tags_genres', 'tags', 'genres']
LEGACY_ARTIST_EMBEDDING_TYPES = ['musical_style', 'lyrical_themes', 'mood', 'full_profile']

# Remove these legacy constants (no longer needed with descriptor system)
# DEFAULT_ARTIST_MATRIX_EMBED_TYPE = 'tags' 
# DEFAULT_ARTIST_EMBED_TYPE = 'musical_style'

# Flask Configuration
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000

# Search Configuration
DEFAULT_SUGGESTION_LIMIT = 20

# Spotify Configuration
SPOTIFY_REDIRECT_URI_LOCAL = "http://127.0.0.1:5000/callback"
SPOTIFY_SCOPES = "user-read-private user-read-email user-top-read playlist-modify-public playlist-modify-private user-read-playback-state user-modify-playback-state streaming"


# ---------------------------------------------------------------------------
# Playlist-builder session model (see docs/session_model_v2.md)
# Single backend source of truth for the session-model defaults. The
# /api/playlist/suggest endpoint falls back to these per-request, and the
# frontend reads them (injected into playlist.html) instead of hardcoding, so
# backend and frontend defaults cannot drift.
# ---------------------------------------------------------------------------
SESSION_DEFAULTS = {
    # Top-level component weights (omega). Sum to 1. Era (a4) given a small
    # nonzero default so it is actually exercised in the testbed.
    "top_level_weights": {
        "a0_song_sim": 0.55,
        "a1_artist_sim": 0.30,
        "a2_total_streams": 0.05,
        "a3_daily_streams": 0.05,
        "a4_release_date": 0.05,
    },
    # Song descriptor (aspect) preset weights (alpha^pref). Sum to 1.
    "song_weights": {
        "genres": 0.30,
        "vocal_style": 0.15,
        "production_sound_design": 0.15,
        "lyrical_meaning": 0.10,
        "mood_atmosphere": 0.20,
        "tags": 0.10,
    },
    # Artist descriptor preset weights (beta^pref), keyed by artist aspect. Sum to 1.
    "artist_weights": {
        "genres": 0.375,
        "vocal_style": 0.15,
        "production_sound_design": 0.15,
        "lyrical_themes": 0.075,
        "mood_atmosphere": 0.125,
        "cultural_context_scene": 0.125,
    },
    # Sampler + shrinkage hyperparameters (live-editable in the dev panel).
    "kappa": 3.0,          # shrinkage pseudo-count: lambda_n = (n-1)/(n-1+kappa)
    "tau": 0.08,           # softmax temperature
    "k_pool": 50,          # top-K candidate pool for sampling
    "n": 5,                # number of suggestions per slate
    "seed_anchor_weight": 3.0,   # per-track weight for the seed / anchored songs
    "normal_weight": 1.0,        # per-track weight for ordinary selections
    "max_per_artist": 1,   # artist dedup cap per slate; 0/None = unlimited
    "release_date_decay_constant": 10950.0,  # era exp-decay (days)
}