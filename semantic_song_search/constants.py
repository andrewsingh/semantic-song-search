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
DEFAULT_SONGS_FILE = "/Users/andrew/dev/semantic-song-search/data/library_v3_pop_v0/library_v3_pop_v0_metadata_with_streams.json"
DEFAULT_EMBEDDINGS_PATH = "/Users/andrew/dev/semantic-song-search/data/library_v3_pop_v0/library_v3_pop_v0_embeddings"
DEFAULT_ARTIST_EMBEDDINGS_PATH = "/Users/andrew/dev/semantic-song-search/data/library_v3_pop_v0/library_v3_pop_v0_artist_embeddings"
DEFAULT_SHARED_GENRE_STORE_PATH = "/Users/andrew/dev/semantic-song-search/data/genre_embedding_store.npz"

# New Descriptor-Based Embedding Types  
SONG_EMBEDDING_TYPES = ['genres', 'vocal_style', 'production_sound_design', 'lyrical_meaning', 'mood_atmosphere']
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