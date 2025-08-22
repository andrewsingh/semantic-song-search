"""
Constants and configuration values shared across the semantic song search application.
"""

# OpenAI Configuration
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"

# Text Search Configuration
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 5000
TEXT_SEARCH_MIN_SCORE = 0.01

# Default File Paths  
DEFAULT_SONGS_FILE = "data/eval_set_v2/eval_set_v2_metadata_ready.json"
DEFAULT_EMBEDDINGS_PATH = "data/eval_set_v2/eval_set_v2_embeddings"
DEFAULT_ARTIST_EMBEDDINGS_PATH = "data/eval_set_v2/eval_set_v2_artist_embeddings"

# Embedding Types
EMBEDDING_TYPES = ['full_profile', 'sound_aspect', 'meaning_aspect', 'mood_aspect', 'tags_genres', 'tags', 'genres']
DEFAULT_ARTIST_MATRIX_EMBED_TYPE = 'tags'
DEFAULT_ARTIST_EMBED_TYPE = 'full_profile'

# Flask Configuration
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000

# Search Configuration
DEFAULT_SEARCH_LIMIT = 20
DEFAULT_SUGGESTION_LIMIT = 10

# Spotify Configuration
SPOTIFY_REDIRECT_URI_LOCAL = "http://127.0.0.1:5000/callback"
SPOTIFY_SCOPES = "user-read-private user-read-email user-top-read playlist-modify-public playlist-modify-private user-read-playback-state user-modify-playback-state streaming"