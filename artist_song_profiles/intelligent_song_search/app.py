#!/usr/bin/env python3
"""
Intelligent Song Search App
Supports both text-to-song and song-to-song search with multiple embedding types.
"""
import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib
from openai import OpenAI
from typing import List, Dict, Tuple, Optional
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command line arguments
def parse_arguments():
    """Parse command line arguments for data file paths."""
    parser = argparse.ArgumentParser(
        description="Intelligent Song Search - AI-powered music discovery with Spotify integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py
  python app.py --songs custom_songs.json --embeddings custom_embeddings.npz
  python app.py -s /path/to/songs.json -e /path/to/embeddings.npz
        """
    )
    
    # Default paths (relative to the script location)
    default_songs_file = Path(__file__).parent.parent / 'pop_eval_set_v0' / 'pop_eval_set_v0_results_enriched.json'
    default_embeddings_file = Path(__file__).parent.parent / 'pop_eval_set_v0' / 'pop_eval_set_v0_embeddings.npz'
    
    parser.add_argument(
        '-s', '--songs',
        type=str,
        default=str(default_songs_file),
        help=f'Path to song profiles JSON file (default: {default_songs_file})'
    )
    
    parser.add_argument(
        '-e', '--embeddings',
        type=str,
        default=str(default_embeddings_file),
        help=f'Path to embeddings NPZ file (default: {default_embeddings_file})'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to run the server on (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to run the server on (default: 5000)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )
    
    return parser.parse_args()

# Global variable to store arguments (will be set in main)
args = None

app = Flask(__name__)
# Use persistent secret key or generate one (sessions reset on app restart with random key)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', os.urandom(24))
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Spotify configuration
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET') 
SPOTIFY_SCOPES = "streaming user-read-email user-read-private user-read-playback-state user-modify-playback-state"

# Validate required environment variables
if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    logger.error("Missing required Spotify credentials. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
    exit(1)

# OpenAI configuration
openai_client = OpenAI()

# Validate OpenAI API key exists
if not os.getenv('OPENAI_API_KEY'):
    logger.warning("OPENAI_API_KEY not set. Text-to-song search will not work.")

class MusicSearchEngine:
    """Core search engine for intelligent music search."""
    
    def __init__(self, songs_file: str, embeddings_file: str):
        self.songs_file = songs_file
        self.embeddings_file = embeddings_file
        self.songs = []
        self.embeddings_data = None
        self.song_lookup = {}
        self.embedding_indices = {}
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        self._load_data()
        self._build_indices()
        self._build_text_search_index()
    
    def _load_data(self):
        """Load song profiles and embeddings."""
        logger.info("Loading song profiles...")
        with open(self.songs_file, 'r') as f:
            self.songs = json.load(f)
        
        # Create lookup for quick access
        for i, song in enumerate(self.songs):
            song_key = (song['original_song'], song['original_artist'])
            self.song_lookup[song_key] = i
        
        logger.info(f"Loaded {len(self.songs)} songs")
        
        # Load embeddings
        logger.info("Loading embeddings...")
        self.embeddings_data = np.load(self.embeddings_file, allow_pickle=True)
        logger.info(f"Loaded {len(self.embeddings_data['embeddings'])} embeddings")
        
        # Check if field_values exists (for backwards compatibility)
        if 'field_values' not in self.embeddings_data:
            logger.warning("Embeddings file does not contain field_values - accordion functionality will be disabled")
            # Create placeholder field_values array
            self.embeddings_data = dict(self.embeddings_data)  # Convert to regular dict for modification
            self.embeddings_data['field_values'] = np.array(['N/A'] * len(self.embeddings_data['embeddings']), dtype=object)
    
    def _build_indices(self):
        """Build indices for each embedding type."""
        logger.info("Building embedding indices...")
        
        # Group embeddings by type
        for embed_type in ['full_profile', 'sound_aspect', 'meaning_aspect', 'mood_aspect', 'tags_genres']:
            mask = self.embeddings_data['field_types'] == embed_type
            if mask.any():
                self.embedding_indices[embed_type] = {
                    'embeddings': self.embeddings_data['embeddings'][mask],
                    'song_indices': self.embeddings_data['song_indices'][mask],
                    'field_values': self.embeddings_data['field_values'][mask]
                }
                logger.info(f"Built index for {embed_type}: {len(self.embedding_indices[embed_type]['embeddings'])} embeddings")
    
    def _build_text_search_index(self):
        """Build text search index for song/artist/album matching."""
        logger.info("Building text search index...")
        
        # Create searchable text for each song
        search_texts = []
        for song in self.songs:
            metadata = song.get('metadata', {})
            searchable_text = f"{song['original_song']} {song['original_artist']} {metadata.get('album_name', '')}"
            search_texts.append(searchable_text.lower())
        
        # Build TF-IDF index
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(search_texts)
    
    def search_songs_by_text(self, query: str, limit: int = 10) -> List[Tuple[int, float, str]]:
        """Search for songs using text similarity (for song-to-song search suggestions)."""
        if not query or not self.tfidf_vectorizer:
            return []
        
        query_lower = query.lower().strip()
        
        # Transform query for TF-IDF
        query_vec = self.tfidf_vectorizer.transform([query_lower])
        
        # Compute similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top matches with original threshold
        top_indices = similarities.argsort()[::-1][:limit]
        
        results = []
        tfidf_results_count = 0
        
        # First, try TF-IDF results with original threshold
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Original threshold
                song = self.songs[idx]
                label = f"{song['original_song']} - {song['original_artist']}"
                results.append((int(idx), float(similarities[idx]), label))
                tfidf_results_count += 1
        
        # If TF-IDF didn't find good matches, fall back to exact string matching
        if tfidf_results_count < 3:  # If we have fewer than 3 good TF-IDF matches
            logger.info(f"TF-IDF found only {tfidf_results_count} matches for '{query}', using fallback search")
            
            # Build a list of candidates with their match scores
            fallback_candidates = []
            
            for idx, song in enumerate(self.songs):
                metadata = song.get('metadata', {})
                song_title = song['original_song'].lower()
                artist_name = song['original_artist'].lower()
                album_name = metadata.get('album_name', '').lower()
                
                # Calculate simple string similarity scores
                match_score = 0.0
                
                # Exact matches get highest priority
                if query_lower == song_title:
                    match_score = 1.0
                elif query_lower == artist_name:
                    match_score = 0.9
                elif query_lower == album_name:
                    match_score = 0.8
                # Substring matches
                elif query_lower in song_title:
                    match_score = 0.7
                elif query_lower in artist_name:
                    match_score = 0.6
                elif query_lower in album_name:
                    match_score = 0.5
                # Word boundary matches (for multi-word titles)
                elif any(word.startswith(query_lower) for word in song_title.split()):
                    match_score = 0.4
                elif any(word.startswith(query_lower) for word in artist_name.split()):
                    match_score = 0.3
                
                if match_score > 0:
                    fallback_candidates.append((idx, match_score, song))
            
            # Sort by match score and add to results (avoid duplicates)
            existing_indices = {result[0] for result in results}
            fallback_candidates.sort(key=lambda x: x[1], reverse=True)
            
            for idx, score, song in fallback_candidates:
                if idx not in existing_indices and len(results) < limit:
                    label = f"{song['original_song']} - {song['original_artist']}"
                    results.append((int(idx), float(score), label))
        
        # Sort results by score (descending) and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def get_song_embedding(self, song_idx: int, embed_type: str) -> Optional[np.ndarray]:
        """Get embedding for a specific song and embedding type."""
        if embed_type not in self.embedding_indices:
            return None
        
        indices = self.embedding_indices[embed_type]
        mask = indices['song_indices'] == song_idx
        
        if mask.any():
            return indices['embeddings'][mask][0]
        return None
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding for text query."""
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text,
                encoding_format="float"
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error getting text embedding: {e}")
            raise
    
    def similarity_search(self, query_embedding: np.ndarray, embed_type: str, k: int = 20, offset: int = 0) -> Tuple[List[Tuple[int, float, str]], int]:
        """Perform similarity search using cosine similarity.
        
        Returns:
            Tuple[List[Tuple[int, float, str]], int]: (paginated_results_with_field_values, total_count)
            Each result tuple contains: (song_idx, similarity, field_value)
        """
        if embed_type not in self.embedding_indices:
            return [], 0
        
        indices = self.embedding_indices[embed_type]
        embeddings = indices['embeddings']
        song_indices = indices['song_indices']
        field_values = indices['field_values']  # Get the field values for accordion display
        
        # Normalize embeddings for cosine similarity
        query_norm_value = np.linalg.norm(query_embedding)
        if query_norm_value == 0:
            logger.warning("Query embedding is zero vector, returning empty results")
            return [], 0
        
        query_norm = query_embedding / query_norm_value
        embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero for embedding normalization
        embedding_norms = np.where(embedding_norms == 0, 1, embedding_norms)
        embeddings_norm = embeddings / embedding_norms
        
        # Compute similarities
        similarities = np.dot(embeddings_norm, query_norm)
        
        # Get all results sorted by similarity
        all_indices = similarities.argsort()[::-1]
        total_count = len(all_indices)
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + k
        paginated_indices = all_indices[start_idx:end_idx]
        
        results = []
        for idx in paginated_indices:
            song_idx = song_indices[idx]
            similarity = similarities[idx]
            field_value = field_values[idx]  # Get the original text that was embedded
            results.append((int(song_idx), float(similarity), str(field_value)))  # Convert all to native Python types
        
        return results, total_count

# Initialize search engine
search_engine = None

def init_search_engine(songs_file: str = None, embeddings_file: str = None):
    """Initialize the search engine with data files."""
    global search_engine
    if search_engine is None:
        # Use provided file paths or fall back to parsed arguments or defaults
        if songs_file and embeddings_file:
            songs_path = songs_file
            embeddings_path = embeddings_file
        elif args:
            songs_path = songs_file or args.songs
            embeddings_path = embeddings_file or args.embeddings
        else:
            # Fallback defaults when no args available (e.g., when imported)
            default_songs = Path(__file__).parent.parent / 'pop_eval_set_v0' / 'pop_eval_set_v0_results_enriched.json'
            default_embeddings = Path(__file__).parent.parent / 'pop_eval_set_v0' / 'pop_eval_set_v0_embeddings.npz'
            songs_path = songs_file or str(default_songs)
            embeddings_path = embeddings_file or str(default_embeddings)
        
        # Validate that files exist
        if not Path(songs_path).exists():
            logger.error(f"Songs file not found: {songs_path}")
            raise FileNotFoundError(f"Songs file not found: {songs_path}")
        
        if not Path(embeddings_path).exists():
            logger.error(f"Embeddings file not found: {embeddings_path}")
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        logger.info(f"Initializing search engine with:")
        logger.info(f"  Songs file: {songs_path}")
        logger.info(f"  Embeddings file: {embeddings_path}")
        
        search_engine = MusicSearchEngine(songs_path, embeddings_path)

# Spotify OAuth setup
def get_spotify_oauth():
    # Construct redirect URI dynamically based on current host/port
    host = args.host if args else '127.0.0.1'
    port = args.port if args else 5000
    redirect_uri = f"http://{host}:{port}/callback"
    
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=redirect_uri,
        scope=SPOTIFY_SCOPES,
        cache_path=None
    )

@app.route('/')
def index():
    """Main page."""
    init_search_engine()
    return render_template('index.html')

@app.route('/login')
def login():
    """Spotify login."""
    sp_oauth = get_spotify_oauth()
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    """Spotify OAuth callback."""
    sp_oauth = get_spotify_oauth()
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    session['token_info'] = token_info
    return redirect(url_for('index'))

@app.route('/api/search_suggestions')
def search_suggestions():
    """Get song suggestions for song-to-song search."""
    if search_engine is None:
        return jsonify({'error': 'Search engine not initialized'}), 500
        
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify([])
    
    results = search_engine.search_songs_by_text(query, limit=100)
    
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

@app.route('/api/search', methods=['POST'])
def search():
    """Main search endpoint."""
    if search_engine is None:
        return jsonify({'error': 'Search engine not initialized'}), 500
        
    data = request.json
    search_type = data.get('search_type')  # 'text' or 'song'
    embed_type = data.get('embed_type')
    query = data.get('query', '').strip()
    song_idx = data.get('song_idx')  # For song-to-song search
    k = data.get('k', 20)
    offset = data.get('offset', 0)  # For pagination
    
    # Validate pagination parameters
    if k <= 0 or k > 100:  # Reasonable limits
        return jsonify({'error': 'k must be between 1 and 100'}), 400
    if offset < 0:
        return jsonify({'error': 'offset must be non-negative'}), 400
    
    if not query and song_idx is None:
        return jsonify({'error': 'Query or song_idx required'}), 400
    
    try:
        if search_type == 'text':
            # Text-to-song search
            query_embedding = search_engine.get_text_embedding(query)
            results, total_count = search_engine.similarity_search(query_embedding, embed_type, k, offset)
            
        elif search_type == 'song':
            # Song-to-song search
            if song_idx is None:
                return jsonify({'error': 'song_idx required for song-to-song search'}), 400
            
            query_embedding = search_engine.get_song_embedding(song_idx, embed_type)
            if query_embedding is None:
                return jsonify({'error': 'Song embedding not found'}), 404
            
            results, total_count = search_engine.similarity_search(query_embedding, embed_type, k, offset)
            
        else:
            return jsonify({'error': 'Invalid search_type'}), 400
        
        # Format results
        formatted_results = []
        for result in results:
            try:
                # Handle both old format (song_idx, similarity) and new format (song_idx, similarity, field_value)
                if len(result) == 3:
                    result_song_idx, similarity, field_value = result
                elif len(result) == 2:
                    result_song_idx, similarity = result
                    field_value = "N/A"  # Fallback for backwards compatibility
                    logger.warning("Old format results detected - using fallback field_value")
                else:
                    logger.error(f"Unexpected result format: {result}")
                    continue
                    
                song = search_engine.songs[result_song_idx]
                metadata = song.get('metadata', {})
                
                formatted_results.append({
                    'song_idx': int(result_song_idx),  # Convert numpy.int64 to native Python int
                    'song': song['original_song'],
                    'artist': song['original_artist'],
                    'album': metadata.get('album_name', ''),
                    'cover_url': metadata.get('cover_url', ''),
                    'spotify_id': metadata.get('song_id', ''),
                    'similarity': float(similarity),  # Convert numpy.float64 to native Python float
                    'field_value': field_value, # Include the original field value
                    'genres': song.get('genres', []),
                    'tags': song.get('tags', [])
                })
            except (ValueError, IndexError, KeyError) as e:
                logger.error(f"Error processing search result {result}: {e}")
                continue
        
        return jsonify({
            'results': formatted_results,
            'search_type': search_type,
            'embed_type': embed_type,
            'query': query,
            'pagination': {
                'offset': offset,
                'limit': k,
                'total_count': total_count,
                'has_more': offset + k < total_count
            }
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

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

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    
    # Print startup information
    logger.info("Starting Intelligent Song Search application...")
    logger.info(f"Server will run on: http://{args.host}:{args.port}")
    logger.info(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    
    # Initialize search engine early to catch any data file issues
    try:
        init_search_engine()
        logger.info("Search engine initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        logger.error("Please check your data file paths and try again.")
        exit(1)
    
    # Start the Flask application
    app.run(debug=args.debug, host=args.host, port=args.port) 