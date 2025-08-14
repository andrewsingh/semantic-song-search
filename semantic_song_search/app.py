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
    
    def __init__(self, songs_file: str, embeddings_file: str, history_path: str = None):
        self.songs_file = songs_file
        self.embeddings_file = embeddings_file
        self.history_path = history_path
        
        # Data structures
        self.songs = []
        self.embeddings_data = None
        self.embedding_indices = {}
        self.embedding_lookup = {}  # (song, artist) -> embedding
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # V2 Ranking engine
        self.ranking_engine = None
        self.has_history = False
        self.history_df = None
        
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
        
        # Build embedding lookup (using full_profile embeddings)
        self.embedding_lookup = data_utils.build_embedding_lookup(
            self.embedding_indices, self.songs, 'full_profile'
        )
        
        # Build text search index
        self._build_text_search_index()
        
        # Initialize ranking engine
        self._initialize_ranking_engine()
        
        logger.info(f"Loaded {len(self.songs)} songs with embeddings")
    
    def _build_text_search_index(self):
        """Build text search index for song/artist/album matching."""
        self.tfidf_vectorizer, self.tfidf_matrix = data_utils.build_text_search_index(self.songs)
    
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
                        self.history_df, self.songs, self.embedding_lookup, config
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
            logger.info("Initialized ranking engine without history")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding for text query."""
        return data_utils.get_openai_embedding(text, normalize=True)
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 20, offset: int = 0, 
                         discovery_slider: float = 0.5) -> Tuple[List[Dict], int]:
        """
        Perform V2.5 similarity search with personalized ranking.
        
        Args:
            query_embedding: Normalized query embedding
            k: Number of results to return
            offset: Pagination offset
            discovery_slider: Discovery slider value (0=familiar, 1=new)
        
        Returns:
            Tuple of (results, total_count)
        """
        # Update discovery slider in config
        self.ranking_engine.config.d = discovery_slider
        
        # V2.5: Compute candidates and their priors for quantile normalization
        candidates_data = []
        
        for i, song in enumerate(self.songs):
            song_key = (song['original_song'], song['original_artist'])
            
            # Get song embedding
            if song_key in self.embedding_lookup:
                song_embedding = self.embedding_lookup[song_key]
                semantic_similarity = np.dot(query_embedding, song_embedding)
                
                candidates_data.append({
                    'song_idx': i,
                    'song_key': song_key,
                    'song': song,
                    'semantic_similarity': semantic_similarity,
                })
            else:
                # No embedding available, skip
                continue
        
        if not candidates_data:
            return [], 0
        
        
        # Compute V2.5 final scores
        candidate_scores = []
        
        for candidate in candidates_data:
            final_score, components = self.ranking_engine.compute_v25_final_score(
                candidate['semantic_similarity'], 
                candidate['song_key'], 
                candidate['song'],
            )
            
            candidate_scores.append({
                'song_idx': candidate['song_idx'],
                'song_key': candidate['song_key'],
                'final_score': final_score,
                'semantic_similarity': candidate['semantic_similarity'],
                **components  # Include all component scores for debugging
            })
        
        # Sort by final score
        candidate_scores.sort(key=lambda x: x['final_score'], reverse=True)
        total_count = len(candidate_scores)
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + k
        paginated_results = candidate_scores[start_idx:end_idx]
        
        # Convert to V2.5 result format
        results = []
        for result in paginated_results:
            song_idx = result['song_idx']
            song = self.songs[song_idx]
            song_key = result['song_key']
            
            # Build result dict with V2.5 structure
            result_dict = {
                'song_idx': song_idx,
                'song': song['original_song'],
                'artist': song['original_artist'],
                'cover_url': song.get('metadata', {}).get('cover_url'),
                'album': song.get('metadata', {}).get('album_name', 'Unknown Album'),
                'spotify_id': song.get('metadata', {}).get('song_id', ''),
                'field_value': 'N/A',  # V2.5 uses full_profile by default
                'genres': song.get('genres', []),
                'tags': song.get('tags', []),
                'final_score': result['final_score'],
                'scoring_components': result  # Include all V2.5 components
            }
            
            # Add has_history flag for compatibility
            result_dict['scoring_components']['has_history'] = song_key in (
                self.ranking_engine.track_stats if self.ranking_engine.track_stats else {}
            )
            
            results.append(result_dict)
        
        return results, total_count
    
    def search_songs_by_text(self, query: str, limit: int = 10) -> List[Tuple[int, float, str]]:
        """Search for songs using text similarity (for song-to-song search suggestions)."""
        if not query or not self.tfidf_vectorizer:
            return []
        
        return data_utils.search_songs_by_text(
            query, self.tfidf_vectorizer, self.tfidf_matrix, self.songs, limit
        )
    
    def get_ranking_weights(self) -> Dict:
        """Get current V2.5 ranking weights for display."""
        config = self.ranking_engine.config
        weights = config.to_dict()
        weights.update({
            'version': '2.5',
            'has_history': self.has_history,
            'history_songs_count': len(self.ranking_engine.track_stats) if self.ranking_engine.track_stats else 0
        })
        return weights


# Initialize search engine
search_engine = None

def init_search_engine(songs_file: str = None, embeddings_file: str = None, history_path: str = None):
    """Initialize the search engine with data files."""
    global search_engine
    if search_engine is None:
        # Use provided file paths or fall back to parsed arguments or defaults
        if songs_file and embeddings_file:
            songs_path = songs_file
            embeddings_path = embeddings_file
            history_path_arg = history_path
        elif 'args' in globals() and args:
            songs_path = songs_file or args.songs
            embeddings_path = embeddings_file or args.embeddings
            history_path_arg = history_path or args.history
        else:
            # Fallback defaults
            default_songs = Path(__file__).parent.parent / constants.DEFAULT_SONGS_FILE
            default_embeddings = Path(__file__).parent.parent / constants.DEFAULT_EMBEDDINGS_PATH
            songs_path = songs_file or str(default_songs)
            embeddings_path = embeddings_file or str(default_embeddings)
            history_path_arg = history_path
        
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
        
        # Create search engine
        search_engine = MusicSearchEngine(songs_path, embeddings_path, history_path_arg)


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
            'page_title': 'Semantic Song Search',
            'is_new_session': True
        })
    
    # Pass debug flag to template  
    debug_mode = getattr(args, 'debug', False) if 'args' in globals() and args else False
    return render_template('index.html', debug_mode=debug_mode)

@app.route('/api/search', methods=['POST'])
def search():
    """Main search endpoint with personalized ranking."""
    if search_engine is None:
        return jsonify({'error': 'Search engine not initialized'}), 500
    
    start_time = datetime.now()
    
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        search_type = data.get('type', 'text')
        limit = int(data.get('limit', 20))
        offset = int(data.get('offset', 0))
        song_idx = data.get('song_idx')
        discovery_slider = float(data.get('discovery_slider', 0.5))  # Discovery parameter
        discovery_slider = np.clip(discovery_slider, 0.0, 1.0)  # Ensure valid range
        
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
                query_embedding, k=limit, offset=offset, discovery_slider=discovery_slider
            )
        
        elif search_type == 'song':
            # Song-to-song search
            if song_idx is None:
                return jsonify({'error': 'song_idx is required for song search'}), 400
            
            if song_idx >= len(search_engine.songs):
                return jsonify({'error': 'Invalid song_idx'}), 400
            
            reference_song = search_engine.songs[song_idx]
            song_key = (reference_song['original_song'], reference_song['original_artist'])
            
            if song_key in search_engine.embedding_lookup:
                query_embedding = search_engine.embedding_lookup[song_key]
                results, total_count = search_engine.similarity_search(
                    query_embedding, k=limit, offset=offset, discovery_slider=discovery_slider
                )
            else:
                return jsonify({'error': 'No embedding available for reference song'}), 400
        
        else:
            return jsonify({'error': 'Invalid search type'}), 400
        
        # Calculate search performance
        search_duration = (datetime.now() - start_time).total_seconds()
        
        # Track successful search with enhanced context
        search_properties = {
            'search_type': search_type,
            'embed_type': 'full_profile',  # V2 uses full_profile embedding by default
            'results_returned': total_count,
            'results_requested': limit,
            'search_offset': offset,
            'search_duration_seconds': round(search_duration, 3),
            'is_paginated_search': offset > 0,
            'returned_count': len(results),
            'discovery_slider': discovery_slider
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
            'embed_type': 'full_profile',
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
            'search_type': search_type,
            'embed_type': 'full_profile',
            'query_length': len(query_text) if query_text else 0,
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
    if os.getenv('RAILWAY_ENVIRONMENT') or request.host not in ['127.0.0.1:5000', 'localhost:5000']:
        # Production: use the current request host with HTTPS
        redirect_uri = f"https://{request.host}/callback"
    else:
        # Local development: use localhost with HTTP
        redirect_uri = f"http://{constants.DEFAULT_HOST}:{constants.DEFAULT_PORT}/callback"
    
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
        playlist_name = data.get('playlist_name', 'Semantic Song Search Playlist')
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
            description=f"Created by Semantic Song Search - {len(valid_songs)} songs"
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
            'artists': artists_list[:50],  # Limit to top 50
            'time_ranges': time_ranges
        })
        
    except Exception as e:
        logger.error(f"Error getting top artists: {e}")
        return jsonify({'error': 'Failed to retrieve top artists'}), 500

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
        init_search_engine(args.songs, args.embeddings, args.history)
        logger.info("Search engine initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        logger.error("Please check your data file paths and try again.")
        exit(1)
    
    # Start the Flask application
    app.run(debug=args.debug, host=args.host, port=args.port)