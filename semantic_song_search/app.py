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
import time
from search import MusicSearchEngine, SearchConfig

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



# Initialize search engine and optional ranking engine
search_engine = None
ranking_engine = None

def init_search_engine(songs_file: str = None, embeddings_file: str = None, history_path: str = None, artist_embeddings_file: str = None, shared_genre_store_file: str = None, profiles_file: str = None):
    """Initialize the search engine with data files."""
    global search_engine, ranking_engine
    if search_engine is None:
        # Use provided file paths or fall back to parsed arguments or defaults
        if songs_file and embeddings_file:
            songs_path = songs_file
            embeddings_path = embeddings_file
            history_path_arg = history_path
            artist_embeddings_path = artist_embeddings_file
            shared_genre_store_path = shared_genre_store_file
        elif 'args' in globals() and args:
            songs_path = songs_file or args.songs
            embeddings_path = embeddings_file or args.embeddings
            history_path_arg = history_path or args.history
            artist_embeddings_path = artist_embeddings_file or getattr(args, 'artist_embeddings', constants.DEFAULT_ARTIST_EMBEDDINGS_PATH)
            shared_genre_store_path = shared_genre_store_file or getattr(args, 'shared_genre_store', constants.DEFAULT_SHARED_GENRE_STORE_PATH)
        else:
            # Fallback defaults
            default_songs = Path(__file__).parent.parent / constants.DEFAULT_SONGS_FILE
            default_embeddings = Path(__file__).parent.parent / constants.DEFAULT_EMBEDDINGS_PATH
            default_artist_embeddings = (Path(__file__).parent.parent / constants.DEFAULT_ARTIST_EMBEDDINGS_PATH) if constants.DEFAULT_ARTIST_EMBEDDINGS_PATH else None
            default_shared_genre_store = (Path(__file__).parent.parent / constants.DEFAULT_SHARED_GENRE_STORE_PATH) if constants.DEFAULT_SHARED_GENRE_STORE_PATH else None
            songs_path = songs_file or str(default_songs)
            embeddings_path = embeddings_file or str(default_embeddings)
            history_path_arg = history_path
            artist_embeddings_path = artist_embeddings_file or (str(default_artist_embeddings) if default_artist_embeddings else None)
            shared_genre_store_path = shared_genre_store_file or (str(default_shared_genre_store) if default_shared_genre_store else None)
        
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
        if artist_embeddings_path and Path(artist_embeddings_path).exists():
            logger.info(f"  Artist embeddings: {artist_embeddings_path}")
        else:
            logger.info(f"  Artist embeddings: None (artist similarity disabled)")
            artist_embeddings_path = None  # Don't pass invalid path to search engine
        
        if shared_genre_store_path and Path(shared_genre_store_path).exists():
            logger.info(f"  Shared genre store: {shared_genre_store_path}")
        else:
            logger.info(f"  Shared genre store: None (using fallback genre loading)")
            shared_genre_store_path = None  # Will use fallback loading
        
        # Create search engine (no history - clean separation)
        profiles_path = profiles_file or getattr(constants, 'DEFAULT_PROFILES_FILE', None)
        search_config = SearchConfig()
        search_engine = MusicSearchEngine(songs_path, embeddings_path, artist_embeddings_path, shared_genre_store_path, profiles_path, config=search_config)

        # Store history path for ranking engine initialization if needed
        search_engine._history_path = history_path_arg

        # Initialize ranking engine if history is available
        global ranking_engine
        if history_path_arg:
            try:
                ranking_engine = init_ranking_engine(history_path_arg)
                logger.info("Initialized ranking engine for personalized search")
            except Exception as e:
                logger.warning(f"Failed to initialize ranking engine: {e}")
                ranking_engine = None
        else:
            ranking_engine = None
            logger.info("No history path provided - using semantic search only")


def init_ranking_engine(history_path: str):
    """Initialize ranking engine with history data."""
    if not search_engine:
        raise ValueError("Search engine must be initialized first")

    # Load and process history
    history_df, success = data_utils.load_and_process_spotify_history(Path(history_path))

    if not success or history_df.empty:
        raise ValueError("Failed to load history data")

    # Filter to known songs
    history_df = data_utils.filter_history_to_known_songs(history_df, search_engine.songs)

    if len(history_df) == 0:
        raise ValueError("No history entries match known songs")

    # Initialize ranking engine with history and stream scores from search engine
    ranking_config = ranking.RankingConfig()
    engine = ranking.initialize_ranking_engine(
        history_df, search_engine.songs, search_engine.embedding_lookups, ranking_config, search_engine.track_streams
    )

    logger.info(f"Initialized ranking engine with {len(history_df)} history entries")
    return engine


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
            'page_title': 'SongMatch',
            'is_new_session': True
        })
    
    # Pass debug flag and default config values to template
    debug_mode = getattr(args, 'debug', False) if 'args' in globals() and args else False
    default_config = SearchConfig()
    return render_template('index.html', debug_mode=debug_mode, default_config=default_config)

@app.route('/api/search', methods=['POST'])
def search():
    """Main search endpoint with personalized ranking."""
    if search_engine is None:
        return jsonify({'error': 'Search engine not initialized'}), 500
    
    start_time = datetime.now()
    
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        search_type = data.get('search_type', data.get('type', 'text'))  # Accept both for compatibility
        # embed_type removed - using all descriptors simultaneously
        limit = int(data.get('k', data.get('limit', 20)))  # Accept both 'k' and 'limit'
        offset = int(data.get('offset', 0))
        song_idx = data.get('song_idx')
        
        # Personalization parameters
        lambda_val = float(data.get('lambda_val', 0.5))  # Semantic vs personal weight
        lambda_val = np.clip(lambda_val, 0.0, 1.0)  # Ensure valid range
        familiarity_min = float(data.get('familiarity_min', 0.0))  # Min familiarity
        familiarity_max = float(data.get('familiarity_max', 1.0))  # Max familiarity
        familiarity_min = np.clip(familiarity_min, 0.0, 1.0)
        familiarity_max = np.clip(familiarity_max, 0.0, 1.0)
        
        # Ensure min <= max
        if familiarity_min > familiarity_max:
            familiarity_min, familiarity_max = familiarity_max, familiarity_min
        
        # Extract advanced ranking parameters
        advanced_params = {}
        # List of valid advanced parameter names (matching RankingConfig)
        valid_advanced_params = {
            'H_c', 'H_E', 'gamma_s', 'gamma_f', 'kappa', 'alpha_0', 'beta_0', 'K_s',
            'K_E', 'gamma_A', 'eta', 'tau', 'beta_f', 'K_life', 'K_recent', 'psi',
            'k_neighbors', 'sigma', 'knn_embed_type', 'beta_p', 'beta_s', 'beta_a',
            'kappa_E', 'theta_c', 'tau_c', 'K_c', 'tau_K', 'M_A', 'K_fam', 'R_min',
            'C_fam', 'min_plays',
            # New 9-weight system parameters
            'a0_song_sim', 'a1_artist_sim', 'a2_total_streams', 'a3_daily_streams',
            'b0_genres', 'b1_vocal_style', 'b2_production_sound_design', 'b3_lyrical_meaning', 'b4_mood_atmosphere'
        }
        
        for param_name in valid_advanced_params:
            if param_name in data:
                advanced_params[param_name] = data[param_name]
        
        logger.info(f"🔧 Advanced parameters received: {advanced_params}")
        logger.info(f"🔧 Lambda value: {lambda_val}")
        logger.info(f"🔧 Familiarity range: [{familiarity_min}, {familiarity_max}]")
        
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
                query_embedding, k=limit, offset=offset,
                lambda_val=lambda_val, familiarity_min=familiarity_min, familiarity_max=familiarity_max,
                ranking_engine=ranking_engine,
                **advanced_params
            )
        
        elif search_type == 'song':
            # Song-to-song search
            if song_idx is None:
                return jsonify({'error': 'song_idx is required for song search'}), 400
            
            try:
                if song_idx >= len(search_engine.songs):
                    return jsonify({'error': 'Invalid song_idx'}), 400
                
                reference_song = search_engine.songs[song_idx]
                track_id = reference_song.get('track_id') or reference_song.get('id')
                
                if not track_id:
                    return jsonify({'error': 'Reference song missing track_id'}), 400
                
                logger.info(f"Song-to-song search for track_id: {track_id} using all descriptors")
                
                # For song-to-song search, we pass any single embedding as query_embedding
                # The similarity_search method will internally build the complete descriptor dict
                query_embedding = None
                for embed_type in constants.SONG_EMBEDDING_TYPES:
                    embedding_lookup = search_engine.embedding_lookups.get(embed_type, {})
                    if embedding_lookup and track_id in embedding_lookup:
                        query_embedding = embedding_lookup[track_id]
                        logger.info(f"Found query embedding for track_id: {track_id}, using {embed_type}, shape: {query_embedding.shape}")
                        break
                
                if query_embedding is None:
                    logger.error(f"No embeddings found for track_id: {track_id}")
                    return jsonify({'error': 'No embeddings available for reference song'}), 400
                
                logger.info(f"Starting similarity search with {len(search_engine.songs)} total songs")
                results, total_count = search_engine.similarity_search(
                    query_embedding, k=limit, offset=offset,
                    lambda_val=lambda_val, familiarity_min=familiarity_min, familiarity_max=familiarity_max,
                    query_track_id=track_id,  # Enable song-to-song artist similarities
                    ranking_engine=ranking_engine,
                    **advanced_params
                )
                logger.info(f"Similarity search completed, found {total_count} results")
            
            except Exception as e:
                logger.error(f"Error in song-to-song search: {str(e)}", exc_info=True)
                return jsonify({'error': f'Song search failed: {str(e)}'}), 500
        
        else:
            return jsonify({'error': 'Invalid search type'}), 400
        
        # Calculate search performance
        search_duration = (datetime.now() - start_time).total_seconds()
        
        # Track successful search with enhanced context
        search_properties = {
            'search_type': search_type,
            'descriptors': 'all',  # Using all descriptors simultaneously
            'results_returned': total_count,
            'results_requested': limit,
            'search_offset': offset,
            'search_duration_seconds': round(search_duration, 3),
            'is_paginated_search': offset > 0,
            'returned_count': len(results),
            'lambda_val': lambda_val,
            'familiarity_min': familiarity_min,
            'familiarity_max': familiarity_max
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
            'descriptors': 'all',  # Using all descriptors simultaneously
            'query': query_text,
            'search_weights': search_engine.get_search_weights(),
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
            'search_type': search_type if 'search_type' in locals() else 'unknown',
            'descriptors': 'all',  # Using all descriptors simultaneously
            'query_length': len(query_text) if 'query_text' in locals() and query_text else 0,
            'error_message': str(e)[:200],
            'search_duration_seconds': round(search_duration, 3)
        })
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

@app.route('/api/search_weights', methods=['GET'])
def get_search_weights():
    """Get current search weights and configuration."""
    if search_engine is None:
        return jsonify({'error': 'Search engine not initialized'}), 500

    try:
        weights = search_engine.get_search_weights()
        return jsonify(weights)
    except Exception as e:
        logger.error(f"Failed to get search weights: {e}")
        return jsonify({'error': 'Failed to retrieve search weights'}), 500

@app.route('/api/search_weights', methods=['PUT'])
def update_search_weights():
    """Update search weights."""
    if search_engine is None:
        return jsonify({'error': 'Search engine not initialized'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Update the search engine config
        search_engine.config.update_weights(data)

        logger.info(f"Updated search weights: {data}")

        # Return the updated weights
        weights = search_engine.get_search_weights()
        return jsonify(weights)

    except Exception as e:
        logger.error(f"Failed to update search weights: {e}")
        return jsonify({'error': 'Failed to update search weights'}), 500

# Keep ranking weights endpoints for backward compatibility (but with proper logic)
@app.route('/api/ranking_weights', methods=['GET'])
def get_ranking_weights():
    """Get current ranking weights (legacy endpoint)."""
    if ranking_engine is None:
        # Return search weights as fallback
        return get_search_weights()

    try:
        weights = ranking_engine.config.to_dict()
        weights.update({
            'version': '3.0',
            'has_history': getattr(ranking_engine, 'has_history', False),
            'ranking_config': True
        })
        return jsonify(weights)
    except Exception as e:
        logger.error(f"Failed to get ranking weights: {e}")
        return jsonify({'error': 'Failed to retrieve ranking weights'}), 500

@app.route('/api/ranking_weights', methods=['PUT'])
def update_ranking_weights():
    """Update ranking weights (legacy endpoint)."""
    if ranking_engine is None:
        # Fallback to search weights
        return update_search_weights()

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Update the ranking engine config
        ranking_engine.config.update_weights(data)

        logger.info(f"Updated ranking weights: {data}")

        # Return the updated weights
        weights = ranking_engine.config.to_dict()
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
    # Local development: host is localhost or 127.0.0.1 (any port)
    host_without_port = request.host.split(':')[0]
    is_local = host_without_port in ['127.0.0.1', 'localhost']
    is_production = os.getenv('RAILWAY_ENVIRONMENT') or not is_local
    
    if is_production:
        # Production: use the current request host with HTTPS
        redirect_uri = f"https://{request.host}/callback"
    else:
        # Local development: use current host with HTTP (preserves custom port)
        redirect_uri = f"http://{request.host}/callback"
    
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
        
        # Extract data from new Spotify metadata structure
        track_id = song.get('track_id') or song.get('id', '')
        song_name = song.get('name', song.get('original_song', ''))
        
        # Handle multiple artists
        artists_list = song.get('artists', [])
        primary_artist = artists_list[0]['name'] if artists_list else song.get('original_artist', '')
        
        # Get album info and cover art
        album_name = song.get('album', {}).get('name', '')
        album_images = song.get('album', {}).get('images', [])
        cover_url = album_images[-1]['url'] if album_images else ''
        
        suggestions.append({
            'song_idx': int(song_idx),  # Convert numpy.int64 to native Python int
            'label': label,
            'song': song_name,
            'artist': primary_artist,
            'album': album_name,
            'cover_url': cover_url,
            'spotify_id': track_id,
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
        playlist_name = data.get('playlist_name', 'SongMatch Playlist')
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
            description=f"Created by SongMatch - {len(valid_songs)} songs"
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
    
    # Extract data from new Spotify metadata structure
    track_id = song.get('track_id') or song.get('id', '')
    song_name = song.get('name', song.get('original_song', ''))
    
    # Handle multiple artists
    artists_list = song.get('artists', [])
    primary_artist = artists_list[0]['name'] if artists_list else song.get('original_artist', '')
    all_artists = [artist['name'] for artist in artists_list] if artists_list else [primary_artist]
    
    # Get album info and cover art
    album_name = song.get('album', {}).get('name', '')
    album_images = song.get('album', {}).get('images', [])
    cover_url = album_images[-1]['url'] if album_images else ''
    
    # Get duration
    duration_ms = song.get('duration_ms', 0)
    
    return jsonify({
        'song_idx': song_idx,
        'song': song_name,
        'artist': primary_artist,
        'all_artists': all_artists,  # New: support multiple artists
        'album': album_name,
        'cover_url': cover_url,
        'spotify_id': track_id,
        'duration_ms': duration_ms,  # New: duration in milliseconds
        'genres': search_engine.genres_lookup.get(track_id, []),
        'tags': search_engine.tags_lookup.get(track_id, []),
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
            'artists': artists_list,  # Return all deduplicated artists (no artificial limit)
            'time_ranges': time_ranges
        })
        
    except Exception as e:
        logger.error(f"Error getting top artists: {e}")
        return jsonify({'error': 'Failed to retrieve top artists'}), 500

@app.route('/api/default_search_config')
def get_default_search_config():
    """Get default search configuration parameters."""
    try:
        # Create a default SearchConfig instance and return its parameters
        default_config = SearchConfig()
        return jsonify(default_config.to_dict())
    except Exception as e:
        logger.error(f"Error getting default search config: {e}")
        return jsonify({'error': 'Failed to retrieve default search configuration'}), 500

@app.route('/api/default_ranking_config')
def get_default_ranking_config():
    """Get default ranking configuration parameters (legacy endpoint)."""
    try:
        # For backward compatibility, return search config if no ranking engine
        if ranking_engine is None:
            return get_default_search_config()

        # Create a default RankingConfig instance and return its parameters
        default_config = ranking.RankingConfig()
        return jsonify(default_config.to_dict())
    except Exception as e:
        logger.error(f"Error getting default ranking config: {e}")
        return jsonify({'error': 'Failed to retrieve default ranking configuration'}), 500



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
    parser.add_argument('--artist-embeddings', type=str, default=constants.DEFAULT_ARTIST_EMBEDDINGS_PATH, help=f'Path to artist embeddings directory (default: {constants.DEFAULT_ARTIST_EMBEDDINGS_PATH})')
    parser.add_argument('--shared-genre-store', type=str, default=constants.DEFAULT_SHARED_GENRE_STORE_PATH, help=f'Path to shared genre embedding store file (default: {constants.DEFAULT_SHARED_GENRE_STORE_PATH})')
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
        init_search_engine(args.songs, args.embeddings, args.history, getattr(args, 'artist_embeddings', None), getattr(args, 'shared_genre_store', None))
        logger.info("Search engine initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        logger.error("Please check your data file paths and try again.")
        exit(1)
    
    # Start the Flask application
    app.run(debug=args.debug, host=args.host, port=args.port)