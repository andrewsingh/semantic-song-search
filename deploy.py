#!/usr/bin/env python3
"""
Deployment entry point for the Semantic Song Search app.
Uses the correct data files for production deployment.
"""
import os
import sys
from pathlib import Path

# Add the semantic_song_search directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'semantic_song_search'))

from app import app, init_search_engine

RAILWAY_VOLUME_MOUNT_PATH = os.getenv('RAILWAY_VOLUME_MOUNT_PATH', '/data')
DATASET_NAME = os.getenv('DATASET_NAME', 'library_v3.1')

def main():
    """Initialize the app with production data files and start the server."""

    # Get port from environment (Railway/Render/Heroku style)
    port = int(os.environ.get('PORT', 5000))

    # Convert to Path objects for proper path operations
    volume_path = Path(RAILWAY_VOLUME_MOUNT_PATH)
    dataset_path = volume_path / DATASET_NAME

    # Use eval_set_v2 data files (adjust paths as needed for your deployment)
    songs_file = dataset_path / f'{DATASET_NAME}_metadata_with_streams.json'

    # Verify files exist before starting
    if not songs_file.exists():
        print(f"Error: Songs file not found: {songs_file}")
        sys.exit(1)

    # Check for embeddings (either combined file or directory)
    embeddings_npz = dataset_path / f'{DATASET_NAME}_embeddings.npz'
    embeddings_dir = dataset_path / f'{DATASET_NAME}_embeddings'

    if embeddings_npz.exists():
        embeddings_path = embeddings_npz
        print(f"Using combined embeddings file: {embeddings_npz}")
    elif embeddings_dir.exists():
        embeddings_path = embeddings_dir
        print(f"Using separate embeddings directory: {embeddings_dir}")
    else:
        print(f"Error: No embeddings found at {embeddings_npz} or {embeddings_dir}")
        sys.exit(1)

    # Check for artist embeddings (optional)
    artist_embeddings_path = dataset_path / f'{DATASET_NAME}_artist_embeddings'
    if not artist_embeddings_path.exists():
        print(f"Warning: Artist embeddings not found at {artist_embeddings_path} - artist similarity will be disabled")
        artist_embeddings_path = None
    else:
        print(f"Found artist embeddings: {artist_embeddings_path}")

    # Check for shared genre store (optional)
    shared_genre_store_path = volume_path / 'genre_embedding_store.npz'
    if not shared_genre_store_path.exists():
        print(f"Warning: Shared genre store not found at {shared_genre_store_path} - using fallback genre loading")
        shared_genre_store_path = None
    else:
        print(f"Found shared genre store: {shared_genre_store_path}")

    # Check for profiles file (optional)
    profiles_file = dataset_path / f'{DATASET_NAME}_profiles_v4.2.jsonl'
    if not profiles_file.exists():
        print(f"Warning: Profiles file not found at {profiles_file} - tags and genres will not be available")
        profiles_file = None
    else:
        print(f"Found profiles file: {profiles_file}")

    print(f"Initializing app with:")
    print(f"  Songs: {songs_file}")
    print(f"  Embeddings: {embeddings_path}")
    print(f"  Artist embeddings: {artist_embeddings_path or 'None'}")
    print(f"  Shared genre store: {shared_genre_store_path or 'None'}")
    print(f"  Profiles file: {profiles_file or 'None'}")

    # Initialize search engine with correct paths
    init_search_engine(
        str(songs_file),
        str(embeddings_path),
        history_path=None,  # No history in production deployment
        artist_embeddings_file=str(artist_embeddings_path) if artist_embeddings_path else None,
        shared_genre_store_file=str(shared_genre_store_path) if shared_genre_store_path else None,
        profiles_file=str(profiles_file) if profiles_file else None
    )

    # Start the app
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    main()