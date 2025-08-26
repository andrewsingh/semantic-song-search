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
DATASET_NAME = os.getenv('DATASET_NAME', 'library_v2.1')

def main():
    """Initialize the app with production data files and start the server."""
    
    # Get port from environment (Railway/Render/Heroku style)
    port = int(os.environ.get('PORT', 5000))
    
    # Convert to Path objects for proper path operations
    volume_path = Path(RAILWAY_VOLUME_MOUNT_PATH)
    dataset_path = volume_path / DATASET_NAME
    
    # Use eval_set_v2 data files (adjust paths as needed for your deployment)
    songs_file = dataset_path / 'library_v2.1_metadata_with_streams.json'
    
    # Verify files exist before starting
    if not songs_file.exists():
        print(f"Error: Songs file not found: {songs_file}")
        sys.exit(1)
    
    # Check for embeddings (either combined file or directory)
    embeddings_npz = dataset_path / 'library_v2.1_embeddings.npz'
    embeddings_dir = dataset_path / 'library_v2.1_embeddings'
    
    if embeddings_npz.exists():
        embeddings_path = embeddings_npz
        print(f"Using combined embeddings file: {embeddings_npz}")
    elif embeddings_dir.exists():
        embeddings_path = embeddings_dir
        print(f"Using separate embeddings directory: {embeddings_dir}")
    else:
        print(f"Error: No embeddings found at {embeddings_npz} or {embeddings_dir}")
        sys.exit(1)
    
    print(f"Initializing app with:")
    print(f"  Songs: {songs_file}")
    print(f"  Embeddings: {embeddings_path}")
    
    # Initialize search engine with correct paths
    init_search_engine(str(songs_file), str(embeddings_path))
    
    # Start the app
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    main()