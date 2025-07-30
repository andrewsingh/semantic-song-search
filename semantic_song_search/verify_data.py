#!/usr/bin/env python3
"""
Verify that data files are accessible and properly formatted for the semantic song search app.
"""
import os
import json
import numpy as np
import argparse
from pathlib import Path

def parse_arguments():
    """Parse command line arguments for data file paths."""
    parser = argparse.ArgumentParser(
        description="Verify data files for Semantic Song Search app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_data.py
  python verify_data.py --songs custom_songs.json --embeddings custom_embeddings.npz
  python verify_data.py -s /path/to/songs.json -e /path/to/embeddings.npz
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
    
    return parser.parse_args()

def verify_data_files(songs_file: str, embeddings_file: str):
    """Verify that all required data files exist and are properly formatted."""
    print("🔍 Verifying data files for Semantic Song Search...")
    print(f"  Songs file: {songs_file}")
    print(f"  Embeddings file: {embeddings_file}")
    
    # Check song profiles
    songs_path = Path(songs_file)
    if not songs_path.exists():
        print(f"❌ Song profiles file not found: {songs_path}")
        return False
    
    try:
        with open(songs_path, 'r') as f:
            songs = json.load(f)
        
        print(f"✅ Song profiles loaded: {len(songs)} songs")
        
        # Check first song structure
        sample_song = songs[0]
        required_fields = ['original_song', 'original_artist', 'metadata']
        missing_fields = [field for field in required_fields if field not in sample_song]
        
        if missing_fields:
            print(f"❌ Missing required fields in song profiles: {missing_fields}")
            return False
        
        # Check metadata structure
        metadata = sample_song['metadata']
        required_metadata = ['song_id', 'cover_url', 'album_name']
        missing_metadata = [field for field in required_metadata if field not in metadata]
        
        if missing_metadata:
            print(f"⚠️  Missing recommended metadata fields: {missing_metadata}")
        
        print("✅ Song profiles structure verified")
        
    except Exception as e:
        print(f"❌ Error loading song profiles: {e}")
        return False
    
    # Check embeddings
    embeddings_path = Path(embeddings_file)
    if not embeddings_path.exists():
        print(f"❌ Embeddings file not found: {embeddings_path}")
        return False
    
    try:
        data = np.load(embeddings_path, allow_pickle=True)
        
        required_keys = ['songs', 'artists', 'embeddings', 'song_indices', 'field_types', 'field_values']
        missing_keys = [key for key in required_keys if key not in data.keys()]
        
        if missing_keys:
            print(f"❌ Missing required keys in embeddings file: {missing_keys}")
            return False
        
        print(f"✅ Embeddings loaded: {len(data['embeddings'])} embeddings")
        print(f"   - Embedding dimension: {data['embeddings'].shape[1]}")
        print(f"   - Songs with embeddings: {len(data['songs'])}")
        
        # Check embedding types
        unique_types, counts = np.unique(data['field_types'], return_counts=True)
        print("   - Embedding types:")
        for embed_type, count in zip(unique_types, counts):
            print(f"     * {embed_type}: {count}")
        
        expected_types = {'full_profile', 'sound_aspect', 'meaning_aspect', 'mood_aspect', 'tags_genres'}
        found_types = set(unique_types)
        missing_types = expected_types - found_types
        
        if missing_types:
            print(f"⚠️  Missing expected embedding types: {missing_types}")
        
        print("✅ Embeddings structure verified")
        
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False
    
    # Check data consistency
    try:
        song_count_profiles = len(songs)
        song_count_embeddings = len(data['songs'])
        
        if song_count_profiles != song_count_embeddings:
            print(f"⚠️  Song count mismatch: {song_count_profiles} profiles vs {song_count_embeddings} embeddings")
        else:
            print(f"✅ Data consistency verified: {song_count_profiles} songs")
        
    except Exception as e:
        print(f"❌ Error checking data consistency: {e}")
        return False
    
    print("\n🎉 All data verification checks passed!")
    print("The semantic song search app should work correctly with this data.")
    
    return True

def check_environment():
    """Check that required environment variables are set."""
    print("\n🔧 Checking environment variables...")
    
    required_vars = ['SPOTIFY_CLIENT_ID', 'SPOTIFY_CLIENT_SECRET', 'OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            # Show first few characters for verification without exposing full key
            value = os.getenv(var)
            masked_value = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
            print(f"✅ {var}: {masked_value}")
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("\nPlease set these environment variables:")
        for var in missing_vars:
            print(f"export {var}='your_{var.lower()}'")
        return False
    
    print("✅ All required environment variables are set")
    return True

if __name__ == "__main__":
    print("🎵 Semantic Song Search - Data Verification\n")
    
    args = parse_arguments()
    
    data_ok = verify_data_files(args.songs, args.embeddings)
    env_ok = check_environment()
    
    if data_ok and env_ok:
        print("\n🚀 Ready to run the semantic song search app!")
        print("Run: python app.py")
        if args.songs != str(Path(__file__).parent.parent / 'pop_eval_set_v0' / 'pop_eval_set_v0_results_enriched.json') or \
           args.embeddings != str(Path(__file__).parent.parent / 'pop_eval_set_v0' / 'pop_eval_set_v0_embeddings.npz'):
            print(f"With custom files: python app.py --songs '{args.songs}' --embeddings '{args.embeddings}'")
    else:
        print("\n❌ Please fix the issues above before running the app.")
        exit(1) 