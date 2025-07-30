#!/usr/bin/env python3
"""
Split combined embeddings file into separate files by embedding type.

This script takes a combined embeddings .npz file (with all 5 embedding types)
and splits it into 5 separate .npz files, one for each embedding type:
- full_profile_embeddings.npz
- sound_aspect_embeddings.npz  
- meaning_aspect_embeddings.npz
- mood_aspect_embeddings.npz
- tags_genres_embeddings.npz

Usage:
  python split_embeddings.py -i combined_embeddings.npz -o output_directory/
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

def load_combined_embeddings(input_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the combined embeddings file."""
    print(f"Loading combined embeddings from: {input_file}")
    
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    data = np.load(input_file, allow_pickle=True)
    
    # Validate required fields
    required_fields = ['songs', 'artists', 'embeddings', 'song_indices', 'field_types', 'field_values']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields in input file: {missing_fields}")
    
    songs = data['songs']
    artists = data['artists']
    embeddings = data['embeddings']
    song_indices = data['song_indices']
    field_types = data['field_types']
    field_values = data['field_values']
    
    print(f"Loaded {len(songs)} songs with {len(embeddings)} total embeddings")
    
    # Show breakdown by field type
    unique_types, counts = np.unique(field_types, return_counts=True)
    print(f"Embedding breakdown:")
    for field_type, count in zip(unique_types, counts):
        print(f"  {field_type}: {count} embeddings")
    
    return songs, artists, embeddings, song_indices, field_types, field_values

def split_and_save_embeddings(songs: np.ndarray, artists: np.ndarray, 
                             embeddings: np.ndarray, song_indices: np.ndarray,
                             field_types: np.ndarray, field_values: np.ndarray,
                             output_dir: str):
    """Split embeddings by type and save to separate files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define the 5 embedding types
    embedding_types = ['full_profile', 'sound_aspect', 'meaning_aspect', 'mood_aspect', 'tags_genres']
    
    print(f"\nSplitting embeddings into separate files...")
    print(f"Output directory: {output_path}")
    
    created_files = []
    
    for embed_type in embedding_types:
        # Find embeddings of this type
        mask = field_types == embed_type
        
        if not mask.any():
            print(f"  Warning: No embeddings found for type '{embed_type}', skipping...")
            continue
        
        # Extract embeddings for this type
        type_embeddings = embeddings[mask]
        type_song_indices = song_indices[mask]
        type_field_values = field_values[mask]
        
        # Create output filename
        output_file = output_path / f"{embed_type}_embeddings.npz"
        
        # Save to separate file
        np.savez_compressed(
            output_file,
            # Song metadata (same for all files)
            songs=songs,
            artists=artists,
            # Embedding data (filtered for this type)
            embeddings=type_embeddings,
            song_indices=type_song_indices,
            field_values=type_field_values
        )
        
        created_files.append(output_file)
        print(f"  ✅ {embed_type}: {len(type_embeddings)} embeddings → {output_file}")
    
    return created_files

def verify_split_files(created_files: list, original_songs: np.ndarray, 
                      original_embeddings: np.ndarray):
    """Verify that the split files contain the same data as the original."""
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    
    total_embeddings_in_splits = 0
    
    for file_path in created_files:
        data = np.load(file_path, allow_pickle=True)
        embed_count = len(data['embeddings'])
        total_embeddings_in_splits += embed_count
        
        # Verify songs metadata is consistent
        if not np.array_equal(data['songs'], original_songs):
            print(f"  ❌ Songs metadata mismatch in {file_path.name}")
        else:
            print(f"  ✅ {file_path.name}: {embed_count} embeddings, songs metadata consistent")
    
    # Check total embedding count
    if total_embeddings_in_splits == len(original_embeddings):
        print(f"  ✅ Total embeddings: {total_embeddings_in_splits} (matches original)")
    else:
        print(f"  ❌ Total embeddings: {total_embeddings_in_splits} (original had {len(original_embeddings)})")
    
    print(f"\nAll files saved to: {created_files[0].parent}")

def main():
    parser = argparse.ArgumentParser(
        description="Split combined embeddings file into separate files by embedding type",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python split_embeddings.py -i pop_eval_set_v0_embeddings.npz -o embeddings_split/
  python split_embeddings.py --input combined.npz --output /path/to/output/dir/
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to combined embeddings .npz file'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True, 
        help='Directory to save the 5 separate embedding files'
    )
    
    args = parser.parse_args()
    
    try:
        # Load combined embeddings
        songs, artists, embeddings, song_indices, field_types, field_values = load_combined_embeddings(args.input)
        
        # Split and save
        created_files = split_and_save_embeddings(
            songs, artists, embeddings, song_indices, field_types, field_values, args.output
        )
        
        # Verify the split
        verify_split_files(created_files, songs, embeddings)
        
        print(f"\n{'='*60}")
        print("SPLIT COMPLETE!")
        print(f"{'='*60}")
        print(f"Created {len(created_files)} embedding files:")
        for file_path in created_files:
            print(f"  - {file_path}")
        print(f"\nYou can now use these separate files with the updated embed_song_profiles.py script.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())