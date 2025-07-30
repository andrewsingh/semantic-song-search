#!/usr/bin/env python3
"""
Merge and deduplicate song embedding datasets from multiple pairs of .npz and .json files.

This script loads pairs of embedding (.npz) and metadata (.json) files, combines them,
deduplicates songs using song name + artist name as the key, and outputs both a merged
.npz file and a merged .json file with perfectly aligned indices.

Usage:
    python merge_embeddings.py -i file1.npz file1.json file2.npz file2.json -o merged
"""

import argparse
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import logging
import os
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_input_pairs(input_files: List[str]) -> List[Tuple[str, str]]:
    """Parse input files into (npz, json) pairs."""
    if len(input_files) % 2 != 0:
        raise ValueError(f"Input files must be provided in pairs (npz, json). Got {len(input_files)} files.")
    
    pairs = []
    for i in range(0, len(input_files), 2):
        npz_file = input_files[i]
        json_file = input_files[i + 1]
        
        # Validate file extensions
        if not npz_file.endswith('.npz'):
            raise ValueError(f"Expected .npz file, got: {npz_file}")
        if not json_file.endswith('.json'):
            raise ValueError(f"Expected .json file, got: {json_file}")
        
        # Validate files exist
        if not Path(npz_file).exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_file}")
        if not Path(json_file).exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        
        pairs.append((npz_file, json_file))
    
    logger.info(f"Parsed {len(pairs)} input pairs: {pairs}")
    return pairs

def load_json_metadata(file_path: str) -> List[Dict]:
    """Load song metadata from JSON file."""
    logger.info(f"Loading JSON metadata from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"JSON metadata must be a list of song objects, got {type(data)}")
    
    # Validate required fields in metadata
    for i, song in enumerate(data):
        if not isinstance(song, dict):
            raise ValueError(f"Song {i} in JSON {file_path} is not a dictionary: {type(song)}")
        
        required_fields = ['original_song', 'original_artist']
        missing_fields = [field for field in required_fields if field not in song]
        
        if missing_fields:
            raise ValueError(f"Song {i} in {file_path} missing required fields: {missing_fields}")
    
    logger.info(f"Loaded {len(data)} songs from JSON metadata")
    return data

def load_embedding_dataset(file_path: str) -> Dict:
    """Load embedding dataset from .npz file."""
    logger.info(f"Loading embedding dataset from {file_path}")
    
    data = np.load(file_path, allow_pickle=True)
    
    # Validate required fields
    required_fields = ['songs', 'artists', 'embeddings', 'song_indices', 'field_types']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        raise ValueError(f"Missing required fields in {file_path}: {missing_fields}")
    
    # Validate array lengths and consistency
    songs = data['songs']
    artists = data['artists']
    embeddings = data['embeddings']
    song_indices = data['song_indices']
    field_types = data['field_types']
    
    if len(songs) != len(artists):
        raise ValueError(f"Songs and artists arrays have mismatched lengths in {file_path}: {len(songs)} vs {len(artists)}")
    
    if len(embeddings) != len(song_indices) or len(embeddings) != len(field_types):
        raise ValueError(f"Embeddings, song_indices, and field_types have mismatched lengths in {file_path}: "
                        f"{len(embeddings)} vs {len(song_indices)} vs {len(field_types)}")
    
    # Validate song_indices are within bounds
    if len(song_indices) > 0:
        max_song_idx = np.max(song_indices)
        min_song_idx = np.min(song_indices)
        if max_song_idx >= len(songs) or min_song_idx < 0:
            raise ValueError(f"song_indices out of bounds in {file_path}: "
                           f"indices range [{min_song_idx}, {max_song_idx}] but only {len(songs)} songs available")
    
    # Validate embeddings have consistent dimensions
    if len(embeddings) > 0 and embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array in {file_path}, got shape {embeddings.shape}")
    
    # Handle optional field_values (for backwards compatibility)
    if 'field_values' not in data:
        logger.warning(f"No field_values found in {file_path}, creating placeholder values")
        field_values = np.array(['N/A'] * len(embeddings), dtype=object)
    else:
        field_values = data['field_values']
        if len(field_values) != len(embeddings):
            raise ValueError(f"field_values length mismatch in {file_path}: {len(field_values)} vs {len(embeddings)}")
    
    dataset = {
        'songs': songs,
        'artists': artists, 
        'embeddings': embeddings,
        'song_indices': song_indices,
        'field_types': field_types,
        'field_values': field_values,
        'source_file': file_path
    }
    
    logger.info(f"Loaded {len(dataset['songs'])} songs with {len(dataset['embeddings'])} embeddings")
    return dataset

def validate_npz_json_alignment(npz_dataset: Dict, json_metadata: List[Dict], npz_file: str, json_file: str):
    """Validate that NPZ and JSON files are properly aligned."""
    npz_songs = len(npz_dataset['songs'])
    json_songs = len(json_metadata)
    
    if npz_songs != json_songs:
        raise ValueError(f"Mismatched song counts between {npz_file} ({npz_songs} songs) and {json_file} ({json_songs} songs)")
    
    # Check that song names and artists match
    mismatches = []
    for i in range(min(npz_songs, json_songs)):
        npz_song = str(npz_dataset['songs'][i])
        npz_artist = str(npz_dataset['artists'][i])
        json_song = json_metadata[i]['original_song']
        json_artist = json_metadata[i]['original_artist']
        
        if npz_song != json_song or npz_artist != json_artist:
            mismatches.append({
                'index': i,
                'npz': f"'{npz_song}' by '{npz_artist}'",
                'json': f"'{json_song}' by '{json_artist}'"
            })
    
    if mismatches:
        logger.error(f"Song mismatches found between {npz_file} and {json_file}:")
        for mismatch in mismatches[:5]:  # Show first 5 mismatches
            logger.error(f"  Index {mismatch['index']}: NPZ={mismatch['npz']}, JSON={mismatch['json']}")
        if len(mismatches) > 5:
            logger.error(f"  ... and {len(mismatches) - 5} more mismatches")
        raise ValueError(f"NPZ and JSON files are not aligned: {npz_file} and {json_file}")
    
    logger.info(f"✅ NPZ and JSON files are properly aligned: {npz_file} ↔ {json_file}")

def create_song_key(song_name: str, artist_name: str) -> str:
    """Create a normalized key for song deduplication."""
    # Handle None or non-string values gracefully
    song_name = str(song_name) if song_name is not None else ""
    artist_name = str(artist_name) if artist_name is not None else ""
    
    # Normalize by converting to lowercase and stripping whitespace
    normalized_song = song_name.lower().strip()
    normalized_artist = artist_name.lower().strip()
    
    return f"{normalized_song}||{normalized_artist}"

def merge_and_deduplicate_datasets(dataset_pairs: List[Tuple[Dict, List[Dict]]]) -> Tuple[Dict, List[Dict]]:
    """Merge multiple dataset pairs, deduplicate, and return (merged_npz_data, merged_json_data)."""
    logger.info(f"Merging and deduplicating {len(dataset_pairs)} dataset pairs")
    
    if not dataset_pairs:
        raise ValueError("No dataset pairs provided for merging")
    
    # Validate embedding dimensions are consistent across all datasets
    embedding_dims = []
    for i, (npz_dataset, _) in enumerate(dataset_pairs):
        if len(npz_dataset['embeddings']) > 0:
            embedding_dims.append(npz_dataset['embeddings'].shape[1])
        else:
            logger.warning(f"Dataset pair {i + 1} has no embeddings")
    
    if embedding_dims:
        unique_dims = set(embedding_dims)
        if len(unique_dims) > 1:
            raise ValueError(f"Inconsistent embedding dimensions across datasets: {unique_dims}")
        logger.info(f"All datasets have consistent embedding dimension: {embedding_dims[0]}")
    
    # Collect all data organized by song key
    song_key_to_data = {}  # key -> (json_metadata, embeddings_data, source_pair_idx)
    song_key_to_embeddings = defaultdict(list)  # key -> list of (embedding, field_type, field_value, source_pair_idx)
    
    duplication_stats = defaultdict(int)
    
    # Process each dataset pair
    for pair_idx, (npz_dataset, json_metadata) in enumerate(dataset_pairs):
        logger.info(f"Processing dataset pair {pair_idx + 1}")
        
        # Group embeddings by song index
        song_to_embeddings = defaultdict(list)
        for emb_idx, song_idx in enumerate(npz_dataset['song_indices']):
            if 0 <= song_idx < len(npz_dataset['songs']):
                song_to_embeddings[song_idx].append(emb_idx)
            else:
                logger.error(f"Invalid song_idx {song_idx} in dataset pair {pair_idx + 1}, skipping embedding")
        
        # Process each song in this dataset
        for song_idx in range(len(npz_dataset['songs'])):
            # Get song info from both NPZ and JSON
            npz_song = str(npz_dataset['songs'][song_idx])
            npz_artist = str(npz_dataset['artists'][song_idx])
            
            if song_idx < len(json_metadata):
                json_song_data = json_metadata[song_idx]
                song_key = create_song_key(npz_song, npz_artist)
                
                # Skip songs with empty names
                if not npz_song.strip() or not npz_artist.strip():
                    logger.warning(f"Skipping song with empty name/artist in pair {pair_idx + 1}: '{npz_song}' by '{npz_artist}'")
                    continue
                
                # Handle deduplication
                if song_key not in song_key_to_data:
                    # First occurrence - keep this song
                    song_key_to_data[song_key] = (json_song_data, npz_dataset, pair_idx)
                    duplication_stats[f"pair_{pair_idx + 1}_chosen"] += 1
                else:
                    # Duplicate - keep the first occurrence
                    existing_pair_idx = song_key_to_data[song_key][2]
                    duplication_stats[f"pair_{existing_pair_idx + 1}_over_pair_{pair_idx + 1}"] += 1
                    logger.debug(f"Duplicate song found: '{npz_song}' by '{npz_artist}' "
                               f"(keeping from pair {existing_pair_idx + 1}, skipping from pair {pair_idx + 1})")
                
                # Collect embeddings for this song (even if duplicate, for potential fallback)
                if song_idx in song_to_embeddings:
                    for emb_idx in song_to_embeddings[song_idx]:
                        embedding = npz_dataset['embeddings'][emb_idx]
                        field_type = npz_dataset['field_types'][emb_idx]
                        field_value = npz_dataset['field_values'][emb_idx]
                        
                        song_key_to_embeddings[song_key].append((
                            embedding, field_type, field_value, pair_idx
                        ))
            else:
                logger.warning(f"Song index {song_idx} out of bounds for JSON metadata in pair {pair_idx + 1}")
    
    logger.info(f"Found {len(song_key_to_data)} unique songs after deduplication")
    
    # Log deduplication statistics
    logger.info("Deduplication statistics:")
    for stat, count in sorted(duplication_stats.items()):
        logger.info(f"  {stat}: {count} songs")
    
    # Build merged datasets
    merged_json_metadata = []
    merged_songs = []
    merged_artists = []
    merged_embeddings = []
    merged_song_indices = []
    merged_field_types = []
    merged_field_values = []
    
    # Process unique songs to build merged data
    for final_idx, (song_key, (json_data, npz_data, chosen_pair_idx)) in enumerate(song_key_to_data.items()):
        # Add to merged JSON metadata
        merged_json_metadata.append(json_data)
        
        # Add to merged NPZ metadata
        merged_songs.append(json_data['original_song'])
        merged_artists.append(json_data['original_artist'])
        
        # Add embeddings for this song (only from the chosen dataset)
        embeddings_for_song = song_key_to_embeddings[song_key]
        chosen_embeddings = [
            (emb, field_type, field_value) 
            for emb, field_type, field_value, pair_idx in embeddings_for_song 
            if pair_idx == chosen_pair_idx
        ]
        
        # Add the chosen embeddings
        for embedding, field_type, field_value in chosen_embeddings:
            merged_embeddings.append(embedding)
            merged_song_indices.append(final_idx)  # Index in the merged dataset
            merged_field_types.append(field_type)
            merged_field_values.append(field_value)
    
    # Convert NPZ data to numpy arrays
    if merged_embeddings:
        embeddings_array = np.array(merged_embeddings, dtype=np.float32)
    else:
        dim = embedding_dims[0] if embedding_dims else 0
        embeddings_array = np.array([], dtype=np.float32).reshape(0, dim)
    
    merged_npz_data = {
        'songs': np.array(merged_songs, dtype=object),
        'artists': np.array(merged_artists, dtype=object),
        'embeddings': embeddings_array,
        'song_indices': np.array(merged_song_indices, dtype=int),
        'field_types': np.array(merged_field_types, dtype=object),
        'field_values': np.array(merged_field_values, dtype=object)
    }
    
    logger.info(f"Final merged dataset: {len(merged_json_metadata)} songs, {len(merged_embeddings)} embeddings")
    
    return merged_npz_data, merged_json_metadata

def save_merged_datasets(npz_data: Dict, json_data: List[Dict], output_prefix: str):
    """Save merged NPZ and JSON datasets."""
    npz_output = f"{output_prefix}.npz"
    json_output = f"{output_prefix}.json"
    
    # Ensure output directory exists
    output_path = Path(output_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save NPZ file
    logger.info(f"Saving merged NPZ dataset to {npz_output}")
    np.savez_compressed(
        npz_output,
        songs=npz_data['songs'],
        artists=npz_data['artists'],
        embeddings=npz_data['embeddings'],
        song_indices=npz_data['song_indices'],
        field_types=npz_data['field_types'],
        field_values=npz_data['field_values']
    )
    
    # Save JSON file
    logger.info(f"Saving merged JSON metadata to {json_output}")
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    if len(npz_data['field_types']) > 0:
        unique_field_types, counts = np.unique(npz_data['field_types'], return_counts=True)
    else:
        unique_field_types, counts = [], []
    
    print(f"\n{'='*70}")
    print("MERGED DATASETS SUMMARY")
    print(f"{'='*70}")
    print(f"NPZ output file: {npz_output}")
    print(f"JSON output file: {json_output}")
    print(f"Total songs: {len(npz_data['songs'])}")
    print(f"Total embeddings: {len(npz_data['embeddings'])}")
    print(f"Embedding dimension: {npz_data['embeddings'].shape[1] if len(npz_data['embeddings']) > 0 else 'N/A'}")
    
    if len(unique_field_types) > 0:
        print(f"\nEmbeddings by field type:")
        for field_type, count in zip(unique_field_types, counts):
            print(f"  {field_type}: {count}")
    
    npz_size_mb = Path(npz_output).stat().st_size / (1024 * 1024)
    json_size_mb = Path(json_output).stat().st_size / (1024 * 1024)
    print(f"\nFile sizes:")
    print(f"  NPZ file: {npz_size_mb:.1f} MB")
    print(f"  JSON file: {json_size_mb:.1f} MB")
    
    print(f"\n✅ SUCCESS: The song_indices in the NPZ file now perfectly align with the JSON file!")
    print(f"   embeddings[i] corresponds to json_data[song_indices[i]]")
    print(f"{'='*70}")

def main():
    parser = argparse.ArgumentParser(
        description="Merge and deduplicate pairs of embedding (.npz) and metadata (.json) files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_embeddings.py -i file1.npz file1.json file2.npz file2.json -o merged
  python merge_embeddings.py -i pop.npz pop.json rap.npz rap.json rock.npz rock.json -o combined
  
Note: Input files must be provided in pairs (npz, json, npz, json, ...)
Output will be: <output_prefix>.npz and <output_prefix>.json
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        nargs='+',
        required=True,
        help='Input files in pairs: npz1 json1 npz2 json2 ...'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output prefix (will create <prefix>.npz and <prefix>.json)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Parse input file pairs
        input_pairs = parse_input_pairs(args.input)
        
        # Load and validate all dataset pairs
        dataset_pairs = []
        for npz_file, json_file in input_pairs:
            try:
                # Load NPZ dataset
                npz_dataset = load_embedding_dataset(npz_file)
                
                # Load JSON metadata
                json_metadata = load_json_metadata(json_file)
                
                # Validate alignment
                validate_npz_json_alignment(npz_dataset, json_metadata, npz_file, json_file)
                
                dataset_pairs.append((npz_dataset, json_metadata))
                
            except Exception as e:
                logger.error(f"Failed to load dataset pair ({npz_file}, {json_file}): {e}")
                exit(1)
        
        # Validate output path
        output_path = Path(args.output)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Cannot create output directory for {args.output}: {e}")
            exit(1)
        
        # Merge and deduplicate
        merged_npz, merged_json = merge_and_deduplicate_datasets(dataset_pairs)
        
        # Save results
        save_merged_datasets(merged_npz, merged_json, args.output)
        
        logger.info("Merge completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Merge interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Error during merge: {e}")
        logger.debug(traceback.format_exc())
        exit(1)

if __name__ == "__main__":
    main() 