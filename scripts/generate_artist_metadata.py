#!/usr/bin/env python3
"""
Generate artist_metadata.json from artist profiles JSONL file.

This script reads the artist profiles JSONL and creates the metadata file
required for genre similarity computation.

Usage:
  python generate_artist_metadata.py -i artist_profiles_v7.1.jsonl -o artist_metadata.json
"""

import argparse
import json
import sys
from pathlib import Path

def generate_artist_metadata(input_file: str, output_file: str):
    """
    Convert artist profiles JSONL to artist metadata JSON.

    Args:
        input_file: Path to artist profiles JSONL file
        output_file: Path to output metadata JSON file
    """
    print(f"Reading artist profiles from: {input_file}")

    artist_metadata = {}
    processed_count = 0
    duplicate_count = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    profile = json.loads(line.strip())

                    # Extract required fields
                    artist_name = profile.get('artist', '').strip()
                    if not artist_name:
                        print(f"Warning: Line {line_num} has no artist name, skipping")
                        continue

                    # Check for duplicates
                    if artist_name in artist_metadata:
                        duplicate_count += 1
                        print(f"Warning: Duplicate artist '{artist_name}' on line {line_num}, skipping")
                        continue

                    # Extract lead vocalist gender
                    lead_vocalist_gender = profile.get('lead_vocalist_gender', '').strip()

                    # Extract and format genres
                    genres_raw = profile.get('genres', [])
                    genres_formatted = []

                    for genre in genres_raw:
                        if isinstance(genre, dict) and 'name' in genre:
                            genre_formatted = {
                                'name': genre['name'],
                                'prominence': genre.get('prominence', 0),
                                'key': f"genre: {genre['name']}"  # Add the required key field
                            }
                            genres_formatted.append(genre_formatted)
                        else:
                            print(f"Warning: Invalid genre format for {artist_name}: {genre}")

                    # Create metadata entry
                    artist_metadata[artist_name] = {
                        'lead_vocalist_gender': lead_vocalist_gender,
                        'genres': genres_formatted
                    }

                    processed_count += 1

                    if processed_count % 50 == 0:
                        print(f"Processed {processed_count} artists...")

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error reading input file: {e}")
        return False

    print(f"\nProcessed {processed_count} artists")
    print(f"Found {duplicate_count} duplicate artists")
    print(f"Writing metadata to: {output_file}")

    # Write the metadata file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(artist_metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ Successfully generated artist metadata with {len(artist_metadata)} artists")

        # Show a sample entry
        if artist_metadata:
            sample_artist = next(iter(artist_metadata.keys()))
            sample_data = artist_metadata[sample_artist]
            print(f"\nSample entry for '{sample_artist}':")
            print(json.dumps({sample_artist: sample_data}, indent=2)[:300] + "...")

        return True

    except Exception as e:
        print(f"Error writing output file: {e}")
        return False

# ──────────────────────────────── ARGPARSE ────────────────────────────────
parser = argparse.ArgumentParser(
    description="Generate artist_metadata.json from artist profiles JSONL file."
)
parser.add_argument("-i", "--input", required=True,
                    help="Path to input artist profiles JSONL file")
parser.add_argument("-o", "--output", required=True,
                    help="Path to output artist metadata JSON file")

def main():
    """Main function to generate artist metadata."""
    cli_args = parser.parse_args()

    input_file = cli_args.input
    output_file = cli_args.output

    print("=== Artist Metadata Generator ===")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print()

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate the metadata
    success = generate_artist_metadata(input_file, output_file)

    if success:
        print("\n🎉 Artist metadata generation completed successfully!")
    else:
        print("\n❌ Artist metadata generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()