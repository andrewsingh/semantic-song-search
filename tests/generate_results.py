#!/usr/bin/env python3
"""
Generate search results for regression testing.

This script runs the search engine on test cases and saves the results
for comparison with other branches or versions.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add the semantic_song_search module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'semantic_song_search'))

from search import MusicSearchEngine, SearchConfig
from constants import DEFAULT_SONGS_FILE, DEFAULT_EMBEDDINGS_PATH, DEFAULT_ARTIST_EMBEDDINGS_PATH, DEFAULT_SHARED_GENRE_STORE_PATH, DEFAULT_PROFILES_FILE

logger = logging.getLogger(__name__)

DEFAULT_K = 48

# Default configuration values for required parameters (matching frontend defaults)
# Note: lambda_val, familiarity_min, familiarity_max have defaults in SearchConfig, so not needed here
DEFAULT_CONFIG = {
    # Top-level component weights (a_i) - should sum to 1.0
    'a0_song_sim': 0.6,        # Weight for song descriptor similarity
    'a1_artist_sim': 0.3,      # Weight for artist descriptor similarity
    'a2_total_streams': 0.05,  # Weight for total streams score
    'a3_daily_streams': 0.05,  # Weight for daily streams score
    'a4_release_date': 0.0,    # Weight for release date similarity

    # Song descriptor weights (b_i) - should sum to 1.0
    'b0_genres': 0.3,                   # Weight for song genres similarity
    'b1_vocal_style': 0.15,             # Weight for vocal style similarity
    'b2_production_sound_design': 0.15, # Weight for production & sound design similarity
    'b3_lyrical_meaning': 0.1,         # Weight for lyrical meaning similarity
    'b4_mood_atmosphere': 0.2,         # Weight for mood & atmosphere similarity
    'b5_tags': 0.1,                    # Weight for tags similarity
}

class TestResultGenerator:
    """Generates search results for regression testing."""

    def __init__(self):
        self.search_engine = None

    def initialize_search_engine(self) -> bool:
        """Initialize the search engine with default configuration."""
        try:
            # Try to initialize with default paths - this might fail if files don't exist
            # In that case, user needs to ensure the search engine is properly configured
            # Note: SearchConfig will be created dynamically during search operations
            self.search_engine = MusicSearchEngine(
                songs_file=DEFAULT_SONGS_FILE,
                embeddings_file=DEFAULT_EMBEDDINGS_PATH,
                artist_embeddings_file=DEFAULT_ARTIST_EMBEDDINGS_PATH,
                shared_genre_store_path=DEFAULT_SHARED_GENRE_STORE_PATH,
                profiles_file=DEFAULT_PROFILES_FILE
            )
            logger.info("Search engine initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")
            logger.error("Make sure the data files are available and paths are correct")
            return False

    def find_song_idx_by_track_id(self, track_id: str) -> int:
        """Find song index by track ID."""
        for idx, song in enumerate(self.search_engine.songs):
            # Check multiple possible track ID fields
            song_track_id = (song.get('track_id') or
                           song.get('id') or
                           song.get('metadata', {}).get('song_id'))
            if song_track_id == track_id:
                return idx
        raise ValueError(f"Track ID {track_id} not found in song database")

    def run_text_query(self, query: str, config: Dict[str, Any]) -> List[str]:
        """Run a text-to-song search query."""
        try:
            # Get text embedding
            query_embedding = self.search_engine.get_text_embedding(query)

            # Merge default config with test-specific config
            full_config = {**DEFAULT_CONFIG, **config}

            # Remove k from config to avoid conflict with explicit k parameter
            search_config = {k: v for k, v in full_config.items() if k != 'k'}

            # Run similarity search
            results, total_count = self.search_engine.similarity_search(
                query_embedding,
                k=config.get('k', DEFAULT_K),
                offset=0,
                ranking_engine=None,  # No ranking engine for consistency
                **search_config  # Pass all config parameters through the unified parameter system
            )

            # Debug: Check what we got back
            logger.info(f"Search returned results type: {type(results)}, total_count: {total_count}")
            if results:
                logger.info(f"First result type: {type(results[0])}")
                logger.info(f"First result keys: {list(results[0].keys())}")
                logger.info(f"First result: {results[0] if len(str(results[0])) < 500 else str(results[0])[:500] + '...'}")

            # Extract track IDs and scores from results for debugging
            track_ids = []
            track_scores = []
            for i, result in enumerate(results):
                # Debug: Check what type result is
                if not isinstance(result, dict):
                    logger.error(f"Result {i} is not a dict, it's {type(result)}: {result}")
                    continue

                # Get track_id directly from result (current format has it as top-level field)
                track_id = result.get('track_id')
                if track_id:
                    track_ids.append(track_id)
                    # Also capture the final score for debugging
                    final_score = result.get('final_score', 0.0)
                    track_scores.append(final_score)
                else:
                    logger.warning(f"No track_id found in result {i}: {list(result.keys())}")

            # Log top results with scores for debugging
            logger.info("Top 20 results with scores:")
            for i, (track_id, score) in enumerate(zip(track_ids[:20], track_scores[:20])):
                logger.info(f"  {i+1:2d}. {track_id} (score: {score:.8f})")

            return track_ids

        except Exception as e:
            logger.error(f"Error running text query '{query}': {e}")
            return []

    def run_song_query(self, track_id: str, config: Dict[str, Any]) -> List[str]:
        """Run a song-to-song search query."""
        try:
            # Find song index
            song_idx = self.find_song_idx_by_track_id(track_id)

            # Get reference song
            reference_song = self.search_engine.songs[song_idx]

            # Get query embedding (use first available embedding type)
            query_embedding = None
            for embed_type in self.search_engine.embedding_lookups:
                embedding_lookup = self.search_engine.embedding_lookups[embed_type]
                if track_id in embedding_lookup:
                    query_embedding = embedding_lookup[track_id]
                    break

            if query_embedding is None:
                raise ValueError(f"No embeddings found for track_id: {track_id}")

            # Merge default config with test-specific config
            full_config = {**DEFAULT_CONFIG, **config}

            # Remove k from config to avoid conflict with explicit k parameter
            search_config = {k: v for k, v in full_config.items() if k != 'k'}

            # Run similarity search
            results, total_count = self.search_engine.similarity_search(
                query_embedding,
                k=config.get('k', DEFAULT_K),
                offset=0,
                query_track_id=track_id,  # Enable song-to-song search
                ranking_engine=None,  # No ranking engine for consistency
                **search_config  # Pass all config parameters through the unified parameter system
            )

            # Extract track IDs and scores from results for debugging
            track_ids = []
            track_scores = []
            for i, result in enumerate(results):
                # Debug: Check what type result is
                if not isinstance(result, dict):
                    logger.error(f"Song query result {i} is not a dict, it's {type(result)}: {result}")
                    continue

                # Get track_id directly from result (current format has it as top-level field)
                result_track_id = result.get('track_id')
                if result_track_id:
                    track_ids.append(result_track_id)
                    # Also capture the final score for debugging
                    final_score = result.get('final_score', 0.0)
                    track_scores.append(final_score)
                else:
                    logger.warning(f"No track_id found in song query result {i}: {list(result.keys())}")

            # Log top results with scores for debugging
            logger.info("Top 20 song query results with scores:")
            for i, (track_id, score) in enumerate(zip(track_ids[:20], track_scores[:20])):
                logger.info(f"  {i+1:2d}. {track_id} (score: {score:.8f})")

            return track_ids

        except Exception as e:
            logger.error(f"Error running song query with track_id '{track_id}': {e}")
            return []

    def process_test_case(self, test_case: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single test case and return results."""
        test_id = test_case.get('id', 'unknown')
        test_type = test_case.get('type')

        logger.info(f"Processing test case: {test_id} (type: {test_type})")

        if test_type == 'text':
            query = test_case.get('query', '')
            track_ids = self.run_text_query(query, config)
        elif test_type == 'song':
            track_id = test_case.get('track_id', '')
            track_ids = self.run_song_query(track_id, config)
        else:
            logger.error(f"Unknown test type: {test_type}")
            track_ids = []

        return {
            'id': test_id,
            'type': test_type,
            'query': test_case.get('query') if test_type == 'text' else test_case.get('track_id'),
            'track_ids': track_ids,
            'num_results': len(track_ids)
        }

    def process_input_file(self, input_file: Path) -> Dict[str, Any]:
        """Process a single input file and return all results."""
        logger.info(f"Processing input file: {input_file}")

        try:
            with open(input_file, 'r') as f:
                data = json.load(f)

            test_cases = data.get('test_cases', [])
            config = data.get('config', {})

            results = {
                'input_file': str(input_file),
                'config': config,
                'test_results': []
            }

            for test_case in test_cases:
                result = self.process_test_case(test_case, config)
                results['test_results'].append(result)

            logger.info(f"Processed {len(test_cases)} test cases from {input_file}")
            return results

        except Exception as e:
            logger.error(f"Error processing input file {input_file}: {e}")
            return {
                'input_file': str(input_file),
                'error': str(e),
                'test_results': []
            }


def main():
    """Main function to run the test result generator."""
    parser = argparse.ArgumentParser(description='Generate search results for regression testing')
    parser.add_argument('--input', type=str, help='Specific input file to process')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory (labels or preds)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize generator
    generator = TestResultGenerator()
    if not generator.initialize_search_engine():
        sys.exit(1)

    # Determine input files to process
    if args.input:
        input_files = [Path(args.input)]
    else:
        # Process all JSON files in inputs/
        inputs_dir = Path(__file__).parent / 'inputs'
        input_files = list(inputs_dir.glob('*.json'))

    if not input_files:
        logger.error("No input files found to process")
        sys.exit(1)

    # Process each input file
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)

    for input_file in input_files:
        results = generator.process_input_file(input_file)

        # Save results with same filename in output directory
        output_file = output_dir / input_file.name

        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}")

    logger.info("Test result generation completed")


if __name__ == '__main__':
    main()