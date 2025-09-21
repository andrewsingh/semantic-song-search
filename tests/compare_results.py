#!/usr/bin/env python3
"""
Compare search results for regression testing.

This script compares prediction results against label results and reports
exact match percentages for the top-k search results.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class ResultComparator:
    """Compares search results for regression testing."""

    def __init__(self):
        pass

    def load_results_file(self, file_path: Path) -> Dict[str, Any]:
        """Load results from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading results file {file_path}: {e}")
            return {}

    def compare_track_lists(self, expected: List[str], actual: List[str], k: int = None) -> Dict[str, Any]:
        """Compare two lists of track IDs and return match statistics."""
        if k is None:
            k = min(len(expected), len(actual))
        else:
            k = min(k, len(expected), len(actual))

        if k == 0:
            return {
                'exact_match': True,
                'match_percentage': 100.0,
                'k': k,
                'expected_count': len(expected),
                'actual_count': len(actual),
                'matches': 0,
                'differences': []
            }

        # Compare top-k results
        expected_k = expected[:k]
        actual_k = actual[:k]

        matches = 0
        differences = []

        for i in range(k):
            if i < len(expected_k) and i < len(actual_k):
                if expected_k[i] == actual_k[i]:
                    matches += 1
                else:
                    differences.append({
                        'position': i,
                        'expected': expected_k[i],
                        'actual': actual_k[i]
                    })
            elif i < len(expected_k):
                differences.append({
                    'position': i,
                    'expected': expected_k[i],
                    'actual': None
                })
            elif i < len(actual_k):
                differences.append({
                    'position': i,
                    'expected': None,
                    'actual': actual_k[i]
                })

        match_percentage = (matches / k) * 100.0 if k > 0 else 100.0
        exact_match = matches == k

        return {
            'exact_match': exact_match,
            'match_percentage': match_percentage,
            'k': k,
            'expected_count': len(expected),
            'actual_count': len(actual),
            'matches': matches,
            'differences': differences
        }

    def compare_test_case(self, expected_case: Dict[str, Any], actual_case: Dict[str, Any], k: int = None) -> Dict[str, Any]:
        """Compare a single test case."""
        test_id = expected_case.get('id', 'unknown')

        # Extract track ID lists
        expected_tracks = expected_case.get('track_ids', [])
        actual_tracks = actual_case.get('track_ids', [])

        # Use the specified k or the minimum from config
        if k is None:
            # Try to get k from config or use length of expected results
            k = min(len(expected_tracks), len(actual_tracks))

        comparison = self.compare_track_lists(expected_tracks, actual_tracks, k)

        return {
            'test_id': test_id,
            'test_type': expected_case.get('type'),
            'query': expected_case.get('query'),
            **comparison
        }

    def compare_results_file(self, labels_file: Path, preds_file: Path, k: int = None) -> Dict[str, Any]:
        """Compare predictions against labels."""
        logger.info(f"Comparing predictions ({preds_file.name}) against labels ({labels_file.name})")

        # Load both files
        labels_data = self.load_results_file(labels_file)
        preds_data = self.load_results_file(preds_file)

        if not labels_data or not preds_data:
            return {
                'labels_file': labels_file.name,
                'preds_file': preds_file.name,
                'error': 'Failed to load one or both files',
                'test_comparisons': []
            }

        # Get test results (labels are expected/ground truth, preds are actual)
        expected_tests = labels_data.get('test_results', [])
        actual_tests = preds_data.get('test_results', [])

        # Create lookup for actual tests by ID
        actual_lookup = {test.get('id'): test for test in actual_tests}

        test_comparisons = []
        missing_tests = []

        for expected_test in expected_tests:
            test_id = expected_test.get('id')

            if test_id in actual_lookup:
                comparison = self.compare_test_case(expected_test, actual_lookup[test_id], k)
                test_comparisons.append(comparison)
            else:
                missing_tests.append(test_id)
                logger.warning(f"Test case {test_id} not found in predictions")

        # Calculate overall statistics
        total_tests = len(test_comparisons)
        exact_matches = sum(1 for comp in test_comparisons if comp['exact_match'])

        if total_tests > 0:
            exact_match_percentage = (exact_matches / total_tests) * 100.0
            avg_match_percentage = sum(comp['match_percentage'] for comp in test_comparisons) / total_tests
        else:
            exact_match_percentage = 0.0
            avg_match_percentage = 0.0

        return {
            'labels_file': labels_file.name,
            'preds_file': preds_file.name,
            'total_tests': total_tests,
            'exact_matches': exact_matches,
            'exact_match_percentage': exact_match_percentage,
            'avg_match_percentage': avg_match_percentage,
            'missing_tests': missing_tests,
            'test_comparisons': test_comparisons,
            'labels_config': labels_data.get('config', {}),
            'preds_config': preds_data.get('config', {})
        }

    def generate_report(self, comparisons: List[Dict[str, Any]], k: int = None) -> str:
        """Generate a human-readable report."""
        report_lines = []

        # Header
        if k:
            report_lines.append(f"# Regression Test Results (Top-{k} Comparison)")
        else:
            report_lines.append("# Regression Test Results")
        report_lines.append("")

        # Overall summary
        total_files = len(comparisons)
        total_tests = sum(comp['total_tests'] for comp in comparisons)
        total_exact_matches = sum(comp['exact_matches'] for comp in comparisons)

        if total_tests > 0:
            overall_exact_percentage = (total_exact_matches / total_tests) * 100.0
            overall_avg_percentage = sum(
                comp['avg_match_percentage'] * comp['total_tests'] for comp in comparisons
            ) / total_tests
        else:
            overall_exact_percentage = 0.0
            overall_avg_percentage = 0.0

        report_lines.append("## Overall Summary")
        report_lines.append(f"- **Files compared:** {total_files}")
        report_lines.append(f"- **Total test cases:** {total_tests}")
        report_lines.append(f"- **Exact matches:** {total_exact_matches}/{total_tests} ({overall_exact_percentage:.1f}%)")
        report_lines.append(f"- **Average match percentage:** {overall_avg_percentage:.1f}%")
        report_lines.append("")

        # Per-file results
        report_lines.append("## Per-File Results")
        report_lines.append("")

        for comp in comparisons:
            if 'error' in comp:
                # Handle both old and new format
                file_name = comp.get('preds_file', comp.get('file', 'unknown'))
                report_lines.append(f"### {file_name} ❌")
                report_lines.append(f"**Error:** {comp['error']}")
                if 'labels_file' in comp and 'preds_file' in comp:
                    report_lines.append(f"**Labels:** {comp['labels_file']}")
                    report_lines.append(f"**Predictions:** {comp['preds_file']}")
                report_lines.append("")
                continue

            status = "✅" if comp['exact_match_percentage'] == 100.0 else "⚠️"
            # Handle both old and new format
            file_name = comp.get('preds_file', comp.get('file', 'unknown'))
            report_lines.append(f"### {file_name} {status}")

            if 'labels_file' in comp and 'preds_file' in comp:
                report_lines.append(f"- **Labels file:** {comp['labels_file']}")
                report_lines.append(f"- **Predictions file:** {comp['preds_file']}")

            report_lines.append(f"- **Test cases:** {comp['total_tests']}")
            report_lines.append(f"- **Exact matches:** {comp['exact_matches']}/{comp['total_tests']} ({comp['exact_match_percentage']:.1f}%)")
            report_lines.append(f"- **Average match:** {comp['avg_match_percentage']:.1f}%")

            if comp['missing_tests']:
                report_lines.append(f"- **Missing from predictions:** {', '.join(comp['missing_tests'])}")

            report_lines.append("")

        # Detailed failures (only if there are failures)
        failed_tests = []
        for comp in comparisons:
            if 'test_comparisons' in comp:
                for test_comp in comp['test_comparisons']:
                    if not test_comp['exact_match']:
                        # Handle both old and new format
                        file_name = comp.get('preds_file', comp.get('file', 'unknown'))
                        failed_tests.append((file_name, test_comp))

        if failed_tests:
            report_lines.append("## Failed Test Cases")
            report_lines.append("")

            for file_name, test_comp in failed_tests:
                report_lines.append(f"### {file_name} - {test_comp['test_id']}")
                report_lines.append(f"- **Query:** {test_comp['query']}")
                report_lines.append(f"- **Type:** {test_comp['test_type']}")
                report_lines.append(f"- **Match percentage:** {test_comp['match_percentage']:.1f}%")
                report_lines.append(f"- **Matches:** {test_comp['matches']}/{test_comp['k']}")

                if test_comp['differences']:
                    report_lines.append("- **Differences (Labels → Predictions):**")
                    for diff in test_comp['differences'][:5]:  # Show first 5 differences
                        pos = diff['position']
                        expected = diff['expected'] or 'None'
                        actual = diff['actual'] or 'None'
                        report_lines.append(f"  - Position {pos}: `{expected}` → `{actual}`")

                    if len(test_comp['differences']) > 5:
                        remaining = len(test_comp['differences']) - 5
                        report_lines.append(f"  - ... and {remaining} more differences")

                report_lines.append("")

        return "\n".join(report_lines)


def main():
    """Main function to run the result comparison."""
    parser = argparse.ArgumentParser(description='Compare predictions against labels for regression testing')

    # Explicit prediction and label file arguments
    parser.add_argument('--preds', type=str, help='Predictions file or directory (required)')
    parser.add_argument('--labels', type=str, help='Labels/ground truth file or directory (required)')

    parser.add_argument('--k', type=int, help='Number of top results to compare (default: compare all available)')
    parser.add_argument('--output', type=str, help='Output file for the comparison report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize comparator
    comparator = ResultComparator()
    tests_dir = Path(__file__).parent

    # Require both preds and labels arguments
    if not args.preds or not args.labels:
        parser.error("Both --preds and --labels arguments are required")

    # Set up paths
    preds_path = Path(args.preds)
    labels_path = Path(args.labels)

    # If relative paths, resolve relative to tests directory
    if not preds_path.is_absolute():
        preds_path = tests_dir / preds_path
    if not labels_path.is_absolute():
        labels_path = tests_dir / labels_path

    if preds_path.is_file() and labels_path.is_file():
        # Single file comparison
        comparison = comparator.compare_results_file(labels_path, preds_path, args.k)
        comparisons = [comparison]
    elif preds_path.is_dir() and labels_path.is_dir():
        # Directory comparison - match files by name
        comparisons = []
        label_files = {f.name: f for f in labels_path.glob('*.json')}

        for pred_file in preds_path.glob('*.json'):
            if pred_file.name in label_files:
                label_file = label_files[pred_file.name]
                comparison = comparator.compare_results_file(label_file, pred_file, args.k)
                comparisons.append(comparison)
            else:
                logger.warning(f"No matching label file found for prediction: {pred_file.name}")
    else:
        logger.error("Both --preds and --labels must be either files or directories")
        return 1

    if not comparisons:
        logger.error("No comparisons to perform")
        return 1

    # Generate report
    report = comparator.generate_report(comparisons, args.k)

    # Output report
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {args.output}")
        except Exception as e:
            logger.error(f"Error saving report to {args.output}: {e}")
            print(report)
    else:
        print(report)

    # Exit with error code if any tests failed
    total_tests = sum(comp.get('total_tests', 0) for comp in comparisons)
    total_exact_matches = sum(comp.get('exact_matches', 0) for comp in comparisons)

    if total_tests > 0 and total_exact_matches < total_tests:
        return 1

    return 0


if __name__ == '__main__':
    exit(main())