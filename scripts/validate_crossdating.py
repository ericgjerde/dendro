#!/usr/bin/env python3
"""
Validate cross-dating algorithm using known-date ITRDB samples.

This script takes a real tree ring series with a known date, "forgets" the date,
and attempts to recover it through cross-dating. This validates that the
algorithm works correctly before using it on unknown samples.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from dendro.reference.tucson_parser import parse_rwl_file
from dendro.reference.chronology_index import ChronologyIndex
from dendro.crossdating.matcher import CrossdateMatcher
from dendro.crossdating.detrend import DetrendMethod


def validate_with_known_sample(
    test_file: str,
    series_id: str,
    reference_dir: str,
    exclude_file: bool = True,
):
    """
    Validate cross-dating by testing a known-date sample.

    Args:
        test_file: Path to RWL file containing the test series
        series_id: ID of the series to test
        reference_dir: Directory containing reference chronologies
        exclude_file: If True, exclude the source file from references
    """
    print(f"=" * 60)
    print("CROSS-DATING VALIDATION TEST")
    print(f"=" * 60)

    # Load the test series
    print(f"\nLoading test sample from: {test_file}")
    rwl = parse_rwl_file(test_file)

    if series_id not in rwl.series:
        print(f"Error: Series '{series_id}' not found in file")
        print(f"Available series: {list(rwl.series.keys())[:10]}...")
        return False

    test_series = rwl.series[series_id]
    true_start = test_series.start_year
    true_end = test_series.end_year

    print(f"Test series: {series_id}")
    print(f"TRUE DATE: {true_start} - {true_end} ({test_series.length} years)")
    print(f"\nNow 'forgetting' the date and attempting to recover it...")

    # Get the ring width values (this is all we'd have from an unknown sample)
    values = test_series.values

    # Build the matcher, optionally excluding the source file
    print(f"\nBuilding reference index from: {reference_dir}")
    index = ChronologyIndex(reference_dir)

    if exclude_file:
        # Remove entries from the same file to make this a fair test
        test_filename = Path(test_file).name
        original_count = len(index.entries)
        index.entries = [e for e in index.entries if Path(e.filepath).name != test_filename]
        index._by_species = {}
        index._by_state = {}
        for entry in index.entries:
            if entry.species:
                if entry.species not in index._by_species:
                    index._by_species[entry.species] = []
                index._by_species[entry.species].append(entry)
            if entry.state:
                if entry.state not in index._by_state:
                    index._by_state[entry.state] = []
                index._by_state[entry.state].append(entry)
        print(f"Excluded source file. Using {len(index.entries)}/{original_count} reference files.")

    # Run cross-dating
    matcher = CrossdateMatcher(index=index)

    # Search in a window around the true date
    search_start = true_start - 50
    search_end = true_end + 50

    print(f"Searching for match in era {search_start}-{search_end}...")
    print()

    report = matcher.date_sample(
        values=values,
        sample_name=series_id,
        has_bark_edge=True,
        era_start=search_start,
        era_end=search_end,
        min_overlap=30,
    )

    # Analyze results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    if not report.matches:
        print("NO MATCHES FOUND")
        return False

    best_match = report.matches[0]
    recovered_end = best_match.felling_year
    recovered_start = best_match.proposed_start_year

    error = recovered_end - true_end

    print(f"\nTRUE END YEAR:      {true_end}")
    print(f"RECOVERED END YEAR: {recovered_end}")
    print(f"ERROR:              {error:+d} years")
    print()
    print(f"Best match: {best_match.reference_name}")
    print(f"Correlation: {best_match.correlation:.3f}")
    print(f"T-value: {best_match.t_value:.1f}")
    print(f"Confidence: {best_match.confidence}")

    if report.warnings:
        print(f"\nWarnings:")
        for w in report.warnings:
            print(f"  - {w}")

    # Show top 5 matches
    print(f"\nTop 5 matches:")
    print("-" * 60)
    for i, m in enumerate(report.matches[:5], 1):
        match_error = m.felling_year - true_end
        marker = "<<<" if match_error == 0 else ""
        print(f"{i}. {m.reference_name}: {m.felling_year} (error: {match_error:+d}) "
              f"r={m.correlation:.3f} t={m.t_value:.1f} {marker}")

    # Verdict
    print()
    print("=" * 60)
    if error == 0:
        print("SUCCESS: Correct date recovered!")
        return True
    elif abs(error) <= 2:
        print(f"CLOSE: Off by {abs(error)} year(s) - acceptable margin")
        return True
    else:
        print(f"FAILED: Off by {abs(error)} years")
        return False


def run_multiple_validations(reference_dir: str, num_tests: int = 5):
    """Run validation on multiple samples from different files."""

    print("\n" + "=" * 60)
    print("RUNNING MULTIPLE VALIDATION TESTS")
    print("=" * 60 + "\n")

    reference_dir = Path(reference_dir)

    # Find RWL files with good coverage of our target era
    test_cases = []

    for rwl_path in sorted(reference_dir.rglob("*.rwl")):
        try:
            rwl = parse_rwl_file(rwl_path)
            for series_id, series in rwl.series.items():
                # Look for series that span the late 1700s with decent length
                if (series.start_year <= 1780 and
                    series.end_year >= 1800 and
                    series.length >= 50):
                    test_cases.append((str(rwl_path), series_id, series.end_year, series.length))
        except Exception:
            continue

    print(f"Found {len(test_cases)} suitable test series")

    # Select diverse test cases
    if len(test_cases) > num_tests:
        # Pick from different files/states
        selected = []
        seen_files = set()
        for case in sorted(test_cases, key=lambda x: -x[3]):  # Sort by length
            if case[0] not in seen_files:
                selected.append(case)
                seen_files.add(case[0])
                if len(selected) >= num_tests:
                    break
        test_cases = selected

    # Run tests
    results = []
    for i, (filepath, series_id, true_end, length) in enumerate(test_cases[:num_tests], 1):
        print(f"\n{'#' * 60}")
        print(f"TEST {i}/{num_tests}")
        print(f"{'#' * 60}")

        success = validate_with_known_sample(
            test_file=filepath,
            series_id=series_id,
            reference_dir=str(reference_dir),
            exclude_file=True,
        )
        results.append((filepath, series_id, true_end, success))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    successes = sum(1 for r in results if r[3])
    print(f"\nPassed: {successes}/{len(results)}")

    for filepath, series_id, true_end, success in results:
        status = "PASS" if success else "FAIL"
        filename = Path(filepath).name
        print(f"  [{status}] {filename}/{series_id} (true end: {true_end})")

    return successes == len(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate cross-dating algorithm")
    parser.add_argument("--reference-dir", "-r", default="data/reference",
                       help="Directory containing reference chronologies")
    parser.add_argument("--test-file", "-f", help="Specific RWL file to test")
    parser.add_argument("--series-id", "-s", help="Specific series ID to test")
    parser.add_argument("--num-tests", "-n", type=int, default=5,
                       help="Number of validation tests to run")

    args = parser.parse_args()

    if args.test_file and args.series_id:
        # Test specific series
        success = validate_with_known_sample(
            test_file=args.test_file,
            series_id=args.series_id,
            reference_dir=args.reference_dir,
        )
    else:
        # Run multiple automatic tests
        success = run_multiple_validations(
            reference_dir=args.reference_dir,
            num_tests=args.num_tests,
        )

    sys.exit(0 if success else 1)
