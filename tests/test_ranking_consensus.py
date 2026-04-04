from pathlib import Path

import pytest

from dendro.crossdating.matcher import CrossdateMatcher, DatingCandidate
from dendro.reference.tucson_parser import parse_rwl_file


REF_DIR = Path(__file__).parent.parent / "data" / "reference"
SKIP_IF_NO_DATA = pytest.mark.skipif(
    not REF_DIR.exists() or not any(REF_DIR.rglob("*.rwl")),
    reason="Reference data not downloaded. Run 'dendro download' first.",
)


def _candidate(reference_id: str, reference_name: str, year: int, score: float) -> DatingCandidate:
    return DatingCandidate(
        reference_id=reference_id,
        reference_name=reference_name,
        reference_species="TSCA",
        reference_state="NY",
        reference_file_type="rwl",
        proposed_start_year=year - 99,
        proposed_end_year=year,
        correlation=0.45,
        t_value=6.0,
        p_value=0.001,
        overlap=100,
        gleichlauf=65.0,
        composite_score=score,
    )


def test_year_consensus_bonus_promotes_supported_cluster():
    matcher = object.__new__(CrossdateMatcher)
    candidates = [
        _candidate("a", "Wrong Isolated Site", 1800, 0.70),
        _candidate("b", "Consensus Site One", 1900, 0.66),
        _candidate("c", "Consensus Site Two", 1901, 0.65),
        _candidate("d", "Consensus Site Three", 1900, 0.64),
    ]

    matcher._apply_year_consensus_bonus(candidates)
    candidates.sort(key=lambda candidate: candidate.composite_score, reverse=True)

    assert candidates[0].outer_ring_year in {1900, 1901}
    assert candidates[0].year_cluster_support >= 3
    assert candidates[-1].reference_name == "Wrong Isolated Site"
    assert candidates[-1].year_cluster_bonus < 0


@SKIP_IF_NO_DATA
def test_year_consensus_recovers_known_clustered_case():
    matcher = CrossdateMatcher(reference_dir=REF_DIR)
    rwl = parse_rwl_file(REF_DIR / "ny" / "ny001.rwl")
    series = rwl.series["046011"]

    report = matcher.date_sample(
        series.values,
        sample_name="046011",
        species_filter=["TSCA"],
        era_start=series.start_year - 50,
        era_end=series.end_year + 50,
        top_n=5,
    )

    assert report.best_candidate is not None
    assert report.best_candidate.outer_ring_year == series.end_year
