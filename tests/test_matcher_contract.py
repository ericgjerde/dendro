import json
from pathlib import Path

import pytest

from dendro.crossdating.matcher import CrossdateMatcher, DatingCandidate
from dendro.reference.tucson_parser import parse_rwl_file


REF_DIR = Path(__file__).parent.parent / "data" / "reference"
SKIP_IF_NO_DATA = pytest.mark.skipif(
    not REF_DIR.exists() or not any(REF_DIR.rglob("*.rwl")),
    reason="Reference data not downloaded. Run 'dendro download' first.",
)


@SKIP_IF_NO_DATA
def test_known_series_report_is_serializable_and_ranked():
    matcher = CrossdateMatcher(reference_dir=REF_DIR)
    rwl = parse_rwl_file(REF_DIR / "nh" / "nh001.rwl")
    series = rwl.series["297031"]

    report = matcher.date_sample(
        series.values,
        sample_name="297031",
        era_start=1500,
        era_end=2000,
        top_n=5,
    )
    payload = report.to_dict()

    assert report.status in {"recommended", "ranked"}
    assert payload["best_candidate"] is not None
    assert payload["best_candidate"]["outer_ring_year"] == series.end_year
    assert payload["sample"]["chosen_orientation"] == "oldest_to_newest"
    assert payload["candidates"][0]["composite_score"] >= payload["candidates"][1]["composite_score"]

    json.dumps(payload)


@SKIP_IF_NO_DATA
def test_auto_orientation_recovers_reversed_measurements():
    matcher = CrossdateMatcher(reference_dir=REF_DIR)
    rwl = parse_rwl_file(REF_DIR / "nh" / "nh001.rwl")
    series = rwl.series["297031"]

    report = matcher.date_sample(
        series.values[::-1],
        sample_name="297031-reversed",
        orientation="auto",
        era_start=1500,
        era_end=2000,
        top_n=3,
    )

    assert report.best_candidate is not None
    assert report.best_candidate.outer_ring_year == series.end_year
    assert report.chosen_orientation == "bark_to_pith"


def test_year_consensus_boost_prefers_dense_year_cluster():
    matcher = CrossdateMatcher.__new__(CrossdateMatcher)
    candidates = [
        DatingCandidate(
            reference_id="wrong-1",
            reference_name="Wrong Peak",
            reference_species="TSCA",
            reference_state="NY",
            reference_file_type="rwl",
            proposed_start_year=1700,
            proposed_end_year=1850,
            correlation=0.55,
            t_value=7.2,
            p_value=0.0,
            overlap=180,
            gleichlauf=62.0,
            segment_consistency=0.7,
            composite_score=0.66,
        ),
        DatingCandidate(
            reference_id="correct-1",
            reference_name="Correct Cluster A",
            reference_species="TSCA",
            reference_state="NY",
            reference_file_type="rwl",
            proposed_start_year=1831,
            proposed_end_year=1981,
            correlation=0.42,
            t_value=6.1,
            p_value=0.0,
            overlap=180,
            gleichlauf=61.0,
            segment_consistency=0.7,
            composite_score=0.63,
        ),
        DatingCandidate(
            reference_id="correct-2",
            reference_name="Correct Cluster B",
            reference_species="TSCA",
            reference_state="NH",
            reference_file_type="rwl",
            proposed_start_year=1831,
            proposed_end_year=1981,
            correlation=0.41,
            t_value=6.0,
            p_value=0.0,
            overlap=180,
            gleichlauf=60.0,
            segment_consistency=0.7,
            composite_score=0.625,
        ),
    ]

    matcher._apply_year_consensus(candidates)
    candidates.sort(key=lambda candidate: candidate.composite_score, reverse=True)

    assert candidates[0].outer_ring_year == 1981
    assert any("year-consensus=" in rationale for rationale in candidates[0].rationale)
