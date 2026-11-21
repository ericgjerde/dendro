"""Tests for the gate that promotes a ranked date to ``recommended``."""

from pathlib import Path

import numpy as np
import pytest

from dendro.crossdating.matcher import CrossdateMatcher, DatingCandidate


REF_DIR = Path(__file__).parent.parent / "data" / "reference"
SKIP_IF_NO_DATA = pytest.mark.skipif(
    not REF_DIR.exists() or not any(REF_DIR.rglob("*.rwl")),
    reason="Reference data not downloaded. Run 'dendro download' first.",
)


def _candidate(**overrides) -> DatingCandidate:
    """A 120-ring candidate that clears every gate; each test weakens one property."""
    fields = dict(
        reference_id="REF1",
        reference_name="Reference One",
        reference_species="TSCA",
        reference_state="NH",
        reference_file_type="rwl",
        reference_material_group="hemlock",
        proposed_start_year=1700,
        proposed_end_year=1819,
        correlation=0.60,
        t_value=8.0,
        p_value=1e-9,
        overlap=120,
        gleichlauf=70.0,
        segment_consistency=0.8,
        composite_score=1.2,
        score_gap_to_next=0.3,
        year_cluster_support=4,
    )
    fields.update(overrides)
    return DatingCandidate(**fields)


def _is_recommended(candidate: DatingCandidate) -> bool:
    return CrossdateMatcher()._is_recommended(candidate)


def test_strong_full_overlap_candidate_is_recommended():
    assert _is_recommended(_candidate()) is True


def test_t_value_within_reach_of_chance_matches_is_not_recommended():
    # The best chance alignment of signal-free input reaches t of about 6 across the
    # default search, and nearly all wrong dates on real series had t below 7.
    assert _is_recommended(_candidate(t_value=6.9)) is False
    assert _is_recommended(_candidate(t_value=7.0)) is True


def test_correlation_below_0_40_is_not_recommended():
    assert _is_recommended(_candidate(correlation=0.39)) is False


def test_reference_covering_under_half_the_sample_is_not_recommended():
    # 59 of 120 rings overlap the reference: most of the sample was never compared.
    assert _is_recommended(_candidate(overlap=59)) is False
    assert _is_recommended(_candidate(overlap=60)) is True


@SKIP_IF_NO_DATA
def test_signal_free_samples_are_never_recommended():
    """Negative control: noise run through the CLI defaults must not get a recommended date."""
    matcher = CrossdateMatcher(reference_dir=REF_DIR)
    rng = np.random.default_rng(20261120)
    recommended = []
    for index in range(40):
        length = int(rng.choice([60, 80, 100, 150]))
        innovations = rng.normal(0.0, 0.3, size=length)
        log_width = np.zeros(length)
        for year in range(1, length):
            log_width[year] = 0.6 * log_width[year - 1] + innovations[year]
        report = matcher.date_sample(
            np.exp(log_width),
            sample_name=f"noise-{index}",
            has_bark_edge=True,
            orientation="auto",
            era_start=1600,
            era_end=1900,
            top_n=10,
        )
        if report.status == "recommended":
            recommended.append((index, length, report.best_candidate.outer_ring_year))

    assert recommended == []
