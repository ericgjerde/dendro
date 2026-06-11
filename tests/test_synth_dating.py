"""
End-to-end dating validation against committed synthetic fixtures.

Unlike ``test_validation.py`` (which needs downloaded ITRDB data and therefore
skips in CI), these tests run against the deterministic fixtures under
``tests/fixtures`` and exercise the full pipeline: parse -> per-series detrend ->
robust master chronology -> sliding correlation -> felling interpretation.

The fixtures are built so the known felling year is 1789; recovering it proves
the master-chronology, orientation, and felling fixes work together.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dendro.crossdating.matcher import CrossdateMatcher
from dendro.reference.chronology_index import ChronologyIndex

FIX = Path(__file__).parent / "fixtures"
REF = FIX / "reference"
SAMPLE = FIX / "samples" / "known_1789.csv"


@pytest.fixture(scope="module")
def matcher() -> CrossdateMatcher:
    return CrossdateMatcher(index=ChronologyIndex(REF))


@pytest.fixture(scope="module")
def sample_widths() -> np.ndarray:
    return pd.read_csv(SAMPLE)["width_mm"].values


def test_recovers_known_felling_year(matcher, sample_widths):
    report = matcher.date_sample(
        values=sample_widths,
        sample_name="known_1789",
        has_bark_edge=True,
        era_start=1700,
        era_end=1850,
        min_overlap=40,
    )
    assert report.consensus_year == 1789
    assert report.consensus_confidence == "HIGH"
    # Strong, unambiguous match expected from clean synthetic data.
    assert report.matches[0].correlation > 0.8
    assert report.matches[0].t_value > 10


def test_bark_edge_gives_exact_felling(matcher, sample_widths):
    report = matcher.date_sample(
        values=sample_widths, has_bark_edge=True,
        era_start=1700, era_end=1850, min_overlap=40,
    )
    fe = report.felling_estimate
    assert fe is not None
    assert fe.felling_type.value == "exact"
    assert fe.earliest_felling == fe.latest_felling == 1789


def test_no_bark_edge_is_terminus_post_quem(matcher, sample_widths):
    report = matcher.date_sample(
        values=sample_widths, has_bark_edge=False,
        era_start=1700, era_end=1850, min_overlap=40,
    )
    fe = report.felling_estimate
    assert fe is not None
    assert fe.felling_type.value == "after"
    # Felled strictly after the last dated ring.
    assert fe.earliest_felling > 1789
    assert fe.latest_felling is None


def test_orientation_reversed_still_dates(matcher, sample_widths):
    # The same series measured bark-first must recover the same year.
    report = matcher.date_sample(
        values=sample_widths[::-1], has_bark_edge=True,
        orientation="bark_to_pith",
        era_start=1700, era_end=1850, min_overlap=40,
    )
    assert report.consensus_year == 1789


def test_master_chronology_is_cached(matcher, sample_widths):
    matcher.date_sample(values=sample_widths, era_start=1700, era_end=1850, min_overlap=40)
    # Every indexed reference should now have a cache entry.
    assert len(matcher._master_cache) >= 1
