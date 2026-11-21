from pathlib import Path

import numpy as np
import pytest

from dendro.crossdating.matcher import CrossdateMatcher
from dendro.reference.chronology_index import ChronologyIndex, MasterChronology, ReferenceManifestEntry
from dendro.reference.tucson_parser import parse_rwl_file


REF_DIR = Path(__file__).parent.parent / "data" / "reference"
SKIP_IF_NO_DATA = pytest.mark.skipif(
    not REF_DIR.exists() or not any(REF_DIR.rglob("*.rwl")),
    reason="Reference data not downloaded. Run 'dendro download' first.",
)


def _reference(site_id: str, site_name: str, values: np.ndarray) -> ReferenceManifestEntry:
    """A 1700..1999 reference whose master is used by the matcher exactly as given."""
    return ReferenceManifestEntry(
        filepath=f"/nonexistent/{site_id}.rwl",
        site_id=site_id,
        site_name=site_name,
        species="TSCA",
        state="NY",
        start_year=1700,
        end_year=1999,
        num_years=300,
        num_series=10,
        file_type="rwl",
        material_group="hemlock",
        master=MasterChronology(
            start_year=1700,
            end_year=1999,
            values=[float(value) for value in values],
            sample_depth=[10] * 300,
            build_method="rwl-log-hp20-ar1-biweight",
            detrend_method="spline",
        ),
    )


def test_year_consensus_bonus_promotes_supported_cluster():
    """Three sites agreeing on 1949 outrank one stronger, unsupported peak at 1819."""
    rng = np.random.default_rng(21)
    signal = np.exp(rng.normal(0.0, 0.3, size=300))  # regional signal, 1700..1999
    sample = signal[150:250].copy()  # true rings 1850..1949

    index = ChronologyIndex()
    for site_id, site_name in [("b", "Consensus Site One"), ("c", "Consensus Site Two"), ("d", "Consensus Site Three")]:
        index._add_entry(_reference(site_id, site_name, signal * np.exp(rng.normal(0.0, 0.33, size=300))))
    # An unrelated site holding a cleaner copy of the sample at the wrong date (1720..1819).
    isolated_values = np.exp(rng.normal(0.0, 0.3, size=300))
    isolated_values[20:120] = sample * np.exp(rng.normal(0.0, 0.22, size=100))
    index._add_entry(_reference("a", "Wrong Isolated Site", isolated_values))

    report = CrossdateMatcher(index=index).date_sample(
        sample,
        sample_name="consensus-sample",
        orientation="oldest_to_newest",
        era_start=1700,
        era_end=1999,
        min_overlap=30,
    )
    candidates = report.candidates

    def without_bonus(candidate):
        return candidate.composite_score - candidate.year_cluster_bonus

    isolated = next(
        candidate
        for candidate in candidates
        if candidate.reference_name == "Wrong Isolated Site" and candidate.outer_ring_year == 1819
    )
    consensus = [
        candidate
        for candidate in candidates
        if candidate.reference_name.startswith("Consensus Site") and candidate.outer_ring_year == 1949
    ]
    assert len(consensus) == 3
    # Premise: before the consensus bonus the isolated peak is the strongest alignment.
    assert without_bonus(isolated) > max(without_bonus(candidate) for candidate in consensus)

    assert candidates[0].outer_ring_year == 1949
    assert candidates[0].year_cluster_support >= 3
    assert candidates[-1].reference_name == "Wrong Isolated Site"
    assert isolated.year_cluster_support == 1
    assert isolated.year_cluster_bonus < min(candidate.year_cluster_bonus for candidate in consensus)


def test_consensus_bonus_penalizes_isolated_peak_without_dominant_evidence():
    matcher = object.__new__(CrossdateMatcher)

    isolated = matcher._compute_consensus_bonus(
        support=1, unique_families=1, ev_ratio=0.3, exact_ev=3.0, max_year_ev=10.0
    )
    pair = matcher._compute_consensus_bonus(
        support=2, unique_families=2, ev_ratio=1.0, exact_ev=10.0, max_year_ev=10.0
    )
    trio = matcher._compute_consensus_bonus(
        support=3, unique_families=3, ev_ratio=1.0, exact_ev=10.0, max_year_ev=10.0
    )

    assert isolated < 0
    assert 0 < pair < trio


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
