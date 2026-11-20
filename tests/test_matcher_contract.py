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
            reference_material_group="hemlock",
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
            reference_material_group="hemlock",
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
            reference_material_group="hemlock",
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


def test_era_filter_finds_in_era_match_even_with_stronger_out_of_era_peak():
    """Era bounds must restrict the lag search rather than dropping the whole reference."""
    import numpy as np
    from dendro.reference.chronology_index import ChronologyIndex, MasterChronology, ReferenceManifestEntry

    rng = np.random.default_rng(42)
    ref_vals = rng.normal(0.0, 1.0, size=300)
    # Plant a strong match inside 1750..1850 (end year 1829) and an exact copy outside (end year 1979)
    sample = ref_vals[70:130].copy() + rng.normal(0.0, 0.25, size=60)
    ref_vals[220:280] = sample.copy()  # Stronger peak at end year 1979 (outside era!)

    master = MasterChronology(
        start_year=1700,
        end_year=1999,
        values=ref_vals.tolist(),
        sample_depth=[10] * 300,
        build_method="rwl-log-hp20-ar1-biweight",
        detrend_method="spline",
    )
    entry = ReferenceManifestEntry(
        filepath="/tmp/era_test.rwl",
        site_id="ERATEST",
        site_name="Era Test Site",
        species="TSCA",
        state="NH",
        start_year=1700,
        end_year=1999,
        num_years=300,
        num_series=10,
        file_type="rwl",
        material_group="hemlock",
        master=master,
    )
    index = ChronologyIndex()
    index._add_entry(entry)
    matcher = CrossdateMatcher(index=index)

    report = matcher.date_sample(
        sample,
        sample_name="era-sample",
        orientation="oldest_to_newest",
        era_start=1750,
        era_end=1860,
        min_overlap=30,
    )
    assert report.best_candidate is not None
    assert report.best_candidate.outer_ring_year == 1829


def test_partial_overlap_computes_segment_consistency():
    """Partial edge overlap must still compute segment correlations instead of returning 0."""
    import numpy as np
    from dendro.reference.chronology_index import ChronologyIndex, MasterChronology, ReferenceManifestEntry

    rng = np.random.default_rng(77)
    ref_vals = rng.normal(0.0, 1.0, size=120)  # 1700..1819
    # Sample spans 1750..1849 (70-year overlap with ref end, 30 years past ref end)
    sample = np.concatenate([ref_vals[50:120], rng.normal(0.0, 1.0, size=30)])

    master = MasterChronology(
        start_year=1700,
        end_year=1819,
        values=ref_vals.tolist(),
        sample_depth=[8] * 120,
        build_method="rwl-log-hp20-ar1-biweight",
        detrend_method="spline",
    )
    entry = ReferenceManifestEntry(
        filepath="/tmp/partial.rwl",
        site_id="PARTIAL",
        site_name="Partial Overlap Site",
        species="TSCA",
        state="VT",
        start_year=1700,
        end_year=1819,
        num_years=120,
        num_series=8,
        file_type="rwl",
        material_group="hemlock",
        master=master,
    )
    index = ChronologyIndex()
    index._add_entry(entry)
    matcher = CrossdateMatcher(index=index)

    report = matcher.date_sample(
        sample,
        sample_name="partial-sample",
        orientation="oldest_to_newest",
        era_start=1800,
        era_end=1870,
        min_overlap=30,
    )
    assert report.best_candidate is not None
    assert report.best_candidate.outer_ring_year == 1849
    assert len(report.best_candidate.segment_correlations) > 0
    assert report.best_candidate.segment_consistency > 0.5


def test_distinct_year_score_gap_does_not_penalize_multi_site_agreement():
    """Two sites agreeing on the same year must not collapse score_gap_to_next to 0."""
    import numpy as np
    from dendro.reference.chronology_index import ChronologyIndex, MasterChronology, ReferenceManifestEntry

    rng = np.random.default_rng(99)
    signal = rng.normal(0.0, 1.0, size=200)
    sample = signal[60:160].copy()

    index = ChronologyIndex()
    for i in range(2):
        noisy = signal + rng.normal(0.0, 0.1, size=200)
        master = MasterChronology(
            start_year=1700,
            end_year=1899,
            values=noisy.tolist(),
            sample_depth=[10] * 200,
            build_method="rwl-log-hp20-ar1-biweight",
            detrend_method="spline",
        )
        index._add_entry(
            ReferenceManifestEntry(
                filepath=f"/tmp/site_{i}.rwl",
                site_id=f"SITE{i}",
                site_name=f"Independent Site {i}",
                species="TSCA",
                state="NH",
                start_year=1700,
                end_year=1899,
                num_years=200,
                num_series=10,
                file_type="rwl",
                material_group="hemlock",
                master=master,
            )
        )

    matcher = CrossdateMatcher(index=index)
    report = matcher.date_sample(
        sample,
        sample_name="consensus-sample",
        orientation="oldest_to_newest",
        era_start=1750,
        era_end=1900,
        min_overlap=30,
    )
    assert report.best_candidate is not None
    assert report.best_candidate.outer_ring_year == 1859
    assert report.best_candidate.score_gap_to_next > 0.2
    assert report.status == "recommended"

