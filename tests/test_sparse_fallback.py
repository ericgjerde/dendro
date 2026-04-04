import numpy as np
from pathlib import Path
import pytest

from dendro.crossdating.matcher import CrossdateMatcher, DatingCandidate
from dendro.reference.chronology_index import ChronologyIndex, MasterChronology, ReferenceManifestEntry
from dendro.reference.tucson_parser import parse_rwl_file


REF_DIR = Path(__file__).parent.parent / "data" / "reference"
SKIP_IF_NO_DATA = pytest.mark.skipif(
    not REF_DIR.exists() or not any(REF_DIR.rglob("*.rwl")),
    reason="Reference data not downloaded. Run 'dendro download' first.",
)


def _entry(site_id: str, species: str, state: str, *, years: int = 250) -> ReferenceManifestEntry:
    start_year = 1700
    values = np.linspace(-1.0, 1.0, years).tolist()
    sample_depth = [5] * years
    master = MasterChronology(
        start_year=start_year,
        end_year=start_year + years - 1,
        values=values,
        sample_depth=sample_depth,
        build_method="test",
        detrend_method="spline",
    )
    return ReferenceManifestEntry(
        filepath=f"/tmp/{site_id.lower()}.rwl",
        site_id=site_id,
        site_name=site_id,
        species=species,
        state=state,
        start_year=start_year,
        end_year=start_year + years - 1,
        num_years=years,
        num_series=5,
        file_type="rwl",
        master=master,
    )


def _matcher_with_entries(*entries: ReferenceManifestEntry) -> CrossdateMatcher:
    index = ChronologyIndex()
    index.entries = []
    index._by_species = {}
    index._by_state = {}
    for entry in entries:
        index._add_entry(entry)
    return CrossdateMatcher(index=index)


def test_sparse_same_species_plan_adds_genus_and_broad_fallback():
    matcher = _matcher_with_entries(
        _entry("PIST01", "PIST", "NY"),
        _entry("PIST02", "PIST", "ME"),
        _entry("PIRI01", "PIRI", "NY"),
        _entry("PIRI02", "PIRI", "ME"),
        _entry("PIPA01", "PIPA", "MA"),
        _entry("PIRE01", "PIRE", "NH"),
        _entry("PIRE02", "PIRE", "ME"),
        _entry("PIRI03", "PIRI", "VT"),
        _entry("PIPA02", "PIPA", "CT"),
        _entry("PIRI04", "PIRI", "RI"),
        _entry("TSCA01", "TSCA", "CT"),
        _entry("PCRU01", "PCRU", "NH"),
    )

    plan = matcher._build_search_plan(
        species_filter=["PIST"],
        state_filter=None,
        min_year=1650,
        max_year=2000,
        min_overlap=30,
        max_references=50,
    )

    lane_names = [lane.name for lane in plan.lanes]
    assert plan.reference_count == 2
    assert plan.recommendation_blocked is True
    assert lane_names == ["species_primary", "pi_genus_fallback", "broad_fallback"]
    assert any("extremely sparse" in warning for warning in plan.warnings)


def test_no_same_species_plan_uses_broad_fallback_when_genus_is_not_supported():
    matcher = _matcher_with_entries(
        _entry("TSCA01", "TSCA", "CT"),
        _entry("TSCA02", "TSCA", "VT"),
        _entry("PCRU01", "PCRU", "NH"),
        _entry("PCRU02", "PCRU", "ME"),
    )

    plan = matcher._build_search_plan(
        species_filter=["JUVI"],
        state_filter=None,
        min_year=1650,
        max_year=2000,
        min_overlap=30,
        max_references=50,
    )

    assert plan.reference_count == 0
    assert [lane.name for lane in plan.lanes] == ["broad_fallback"]
    assert any("No same-species references matched" in warning for warning in plan.warnings)


def test_sparse_coverage_blocks_recommendation_even_for_strong_candidate():
    matcher = CrossdateMatcher.__new__(CrossdateMatcher)
    strong_candidate = DatingCandidate(
        reference_id="NH001",
        reference_name="Strong Site",
        reference_species="PCRU",
        reference_state="NH",
        reference_file_type="rwl",
        reference_material_group="hemlock",
        proposed_start_year=1700,
        proposed_end_year=1900,
        correlation=0.5,
        t_value=7.0,
        p_value=0.0,
        overlap=120,
        gleichlauf=70.0,
        segment_consistency=0.8,
        composite_score=0.8,
        year_cluster_support=3,
        score_gap_to_next=0.1,
    )

    assert matcher._is_recommended(strong_candidate, recommendation_blocked=True) is False
    assert matcher._is_recommended(strong_candidate, recommendation_blocked=False) is True


@SKIP_IF_NO_DATA
def test_real_sparse_qupr_case_stays_ranked_with_sparse_diagnostics():
    matcher = CrossdateMatcher(reference_dir=REF_DIR)
    rwl = parse_rwl_file(REF_DIR / "ny" / "ny003.rwl")
    series = rwl.series["367041"]

    report = matcher.date_sample(
        series.values,
        sample_name="367041",
        species_filter=["QUPR"],
        era_start=series.start_year - 50,
        era_end=series.end_year + 50,
        top_n=5,
    )

    assert report.best_candidate is not None
    assert report.best_candidate.outer_ring_year == series.end_year
    assert report.status == "ranked"
    assert report.diagnostics["reference_count"] == 4
    assert report.diagnostics["sparse_reference_coverage"] is True
    assert [lane["name"] for lane in report.diagnostics["search_lanes"]] == [
        "species_primary",
        "qu_genus_fallback",
    ]
    assert any("recommendation is disabled" in warning for warning in report.warnings)


@SKIP_IF_NO_DATA
def test_real_zero_coverage_case_reports_fallback_lanes():
    matcher = CrossdateMatcher(reference_dir=REF_DIR)
    rwl = parse_rwl_file(REF_DIR / "ny" / "ny019.rwl")
    series = rwl.series["ADS01a"]

    report = matcher.date_sample(
        series.values,
        sample_name="ADS01a",
        species_filter=["QUST"],
        era_start=series.start_year - 50,
        era_end=series.end_year + 50,
        top_n=5,
    )

    assert report.best_candidate is not None
    assert report.status == "ranked"
    assert report.diagnostics["reference_count"] == 1
    assert report.diagnostics["sparse_reference_coverage"] is True
    assert [lane["name"] for lane in report.diagnostics["search_lanes"]] == [
        "species_primary",
        "qu_genus_fallback",
        "broad_fallback",
    ]
    assert any("extremely sparse" in warning for warning in report.warnings)
