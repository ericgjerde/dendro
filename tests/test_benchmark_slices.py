from pathlib import Path

from dendro.crossdating.benchmark import (
    BenchmarkCase,
    BenchmarkCaseResult,
    assign_benchmark_slices,
    summarize_corpus_results,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "benchmark_slices_v1.json"


def _case(**overrides) -> BenchmarkCase:
    payload = {
        "test_file": "/tmp/sample.rwl",
        "series_id": "abc123",
        "state": "CT",
        "species": "TSCA",
        "site_id": "CT001",
        "site_name": "Sample Site",
        "true_start_year": 1700,
        "true_end_year": 1985,
        "length": 180,
        "curated_suite_case": False,
    }
    payload.update(overrides)
    return BenchmarkCase(**payload)


def _result(**overrides) -> BenchmarkCaseResult:
    payload = {
        "case": _case(),
        "status": "recommended",
        "reference_count": 18,
        "candidate_count": 10,
        "top1_outer_ring_year": 1985,
        "top1_error": 0,
        "top1_reference_name": "Sample Site",
        "top1_reference_state": "CT",
        "top1_reference_species": "TSCA",
        "top1_score": 0.91,
        "top1_correlation": 0.62,
        "top1_t_value": 8.1,
        "correct_year_rank": 1,
        "correct_year_in_top5": True,
        "correct_year_in_top10": True,
        "category": "correct_recommended",
        "likely_causes": [],
        "warning_count": 0,
    }
    payload.update(overrides)
    return BenchmarkCaseResult(**payload)


def test_assigns_primary_slice_for_dense_conifer_case():
    result = _result()

    assign_benchmark_slices(result, slice_config_path=FIXTURE_PATH)

    assert result.primary_slice_id == "core_dense_conifer"
    assert result.primary_slice_matched is True
    assert result.overlay_ids == []
    assert result.to_dict()["benchmark_slices"]["primary_slice_id"] == "core_dense_conifer"


def test_assigns_sparse_slice_and_overlays():
    result = _result(
        case=_case(
            state="NH",
            species="JUVI",
            site_id="NH999",
            site_name="Sparse Site",
            true_end_year=1971,
            length=120,
        ),
        reference_count=2,
        candidate_count=0,
        status="inconclusive",
        top1_outer_ring_year=None,
        top1_error=None,
        top1_reference_name="",
        top1_reference_state="",
        top1_reference_species="",
        top1_score=None,
        top1_correlation=None,
        top1_t_value=None,
        correct_year_rank=None,
        correct_year_in_top5=False,
        correct_year_in_top10=False,
        category="no_match",
        likely_causes=["sparse_filtered_reference_coverage"],
    )

    assign_benchmark_slices(result, slice_config_path=FIXTURE_PATH)

    assert result.primary_slice_id == "sparse_coverage_fallback"
    assert result.primary_slice_matched is True
    assert result.overlay_ids == ["short_series", "thin_search_space"]


def test_corpus_summary_exposes_slice_targets():
    dense = _result()
    sparse = _result(
        case=_case(
            state="NH",
            species="JUVI",
            site_id="NH999",
            site_name="Sparse Site",
            true_end_year=1971,
            length=120,
        ),
        reference_count=2,
        candidate_count=0,
        status="inconclusive",
        top1_outer_ring_year=None,
        top1_error=None,
        top1_reference_name="",
        top1_reference_state="",
        top1_reference_species="",
        top1_score=None,
        top1_correlation=None,
        top1_t_value=None,
        correct_year_rank=None,
        correct_year_in_top5=False,
        correct_year_in_top10=False,
        category="no_match",
        likely_causes=["sparse_filtered_reference_coverage"],
    )

    assign_benchmark_slices(dense, slice_config_path=FIXTURE_PATH)
    assign_benchmark_slices(sparse, slice_config_path=FIXTURE_PATH)

    summary = summarize_corpus_results([dense, sparse], slice_config_path=FIXTURE_PATH)
    benchmark_slices = summary["benchmark_slices"]
    primary_summaries = {item["id"]: item for item in benchmark_slices["primary_slices"]}
    overlay_summaries = {item["id"]: item for item in benchmark_slices["overlays"]}

    assert primary_summaries["core_dense_conifer"]["targets_met"] is True
    assert primary_summaries["core_dense_conifer"]["target_checks"]["top1_within_2_years_rate"]["passed"] is True
    assert primary_summaries["sparse_coverage_fallback"]["targets_met"] is False
    assert primary_summaries["sparse_coverage_fallback"]["target_checks"]["recommended_precision"]["passed"] is False
    assert overlay_summaries["short_series"]["count"] == 1
    assert overlay_summaries["thin_search_space"]["count"] == 1
