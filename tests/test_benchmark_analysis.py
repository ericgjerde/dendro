from dendro.crossdating.benchmark import (
    BenchmarkCase,
    BenchmarkCaseResult,
    classify_case_result,
    summarize_corpus_results,
)


def _case(**overrides) -> BenchmarkCase:
    payload = {
        "test_file": "/tmp/sample.rwl",
        "series_id": "abc123",
        "state": "NH",
        "species": "PCRU",
        "site_id": "NH001",
        "site_name": "Nancy Brook",
        "true_start_year": 1700,
        "true_end_year": 1971,
        "length": 180,
        "curated_suite_case": False,
    }
    payload.update(overrides)
    return BenchmarkCase(**payload)


def _result(**overrides) -> BenchmarkCaseResult:
    payload = {
        "case": _case(),
        "status": "ranked",
        "reference_count": 18,
        "candidate_count": 10,
        "top1_outer_ring_year": 1850,
        "top1_error": -121,
        "top1_reference_name": "Some Site",
        "top1_reference_state": "ME",
        "top1_reference_species": "PCRU",
        "top1_score": 0.7,
        "top1_correlation": 0.55,
        "top1_t_value": 6.1,
        "correct_year_rank": None,
        "correct_year_in_top5": False,
        "correct_year_in_top10": False,
        "warning_count": 1,
    }
    payload.update(overrides)
    return BenchmarkCaseResult(**payload)


def test_classify_ranking_miss_when_correct_year_survives():
    result = _result(correct_year_rank=2, correct_year_in_top5=True)
    category, causes = classify_case_result(result)

    assert category == "ranking_miss"
    assert "correct_year_survives_but_scores_below_false_peak" in causes


def test_classify_no_match_uses_coverage_hint():
    result = _result(
        reference_count=2,
        candidate_count=0,
        top1_outer_ring_year=None,
        top1_error=None,
        top1_reference_name="",
        top1_reference_state="",
        top1_reference_species="",
        top1_score=None,
        top1_correlation=None,
        top1_t_value=None,
    )
    category, causes = classify_case_result(result)

    assert category == "no_match"
    assert "sparse_filtered_reference_coverage" in causes


def test_summarize_corpus_results_tracks_rates():
    correct = _result(
        status="recommended",
        top1_outer_ring_year=1971,
        top1_error=0,
        correct_year_rank=1,
        correct_year_in_top5=True,
        correct_year_in_top10=True,
        category="correct_recommended",
        likely_causes=[],
    )
    ranking_miss = _result(
        correct_year_rank=2,
        correct_year_in_top5=True,
        category="ranking_miss",
        likely_causes=["correct_year_survives_but_scores_below_false_peak"],
    )

    summary = summarize_corpus_results([correct, ranking_miss])

    assert summary["overall"]["count"] == 2
    assert summary["overall"]["top1_within_2_years"] == 1
    assert summary["overall"]["top5_within_2_years"] == 2
    assert summary["category_counts"]["ranking_miss"] == 1
