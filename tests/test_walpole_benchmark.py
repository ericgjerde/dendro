import json
from pathlib import Path

from dendro.crossdating.benchmark import load_walpole_benchmark_suite, run_walpole_benchmark_suite
from scripts.validate_crossdating import run_walpole_analysis


FIXTURE_DIR = Path(__file__).parent / "fixtures"
WALPOLE_SUITE = FIXTURE_DIR / "walpole_benchmark_v1.json"
REFERENCE_DIR = Path(__file__).parent.parent / "data" / "reference"


def test_load_walpole_benchmark_suite_schema():
    suite = load_walpole_benchmark_suite(WALPOLE_SUITE)

    assert suite.name == "walpole_benchmark_v1"
    assert suite.scope == "walpole_nh_late_1700s_house"
    assert suite.town == "Walpole"
    assert suite.state == "NH"
    assert suite.built_year_range == (1760, 1800)
    assert suite.supported_material_groups == ("hemlock", "white_pine", "hard_pine", "oak", "chestnut")
    assert [track.id for track in suite.tracks] == ["measurements_to_date", "scan_session_to_date"]
    assert [case.material_group for case in suite.tracks[1].cases] == [
        "hemlock",
        "white_pine",
        "hard_pine",
        "oak",
        "chestnut",
    ]


def test_run_walpole_benchmark_suite_supports_measurements_and_scan_sessions():
    payload = run_walpole_benchmark_suite(REFERENCE_DIR, WALPOLE_SUITE)

    assert payload["summary"]["overall"]["count"] == 6
    assert payload["summary"]["overall"]["passed"] == 2
    assert payload["summary"]["overall"]["skipped"] == 4
    assert payload["summary"]["track_summary"]["measurements_to_date"]["passed"] == 1
    assert payload["summary"]["track_summary"]["scan_session_to_date"]["passed"] == 1
    assert payload["summary"]["track_summary"]["scan_session_to_date"]["skipped"] == 4
    assert payload["summary"]["material_group_summary"]["hemlock"]["passed"] == 2
    assert payload["summary"]["material_group_summary"]["white_pine"]["skipped"] == 1
    assert payload["summary"]["material_group_summary"]["chestnut"]["skipped"] == 1
    assert payload["summary"]["skip_reasons"]["Missing scan artifact"] == 4

    hemlock_results = [result for result in payload["results"] if result["case"]["material_group"] == "hemlock"]
    assert len(hemlock_results) == 2
    assert all(result["status"] == "passed" for result in hemlock_results)
    assert {result["material_inference_status"] for result in hemlock_results} == {"recommended"}
    assert {result["recommended_material"] for result in hemlock_results} == {"hemlock"}
    assert {result["top_material_group"] for result in hemlock_results} == {"hemlock"}
    assert {result["analysis_status"] for result in hemlock_results} == {"ranked"}
    assert {result["top1_outer_ring_year"] for result in hemlock_results} == {1779}

    skipped = [result for result in payload["results"] if result["status"] == "skipped"]
    assert len(skipped) == 4
    assert all(result["skip_reason"] == "Missing scan artifact" for result in skipped)

    json.dumps(payload)


def test_run_walpole_analysis_writes_json(tmp_path: Path):
    output_json = tmp_path / "walpole-benchmark.json"
    payload = run_walpole_analysis(
        reference_dir=str(REFERENCE_DIR),
        suite_file=str(WALPOLE_SUITE),
        output_json=str(output_json),
        top_n=10,
        min_overlap=30,
    )

    assert output_json.exists()
    saved = json.loads(output_json.read_text())
    assert saved["summary"]["overall"]["passed"] == 2
    assert saved["summary"]["overall"]["skipped"] == 4
    assert saved["summary"]["track_summary"]["scan_session_to_date"]["skipped"] == 4
    assert payload["summary"]["overall"]["passed"] == 2
