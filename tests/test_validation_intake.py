import json
from pathlib import Path

from click.testing import CliRunner

from dendro.cli.main import cli


def _write_dummy_measurement_artifacts(case_dir: Path):
    artifacts = case_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "measurements.csv").write_text("ring_index,width_mm\n1,1.0\n2,0.9\n")
    (artifacts / "measurement.session.json").write_text(
        json.dumps(
            {
                "image_path": "scan.tif",
                "dpi": 1200,
                "image_shape": [100, 200],
                "exported_widths_mm_oldest_to_newest": [1.0, 0.9],
            }
        )
    )
    (artifacts / "scan.tif").write_bytes(b"fake")


def test_init_validation_suite_scaffolds_manifest(tmp_path: Path):
    runner = CliRunner()
    suite_dir = tmp_path / "local_suite"

    result = runner.invoke(
        cli,
        [
            "init-validation-suite",
            str(suite_dir),
            "--name",
            "House Firewood Cases",
            "--town",
            "Walpole",
            "--state",
            "NH",
            "--built-year-range",
            "1760:1800",
        ],
    )

    assert result.exit_code == 0, result.output
    suite_file = suite_dir / "suite.json"
    assert suite_file.exists()
    payload = json.loads(suite_file.read_text())
    assert payload["name"] == "House Firewood Cases"
    assert payload["built_year_range"] == [1760, 1800]
    assert (suite_dir / "cases").exists()
    assert (suite_dir / "README.md").exists()


def test_add_validation_case_scaffolds_supported_case(tmp_path: Path):
    runner = CliRunner()
    suite_dir = tmp_path / "local_suite"
    runner.invoke(cli, ["init-validation-suite", str(suite_dir)])

    result = runner.invoke(
        cli,
        [
            "add-validation-case",
            str(suite_dir),
            "hemlock_firewood_001",
            "--species-name",
            "Eastern Hemlock",
            "--species-code",
            "TSCA",
            "--material-group",
            "hemlock",
            "--true-outer-ring-year",
            "2025",
            "--sample-origin",
            "firewood",
            "--scan-dpi",
            "1200",
        ],
    )

    assert result.exit_code == 0, result.output
    case_file = suite_dir / "cases" / "hemlock_firewood_001" / "case.json"
    assert case_file.exists()
    payload = json.loads(case_file.read_text())
    assert payload["known_material_group"] == "hemlock"
    assert payload["known_species_code"] == "TSCA"
    assert payload["artifacts"]["scan_image"] == "artifacts/scan.tif"
    suite_payload = json.loads((suite_dir / "suite.json").read_text())
    assert suite_payload["case_ids"] == ["hemlock_firewood_001"]


def test_validation_info_reports_supported_and_unsupported_case_readiness(tmp_path: Path):
    runner = CliRunner()
    suite_dir = tmp_path / "local_suite"
    runner.invoke(cli, ["init-validation-suite", str(suite_dir)])

    supported_result = runner.invoke(
        cli,
        [
            "add-validation-case",
            str(suite_dir),
            "hemlock_firewood_001",
            "--species-name",
            "Eastern Hemlock",
            "--species-code",
            "TSCA",
            "--material-group",
            "hemlock",
            "--true-outer-ring-year",
            "2025",
            "--sample-origin",
            "firewood",
            "--scan-dpi",
            "1200",
        ],
    )
    assert supported_result.exit_code == 0, supported_result.output

    unsupported_result = runner.invoke(
        cli,
        [
            "add-validation-case",
            str(suite_dir),
            "ash_firewood_001",
            "--species-name",
            "White Ash",
            "--species-code",
            "FRAM",
            "--expected-policy-outcome",
            "unsupported_inconclusive",
            "--true-outer-ring-year",
            "2025",
            "--sample-origin",
            "firewood",
            "--scale-included",
        ],
    )
    assert unsupported_result.exit_code == 0, unsupported_result.output

    _write_dummy_measurement_artifacts(suite_dir / "cases" / "hemlock_firewood_001")

    result = runner.invoke(
        cli,
        [
            "validation-info",
            str(suite_dir),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["case_count"] == 2
    assert payload["summary"]["expected_policy_outcomes"]["supported_dateable"] == 1
    assert payload["summary"]["expected_policy_outcomes"]["unsupported_inconclusive"] == 1
    assert payload["summary"]["readiness"]["metadata_ready"] == 2
    assert payload["summary"]["readiness"]["measurement_track_ready"] == 1
    assert payload["summary"]["readiness"]["scan_session_track_ready"] == 1

    by_case = {case["case_id"]: case for case in payload["cases"]}
    assert by_case["hemlock_firewood_001"]["status"] == "ready_for_full_validation"
    assert by_case["ash_firewood_001"]["status"] == "awaiting_artifacts"
    assert by_case["ash_firewood_001"]["known_material_group"] is None
