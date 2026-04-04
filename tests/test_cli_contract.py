import json
from pathlib import Path

from click.testing import CliRunner
import pytest

from dendro.cli.main import cli


REF_DIR = Path(__file__).parent.parent / "data" / "reference"
FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def known_sample_csv(tmp_path: Path) -> Path:
    return FIXTURE_DIR / "known_samples" / "nh001_297031.csv"


@pytest.fixture
def known_sample_expected() -> dict:
    return json.loads((FIXTURE_DIR / "expected" / "nh001_297031.expected.json").read_text())


@pytest.mark.skipif(not REF_DIR.exists(), reason="Reference data missing")
def test_date_json_contract(runner: CliRunner, known_sample_csv: Path, known_sample_expected: dict):
    result = runner.invoke(
        cli,
        [
            "date",
            str(known_sample_csv),
            "--reference",
            str(REF_DIR),
            "--json",
            "--top",
            "3",
            "--era-start",
            "1500",
            "--era-end",
            "2000",
        ],
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    assert set(payload.keys()) == {"status", "policy_version", "sample", "best_candidate", "candidates", "diagnostics", "warnings"}
    assert payload["status"] == known_sample_expected["status"]
    assert payload["sample"]["chosen_orientation"] == known_sample_expected["sample"]["chosen_orientation"]
    assert payload["best_candidate"]["outer_ring_year"] == known_sample_expected["best_candidate"]["outer_ring_year"]
    assert payload["best_candidate"]["reference_name"] == known_sample_expected["best_candidate"]["reference_name"]
    assert payload["best_candidate"]["reference_species"] == known_sample_expected["best_candidate"]["reference_species"]
    assert payload["best_candidate"]["reference_state"] == known_sample_expected["best_candidate"]["reference_state"]


@pytest.mark.skipif(not REF_DIR.exists(), reason="Reference data missing")
def test_info_json_contract(runner: CliRunner):
    result = runner.invoke(cli, ["info", "--reference", str(REF_DIR), "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    assert payload["entries"] > 0
    assert "rwl" in payload["file_types"]
