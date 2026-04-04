import json
from pathlib import Path

from click.testing import CliRunner
import pytest

from dendro.cli.main import cli


REF_DIR = Path(__file__).parent.parent / "data" / "reference"
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "walpole"
SKIP_IF_NO_DATA = pytest.mark.skipif(
    not REF_DIR.exists() or not any(REF_DIR.rglob("*.rwl")),
    reason="Reference data not downloaded. Run 'dendro download' first.",
)


@pytest.fixture
def runner():
    return CliRunner()


@SKIP_IF_NO_DATA
def test_infer_materials_json_recommends_hemlock_for_walpole_holdout(runner: CliRunner):
    result = runner.invoke(
        cli,
        [
            "infer-materials",
            str(FIXTURE_DIR / "bp7s_measurements.csv"),
            "--reference",
            str(REF_DIR),
            "--town",
            "Walpole",
            "--state",
            "NH",
            "--built-year-range",
            "1760:1800",
            "--member-type",
            "frame",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    assert payload["status"] == "recommended"
    assert payload["recommended_material"] == "hemlock"
    assert payload["material_candidates"][0]["material_group"] == "hemlock"
    assert payload["material_candidates"][0]["best_outer_ring_year"] == 1779


@SKIP_IF_NO_DATA
def test_date_auto_material_embeds_material_inference(runner: CliRunner):
    result = runner.invoke(
        cli,
        [
            "date",
            str(FIXTURE_DIR / "bp7s_measurements.csv"),
            "--reference",
            str(REF_DIR),
            "--town",
            "Walpole",
            "--state",
            "NH",
            "--built-year-range",
            "1760:1800",
            "--member-type",
            "frame",
            "--auto-material",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    assert payload["status"] == "recommended"
    assert payload["best_candidate"]["outer_ring_year"] == 1779
    assert payload["best_candidate"]["reference_material_group"] == "hemlock"
    assert payload["material_inference"] is not None
    assert payload["material_inference"]["recommended_material"] == "hemlock"
