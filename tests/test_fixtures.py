"""Ensure the committed synthetic fixtures are reproducible and parseable."""

from pathlib import Path

from dendro.reference.tucson_parser import parse_rwl_file

FIX = Path(__file__).parent / "fixtures"
REF = FIX / "reference" / "synth"


def test_fixture_files_exist():
    assert (REF / "synth01.rwl").exists()
    assert (REF / "synth02.rwl").exists()
    assert (FIX / "samples" / "known_1789.csv").exists()


def test_fixtures_parse():
    rwl = parse_rwl_file(REF / "synth01.rwl")
    assert len(rwl.series) == 8
    for series in rwl.series.values():
        assert series.length > 100
        assert (series.values > 0).all()


def test_fixtures_are_deterministic(tmp_path, monkeypatch):
    # Regenerating into a temp dir must reproduce the committed bytes exactly.
    import tests.fixtures.generate_fixtures as gen

    committed = (REF / "synth01.rwl").read_bytes()
    monkeypatch.setattr(gen, "REF_DIR", tmp_path / "ref")
    monkeypatch.setattr(gen, "SAMPLE_DIR", tmp_path / "samples")
    gen.generate_all()
    regenerated = (tmp_path / "ref" / "synth01.rwl").read_bytes()
    assert regenerated == committed
