"""Tests for the persisted reference manifest."""

import json
import os
import time
from pathlib import Path

from dendro.reference.chronology_index import MANIFEST_FORMAT_VERSION, ChronologyIndex


def _write_rwl(path: Path, n_years: int = 40, start_year: int = 1900) -> None:
    values = [100 + (i * 37) % 60 for i in range(n_years)]
    lines = []
    for decade_start in range(0, n_years, 10):
        chunk = values[decade_start:decade_start + 10]
        line = f"{'AAA01':<8}{start_year + decade_start:4d}" + "".join(f"{v:6d}" for v in chunk)
        if decade_start + 10 >= n_years:
            line += "   999"
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")


def test_offline_rebuild_keeps_metadata_the_files_cannot_supply(tmp_path: Path):
    """Species and site names fetched earlier must survive a rebuild that has no network."""
    rwl_path = tmp_path / "nh999.rwl"
    _write_rwl(rwl_path, n_years=40)

    index = ChronologyIndex(tmp_path)
    entry = index.entries[0]
    assert entry.species == ""  # the raw file alone carries no species
    entry.species = "TSCA"
    entry.site_name = "Example Pond"
    entry.provenance_url = "https://example.invalid/study/1"
    entry.metadata_source = "embedded,remote"
    index.save_manifest(index.manifest_path)

    # The file changes (10 more years), so the manifest is stale and must be rebuilt.
    _write_rwl(rwl_path, n_years=50)
    newer = time.time() + 60
    os.utime(rwl_path, (newer, newer))

    rebuilt = ChronologyIndex(tmp_path)
    rebuilt_entry = rebuilt.entries[0]
    assert rebuilt_entry.end_year == 1949  # data was re-read from the changed file
    assert rebuilt_entry.species == "TSCA"
    assert rebuilt_entry.site_name == "Example Pond"
    assert rebuilt_entry.material_group == "hemlock"
    assert rebuilt_entry.provenance_url == "https://example.invalid/study/1"
    assert rebuilt_entry.metadata_source == "embedded,remote"
    assert [e.site_id for e in rebuilt.search(species=["TSCA"])] == [rebuilt_entry.site_id]

    persisted = json.loads(rebuilt.manifest_path.read_text())["entries"][0]
    assert persisted["species"] == "TSCA"
    assert persisted["site_name"] == "Example Pond"


def _count_scans(monkeypatch) -> list[int]:
    """Count full directory scans while keeping the real behaviour."""
    calls: list[int] = []
    original = ChronologyIndex.scan_directory

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ChronologyIndex, "scan_directory", spy)
    return calls


def test_unparseable_file_does_not_force_a_rescan_on_every_load(tmp_path: Path, monkeypatch):
    _write_rwl(tmp_path / "good.rwl")
    (tmp_path / "bad.rwl").write_text("# truncated download, no series\n")
    assert len(ChronologyIndex(tmp_path)) == 1

    calls = _count_scans(monkeypatch)
    assert len(ChronologyIndex(tmp_path)) == 1
    assert calls == []


def test_unparseable_file_is_indexed_once_it_is_repaired(tmp_path: Path):
    _write_rwl(tmp_path / "good.rwl")
    bad_path = tmp_path / "bad.rwl"
    bad_path.write_text("# truncated download, no series\n")
    assert len(ChronologyIndex(tmp_path)) == 1

    _write_rwl(bad_path)
    newer = time.time() + 60
    os.utime(bad_path, (newer, newer))
    assert len(ChronologyIndex(tmp_path)) == 2


def test_symlinked_reference_file_keeps_a_relative_manifest_path(tmp_path: Path, monkeypatch):
    store = tmp_path / "store"
    reference_dir = tmp_path / "reference"
    (reference_dir / "vt").mkdir(parents=True)
    store.mkdir()
    _write_rwl(store / "vt999.rwl")
    (reference_dir / "vt" / "vt999.rwl").symlink_to(store / "vt999.rwl")

    index = ChronologyIndex(reference_dir)
    persisted = json.loads(index.manifest_path.read_text())["entries"]
    assert [entry["filepath"] for entry in persisted] == [str(Path("vt") / "vt999.rwl")]

    calls = _count_scans(monkeypatch)
    assert len(ChronologyIndex(reference_dir)) == 1
    assert calls == []


def test_cached_master_arrays_cannot_be_modified_by_callers(tmp_path: Path):
    _write_rwl(tmp_path / "good.rwl")
    entry = ChronologyIndex(tmp_path).entries[0]

    for array in (entry.master_values, entry.master_crossdate_values):
        try:
            array[0] = 123.0
        except ValueError:
            continue
        raise AssertionError("cached master array accepted an in-place write")


def test_failed_manifest_save_leaves_the_previous_manifest_intact(tmp_path: Path, monkeypatch):
    _write_rwl(tmp_path / "good.rwl")
    index = ChronologyIndex(tmp_path)
    before = index.manifest_path.read_text()

    def failing_replace(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing_replace)
    index.entries[0].site_name = "Changed"
    try:
        index.save_manifest(index.manifest_path)
    except OSError:
        pass

    assert index.manifest_path.read_text() == before


def test_manifest_written_by_older_code_is_rebuilt_when_raw_files_exist(tmp_path: Path):
    """Parser and conditioning fixes must reach reference folders indexed before them."""
    _write_rwl(tmp_path / "nh999.rwl", n_years=40)
    index = ChronologyIndex(tmp_path)
    raw = json.loads(index.manifest_path.read_text())
    assert raw["format_version"] == MANIFEST_FORMAT_VERSION

    # An older manifest: no version stamp, a span the current parser would not
    # produce, and metadata that is worth keeping.
    del raw["format_version"]
    raw["entries"][0]["end_year"] = 1900
    raw["entries"][0]["species"] = "TSCA"
    index.manifest_path.write_text(json.dumps(raw))
    newer = time.time() + 60
    os.utime(index.manifest_path, (newer, newer))  # newer than the raw file: mtimes alone say "fresh"

    rebuilt = ChronologyIndex(tmp_path)
    assert rebuilt.entries[0].end_year == 1939
    assert rebuilt.entries[0].species == "TSCA"
    assert json.loads(rebuilt.manifest_path.read_text())["format_version"] == MANIFEST_FORMAT_VERSION


def test_older_manifest_is_still_served_when_no_raw_files_are_present(tmp_path: Path):
    """Offline mode has nothing to rebuild from, so an older manifest stays usable and untouched."""
    source_dir = tmp_path / "source"
    offline_dir = tmp_path / "offline"
    source_dir.mkdir()
    offline_dir.mkdir()
    _write_rwl(source_dir / "nh999.rwl")
    raw = json.loads(ChronologyIndex(source_dir).manifest_path.read_text())
    del raw["format_version"]
    offline_manifest = offline_dir / ChronologyIndex(source_dir).manifest_path.name
    offline_manifest.write_text(json.dumps(raw))
    before = offline_manifest.read_text()

    assert len(ChronologyIndex(offline_dir)) == 1
    assert offline_manifest.read_text() == before
