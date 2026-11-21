"""Tests for the persisted reference manifest."""

import json
import os
import time
from pathlib import Path

from dendro.reference.chronology_index import ChronologyIndex


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
