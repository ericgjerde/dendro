import json
from pathlib import Path

import numpy as np

from dendro.reference.chronology_index import ChronologyIndex, ReferenceManifestEntry
from dendro.reference.curated import CuratedChronology, parse_curated_chronology_file


def _write_curated_file(path: Path) -> None:
    payload = {
        "site_id": "WAL001",
        "site_name": "Walpole House Timber",
        "species": "CHTH",
        "state": "NH",
        "material_group": "chestnut",
        "provenance_url": "https://example.invalid/provenance/wal001",
        "study_metadata_url": "https://example.invalid/study/wal001",
        "latitude": 43.076,
        "longitude": -72.423,
        "elevation": 155,
        "source": "partner-curated",
        "chronology": {
            "start_year": 1768,
            "values": [1.0, 1.1, 0.9, 1.2],
            "sample_depth": [4, 4, 3, 2],
            "standardized": True,
            "build_method": "partner-master",
            "detrend_method": "none",
        },
        "warnings": ["synthetic fixture"],
    }
    path.write_text(json.dumps(payload, indent=2))


def test_parse_curated_chronology_file(tmp_path):
    filepath = tmp_path / "walpole-curated.curated.json"
    _write_curated_file(filepath)

    curated = parse_curated_chronology_file(filepath)

    assert isinstance(curated, CuratedChronology)
    assert curated.site_id == "WAL001"
    assert curated.site_name == "Walpole House Timber"
    assert curated.species == "CHTH"
    assert curated.state == "NH"
    assert curated.material_group == "chestnut"
    assert curated.start_year == 1768
    assert curated.end_year == 1771
    assert curated.length == 4
    assert np.allclose(curated.values, np.array([1.0, 1.1, 0.9, 1.2]))
    assert np.array_equal(curated.sample_depth, np.array([4, 4, 3, 2]))
    assert curated.provenance_url.endswith("/wal001")
    assert curated.study_metadata_url.endswith("/wal001")
    assert curated.warnings == ["synthetic fixture"]


def test_chronology_index_loads_curated_json(tmp_path):
    data_dir = tmp_path / "reference"
    data_dir.mkdir()
    filepath = data_dir / "walpole-curated.curated.json"
    _write_curated_file(filepath)

    index = ChronologyIndex()
    count = index.scan_directory(data_dir)

    assert count == 1
    assert len(index.entries) == 1
    entry = index.entries[0]
    assert entry.file_type == "curated_json"
    assert entry.material_group == "chestnut"
    assert entry.species == "CHTH"
    assert entry.state == "NH"
    assert entry.start_year == 1768
    assert entry.end_year == 1771
    assert entry.master is not None
    assert entry.master.start_year == 1768
    assert entry.master.end_year == 1771
    assert entry.master.values == [1.0, 1.1, 0.9, 1.2]
    assert entry.master.sample_depth == [4, 4, 3, 2]
    assert index.get_material_groups() == ["chestnut"]
    assert index.search(material_groups=["chestnut"]) == [entry]

    loaded = index.load_chronology(entry)
    assert isinstance(loaded, CuratedChronology)
    assert loaded.site_id == "WAL001"


def test_curated_manifest_roundtrip_preserves_material_group(tmp_path):
    data_dir = tmp_path / "reference"
    data_dir.mkdir()
    filepath = data_dir / "walpole-curated.curated.json"
    _write_curated_file(filepath)

    index = ChronologyIndex()
    index.scan_directory(data_dir)

    manifest_path = data_dir / ".dendro-reference-manifest.json"
    index.save_manifest(manifest_path)

    reloaded = ChronologyIndex(data_dir=None)
    reloaded.load_manifest(manifest_path)

    assert len(reloaded.entries) == 1
    entry = reloaded.entries[0]
    assert entry.material_group == "chestnut"
    assert entry.file_type == "curated_json"
    assert entry.master is not None
    assert entry.master.values == [1.0, 1.1, 0.9, 1.2]


def test_enrich_missing_metadata_skips_curated_entries(monkeypatch):
    index = ChronologyIndex()
    entry = ReferenceManifestEntry(
        filepath="/tmp/walpole-curated.curated.json",
        site_id="WAL001",
        site_name="",
        species="",
        state="",
        start_year=1768,
        end_year=1771,
        num_years=4,
        num_series=4,
        file_type="curated_json",
        material_group="chestnut",
        parser_warnings=[],
        master=None,
    )
    index._add_entry(entry)

    def _boom(*args, **kwargs):
        raise AssertionError("resolve_reference_metadata should not be called for curated entries")

    monkeypatch.setattr("dendro.reference.chronology_index.resolve_reference_metadata", _boom)

    enriched = index.enrich_missing_metadata()

    assert enriched == 0
    assert index.entries[0].file_type == "curated_json"
