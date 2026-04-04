"""
Persisted reference manifest and master chronology index.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..crossdating.detrend import DetrendMethod, build_chronology, detrend_series, standardize
from .metadata import resolve_reference_metadata
from .tucson_parser import Chronology, RWLFile, parse_crn_file, parse_rwl_file


MANIFEST_FILENAME = ".dendro-reference-manifest.json"


@dataclass
class MasterChronology:
    """Persisted master chronology used for ranking."""

    start_year: int
    end_year: int
    values: list[float]
    sample_depth: list[int]
    build_method: str
    detrend_method: str
    standardized: bool = True

    @property
    def length(self) -> int:
        return len(self.values)

    def values_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)

    def sample_depth_array(self) -> np.ndarray:
        return np.asarray(self.sample_depth, dtype=np.int32)


@dataclass
class ReferenceManifestEntry:
    """Structured metadata for a locally indexed reference chronology."""

    filepath: str
    site_id: str
    site_name: str
    species: str
    state: str
    start_year: int
    end_year: int
    num_years: int
    num_series: int
    file_type: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None
    provenance_url: str = ""
    study_metadata_url: str = ""
    metadata_source: str = ""
    parser_warnings: list[str] = field(default_factory=list)
    master: Optional[MasterChronology] = None

    def covers_period(self, start: int, end: int) -> bool:
        return self.start_year <= start and self.end_year >= end

    def overlap_years(self, start: int, end: int) -> int:
        overlap_start = max(self.start_year, start)
        overlap_end = min(self.end_year, end)
        return max(0, overlap_end - overlap_start + 1)

    @property
    def master_values(self) -> np.ndarray:
        if self.master is None:
            return np.array([], dtype=np.float64)
        return self.master.values_array()

    @property
    def master_start_year(self) -> int:
        if self.master is None:
            return self.start_year
        return self.master.start_year

    @property
    def master_end_year(self) -> int:
        if self.master is None:
            return self.end_year
        return self.master.end_year


# Backwards-compatible alias used by older tests/imports.
ChronologyMetadata = ReferenceManifestEntry


class ChronologyIndex:
    """
    Reference chronology index backed by a persisted manifest.
    """

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
        *,
        allow_remote_metadata: bool = False,
        manifest_name: str = MANIFEST_FILENAME,
    ):
        self.entries: list[ReferenceManifestEntry] = []
        self._by_species: dict[str, list[ReferenceManifestEntry]] = {}
        self._by_state: dict[str, list[ReferenceManifestEntry]] = {}
        self.data_dir: Optional[Path] = Path(data_dir).resolve() if data_dir is not None else None
        self.manifest_name = manifest_name

        if self.data_dir is not None:
            self.load_or_build(self.data_dir, allow_remote_metadata=allow_remote_metadata)

    @property
    def manifest_path(self) -> Optional[Path]:
        if self.data_dir is None:
            return None
        return self.data_dir / self.manifest_name

    def load_or_build(self, data_dir: str | Path, *, allow_remote_metadata: bool = False) -> int:
        data_dir = Path(data_dir).resolve()
        self.data_dir = data_dir

        manifest_path = self.manifest_path
        if manifest_path and manifest_path.exists() and not self._manifest_stale(manifest_path, data_dir):
            self.load_manifest(manifest_path)
            if allow_remote_metadata and self.enrich_missing_metadata():
                self.save_manifest(manifest_path)
            return len(self.entries)

        count = self.scan_directory(data_dir, allow_remote_metadata=allow_remote_metadata)
        if manifest_path:
            self.save_manifest(manifest_path)
        return count

    def _manifest_stale(self, manifest_path: Path, data_dir: Path) -> bool:
        manifest_mtime = manifest_path.stat().st_mtime
        for filepath in self._reference_files(data_dir):
            if filepath.stat().st_mtime > manifest_mtime:
                return True
        return False

    def _reference_files(self, data_dir: Path) -> list[Path]:
        files: list[Path] = []
        for filepath in data_dir.rglob("*"):
            if not filepath.is_file():
                continue
            name = filepath.name.lower()
            if name.endswith(".rwl") or name.endswith(".crn"):
                if "-noaa." in name:
                    continue
                files.append(filepath)
        return sorted(files)

    def scan_directory(self, data_dir: str | Path, *, allow_remote_metadata: bool = False) -> int:
        data_dir = Path(data_dir)
        self.entries = []
        self._by_species = {}
        self._by_state = {}

        if not data_dir.exists():
            return 0

        count = 0
        for filepath in self._reference_files(data_dir):
            entry = self._index_file(filepath, allow_remote_metadata=allow_remote_metadata)
            if entry is None:
                continue
            self._add_entry(entry)
            count += 1

        return count

    def _index_file(
        self,
        filepath: Path,
        *,
        allow_remote_metadata: bool = False,
    ) -> Optional[ReferenceManifestEntry]:
        file_type = "crn" if filepath.suffix.lower() == ".crn" else "rwl"
        resolved = resolve_reference_metadata(filepath, file_type, allow_remote=allow_remote_metadata)

        warnings = list(resolved.warnings)
        master: Optional[MasterChronology] = None
        start_year = 0
        end_year = 0
        num_years = 0
        num_series = 0

        if file_type == "crn":
            chronology = parse_crn_file(filepath)
            if chronology is None:
                return None
            start_year = chronology.start_year
            end_year = chronology.end_year
            num_years = chronology.length
            num_series = int(np.nanmax(chronology.sample_depth)) if len(chronology.sample_depth) else 1
            master = self._build_crn_master(chronology)
        else:
            rwl = parse_rwl_file(filepath)
            if not rwl.series:
                return None
            start_year, end_year, num_years, num_series, master, build_warnings = self._build_rwl_entry(rwl)
            warnings.extend(build_warnings)

        if master is None or not master.values:
            warnings.append("No usable master chronology could be constructed.")

        site_name = resolved.site_name or filepath.stem
        species = resolved.species
        state = resolved.state

        return ReferenceManifestEntry(
            filepath=str(filepath),
            site_id=resolved.site_id or filepath.stem[:8].upper(),
            site_name=site_name,
            species=species,
            state=state,
            start_year=start_year,
            end_year=end_year,
            num_years=num_years,
            num_series=num_series,
            file_type=file_type,
            latitude=resolved.latitude,
            longitude=resolved.longitude,
            elevation=resolved.elevation,
            provenance_url=resolved.provenance_url,
            study_metadata_url=resolved.study_metadata_url,
            metadata_source=resolved.source,
            parser_warnings=warnings,
            master=master,
        )

    def _build_crn_master(self, chronology: Chronology) -> Optional[MasterChronology]:
        values = np.asarray(chronology.values, dtype=np.float64)
        depth = np.asarray(chronology.sample_depth, dtype=np.int32)
        if len(values) == 0:
            return None

        if np.nanmean(values) > 50:
            values = values / 1000.0

        standardized = standardize(values, method="zscore")
        return MasterChronology(
            start_year=chronology.start_year,
            end_year=chronology.end_year,
            values=standardized.tolist(),
            sample_depth=depth.tolist(),
            build_method="crn-index",
            detrend_method="none",
            standardized=True,
        )

    def _build_rwl_entry(
        self,
        rwl: RWLFile,
    ) -> tuple[int, int, int, int, Optional[MasterChronology], list[str]]:
        warnings: list[str] = []
        series_list: list[np.ndarray] = []
        years_list: list[np.ndarray] = []

        start_year = min(series.start_year for series in rwl.series.values())
        end_year = max(series.end_year for series in rwl.series.values())
        num_years = end_year - start_year + 1
        num_series = len(rwl.series)

        for series_id, series in rwl.series.items():
            try:
                if series.length < 10:
                    warnings.append(f"Skipped short series {series_id} ({series.length} rings).")
                    continue
                detrended, _ = detrend_series(series.values, method=DetrendMethod.SPLINE)
                standardized = standardize(detrended, method="zscore")
                years = np.arange(series.start_year, series.end_year + 1)
                series_list.append(standardized)
                years_list.append(years)
            except Exception as exc:
                warnings.append(f"Skipped series {series_id}: {exc}")

        if not series_list:
            return start_year, end_year, num_years, num_series, None, warnings

        master_years, master_values, sample_depth = build_chronology(
            series_list,
            years_list,
            method="biweight",
        )

        valid = ~np.isnan(master_values)
        if not np.any(valid):
            return start_year, end_year, num_years, num_series, None, warnings

        master_years = master_years[valid]
        master_values = master_values[valid]
        sample_depth = sample_depth[valid]

        master = MasterChronology(
            start_year=int(master_years[0]),
            end_year=int(master_years[-1]),
            values=np.asarray(master_values, dtype=np.float64).tolist(),
            sample_depth=np.asarray(sample_depth, dtype=np.int32).tolist(),
            build_method="rwl-biweight-master",
            detrend_method=DetrendMethod.SPLINE.value,
            standardized=True,
        )
        return int(master_years[0]), int(master_years[-1]), len(master_years), num_series, master, warnings

    def _add_entry(self, entry: ReferenceManifestEntry):
        self.entries.append(entry)

        if entry.species:
            self._by_species.setdefault(entry.species, []).append(entry)
        if entry.state:
            self._by_state.setdefault(entry.state, []).append(entry)

    def search(
        self,
        species: Optional[list[str]] = None,
        states: Optional[list[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        min_overlap: int = 30,
        file_type: Optional[str] = None,
    ) -> list[ReferenceManifestEntry]:
        results: list[ReferenceManifestEntry] = []
        candidates = self.entries

        if species:
            species = [s.upper() for s in species]
            candidates = [entry for entry in candidates if entry.species in species]

        if states:
            states = [s.upper() for s in states]
            candidates = [entry for entry in candidates if entry.state in states]

        if file_type:
            candidates = [entry for entry in candidates if entry.file_type == file_type]

        for entry in candidates:
            if entry.master is None or not entry.master.values:
                continue

            if min_year is not None and max_year is not None:
                overlap = entry.overlap_years(min_year, max_year)
                if overlap < min_overlap:
                    continue
            elif min_year is not None and entry.end_year < min_year:
                continue
            elif max_year is not None and entry.start_year > max_year:
                continue

            results.append(entry)

        if min_year is not None and max_year is not None:
            results.sort(key=lambda entry: entry.overlap_years(min_year, max_year), reverse=True)
        else:
            results.sort(key=lambda entry: entry.num_years, reverse=True)

        return results

    def get_species(self) -> list[str]:
        return sorted(self._by_species.keys())

    def get_states(self) -> list[str]:
        return sorted(self._by_state.keys())

    def load_chronology(self, metadata: ReferenceManifestEntry) -> Chronology | RWLFile | None:
        filepath = Path(metadata.filepath)
        if metadata.file_type == "crn":
            return parse_crn_file(filepath)
        return parse_rwl_file(filepath)

    def save_manifest(self, filepath: str | Path):
        filepath = Path(filepath)
        payload = {
            "entries": [self._entry_to_dict(entry) for entry in self.entries],
        }
        filepath.write_text(json.dumps(payload, indent=2))

    def load_manifest(self, filepath: str | Path):
        filepath = Path(filepath)
        data = json.loads(filepath.read_text())
        self.entries = []
        self._by_species = {}
        self._by_state = {}

        for raw_entry in data.get("entries", []):
            entry = self._entry_from_dict(raw_entry)
            self._add_entry(entry)

    def enrich_missing_metadata(self) -> int:
        """
        Fill missing metadata fields from NOAA sidecars and persist the improvements.
        """
        enriched = 0
        refreshed_entries: list[ReferenceManifestEntry] = []

        for entry in self.entries:
            if entry.species and entry.site_name and entry.site_name != Path(entry.filepath).stem:
                refreshed_entries.append(entry)
                continue

            resolved = resolve_reference_metadata(entry.filepath, entry.file_type, allow_remote=True)
            if resolved.species and not entry.species:
                entry.species = resolved.species
                enriched += 1
            if resolved.site_name and (not entry.site_name or entry.site_name == Path(entry.filepath).stem):
                entry.site_name = resolved.site_name
                enriched += 1
            if resolved.state and not entry.state:
                entry.state = resolved.state
            if resolved.latitude is not None and entry.latitude is None:
                entry.latitude = resolved.latitude
            if resolved.longitude is not None and entry.longitude is None:
                entry.longitude = resolved.longitude
            if resolved.elevation is not None and entry.elevation is None:
                entry.elevation = resolved.elevation
            if resolved.provenance_url and not entry.provenance_url:
                entry.provenance_url = resolved.provenance_url
            if resolved.study_metadata_url and not entry.study_metadata_url:
                entry.study_metadata_url = resolved.study_metadata_url
            if resolved.source:
                entry.metadata_source = resolved.source
            entry.parser_warnings.extend(resolved.warnings)
            refreshed_entries.append(entry)

        if enriched:
            self.entries = []
            self._by_species = {}
            self._by_state = {}
            for entry in refreshed_entries:
                self._add_entry(entry)
        return enriched

    def _entry_to_dict(self, entry: ReferenceManifestEntry) -> dict:
        payload = asdict(entry)
        filepath = Path(entry.filepath)
        if self.data_dir is not None:
            try:
                payload["filepath"] = str(filepath.relative_to(self.data_dir))
            except ValueError:
                payload["filepath"] = str(filepath)
        return payload

    def _entry_from_dict(self, raw_entry: dict) -> ReferenceManifestEntry:
        raw_entry = dict(raw_entry)
        master = raw_entry.get("master")
        if master is not None:
            raw_entry["master"] = MasterChronology(**master)
        filepath = Path(raw_entry["filepath"])
        if not filepath.is_absolute() and self.data_dir is not None:
            raw_entry["filepath"] = str(self.data_dir / filepath)
        return ReferenceManifestEntry(**raw_entry)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"ChronologyIndex({len(self.entries)} entries)"
