"""
Persisted reference manifest and master chronology index.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..crossdating.detrend import (
    DetrendMethod,
    build_chronology,
    crossdate_transform,
    detrend_series,
    high_pass_filter,
    standardize,
)
from ..materials.catalog import infer_material_group_from_species
from .metadata import resolve_reference_metadata
from .curated import CuratedChronology, parse_curated_chronology_file
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
        cached = getattr(self, "_cached_values", None)
        if cached is None:
            cached = np.asarray(
                [np.nan if v is None else v for v in self.values],
                dtype=np.float64,
            )
            self._cached_values = cached
        return cached

    def crossdate_values_array(self) -> np.ndarray:
        """Return high-frequency prewhitened array cached for cross-dating."""
        cached = getattr(self, "_cached_crossdate_values", None)
        if cached is None:
            raw = self.values_array()
            if self.build_method == "rwl-log-hp20-ar1-biweight":
                cached = raw
            elif len(raw) >= 12 and int(np.sum(~np.isnan(raw))) >= 12:
                try:
                    cached = high_pass_filter(raw, spline_period=20.0, prewhiten_order=1)
                except Exception:
                    cached = raw
            else:
                cached = raw
            self._cached_crossdate_values = cached
        return cached

    def sample_depth_array(self) -> np.ndarray:
        cached = getattr(self, "_cached_depth", None)
        if cached is None:
            cached = np.asarray(self.sample_depth, dtype=np.int32)
            self._cached_depth = cached
        return cached


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
    material_group: str = ""
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
    def master_crossdate_values(self) -> np.ndarray:
        if self.master is None:
            return np.array([], dtype=np.float64)
        return self.master.crossdate_values_array()

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

# Metadata fields a rebuild keeps from the previous manifest when the raw files
# (and any local sidecars) cannot supply them.
_CARRIED_METADATA_FIELDS = (
    "species",
    "state",
    "latitude",
    "longitude",
    "elevation",
    "provenance_url",
    "study_metadata_url",
)


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
        self._by_material_group: dict[str, list[ReferenceManifestEntry]] = {}
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

        # A rebuild must not discard metadata the raw files cannot supply (species and
        # site names fetched from NOAA earlier), or an offline rebuild silently strips it.
        prior_metadata = self._read_prior_metadata(manifest_path, data_dir)
        count = self.scan_directory(
            data_dir,
            allow_remote_metadata=allow_remote_metadata,
            prior_metadata=prior_metadata,
        )
        if manifest_path:
            self.save_manifest(manifest_path)
        return count

    def _read_prior_metadata(self, manifest_path: Optional[Path], data_dir: Path) -> dict[str, dict]:
        """Return the existing manifest's entries keyed by path relative to ``data_dir``."""
        if manifest_path is None or not manifest_path.exists():
            return {}
        try:
            raw_entries = json.loads(manifest_path.read_text()).get("entries", [])
        except Exception:
            return {}

        prior: dict[str, dict] = {}
        for raw_entry in raw_entries:
            raw_path = raw_entry.get("filepath") if isinstance(raw_entry, dict) else None
            if not raw_path:
                continue
            path = Path(raw_path)
            if path.is_absolute():
                try:
                    path = path.relative_to(data_dir)
                except ValueError:
                    continue
            prior[str(path)] = raw_entry
        return prior

    def _carry_forward_metadata(self, entry: ReferenceManifestEntry, prior: Optional[dict]) -> None:
        """Fill metadata this scan could not resolve from the previous manifest entry."""
        if not prior or entry.file_type not in {"rwl", "crn"}:
            return

        carried = False
        stem = Path(entry.filepath).stem
        prior_site_name = prior.get("site_name")
        if entry.site_name in ("", stem) and prior_site_name not in (None, "", stem):
            entry.site_name = prior_site_name
            carried = True
        for name in _CARRIED_METADATA_FIELDS:
            if getattr(entry, name) in (None, "") and prior.get(name) not in (None, ""):
                setattr(entry, name, prior[name])
                carried = True

        if not entry.material_group:
            entry.material_group = infer_material_group_from_species(entry.species)
        if carried and prior.get("metadata_source"):
            entry.metadata_source = prior["metadata_source"]

    def _manifest_stale(self, manifest_path: Path, data_dir: Path) -> bool:
        disk_files = self._reference_files(data_dir)
        if not disk_files:
            # Offline committed-manifest mode (e.g. CI without raw RWL downloads).
            return False

        manifest_mtime = manifest_path.stat().st_mtime
        for filepath in disk_files:
            if filepath.stat().st_mtime > manifest_mtime:
                return True

        try:
            raw = json.loads(manifest_path.read_text())
            manifest_relpaths = {
                str(Path(entry["filepath"]))
                for entry in raw.get("entries", [])
                if "filepath" in entry
            }
            disk_relpaths = {str(filepath.relative_to(data_dir)) for filepath in disk_files}
            if manifest_relpaths != disk_relpaths:
                return True
        except Exception:
            return True

        return False

    def _reference_files(self, data_dir: Path) -> list[Path]:
        files: list[Path] = []
        for filepath in data_dir.rglob("*"):
            if not filepath.is_file():
                continue
            name = filepath.name.lower()
            if name.endswith(".rwl") or name.endswith(".crn") or name.endswith(".curated.json") or name.endswith(".chronology.json"):
                if "-noaa." in name:
                    continue
                if name == MANIFEST_FILENAME:
                    continue
                files.append(filepath)
        return sorted(files)

    def scan_directory(
        self,
        data_dir: str | Path,
        *,
        allow_remote_metadata: bool = False,
        prior_metadata: Optional[dict[str, dict]] = None,
    ) -> int:
        data_dir = Path(data_dir).resolve()
        self.data_dir = data_dir
        self.entries = []
        self._by_species = {}
        self._by_state = {}
        self._by_material_group = {}

        if not data_dir.exists():
            return 0

        count = 0
        for filepath in self._reference_files(data_dir):
            entry = self._index_file(filepath, allow_remote_metadata=allow_remote_metadata)
            if entry is None:
                continue
            if prior_metadata:
                self._carry_forward_metadata(entry, prior_metadata.get(str(filepath.relative_to(data_dir))))
            self._add_entry(entry)
            count += 1

        return count

    def _index_file(
        self,
        filepath: Path,
        *,
        allow_remote_metadata: bool = False,
    ) -> Optional[ReferenceManifestEntry]:
        name = filepath.name.lower()
        if name.endswith(".curated.json") or name.endswith(".chronology.json"):
            curated = parse_curated_chronology_file(filepath)
            if curated is None:
                return None
            warnings = list(curated.warnings)
            master = MasterChronology(
                start_year=curated.start_year,
                end_year=curated.end_year,
                values=np.asarray(curated.values, dtype=np.float64).tolist(),
                sample_depth=np.asarray(curated.sample_depth, dtype=np.int32).tolist(),
                build_method=curated.build_method,
                detrend_method=curated.detrend_method,
                standardized=curated.standardized,
            )
            site_name = curated.site_name or filepath.stem
            return ReferenceManifestEntry(
                filepath=str(filepath),
                site_id=curated.site_id or filepath.stem[:8].upper(),
                site_name=site_name,
                species=curated.species,
                state=curated.state,
                start_year=curated.start_year,
                end_year=curated.end_year,
                num_years=curated.length,
                num_series=max(1, int(np.nanmax(curated.sample_depth)) if len(curated.sample_depth) else 1),
                file_type="curated_json",
                material_group=curated.material_group or infer_material_group_from_species(curated.species),
                latitude=curated.latitude,
                longitude=curated.longitude,
                elevation=curated.elevation,
                provenance_url=curated.provenance_url,
                study_metadata_url=curated.study_metadata_url,
                metadata_source=curated.source,
                parser_warnings=warnings,
                master=master,
            )

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
            material_group=infer_material_group_from_species(resolved.species),
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
        serialized = [None if np.isnan(val) else float(val) for val in standardized]
        return MasterChronology(
            start_year=chronology.start_year,
            end_year=chronology.end_year,
            values=serialized,
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
                standardized = crossdate_transform(series.values, spline_period=20.0, prewhiten_order=1)
                if int(np.sum(~np.isnan(standardized))) < 10:
                    warnings.append(f"Skipped sparse series {series_id}.")
                    continue
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

        # Trim leading/trailing NaNs only; preserve internal gaps so calendar years never shift.
        first_valid = int(np.argmax(valid))
        last_valid = len(valid) - 1 - int(np.argmax(valid[::-1]))
        master_years = master_years[first_valid:last_valid + 1]
        master_values = standardize(master_values[first_valid:last_valid + 1], method="zscore")
        sample_depth = sample_depth[first_valid:last_valid + 1]

        serialized_values = [
            None if np.isnan(val) else float(val)
            for val in master_values
        ]
        master = MasterChronology(
            start_year=int(master_years[0]),
            end_year=int(master_years[-1]),
            values=serialized_values,
            sample_depth=np.asarray(sample_depth, dtype=np.int32).tolist(),
            build_method="rwl-log-hp20-ar1-biweight",
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
        if entry.material_group:
            self._by_material_group.setdefault(entry.material_group, []).append(entry)

    def search(
        self,
        species: Optional[list[str]] = None,
        states: Optional[list[str]] = None,
        material_groups: Optional[list[str]] = None,
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

        if material_groups:
            material_groups = [group.lower() for group in material_groups]
            candidates = [entry for entry in candidates if entry.material_group in material_groups]

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

    def get_material_groups(self) -> list[str]:
        return sorted(self._by_material_group.keys())

    def load_chronology(self, metadata: ReferenceManifestEntry) -> Chronology | CuratedChronology | RWLFile | None:
        filepath = Path(metadata.filepath)
        if metadata.file_type == "curated_json":
            return parse_curated_chronology_file(filepath)
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
        self._by_material_group = {}

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
            if entry.file_type not in {"rwl", "crn"}:
                refreshed_entries.append(entry)
                continue

            if entry.species and entry.site_name and entry.site_name != Path(entry.filepath).stem:
                refreshed_entries.append(entry)
                continue

            resolved = resolve_reference_metadata(entry.filepath, entry.file_type, allow_remote=True)
            if resolved.species and not entry.species:
                entry.species = resolved.species
                enriched += 1
            if not entry.material_group and entry.species:
                entry.material_group = infer_material_group_from_species(entry.species)
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
            self._by_material_group = {}
            for entry in refreshed_entries:
                self._add_entry(entry)
        return enriched

    def _entry_to_dict(self, entry: ReferenceManifestEntry) -> dict:
        payload = asdict(entry)
        filepath = Path(entry.filepath).resolve()
        if self.data_dir is not None:
            try:
                payload["filepath"] = str(filepath.relative_to(self.data_dir.resolve()))
            except ValueError:
                payload["filepath"] = str(entry.filepath)
        return payload

    def _entry_from_dict(self, raw_entry: dict) -> ReferenceManifestEntry:
        raw_entry = dict(raw_entry)
        master = raw_entry.get("master")
        if master is not None:
            raw_entry["master"] = MasterChronology(**master)
        filepath = Path(raw_entry["filepath"])
        if not filepath.is_absolute() and self.data_dir is not None:
            raw_entry["filepath"] = str(self.data_dir / filepath)
        if not raw_entry.get("material_group") and raw_entry.get("species"):
            raw_entry["material_group"] = infer_material_group_from_species(raw_entry["species"])
        return ReferenceManifestEntry(**raw_entry)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"ChronologyIndex({len(self.entries)} entries)"
