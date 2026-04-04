"""Curated reference chronology ingestion.

This module handles non-ITRDB JSON chronologies that are authored or
assembled outside the Tucson/NOAA file formats. The intent is to support
curated masters with explicit provenance, geographic metadata, species,
material-group hints, and chronology values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CuratedChronology:
    """A curated site chronology loaded from JSON."""

    site_id: str
    site_name: str
    species: str
    state: str
    material_group: str
    start_year: int
    end_year: int
    values: np.ndarray
    sample_depth: np.ndarray
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None
    provenance_url: str = ""
    study_metadata_url: str = ""
    source: str = "curated-json"
    standardized: bool = True
    build_method: str = "curated-json-master"
    detrend_method: str = "none"
    warnings: list[str] = field(default_factory=list)

    @property
    def years(self) -> np.ndarray:
        return np.arange(self.start_year, self.end_year + 1)

    @property
    def length(self) -> int:
        return len(self.values)

    def to_dataframe(self):
        """Return a pandas DataFrame for convenience in tests or callers."""
        import pandas as pd

        return pd.DataFrame(
            {
                "value": self.values,
                "sample_depth": self.sample_depth,
            },
            index=self.years,
        )


def _coerce_text(value: object, *, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _coerce_bool(value: object, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return default


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float_array(values: object, *, allow_empty: bool = False) -> Optional[np.ndarray]:
    if values is None:
        return np.array([], dtype=np.float64) if allow_empty else None

    if isinstance(values, np.ndarray):
        array = values.astype(np.float64, copy=False)
    else:
        try:
            array = np.asarray(list(values), dtype=np.float64)
        except TypeError:
            try:
                array = np.asarray([float(values)], dtype=np.float64)
            except (TypeError, ValueError):
                return None
        except ValueError:
            return None

    if array.size == 0 and not allow_empty:
        return None
    return array


def _coerce_int_array(values: object, *, length: int) -> Optional[np.ndarray]:
    if values is None:
        return np.ones(length, dtype=np.int32)

    if isinstance(values, np.ndarray):
        array = values.astype(np.int32, copy=False)
    else:
        try:
            array = np.asarray(list(values), dtype=np.int32)
        except TypeError:
            try:
                array = np.asarray([int(values)], dtype=np.int32)
            except (TypeError, ValueError):
                return None
        except ValueError:
            return None

    if array.size == 1 and length > 1:
        return np.full(length, int(array[0]), dtype=np.int32)
    if array.size != length:
        return None
    return array


def _choose_first(*values: object) -> object:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return value
            continue
        try:
            if len(value) == 0:  # type: ignore[arg-type]
                continue
        except TypeError:
            pass
        return value
    return None


def parse_curated_chronology_file(filepath: str | Path) -> CuratedChronology | None:
    """Parse a JSON-based curated chronology file.

    Supported schema:
    - top-level metadata fields or a nested ``chronology`` object
    - ``values`` or ``chronology.values`` ring-width/index values
    - optional ``sample_depth`` or ``chronology.sample_depth``
    - ``start_year``/``end_year`` or a contiguous ``years`` list
    """

    filepath = Path(filepath)
    try:
        payload = json.loads(filepath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    chronology = payload.get("chronology")
    if chronology is None:
        chronology = payload
    if not isinstance(chronology, dict):
        return None

    site_id = _coerce_text(_choose_first(payload.get("site_id"), chronology.get("site_id"), filepath.stem[:8].upper()), default=filepath.stem[:8].upper())
    site_name = _coerce_text(_choose_first(payload.get("site_name"), chronology.get("site_name"), filepath.stem), default=filepath.stem)
    species = _coerce_text(_choose_first(payload.get("species"), chronology.get("species")), default="")
    state = _coerce_text(_choose_first(payload.get("state"), chronology.get("state")), default="")
    material_group = _coerce_text(_choose_first(payload.get("material_group"), chronology.get("material_group")), default="").lower()

    values = _coerce_float_array(_choose_first(payload.get("values"), chronology.get("values")))
    if values is None or values.size == 0:
        return None

    years = _choose_first(payload.get("years"), chronology.get("years"))
    start_year = _coerce_int(_choose_first(payload.get("start_year"), chronology.get("start_year")))
    end_year = _coerce_int(_choose_first(payload.get("end_year"), chronology.get("end_year")))

    if years is not None:
        years_array = _coerce_float_array(years)
        if years_array is None or years_array.size != values.size:
            return None
        years_int = years_array.astype(np.int32)
        if not np.all(np.diff(years_int) == 1):
            return None
        start_year = int(years_int[0])
        end_year = int(years_int[-1])
    else:
        if start_year is None:
            return None
        if end_year is None:
            end_year = start_year + len(values) - 1
        if end_year < start_year:
            return None
        if end_year - start_year + 1 != len(values):
            return None

    sample_depth = _coerce_int_array(_choose_first(payload.get("sample_depth"), chronology.get("sample_depth")), length=len(values))
    if sample_depth is None:
        return None

    standardized = _coerce_bool(_choose_first(payload.get("standardized"), chronology.get("standardized")), default=True)
    build_method = _coerce_text(_choose_first(payload.get("build_method"), chronology.get("build_method")), default="curated-json-master")
    detrend_method = _coerce_text(_choose_first(payload.get("detrend_method"), chronology.get("detrend_method")), default="none")
    source = _coerce_text(_choose_first(payload.get("source"), chronology.get("source")), default="curated-json")
    provenance_url = _coerce_text(_choose_first(payload.get("provenance_url"), chronology.get("provenance_url")), default="")
    study_metadata_url = _coerce_text(_choose_first(payload.get("study_metadata_url"), chronology.get("study_metadata_url")), default="")
    latitude = _coerce_float(_choose_first(payload.get("latitude"), chronology.get("latitude")))
    longitude = _coerce_float(_choose_first(payload.get("longitude"), chronology.get("longitude")))
    elevation = _coerce_float(_choose_first(payload.get("elevation"), chronology.get("elevation")))

    warnings: list[str] = []
    raw_warnings = _choose_first(payload.get("warnings"), chronology.get("warnings"))
    if isinstance(raw_warnings, str):
        warnings.append(raw_warnings)
    elif raw_warnings is not None:
        try:
            warnings.extend(str(item) for item in raw_warnings)
        except TypeError:
            warnings.append(str(raw_warnings))

    return CuratedChronology(
        site_id=site_id,
        site_name=site_name,
        species=species,
        state=state,
        material_group=material_group,
        start_year=int(start_year),
        end_year=int(end_year),
        values=values,
        sample_depth=sample_depth,
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        provenance_url=provenance_url,
        study_metadata_url=study_metadata_url,
        source=source,
        standardized=standardized,
        build_method=build_method,
        detrend_method=detrend_method,
        warnings=warnings,
    )
