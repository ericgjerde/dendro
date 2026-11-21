"""
Parser for Tucson format dendrochronology files (.rwl and .crn).

The Tucson format is the standard format used by the International Tree-Ring
Data Bank (ITRDB). This module handles both:
- .rwl files: Raw ring-width measurements (individual series)
- .crn files: Chronology files (site means with sample depth)

Format specification:
- RWL: cols 1-8 series ID, cols 9-12 start year, then 6-char integer values.
  Stop marker is either 999 (0.01 mm precision) or -9999 (0.001 mm precision).
- CRN: cols 1-6 (standard ITRDB) or 1-8 site ID, next 4 cols decade year,
  then ten 7-char (4-char value + 3-char sample depth) pairs; 9990 = missing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RingWidthSeries:
    """A single ring-width series from an RWL file."""

    series_id: str
    start_year: int
    end_year: int
    values: np.ndarray  # Ring widths (contiguous from start_year..end_year, NaN for gaps)

    @property
    def years(self) -> np.ndarray:
        return np.arange(self.start_year, self.end_year + 1)

    @property
    def length(self) -> int:
        return len(self.values)

    def to_series(self) -> pd.Series:
        """Convert to pandas Series indexed by year."""
        return pd.Series(self.values, index=self.years, name=self.series_id)


@dataclass
class Chronology:
    """A site chronology from a CRN file."""

    site_id: str
    site_name: str
    species: str
    start_year: int
    end_year: int
    values: np.ndarray  # Indexed values (typically scaled to 1000 = 1.0)
    sample_depth: np.ndarray  # Number of samples per year
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None
    state: Optional[str] = None

    @property
    def years(self) -> np.ndarray:
        return np.arange(self.start_year, self.end_year + 1)

    @property
    def length(self) -> int:
        return len(self.values)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with year index."""
        return pd.DataFrame(
            {
                "value": self.values,
                "sample_depth": self.sample_depth,
            },
            index=self.years,
        )


@dataclass
class RWLFile:
    """Container for all series in an RWL file."""

    series: dict[str, RingWidthSeries] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all series to a DataFrame with year index."""
        if not self.series:
            return pd.DataFrame()

        series_list = [s.to_series() for s in self.series.values()]
        df = pd.concat(series_list, axis=1)
        df.index.name = "year"
        return df

    def get_master_chronology(self) -> pd.Series:
        """Calculate mean chronology from all series."""
        df = self.to_dataframe()
        return df.mean(axis=1)


def parse_rwl_file(filepath: str | Path) -> RWLFile:
    """
    Parse a Tucson format .rwl file accurately.

    Respects per-series stop markers (`-9999` vs `999`) so that valid `0.999 mm`
    measurements (`999`) in 0.001 mm (`-9999`) files are never treated as stop
    markers, and maps each value to its explicit line calendar year so internal
    gaps (e.g. broken cores) produce NaNs instead of shifting subsequent years.
    """
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    rwl = RWLFile()
    rwl.metadata["filename"] = filepath.name

    # Collect raw parsed lines per series first so we know each series' stop marker.
    raw_lines_by_series: dict[str, list[tuple[int, list[int]]]] = {}

    for line in lines:
        line = line.rstrip("\n\r")
        if not line or len(line) < 12:
            continue
        if line.startswith("#") or "ITRDB" in line.upper():
            continue

        parsed = _parse_rwl_line(line)
        if parsed is None:
            continue

        series_id, year, values, _ = parsed
        raw_lines_by_series.setdefault(series_id, []).append((year, values))

    for series_id, records in raw_lines_by_series.items():
        # The stop marker is a property of each series, not of the file: ITRDB
        # files can mix 0.01 mm (999) and 0.001 mm (-9999) series.
        series_uses_neg9999 = any(-9999 in vals for _, vals in records)
        stop_marker = -9999 if series_uses_neg9999 else 999

        year_to_value: dict[int, float] = {}
        for line_year, vals in records:
            for offset, v in enumerate(vals):
                if v == stop_marker or v == -9999:
                    break
                # Standard Tucson missing-value sentinels inside active series
                if v < 0:
                    continue
                year_to_value[line_year + offset] = float(v)

        if not year_to_value:
            continue

        start_year = min(year_to_value)
        end_year = max(year_to_value)
        length = end_year - start_year + 1
        values_arr = np.full(length, np.nan, dtype=np.float64)
        for yr, val in year_to_value.items():
            values_arr[yr - start_year] = val

        rwl.series[series_id] = RingWidthSeries(
            series_id=series_id,
            start_year=start_year,
            end_year=end_year,
            values=values_arr,
        )

    return rwl


def _parse_rwl_line(line: str) -> tuple[str, int, list[int], bool] | None:
    """
    Parse a single line of an RWL file.

    Standard Tucson format:
    - Cols 1-8: Series ID (left-justified, space-padded)
    - Cols 9-12: Year (right-justified)
    - Cols 13-18, 19-24, etc.: Ring width values (6 chars each, right-justified)

    Returns:
        Tuple of (series_id, year, values, is_end_marker) or None if unparseable
    """
    if len(line) < 12:
        return None

    series_id = line[:8].strip()
    if not series_id:
        return None

    try:
        year_str = line[8:12].strip()
        year = int(year_str)
    except (ValueError, IndexError):
        return None

    values: list[int] = []
    pos = 12
    is_end = False

    while pos + 6 <= len(line):
        val_str = line[pos:pos + 6].strip()
        if val_str:
            try:
                val = int(val_str)
                values.append(val)
                if val == 999 or val == -9999:
                    is_end = True
            except ValueError:
                pass
        pos += 6

    if pos < len(line):
        val_str = line[pos:].strip()
        if val_str:
            try:
                val = int(val_str)
                values.append(val)
                if val == 999 or val == -9999:
                    is_end = True
            except ValueError:
                pass

    if not values:
        return None

    return series_id, year, values, is_end


def _detect_crn_layout(line: str) -> Optional[tuple[int, int]]:
    """
    Detect CRN fixed-column layout on a candidate data line.

    Returns ``(id_width, data_start)`` where standard ITRDB CRN uses
    ``(6, 10)`` (6-char ID, 4-char year, 7-char value/depth pairs) and
    extended 8-char ID variants use ``(8, 12)``.
    """
    if len(line) < 17 or line.startswith("#"):
        return None

    for id_width in (6, 8):
        year_slice = line[id_width:id_width + 4].strip()
        if not year_slice or not re.fullmatch(r"-?\d{3,4}", year_slice):
            continue
        data_start = id_width + 4
        if len(line) < data_start + 7:
            continue
        # Verify at least the first 7-char pair parses as two integers (val + depth).
        first_val = line[data_start:data_start + 4].strip()
        first_depth = line[data_start + 4:data_start + 7].strip()
        try:
            int(first_val)
            if first_depth:
                int(first_depth)
        except ValueError:
            continue
        # Extra check for standard 6-char layout: reject header lines like "PORSTD 1 ..."
        id_part = line[:id_width].strip()
        if not id_part:
            continue
        return (id_width, data_start)

    return None


def parse_crn_file(filepath: str | Path) -> Chronology | None:
    """
    Parse a Tucson format .crn chronology file.

    Supports standard ITRDB 6-char site-ID format (cols 1-6 ID, 7-10 year,
    11-80 ten 7-char value/depth pairs) as well as 8-char site-ID variants.
    Preserves exact calendar years across leading/internal `9990` missing codes.
    """
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip("\n\r") for line in f.readlines()]

    metadata = _parse_crn_header(lines)

    year_to_val: dict[int, float] = {}
    year_to_depth: dict[int, int] = {}
    site_id: Optional[str] = None
    detected_layout: Optional[tuple[int, int]] = None

    for line in lines:
        if not line or line.startswith("#"):
            continue
        # Skip standard 3-line Tucson headers (e.g. "PORSTD 1 ...", "PORSTD 2 ...", "PORSTD 3 ...")
        if len(line) >= 8 and line[6:8] in {" 1", " 2", " 3"} and not line[8:12].strip().isdigit():
            continue

        layout = detected_layout or _detect_crn_layout(line)
        if layout is None:
            continue

        id_width, data_start = layout
        if len(line) < data_start + 7:
            continue
        if "." in line[data_start:]:
            continue
        line_site_id = line[:id_width].strip()
        if not line_site_id:
            continue
        if site_id is not None and line_site_id != site_id:
            continue

        try:
            decade_year = int(line[id_width:data_start].strip())
        except ValueError:
            continue

        if detected_layout is None:
            detected_layout = layout
        if site_id is None:
            site_id = line_site_id

        pos = data_start
        slot = 0
        # Standard CRN has up to 10 value/depth pairs per line (7 chars each).
        while pos + 7 <= len(line) and slot < 10:
            val_str = line[pos:pos + 4].strip()
            depth_str = line[pos + 4:pos + 7].strip()
            pos += 7

            if not val_str or val_str in {"9990", "-999", "-9999"}:
                slot += 1
                continue

            try:
                val = int(val_str)
                depth = int(depth_str) if depth_str else 1
            except ValueError:
                slot += 1
                continue

            if val == 9990 or val == -999 or depth <= 0:
                slot += 1
                continue

            # On a first-decade line starting mid-decade (e.g. year=1703),
            # some files put year=1703 with values starting at slot 0, while
            # standard ITRDB puts year=1700 with 9990 padding in slots 0..2.
            # Using decade_year + slot handles both cleanly!
            yr = decade_year + slot
            year_to_val[yr] = float(val)
            year_to_depth[yr] = depth
            slot += 1

    if not year_to_val:
        return None

    start_year = min(year_to_val)
    end_year = max(year_to_val)
    length = end_year - start_year + 1
    values_arr = np.full(length, np.nan, dtype=np.float64)
    depth_arr = np.zeros(length, dtype=np.int32)
    for yr, val in year_to_val.items():
        idx = yr - start_year
        values_arr[idx] = val
        depth_arr[idx] = year_to_depth[yr]

    return Chronology(
        site_id=site_id or filepath.stem,
        site_name=metadata.get("site_name", ""),
        species=metadata.get("species", ""),
        start_year=start_year,
        end_year=end_year,
        values=values_arr,
        sample_depth=depth_arr,
        latitude=metadata.get("latitude"),
        longitude=metadata.get("longitude"),
        elevation=metadata.get("elevation"),
        state=metadata.get("state"),
    )


def _parse_crn_header(lines: list[str]) -> dict:
    """Extract metadata from CRN file header lines."""
    metadata = {}

    for line in lines[:20]:
        line_upper = line.upper()

        species_match = re.search(
            r"\b(PIST|TSCA|QUAL|QURU|PCRU|PIRE|PIRI|THOC|ACSA|FRAX|CADN)\b",
            line_upper,
        )
        if species_match:
            metadata["species"] = species_match.group(1)

        lat_match = re.search(r"(\d{2})[°\s](\d{2})['′\s]?\s*N", line, re.IGNORECASE)
        if lat_match:
            metadata["latitude"] = float(lat_match.group(1)) + float(lat_match.group(2)) / 60

        lon_match = re.search(r"(\d{2,3})[°\s](\d{2})['′\s]?\s*W", line, re.IGNORECASE)
        if lon_match:
            metadata["longitude"] = -(float(lon_match.group(1)) + float(lon_match.group(2)) / 60)

        elev_match = re.search(r"(\d{2,4})\s*M\b", line_upper)
        if elev_match:
            metadata["elevation"] = float(elev_match.group(1))

        for state in ["NH", "VT", "ME", "MA", "CT", "RI", "NY", "PA", "TN"]:
            if f" {state} " in line_upper or line_upper.endswith(f" {state}"):
                metadata["state"] = state
                break

    return metadata


def load_measurements_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Load ring width measurements from a CSV file.

    Expected CSV format:
    - Column 'year' or first column: measurement year (optional)
    - Column 'width' or second column: ring width in mm or 0.01mm

    If years are not provided, they are assigned sequentially from 1.

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with 'year' and 'width' columns
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath)

    df.columns = df.columns.str.lower().str.strip()

    if "year" not in df.columns:
        if len(df.columns) >= 2:
            first_col = df.iloc[:, 0]
            if first_col.dtype in [np.int64, np.float64] and first_col.min() > 1000:
                df = df.rename(columns={df.columns[0]: "year"})
            else:
                df["year"] = range(1, len(df) + 1)
        else:
            df["year"] = range(1, len(df) + 1)

    if "width" not in df.columns:
        if "width_mm" in df.columns:
            df = df.rename(columns={"width_mm": "width"})
            return df[
                [col for col in ["year", "calendar_year", "ring_index", "width"] if col in df.columns]
            ].dropna(subset=["width"])
        for col in df.columns:
            if col not in {"year", "calendar_year", "ring_index"} and df[col].dtype in [np.int64, np.float64]:
                df = df.rename(columns={col: "width"})
                break

    if "width" not in df.columns:
        raise ValueError("Could not identify ring width column in CSV")

    keep = [col for col in ["year", "calendar_year", "ring_index", "width"] if col in df.columns]
    if "year" not in keep:
        df["year"] = range(1, len(df) + 1)
        keep = [col for col in ["year", "calendar_year", "ring_index", "width"] if col in df.columns]
    return df[keep].dropna(subset=["width"])


def load_measurement_session(filepath: str | Path) -> pd.DataFrame:
    """Load canonical widths from a saved measurement session JSON file."""
    payload = json.loads(Path(filepath).read_text())
    widths = payload.get("exported_widths_mm_oldest_to_newest", [])
    if not widths:
        raise ValueError("Measurement session does not contain exported widths")
    return pd.DataFrame(
        {
            "ring_index": range(1, len(widths) + 1),
            "width": widths,
        }
    )
