"""
Parser for Tucson format dendrochronology files (.rwl and .crn).

The Tucson format is the standard format used by the International Tree-Ring
Data Bank (ITRDB). This module handles both:
- .rwl files: Raw ring-width measurements (individual series)
- .crn files: Chronology files (site means with sample depth)

Format specification:
- RWL: cols 1-8 series ID, cols 9-12 start year, then 6-char integer values.
  Stop marker is either 999 (0.01 mm precision) or -9999 (0.001 mm precision).
- CRN: cols 1-6 (standard ITRDB) or 1-8 site ID, next 4 cols decade year
  (the first line may carry the true first year instead), then ten 7-char
  (4-char value + 3-char sample depth) pairs; 9990 = missing.
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
    raw_lines_by_series: dict[str, list[tuple[int, list[Optional[int]]]]] = {}

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
                # Blank or unreadable field: the year stays a gap (NaN).
                if v is None:
                    continue
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


def _parse_rwl_line(line: str) -> tuple[str, int, list[Optional[int]], bool] | None:
    """
    Parse a single line of an RWL file.

    Standard Tucson format:
    - Cols 1-8: Series ID (left-justified, space-padded)
    - Cols 9-12: Year (right-justified)
    - Cols 13-18, 19-24, etc.: Ring width values (6 chars each, right-justified)

    Values are returned by fixed-width slot position: ``values[i]`` belongs to
    ``year + i``. A blank or non-numeric field followed by later values is
    returned as ``None`` so it leaves a gap instead of shifting later years.

    Returns:
        Tuple of (series_id, year, values, is_end_marker) or None if unparseable
    """
    if len(line) < 12:
        return None

    series_id = line[:8].strip()
    year_str = line[8:12].strip()
    # A 5-character negative year ("ABC01  -1005") borrows column 8 from the
    # ID field. Only accepted when a space separates the ID from the sign.
    if line[6:8] == " -" and line[8:12].isdigit():
        series_id = line[:7].strip()
        year_str = "-" + line[8:12]
    if not series_id:
        return None

    try:
        year = int(year_str)
    except (ValueError, IndexError):
        return None

    values: list[Optional[int]] = []
    is_end = False

    for pos in range(12, len(line), 6):
        val_str = line[pos:pos + 6].strip()
        val: Optional[int] = None
        if val_str:
            try:
                val = int(val_str)
            except ValueError:
                val = None
        if val == 999 or val == -9999:
            is_end = True
        values.append(val)

    while values and values[-1] is None:
        values.pop()

    if not values:
        return None

    return series_id, year, values, is_end


# CRN integers are written right-justified in their fixed-width fields.
_CRN_INT_FIELD = re.compile(r" *-?\d+")
_CRN_MISSING_VALUES = {9990, -999}

# One CRN slot: (index value, sample depth), or None for padding/missing.
_CRNSlot = Optional[tuple[int, int]]


def _parse_crn_data_line(line: str, id_width: int) -> Optional[tuple[str, int, list[_CRNSlot], bool]]:
    """
    Split a CRN line under a fixed-column layout (``id_width`` is 6 or 8).

    Returns ``(line_id, year, slots, clean)`` or None when the line is not a
    data line under this layout (header, statistics trailer, comment, text).
    ``slots[i]`` is the i-th 7-char value/depth pair, None when it is padding
    or missing. ``clean`` is True when every pair is a right-justified integer
    pair and the line ends on a pair boundary, which is how a line read under
    the correct layout looks.
    """
    data_start = id_width + 4
    if line.startswith("#") or len(line) < data_start + 7:
        return None

    line_id = line[:id_width].strip()
    year_field = line[id_width:data_start]
    # The three header lines carry " 1 ", " 2 ", " 3 " after the ID, which is
    # never a right-justified integer, so they are rejected here without a
    # year-range rule that would also drop early-year data lines.
    if not line_id or not _CRN_INT_FIELD.fullmatch(year_field):
        return None
    if "." in line[data_start:]:
        return None

    # Standard CRN has up to 10 value/depth pairs per line (7 chars each).
    body = line[data_start:data_start + 70]
    clean = len(line.rstrip()) <= data_start + 70 and len(body.rstrip()) % 7 == 0
    slots: list[_CRNSlot] = []
    numeric_pairs = 0
    for pos in range(0, len(body) - 6, 7):
        val_field = body[pos:pos + 4]
        depth_field = body[pos + 4:pos + 7]
        if not _CRN_INT_FIELD.fullmatch(val_field) or (
            depth_field.strip() and not _CRN_INT_FIELD.fullmatch(depth_field)
        ):
            clean = False
        try:
            val = int(val_field.strip())
            depth = int(depth_field.strip()) if depth_field.strip() else 1
        except ValueError:
            # Blank or unreadable pair: keep the slot so later years do not shift.
            slots.append(None)
            continue
        numeric_pairs += 1
        if val in _CRN_MISSING_VALUES or depth <= 0:
            slots.append(None)
        else:
            slots.append((val, depth))

    if not numeric_pairs:
        return None

    return line_id, int(year_field), slots, clean


def _score_crn_layout(lines: list[str], id_width: int) -> Optional[tuple[int, int, int]]:
    """
    Score how well a fixed-column layout explains the data lines of a file.

    Returns ``(decade_steps, clean_lines, data_lines)``: the number of
    consecutive data lines whose year advances to the next decade, the number
    of lines that parse cleanly, and the number of data lines. None when no
    line parses as data under this layout.
    """
    parsed = [p for p in (_parse_crn_data_line(line, id_width) for line in lines) if p is not None]
    if not parsed:
        return None
    first_id = parsed[0][0]
    parsed = [p for p in parsed if p[0] == first_id]
    decade_steps = sum(
        1
        for prev, nxt in zip(parsed, parsed[1:])
        if nxt[1] == prev[1] - prev[1] % 10 + 10
    )
    clean_lines = sum(1 for p in parsed if p[3])
    return decade_steps, clean_lines, len(parsed)


def _detect_crn_layout(lines: list[str]) -> Optional[tuple[int, int]]:
    """
    Detect the CRN fixed-column layout from all data lines of a file.

    Returns ``(id_width, data_start)`` where standard ITRDB CRN uses
    ``(6, 10)`` (6-char ID, 4-char year, 7-char value/depth pairs) and
    extended 8-char ID variants use ``(8, 12)``.

    A single line cannot settle this: an 8-char ID ending in digits also reads
    as a 6-char ID followed by a year. The layout whose years advance one
    decade per line, and whose value/depth pairs are well formed, wins; the
    standard 6-char layout wins ties.
    """
    best: Optional[tuple[tuple[int, int, int], int]] = None
    for id_width in (6, 8):
        score = _score_crn_layout(lines, id_width)
        if score is not None and (best is None or score > best[0]):
            best = (score, id_width)
    if best is None:
        return None
    return best[1], best[1] + 4


def _crn_header_site_id(lines: list[str], id_width: int) -> Optional[str]:
    """
    Site ID from the Tucson header lines that precede the data.

    Header lines are ``<6-char ID> 1 ...``, ``<ID> 2 ...``, ``<ID> 3 ...``.
    Data lines of ARSTAN-written files carry the chronology type instead
    (``STNDRD``, ``RESID``, ``ARSTAN``), so the header is the site-specific ID.
    """
    for line in lines:
        if _parse_crn_data_line(line, id_width) is not None:
            break
        if line[6:9] in {" 1 ", " 2 ", " 3 "} and line[:6].strip():
            return line[:6].strip()
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
    data_line_id: Optional[str] = None

    layout = _detect_crn_layout(lines)
    if layout is None:
        return None
    id_width, _ = layout

    for line in lines:
        parsed = _parse_crn_data_line(line, id_width)
        if parsed is None:
            continue

        line_id, line_year, slots, _ = parsed
        if data_line_id is None:
            data_line_id = line_id
        elif line_id != data_line_id:
            continue

        # Slot 0 normally holds the year in the year field. A first line that
        # starts mid-decade comes in two forms:
        #   - the year field is the true first year but the slots stay
        #     decade-aligned, with 9990 padding before it (1641 -> slot 1);
        #   - the values simply start at slot 0 (RWL style).
        # Leading padding up to the year's position in its decade identifies
        # the first form. Python's floor modulo keeps negative years right.
        offset = line_year % 10
        if offset and all(cell is None for cell in slots[:offset]):
            line_year -= offset

        for slot, cell in enumerate(slots):
            if cell is None:
                continue
            year_to_val[line_year + slot] = float(cell[0])
            year_to_depth[line_year + slot] = cell[1]

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
        site_id=_crn_header_site_id(lines, id_width) or data_line_id or filepath.stem,
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
