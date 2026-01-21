"""
Parser for Tucson format dendrochronology files (.rwl and .crn).

The Tucson format is the standard format used by the International Tree-Ring
Data Bank (ITRDB). This module handles both:
- .rwl files: Raw ring-width measurements (individual series)
- .crn files: Chronology files (site means with sample depth)

Format specification:
- Fixed-width columns
- First 8 characters: series ID
- Characters 9-12: start year (or decade for some formats)
- Remaining: ring width values (varies by format)
"""

from __future__ import annotations

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
    values: np.ndarray  # Ring widths in 0.01mm units

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
        return pd.DataFrame({
            "value": self.values,
            "sample_depth": self.sample_depth,
        }, index=self.years)


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
    Parse a Tucson format .rwl file.

    Handles multiple common Tucson format variants:
    - Standard ITRDB format with fixed columns
    - NOAA NCEI modified format

    Args:
        filepath: Path to the .rwl file

    Returns:
        RWLFile containing all parsed series
    """
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    rwl = RWLFile()
    rwl.metadata["filename"] = filepath.name

    # Track data for each series as we parse
    series_data: dict[str, dict] = {}

    for line in lines:
        line = line.rstrip("\n\r")

        # Skip empty lines and obvious header lines
        if not line or len(line) < 12:
            continue

        # Skip header lines (usually marked or contain metadata keywords)
        if line.startswith("#") or "ITRDB" in line.upper():
            continue

        # Parse the line
        try:
            parsed = _parse_rwl_line(line)
            if parsed is None:
                continue

            series_id, year, values, is_end = parsed

            if series_id not in series_data:
                series_data[series_id] = {
                    "start_year": year,
                    "values": [],
                }

            # Filter out stop marker (999) and append valid values
            for v in values:
                if v == -9999 or v == 999:  # Common stop markers
                    break
                series_data[series_id]["values"].append(v)

        except Exception:
            # Skip unparseable lines
            continue

    # Convert accumulated data to RingWidthSeries objects
    for series_id, data in series_data.items():
        if data["values"]:
            values = np.array(data["values"], dtype=np.float64)
            start_year = data["start_year"]
            end_year = start_year + len(values) - 1

            rwl.series[series_id] = RingWidthSeries(
                series_id=series_id,
                start_year=start_year,
                end_year=end_year,
                values=values,
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

    # Extract series ID (first 8 chars, stripped)
    series_id = line[:8].strip()
    if not series_id:
        return None

    # Extract year (chars 8-12)
    try:
        year_str = line[8:12].strip()
        year = int(year_str)
    except (ValueError, IndexError):
        return None

    # Parse values (remaining characters in 6-char blocks)
    values = []
    pos = 12
    is_end = False

    while pos + 6 <= len(line):
        val_str = line[pos:pos+6].strip()
        if val_str:
            try:
                val = int(val_str)
                values.append(val)
                if val == 999 or val == -9999:
                    is_end = True
            except ValueError:
                pass
        pos += 6

    # Handle partial final value
    if pos < len(line):
        val_str = line[pos:].strip()
        if val_str:
            try:
                val = int(val_str)
                values.append(val)
            except ValueError:
                pass

    if not values:
        return None

    return series_id, year, values, is_end


def parse_crn_file(filepath: str | Path) -> Chronology | None:
    """
    Parse a Tucson format .crn chronology file.

    CRN format typically has:
    - Header lines with site metadata
    - Data lines with year, index value, and sample depth

    Args:
        filepath: Path to the .crn file

    Returns:
        Chronology object or None if parsing fails
    """
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    lines = content.split("\n")

    # Parse header information
    metadata = _parse_crn_header(lines)

    # Find where data starts (after header lines)
    data_start = 0
    for i, line in enumerate(lines):
        # Data lines typically start with the site code followed by year
        if len(line) >= 12:
            # Check if this looks like a data line
            try:
                year_part = line[8:12].strip()
                if year_part and year_part.isdigit():
                    data_start = i
                    break
            except (IndexError, ValueError):
                continue

    # Parse data lines
    years = []
    values = []
    sample_depths = []
    site_id = None

    for line in lines[data_start:]:
        if len(line) < 18:
            continue

        try:
            if site_id is None:
                site_id = line[:8].strip()

            year = int(line[8:12].strip())

            # Parse value-depth pairs (each pair is typically 7 chars: 4 value + 3 depth)
            pos = 12
            while pos + 7 <= len(line):
                val_str = line[pos:pos+4].strip()
                depth_str = line[pos+4:pos+7].strip()

                if val_str and val_str != "9990" and val_str != "-999":
                    try:
                        val = int(val_str)
                        depth = int(depth_str) if depth_str else 1

                        years.append(year)
                        values.append(val)
                        sample_depths.append(depth)
                        year += 1
                    except ValueError:
                        pass
                else:
                    break
                pos += 7

        except (ValueError, IndexError):
            continue

    if not years:
        return None

    return Chronology(
        site_id=site_id or filepath.stem,
        site_name=metadata.get("site_name", ""),
        species=metadata.get("species", ""),
        start_year=min(years),
        end_year=max(years),
        values=np.array(values, dtype=np.float64),
        sample_depth=np.array(sample_depths, dtype=np.int32),
        latitude=metadata.get("latitude"),
        longitude=metadata.get("longitude"),
        elevation=metadata.get("elevation"),
        state=metadata.get("state"),
    )


def _parse_crn_header(lines: list[str]) -> dict:
    """Extract metadata from CRN file header lines."""
    metadata = {}

    for line in lines[:20]:  # Check first 20 lines for header info
        line_upper = line.upper()

        # Look for species code
        species_match = re.search(r'\b(PIST|TSCA|QUAL|QURU|PCRU|PIRE|PIRI|THOC|ACSA|FRAX)\b', line_upper)
        if species_match:
            metadata["species"] = species_match.group(1)

        # Look for coordinates
        lat_match = re.search(r'(\d{2})[°\s](\d{2})[\'′\s]?\s*N', line, re.IGNORECASE)
        if lat_match:
            metadata["latitude"] = float(lat_match.group(1)) + float(lat_match.group(2)) / 60

        lon_match = re.search(r'(\d{2,3})[°\s](\d{2})[\'′\s]?\s*W', line, re.IGNORECASE)
        if lon_match:
            metadata["longitude"] = -(float(lon_match.group(1)) + float(lon_match.group(2)) / 60)

        # Look for elevation
        elev_match = re.search(r'(\d{2,4})\s*M\b', line_upper)
        if elev_match:
            metadata["elevation"] = float(elev_match.group(1))

        # State codes
        for state in ["NH", "VT", "ME", "MA", "CT", "RI", "NY", "PA"]:
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

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # Handle year column
    if "year" not in df.columns:
        if len(df.columns) >= 2:
            # Assume first column is year if it looks like years
            first_col = df.iloc[:, 0]
            if first_col.dtype in [np.int64, np.float64] and first_col.min() > 1000:
                df = df.rename(columns={df.columns[0]: "year"})
            else:
                df["year"] = range(1, len(df) + 1)
        else:
            df["year"] = range(1, len(df) + 1)

    # Handle width column
    if "width" not in df.columns:
        # Find the first non-year numeric column
        for col in df.columns:
            if col != "year" and df[col].dtype in [np.int64, np.float64]:
                df = df.rename(columns={col: "width"})
                break

    # Ensure we have the required columns
    if "width" not in df.columns:
        raise ValueError("Could not identify ring width column in CSV")

    return df[["year", "width"]].dropna()
