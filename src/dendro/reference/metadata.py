"""
Metadata helpers for local and NOAA-hosted reference files.

This module resolves reference metadata from three sources, in order:
1. Embedded NOAA template headers inside local files
2. Local NOAA sidecar text files downloaded alongside the main data file
3. Optional remote NOAA sidecar text files for bundled datasets that predate
   the new downloader behavior
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Optional

import requests


ITRDB_MEASUREMENTS_BASE = (
    "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/northamerica/usa/"
)
ITRDB_CHRONOLOGIES_BASE = (
    "https://www.ncei.noaa.gov/pub/data/paleo/treering/chronologies/northamerica/usa/"
)

SPECIES_CODES = [
    "PIST",
    "TSCA",
    "QUAL",
    "QURU",
    "PCRU",
    "PIRE",
    "PIRI",
    "THOC",
    "ACSA",
    "FRAX",
]

STATE_CODES = ["CT", "MA", "ME", "NH", "NY", "RI", "VT", "PA"]


@dataclass
class ReferenceMetadata:
    site_id: str
    site_name: str = ""
    species: str = ""
    state: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None
    provenance_url: str = ""
    study_metadata_url: str = ""
    source: str = ""
    warnings: list[str] = field(default_factory=list)

    def merge(self, other: "ReferenceMetadata") -> "ReferenceMetadata":
        """Merge non-empty values from another metadata object."""
        if other.site_name and not self.site_name:
            self.site_name = other.site_name
        if other.species and not self.species:
            self.species = other.species
        if other.state and not self.state:
            self.state = other.state
        if other.latitude is not None and self.latitude is None:
            self.latitude = other.latitude
        if other.longitude is not None and self.longitude is None:
            self.longitude = other.longitude
        if other.elevation is not None and self.elevation is None:
            self.elevation = other.elevation
        if other.provenance_url and not self.provenance_url:
            self.provenance_url = other.provenance_url
        if other.study_metadata_url and not self.study_metadata_url:
            self.study_metadata_url = other.study_metadata_url
        if other.source:
            if self.source:
                self.source = f"{self.source},{other.source}"
            else:
                self.source = other.source
        self.warnings.extend(other.warnings)
        return self


def _data_line_pattern() -> re.Pattern[str]:
    return re.compile(r"^[A-Za-z0-9]{1,8}\s+\d{3,4}")


def parse_embedded_metadata(filepath: str | Path) -> ReferenceMetadata:
    """Parse embedded NOAA header metadata from a local file, if present."""
    filepath = Path(filepath)
    metadata = ReferenceMetadata(site_id=filepath.stem[:8].upper())

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
            lines = []
            for _ in range(80):
                line = handle.readline()
                if not line:
                    break
                stripped = line.rstrip("\n\r")
                if _data_line_pattern().match(stripped):
                    break
                lines.append(stripped)
    except OSError as exc:
        metadata.warnings.append(f"Could not read embedded metadata: {exc}")
        return metadata

    if not lines:
        return metadata

    return metadata.merge(parse_noaa_template("\n".join(lines), site_id=metadata.site_id, source="embedded"))


def parse_noaa_template(text: str, site_id: str = "", source: str = "") -> ReferenceMetadata:
    """Parse a NOAA sidecar/template header into structured metadata."""
    metadata = ReferenceMetadata(site_id=site_id.upper(), source=source)
    lines = text.splitlines()

    for raw_line in lines[:120]:
        line = raw_line.strip()
        if not line:
            continue

        # NOAA template title line:
        # "# Fritts - Nancy Brook - PCRU - ITRDB NH001"
        title = re.match(
            r"^#\s+.+?\s+-\s+(?P<site>.+?)\s+-\s+(?P<species>[A-Z]{4})\s+-\s+ITRDB\s+(?P<site_id>[A-Z0-9]+)",
            line,
        )
        if title:
            metadata.site_name = title.group("site").strip()
            metadata.species = title.group("species").strip()
            metadata.site_id = title.group("site_id").strip()
            continue

        if line.startswith("# NOAA_Landing_Page:"):
            metadata.provenance_url = line.split(":", 1)[1].strip()
            continue

        if line.startswith("# Study_Level_JSON_Metadata:"):
            metadata.study_metadata_url = line.split(":", 1)[1].strip()
            continue

        species_match = re.search(rf"\b({'|'.join(SPECIES_CODES)})\b", line.upper())
        if species_match and not metadata.species:
            metadata.species = species_match.group(1)

        state_match = re.search(rf"\b({'|'.join(STATE_CODES)})\b", line.upper())
        if state_match and not metadata.state:
            metadata.state = state_match.group(1)

        lat_match = re.search(r"(\d{2})[°\s](\d{2})['′\s]?\s*N", line, re.IGNORECASE)
        if lat_match and metadata.latitude is None:
            metadata.latitude = float(lat_match.group(1)) + float(lat_match.group(2)) / 60

        lon_match = re.search(r"(\d{2,3})[°\s](\d{2})['′\s]?\s*W", line, re.IGNORECASE)
        if lon_match and metadata.longitude is None:
            metadata.longitude = -(
                float(lon_match.group(1)) + float(lon_match.group(2)) / 60
            )

        elev_match = re.search(r"(\d{2,4})\s*M\b", line.upper())
        if elev_match and metadata.elevation is None:
            metadata.elevation = float(elev_match.group(1))

    return metadata


def local_sidecar_path(filepath: str | Path, file_type: str) -> Path:
    """Return the expected local NOAA sidecar path for a reference file."""
    filepath = Path(filepath)
    return filepath.with_name(f"{filepath.stem}-{file_type}-noaa.txt")


def load_local_sidecar(filepath: str | Path, file_type: str) -> ReferenceMetadata:
    """Load metadata from a local NOAA sidecar, if present."""
    filepath = Path(filepath)
    sidecar = local_sidecar_path(filepath, file_type)
    if not sidecar.exists():
        return ReferenceMetadata(site_id=filepath.stem[:8].upper())

    text = sidecar.read_text(encoding="utf-8", errors="replace")
    return parse_noaa_template(text, site_id=filepath.stem[:8].upper(), source="sidecar")


def remote_sidecar_url(site_id: str, file_type: str) -> str:
    """Return the NOAA sidecar URL for a site/file type pair."""
    site_id = site_id.lower()
    if file_type == "crn":
        return f"{ITRDB_CHRONOLOGIES_BASE}{site_id}-crn-noaa.txt"
    return f"{ITRDB_MEASUREMENTS_BASE}{site_id}-rwl-noaa.txt"


def fetch_remote_sidecar(site_id: str, file_type: str, timeout: int = 20) -> ReferenceMetadata:
    """Fetch a NOAA sidecar text file and parse metadata from it."""
    url = remote_sidecar_url(site_id, file_type)
    metadata = ReferenceMetadata(site_id=site_id.upper())

    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            metadata.warnings.append(f"Remote metadata unavailable at {url}")
            return metadata
        return parse_noaa_template(response.text, site_id=site_id.upper(), source="remote").merge(
            ReferenceMetadata(site_id=site_id.upper(), provenance_url=url)
        )
    except requests.RequestException as exc:
        metadata.warnings.append(f"Could not fetch remote metadata: {exc}")
        return metadata


def resolve_reference_metadata(
    filepath: str | Path,
    file_type: str,
    *,
    allow_remote: bool = False,
) -> ReferenceMetadata:
    """Resolve best-effort metadata for a local reference file."""
    filepath = Path(filepath)
    site_id = filepath.stem[:8].upper()
    resolved = ReferenceMetadata(site_id=site_id)

    resolved.merge(parse_embedded_metadata(filepath))
    resolved.merge(load_local_sidecar(filepath, file_type))

    if allow_remote and (not resolved.species or not resolved.site_name):
        resolved.merge(fetch_remote_sidecar(site_id, file_type))

    if not resolved.state:
        parent = filepath.parent.name.upper()
        if len(parent) == 2 and parent.isalpha():
            resolved.state = parent

    return resolved
