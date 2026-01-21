"""
Download reference chronologies from ITRDB (NOAA NCEI).

The International Tree-Ring Data Bank is hosted by NOAA's National Centers
for Environmental Information. This module downloads chronology and raw
measurement files for specified regions and species.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from tqdm import tqdm


# Base URLs for ITRDB data
ITRDB_MEASUREMENTS_BASE = "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/northamerica/usa/"
ITRDB_CHRONOLOGIES_BASE = "https://www.ncei.noaa.gov/pub/data/paleo/treering/chronologies/northamerica/usa/"

# Species codes relevant for northeastern US dating
NORTHEAST_SPECIES = {
    "PIST": "Pinus strobus (Eastern White Pine)",
    "TSCA": "Tsuga canadensis (Eastern Hemlock)",
    "QUAL": "Quercus alba (White Oak)",
    "QURU": "Quercus rubra (Red Oak)",
    "PCRU": "Picea rubens (Red Spruce)",
    "PIRE": "Pinus resinosa (Red Pine)",
    "THOC": "Thuja occidentalis (Northern White Cedar)",
    "ACSA": "Acer saccharum (Sugar Maple)",
}

# State abbreviations for northeastern US
NORTHEAST_STATES = ["ct", "ma", "me", "nh", "ny", "ri", "vt"]


@dataclass
class ChronologyFile:
    """Metadata about a downloadable chronology file."""

    filename: str
    url: str
    state: str
    site_code: str
    species: Optional[str] = None
    file_type: str = "crn"  # 'crn' for chronology, 'rwl' for raw measurements

    @property
    def local_path(self) -> str:
        """Suggested local filename."""
        return f"{self.state}/{self.filename}"


def list_available_files(
    states: Optional[list[str]] = None,
    species: Optional[list[str]] = None,
    base_url: str = ITRDB_MEASUREMENTS_BASE,
) -> list[ChronologyFile]:
    """
    List available chronology files from ITRDB for given states and species.

    ITRDB files are stored in a flat directory with state codes as filename prefixes
    (e.g., nh001.rwl, ma002.rwl).

    Args:
        states: List of state codes (e.g., ['nh', 'vt', 'ma']). Defaults to all NE states.
        species: List of species codes (e.g., ['PIST', 'TSCA']). None = all species.
        base_url: Base URL for the ITRDB archive.

    Returns:
        List of ChronologyFile objects describing available downloads.
    """
    if states is None:
        states = NORTHEAST_STATES

    states = [s.lower() for s in states]
    files = []

    # Fetch the flat directory listing
    try:
        response = requests.get(base_url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Warning: Could not fetch directory listing: {e}")
        return files

    # Parse directory listing - find all .rwl files
    # Files are named like: nh001.rwl, nh001-noaa.rwl, nh001-rwl-noaa.txt
    file_links = re.findall(r'href="([^"]+\.rwl)"', response.text, re.IGNORECASE)

    for filename in file_links:
        # Skip parent directory links
        if filename.startswith("..") or filename.startswith("/"):
            continue

        # Extract state code from filename (first 2 letters)
        state_match = re.match(r'^([a-z]{2})(\d{3})', filename.lower())
        if not state_match:
            continue

        state = state_match.group(1)
        site_code = state_match.group(1) + state_match.group(2)

        # Filter by state
        if state not in states:
            continue

        # Prefer the simple .rwl files over -noaa.rwl variants for cleaner data
        # Skip -noaa variants to avoid duplicates
        if '-noaa.rwl' in filename.lower():
            continue

        file_info = ChronologyFile(
            filename=filename,
            url=urljoin(base_url, filename),
            state=state.upper(),
            site_code=site_code,
            file_type="rwl",
        )

        # Try to identify species from filename
        for sp_code in NORTHEAST_SPECIES:
            if sp_code.lower() in filename.lower():
                file_info.species = sp_code
                break

        # Filter by species if specified (but include unknown species)
        if species is not None and file_info.species is not None:
            if file_info.species not in species:
                continue

        files.append(file_info)

    return files


def download_file(
    file_info: ChronologyFile,
    output_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """
    Download a single chronology file.

    Args:
        file_info: ChronologyFile describing the file to download.
        output_dir: Directory to save the file.
        overwrite: If True, overwrite existing files.

    Returns:
        Path to the downloaded file.
    """
    output_dir = Path(output_dir)
    state_dir = output_dir / file_info.state.lower()
    state_dir.mkdir(parents=True, exist_ok=True)

    output_path = state_dir / file_info.filename

    if output_path.exists() and not overwrite:
        return output_path

    response = requests.get(file_info.url, timeout=60)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    return output_path


def download_chronologies(
    output_dir: str | Path,
    states: Optional[list[str]] = None,
    species: Optional[list[str]] = None,
    file_types: Optional[list[str]] = None,
    overwrite: bool = False,
    progress: bool = True,
) -> list[Path]:
    """
    Download chronologies from ITRDB for specified states and species.

    Args:
        output_dir: Directory to save downloaded files.
        states: List of state codes. Defaults to northeastern US states.
        species: List of species codes. None = all species.
        file_types: List of file types ('rwl', 'crn'). Defaults to both.
        overwrite: If True, re-download existing files.
        progress: If True, show progress bar.

    Returns:
        List of paths to downloaded files.
    """
    if file_types is None:
        file_types = ["rwl", "crn"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []

    # Get files from both measurements and chronologies directories
    all_files = []

    if "rwl" in file_types:
        print("Fetching measurement file listings...")
        all_files.extend(list_available_files(states, species, ITRDB_MEASUREMENTS_BASE))

    if "crn" in file_types:
        print("Fetching chronology file listings...")
        all_files.extend(list_available_files(states, species, ITRDB_CHRONOLOGIES_BASE))

    # Filter by file type
    all_files = [f for f in all_files if f.file_type in file_types]

    print(f"Found {len(all_files)} files to download")

    iterator = tqdm(all_files, desc="Downloading") if progress else all_files

    for file_info in iterator:
        try:
            path = download_file(file_info, output_dir, overwrite)
            downloaded.append(path)
        except requests.RequestException as e:
            print(f"Warning: Failed to download {file_info.filename}: {e}")
            continue

    print(f"Downloaded {len(downloaded)} files to {output_dir}")
    return downloaded


def download_northeast_reference_set(
    output_dir: str | Path,
    species: Optional[list[str]] = None,
) -> list[Path]:
    """
    Download a curated set of reference chronologies for northeastern US dating.

    This downloads both raw measurements (.rwl) and site chronologies (.crn)
    for the northeastern states, focusing on species commonly used in
    historical timber construction.

    Args:
        output_dir: Directory to save downloaded files.
        species: Species to include. Defaults to PIST and TSCA.

    Returns:
        List of paths to downloaded files.
    """
    if species is None:
        species = ["PIST", "TSCA", "QUAL", "QURU"]

    print(f"Downloading reference chronologies for: {', '.join(species)}")
    print(f"States: {', '.join(s.upper() for s in NORTHEAST_STATES)}")

    return download_chronologies(
        output_dir=output_dir,
        states=NORTHEAST_STATES,
        species=species,
        file_types=["rwl", "crn"],
        progress=True,
    )
