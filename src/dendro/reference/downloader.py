"""
Download and inventory Northeast ITRDB reference files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional
from urllib.parse import urljoin

import requests
from tqdm import tqdm

from .metadata import (
    ITRDB_CHRONOLOGIES_BASE,
    ITRDB_MEASUREMENTS_BASE,
    SPECIES_CODES,
    parse_noaa_template,
)


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

NORTHEAST_STATES = ["ct", "ma", "me", "nh", "ny", "ri", "vt"]


@dataclass
class ChronologyFile:
    filename: str
    url: str
    state: str
    site_code: str
    file_type: str
    sidecar_filename: Optional[str] = None
    sidecar_url: str = ""
    species: Optional[str] = None
    site_name: str = ""

    @property
    def local_path(self) -> str:
        return f"{self.state.lower()}/{self.filename}"


def list_available_files(
    states: Optional[list[str]] = None,
    species: Optional[list[str]] = None,
    *,
    file_type: str = "rwl",
) -> list[ChronologyFile]:
    """
    List downloadable RWL or CRN files with NOAA sidecar metadata when available.
    """
    if states is None:
        states = NORTHEAST_STATES
    states = [state.lower() for state in states]
    species = [code.upper() for code in species] if species else None

    base_url = ITRDB_MEASUREMENTS_BASE if file_type == "rwl" else ITRDB_CHRONOLOGIES_BASE
    response = requests.get(base_url, timeout=60)
    response.raise_for_status()

    hrefs = set(re.findall(r'href="([^"]+)"', response.text, re.IGNORECASE))
    grouped: dict[str, dict[str, str]] = {}

    for href in hrefs:
        name = href.lower()
        if href.startswith("..") or href.startswith("/"):
            continue

        if file_type == "rwl":
            main_match = re.match(r"^([a-z]{2}\d+[a-z]?)\.rwl$", name)
            sidecar_match = re.match(r"^([a-z]{2}\d+[a-z]?)-rwl-noaa\.txt$", name)
        else:
            main_match = re.match(r"^([a-z]{2}\d+[a-z]?)\.crn$", name)
            sidecar_match = re.match(r"^([a-z]{2}\d+[a-z]?)-crn-noaa\.txt$", name)

        if main_match:
            site_code = main_match.group(1)
            grouped.setdefault(site_code, {})["main"] = href
            continue

        if sidecar_match:
            site_code = sidecar_match.group(1)
            grouped.setdefault(site_code, {})["sidecar"] = href

    files: list[ChronologyFile] = []
    for site_code, assets in sorted(grouped.items()):
        state = site_code[:2]
        if state not in states or "main" not in assets:
            continue

        item = ChronologyFile(
            filename=assets["main"],
            url=urljoin(base_url, assets["main"]),
            state=state.upper(),
            site_code=site_code.upper(),
            file_type=file_type,
            sidecar_filename=assets.get("sidecar"),
            sidecar_url=urljoin(base_url, assets["sidecar"]) if assets.get("sidecar") else "",
        )

        if item.sidecar_url:
            metadata = _fetch_remote_sidecar_metadata(item.sidecar_url)
            if metadata.species:
                item.species = metadata.species
            if metadata.site_name:
                item.site_name = metadata.site_name

        if species and item.species not in species:
            continue

        files.append(item)

    return files


def _fetch_remote_sidecar_metadata(sidecar_url: str):
    try:
        response = requests.get(sidecar_url, timeout=30)
        response.raise_for_status()
        return parse_noaa_template(response.text, source="remote")
    except requests.RequestException:
        return parse_noaa_template("", source="remote")


def download_file(
    file_info: ChronologyFile,
    output_dir: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    """
    Download a main chronology file and its NOAA sidecar metadata if available.
    """
    output_dir = Path(output_dir)
    state_dir = output_dir / file_info.state.lower()
    state_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []

    main_path = state_dir / file_info.filename
    if overwrite or not main_path.exists():
        response = requests.get(file_info.url, timeout=60)
        response.raise_for_status()
        main_path.write_bytes(response.content)
    downloaded.append(main_path)

    if file_info.sidecar_filename and file_info.sidecar_url:
        sidecar_path = state_dir / file_info.sidecar_filename
        if overwrite or not sidecar_path.exists():
            response = requests.get(file_info.sidecar_url, timeout=60)
            response.raise_for_status()
            sidecar_path.write_bytes(response.content)
        downloaded.append(sidecar_path)

    return downloaded


def download_chronologies(
    output_dir: str | Path,
    states: Optional[list[str]] = None,
    species: Optional[list[str]] = None,
    file_types: Optional[list[str]] = None,
    overwrite: bool = False,
    progress: bool = True,
) -> list[Path]:
    """
    Download Northeast RWL/CRN files and NOAA sidecars truthfully.
    """
    if file_types is None:
        file_types = ["rwl", "crn"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory: list[ChronologyFile] = []
    for file_type in file_types:
        print(f"Fetching {file_type.upper()} listings...")
        inventory.extend(list_available_files(states=states, species=species, file_type=file_type))

    downloaded: list[Path] = []
    iterator = tqdm(inventory, desc="Downloading") if progress else inventory
    for file_info in iterator:
        try:
            downloaded.extend(download_file(file_info, output_dir, overwrite=overwrite))
        except requests.RequestException as exc:
            print(f"Warning: Failed to download {file_info.filename}: {exc}")

    print(f"Downloaded {len(downloaded)} artifacts to {output_dir}")
    return downloaded


def download_northeast_reference_set(
    output_dir: str | Path,
    species: Optional[list[str]] = None,
) -> list[Path]:
    if species is None:
        species = ["PIST", "TSCA", "QUAL", "QURU"]

    print(f"Downloading Northeast references for: {', '.join(species)}")
    print(f"States: {', '.join(state.upper() for state in NORTHEAST_STATES)}")

    return download_chronologies(
        output_dir=output_dir,
        states=NORTHEAST_STATES,
        species=species,
        file_types=["rwl", "crn"],
        progress=True,
    )
