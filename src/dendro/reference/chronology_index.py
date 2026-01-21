"""
Index and search downloaded chronologies by species, location, and time span.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .tucson_parser import parse_rwl_file, parse_crn_file, Chronology, RWLFile


@dataclass
class ChronologyMetadata:
    """Metadata for an indexed chronology."""

    filepath: str
    site_id: str
    site_name: str
    species: str
    state: str
    start_year: int
    end_year: int
    num_years: int
    num_series: int  # For RWL files, number of individual series
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    file_type: str = "rwl"  # 'rwl' or 'crn'

    def covers_period(self, start: int, end: int) -> bool:
        """Check if this chronology covers the given time period."""
        return self.start_year <= start and self.end_year >= end

    def overlap_years(self, start: int, end: int) -> int:
        """Calculate years of overlap with a given period."""
        overlap_start = max(self.start_year, start)
        overlap_end = min(self.end_year, end)
        return max(0, overlap_end - overlap_start + 1)


class ChronologyIndex:
    """
    Index of available reference chronologies.

    Provides efficient lookup by species, location, and time period.
    """

    def __init__(self, data_dir: Optional[str | Path] = None):
        """
        Initialize the chronology index.

        Args:
            data_dir: Directory containing downloaded chronologies.
                     If None, must call scan_directory() later.
        """
        self.entries: list[ChronologyMetadata] = []
        self._by_species: dict[str, list[ChronologyMetadata]] = {}
        self._by_state: dict[str, list[ChronologyMetadata]] = {}

        if data_dir is not None:
            self.scan_directory(data_dir)

    def scan_directory(self, data_dir: str | Path) -> int:
        """
        Scan a directory for chronology files and index them.

        Args:
            data_dir: Directory to scan (recursively).

        Returns:
            Number of files indexed.
        """
        data_dir = Path(data_dir)

        if not data_dir.exists():
            return 0

        count = 0

        # Find all .rwl and .crn files
        for pattern in ["**/*.rwl", "**/*.crn", "**/*-rwl-*.txt", "**/*-crn-*.txt"]:
            for filepath in data_dir.glob(pattern):
                try:
                    metadata = self._index_file(filepath)
                    if metadata is not None:
                        self._add_entry(metadata)
                        count += 1
                except Exception as e:
                    print(f"Warning: Could not index {filepath}: {e}")
                    continue

        print(f"Indexed {count} chronology files")
        return count

    def _index_file(self, filepath: Path) -> Optional[ChronologyMetadata]:
        """Parse a file and extract its metadata."""
        filename = filepath.name.lower()

        # Determine file type
        if ".crn" in filename or "-crn-" in filename:
            file_type = "crn"
        else:
            file_type = "rwl"

        # Extract state from path or filename
        state = self._extract_state(filepath)

        # Parse the file
        if file_type == "crn":
            chronology = parse_crn_file(filepath)
            if chronology is None:
                return None

            return ChronologyMetadata(
                filepath=str(filepath),
                site_id=chronology.site_id,
                site_name=chronology.site_name,
                species=chronology.species or self._extract_species(filepath),
                state=state or chronology.state or "",
                start_year=chronology.start_year,
                end_year=chronology.end_year,
                num_years=chronology.length,
                num_series=1,
                latitude=chronology.latitude,
                longitude=chronology.longitude,
                file_type="crn",
            )
        else:
            rwl = parse_rwl_file(filepath)
            if not rwl.series:
                return None

            # Get overall time span
            all_years = []
            for series in rwl.series.values():
                all_years.extend(range(series.start_year, series.end_year + 1))

            if not all_years:
                return None

            return ChronologyMetadata(
                filepath=str(filepath),
                site_id=filepath.stem[:8].upper(),
                site_name=filepath.stem,
                species=self._extract_species(filepath),
                state=state or "",
                start_year=min(all_years),
                end_year=max(all_years),
                num_years=max(all_years) - min(all_years) + 1,
                num_series=len(rwl.series),
                file_type="rwl",
            )

    def _extract_state(self, filepath: Path) -> Optional[str]:
        """Try to extract state code from filepath."""
        # Check parent directory name
        parent = filepath.parent.name.upper()
        if len(parent) == 2 and parent.isalpha():
            return parent

        # Check filename prefix (e.g., nh001.rwl)
        match = re.match(r'([a-zA-Z]{2})\d{3}', filepath.stem)
        if match:
            return match.group(1).upper()

        return None

    def _extract_species(self, filepath: Path) -> str:
        """Try to extract species code from filepath."""
        filename = filepath.name.upper()

        species_codes = ["PIST", "TSCA", "QUAL", "QURU", "PCRU", "PIRE", "THOC", "ACSA"]
        for code in species_codes:
            if code in filename:
                return code

        return ""

    def _add_entry(self, metadata: ChronologyMetadata):
        """Add an entry to the index."""
        self.entries.append(metadata)

        # Index by species
        if metadata.species:
            if metadata.species not in self._by_species:
                self._by_species[metadata.species] = []
            self._by_species[metadata.species].append(metadata)

        # Index by state
        if metadata.state:
            if metadata.state not in self._by_state:
                self._by_state[metadata.state] = []
            self._by_state[metadata.state].append(metadata)

    def search(
        self,
        species: Optional[list[str]] = None,
        states: Optional[list[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        min_overlap: int = 30,
        file_type: Optional[str] = None,
    ) -> list[ChronologyMetadata]:
        """
        Search for chronologies matching the given criteria.

        Args:
            species: List of species codes to match.
            states: List of state codes to match.
            min_year: Minimum year that must be covered.
            max_year: Maximum year that must be covered.
            min_overlap: Minimum years of overlap required with time range.
            file_type: Filter by 'rwl' or 'crn'.

        Returns:
            List of matching ChronologyMetadata entries.
        """
        results = []

        # Start with all entries or filter by species/state
        candidates = self.entries

        if species:
            species = [s.upper() for s in species]
            candidates = [e for e in candidates if e.species in species]

        if states:
            states = [s.upper() for s in states]
            candidates = [e for e in candidates if e.state in states]

        if file_type:
            candidates = [e for e in candidates if e.file_type == file_type]

        # Filter by time range
        for entry in candidates:
            if min_year is not None and max_year is not None:
                overlap = entry.overlap_years(min_year, max_year)
                if overlap < min_overlap:
                    continue
            elif min_year is not None:
                if entry.end_year < min_year:
                    continue
            elif max_year is not None:
                if entry.start_year > max_year:
                    continue

            results.append(entry)

        # Sort by overlap (most overlap first)
        if min_year is not None and max_year is not None:
            results.sort(key=lambda e: e.overlap_years(min_year, max_year), reverse=True)
        else:
            # Sort by number of years
            results.sort(key=lambda e: e.num_years, reverse=True)

        return results

    def get_species(self) -> list[str]:
        """Get list of all indexed species codes."""
        return sorted(self._by_species.keys())

    def get_states(self) -> list[str]:
        """Get list of all indexed state codes."""
        return sorted(self._by_state.keys())

    def load_chronology(self, metadata: ChronologyMetadata) -> Chronology | RWLFile:
        """Load the actual chronology data for an index entry."""
        filepath = Path(metadata.filepath)

        if metadata.file_type == "crn":
            return parse_crn_file(filepath)
        else:
            return parse_rwl_file(filepath)

    def save_index(self, filepath: str | Path):
        """Save the index to a JSON file."""
        filepath = Path(filepath)

        data = {
            "entries": [asdict(e) for e in self.entries],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_index(self, filepath: str | Path):
        """Load an index from a JSON file."""
        filepath = Path(filepath)

        with open(filepath, "r") as f:
            data = json.load(f)

        self.entries = []
        self._by_species = {}
        self._by_state = {}

        for entry_data in data.get("entries", []):
            metadata = ChronologyMetadata(**entry_data)
            self._add_entry(metadata)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"ChronologyIndex({len(self.entries)} entries)"
