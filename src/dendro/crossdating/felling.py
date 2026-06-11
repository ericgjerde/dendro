"""
Felling-year interpretation from a dated last ring.

Dating a sample establishes the calendar year of its *last measured ring*. What
that implies about the felling (cutting) date depends on what survives on the
sample:

- **Bark edge / waney edge present** -> the last ring is the final year of
  growth, so the felling year is known exactly (to the season).
- **Sapwood present but no bark edge** -> some outer sapwood rings are missing.
  The felling year is *after* the last ring, by an amount estimated from
  species-specific sapwood statistics. The result is a *range*.
- **Heartwood/sapwood boundary only** -> we know the felling year is no earlier
  than the last heartwood ring plus the minimum sapwood count: a *terminus post
  quem* (felled after).

This module turns the dated last-ring year plus those observations into an
explicit ``FellingEstimate`` rather than overloading a single "felling year"
number (the previous behaviour, which silently reported the last ring as the
felling year even with no bark edge).

Sapwood estimates are intentionally coarse and species-grouped; precise sapwood
models (e.g. Hollstein, Baillie & Pilcher, Sohar et al.) should be selected per
region and species for publication-grade work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FellingType(str, Enum):
    """How precisely the felling date is known."""

    EXACT = "exact"  # bark edge present
    RANGE = "range"  # sapwood present, estimated range
    AFTER = "after"  # terminus post quem only


# Typical (min, median, max) number of sapwood rings by broad species group.
# Conifers do not have a reliably countable sapwood band for this purpose, so
# only a nominal minimum is offered. Oak figures are commonly used European/NA
# ranges and should be refined per regional model before publication.
SAPWOOD_ESTIMATES: dict[str, tuple[int, int, int]] = {
    "QUERCUS": (9, 15, 50),  # white/red oak (QUAL, QURU, ...)
    "DEFAULT": (10, 20, 40),
}

# Species codes that belong to the oak group.
_OAK_CODES = {"QUAL", "QURU", "QUVE", "QUPR", "QUST", "QUSP", "QUMO"}


def _sapwood_model(species: Optional[str]) -> tuple[int, int, int]:
    if species and species.upper() in _OAK_CODES:
        return SAPWOOD_ESTIMATES["QUERCUS"]
    return SAPWOOD_ESTIMATES["DEFAULT"]


@dataclass
class FellingEstimate:
    """An interpretation of the felling date from a dated last ring."""

    last_ring_year: int
    felling_type: FellingType
    earliest_felling: int
    latest_felling: Optional[int]  # None for open-ended "felled after"
    note: str

    def summary(self) -> str:
        if self.felling_type is FellingType.EXACT:
            return f"Felling year: {self.earliest_felling} (exact, bark edge present)"
        if self.felling_type is FellingType.RANGE:
            return (
                f"Felling year range: {self.earliest_felling}-{self.latest_felling} "
                "(estimated from sapwood)"
            )
        return f"Felled after {self.earliest_felling} (terminus post quem)"

    def to_dict(self) -> dict:
        return {
            "last_ring_year": self.last_ring_year,
            "felling_type": self.felling_type.value,
            "earliest_felling": self.earliest_felling,
            "latest_felling": self.latest_felling,
            "note": self.note,
        }


def estimate_felling(
    last_ring_year: int,
    has_bark_edge: bool,
    has_sapwood: bool = False,
    sapwood_count: Optional[int] = None,
    species: Optional[str] = None,
) -> FellingEstimate:
    """
    Interpret a dated last ring as a felling date.

    Args:
        last_ring_year: Calendar year of the outermost measured ring.
        has_bark_edge: True if bark/waney edge is present (exact felling year).
        has_sapwood: True if (incomplete) sapwood is present but no bark edge.
        sapwood_count: Number of sapwood rings already measured/present, if known.
        species: Species code, used to pick a sapwood model.

    Returns:
        A ``FellingEstimate`` describing the felling date and its uncertainty.
    """
    if has_bark_edge:
        return FellingEstimate(
            last_ring_year=last_ring_year,
            felling_type=FellingType.EXACT,
            earliest_felling=last_ring_year,
            latest_felling=last_ring_year,
            note="Bark/waney edge present: felling year is exact.",
        )

    sap_min, sap_med, sap_max = _sapwood_model(species)
    present = sapwood_count or 0

    if has_sapwood:
        # Some sapwood present: estimate the missing remainder to bark.
        missing_min = max(0, sap_min - present)
        missing_max = max(missing_min, sap_max - present)
        earliest = last_ring_year + missing_min
        latest = last_ring_year + missing_max
        return FellingEstimate(
            last_ring_year=last_ring_year,
            felling_type=FellingType.RANGE,
            earliest_felling=earliest,
            latest_felling=latest,
            note=(
                f"Sapwood present ({present} rings); estimated total "
                f"{sap_min}-{sap_max} sapwood rings for this species group."
            ),
        )

    # No bark edge, no sapwood retained: only a lower bound on the felling year.
    return FellingEstimate(
        last_ring_year=last_ring_year,
        felling_type=FellingType.AFTER,
        earliest_felling=last_ring_year + sap_min,
        latest_felling=None,
        note=(
            "No bark edge or sapwood retained: felling year is at least the last "
            f"ring plus the minimum sapwood allowance ({sap_min} rings)."
        ),
    )
