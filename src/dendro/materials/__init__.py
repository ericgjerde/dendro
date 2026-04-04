"""Material taxonomy and context profiles for assisted Walpole inference."""

from .walpole import (
    CURRENT_WALPOLE_PROFILE_VERSION,
    WALPOLE_PROFILE_ID,
    WALPOLE_PROFILE_V1,
    Citation,
    MaterialGroup,
    SpeciesMapping,
    WalpoleMaterialProfile,
    load_walpole_profile,
)

__all__ = [
    "CURRENT_WALPOLE_PROFILE_VERSION",
    "WALPOLE_PROFILE_ID",
    "WALPOLE_PROFILE_V1",
    "Citation",
    "MaterialGroup",
    "SpeciesMapping",
    "WalpoleMaterialProfile",
    "load_walpole_profile",
]
