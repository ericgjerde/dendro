"""Material-group taxonomy and Walpole inference helpers."""

from .catalog import (
    MATERIAL_GROUP_DISPLAY_NAMES,
    MATERIAL_GROUP_SPECIES,
    SUPPORTED_MATERIAL_GROUPS,
    infer_material_group_from_species,
    material_group_display_name,
    material_group_species,
    normalize_material_group,
)
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
    "MATERIAL_GROUP_DISPLAY_NAMES",
    "MATERIAL_GROUP_SPECIES",
    "SUPPORTED_MATERIAL_GROUPS",
    "infer_material_group_from_species",
    "material_group_display_name",
    "material_group_species",
    "normalize_material_group",
    "CURRENT_WALPOLE_PROFILE_VERSION",
    "WALPOLE_PROFILE_ID",
    "WALPOLE_PROFILE_V1",
    "Citation",
    "MaterialGroup",
    "SpeciesMapping",
    "WalpoleMaterialProfile",
    "load_walpole_profile",
]
