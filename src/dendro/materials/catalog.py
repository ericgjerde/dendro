"""Shared material-group taxonomy for Walpole timber workflows."""

from __future__ import annotations

from typing import Optional

from .walpole import load_walpole_profile


def _profile():
    return load_walpole_profile()


_PROFILE = _profile()

MATERIAL_GROUP_ALIASES: dict[str, str] = {}
for _group in _PROFILE.material_groups:
    MATERIAL_GROUP_ALIASES[_group.group_id] = _group.group_id
    MATERIAL_GROUP_ALIASES[_group.display_name.strip().lower().replace(" ", "_")] = _group.group_id
    for _alias in _group.aliases:
        MATERIAL_GROUP_ALIASES[str(_alias).strip().lower().replace(" ", "_")] = _group.group_id

MATERIAL_GROUP_SPECIES: dict[str, tuple[str, ...]] = {
    group.group_id: tuple(group.species_codes)
    for group in _PROFILE.material_groups
}

MATERIAL_GROUP_DISPLAY_NAMES: dict[str, str] = {
    group.group_id: group.display_name
    for group in _PROFILE.material_groups
}

SUPPORTED_MATERIAL_GROUPS: tuple[str, ...] = _PROFILE.material_group_ids()

_SPECIES_TO_MATERIAL_GROUP: dict[str, str] = {
    mapping.species_code.upper(): mapping.material_group
    for mapping in _PROFILE.species_mappings
}


def normalize_material_group(material_group: str | None) -> str:
    if material_group is None:
        return ""
    key = str(material_group).strip().lower().replace("-", "_").replace(" ", "_")
    return MATERIAL_GROUP_ALIASES.get(key, key)


def material_group_species(material_group: str | None) -> tuple[str, ...]:
    normalized = normalize_material_group(material_group)
    return MATERIAL_GROUP_SPECIES.get(normalized, ())


def infer_material_group_from_species(species: str | None) -> str:
    if species is None:
        return ""
    return _SPECIES_TO_MATERIAL_GROUP.get(str(species).strip().upper(), "")


def material_group_display_name(material_group: str | None) -> str:
    normalized = normalize_material_group(material_group)
    if not normalized:
        return "Unknown"
    return MATERIAL_GROUP_DISPLAY_NAMES.get(normalized, normalized.replace("_", " ").title())


def material_group_supported(material_group: str | None) -> bool:
    normalized = normalize_material_group(material_group)
    return normalized in SUPPORTED_MATERIAL_GROUPS


def species_codes_for_material_groups(material_groups: list[str] | tuple[str, ...]) -> list[str]:
    codes: list[str] = []
    for material_group in material_groups:
        for code in material_group_species(material_group):
            if code not in codes:
                codes.append(code)
    return codes


def best_material_group_for_species_list(species_codes: list[str] | tuple[str, ...]) -> Optional[str]:
    counts: dict[str, int] = {}
    for code in species_codes:
        material_group = infer_material_group_from_species(code)
        if not material_group:
            continue
        counts[material_group] = counts.get(material_group, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
