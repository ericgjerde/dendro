"""Versioned Walpole late-1700s house materials profile.

This module owns the Walpole-specific context profile artifact requested for
milestone 1. It intentionally stays independent from the dating, matcher, and
reference ingestion modules so other code can consume the profile without
pulling in unrelated policy logic.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Any, Optional


WALPOLE_PROFILE_ID = "walpole_nh_late_1700s_house"
CURRENT_WALPOLE_PROFILE_VERSION = "1.0.0"


@dataclass(frozen=True)
class Citation:
    """Metadata for a source used to justify the profile."""

    citation_id: str
    title: str
    url: str
    publisher: str
    accessed: str
    notes: str = ""
    supports: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaterialGroup:
    """A Walpole material group and its profile metadata."""

    group_id: str
    display_name: str
    support_status: str
    context_weight: float
    species_codes: tuple[str, ...]
    aliases: tuple[str, ...]
    notes: str = ""
    citations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpeciesMapping:
    """Mapping from a species code or label to a Walpole material group."""

    species_code: str
    material_group: str
    label: str
    mapping_quality: str = "direct"
    notes: str = ""
    citations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WalpoleMaterialProfile:
    """The versioned Walpole late-1700s house context profile."""

    profile_id: str
    version: str
    town: str
    state: str
    built_year_range: tuple[int, int]
    description: str
    material_groups: tuple[MaterialGroup, ...]
    species_mappings: tuple[SpeciesMapping, ...]
    member_type_priors: dict[str, dict[str, float]]
    context_prior_weights: dict[str, float]
    citations: tuple[Citation, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self._species_code_to_group = {
            mapping.species_code.upper(): mapping.material_group for mapping in self.species_mappings
        }
        self._species_label_to_group = {
            _normalize_lookup_key(mapping.label): mapping.material_group for mapping in self.species_mappings
        }
        self._group_lookup = {group.group_id: group for group in self.material_groups}
        self._group_alias_to_id: dict[str, str] = {}
        for group in self.material_groups:
            self._group_alias_to_id[_normalize_lookup_key(group.group_id)] = group.group_id
            self._group_alias_to_id[_normalize_lookup_key(group.display_name)] = group.group_id
            for alias in group.aliases:
                self._group_alias_to_id[_normalize_lookup_key(alias)] = group.group_id
        self._citation_lookup = {citation.citation_id: citation for citation in self.citations}
        self._validate_weights()

    def _validate_weights(self) -> None:
        for name, weights in self.member_type_priors.items():
            _validate_probability_distribution(weights, context=f"member_type_priors[{name}]")
        _validate_probability_distribution(self.context_prior_weights, context="context_prior_weights")

    def material_group_ids(self) -> tuple[str, ...]:
        return tuple(group.group_id for group in self.material_groups)

    def supported_material_groups(self) -> tuple[str, ...]:
        return tuple(
            group.group_id
            for group in self.material_groups
            if group.support_status in {"supported", "supported_with_caution"}
        )

    def required_material_groups(self) -> tuple[str, ...]:
        return tuple(
            group.group_id for group in self.material_groups if group.support_status == "required_coverage"
        )

    def material_group_for_species(self, identifier: str) -> Optional[str]:
        normalized = _normalize_lookup_key(identifier)
        if not normalized:
            return None

        direct_code = self._species_code_to_group.get(normalized.upper())
        if direct_code:
            return direct_code

        direct_group = self._group_alias_to_id.get(normalized)
        if direct_group:
            return direct_group

        return self._species_label_to_group.get(normalized)

    def group(self, group_id: str) -> Optional[MaterialGroup]:
        return self._group_lookup.get(group_id)

    def citation(self, citation_id: str) -> Optional[Citation]:
        return self._citation_lookup.get(citation_id)

    def member_type_prior(self, member_type: str) -> dict[str, float]:
        weights = self.member_type_priors.get(_normalize_lookup_key(member_type))
        return dict(weights) if weights else {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "town": self.town,
            "state": self.state,
            "built_year_range": list(self.built_year_range),
            "description": self.description,
            "material_groups": [group.to_dict() for group in self.material_groups],
            "species_mappings": [mapping.to_dict() for mapping in self.species_mappings],
            "member_type_priors": {
                member_type: dict(weights) for member_type, weights in self.member_type_priors.items()
            },
            "context_prior_weights": dict(self.context_prior_weights),
            "citations": [citation.to_dict() for citation in self.citations],
            "notes": list(self.notes),
            "supported_material_groups": list(self.supported_material_groups()),
            "required_material_groups": list(self.required_material_groups()),
        }


def _normalize_lookup_key(value: object) -> str:
    text = str(value or "").strip().lower()
    return " ".join(text.split())


def _validate_probability_distribution(weights: dict[str, float], *, context: str) -> None:
    if not weights:
        raise ValueError(f"{context} cannot be empty")
    total = 0.0
    for key, value in weights.items():
        if value < 0:
            raise ValueError(f"{context} contains a negative weight for {key!r}")
        total += float(value)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{context} must sum to 1.0, got {total!r}")


WALPOLE_PROFILE_V1_DATA: dict[str, Any] = {
    "profile_id": WALPOLE_PROFILE_ID,
    "version": CURRENT_WALPOLE_PROFILE_VERSION,
    "town": "Walpole",
    "state": "NH",
    "built_year_range": (1760, 1800),
    "description": (
        "Walpole, New Hampshire late-1700s house material prior for assisted "
        "species inference. The profile captures historical building-material "
        "preferences and local forest context for likely house timbers."
    ),
    "material_groups": [
        {
            "group_id": "hemlock",
            "display_name": "Eastern hemlock",
            "support_status": "supported",
            "context_weight": 0.28,
            "species_codes": ("TSCA",),
            "aliases": ("hemlock", "eastern hemlock", "tsuga canadensis"),
            "notes": (
                "Strong Walpole candidate for major timbers and contextual forest "
                "coverage in southern New Hampshire."
            ),
            "citations": (
                "gilman_garrison_house",
                "harvard_forest_southern_nh_forests",
                "unh_grafton_county_forest",
            ),
        },
        {
            "group_id": "white_pine",
            "display_name": "Eastern white pine",
            "support_status": "supported",
            "context_weight": 0.31,
            "species_codes": ("PIST",),
            "aliases": ("white pine", "eastern white pine", "pinus strobus"),
            "notes": (
                "Primary Walpole candidate for framing and boards; common in New "
                "Hampshire forests and historic New England wood use."
            ),
            "citations": (
                "hne_a_to_z_primer",
                "harvard_forest_southern_nh_forests",
                "unh_grafton_county_forest",
            ),
        },
        {
            "group_id": "hard_pine",
            "display_name": "Hard pine",
            "support_status": "supported_with_caution",
            "context_weight": 0.09,
            "species_codes": ("PIRI", "PIPA", "PIRE"),
            "aliases": ("hard pine", "pitch pine", "red pine", "longleaf pine"),
            "notes": (
                "Lower-prior pine bucket for Walpole mode. Includes pine codes that "
                "appear in the broader corpus, with more caution than white pine."
            ),
            "citations": (
                "usfs_eastern_region_tree_species_codes",
                "harvard_forest_data_archive_pipa",
            ),
        },
        {
            "group_id": "oak",
            "display_name": "Oak",
            "support_status": "supported_with_caution",
            "context_weight": 0.22,
            "species_codes": ("QUAL", "QURU", "QUPR", "QUST", "QUVE"),
            "aliases": ("oak", "white oak", "red oak", "chestnut oak", "black oak", "post oak"),
            "notes": (
                "Strong structural candidate in late-1700s New England houses, "
                "especially for heavy framing members."
            ),
            "citations": (
                "gilman_garrison_house",
                "unh_native_trees",
                "hne_a_to_z_primer",
            ),
        },
        {
            "group_id": "chestnut",
            "display_name": "American chestnut",
            "support_status": "required_coverage",
            "context_weight": 0.10,
            "species_codes": ("CHTH",),
            "aliases": ("chestnut", "american chestnut", "castanea dentata"),
            "notes": (
                "Historically likely in the region and required for Walpole support "
                "planning, but not yet backed by a local reference corpus in this repo."
            ),
            "citations": (
                "unh_native_trees",
                "harvard_forest_chestnut_history",
            ),
        },
    ],
    "species_mappings": [
        {
            "species_code": "TSCA",
            "material_group": "hemlock",
            "label": "eastern hemlock",
            "mapping_quality": "direct",
            "notes": "Core Walpole hemlock mapping.",
            "citations": ("usfs_eastern_region_tree_species_codes",),
        },
        {
            "species_code": "PIST",
            "material_group": "white_pine",
            "label": "eastern white pine",
            "mapping_quality": "direct",
            "notes": "Core Walpole white pine mapping.",
            "citations": ("usfs_eastern_region_tree_species_codes",),
        },
        {
            "species_code": "PIRI",
            "material_group": "hard_pine",
            "label": "pitch pine",
            "mapping_quality": "direct",
            "notes": "Included as a lower-prior hard pine candidate.",
            "citations": ("usfs_eastern_region_tree_species_codes",),
        },
        {
            "species_code": "PIPA",
            "material_group": "hard_pine",
            "label": "longleaf pine",
            "mapping_quality": "direct",
            "notes": "Included as a lower-prior hard pine candidate from broader corpora.",
            "citations": ("harvard_forest_data_archive_pipa",),
        },
        {
            "species_code": "PIRE",
            "material_group": "hard_pine",
            "label": "red pine",
            "mapping_quality": "direct",
            "notes": "Regional pine addition retained for broader hard pine coverage.",
            "citations": ("usfs_eastern_region_tree_species_codes",),
        },
        {
            "species_code": "QUAL",
            "material_group": "oak",
            "label": "white oak",
            "mapping_quality": "direct",
            "notes": "Core oak mapping for Walpole framing and structural members.",
            "citations": ("usfs_eastern_region_tree_species_codes",),
        },
        {
            "species_code": "QURU",
            "material_group": "oak",
            "label": "red oak",
            "mapping_quality": "direct",
            "notes": "Core oak mapping for Walpole framing and structural members.",
            "citations": ("usfs_eastern_region_tree_species_codes",),
        },
        {
            "species_code": "QUPR",
            "material_group": "oak",
            "label": "chestnut oak",
            "mapping_quality": "direct",
            "notes": "Grouped with Walpole oak because the profile operates at the material-group level.",
            "citations": ("unh_native_trees",),
        },
        {
            "species_code": "QUST",
            "material_group": "oak",
            "label": "post oak",
            "mapping_quality": "direct",
            "notes": "Grouped with Walpole oak because the profile operates at the material-group level.",
            "citations": ("usfs_eastern_region_tree_species_codes",),
        },
        {
            "species_code": "QUVE",
            "material_group": "oak",
            "label": "black oak",
            "mapping_quality": "direct",
            "notes": "Grouped with Walpole oak because the profile operates at the material-group level.",
            "citations": ("usfs_eastern_region_tree_species_codes",),
        },
        {
            "species_code": "CHTH",
            "material_group": "chestnut",
            "label": "american chestnut",
            "mapping_quality": "direct",
            "notes": "Required Walpole support group with no local reference corpus yet.",
            "citations": ("unh_native_trees", "harvard_forest_chestnut_history"),
        },
    ],
    "member_type_priors": {
        "frame": {
            "white_pine": 0.34,
            "hemlock": 0.30,
            "oak": 0.16,
            "hard_pine": 0.10,
            "chestnut": 0.10,
        },
        "brace": {
            "oak": 0.42,
            "hemlock": 0.18,
            "white_pine": 0.14,
            "hard_pine": 0.12,
            "chestnut": 0.14,
        },
        "sill": {
            "oak": 0.46,
            "hemlock": 0.18,
            "white_pine": 0.12,
            "hard_pine": 0.06,
            "chestnut": 0.18,
        },
        "joist": {
            "white_pine": 0.33,
            "hemlock": 0.28,
            "oak": 0.17,
            "hard_pine": 0.12,
            "chestnut": 0.10,
        },
        "rafter": {
            "white_pine": 0.40,
            "hemlock": 0.27,
            "hard_pine": 0.17,
            "oak": 0.10,
            "chestnut": 0.06,
        },
        "board": {
            "white_pine": 0.48,
            "hemlock": 0.16,
            "oak": 0.10,
            "hard_pine": 0.16,
            "chestnut": 0.10,
        },
        "unknown": {
            "white_pine": 0.30,
            "hemlock": 0.24,
            "oak": 0.20,
            "hard_pine": 0.16,
            "chestnut": 0.10,
        },
    },
    "context_prior_weights": {
        "white_pine": 0.31,
        "hemlock": 0.28,
        "oak": 0.22,
        "chestnut": 0.10,
        "hard_pine": 0.09,
    },
    "citations": [
        {
            "citation_id": "gilman_garrison_house",
            "title": "Gilman Garrison House",
            "url": "https://www.historicnewengland.org/property/gilman-garrison-house/",
            "publisher": "Historic New England",
            "accessed": "2026-04-04",
            "notes": (
                "Describes a New Hampshire timber-framed house built of massive "
                "hemlock planks mortised into oak posts."
            ),
            "supports": ("hemlock", "oak"),
        },
        {
            "citation_id": "hne_a_to_z_primer",
            "title": "A to Z Primer for Homeowners",
            "url": "https://www.historicnewengland.org/preservation/for-homeowners-communities/your-old-or-historic-home/a-z-primer-for-homeowners/",
            "publisher": "Historic New England",
            "accessed": "2026-04-04",
            "notes": (
                "Explains that white or yellow pine boards were used from the "
                "seventeenth through nineteenth centuries and that timber framing "
                "is the oldest domestic structure type in New England."
            ),
            "supports": ("white_pine", "oak"),
        },
        {
            "citation_id": "unh_native_trees",
            "title": "List of New Hampshire Native Trees",
            "url": "https://extension.unh.edu/resource/list-new-hampshire-native-trees-0",
            "publisher": "University of New Hampshire Extension",
            "accessed": "2026-04-04",
            "notes": (
                "Lists American chestnut and chestnut oak among New Hampshire "
                "native trees, supporting chestnut/oak regional plausibility."
            ),
            "supports": ("chestnut", "oak"),
        },
        {
            "citation_id": "unh_grafton_county_forest",
            "title": "Timber Harvest on Grafton County Forest this Fall",
            "url": "https://extension.unh.edu/blog/2024/09/timber-harvest-grafton-county-forest-fall",
            "publisher": "University of New Hampshire Extension",
            "accessed": "2026-04-04",
            "notes": (
                "Describes white pine-hemlock stands with scattered oak and other "
                "hardwoods, and calls out red oak and white pine as valuable species."
            ),
            "supports": ("hemlock", "white_pine", "oak"),
        },
        {
            "citation_id": "harvard_forest_southern_nh_forests",
            "title": "Old-Growth Study Reconstructs Southern NH Forests",
            "url": "https://harvardforest.fas.harvard.edu/notes/old-growth-study-reconstructs-southern-nh-forests/",
            "publisher": "Harvard Forest",
            "accessed": "2026-04-04",
            "notes": (
                "Summarizes southern New Hampshire forest composition ranging from "
                "large white pine and eastern hemlock to northern hardwoods."
            ),
            "supports": ("hemlock", "white_pine", "oak"),
        },
        {
            "citation_id": "harvard_forest_chestnut_history",
            "title": "Autumn Foliage Color: Past, Present, and Future",
            "url": "https://harvardforest.fas.harvard.edu/education-opportunities/classic-outreach-resources/autumn-foliage-color/autumn-foliage-color-future/",
            "publisher": "Harvard Forest",
            "accessed": "2026-04-04",
            "notes": (
                "Notes that American chestnut was a very common broadleaf tree in "
                "southern New England forests a century ago before chestnut blight."
            ),
            "supports": ("chestnut", "hemlock"),
        },
        {
            "citation_id": "usfs_eastern_region_tree_species_codes",
            "title": "U.S. Forest Service -- Eastern Region Tree Species",
            "url": "https://www.fs.usda.gov/sites/nfs/files/r09/allegheny/publication/USDA%20Forest%20Service%20Region%209%20Tree%20Species%20Codes.pdf",
            "publisher": "U.S. Forest Service",
            "accessed": "2026-04-04",
            "notes": (
                "Provides standard species code mappings used in the eastern "
                "region, including TSCA, PIST, PIRE, PIRI, and related species."
            ),
            "supports": ("hemlock", "white_pine", "hard_pine", "oak"),
        },
        {
            "citation_id": "harvard_forest_data_archive_pipa",
            "title": "Harvard Forest Data Archive species code references",
            "url": "https://harvardforest1.fas.harvard.edu/exist/apps/datasets/showData.html?id=hf199",
            "publisher": "Harvard Forest",
            "accessed": "2026-04-04",
            "notes": (
                "Data archive reference showing PIPA as Pinus palustris / longleaf "
                "pine in Harvard Forest metadata."
            ),
            "supports": ("hard_pine",),
        },
    ],
    "notes": (
        "This profile is intentionally Walpole-specific and late-1700s-specific.",
        "It is a context artifact for species/material inference, not a dating engine.",
        "Chestnut is included as a required coverage group even though the current repo "
        "does not yet ship a local chestnut reference corpus.",
    ),
}


def _build_profile(data: dict[str, Any]) -> WalpoleMaterialProfile:
    material_groups = tuple(
        MaterialGroup(
            group_id=group["group_id"],
            display_name=group["display_name"],
            support_status=group["support_status"],
            context_weight=float(group["context_weight"]),
            species_codes=tuple(group["species_codes"]),
            aliases=tuple(group["aliases"]),
            notes=group.get("notes", ""),
            citations=tuple(group.get("citations", ())),
        )
        for group in data["material_groups"]
    )
    species_mappings = tuple(
        SpeciesMapping(
            species_code=mapping["species_code"],
            material_group=mapping["material_group"],
            label=mapping["label"],
            mapping_quality=mapping.get("mapping_quality", "direct"),
            notes=mapping.get("notes", ""),
            citations=tuple(mapping.get("citations", ())),
        )
        for mapping in data["species_mappings"]
    )
    citations = tuple(
        Citation(
            citation_id=citation["citation_id"],
            title=citation["title"],
            url=citation["url"],
            publisher=citation["publisher"],
            accessed=citation["accessed"],
            notes=citation.get("notes", ""),
            supports=tuple(citation.get("supports", ())),
        )
        for citation in data["citations"]
    )
    return WalpoleMaterialProfile(
        profile_id=data["profile_id"],
        version=data["version"],
        town=data["town"],
        state=data["state"],
        built_year_range=tuple(data["built_year_range"]),
        description=data["description"],
        material_groups=material_groups,
        species_mappings=species_mappings,
        member_type_priors={name: dict(weights) for name, weights in data["member_type_priors"].items()},
        context_prior_weights=dict(data["context_prior_weights"]),
        citations=citations,
        notes=tuple(data.get("notes", ())),
    )


@lru_cache(maxsize=None)
def load_walpole_profile(version: str = CURRENT_WALPOLE_PROFILE_VERSION) -> WalpoleMaterialProfile:
    """Load the requested Walpole profile version."""

    if version != CURRENT_WALPOLE_PROFILE_VERSION:
        raise KeyError(f"Unknown Walpole profile version: {version!r}")
    return _build_profile(WALPOLE_PROFILE_V1_DATA)


WALPOLE_PROFILE_V1 = load_walpole_profile()

