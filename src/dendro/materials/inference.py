"""Context- and dating-aware material inference for Walpole house timbers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..crossdating.matcher import CrossdateMatcher, DatingReport
from ..reference.chronology_index import ChronologyIndex
from .catalog import infer_material_group_from_species, material_group_display_name, material_group_species
from .walpole import WALPOLE_PROFILE_ID, WalpoleMaterialProfile, load_walpole_profile


DEFAULT_SUPPORT_STATES = ("NH", "VT", "MA", "CT", "RI", "ME", "NY")


@dataclass(frozen=True)
class MaterialInferenceContext:
    """Runtime context for Walpole material inference."""

    town: Optional[str] = None
    state: Optional[str] = None
    built_year_range: Optional[tuple[int, int]] = None
    member_type: str = "unknown"
    profile_id: Optional[str] = None

    def normalized_member_type(self) -> str:
        member_type = (self.member_type or "unknown").strip().lower()
        profile = load_walpole_profile()
        if member_type not in profile.member_type_priors:
            return "unknown"
        return member_type

    def to_dict(self) -> dict:
        return {
            "town": self.town,
            "state": self.state,
            "built_year_range": list(self.built_year_range) if self.built_year_range is not None else None,
            "member_type": self.normalized_member_type(),
            "profile_id": self.profile_id or WALPOLE_PROFILE_ID,
        }


@dataclass
class MaterialCandidate:
    """Ranked material-group candidate."""

    material_group: str
    display_name: str
    context_weight: float
    support_reference_count: int
    support_state_count: int
    support_status: str
    support_strength: float
    score: float
    dating_status: str
    best_outer_ring_year: Optional[int]
    best_reference_name: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "material_group": self.material_group,
            "display_name": self.display_name,
            "context_weight": round(float(self.context_weight), 4),
            "support_reference_count": int(self.support_reference_count),
            "support_state_count": int(self.support_state_count),
            "support_status": self.support_status,
            "support_strength": round(float(self.support_strength), 4),
            "score": round(float(self.score), 4),
            "dating_status": self.dating_status,
            "best_outer_ring_year": int(self.best_outer_ring_year) if self.best_outer_ring_year is not None else None,
            "best_reference_name": self.best_reference_name,
            "evidence": list(self.evidence),
        }


@dataclass
class MaterialInferenceReport:
    """Stable machine-readable material inference report."""

    status: str
    context: MaterialInferenceContext
    context_profile: Optional[WalpoleMaterialProfile]
    support_status: str
    recommended_material: Optional[str]
    candidates: list[MaterialCandidate]
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "context": self.context.to_dict(),
            "context_profile": self.context_profile.to_dict() if self.context_profile is not None else None,
            "support_status": self.support_status,
            "recommended_material": self.recommended_material,
            "material_candidates": [candidate.to_dict() for candidate in self.candidates],
            "warnings": list(self.warnings),
            "diagnostics": self._normalize_scalars(self.diagnostics),
        }

    def _normalize_scalars(self, value):
        if isinstance(value, dict):
            return {str(key): self._normalize_scalars(subvalue) for key, subvalue in value.items()}
        if isinstance(value, list):
            return [self._normalize_scalars(item) for item in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value


def parse_built_year_range(raw_value: str | None) -> Optional[tuple[int, int]]:
    if raw_value in {None, ""}:
        return None
    text = str(raw_value).strip()
    delimiter = ":" if ":" in text else "-"
    parts = [part.strip() for part in text.split(delimiter, 1)]
    if len(parts) != 2:
        raise ValueError("Built year range must be START:END")
    start, end = int(parts[0]), int(parts[1])
    if end < start:
        raise ValueError("Built year range end must be greater than or equal to start")
    return (start, end)


def _resolve_walpole_profile(
    *,
    town: Optional[str],
    state: Optional[str],
    built_year_range: Optional[tuple[int, int]],
    explicit_profile_id: Optional[str] = None,
) -> Optional[WalpoleMaterialProfile]:
    profile = load_walpole_profile()
    requested_profile = explicit_profile_id or profile.profile_id
    if requested_profile != profile.profile_id:
        return None
    if town is not None and town.strip().lower() != profile.town.lower():
        return None
    if state is not None and state.strip().upper() != profile.state.upper():
        return None
    if built_year_range is not None:
        start, end = built_year_range
        profile_start, profile_end = profile.built_year_range
        if end < profile_start or start > profile_end:
            return None
    return profile


class MaterialInferenceEngine:
    """Run Walpole material inference using context and dating evidence."""

    def __init__(
        self,
        matcher: Optional[CrossdateMatcher] = None,
        *,
        reference_dir: Optional[str | Path] = None,
        index: Optional[ChronologyIndex] = None,
    ):
        if matcher is not None:
            self.matcher = matcher
        else:
            self.matcher = CrossdateMatcher(index=index, reference_dir=reference_dir)

    @property
    def index(self) -> ChronologyIndex:
        return self.matcher.index

    def infer(
        self,
        values: np.ndarray,
        *,
        sample_name: str,
        context: MaterialInferenceContext,
        has_bark_edge: bool = True,
        orientation: str = "auto",
        era_start: Optional[int] = None,
        era_end: Optional[int] = None,
        min_overlap: int = 30,
        top_n: int = 5,
    ) -> MaterialInferenceReport:
        values = np.asarray(values, dtype=np.float64)
        warnings: list[str] = []
        profile = _resolve_walpole_profile(
            town=context.town,
            state=context.state,
            built_year_range=context.built_year_range,
            explicit_profile_id=context.profile_id,
        )
        if profile is None:
            warnings.append(
                "No context profile matched the supplied town/state/year range. Walpole-only inference is disabled."
            )
            return MaterialInferenceReport(
                status="inconclusive",
                context=context,
                context_profile=None,
                support_status="unsupported_context",
                recommended_material=None,
                candidates=[],
                warnings=warnings,
                diagnostics={"sample_name": sample_name},
            )

        member_type = context.normalized_member_type()
        if member_type != (context.member_type or "unknown").strip().lower():
            warnings.append(
                f"Unsupported member type '{context.member_type}'. Using 'unknown' priors."
            )

        diagnostics = {
            "sample_name": sample_name,
            "profile_id": profile.profile_id,
            "profile_version": profile.version,
            "member_type": member_type,
        }

        candidates: list[MaterialCandidate] = []
        dating_reports: dict[str, DatingReport] = {}
        member_priors = profile.member_type_prior(member_type) or profile.member_type_prior("unknown")
        support_states = [profile.state] + [state for state in DEFAULT_SUPPORT_STATES if state != profile.state]
        built_year_range = context.built_year_range or profile.built_year_range
        analysis_era_start = era_start if era_start is not None else built_year_range[0] - 25
        analysis_era_end = era_end if era_end is not None else built_year_range[1] + 25

        for material_group in profile.material_group_ids():
            species_filter = list(material_group_species(material_group))
            if not species_filter:
                continue

            entries = self.index.search(
                species=species_filter,
                states=support_states,
                min_year=analysis_era_start - len(values),
                max_year=analysis_era_end,
                min_overlap=min_overlap,
            )
            support_reference_count = len(entries)
            support_state_count = len({entry.state for entry in entries if entry.state})
            minimum_references = 4
            minimum_states = 2

            group = profile.group(material_group)
            support_status = group.support_status if group is not None else "unsupported"
            if support_reference_count < minimum_references or support_state_count < minimum_states:
                support_status = {
                    "required_coverage": "required_coverage_missing",
                    "supported_with_caution": "insufficient",
                    "supported": "insufficient",
                }.get(support_status, "insufficient")

            report = self.matcher.date_sample(
                values=values,
                sample_name=f"{sample_name}:{material_group}",
                has_bark_edge=has_bark_edge,
                orientation=orientation,
                species_filter=species_filter,
                state_filter=support_states,
                era_start=analysis_era_start,
                era_end=analysis_era_end,
                min_overlap=min_overlap,
                top_n=top_n,
            )
            dating_reports[material_group] = report

            best = report.best_candidate
            context_prior_weight = float(profile.context_prior_weights.get(material_group, 0.0))
            member_prior_weight = float(member_priors.get(material_group, 0.0))
            context_weight = round((context_prior_weight * 0.45) + (member_prior_weight * 0.55), 4)
            top_score = float(best.composite_score) if best is not None else 0.0
            score_gap = float(best.score_gap_to_next) if best is not None else 0.0
            support_strength = min(
                1.0,
                (support_reference_count / max(minimum_references, 1)) * 0.6
                + (support_state_count / max(minimum_states, 1)) * 0.4,
            )

            score = (context_weight * 0.35) + (top_score * 0.45) + (support_strength * 0.15) + (score_gap * 0.05)
            if support_status == "supported_with_caution":
                score *= 0.9
            elif support_status in {"required_coverage_missing", "insufficient"}:
                score *= 0.2 if support_status == "required_coverage_missing" else 0.4

            evidence: list[str] = [
                f"context-weight={context_weight:.3f}",
                f"context-prior={context_prior_weight:.3f}",
                f"member-prior={member_prior_weight:.3f}",
                f"support={support_reference_count} refs/{support_state_count} states",
                f"dating-status={report.status}",
            ]
            if best is not None:
                evidence.append(f"top-score={best.composite_score:.3f}")
                evidence.append(f"top-reference={best.reference_name}")
                material_from_reference = infer_material_group_from_species(best.reference_species)
                if material_from_reference and material_from_reference != material_group:
                    evidence.append(f"top-reference-material={material_from_reference}")

            candidates.append(
                MaterialCandidate(
                    material_group=material_group,
                    display_name=material_group_display_name(material_group),
                    context_weight=context_weight,
                    support_reference_count=support_reference_count,
                    support_state_count=support_state_count,
                    support_status=support_status,
                    support_strength=support_strength,
                    score=score,
                    dating_status=report.status,
                    best_outer_ring_year=best.outer_ring_year if best is not None else None,
                    best_reference_name=best.reference_name if best is not None else "",
                    evidence=evidence,
                )
            )

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        if not candidates:
            warnings.append("No material groups were evaluable for the requested Walpole context.")
            return MaterialInferenceReport(
                status="inconclusive",
                context=context,
                context_profile=profile,
                support_status="no_candidates",
                recommended_material=None,
                candidates=[],
                warnings=warnings,
                diagnostics=diagnostics,
            )

        best_candidate = candidates[0]
        recommended_material: Optional[str] = None
        best_report = dating_reports.get(best_candidate.material_group)
        next_score = candidates[1].score if len(candidates) > 1 else 0.0
        score_separation = best_candidate.score - next_score

        diagnostics["score_separation"] = score_separation
        diagnostics["dating_status_by_material"] = {
            key: report.status for key, report in dating_reports.items()
        }

        supported_statuses = {"supported", "supported_with_caution"}
        support_status = "supported" if any(candidate.support_status in supported_statuses for candidate in candidates) else "insufficient"
        status = "ranked"

        if (
            best_candidate.support_status in supported_statuses
            and best_report is not None
            and best_report.best_candidate is not None
            and best_report.status in {"recommended", "ranked"}
            and best_candidate.score >= 0.46
            and score_separation >= 0.05
            and best_candidate.context_weight >= 0.16
        ):
            recommended_material = best_candidate.material_group
            status = "recommended"
        elif best_candidate.support_status in {"required_coverage_missing", "insufficient"}:
            warnings.append(
                "Top material group lacks sufficient Walpole-focused reference coverage. Returning inconclusive."
            )
            status = "inconclusive"
        else:
            warnings.append(
                "Material candidates were ranked, but no group cleared the recommendation policy."
            )

        return MaterialInferenceReport(
            status=status,
            context=context,
            context_profile=profile,
            support_status=support_status,
            recommended_material=recommended_material,
            candidates=candidates,
            warnings=warnings,
            diagnostics=diagnostics,
        )
