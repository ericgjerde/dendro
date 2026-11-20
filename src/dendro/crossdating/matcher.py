"""
Benchmark-driven assisted ranking for cross-dating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Optional

import numpy as np
import pandas as pd

from ..materials.catalog import infer_material_group_from_species
from .correlator import (
    CorrelationResult,
    _compute_sliding_stats,
    segment_correlation,
)
from .detrend import DetrendMethod, crossdate_transform, detrend_series, standardize
from ..reference.chronology_index import ChronologyIndex, ReferenceManifestEntry


POLICY_VERSION = "2026.04-assisted-ranking-v1"
YEAR_CONSENSUS_WINDOW = 2
TOP_PEAKS_PER_REFERENCE = 8
SPARSE_REFERENCE_WARNING_THRESHOLD = 8
SPARSE_REFERENCE_FORCE_BROAD_SEARCH_THRESHOLD = 3
GENUS_FALLBACK_MIN_REFERENCES = 8
GENUS_FALLBACK_MIN_STATES = 2
GENUS_FALLBACK_PENALTY = 0.025
BROAD_FALLBACK_PENALTY = 0.04
SUPPORTED_GENUS_FALLBACKS = {"PI", "QU"}


@dataclass
class SearchLane:
    """A single reference search lane used during candidate ranking."""

    name: str
    entries: list[ReferenceManifestEntry]
    lane_penalty: float = 0.0
    species_filter: Optional[list[str]] = None
    rationale: str = ""
    fallback: bool = False


@dataclass
class SearchPlan:
    """Resolved search lanes and diagnostics for a dating run."""

    lanes: list[SearchLane]
    reference_count: int
    combined_reference_count: int
    recommendation_blocked: bool = False
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


@dataclass
class DatingCandidate:
    """Ranked candidate alignment for a sample."""

    reference_id: str
    reference_name: str
    reference_species: str
    reference_state: str
    reference_file_type: str
    reference_material_group: str
    proposed_start_year: int
    proposed_end_year: int
    correlation: float
    t_value: float
    p_value: float
    overlap: int
    gleichlauf: float
    segment_correlations: list[tuple[int, float, float]] = field(default_factory=list)
    segment_consistency: float = 0.0
    composite_score: float = 0.0
    score_gap_to_next: float = 0.0
    year_cluster_support: int = 1
    year_cluster_unique_families: int = 1
    year_cluster_bonus: float = 0.0
    search_lane: str = "species_primary"
    search_lane_penalty: float = 0.0
    is_recommended: bool = False
    rationale: list[str] = field(default_factory=list)

    @property
    def outer_ring_year(self) -> int:
        return self.proposed_end_year

    @property
    def felling_year(self) -> int:
        # Backwards-compatible alias. The CLI treats this as a candidate outer ring
        # year unless bark-edge and recommendation criteria are both satisfied.
        return self.outer_ring_year

    @property
    def confidence(self) -> str:
        if self.is_recommended:
            return "RECOMMENDED"
        if self.composite_score >= 0.4:
            return "RANKED"
        return "WEAK"

    def to_dict(self) -> dict:
        return {
            "reference_id": self.reference_id,
            "reference_name": self.reference_name,
            "reference_species": self.reference_species,
            "reference_material_group": self.reference_material_group,
            "reference_state": self.reference_state,
            "reference_file_type": self.reference_file_type,
            "proposed_start_year": int(self.proposed_start_year),
            "outer_ring_year": int(self.outer_ring_year),
            "correlation": float(round(self.correlation, 4)),
            "t_value": float(round(self.t_value, 3)),
            "p_value": float(round(self.p_value, 6)),
            "overlap": int(self.overlap),
            "gleichlauf": float(round(self.gleichlauf, 2)),
            "segment_consistency": float(round(self.segment_consistency, 4)),
            "composite_score": float(round(self.composite_score, 4)),
            "score_gap_to_next": float(round(self.score_gap_to_next, 4)),
            "year_cluster_support": int(self.year_cluster_support),
            "year_cluster_unique_families": int(self.year_cluster_unique_families),
            "year_cluster_bonus": float(round(self.year_cluster_bonus, 4)),
            "search_lane": self.search_lane,
            "search_lane_penalty": float(round(self.search_lane_penalty, 4)),
            "is_recommended": bool(self.is_recommended),
            "rationale": list(self.rationale),
            "segment_correlations": [
                {
                    "start_index": int(start),
                    "correlation": float(round(corr, 4)),
                    "t_value": float(round(t_val, 3)),
                }
                for start, corr, t_val in self.segment_correlations
            ],
        }


@dataclass
class DatingReport:
    """Stable machine-readable dating report."""

    sample_name: str
    sample_length: int
    bark_edge: bool
    requested_orientation: str
    chosen_orientation: str
    analysis_orientation: str
    status: str
    policy_version: str
    candidates: list[DatingCandidate]
    material_inference: Optional[dict] = None
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def best_candidate(self) -> Optional[DatingCandidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def matches(self) -> list[DatingCandidate]:
        return self.candidates

    @property
    def has_bark_edge(self) -> bool:
        return self.bark_edge

    @property
    def consensus_year(self) -> Optional[int]:
        if self.status == "recommended" and self.best_candidate is not None:
            return self.best_candidate.outer_ring_year
        return None

    @property
    def consensus_confidence(self) -> str:
        return self.status.upper()

    @property
    def detrend_method(self) -> str:
        return str(self.diagnostics.get("detrend_method", "spline"))

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "policy_version": self.policy_version,
            "sample": {
                "name": self.sample_name,
                "length": int(self.sample_length),
                "bark_edge": bool(self.bark_edge),
                "requested_orientation": self.requested_orientation,
                "chosen_orientation": self.chosen_orientation,
                "analysis_orientation": self.analysis_orientation,
            },
            "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "material_inference": self._normalize_scalars(self.material_inference),
            "diagnostics": self._normalize_scalars(self.diagnostics),
            "warnings": list(self.warnings),
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


# Backwards-compatible aliases for older imports/tests.
MatchResult = DatingCandidate
CrossdateReport = DatingReport


class CrossdateMatcher:
    """Cross-dating engine producing ranked candidate reports."""

    def __init__(
        self,
        index: Optional[ChronologyIndex] = None,
        reference_dir: Optional[str | Path] = None,
        *,
        allow_remote_metadata: bool = False,
    ):
        if index is not None:
            self.index = index
        elif reference_dir is not None:
            self.index = ChronologyIndex(reference_dir, allow_remote_metadata=allow_remote_metadata)
        else:
            self.index = ChronologyIndex(allow_remote_metadata=allow_remote_metadata)

    def date_sample(
        self,
        values: np.ndarray,
        sample_name: str = "sample",
        has_bark_edge: bool = True,
        orientation: str = "auto",
        species_filter: Optional[list[str]] = None,
        state_filter: Optional[list[str]] = None,
        era_start: int = 1600,
        era_end: int = 1900,
        detrend_method: DetrendMethod = DetrendMethod.SPLINE,
        min_overlap: int = 30,
        max_references: int = 250,
        top_n: int = 10,
        allow_sparse_fallback: bool = True,
    ) -> DatingReport:
        values = np.asarray(values, dtype=np.float64)
        warnings: list[str] = []
        requested_species = [code.upper() for code in species_filter] if species_filter else None
        requested_states = [code.upper() for code in state_filter] if state_filter else None

        if len(values) < 10:
            warnings.append("Sample too short for assisted ranking. Minimum 10 rings required.")
            return self._build_report(
                sample_name=sample_name,
                sample_length=len(values),
                bark_edge=has_bark_edge,
                requested_orientation=orientation,
                chosen_orientation="oldest_to_newest",
                status="inconclusive",
                candidates=[],
                warnings=warnings,
                diagnostics={
                    "detrend_method": detrend_method.value,
                    "reference_count": 0,
                    "min_overlap": min_overlap,
                },
            )

        search_plan = self._build_search_plan(
            species_filter=requested_species,
            state_filter=requested_states,
            min_year=era_start - len(values),
            max_year=era_end,
            min_overlap=min_overlap,
            max_references=max_references,
            allow_sparse_fallback=allow_sparse_fallback,
        )
        warnings.extend(search_plan.warnings)

        if not search_plan.lanes:
            warnings.append("No reference chronologies matched the requested filters and era.")
            return self._build_report(
                sample_name=sample_name,
                sample_length=len(values),
                bark_edge=has_bark_edge,
                requested_orientation=orientation,
                chosen_orientation="oldest_to_newest",
                status="inconclusive",
                candidates=[],
                warnings=warnings,
                diagnostics={
                    "detrend_method": detrend_method.value,
                    "reference_count": int(search_plan.reference_count),
                    "combined_reference_count": int(search_plan.combined_reference_count),
                    "min_overlap": min_overlap,
                    **search_plan.diagnostics,
                },
            )

        use_high_pass = detrend_method == DetrendMethod.SPLINE
        orientation_runs = self._resolve_orientations(values, orientation)
        best_run: Optional[tuple[str, list[DatingCandidate], np.ndarray]] = None

        for assumed_orientation, normalized_values in orientation_runs:
            try:
                if use_high_pass:
                    sample_std = crossdate_transform(
                        normalized_values,
                        spline_period=20.0,
                        prewhiten_order=1,
                    )
                else:
                    detrended, _ = detrend_series(normalized_values, method=detrend_method)
                    sample_std = standardize(detrended, method="zscore")
            except Exception as exc:
                warnings.append(f"Detrending failed for {assumed_orientation}: {exc}")
                continue

            ranked_candidates = self._rank_candidates(
                sample_std=sample_std,
                bark_edge=has_bark_edge,
                search_lanes=search_plan.lanes,
                era_start=era_start,
                era_end=era_end,
                min_overlap=min_overlap,
                top_n=top_n,
                species_filter=requested_species,
                state_filter=requested_states,
                use_high_pass=use_high_pass,
            )

            if best_run is None:
                best_run = (assumed_orientation, ranked_candidates, sample_std)
                continue

            current_best = ranked_candidates[0].composite_score if ranked_candidates else -1.0
            previous_best = best_run[1][0].composite_score if best_run[1] else -1.0
            if current_best > previous_best:
                best_run = (assumed_orientation, ranked_candidates, sample_std)

        if best_run is None:
            warnings.append("All analysis attempts failed during sample preparation.")
            return self._build_report(
                sample_name=sample_name,
                sample_length=len(values),
                bark_edge=has_bark_edge,
                requested_orientation=orientation,
                chosen_orientation="oldest_to_newest",
                status="inconclusive",
                candidates=[],
                warnings=warnings,
                diagnostics={
                    "detrend_method": detrend_method.value,
                    "reference_count": int(search_plan.reference_count),
                    "combined_reference_count": int(search_plan.combined_reference_count),
                    "min_overlap": min_overlap,
                    **search_plan.diagnostics,
                },
            )

        chosen_orientation, ranked_candidates, sample_std = best_run
        if ranked_candidates:
            ranked_candidates[0].is_recommended = self._is_recommended(
                ranked_candidates[0],
                recommendation_blocked=search_plan.recommendation_blocked,
            )
            ranked_candidates = self._select_top_candidates(
                ranked_candidates,
                top_n=top_n,
                is_recommended=ranked_candidates[0].is_recommended,
            )
        status = self._determine_status(ranked_candidates, warnings)

        if status == "ranked" and ranked_candidates:
            warnings.append(
                "Candidates were ranked, but no alignment cleared the recommendation policy. "
                "Treat the output as assisted ranking rather than a secure date."
            )
        if status == "inconclusive" and not ranked_candidates:
            warnings.append("No usable candidate alignments were produced.")

        return self._build_report(
            sample_name=sample_name,
            sample_length=len(values),
            bark_edge=has_bark_edge,
            requested_orientation=orientation,
            chosen_orientation=chosen_orientation,
            status=status,
            candidates=ranked_candidates[:top_n],
            warnings=warnings,
            diagnostics={
                "detrend_method": detrend_method.value,
                "reference_count": int(search_plan.reference_count),
                "combined_reference_count": int(search_plan.combined_reference_count),
                "min_overlap": int(min_overlap),
                "top_score": float(ranked_candidates[0].composite_score) if ranked_candidates else None,
                "top_correlation": float(ranked_candidates[0].correlation) if ranked_candidates else None,
                "top_t_value": float(ranked_candidates[0].t_value) if ranked_candidates else None,
                **search_plan.diagnostics,
            },
        )

    def _resolve_orientations(
        self,
        values: np.ndarray,
        orientation: str,
    ) -> list[tuple[str, np.ndarray]]:
        if orientation == "oldest_to_newest":
            return [("oldest_to_newest", values)]
        if orientation == "bark_to_pith":
            return [("bark_to_pith", values[::-1])]
        return [
            ("oldest_to_newest", values),
            ("bark_to_pith", values[::-1]),
        ]

    def _build_search_plan(
        self,
        *,
        species_filter: Optional[list[str]],
        state_filter: Optional[list[str]],
        min_year: int,
        max_year: int,
        min_overlap: int,
        max_references: int,
        allow_sparse_fallback: bool = True,
    ) -> SearchPlan:
        warnings: list[str] = []
        lanes: list[SearchLane] = []
        lane_summaries: list[dict] = []

        if not species_filter:
            entries = self.index.search(
                states=state_filter,
                min_year=min_year,
                max_year=max_year,
                min_overlap=min_overlap,
            )[:max_references]
            if entries:
                lanes.append(
                    SearchLane(
                        name="all_references",
                        entries=entries,
                        rationale="all-reference search lane",
                    )
                )
                lane_summaries.append(
                    {
                        "name": "all_references",
                        "reference_count": len(entries),
                        "fallback": False,
                        "lane_penalty": 0.0,
                    }
                )
            return SearchPlan(
                lanes=lanes,
                reference_count=len(entries),
                combined_reference_count=len(entries),
                diagnostics={
                    "search_strategy": "all_references",
                    "search_lanes": lane_summaries,
                    "sparse_reference_coverage": False,
                },
            )

        primary_entries = self.index.search(
            species=species_filter,
            states=state_filter,
            min_year=min_year,
            max_year=max_year,
            min_overlap=min_overlap,
        )
        primary_count = len(primary_entries)
        sparse_coverage = primary_count < SPARSE_REFERENCE_WARNING_THRESHOLD

        if primary_entries:
            primary_lane_entries = primary_entries[:max_references]
            lanes.append(
                SearchLane(
                    name="species_primary",
                    entries=primary_lane_entries,
                    species_filter=species_filter,
                    rationale="same-species lane",
                )
            )
            lane_summaries.append(
                {
                    "name": "species_primary",
                    "reference_count": len(primary_lane_entries),
                    "fallback": False,
                    "lane_penalty": 0.0,
                }
            )

        fallback_species: set[str] = set(species_filter)
        if sparse_coverage and allow_sparse_fallback:
            genus_lane = self._build_genus_fallback_lane(
                species_filter=species_filter,
                state_filter=state_filter,
                min_year=min_year,
                max_year=max_year,
                min_overlap=min_overlap,
                max_references=max_references,
            )
            if genus_lane is not None:
                lanes.append(genus_lane)
                lane_summaries.append(
                    {
                        "name": genus_lane.name,
                        "reference_count": len(genus_lane.entries),
                        "fallback": True,
                        "lane_penalty": genus_lane.lane_penalty,
                    }
                )
                fallback_species.update(genus_lane.species_filter or [])

            if primary_count <= SPARSE_REFERENCE_FORCE_BROAD_SEARCH_THRESHOLD or genus_lane is None:
                broad_lane = self._build_broad_fallback_lane(
                    state_filter=state_filter,
                    min_year=min_year,
                    max_year=max_year,
                    min_overlap=min_overlap,
                    max_references=max_references,
                    exclude_species=sorted(fallback_species),
                )
                if broad_lane is not None:
                    lanes.append(broad_lane)
                    lane_summaries.append(
                        {
                            "name": broad_lane.name,
                            "reference_count": len(broad_lane.entries),
                            "fallback": True,
                            "lane_penalty": broad_lane.lane_penalty,
                        }
                    )

        if primary_count == 0:
            if allow_sparse_fallback:
                warnings.append(
                    "No same-species references matched the requested filters. Broader fallback lanes were used."
                )
            else:
                warnings.append("No same-species references matched the requested filters.")
        elif primary_count <= SPARSE_REFERENCE_FORCE_BROAD_SEARCH_THRESHOLD:
            if allow_sparse_fallback:
                warnings.append(
                    f"Same-species coverage is extremely sparse ({primary_count} references). "
                    "Broader fallback lanes were evaluated and recommendation is disabled."
                )
            else:
                warnings.append(
                    f"Same-species coverage is extremely sparse ({primary_count} references). "
                    "Recommendation is disabled."
                )
        elif sparse_coverage:
            if allow_sparse_fallback:
                warnings.append(
                    f"Same-species coverage is sparse ({primary_count} references). "
                    "Fallback lanes were evaluated and recommendation is disabled."
                )
            else:
                warnings.append(
                    f"Same-species coverage is sparse ({primary_count} references). "
                    "Recommendation is disabled."
                )

        combined_reference_count = sum(len(lane.entries) for lane in lanes)
        diagnostics = {
            "search_strategy": (
                "sparse_fallback"
                if (sparse_coverage and allow_sparse_fallback)
                else "species_primary"
            ),
            "search_lanes": lane_summaries,
            "sparse_reference_coverage": sparse_coverage,
            "requested_species_filter": list(species_filter),
        }
        if state_filter:
            diagnostics["requested_state_filter"] = list(state_filter)

        return SearchPlan(
            lanes=lanes,
            reference_count=primary_count,
            combined_reference_count=combined_reference_count,
            recommendation_blocked=sparse_coverage,
            warnings=warnings,
            diagnostics=diagnostics,
        )

    def _build_genus_fallback_lane(
        self,
        *,
        species_filter: list[str],
        state_filter: Optional[list[str]],
        min_year: int,
        max_year: int,
        min_overlap: int,
        max_references: int,
    ) -> Optional[SearchLane]:
        genera = {self._species_genus_code(code) for code in species_filter if self._species_genus_code(code)}
        if len(genera) != 1:
            return None

        genus = next(iter(genera))
        if genus not in SUPPORTED_GENUS_FALLBACKS:
            return None

        alt_species = sorted(
            code
            for code in self.index.get_species()
            if self._species_genus_code(code) == genus and code not in species_filter
        )
        if not alt_species:
            return None

        entries = self.index.search(
            species=alt_species,
            states=state_filter,
            min_year=min_year,
            max_year=max_year,
            min_overlap=min_overlap,
        )
        fallback_states = {entry.state for entry in entries if entry.state}
        if len(entries) < GENUS_FALLBACK_MIN_REFERENCES or len(fallback_states) < GENUS_FALLBACK_MIN_STATES:
            return None

        return SearchLane(
            name=f"{genus.lower()}_genus_fallback",
            entries=entries[:max_references],
            lane_penalty=GENUS_FALLBACK_PENALTY,
            species_filter=alt_species,
            rationale=f"same-genus fallback ({genus}*)",
            fallback=True,
        )

    def _build_broad_fallback_lane(
        self,
        *,
        state_filter: Optional[list[str]],
        min_year: int,
        max_year: int,
        min_overlap: int,
        max_references: int,
        exclude_species: list[str],
    ) -> Optional[SearchLane]:
        entries = self.index.search(
            states=state_filter,
            min_year=min_year,
            max_year=max_year,
            min_overlap=min_overlap,
        )
        filtered_entries = [
            entry for entry in entries
            if not entry.species or entry.species not in exclude_species
        ]
        if not filtered_entries:
            return None

        return SearchLane(
            name="broad_fallback",
            entries=filtered_entries[:max_references],
            lane_penalty=BROAD_FALLBACK_PENALTY,
            rationale="broad fallback lane",
            fallback=True,
        )

    def _species_genus_code(self, species_code: str) -> str:
        code = (species_code or "").upper()
        return code[:2] if len(code) >= 2 else ""

    def _rank_candidates(
        self,
        *,
        sample_std: np.ndarray,
        bark_edge: bool,
        search_lanes: list[SearchLane],
        era_start: int,
        era_end: int,
        min_overlap: int,
        top_n: int,
        species_filter: Optional[list[str]],
        state_filter: Optional[list[str]],
        use_high_pass: bool = True,
    ) -> list[DatingCandidate]:
        n_sample = len(sample_std)
        min_position = int(era_start) - n_sample + 1
        max_position = int(era_end) - n_sample + 1

        # Pass 1: Collect top peaks per reference within the requested era and
        # build a cross-reference calendar-year evidence grid.
        raw_peaks: list[tuple[SearchLane, ReferenceManifestEntry, np.ndarray, CorrelationResult, float]] = []
        year_evidence: dict[int, float] = {}
        year_support_refs: dict[int, set[str]] = {}
        year_support_families: dict[int, set[str]] = {}
        seen_ref_ids: set[str] = set()

        primary_lane_count = sum(
            len(lane.entries) for lane in search_lanes if not lane.fallback
        )

        for lane in search_lanes:
            if not lane.fallback:
                lane_weight = 1.0
            elif lane.name.endswith("_genus_fallback"):
                lane_weight = 0.40 if primary_lane_count >= 2 else 0.70
            else:
                lane_weight = 0.15 if primary_lane_count >= 2 else 0.60

            for entry in lane.entries:
                ref_key = f"{entry.site_id}:{entry.filepath}"
                if ref_key in seen_ref_ids:
                    continue
                seen_ref_ids.add(ref_key)

                reference = entry.master_crossdate_values if use_high_pass else entry.master_values
                if len(reference) < min_overlap:
                    continue

                positions, corrs, t_vals, p_vals, overlaps, glks = _compute_sliding_stats(
                    sample_std,
                    reference,
                    entry.master_start_year,
                    min_overlap=min_overlap,
                    min_position=min_position,
                    max_position=max_position,
                    compute_glk=True,
                )
                if len(positions) == 0:
                    continue

                t_finite = np.where(np.isfinite(t_vals), t_vals, 15.0)
                t_med = float(np.median(t_finite))
                t_mad = float(np.median(np.abs(t_finite - t_med))) * 1.4826
                t_z_arr = (t_finite - t_med) / max(t_mad, 0.5)

                k = min(TOP_PEAKS_PER_REFERENCE, len(t_vals))
                top_idx = np.argpartition(t_finite, -k)[-k:]
                top_idx = top_idx[np.argsort(t_finite[top_idx])[::-1]]

                family_name = self._normalize_reference_family(entry.site_name)
                kept_any = False
                for rank_i, idx in enumerate(top_idx):
                    tf = float(t_finite[idx])
                    rf = float(corrs[idx])
                    if rank_i > 0 and (tf <= 2.0 or rf <= 0.12):
                        continue

                    result = CorrelationResult(
                        position=int(positions[idx]),
                        correlation=rf,
                        t_value=float(t_vals[idx]),
                        p_value=float(p_vals[idx]),
                        overlap=int(overlaps[idx]),
                        gleichlauf=float(glks[idx]),
                    )
                    tz = float(t_z_arr[idx])
                    raw_peaks.append((lane, entry, reference, result, tz))
                    kept_any = True

                    if tf > 2.0 and rf > 0.12:
                        end_yr = int(result.position + n_sample - 1)
                        ov_ratio = float(result.overlap) / max(n_sample, 1)
                        contrib = (
                            max(0.0, tf - 2.0)
                            * max(0.0, rf)
                            * (ov_ratio ** 0.75)
                            * (1.0 + 0.2 * max(0.0, tz - 2.5))
                            * lane_weight
                        )
                        year_evidence[end_yr] = year_evidence.get(end_yr, 0.0) + contrib
                        year_support_refs.setdefault(end_yr, set()).add(entry.site_id)
                        year_support_families.setdefault(end_yr, set()).add(family_name)

                if not kept_any and len(top_idx) > 0:
                    idx = top_idx[0]
                    result = CorrelationResult(
                        position=int(positions[idx]),
                        correlation=float(corrs[idx]),
                        t_value=float(t_vals[idx]),
                        p_value=float(p_vals[idx]),
                        overlap=int(overlaps[idx]),
                        gleichlauf=float(glks[idx]),
                    )
                    raw_peaks.append((lane, entry, reference, result, float(t_z_arr[idx])))

        if not raw_peaks:
            return []

        max_year_ev = max(year_evidence.values()) if year_evidence else 0.0

        # Pass 2: Score peaks with multi-site year consensus and keep the best
        # alignment per reference site.
        best_per_ref: dict[str, tuple[float, SearchLane, ReferenceManifestEntry, np.ndarray, CorrelationResult, float, int, int, float]] = {}

        for lane, entry, reference, result, tz in raw_peaks:
            end_yr = int(result.position + n_sample - 1)

            # Aggregate cluster support within +/- YEAR_CONSENSUS_WINDOW,
            # weighting exact-year evidence highest.
            cluster_refs: set[str] = set()
            cluster_families: set[str] = set()
            exact_ev = year_evidence.get(end_yr, 0.0)
            neighbor_ev = 0.0
            for delta in range(-YEAR_CONSENSUS_WINDOW, YEAR_CONSENSUS_WINDOW + 1):
                yr_d = end_yr + delta
                if yr_d in year_support_refs:
                    cluster_refs.update(year_support_refs[yr_d])
                    cluster_families.update(year_support_families[yr_d])
                if delta != 0 and yr_d in year_evidence:
                    neighbor_ev += year_evidence[yr_d] * (0.25 if abs(delta) == 1 else 0.1)

            support = max(1, len(cluster_refs))
            unique_families = max(1, len(cluster_families))
            ev_ratio = ((exact_ev + neighbor_ev) / max_year_ev) if max_year_ev > 0.0 else 0.0

            # Fast pre-score (without segment_correlation yet) to select best peak per reference.
            base_fast = self._score_candidate(
                result=result,
                segment_consistency=0.0,
                entry=entry,
                species_filter=species_filter,
                state_filter=state_filter,
                sample_length=n_sample,
                peak_z=tz,
            ) - lane.lane_penalty

            # Consensus bonus rewards multi-site agreement and normalized year-grid dominance.
            cluster_bonus = self._compute_consensus_bonus(
                support=support,
                unique_families=unique_families,
                ev_ratio=ev_ratio,
                exact_ev=exact_ev,
                max_year_ev=max_year_ev,
            )
            prelim_score = base_fast + cluster_bonus

            # Keep the strongest peak per (reference, year_bucket) so a strong
            # runner-up match at an alternate calendar year is not erased when
            # the same reference also happens to peak near another cluster.
            year_bucket = end_yr // (YEAR_CONSENSUS_WINDOW * 2 + 1)
            ref_key = f"{entry.site_id}:{entry.filepath}:{year_bucket}"
            prev = best_per_ref.get(ref_key)
            if prev is None or prelim_score > prev[0]:
                best_per_ref[ref_key] = (
                    prelim_score,
                    lane,
                    entry,
                    reference,
                    result,
                    tz,
                    support,
                    unique_families,
                    cluster_bonus,
                )

        # Pass 3: Compute exact segment correlations (supporting partial overlaps)
        # for the top candidate peaks and construct DatingCandidate objects.
        top_prelim = sorted(best_per_ref.values(), key=lambda item: item[0], reverse=True)
        eval_limit = max(top_n * 4, 50)
        ranked: list[DatingCandidate] = []
        for (
            _prelim,
            lane,
            entry,
            reference,
            best,
            tz,
            support,
            unique_families,
            cluster_bonus,
        ) in top_prelim[:eval_limit]:
            proposed_end = int(best.position + n_sample - 1)
            offset = int(best.position - entry.master_start_year)
            aligned_sample, aligned_reference = self._aligned_slices(sample_std, reference, offset)
            if aligned_sample is not None and aligned_reference is not None and len(aligned_sample) >= 20:
                seg_len = min(50, max(20, len(aligned_sample) // 2))
                seg_lag = max(10, min(25, len(aligned_sample) // 4))
                seg_corrs = segment_correlation(
                    aligned_sample,
                    aligned_reference,
                    segment_length=seg_len,
                    lag=seg_lag,
                )
            else:
                seg_corrs = []

            segment_consistency = self._segment_consistency(seg_corrs)
            base_score = self._score_candidate(
                result=best,
                segment_consistency=segment_consistency,
                entry=entry,
                species_filter=species_filter,
                state_filter=state_filter,
                sample_length=n_sample,
                peak_z=tz,
            ) - lane.lane_penalty

            composite_score = max(0.0, base_score + cluster_bonus)
            rationale = self._candidate_rationale(best, segment_consistency, bark_edge, entry)
            if lane.rationale:
                rationale.append(lane.rationale)
            if unique_families >= 3:
                rationale.append("multi-family year consensus")
            elif support >= 2:
                rationale.append("limited year consensus")
            else:
                rationale.append("isolated-year candidate")
            if support >= 2:
                rationale.append(f"year-consensus={support}")

            ranked.append(
                DatingCandidate(
                    reference_id=entry.site_id,
                    reference_name=entry.site_name,
                    reference_species=entry.species,
                    reference_material_group=entry.material_group or infer_material_group_from_species(entry.species),
                    reference_state=entry.state,
                    reference_file_type=entry.file_type,
                    proposed_start_year=int(best.position),
                    proposed_end_year=proposed_end,
                    correlation=float(best.correlation),
                    t_value=float(best.t_value),
                    p_value=float(best.p_value),
                    overlap=int(best.overlap),
                    gleichlauf=float(best.gleichlauf),
                    segment_correlations=seg_corrs,
                    segment_consistency=float(segment_consistency),
                    composite_score=float(composite_score),
                    year_cluster_support=int(support),
                    year_cluster_unique_families=int(unique_families),
                    year_cluster_bonus=float(cluster_bonus),
                    search_lane=lane.name,
                    search_lane_penalty=float(lane.lane_penalty),
                    rationale=rationale,
                )
            )

        ranked.sort(
            key=lambda candidate: (
                candidate.composite_score,
                candidate.year_cluster_bonus,
                candidate.year_cluster_support,
                candidate.t_value,
                candidate.correlation,
            ),
            reverse=True,
        )

        # Compute score_gap_to_next against the strongest candidate from a
        # DISTINCT calendar year (outside +/- YEAR_CONSENSUS_WINDOW) so that
        # strong multi-site agreement on the same year does not penalize itself!
        for candidate in ranked:
            next_distinct_score = 0.0
            for other in ranked:
                if abs(other.outer_ring_year - candidate.outer_ring_year) > YEAR_CONSENSUS_WINDOW:
                    next_distinct_score = other.composite_score
                    break
            candidate.score_gap_to_next = float(candidate.composite_score - next_distinct_score)

        return ranked

    def _compute_consensus_bonus(
        self,
        *,
        support: int,
        unique_families: int,
        ev_ratio: float,
        exact_ev: float,
        max_year_ev: float,
    ) -> float:
        if support <= 1 and max_year_ev > 0.0:
            # Isolated single-site peak: penalize slightly unless it IS the dominant evidence.
            return 0.06 * ev_ratio - 0.04
        family_bonus = min(0.18, max(0.0, 0.025 * (support - 1) + 0.02 * (unique_families - 1)))
        exact_ratio = (exact_ev / max_year_ev) if max_year_ev > 0.0 else 0.0
        evidence_bonus = 0.28 * min(1.0, ev_ratio) + 0.10 * min(1.0, exact_ratio)
        return family_bonus + evidence_bonus

    def _apply_year_consensus_bonus(self, ranked: list[DatingCandidate]):
        """Apply standalone year-consensus bonus to a pre-built candidate list."""
        if not ranked:
            return

        family_cache = {
            candidate.reference_id: self._normalize_reference_family(candidate.reference_name)
            for candidate in ranked
        }

        for candidate in ranked:
            cluster = [
                other
                for other in ranked
                if abs(other.outer_ring_year - candidate.outer_ring_year) <= YEAR_CONSENSUS_WINDOW
            ]
            support = len(cluster)
            unique_families = len({family_cache[other.reference_id] for other in cluster})
            raw_score = candidate.composite_score
            bonus = min(0.18, max(0.0, 0.03 * (support - 1) + 0.025 * (unique_families - 1)))
            if support == 1:
                bonus -= 0.03

            candidate.year_cluster_support = support
            candidate.year_cluster_unique_families = unique_families
            candidate.year_cluster_bonus = bonus
            candidate.composite_score = raw_score + bonus

            if unique_families >= 3:
                candidate.rationale.append("multi-family year consensus")
            elif support >= 2:
                candidate.rationale.append("limited year consensus")
            else:
                candidate.rationale.append("isolated-year candidate")
            if support >= 2:
                candidate.rationale.append(f"year-consensus={support}")

    def _apply_year_consensus(self, candidates: list[DatingCandidate]):
        # Backwards-compatible helper retained for older tests and callers.
        self._apply_year_consensus_bonus(candidates)

    def _normalize_reference_family(self, reference_name: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", reference_name.lower()).strip()
        tokens = [
            token
            for token in normalized.split()
            if token not in {"update", "historical", "recollection", "core", "cores", "long", "new"}
        ]
        return " ".join(tokens) or normalized

    def _aligned_slices(
        self,
        sample: np.ndarray,
        reference: np.ndarray,
        offset: int,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return aligned overlapping slices of (sample, reference), supporting partial overlaps."""
        n_sample = len(sample)
        n_ref = len(reference)
        s_start = max(0, -offset)
        s_end = min(n_sample, n_ref - offset)
        r_start = max(0, offset)
        r_end = min(n_ref, offset + n_sample)
        if s_end - s_start < 20:
            return None, None
        return sample[s_start:s_end], reference[r_start:r_end]

    def _aligned_reference(
        self,
        reference: np.ndarray,
        offset: int,
        sample_length: int,
    ) -> Optional[np.ndarray]:
        if offset < 0 or offset + sample_length > len(reference):
            return None
        return reference[offset:offset + sample_length]

    def _segment_consistency(self, segment_results: list[tuple[int, float, float]]) -> float:
        if not segment_results:
            return 0.0
        strong = sum(1 for _, _, t_val in segment_results if t_val >= 3.0)
        return strong / len(segment_results)

    def _score_candidate(
        self,
        *,
        result: CorrelationResult,
        segment_consistency: float,
        entry: ReferenceManifestEntry,
        species_filter: Optional[list[str]],
        state_filter: Optional[list[str]],
        sample_length: Optional[int] = None,
        peak_z: float = 0.0,
    ) -> float:
        corr_score = max(0.0, min(result.correlation, 1.0))
        t_val = result.t_value if np.isfinite(result.t_value) else 15.0
        t_score = max(0.0, min(t_val / 8.0, 1.0))
        abs_overlap_score = max(0.0, min(result.overlap / 100.0, 1.0))
        if sample_length and sample_length > 0:
            frac_overlap = max(0.0, min(result.overlap / float(sample_length), 1.0))
            overlap_score = 0.45 * abs_overlap_score + 0.55 * (frac_overlap ** 0.75)
            # Attenuate short edge-sliver correlations on long samples so a 30-year
            # sliver cannot outrank a full-length match purely on inflated raw r.
            edge_factor = 0.65 + 0.35 * (frac_overlap ** 0.5)
        else:
            overlap_score = abs_overlap_score
            edge_factor = 1.0
        glk_score = max(0.0, min(result.gleichlauf / 100.0, 1.0))
        prominence_score = max(0.0, min((peak_z - 2.0) / 6.0, 1.0))

        meta_score = 1.0
        if not entry.species:
            meta_score -= 0.05
        if not entry.state:
            meta_score -= 0.03
        if species_filter and entry.species in [code.upper() for code in species_filter]:
            meta_score += 0.02
        if state_filter and entry.state in [code.upper() for code in state_filter]:
            meta_score += 0.02

        composite = (
            corr_score * 0.24
            + t_score * 0.32
            + overlap_score * 0.16
            + glk_score * 0.08
            + segment_consistency * 0.12
            + prominence_score * 0.08
        ) * edge_factor * max(0.0, meta_score)
        return composite

    def _candidate_rationale(
        self,
        result: CorrelationResult,
        segment_consistency: float,
        bark_edge: bool,
        entry: ReferenceManifestEntry,
    ) -> list[str]:
        rationale: list[str] = []

        if result.correlation >= 0.5:
            rationale.append("strong correlation")
        elif result.correlation >= 0.35:
            rationale.append("moderate correlation")
        else:
            rationale.append("weak correlation")

        if result.t_value >= 6.0:
            rationale.append("high t-value")
        elif result.t_value >= 4.0:
            rationale.append("useful but not secure t-value")
        else:
            rationale.append("borderline t-value")

        if segment_consistency >= 0.7:
            rationale.append("consistent segments")
        elif segment_consistency > 0:
            rationale.append("mixed segment support")
        else:
            rationale.append("no segment consistency")

        if bark_edge:
            rationale.append("bark-edge supplied")
        else:
            rationale.append("outer-ring year only")

        if entry.species:
            rationale.append(f"species={entry.species}")

        return rationale

    def _select_top_candidates(
        self,
        ranked: list[DatingCandidate],
        *,
        top_n: int,
        is_recommended: bool,
    ) -> list[DatingCandidate]:
        """
        Select top_n candidates in strict descending composite_score order.

        For recommended dates, return the top-scoring distinct corroborating
        references directly. For non-recommended (assisted-ranking) dates, keep
        the strongest representative per +/-2-year cluster so runner-up
        alternative calendar years remain visible in the top-N output, then
        backfill any remaining slots.
        """
        if is_recommended:
            unique_refs: list[DatingCandidate] = []
            seen_refs: set[str] = set()
            for candidate in ranked:
                if candidate.reference_id in seen_refs:
                    continue
                seen_refs.add(candidate.reference_id)
                unique_refs.append(candidate)
                if len(unique_refs) >= top_n:
                    break
            return unique_refs

        selected: list[DatingCandidate] = []
        deferred: list[DatingCandidate] = []
        cluster_counts: list[tuple[int, int]] = []  # (anchor_year, count)

        for candidate in ranked:
            matched_idx = None
            for idx, (anchor_year, count) in enumerate(cluster_counts):
                if abs(candidate.outer_ring_year - anchor_year) <= YEAR_CONSENSUS_WINDOW:
                    matched_idx = idx
                    break
            if matched_idx is None:
                cluster_counts.append((candidate.outer_ring_year, 1))
                selected.append(candidate)
            else:
                deferred.append(candidate)
            if len(selected) >= top_n:
                break

        if len(selected) < top_n and deferred:
            selected.extend(deferred[: top_n - len(selected)])
            selected.sort(
                key=lambda candidate: (
                    candidate.composite_score,
                    candidate.year_cluster_bonus,
                    candidate.year_cluster_support,
                    candidate.t_value,
                    candidate.correlation,
                ),
                reverse=True,
            )

        return selected[:top_n]

    def _is_recommended(self, candidate: DatingCandidate, *, recommendation_blocked: bool = False) -> bool:
        if recommendation_blocked:
            return False
        return bool(
            candidate.composite_score >= 0.62
            and candidate.correlation >= 0.35
            and candidate.t_value >= 5.0
            and candidate.overlap >= 50
            and candidate.segment_consistency >= 0.5
            and candidate.year_cluster_support >= 2
            and candidate.score_gap_to_next >= 0.05
        )

    def _determine_status(self, candidates: list[DatingCandidate], warnings: list[str]) -> str:
        if not candidates:
            return "inconclusive"

        best = candidates[0]
        if best.is_recommended:
            return "recommended"

        if best.composite_score >= 0.25:
            return "ranked"

        warnings.append("Candidate scores were too weak to support even assisted ranking.")
        return "inconclusive"

    def _build_report(
        self,
        *,
        sample_name: str,
        sample_length: int,
        bark_edge: bool,
        requested_orientation: str,
        chosen_orientation: str,
        status: str,
        candidates: list[DatingCandidate],
        warnings: list[str],
        diagnostics: dict,
        material_inference: Optional[dict] = None,
    ) -> DatingReport:
        return DatingReport(
            sample_name=sample_name,
            sample_length=sample_length,
            bark_edge=bark_edge,
            requested_orientation=requested_orientation,
            chosen_orientation=chosen_orientation,
            analysis_orientation="oldest_to_newest",
            status=status,
            policy_version=POLICY_VERSION,
            candidates=candidates,
            material_inference=material_inference,
            warnings=warnings,
            diagnostics=diagnostics,
        )


def date_measurements(
    measurements: np.ndarray | pd.DataFrame,
    reference_dir: str | Path,
    sample_name: str = "sample",
    has_bark_edge: bool = True,
    orientation: str = "auto",
    species: Optional[list[str]] = None,
    states: Optional[list[str]] = None,
    era_start: int = 1600,
    era_end: int = 1900,
) -> DatingReport:
    if isinstance(measurements, pd.DataFrame):
        for preferred in ("width_mm", "width"):
            if preferred in measurements.columns:
                values = measurements[preferred].values
                break
        else:
            values = measurements.select_dtypes(include=[np.number]).iloc[:, -1].values
    else:
        values = np.asarray(measurements)

    matcher = CrossdateMatcher(reference_dir=reference_dir)
    return matcher.date_sample(
        values=values,
        sample_name=sample_name,
        has_bark_edge=has_bark_edge,
        orientation=orientation,
        species_filter=species,
        state_filter=states,
        era_start=era_start,
        era_end=era_end,
    )
