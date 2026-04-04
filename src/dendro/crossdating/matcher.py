"""
Benchmark-driven assisted ranking for cross-dating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .correlator import CorrelationResult, find_best_match, segment_correlation
from .detrend import DetrendMethod, detrend_series, standardize
from ..reference.chronology_index import ChronologyIndex, ReferenceManifestEntry


POLICY_VERSION = "2026.04-assisted-ranking-v1"


@dataclass
class DatingCandidate:
    """Ranked candidate alignment for a sample."""

    reference_id: str
    reference_name: str
    reference_species: str
    reference_state: str
    reference_file_type: str
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
        max_references: int = 50,
        top_n: int = 10,
    ) -> DatingReport:
        values = np.asarray(values, dtype=np.float64)
        warnings: list[str] = []

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

        candidates = self.index.search(
            species=species_filter,
            states=state_filter,
            min_year=era_start - len(values),
            max_year=era_end,
            min_overlap=min_overlap,
        )

        if not candidates:
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
                    "reference_count": 0,
                    "min_overlap": min_overlap,
                },
            )

        candidates = candidates[:max_references]

        orientation_runs = self._resolve_orientations(values, orientation)
        best_run: Optional[tuple[str, list[DatingCandidate], np.ndarray]] = None

        for assumed_orientation, normalized_values in orientation_runs:
            try:
                detrended, _ = detrend_series(normalized_values, method=detrend_method)
                sample_std = standardize(detrended, method="zscore")
            except Exception as exc:
                warnings.append(f"Detrending failed for {assumed_orientation}: {exc}")
                continue

            ranked_candidates = self._rank_candidates(
                sample_std=sample_std,
                bark_edge=has_bark_edge,
                candidates=candidates,
                era_start=era_start,
                era_end=era_end,
                min_overlap=min_overlap,
                top_n=top_n,
                species_filter=species_filter,
                state_filter=state_filter,
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
                    "reference_count": len(candidates),
                    "min_overlap": min_overlap,
                },
            )

        chosen_orientation, ranked_candidates, sample_std = best_run
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
                "reference_count": len(candidates),
                "min_overlap": int(min_overlap),
                "top_score": float(ranked_candidates[0].composite_score) if ranked_candidates else None,
                "top_correlation": float(ranked_candidates[0].correlation) if ranked_candidates else None,
                "top_t_value": float(ranked_candidates[0].t_value) if ranked_candidates else None,
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

    def _rank_candidates(
        self,
        *,
        sample_std: np.ndarray,
        bark_edge: bool,
        candidates: list[ReferenceManifestEntry],
        era_start: int,
        era_end: int,
        min_overlap: int,
        top_n: int,
        species_filter: Optional[list[str]],
        state_filter: Optional[list[str]],
    ) -> list[DatingCandidate]:
        ranked: list[DatingCandidate] = []

        for entry in candidates:
            reference = entry.master_values
            if len(reference) < min_overlap:
                continue

            best_matches = find_best_match(
                sample_std,
                reference,
                entry.master_start_year,
                min_overlap=min_overlap,
                n_best=1,
            )
            if not best_matches:
                continue

            best = best_matches[0]
            proposed_end = best.position + len(sample_std) - 1
            if proposed_end < era_start or proposed_end > era_end:
                continue

            offset = best.position - entry.master_start_year
            aligned_reference = self._aligned_reference(reference, offset, len(sample_std))
            seg_corrs = (
                segment_correlation(
                    sample_std,
                    aligned_reference,
                    segment_length=min(50, max(20, len(sample_std) // 2)),
                    lag=max(10, min(25, len(sample_std) // 4)),
                )
                if aligned_reference is not None
                else []
            )

            segment_consistency = self._segment_consistency(seg_corrs)
            composite_score = self._score_candidate(
                result=best,
                segment_consistency=segment_consistency,
                entry=entry,
                species_filter=species_filter,
                state_filter=state_filter,
            )
            rationale = self._candidate_rationale(best, segment_consistency, bark_edge, entry)

            ranked.append(
                DatingCandidate(
                    reference_id=entry.site_id,
                    reference_name=entry.site_name,
                    reference_species=entry.species,
                    reference_state=entry.state,
                    reference_file_type=entry.file_type,
                    proposed_start_year=int(best.position),
                    proposed_end_year=int(proposed_end),
                    correlation=float(best.correlation),
                    t_value=float(best.t_value),
                    p_value=float(best.p_value),
                    overlap=int(best.overlap),
                    gleichlauf=float(best.gleichlauf),
                    segment_correlations=seg_corrs,
                    segment_consistency=float(segment_consistency),
                    composite_score=float(composite_score),
                    rationale=rationale,
                )
            )

        ranked.sort(key=lambda candidate: (candidate.composite_score, candidate.t_value, candidate.correlation), reverse=True)
        for index, candidate in enumerate(ranked):
            next_score = ranked[index + 1].composite_score if index + 1 < len(ranked) else 0.0
            candidate.score_gap_to_next = float(candidate.composite_score - next_score)

        if ranked:
            ranked[0].is_recommended = self._is_recommended(ranked[0])

        return ranked[:top_n]

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
        strong = sum(1 for _, _, t_val in segment_results if t_val >= 3.5)
        return strong / len(segment_results)

    def _score_candidate(
        self,
        *,
        result: CorrelationResult,
        segment_consistency: float,
        entry: ReferenceManifestEntry,
        species_filter: Optional[list[str]],
        state_filter: Optional[list[str]],
    ) -> float:
        corr_score = max(0.0, min(result.correlation, 1.0))
        t_score = max(0.0, min(result.t_value / 8.0, 1.0))
        overlap_score = max(0.0, min(result.overlap / 100.0, 1.0))
        glk_score = max(0.0, min(result.gleichlauf / 100.0, 1.0))
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
            corr_score * 0.34
            + t_score * 0.28
            + overlap_score * 0.16
            + glk_score * 0.10
            + segment_consistency * 0.12
        ) * max(0.0, meta_score)
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

    def _is_recommended(self, candidate: DatingCandidate) -> bool:
        return bool(
            candidate.composite_score >= 0.62
            and candidate.correlation >= 0.4
            and candidate.t_value >= 5.5
            and candidate.overlap >= 50
            and candidate.segment_consistency >= 0.5
            and candidate.score_gap_to_next >= 0.04
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
