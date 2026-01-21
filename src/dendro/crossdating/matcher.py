"""
Multi-chronology matching and ranking for cross-dating.

This module orchestrates the cross-dating process:
1. Prepares the sample series (detrending, standardization)
2. Matches against multiple reference chronologies
3. Ranks and combines results
4. Provides confidence assessment

The goal is to find the calendar year assignment for an undated sample
with high confidence, or indicate when confidence is insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .correlator import (
    CorrelationResult,
    find_best_match,
    segment_correlation,
    dating_confidence,
)
from .detrend import detrend_series, standardize, DetrendMethod
from ..reference.chronology_index import ChronologyIndex, ChronologyMetadata
from ..reference.tucson_parser import parse_rwl_file, parse_crn_file, Chronology, RWLFile


@dataclass
class MatchResult:
    """Result of matching a sample against a single reference chronology."""

    reference_name: str
    reference_species: str
    reference_state: str
    proposed_start_year: int
    proposed_end_year: int
    correlation: float
    t_value: float
    p_value: float
    overlap: int
    gleichlauf: float
    confidence: str
    segment_correlations: list[tuple[int, float, float]] = field(default_factory=list)

    @property
    def felling_year(self) -> int:
        """The proposed felling year (last ring + bark edge)."""
        return self.proposed_end_year


@dataclass
class CrossdateReport:
    """Complete cross-dating report for a sample."""

    sample_name: str
    sample_length: int
    has_bark_edge: bool
    detrend_method: str
    matches: list[MatchResult]
    consensus_year: Optional[int] = None
    consensus_confidence: str = "LOW"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert report to dictionary for serialization."""
        return {
            "sample_name": self.sample_name,
            "sample_length": self.sample_length,
            "has_bark_edge": self.has_bark_edge,
            "detrend_method": self.detrend_method,
            "consensus_year": self.consensus_year,
            "consensus_confidence": self.consensus_confidence,
            "warnings": self.warnings,
            "matches": [
                {
                    "reference": m.reference_name,
                    "species": m.reference_species,
                    "state": m.reference_state,
                    "felling_year": m.felling_year,
                    "correlation": round(m.correlation, 4),
                    "t_value": round(m.t_value, 2),
                    "overlap": m.overlap,
                    "confidence": m.confidence,
                }
                for m in self.matches
            ],
        }


class CrossdateMatcher:
    """
    Cross-dating engine that matches samples against reference chronologies.
    """

    def __init__(
        self,
        index: Optional[ChronologyIndex] = None,
        reference_dir: Optional[str | Path] = None,
    ):
        """
        Initialize the matcher.

        Args:
            index: Pre-built chronology index. If None, creates from reference_dir.
            reference_dir: Directory containing reference chronologies.
        """
        if index is not None:
            self.index = index
        elif reference_dir is not None:
            self.index = ChronologyIndex(reference_dir)
        else:
            self.index = ChronologyIndex()

    def date_sample(
        self,
        values: np.ndarray,
        sample_name: str = "sample",
        has_bark_edge: bool = True,
        species_filter: Optional[list[str]] = None,
        state_filter: Optional[list[str]] = None,
        era_start: int = 1600,
        era_end: int = 1900,
        detrend_method: DetrendMethod = DetrendMethod.SPLINE,
        min_overlap: int = 30,
        max_references: int = 20,
    ) -> CrossdateReport:
        """
        Cross-date a sample against available reference chronologies.

        Args:
            values: Ring width measurements (from bark to pith, outer to inner).
            sample_name: Identifier for the sample.
            has_bark_edge: Whether sample includes the outermost ring (bark).
            species_filter: Only match against these species.
            state_filter: Only match against these states.
            era_start: Earliest possible felling year.
            era_end: Latest possible felling year.
            detrend_method: Method for removing growth trend.
            min_overlap: Minimum overlapping years required.
            max_references: Maximum number of references to try.

        Returns:
            CrossdateReport with all match results and assessment.
        """
        values = np.asarray(values, dtype=np.float64)

        # Validate input
        if len(values) < min_overlap:
            report = CrossdateReport(
                sample_name=sample_name,
                sample_length=len(values),
                has_bark_edge=has_bark_edge,
                detrend_method=detrend_method.value,
                matches=[],
            )
            report.warnings.append(
                f"Sample too short ({len(values)} rings). "
                f"Minimum {min_overlap} required for reliable dating."
            )
            return report

        # Detrend and standardize the sample
        try:
            detrended, _ = detrend_series(values, method=detrend_method)
            sample_std = standardize(detrended)
        except Exception as e:
            report = CrossdateReport(
                sample_name=sample_name,
                sample_length=len(values),
                has_bark_edge=has_bark_edge,
                detrend_method=detrend_method.value,
                matches=[],
            )
            report.warnings.append(f"Detrending failed: {e}")
            return report

        # Find candidate references
        # The sample spans era_end - len(values) to era_end approximately
        candidates = self.index.search(
            species=species_filter,
            states=state_filter,
            min_year=era_start - len(values),
            max_year=era_end,
            min_overlap=min_overlap,
        )

        if not candidates:
            report = CrossdateReport(
                sample_name=sample_name,
                sample_length=len(values),
                has_bark_edge=has_bark_edge,
                detrend_method=detrend_method.value,
                matches=[],
            )
            report.warnings.append(
                "No reference chronologies found matching criteria. "
                "Try downloading more references or relaxing filters."
            )
            return report

        # Limit number of references to try
        candidates = candidates[:max_references]

        # Match against each reference
        all_matches: list[MatchResult] = []

        for meta in candidates:
            try:
                match = self._match_against_reference(
                    sample_std, meta, era_start, era_end, min_overlap
                )
                if match is not None:
                    all_matches.append(match)
            except Exception as e:
                continue  # Skip problematic references

        # Sort by t-value
        all_matches.sort(key=lambda m: m.t_value, reverse=True)

        # Build report
        report = CrossdateReport(
            sample_name=sample_name,
            sample_length=len(values),
            has_bark_edge=has_bark_edge,
            detrend_method=detrend_method.value,
            matches=all_matches,
        )

        # Assess consensus
        self._assess_consensus(report)

        return report

    def _match_against_reference(
        self,
        sample: np.ndarray,
        meta: ChronologyMetadata,
        era_start: int,
        era_end: int,
        min_overlap: int,
    ) -> Optional[MatchResult]:
        """Match sample against a single reference chronology."""

        # Load the reference data
        data = self.index.load_chronology(meta)

        if data is None:
            return None

        # Get reference values and years
        if isinstance(data, Chronology):
            ref_values = data.values
            ref_start = data.start_year

            # Detrend/standardize reference (CRN files are often already indexed)
            if np.mean(ref_values) > 500:  # Likely scaled to 1000
                ref_std = ref_values / 1000.0
            else:
                ref_std = standardize(ref_values)
        elif isinstance(data, RWLFile):
            # For RWL files, build a master chronology
            if not data.series:
                return None

            # Get chronology
            df = data.to_dataframe()
            ref_values = df.mean(axis=1).values
            ref_start = int(df.index.min())

            # Detrend/standardize
            try:
                detrended, _ = detrend_series(ref_values)
                ref_std = standardize(detrended)
            except Exception:
                ref_std = standardize(ref_values)
        else:
            return None

        # Find best match
        best_matches = find_best_match(
            sample, ref_std, ref_start, min_overlap, n_best=1
        )

        if not best_matches:
            return None

        best = best_matches[0]

        # Filter by era
        sample_end = best.position + len(sample) - 1
        if best.position < era_start - len(sample) or sample_end > era_end + 50:
            return None

        # Calculate segment correlations for detailed analysis
        # Align sample to best position
        offset = best.position - ref_start
        if offset >= 0 and offset + len(sample) <= len(ref_std):
            seg_corrs = segment_correlation(
                sample,
                ref_std[offset:offset + len(sample)],
                segment_length=min(50, len(sample) // 2),
                lag=25,
            )
        else:
            seg_corrs = []

        return MatchResult(
            reference_name=meta.site_name,
            reference_species=meta.species,
            reference_state=meta.state,
            proposed_start_year=best.position,
            proposed_end_year=best.position + len(sample) - 1,
            correlation=best.correlation,
            t_value=best.t_value,
            p_value=best.p_value,
            overlap=best.overlap,
            gleichlauf=best.gleichlauf,
            confidence=dating_confidence(best),
            segment_correlations=seg_corrs,
        )

    def _assess_consensus(self, report: CrossdateReport):
        """Assess consensus among top matches."""

        if not report.matches:
            report.consensus_confidence = "LOW"
            report.warnings.append("No matches found.")
            return

        # Get top matches with HIGH or MEDIUM confidence
        good_matches = [m for m in report.matches if m.confidence in ("HIGH", "MEDIUM")]

        if not good_matches:
            report.consensus_confidence = "LOW"
            report.warnings.append(
                "No matches achieved MEDIUM or HIGH confidence. "
                "Consider longer sample or different references."
            )
            return

        # Check for consensus on felling year
        felling_years = [m.felling_year for m in good_matches[:5]]  # Top 5
        most_common = max(set(felling_years), key=felling_years.count)
        agreement = felling_years.count(most_common) / len(felling_years)

        report.consensus_year = most_common

        # Assess overall confidence
        best_match = report.matches[0]

        if agreement >= 0.8 and best_match.confidence == "HIGH":
            report.consensus_confidence = "HIGH"
        elif agreement >= 0.6 and best_match.t_value >= 4.0:
            report.consensus_confidence = "MEDIUM"
        else:
            report.consensus_confidence = "LOW"
            if agreement < 0.6:
                report.warnings.append(
                    f"Low agreement among references: only {agreement:.0%} "
                    f"agree on {most_common}."
                )

        # Additional warnings
        if best_match.overlap < 50:
            report.warnings.append(
                f"Limited overlap ({best_match.overlap} years). "
                "Results would be stronger with longer sample."
            )

        if report.sample_length < 50:
            report.warnings.append(
                f"Short sample ({report.sample_length} rings). "
                "50+ rings recommended for reliable dating."
            )

        # Stricter t-value warnings
        if 4.0 <= best_match.t_value < 5.0:
            report.warnings.append(
                f"T-value ({best_match.t_value:.1f}) is in the borderline range (4-5). "
                "This range commonly produces spurious matches in dendro dating."
            )

        if best_match.t_value < 6.0:
            report.warnings.append(
                "T-value below 6.0. Professional publication standards "
                "typically require t≥6 for secure dating."
            )

        # Check segment consistency
        if best_match.segment_correlations:
            weak_segs = [s for s in best_match.segment_correlations if s[2] < 3.5]
            if weak_segs:
                report.warnings.append(
                    f"{len(weak_segs)} segment(s) show weak correlation (t<3.5). "
                    "Check measurements in these regions."
                )

        # Geographic diversity warning
        states = set(m.reference_state for m in good_matches[:5])
        if len(states) == 1:
            report.warnings.append(
                f"All top matches from single state ({list(states)[0]}). "
                "Geographic diversity would strengthen confidence."
            )


def date_measurements(
    measurements: np.ndarray | pd.DataFrame,
    reference_dir: str | Path,
    sample_name: str = "sample",
    has_bark_edge: bool = True,
    species: Optional[list[str]] = None,
    states: Optional[list[str]] = None,
    era_start: int = 1600,
    era_end: int = 1900,
) -> CrossdateReport:
    """
    Convenience function to date ring measurements against references.

    Args:
        measurements: Ring width values (array or DataFrame with 'width' column).
        reference_dir: Directory containing downloaded ITRDB chronologies.
        sample_name: Identifier for the sample.
        has_bark_edge: Whether sample includes bark/outermost ring.
        species: Species codes to match against.
        states: State codes to match against.
        era_start: Earliest expected felling year.
        era_end: Latest expected felling year.

    Returns:
        CrossdateReport with match results.
    """
    # Extract values from DataFrame if needed
    if isinstance(measurements, pd.DataFrame):
        if "width" in measurements.columns:
            values = measurements["width"].values
        else:
            values = measurements.iloc[:, -1].values
    else:
        values = np.asarray(measurements)

    # Create matcher and run
    matcher = CrossdateMatcher(reference_dir=reference_dir)

    return matcher.date_sample(
        values=values,
        sample_name=sample_name,
        has_bark_edge=has_bark_edge,
        species_filter=species,
        state_filter=states,
        era_start=era_start,
        era_end=era_end,
    )
