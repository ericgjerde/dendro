"""
Sliding correlation algorithms for cross-dating.

Cross-dating works by sliding a sample series along a reference chronology
and calculating correlation at each position. The position with highest
correlation indicates the calendar year assignment.

Key metrics:
- Pearson correlation coefficient (r)
- Student's t-value (measures statistical significance)
- Gleichläufigkeit (% of agreeing year-to-year signs)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class CorrelationResult:
    """Result of correlation at a single position."""

    position: int  # Year offset (sample start year if matched here)
    correlation: float  # Pearson r
    t_value: float  # Student's t
    p_value: float  # Statistical significance
    overlap: int  # Number of overlapping years
    gleichlauf: float  # Gleichläufigkeit (sign agreement)


def sliding_correlation(
    sample: np.ndarray,
    reference: np.ndarray,
    reference_start_year: int,
    min_overlap: int = 30,
) -> list[CorrelationResult]:
    """
    Calculate correlation between sample and reference at all valid positions.

    The sample is slid along the reference chronology. At each position,
    correlation and significance statistics are calculated.

    Args:
        sample: The undated sample series (ring width indices).
        reference: The dated reference chronology.
        reference_start_year: Calendar year of first reference value.
        min_overlap: Minimum years of overlap required.

    Returns:
        List of CorrelationResult for each valid position.
    """
    sample = np.asarray(sample, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    n_sample = len(sample)
    n_reference = len(reference)

    if n_sample < min_overlap or n_reference < min_overlap:
        return []

    results = []

    # Slide sample along reference
    # Position is the reference index where sample[0] would align
    for pos in range(-n_sample + min_overlap, n_reference - min_overlap + 1):
        # Calculate overlap indices
        sample_start = max(0, -pos)
        sample_end = min(n_sample, n_reference - pos)
        ref_start = max(0, pos)
        ref_end = min(n_reference, pos + n_sample)

        # Extract overlapping segments
        sample_segment = sample[sample_start:sample_end]
        ref_segment = reference[ref_start:ref_end]

        # Handle NaN values
        valid_mask = ~(np.isnan(sample_segment) | np.isnan(ref_segment))
        n_valid = np.sum(valid_mask)

        if n_valid < min_overlap:
            continue

        sample_valid = sample_segment[valid_mask]
        ref_valid = ref_segment[valid_mask]

        # Calculate correlation
        corr, p_value = stats.pearsonr(sample_valid, ref_valid)

        # Calculate t-value
        t_val = calculate_tvalue(corr, n_valid)

        # Calculate Gleichläufigkeit
        glk = calculate_gleichlauf(sample_valid, ref_valid)

        # The sample's first ring would be at this calendar year
        sample_start_year = reference_start_year + pos

        results.append(CorrelationResult(
            position=sample_start_year,
            correlation=corr,
            t_value=t_val,
            p_value=p_value,
            overlap=n_valid,
            gleichlauf=glk,
        ))

    return results


def calculate_tvalue(r: float, n: int) -> float:
    """
    Calculate Student's t-value for a correlation coefficient.

    t = r * sqrt((n-2) / (1-r^2))

    Higher t-values indicate more statistically significant correlations.
    Rule of thumb: t > 3.5 is often considered significant in dendro.

    Args:
        r: Pearson correlation coefficient.
        n: Number of observations (overlapping years).

    Returns:
        Student's t-value.
    """
    if abs(r) >= 1.0:
        return float('inf') if r > 0 else float('-inf')

    if n <= 2:
        return 0.0

    t = r * np.sqrt((n - 2) / (1 - r**2))
    return t


def calculate_gleichlauf(series1: np.ndarray, series2: np.ndarray) -> float:
    """
    Calculate Gleichläufigkeit (coefficient of parallel variation).

    This is the percentage of years where both series change in the
    same direction (both increase or both decrease). It's a non-parametric
    measure that's robust to outliers.

    Formula: GLK = 100 * (# agreements + 0.5 * # zeros) / (n - 1)

    Args:
        series1: First series.
        series2: Second series.

    Returns:
        Gleichläufigkeit as percentage (0-100).
    """
    if len(series1) != len(series2) or len(series1) < 2:
        return 0.0

    # Year-to-year interval signs (Eckstein & Bauch 1969).
    sign1 = np.sign(np.diff(series1))
    sign2 = np.sign(np.diff(series2))

    n = len(sign1)

    # Full agreement when both intervals share a sign (including both flat);
    # half credit when exactly one interval is flat; no credit on opposite signs.
    same = sign1 == sign2
    one_zero = (~same) & ((sign1 == 0) | (sign2 == 0))
    score = np.sum(same) + 0.5 * np.sum(one_zero)

    return 100.0 * score / n


def gleichlauf_significance(glk_percent: float, n: int) -> float:
    """
    Approximate two-sided significance (p-value) of a Gleichläufigkeit value.

    Under the null hypothesis of independent series, GLK is ~50% with standard
    error ``1 / (2 * sqrt(n-1))`` (Eckstein & Bauch 1969). This returns the
    probability of observing a deviation from 50% at least this large by chance.

    Args:
        glk_percent: Gleichläufigkeit as a percentage (0-100).
        n: Number of overlapping years (intervals used = n - 1).

    Returns:
        Two-sided p-value (lower = more significant).
    """
    if n < 3:
        return 1.0
    se = 1.0 / (2.0 * np.sqrt(n - 1))
    z = abs(glk_percent / 100.0 - 0.5) / se
    return float(2.0 * (1.0 - stats.norm.cdf(z)))


def cross_correlation(
    sample: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Calculate normalized cross-correlation using FFT.

    This is faster than explicit sliding for long series.

    Args:
        sample: Sample series.
        reference: Reference series.

    Returns:
        Array of correlation coefficients at each lag.
    """
    # Normalize both series
    sample = (sample - np.nanmean(sample)) / np.nanstd(sample)
    reference = (reference - np.nanmean(reference)) / np.nanstd(reference)

    # Replace NaN with 0 for FFT
    sample = np.nan_to_num(sample)
    reference = np.nan_to_num(reference)

    # Cross-correlation via FFT
    n = len(reference)
    xcorr = np.correlate(reference, sample, mode='full')

    # Normalize by overlap count
    overlap = np.correlate(
        np.ones(n),
        np.ones(len(sample)),
        mode='full'
    )

    xcorr = xcorr / overlap

    return xcorr


def segment_correlation(
    sample: np.ndarray,
    reference: np.ndarray,
    segment_length: int = 50,
    lag: int = 25,
) -> list[tuple[int, float, float]]:
    """
    Calculate correlation in overlapping segments (COFECHA-style).

    This helps identify segments where correlation is weak, which
    may indicate dating errors or problem areas.

    Args:
        sample: Sample series aligned to reference.
        reference: Reference chronology.
        segment_length: Length of each segment in years.
        lag: Offset between segment starts.

    Returns:
        List of (start_index, correlation, t_value) tuples.
    """
    n = min(len(sample), len(reference))

    if n < segment_length:
        return []

    results = []

    for start in range(0, n - segment_length + 1, lag):
        end = start + segment_length

        seg_sample = sample[start:end]
        seg_ref = reference[start:end]

        # Handle NaN
        valid = ~(np.isnan(seg_sample) | np.isnan(seg_ref))
        n_valid = np.sum(valid)

        if n_valid < 20:
            continue

        r, _ = stats.pearsonr(seg_sample[valid], seg_ref[valid])
        t = calculate_tvalue(r, n_valid)

        results.append((start, r, t))

    return results


def find_best_match(
    sample: np.ndarray,
    reference: np.ndarray,
    reference_start_year: int,
    min_overlap: int = 30,
    n_best: int = 5,
) -> list[CorrelationResult]:
    """
    Find the best matching positions for a sample.

    Args:
        sample: Undated sample series.
        reference: Dated reference chronology.
        reference_start_year: First year of reference.
        min_overlap: Minimum overlap required.
        n_best: Number of top matches to return.

    Returns:
        List of top N correlation results, sorted by t-value descending.
    """
    results = sliding_correlation(
        sample, reference, reference_start_year, min_overlap
    )

    # Sort by t-value (highest first)
    results.sort(key=lambda x: x.t_value, reverse=True)

    return results[:n_best]


@dataclass
class MissingRingHint:
    """A heuristic suggestion that a sample has a missing or false ring."""

    index: int  # sample index where the discontinuity is suggested
    kind: str  # "missing" (sample omits a ring) or "false" (sample has an extra)
    baseline_correlation: float
    improved_correlation: float

    @property
    def gain(self) -> float:
        return self.improved_correlation - self.baseline_correlation


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3 or len(a) != len(b):
        return 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def detect_missing_ring(
    sample: np.ndarray,
    reference: np.ndarray,
    min_segment: int = 15,
    min_gain: float = 0.1,
) -> Optional[MissingRingHint]:
    """
    Heuristically locate a single missing or false ring causing a dating misfit.

    A locally absent ring (a year with no measurable growth) or a false/double
    ring shifts every later ring by one, so a sample can correlate only weakly
    even at its true position. This scans candidate break points: at each one it
    tests whether re-aligning the later part of the sample by +/-1 ring against
    the reference substantially improves correlation over the rigid alignment.

    The reference must be at least one ring longer than the sample and aligned so
    that ``sample[0]`` corresponds to ``reference[0]``.

    Args:
        sample: Standardized sample series (oldest -> newest).
        reference: Standardized reference series aligned to the sample start;
            should extend at least ``len(sample) + 1``.
        min_segment: Minimum rings on each side of a candidate break.
        min_gain: Minimum correlation improvement to report a hint.

    Returns:
        A ``MissingRingHint`` for the best-supported location, or ``None``.
    """
    sample = np.asarray(sample, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    n = len(sample)
    if n < 2 * min_segment + 1 or len(reference) < n + 1:
        return None

    baseline = _safe_corr(sample, reference[:n])
    best: Optional[MissingRingHint] = None

    for k in range(min_segment, n - min_segment):
        head = _safe_corr(sample[:k], reference[:k])

        # "missing": sample omits a ring at k, so its tail lines up one ring
        # later in the reference.
        tail_missing = _safe_corr(sample[k:], reference[k + 1 : k + 1 + (n - k)])
        # "false": sample has an extra ring at k, so its tail lines up one ring
        # earlier in the reference.
        tail_false = _safe_corr(sample[k:], reference[k - 1 : k - 1 + (n - k)])

        for kind, tail in (("missing", tail_missing), ("false", tail_false)):
            combined = 0.5 * (head + tail)
            if combined - baseline >= min_gain and (best is None or combined > best.improved_correlation):
                best = MissingRingHint(
                    index=k,
                    kind=kind,
                    baseline_correlation=baseline,
                    improved_correlation=combined,
                )

    return best


def dating_confidence(result: CorrelationResult) -> str:
    """
    Assess confidence level of a dating result.

    Based on standard dendrochronological thresholds.

    Args:
        result: CorrelationResult to assess.

    Returns:
        Confidence level: "HIGH", "MEDIUM", or "LOW".
    """
    # High confidence: strong correlation, good overlap, and parallel variation.
    if (
        result.t_value >= 6.0
        and result.correlation >= 0.55
        and result.overlap >= 50
        and result.gleichlauf >= 62.0
    ):
        return "HIGH"

    # Medium confidence: decent statistics.
    if (
        result.t_value >= 4.0
        and result.correlation >= 0.45
        and result.overlap >= 30
        and result.gleichlauf >= 58.0
    ):
        return "MEDIUM"

    # Low confidence
    return "LOW"
