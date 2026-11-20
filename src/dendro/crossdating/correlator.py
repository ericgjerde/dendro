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

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal, stats


@dataclass
class CorrelationResult:
    """Result of correlation at a single position."""

    position: int  # Year offset (sample start year if matched here)
    correlation: float  # Pearson r
    t_value: float  # Student's t
    p_value: float  # Statistical significance
    overlap: int  # Number of overlapping years
    gleichlauf: float  # Gleichläufigkeit (sign agreement)


def _compute_sliding_stats(
    sample: np.ndarray,
    reference: np.ndarray,
    reference_start_year: int,
    min_overlap: int = 30,
    min_position: Optional[int] = None,
    max_position: Optional[int] = None,
    compute_glk: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized computation of sliding Pearson r, t-value, p-value, overlap, and GLK.

    Returns arrays ``(positions, correlations, t_values, p_values, overlaps, glks)``
    filtered to valid lags with ``overlap >= min_overlap`` and non-constant inputs.
    """
    sample = np.asarray(sample, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    n_sample = len(sample)
    n_reference = len(reference)
    empty = (
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
    )

    if n_sample < min_overlap or n_reference < min_overlap:
        return empty

    lag_min = -n_sample + min_overlap
    lag_max = n_reference - min_overlap
    if min_position is not None:
        lag_min = max(lag_min, int(min_position) - int(reference_start_year))
    if max_position is not None:
        lag_max = min(lag_max, int(max_position) - int(reference_start_year))

    if lag_min > lag_max:
        return empty

    lags = np.arange(lag_min, lag_max + 1, dtype=np.int64)
    full_idx = lags + (n_sample - 1)

    sample_has_nan = bool(np.isnan(sample).any())
    ref_has_nan = bool(np.isnan(reference).any())

    if not sample_has_nan and not ref_has_nan:
        # Center globally for numerical stability (invariant for Pearson r).
        s_centered = sample - float(np.mean(sample))
        r_centered = reference - float(np.mean(reference))

        s_start = np.maximum(0, -lags)
        s_end = np.minimum(n_sample, n_reference - lags)
        r_start = np.maximum(0, lags)
        r_end = np.minimum(n_reference, lags + n_sample)

        overlaps = (s_end - s_start).astype(np.int64)

        cs_s = np.empty(n_sample + 1, dtype=np.float64)
        cs_s[0] = 0.0
        np.cumsum(s_centered, out=cs_s[1:])

        cs_s2 = np.empty(n_sample + 1, dtype=np.float64)
        cs_s2[0] = 0.0
        np.cumsum(s_centered * s_centered, out=cs_s2[1:])

        cs_r = np.empty(n_reference + 1, dtype=np.float64)
        cs_r[0] = 0.0
        np.cumsum(r_centered, out=cs_r[1:])

        cs_r2 = np.empty(n_reference + 1, dtype=np.float64)
        cs_r2[0] = 0.0
        np.cumsum(r_centered * r_centered, out=cs_r2[1:])

        sum_x = cs_s[s_end] - cs_s[s_start]
        sum_x2 = cs_s2[s_end] - cs_s2[s_start]
        sum_y = cs_r[r_end] - cs_r[r_start]
        sum_y2 = cs_r2[r_end] - cs_r2[r_start]

        sum_xy_full = signal.correlate(r_centered, s_centered, mode="full", method="auto")
        sum_xy = sum_xy_full[full_idx]
    else:
        ms = (~np.isnan(sample)).astype(np.float64)
        mr = (~np.isnan(reference)).astype(np.float64)
        if np.sum(ms) < min_overlap or np.sum(mr) < min_overlap:
            return empty

        s_mean = float(np.nanmean(sample))
        r_mean = float(np.nanmean(reference))
        s0 = np.where(np.isnan(sample), 0.0, sample - s_mean)
        r0 = np.where(np.isnan(reference), 0.0, reference - r_mean)

        overlaps = np.rint(
            signal.correlate(mr, ms, mode="full", method="auto")[full_idx]
        ).astype(np.int64)
        sum_x = signal.correlate(mr, s0, mode="full", method="auto")[full_idx]
        sum_x2 = signal.correlate(mr, s0 * s0, mode="full", method="auto")[full_idx]
        sum_y = signal.correlate(r0, ms, mode="full", method="auto")[full_idx]
        sum_y2 = signal.correlate(r0 * r0, ms, mode="full", method="auto")[full_idx]
        sum_xy = signal.correlate(r0, s0, mode="full", method="auto")[full_idx]

    n_float = overlaps.astype(np.float64)
    # Guard division by zero for overlaps == 0 before masking.
    safe_n = np.where(overlaps > 0, n_float, 1.0)
    var_x = sum_x2 - (sum_x * sum_x) / safe_n
    var_y = sum_y2 - (sum_y * sum_y) / safe_n
    cov_xy = sum_xy - (sum_x * sum_y) / safe_n

    # Scale-aware variance tolerance so truly constant segments are excluded cleanly.
    var_tol = 1e-18
    valid = (overlaps >= min_overlap) & (var_x > var_tol) & (var_y > var_tol)
    if not np.any(valid):
        return empty

    lags_v = lags[valid]
    overlaps_v = overlaps[valid]
    n_v = overlaps_v.astype(np.float64)
    denom = np.sqrt(var_x[valid] * var_y[valid])
    corrs = np.clip(cov_xy[valid] / denom, -1.0, 1.0)

    # Vectorized Student's t-value and two-sided p-value.
    one_minus_r2 = 1.0 - corrs * corrs
    t_vals = np.empty_like(corrs)
    perfect_pos = corrs >= 1.0 - 1e-15
    perfect_neg = corrs <= -1.0 + 1e-15
    normal = ~(perfect_pos | perfect_neg)
    t_vals[perfect_pos] = np.inf
    t_vals[perfect_neg] = -np.inf
    t_vals[normal] = corrs[normal] * np.sqrt((n_v[normal] - 2.0) / one_minus_r2[normal])

    p_vals = np.zeros_like(corrs)
    if np.any(normal):
        p_vals[normal] = 2.0 * stats.t.sf(np.abs(t_vals[normal]), df=n_v[normal] - 2.0)

    if compute_glk:
        if not sample_has_nan and not ref_has_nan:
            d1 = np.sign(np.diff(sample))
            d2 = np.sign(np.diff(reference))
            diff_corr_full = signal.correlate(d2, d1, mode="full", method="auto")
            diff_idx = lags_v + (n_sample - 2)
            m = np.maximum(n_v - 1.0, 1.0)
            glks = 50.0 * (1.0 + diff_corr_full[diff_idx] / m)
        else:
            glks = np.empty_like(corrs)
            for idx, lag in enumerate(lags_v):
                lag_int = int(lag)
                s0_i = max(0, -lag_int)
                s1_i = min(n_sample, n_reference - lag_int)
                r0_i = max(0, lag_int)
                r1_i = min(n_reference, lag_int + n_sample)
                s_seg = sample[s0_i:s1_i]
                r_seg = reference[r0_i:r1_i]
                vm = ~(np.isnan(s_seg) | np.isnan(r_seg))
                glks[idx] = calculate_gleichlauf(s_seg[vm], r_seg[vm])
    else:
        glks = np.zeros_like(corrs)

    positions = lags_v + int(reference_start_year)
    return positions, corrs, t_vals, p_vals, overlaps_v, glks


def sliding_correlation(
    sample: np.ndarray,
    reference: np.ndarray,
    reference_start_year: int,
    min_overlap: int = 30,
    min_position: Optional[int] = None,
    max_position: Optional[int] = None,
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
        min_position: Optional minimum sample start year to evaluate.
        max_position: Optional maximum sample start year to evaluate.

    Returns:
        List of CorrelationResult for each valid position.
    """
    positions, corrs, t_vals, p_vals, overlaps, glks = _compute_sliding_stats(
        sample=sample,
        reference=reference,
        reference_start_year=reference_start_year,
        min_overlap=min_overlap,
        min_position=min_position,
        max_position=max_position,
        compute_glk=True,
    )
    if len(positions) == 0:
        return []

    return [
        CorrelationResult(
            position=int(positions[i]),
            correlation=float(corrs[i]),
            t_value=float(t_vals[i]),
            p_value=float(p_vals[i]),
            overlap=int(overlaps[i]),
            gleichlauf=float(glks[i]),
        )
        for i in range(len(positions))
    ]


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
        return float("inf") if r > 0 else float("-inf")

    if n <= 2:
        return 0.0

    return float(r * np.sqrt((n - 2) / (1 - r**2)))


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

    diff1 = np.diff(series1)
    diff2 = np.diff(series2)

    sign1 = np.sign(diff1)
    sign2 = np.sign(diff2)

    n = len(diff1)
    return float(50.0 * (1.0 + np.mean(sign1 * sign2)))


def cross_correlation(
    sample: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Calculate normalized cross-correlation using FFT.

    Args:
        sample: Sample series.
        reference: Reference series.

    Returns:
        Array of correlation coefficients at each lag.
    """
    sample = np.asarray(sample, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    s_std = float(np.nanstd(sample))
    r_std = float(np.nanstd(reference))
    if s_std == 0.0 or r_std == 0.0:
        return np.zeros(len(reference) + len(sample) - 1, dtype=np.float64)

    s_norm = np.nan_to_num((sample - float(np.nanmean(sample))) / s_std)
    r_norm = np.nan_to_num((reference - float(np.nanmean(reference))) / r_std)

    xcorr = signal.correlate(r_norm, s_norm, mode="full", method="auto")
    overlap = signal.correlate(
        np.ones(len(reference), dtype=np.float64),
        np.ones(len(sample), dtype=np.float64),
        mode="full",
        method="auto",
    )
    return xcorr / np.maximum(overlap, 1.0)


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
    sample = np.asarray(sample, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    n = min(len(sample), len(reference))

    if n < segment_length or segment_length < 3:
        return []

    results: list[tuple[int, float, float]] = []

    for start in range(0, n - segment_length + 1, lag):
        end = start + segment_length
        seg_sample = sample[start:end]
        seg_ref = reference[start:end]

        valid = ~(np.isnan(seg_sample) | np.isnan(seg_ref))
        n_valid = int(np.sum(valid))
        if n_valid < 20:
            continue

        s_v = seg_sample[valid]
        r_v = seg_ref[valid]
        s_c = s_v - np.mean(s_v)
        r_c = r_v - np.mean(r_v)
        denom = float(np.sqrt(np.dot(s_c, s_c) * np.dot(r_c, r_c)))
        if denom <= 1e-18:
            continue

        r = float(np.clip(np.dot(s_c, r_c) / denom, -1.0, 1.0))
        t = calculate_tvalue(r, n_valid)
        results.append((start, r, t))

    return results


def find_best_match(
    sample: np.ndarray,
    reference: np.ndarray,
    reference_start_year: int,
    min_overlap: int = 30,
    n_best: int = 5,
    min_position: Optional[int] = None,
    max_position: Optional[int] = None,
    min_peak_separation: int = 0,
) -> list[CorrelationResult]:
    """
    Find the best matching positions for a sample.

    Args:
        sample: Undated sample series.
        reference: Dated reference chronology.
        reference_start_year: First year of reference.
        min_overlap: Minimum overlap required.
        n_best: Number of top matches to return.
        min_position: Optional minimum sample start year.
        max_position: Optional maximum sample start year.
        min_peak_separation: Optional minimum year gap between returned peaks.

    Returns:
        List of top N correlation results, sorted by t-value descending.
    """
    positions, corrs, t_vals, p_vals, overlaps, glks = _compute_sliding_stats(
        sample=sample,
        reference=reference,
        reference_start_year=reference_start_year,
        min_overlap=min_overlap,
        min_position=min_position,
        max_position=max_position,
        compute_glk=True,
    )
    if len(positions) == 0 or n_best <= 0:
        return []

    order = np.argsort(t_vals)[::-1]
    selected: list[CorrelationResult] = []
    for idx in order:
        pos = int(positions[idx])
        if min_peak_separation > 0 and any(abs(pos - prev.position) < min_peak_separation for prev in selected):
            continue
        selected.append(
            CorrelationResult(
                position=pos,
                correlation=float(corrs[idx]),
                t_value=float(t_vals[idx]),
                p_value=float(p_vals[idx]),
                overlap=int(overlaps[idx]),
                gleichlauf=float(glks[idx]),
            )
        )
        if len(selected) >= n_best:
            break

    return selected


def dating_confidence(result: CorrelationResult) -> str:
    """
    Assess confidence level of a dating result.

    Based on standard dendrochronological thresholds.

    Args:
        result: CorrelationResult to assess.

    Returns:
        Confidence level: "HIGH", "MEDIUM", or "LOW".
    """
    if result.t_value >= 6.0 and result.correlation >= 0.55 and result.overlap >= 50:
        return "HIGH"

    if result.t_value >= 4.0 and result.correlation >= 0.45 and result.overlap >= 30:
        return "MEDIUM"

    return "LOW"
