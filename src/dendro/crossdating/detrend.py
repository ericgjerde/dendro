"""
Detrending and standardization algorithms for ring-width series.

Tree ring widths exhibit age-related growth trends (young trees typically grow
faster). Detrending removes this trend to reveal climate signal, making series
from different trees comparable.

Detrending methods:
1. Negative exponential curve
2. Cook & Peters (1981) cubic smoothing spline (exact 50% frequency response)
3. Modified negative exponential (Hugershoff)
4. Linear / Mean / None

After detrending, series are typically standardized to zero mean, unit variance
or ratio indices, and optionally prewhitened with an AR model.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from scipy.linalg import solveh_banded
from scipy.optimize import curve_fit


class DetrendMethod(Enum):
    """Available detrending methods."""

    NONE = "none"
    MEAN = "mean"
    LINEAR = "linear"
    NEGATIVE_EXPONENTIAL = "negexp"
    SPLINE = "spline"
    MODIFIED_NEGEXP = "modnegexp"


def detrend_series(
    values: np.ndarray,
    method: DetrendMethod = DetrendMethod.SPLINE,
    spline_period: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detrend a ring-width series to remove age-related growth trend.

    Args:
        values: Raw ring width values.
        method: Detrending method to use.
        spline_period: For spline method, the wavelength (years) at which the
            spline has a 50% amplitude (frequency) response. If None, uses
            32 years, or 2/3 of the series length when that is shorter.

    Returns:
        Tuple of (detrended_values, fitted_curve).
        Detrended values are ratios (values / fitted_curve) except for NONE.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 10:
        raise ValueError("Series too short for detrending (minimum 10 values)")

    x = np.arange(len(values), dtype=np.float64)

    valid_mask = ~np.isnan(values) & (values > 0)
    if int(np.sum(valid_mask)) < 10:
        raise ValueError("Not enough valid values for detrending")

    if method == DetrendMethod.NONE:
        return values.copy(), np.ones_like(values)

    if method == DetrendMethod.MEAN:
        mean_val = float(np.nanmean(values[valid_mask]))
        curve = np.full_like(values, max(mean_val, 0.01))
        return values / curve, curve

    if method == DetrendMethod.LINEAR:
        coeffs = np.polyfit(x[valid_mask], values[valid_mask], 1)
        curve = np.maximum(np.polyval(coeffs, x), 0.01)
        return values / curve, curve

    if method == DetrendMethod.NEGATIVE_EXPONENTIAL:
        curve = _fit_negative_exponential(x, values, valid_mask)
        return values / curve, curve

    if method == DetrendMethod.MODIFIED_NEGEXP:
        curve = _fit_modified_negexp(x, values, valid_mask)
        return values / curve, curve

    if method == DetrendMethod.SPLINE:
        if spline_period is None:
            # COFECHA cross-dating default: 32-year 50% frequency-response spline
            # (capped at 2/3 of series length for short series).
            spline_period = min(32.0, len(values) * 2.0 / 3.0)
        curve = _fit_cubic_spline(x, values, valid_mask, spline_period)
        return values / curve, curve

    raise ValueError(f"Unknown detrend method: {method}")


def _fit_negative_exponential(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Fit a negative exponential curve: y = a * exp(-b * x) + c.

    Falls back to linear if exponential fit fails.
    """

    def neg_exp(t, a, b, c):
        return a * np.exp(-b * t) + c

    try:
        y_valid = y[mask]
        a0 = max(float(y_valid[0] - y_valid[-1]), 0.01)
        c0 = max(float(y_valid[-1]), 0.01)
        b0 = 0.01

        popt, _ = curve_fit(
            neg_exp,
            x[mask],
            y_valid,
            p0=[a0, b0, c0],
            bounds=([0, 0, 0], [np.inf, 1, np.inf]),
            maxfev=5000,
        )

        curve = neg_exp(x, *popt)
        return np.maximum(curve, 0.01)

    except (RuntimeError, ValueError):
        coeffs = np.polyfit(x[mask], y[mask], 1)
        return np.maximum(np.polyval(coeffs, x), 0.01)


def _fit_modified_negexp(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Fit modified negative exponential (Hugershoff growth curve):

    y = a * t^b * exp(-c * t) + d
    """

    def hugershoff(t, a, b, c, d):
        t_safe = np.maximum(t, 0.1)
        return a * np.power(t_safe, b) * np.exp(-c * t_safe) + d

    try:
        y_valid = y[mask]
        x_valid = x[mask] + 1.0

        popt, _ = curve_fit(
            hugershoff,
            x_valid,
            y_valid,
            p0=[float(np.max(y_valid)), 0.5, 0.01, max(float(np.min(y_valid)), 0.0)],
            bounds=([0, 0, 0, 0], [np.inf, 5, 1, np.inf]),
            maxfev=5000,
        )

        curve = hugershoff(x + 1.0, *popt)
        return np.maximum(curve, 0.01)

    except (RuntimeError, ValueError):
        return _fit_negative_exponential(x, y, mask)


def cook_peters_spline(
    values: np.ndarray,
    period: float,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Cook & Peters (1981) cubic smoothing spline (Reinsch formulation).

    Exact 50% frequency response at wavelength ``period`` (in years).
    Solved in O(N) via symmetric positive-definite pentadiagonal Cholesky solve.

    Args:
        values: 1D array of observations (may contain NaNs if masked).
        period: Wavelength (years) at 50% frequency response. Must be > 2.
        valid_mask: Optional boolean mask of valid observations.

    Returns:
        Fitted smooth curve of the same length as ``values``.
    """
    y = np.asarray(values, dtype=np.float64)
    n = len(y)
    if n < 4:
        return np.full(n, float(np.nanmean(y)) if n > 0 else 1.0, dtype=np.float64)

    if valid_mask is None:
        valid_mask = ~np.isnan(y)

    if not np.all(valid_mask):
        idx = np.arange(n, dtype=np.float64)
        y_filled = np.interp(idx, idx[valid_mask], y[valid_mask])
    else:
        y_filled = y

    period = max(float(period), 2.5)
    cos_w = np.cos(2.0 * np.pi / period)
    denom = 12.0 * (1.0 - cos_w) ** 2
    if denom <= 1e-18:
        coeffs = np.polyfit(np.arange(n, dtype=np.float64), y_filled, 1)
        return np.polyval(coeffs, np.arange(n, dtype=np.float64))

    lam = (2.0 + cos_w) / denom

    # Reinsch system: (T + lam * Q^T Q) c = Q^T y, g = y - lam * Q c
    # where Q has columns [1, -2, 1] and T has diag 2/3, off-diag 1/6.
    m = n - 2
    rhs = y_filled[:-2] - 2.0 * y_filled[1:-1] + y_filled[2:]

    # Lower banded storage for solveh_banded (3 x m):
    # row 0 = main diagonal, row 1 = 1st subdiagonal, row 2 = 2nd subdiagonal.
    ab = np.zeros((3, m), dtype=np.float64)
    ab[0, :] = (2.0 / 3.0) + 6.0 * lam
    if m > 1:
        ab[1, :-1] = (1.0 / 6.0) - 4.0 * lam
    if m > 2:
        ab[2, :-2] = lam

    c = solveh_banded(ab, rhs, lower=True, overwrite_ab=True, overwrite_b=True, check_finite=False)

    # Apply Q c to recover smooth curve g = y - lam * Q c.
    qc = np.zeros(n, dtype=np.float64)
    qc[:-2] += c
    qc[1:-1] -= 2.0 * c
    qc[2:] += c

    return y_filled - lam * qc


def _fit_cubic_spline(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    period: float,
) -> np.ndarray:
    """
    Fit a Cook & Peters (1981) cubic smoothing spline clamped strictly positive.
    """
    try:
        curve = cook_peters_spline(y, period=period, valid_mask=mask)
        return np.maximum(curve, 0.01)
    except Exception:
        coeffs = np.polyfit(x[mask], y[mask], 1)
        return np.maximum(np.polyval(coeffs, x), 0.01)


def standardize(
    values: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    """
    Standardize a detrended series.

    Args:
        values: Detrended ring-width indices.
        method: Standardization method:
            - "zscore": Transform to zero mean, unit variance
            - "ratio": Values already as ratios (center on 1.0)

    Returns:
        Standardized series.
    """
    values = np.asarray(values, dtype=np.float64)

    if method == "zscore":
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        if std > 0:
            return (values - mean) / std
        return values - mean

    if method == "ratio":
        mean = float(np.nanmean(values))
        if mean != 0.0 and not np.isnan(mean):
            return values / mean
        return values.copy()

    raise ValueError(f"Unknown standardization method: {method}")


def prewhiten(values: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Remove autocorrelation using a mean-centered Yule-Walker AR(p) model.

    Prewhitening removes biological persistence (red noise) from a series,
    restoring effective degrees of freedom for cross-dating statistics.

    Args:
        values: Ring-width indices (may contain NaNs).
        order: Order of autoregressive model (default 1).

    Returns:
        Mean-centered residuals from the AR(p) model fit (preserving NaN positions).
    """
    values = np.asarray(values, dtype=np.float64)
    if order < 1:
        return values.copy()

    valid_mask = ~np.isnan(values)
    n_valid = int(np.sum(valid_mask))
    if n_valid < order + 10:
        return values.copy()

    mean_val = float(np.nanmean(values))
    centered = values - mean_val

    # Estimate autocovariance over contiguous valid pairs at lags 0..order.
    r = np.zeros(order + 1, dtype=np.float64)
    valid_centered = centered[valid_mask]
    c0 = float(np.dot(valid_centered, valid_centered)) / n_valid
    if c0 <= 1e-18:
        return np.where(valid_mask, 0.0, np.nan)

    r[0] = 1.0
    for lag in range(1, order + 1):
        pair_mask = valid_mask[:-lag] & valid_mask[lag:]
        n_pairs = int(np.sum(pair_mask))
        if n_pairs < 5:
            return centered.copy()
        cov = float(np.dot(centered[:-lag][pair_mask], centered[lag:][pair_mask])) / n_pairs
        r[lag] = np.clip(cov / c0, -0.99, 0.99)

    # Solve Yule-Walker Toeplitz equations R @ phi = r[1:].
    R = np.empty((order, order), dtype=np.float64)
    for i in range(order):
        for j in range(order):
            R[i, j] = r[abs(i - j)]

    try:
        ar_coeffs = np.linalg.solve(R, r[1:order + 1])
    except np.linalg.LinAlgError:
        return centered.copy()

    # Apply AR filter to centered values, handling NaNs without propagating them.
    residuals = centered.copy()
    filled = np.where(valid_mask, centered, 0.0)
    for i in range(order, len(values)):
        if not valid_mask[i]:
            residuals[i] = np.nan
            continue
        predicted = float(np.dot(ar_coeffs, filled[i - order:i][::-1]))
        residuals[i] = centered[i] - predicted

    return residuals


def high_pass_filter(
    values: np.ndarray,
    spline_period: float = 32.0,
    prewhiten_order: int = 1,
) -> np.ndarray:
    """
    Prepare an already-detrended/standardized series for high-frequency cross-dating.

    Removes residual multi-decadal trends with a short Cook & Peters spline
    (COFECHA-style) and optionally prewhitens AR(p) persistence, then
    re-standardizes to zero mean and unit variance.
    """
    arr = np.asarray(values, dtype=np.float64)
    valid = ~np.isnan(arr)
    if int(np.sum(valid)) < 10:
        return arr.copy()

    if spline_period is not None and spline_period > 2.5 and len(arr) >= 12:
        effective_period = min(float(spline_period), max(4.0, len(arr) * 0.8))
        smooth = cook_peters_spline(arr, period=effective_period, valid_mask=valid)
        residual = np.where(valid, arr - smooth, np.nan)
    else:
        residual = arr.copy()

    if prewhiten_order > 0:
        residual = prewhiten(residual, order=prewhiten_order)

    return standardize(residual, method="zscore")


def crossdate_transform(
    values: np.ndarray,
    spline_period: float = 20.0,
    prewhiten_order: int = 1,
) -> np.ndarray:
    """
    COFECHA/ARSTAN-style cross-dating conditioning for ring-width series.

    The branch is chosen from the values, not from what the series represents:

    - No negative value and at least one positive value (raw ring widths, and
      equally any all-positive index series such as ratio or CRN indices): takes
      the natural log of the positive values (stabilizing multiplicative growth
      variance), removes low-frequency stand/age trends with a Cook & Peters
      cubic smoothing spline, removes AR(p) biological persistence via
      Yule-Walker prewhitening, and standardizes to zero mean and unit variance.
      Zero values are not floored: they are set to NaN (treated as missing)
      before the log and stay NaN in the output.

    - Any negative value (e.g. z-scores), or no positive value at all: the log
      is skipped and the spline, AR(p) prewhitening and standardization are
      applied to the values as given.
    """
    arr = np.asarray(values, dtype=np.float64)
    valid = ~np.isnan(arr)
    if int(np.sum(valid)) < 10:
        raise ValueError("Not enough valid values for cross-dating transform")

    valid_vals = arr[valid]
    if float(np.min(valid_vals)) >= 0.0 and float(np.max(valid_vals)) > 0.0:
        # Non-negative series with at least one positive value: log-transform it.
        # Zero-width rings are excluded by positive_mask and become NaN (missing);
        # they are never replaced by a small positive value. The 1e-4 inside
        # np.maximum only keeps np.log from seeing 0 at those masked positions;
        # the one case where it alters an output is a positive value below 1e-4,
        # which is raised to 1e-4 before the log.
        positive_mask = valid & (arr > 0)
        if int(np.sum(positive_mask)) < 10:
            raise ValueError("Not enough positive values for cross-dating transform")
        working = np.where(positive_mask, np.log(np.maximum(arr, 1e-4)), np.nan)
    else:
        working = arr.copy()

    return high_pass_filter(
        working,
        spline_period=spline_period,
        prewhiten_order=prewhiten_order,
    )


def build_chronology(
    series_list: list[np.ndarray],
    years_list: list[np.ndarray],
    method: str = "biweight",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a master chronology from multiple series.

    Args:
        series_list: List of detrended/standardized series.
        years_list: List of year arrays corresponding to each series.
        method: Averaging method:
            - "mean": Simple arithmetic mean
            - "biweight": Tukey's biweight robust mean

    Returns:
        Tuple of (years, chronology_values, sample_depth).
    """
    all_years = set()
    for years in years_list:
        all_years.update(years)

    if not all_years:
        return np.array([]), np.array([]), np.array([])

    min_year = int(min(all_years))
    max_year = int(max(all_years))

    n_years = max_year - min_year + 1
    result_years = np.arange(min_year, max_year + 1)
    values_matrix = np.full((n_years, len(series_list)), np.nan)

    for i, (values, years) in enumerate(zip(series_list, years_list)):
        for v, y in zip(values, years):
            idx = int(y) - min_year
            if 0 <= idx < n_years:
                values_matrix[idx, i] = v

    if method == "mean":
        with np.errstate(invalid="ignore"):
            chronology = np.nanmean(values_matrix, axis=1)
    elif method == "biweight":
        chronology = np.array(
            [_biweight_mean(values_matrix[i, :]) for i in range(n_years)],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    sample_depth = np.sum(~np.isnan(values_matrix), axis=1)
    return result_years, chronology, sample_depth


def _biweight_mean(values: np.ndarray, c: float = 9.0) -> float:
    """Calculate Tukey's biweight robust mean."""
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan
    if len(values) == 1:
        return float(values[0])

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    if mad == 0.0:
        return float(np.mean(values))

    u = (values - median) / (c * mad)
    mask = np.abs(u) < 1.0

    if not np.any(mask):
        return median

    weights = (1.0 - u**2) ** 2
    weights[~mask] = 0.0

    return float(np.sum(weights * values) / np.sum(weights))
