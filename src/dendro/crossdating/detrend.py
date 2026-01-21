"""
Detrending and standardization algorithms for ring-width series.

Tree ring widths exhibit age-related growth trends (young trees typically grow
faster). Detrending removes this trend to reveal climate signal, making series
from different trees comparable.

Common detrending methods:
1. Negative exponential curve
2. Cubic smoothing spline
3. Modified negative exponential
4. Regional curve standardization (RCS)

After detrending, series are typically standardized to unit variance.
"""

from enum import Enum
from typing import Optional

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline


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
        spline_period: For spline method, the wavelength of 50% frequency
                      response. If None, uses 2/3 of series length.

    Returns:
        Tuple of (detrended_values, fitted_curve).
        Detrended values are the residuals or ratios depending on method.
    """
    if len(values) < 10:
        raise ValueError("Series too short for detrending (minimum 10 values)")

    x = np.arange(len(values))
    values = np.asarray(values, dtype=np.float64)

    # Handle missing values
    valid_mask = ~np.isnan(values) & (values > 0)
    if np.sum(valid_mask) < 10:
        raise ValueError("Not enough valid values for detrending")

    if method == DetrendMethod.NONE:
        return values.copy(), np.ones_like(values)

    elif method == DetrendMethod.MEAN:
        mean_val = np.nanmean(values)
        curve = np.full_like(values, mean_val)
        return values / curve, curve

    elif method == DetrendMethod.LINEAR:
        # Fit linear trend
        coeffs = np.polyfit(x[valid_mask], values[valid_mask], 1)
        curve = np.polyval(coeffs, x)
        # Ensure curve is positive
        curve = np.maximum(curve, 0.01)
        return values / curve, curve

    elif method == DetrendMethod.NEGATIVE_EXPONENTIAL:
        curve = _fit_negative_exponential(x, values, valid_mask)
        return values / curve, curve

    elif method == DetrendMethod.MODIFIED_NEGEXP:
        curve = _fit_modified_negexp(x, values, valid_mask)
        return values / curve, curve

    elif method == DetrendMethod.SPLINE:
        if spline_period is None:
            # Default: 2/3 of series length (67% variance retained at that wavelength)
            spline_period = len(values) * 2 / 3

        curve = _fit_cubic_spline(x, values, valid_mask, spline_period)
        return values / curve, curve

    else:
        raise ValueError(f"Unknown detrend method: {method}")


def _fit_negative_exponential(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Fit a negative exponential curve: y = a * exp(-b * x) + c

    Falls back to linear if exponential fit fails.
    """

    def neg_exp(t, a, b, c):
        return a * np.exp(-b * t) + c

    try:
        # Initial guesses
        y_valid = y[mask]
        a0 = y_valid[0] - y_valid[-1]
        c0 = y_valid[-1]
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
        curve = np.maximum(curve, 0.01)  # Ensure positive
        return curve

    except (RuntimeError, ValueError):
        # Fall back to linear
        coeffs = np.polyfit(x[mask], y[mask], 1)
        curve = np.polyval(coeffs, x)
        return np.maximum(curve, 0.01)


def _fit_modified_negexp(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Fit modified negative exponential (Hugershoff growth curve).

    y = a * t^b * exp(-c * t) + d

    This allows for initial juvenile growth increase.
    """

    def hugershoff(t, a, b, c, d):
        t_safe = np.maximum(t, 0.1)
        return a * np.power(t_safe, b) * np.exp(-c * t_safe) + d

    try:
        y_valid = y[mask]
        x_valid = x[mask] + 1  # Shift to avoid t=0

        # Initial guesses
        popt, _ = curve_fit(
            hugershoff,
            x_valid,
            y_valid,
            p0=[np.max(y_valid), 0.5, 0.01, np.min(y_valid)],
            bounds=([0, 0, 0, 0], [np.inf, 5, 1, np.inf]),
            maxfev=5000,
        )

        curve = hugershoff(x + 1, *popt)
        return np.maximum(curve, 0.01)

    except (RuntimeError, ValueError):
        # Fall back to negative exponential
        return _fit_negative_exponential(x, y, mask)


def _fit_cubic_spline(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    period: float,
) -> np.ndarray:
    """
    Fit a cubic smoothing spline with specified frequency response.

    Args:
        period: Wavelength at which 50% of variance is retained.
    """
    # Convert period to smoothing parameter
    # Approximate relationship between period and smoothing
    n = len(y)
    s = n / (period / 2)  # Smoothing factor

    try:
        spline = UnivariateSpline(
            x[mask],
            y[mask],
            s=s * np.var(y[mask]),
            k=3,
        )
        curve = spline(x)
        return np.maximum(curve, 0.01)

    except Exception:
        # Fall back to linear
        coeffs = np.polyfit(x[mask], y[mask], 1)
        curve = np.polyval(coeffs, x)
        return np.maximum(curve, 0.01)


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
            - "ratio": Values already as ratios (just center on 1.0)

    Returns:
        Standardized series.
    """
    values = np.asarray(values, dtype=np.float64)

    if method == "zscore":
        mean = np.nanmean(values)
        std = np.nanstd(values)
        if std > 0:
            return (values - mean) / std
        else:
            return values - mean

    elif method == "ratio":
        # For ratio indices (from division by fitted curve)
        # Center on 1.0 (already should be approximately 1.0)
        return values / np.nanmean(values)

    else:
        raise ValueError(f"Unknown standardization method: {method}")


def prewhiten(values: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Remove autocorrelation using AR model.

    Prewhitening removes persistence (autocorrelation) from a series,
    which can improve cross-dating statistics.

    Args:
        values: Ring-width indices.
        order: Order of autoregressive model.

    Returns:
        Residuals from AR model fit.
    """
    values = np.asarray(values, dtype=np.float64)

    # Handle NaN values
    valid_mask = ~np.isnan(values)
    if np.sum(valid_mask) < order + 10:
        return values

    # Fit AR model using Yule-Walker equations
    try:
        from scipy.signal import lfilter

        # Estimate AR coefficients
        r = np.correlate(values[valid_mask] - np.mean(values[valid_mask]),
                        values[valid_mask] - np.mean(values[valid_mask]),
                        mode='full')
        r = r[len(r)//2:]
        r = r / r[0]

        # Solve Yule-Walker equations
        R = np.zeros((order, order))
        for i in range(order):
            for j in range(order):
                R[i, j] = r[abs(i - j)]

        try:
            ar_coeffs = np.linalg.solve(R, r[1:order+1])
        except np.linalg.LinAlgError:
            return values

        # Apply AR filter to get residuals
        residuals = np.zeros_like(values)
        for i in range(order, len(values)):
            predicted = np.sum(ar_coeffs * values[i-order:i][::-1])
            residuals[i] = values[i] - predicted

        # First 'order' values are undefined, copy original
        residuals[:order] = values[:order] - np.mean(values[:order])

        return residuals

    except Exception:
        return values


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
    # Find overall time span
    all_years = set()
    for years in years_list:
        all_years.update(years)

    if not all_years:
        return np.array([]), np.array([]), np.array([])

    min_year = min(all_years)
    max_year = max(all_years)

    # Create arrays for results
    n_years = max_year - min_year + 1
    result_years = np.arange(min_year, max_year + 1)
    values_matrix = np.full((n_years, len(series_list)), np.nan)

    # Fill in values
    for i, (values, years) in enumerate(zip(series_list, years_list)):
        for v, y in zip(values, years):
            idx = y - min_year
            if 0 <= idx < n_years:
                values_matrix[idx, i] = v

    # Calculate chronology
    if method == "mean":
        chronology = np.nanmean(values_matrix, axis=1)
    elif method == "biweight":
        chronology = np.array([
            _biweight_mean(values_matrix[i, :])
            for i in range(n_years)
        ])
    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate sample depth
    sample_depth = np.sum(~np.isnan(values_matrix), axis=1)

    return result_years, chronology, sample_depth


def _biweight_mean(values: np.ndarray, c: float = 9.0) -> float:
    """
    Calculate Tukey's biweight robust mean.

    This is less sensitive to outliers than arithmetic mean.
    """
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan
    if len(values) == 1:
        return values[0]

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad == 0:
        return np.mean(values)

    u = (values - median) / (c * mad)

    # Values outside |u| > 1 get zero weight
    mask = np.abs(u) < 1

    if not np.any(mask):
        return median

    weights = (1 - u**2) ** 2
    weights[~mask] = 0

    return np.sum(weights * values) / np.sum(weights)
