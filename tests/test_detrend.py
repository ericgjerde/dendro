"""Tests for detrending algorithms."""

import numpy as np
import pytest

from dendro.crossdating.detrend import (
    detrend_series,
    standardize,
    prewhiten,
    build_chronology,
    DetrendMethod,
)


class TestDetrendSeries:
    """Tests for detrending methods."""

    def test_mean_detrend(self):
        """Test mean detrending."""
        values = np.array([100, 110, 90, 105, 95, 100, 100, 105, 95, 100])

        detrended, curve = detrend_series(values, DetrendMethod.MEAN)

        # Curve should be constant (the mean)
        assert np.allclose(curve, np.mean(values))

        # Detrended values should be ratios
        assert np.allclose(detrended, values / np.mean(values))

    def test_linear_detrend(self):
        """Test linear detrending."""
        # Create series with linear trend
        trend = np.linspace(100, 50, 50)  # Decreasing trend
        noise = np.random.randn(50) * 5
        values = trend + noise

        detrended, curve = detrend_series(values, DetrendMethod.LINEAR)

        # Curve should roughly follow the trend
        assert curve[0] > curve[-1]  # Decreasing

        # Detrended should have reduced trend
        detrended_trend = np.polyfit(np.arange(len(detrended)), detrended, 1)[0]
        original_trend = np.polyfit(np.arange(len(values)), values, 1)[0]
        assert abs(detrended_trend) < abs(original_trend)

    def test_spline_detrend(self):
        """Test spline detrending."""
        # Create series with curved trend
        x = np.arange(100)
        trend = 100 * np.exp(-0.01 * x)  # Exponential decay
        noise = np.random.randn(100) * 5
        values = trend + noise + 10

        detrended, curve = detrend_series(values, DetrendMethod.SPLINE)

        # Curve should be smooth and follow general trend
        assert len(curve) == len(values)
        assert all(curve > 0)  # Positive curve

    def test_none_detrend(self):
        """Test no detrending."""
        # Need at least 10 values for detrending
        values = np.array([100, 110, 90, 105, 95, 100, 102, 98, 101, 99])

        detrended, curve = detrend_series(values, DetrendMethod.NONE)

        # Should return original values
        assert np.allclose(detrended, values)
        assert np.allclose(curve, 1.0)

    def test_short_series_rejected(self):
        """Test that too-short series raise error."""
        values = np.array([100, 110, 90])

        with pytest.raises(ValueError, match="too short"):
            detrend_series(values, DetrendMethod.SPLINE)


class TestStandardize:
    """Tests for standardization."""

    def test_zscore_standardization(self):
        """Test z-score standardization."""
        values = np.array([100, 110, 90, 105, 95])

        standardized = standardize(values, method="zscore")

        # Should have mean ~0 and std ~1
        assert abs(np.mean(standardized)) < 0.01
        assert abs(np.std(standardized) - 1.0) < 0.01

    def test_ratio_standardization(self):
        """Test ratio standardization."""
        values = np.array([0.9, 1.1, 0.95, 1.05, 1.0])

        standardized = standardize(values, method="ratio")

        # Should be centered on 1.0
        assert abs(np.mean(standardized) - 1.0) < 0.01


class TestPrewhiten:
    """Tests for prewhitening."""

    def test_reduces_autocorrelation(self):
        """Test that prewhitening reduces autocorrelation."""
        # Create series with high autocorrelation
        np.random.seed(42)
        n = 100
        values = np.zeros(n)
        values[0] = 1
        for i in range(1, n):
            values[i] = 0.8 * values[i-1] + np.random.randn() * 0.2

        # Calculate original autocorrelation
        original_ac = np.corrcoef(values[:-1], values[1:])[0, 1]

        # Prewhiten
        residuals = prewhiten(values, order=1)

        # Calculate residual autocorrelation
        residual_ac = np.corrcoef(residuals[1:-1], residuals[2:])[0, 1]

        # Residuals should have lower autocorrelation
        assert abs(residual_ac) < abs(original_ac)


class TestBuildChronology:
    """Tests for chronology building."""

    def test_simple_mean(self):
        """Test building chronology with simple mean."""
        series1 = np.array([1.0, 1.0, 1.0, 1.0])
        series2 = np.array([2.0, 2.0, 2.0, 2.0])

        years1 = np.array([1800, 1801, 1802, 1803])
        years2 = np.array([1800, 1801, 1802, 1803])

        years, chron, depth = build_chronology(
            [series1, series2],
            [years1, years2],
            method="mean",
        )

        assert len(years) == 4
        assert np.allclose(chron, 1.5)  # Mean of 1 and 2
        assert all(depth == 2)  # Both series present

    def test_partial_overlap(self):
        """Test chronology with partial overlap."""
        series1 = np.array([1.0, 1.0, 1.0])
        series2 = np.array([2.0, 2.0, 2.0])

        years1 = np.array([1800, 1801, 1802])
        years2 = np.array([1801, 1802, 1803])

        years, chron, depth = build_chronology(
            [series1, series2],
            [years1, years2],
            method="mean",
        )

        assert len(years) == 4  # 1800-1803
        assert years[0] == 1800
        assert years[-1] == 1803

        # 1800: only series1 (depth=1)
        # 1801, 1802: both (depth=2)
        # 1803: only series2 (depth=1)
        assert depth[0] == 1
        assert depth[1] == 2
        assert depth[2] == 2
        assert depth[3] == 1

    def test_biweight_mean(self):
        """Test biweight robust mean."""
        # Use varying data so MAD is non-zero
        series1 = np.array([0.9, 1.0, 1.1])
        series2 = np.array([1.1, 1.0, 0.9])
        series3 = np.array([10.0, 10.0, 10.0])  # Outlier (10x the others)

        years = np.array([1800, 1801, 1802])

        _, chron_mean, _ = build_chronology(
            [series1, series2, series3],
            [years, years, years],
            method="mean",
        )

        _, chron_biweight, _ = build_chronology(
            [series1, series2, series3],
            [years, years, years],
            method="biweight",
        )

        # Verify biweight produces valid values
        assert not np.isnan(chron_biweight[0])
        # Mean would be ~4, biweight should be closer to 1 (downweighting outlier)
        assert chron_biweight[0] < chron_mean[0]
