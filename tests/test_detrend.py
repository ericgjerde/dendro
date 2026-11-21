"""Tests for detrending algorithms."""

import numpy as np
import pytest

from dendro.crossdating.detrend import (
    cook_peters_spline,
    crossdate_transform,
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


class TestCookPetersAndCrossdateTransform:
    """Tests for Cook & Peters spline frequency response, prewhiten NaN safety, and crossdate_transform."""

    @pytest.mark.parametrize("period", [20.0, 32.0, 64.0])
    def test_cook_peters_spline_50_percent_frequency_response(self, period):
        t = np.arange(500, dtype=np.float64)
        sig = np.sin(2.0 * np.pi * t / period)
        fit = cook_peters_spline(sig, period=period)
        ratio = np.std(fit[100:400]) / np.std(sig[100:400])
        assert abs(ratio - 0.5) < 0.01

    def test_prewhiten_preserves_mean_and_nans(self):
        rng = np.random.default_rng(7)
        n = 120
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.7 * x[i - 1] + rng.normal(0.0, 0.2)
        ratio_series = x + 1.0
        ratio_series[50] = np.nan

        residuals = prewhiten(ratio_series, order=1)
        assert np.isnan(residuals[50])
        assert not np.isnan(residuals[51])
        assert abs(float(np.nanmean(residuals))) < 0.1

    def test_crossdate_transform_handles_positive_and_zscore_series(self):
        rng = np.random.default_rng(11)
        raw_widths = np.exp(0.5 + rng.normal(0.0, 0.3, size=80))
        z_series = rng.normal(0.0, 1.0, size=80)

        t_raw = crossdate_transform(raw_widths)
        t_z = crossdate_transform(z_series)
        assert len(t_raw) == 80 and abs(float(np.nanmean(t_raw))) < 0.05
        assert len(t_z) == 80 and abs(float(np.nanmean(t_z))) < 0.05



def _lag_autocorrelation(values, lag=1):
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    return float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x))


def _ar1_whiten_by_hand(series):
    """Mean-centre, estimate lag-1 autocorrelation, subtract phi * previous value."""
    centered = series - series.mean()
    n = len(centered)
    phi = (np.dot(centered[:-1], centered[1:]) / (n - 1)) / (np.dot(centered, centered) / n)
    white = centered.copy()
    white[1:] = centered[1:] - phi * centered[:-1]
    return white, phi


def _zscore_by_hand(series):
    return (series - series.mean()) / series.std()


def _ar_process(coefficients, n, seed, noise_scale=1.0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_scale, size=n)
    x = np.zeros(n)
    order = len(coefficients)
    for i in range(order, n):
        x[i] = noise[i] + sum(c * x[i - k - 1] for k, c in enumerate(coefficients))
    return x, noise


class TestDetrendSeriesDefaultSpline:
    """The default spline is 32 years, or 2/3 of the series length when shorter."""

    def test_long_series_default_is_the_32_year_spline(self):
        rng = np.random.default_rng(41)
        values = 1.0 + np.exp(rng.normal(0.0, 0.3, size=120))

        detrended, curve = detrend_series(values, DetrendMethod.SPLINE)
        _, curve_32 = detrend_series(values, DetrendMethod.SPLINE, spline_period=32.0)
        _, curve_two_thirds = detrend_series(values, DetrendMethod.SPLINE, spline_period=80.0)

        assert np.array_equal(curve, curve_32)
        assert np.array_equal(curve, cook_peters_spline(values, period=32.0))
        assert np.max(np.abs(curve - curve_two_thirds)) > 0.05
        assert np.array_equal(detrended, values / curve)

    def test_short_series_default_is_two_thirds_of_length(self):
        rng = np.random.default_rng(42)
        values = 1.0 + np.exp(rng.normal(0.0, 0.3, size=30))

        _, curve = detrend_series(values, DetrendMethod.SPLINE)
        _, curve_20 = detrend_series(values, DetrendMethod.SPLINE, spline_period=20.0)
        _, curve_32 = detrend_series(values, DetrendMethod.SPLINE, spline_period=32.0)

        assert np.array_equal(curve, curve_20)
        assert np.max(np.abs(curve - curve_32)) > 0.01


class TestPrewhitenRemovesKnownPersistence:
    """prewhiten must apply the fitted AR coefficients, each to the right lag."""

    def test_ar1_series_is_whitened(self):
        series, noise = _ar_process([0.7], n=3000, seed=51)
        assert _lag_autocorrelation(series) > 0.6

        residuals = prewhiten(series, order=1)

        assert abs(_lag_autocorrelation(residuals)) < 0.05
        # What is left is the innovation series, not a rescaled copy of the input.
        assert np.corrcoef(residuals[1:], noise[1:])[0, 1] > 0.99
        assert np.var(residuals) / np.var(noise) == pytest.approx(1.0, abs=0.05)

    def test_ar1_matches_hand_computed_filter(self):
        series, _ = _ar_process([0.7], n=200, seed=52)
        expected, phi = _ar1_whiten_by_hand(series)
        assert 0.5 < phi < 0.9
        np.testing.assert_allclose(prewhiten(series, order=1), expected, rtol=0, atol=1e-12)

    def test_ar2_series_is_whitened(self):
        series, noise = _ar_process([0.6, -0.3], n=5000, seed=53)
        assert _lag_autocorrelation(series, 1) > 0.4

        residuals = prewhiten(series, order=2)

        assert abs(_lag_autocorrelation(residuals, 1)) < 0.05
        assert abs(_lag_autocorrelation(residuals, 2)) < 0.05
        assert np.corrcoef(residuals[2:], noise[2:])[0, 1] > 0.99
        assert np.var(residuals) / np.var(noise) == pytest.approx(1.0, abs=0.05)

    def test_ar2_matches_hand_computed_yule_walker_filter(self):
        series, _ = _ar_process([0.6, -0.3], n=400, seed=54)
        centered = series - series.mean()
        n = len(centered)
        c0 = np.dot(centered, centered) / n
        r1 = np.dot(centered[:-1], centered[1:]) / (n - 1) / c0
        r2 = np.dot(centered[:-2], centered[2:]) / (n - 2) / c0
        phi1, phi2 = np.linalg.solve([[1.0, r1], [r1, 1.0]], [r1, r2])
        # The two coefficients are clearly different, so swapping lags is visible.
        assert phi1 > 0.4 and phi2 < -0.15
        expected = centered.copy()
        expected[2:] = centered[2:] - phi1 * centered[1:-1] - phi2 * centered[:-2]

        np.testing.assert_allclose(prewhiten(series, order=2), expected, rtol=0, atol=1e-12)


class TestCrossdateTransformPipeline:
    """crossdate_transform = log -> 20-year spline residual -> AR(1) whitening -> z-score."""

    @staticmethod
    def _widths(seed, n=120):
        """Positive widths with AR(1) persistence and an age trend."""
        persistent, _ = _ar_process([0.6], n=n, seed=seed, noise_scale=0.25)
        return np.exp(0.5 + persistent - 0.004 * np.arange(n))

    def test_positive_widths_equal_the_explicit_pipeline(self):
        widths = self._widths(61)

        log_widths = np.log(widths)
        residual = log_widths - cook_peters_spline(log_widths, period=20.0)
        white, phi = _ar1_whiten_by_hand(residual)
        expected = _zscore_by_hand(white)

        got = crossdate_transform(widths)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)
        # Each stage matters for this series: leaving one out moves the answer.
        assert abs(phi) > 0.05
        assert np.max(np.abs(_zscore_by_hand(residual) - expected)) > 0.05
        assert np.max(np.abs(got - got[::-1])) > 0.5

    def test_positive_index_series_also_take_the_log_branch(self):
        rng = np.random.default_rng(62)
        indices = np.clip(1.0 + rng.normal(0.0, 0.2, size=100), 0.3, None)

        log_indices = np.log(indices)
        residual = log_indices - cook_peters_spline(log_indices, period=20.0)
        expected = _zscore_by_hand(_ar1_whiten_by_hand(residual)[0])

        np.testing.assert_allclose(crossdate_transform(indices), expected, rtol=0, atol=1e-10)

    def test_series_with_negative_values_skip_the_log(self):
        rng = np.random.default_rng(63)
        z_series = rng.normal(0.0, 1.0, size=100)
        assert z_series.min() < 0

        residual = z_series - cook_peters_spline(z_series, period=20.0)
        expected = _zscore_by_hand(_ar1_whiten_by_hand(residual)[0])

        np.testing.assert_allclose(crossdate_transform(z_series), expected, rtol=0, atol=1e-10)

    def test_output_is_standardized(self):
        out = crossdate_transform(self._widths(64))
        assert abs(float(np.mean(out))) < 1e-12
        assert float(np.std(out)) == pytest.approx(1.0, abs=1e-12)

    def test_log_makes_exponential_growth_trend_irrelevant(self):
        """exp(trend) * widths is log(widths) + a straight line, which the spline removes.

        Without the log the trend would leave the variance changing along the series.
        """
        widths = self._widths(65)
        trended = widths * np.exp(-0.01 * np.arange(len(widths)))
        np.testing.assert_allclose(
            crossdate_transform(trended), crossdate_transform(widths), rtol=0, atol=1e-9
        )

    def test_slow_variation_is_removed(self):
        rng = np.random.default_rng(66)
        t = np.arange(300)
        slow = np.sin(2.0 * np.pi * t / 150.0)
        widths = np.exp(slow + rng.normal(0.0, 0.2, size=300))
        assert np.corrcoef(np.log(widths), slow)[0, 1] > 0.9

        out = crossdate_transform(widths)
        assert abs(np.corrcoef(out, slow)[0, 1]) < 0.05

    def test_lag_one_autocorrelation_falls(self):
        widths = self._widths(67, n=300)
        assert _lag_autocorrelation(np.log(widths)) > 0.5

        whitened = crossdate_transform(widths)
        spline_only = crossdate_transform(widths, prewhiten_order=0)

        assert abs(_lag_autocorrelation(whitened)) < 0.1
        assert _lag_autocorrelation(spline_only) > 0.2

    def test_output_keeps_time_order(self):
        rng = np.random.default_rng(68)
        widths = np.exp(rng.normal(0.0, 0.2, size=80))
        widths[17] *= 0.05  # one very narrow ring, off-centre

        out = crossdate_transform(widths)
        assert int(np.argmin(out)) == 17
        assert out[17] < -4.0
