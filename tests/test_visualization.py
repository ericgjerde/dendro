"""
Tests for the visualization module.
"""

import numpy as np
import pytest

from dendro.visualization.plots import (
    detect_marker_years,
    identify_known_markers,
    MARKER_YEARS,
)


class TestMarkerYearDetection:
    """Tests for marker year detection."""

    def test_detect_anomalous_narrow_ring(self):
        """Detect anomalously narrow rings."""
        # Create series with one very narrow ring
        values = np.ones(100) * 1.0
        values[50] = 0.1  # Very narrow ring

        anomalies = detect_marker_years(values, threshold_sigma=2.0)

        assert len(anomalies) >= 1
        # Find the anomaly at index 50
        narrow_anomaly = [a for a in anomalies if a[0] == 50]
        assert len(narrow_anomaly) == 1
        assert narrow_anomaly[0][1] < -2.0  # Negative z-score
        assert "narrow" in narrow_anomaly[0][2].lower()

    def test_detect_anomalous_wide_ring(self):
        """Detect anomalously wide rings."""
        values = np.ones(100) * 1.0
        values[30] = 3.0  # Very wide ring

        anomalies = detect_marker_years(values, threshold_sigma=2.0)

        assert len(anomalies) >= 1
        wide_anomaly = [a for a in anomalies if a[0] == 30]
        assert len(wide_anomaly) == 1
        assert wide_anomaly[0][1] > 2.0  # Positive z-score
        assert "wide" in wide_anomaly[0][2].lower()

    def test_no_anomalies_uniform_series(self):
        """No anomalies detected in uniform series."""
        values = np.ones(100) * 1.0
        anomalies = detect_marker_years(values, threshold_sigma=2.0)
        assert len(anomalies) == 0

    def test_marker_year_1816_identification(self):
        """Identify Year Without Summer (1816) marker."""
        # Create anomalies list where index 66 corresponds to 1816
        # if start year is 1750
        anomalies = [(66, -2.5, "extremely narrow (drought/cold?)")]
        proposed_start_year = 1750  # So index 66 = year 1816

        matches = identify_known_markers(anomalies, proposed_start_year)

        assert len(matches) >= 1
        years_matched = [m[0] for m in matches]
        assert 1816 in years_matched

        # Check description mentions Tambora
        match_1816 = [m for m in matches if m[0] == 1816][0]
        assert "Tambora" in match_1816[2] or "Summer" in match_1816[2]


class TestMarkerYearsData:
    """Tests for the MARKER_YEARS constant."""

    def test_marker_years_includes_1816(self):
        """Year Without Summer should be included."""
        assert 1816 in MARKER_YEARS
        assert "Summer" in MARKER_YEARS[1816] or "Tambora" in MARKER_YEARS[1816]

    def test_marker_years_includes_1709(self):
        """Great Frost should be included."""
        assert 1709 in MARKER_YEARS

    def test_all_marker_years_have_descriptions(self):
        """All marker years should have non-empty descriptions."""
        for year, description in MARKER_YEARS.items():
            assert isinstance(year, int)
            assert isinstance(description, str)
            assert len(description) > 0


class TestPlotFunctions:
    """Basic tests for plot functions (without rendering)."""

    def test_import_plot_functions(self):
        """All plot functions should be importable."""
        from dendro.visualization.plots import (
            plot_crossdate_results,
            plot_correlation_profile,
            plot_segment_analysis,
            plot_sample_reference_overlay,
            save_diagnostic_plots,
        )

        assert callable(plot_crossdate_results)
        assert callable(plot_correlation_profile)
        assert callable(plot_segment_analysis)
        assert callable(plot_sample_reference_overlay)
        assert callable(save_diagnostic_plots)
