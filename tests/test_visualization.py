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


class TestDiagnosticPlotInputs:
    """`dendro date --plot` must draw the series the reported statistics came from."""

    @pytest.mark.parametrize("reverse", [False, True], ids=["oldest_to_newest", "bark_to_pith"])
    def test_generate_plots_passes_the_series_the_matcher_scored(self, monkeypatch, tmp_path, reverse):
        import dendro.visualization.plots as plots
        from dendro.cli.main import _generate_plots
        from dendro.crossdating.matcher import CrossdateMatcher
        from dendro.reference.chronology_index import ChronologyIndex, MasterChronology, ReferenceManifestEntry

        rng = np.random.default_rng(11)
        years = np.arange(200)
        common = rng.normal(0.0, 0.15, size=200)
        # A chronology index with slow swings: its raw values differ from the
        # high-pass, prewhitened series the matcher correlates against.
        reference = 1.0 + 0.4 * np.sin(2.0 * np.pi * years / 90.0) + common  # 1700..1899
        age_trend = 2.0 * np.exp(-np.arange(100) / 60.0) + 0.5
        widths = age_trend * np.exp(1.5 * common[60:160] + rng.normal(0.0, 0.05, size=100))  # 1760..1859

        index = ChronologyIndex()
        index._add_entry(
            ReferenceManifestEntry(
                filepath="/nonexistent/plotref.crn",
                site_id="PLOTREF",
                site_name="Plot Reference Site",
                species="TSCA",
                state="NH",
                start_year=1700,
                end_year=1899,
                num_years=200,
                num_series=12,
                file_type="crn",
                material_group="hemlock",
                master=MasterChronology(
                    start_year=1700,
                    end_year=1899,
                    values=reference.tolist(),
                    sample_depth=[12] * 200,
                    build_method="crn-index",
                    detrend_method="spline",
                ),
            )
        )
        matcher = CrossdateMatcher(index=index)
        measurements = widths[::-1] if reverse else widths

        report = matcher.date_sample(
            measurements,
            sample_name="plot-sample",
            orientation="auto",
            era_start=1750,
            era_end=1900,
        )
        best = report.best_candidate
        assert best is not None
        assert best.outer_ring_year == 1859
        assert best.overlap == len(widths)
        assert report.chosen_orientation == ("bark_to_pith" if reverse else "oldest_to_newest")

        captured = {}

        def capture_plot_inputs(**kwargs):
            captured.update(kwargs)
            return kwargs["output_path"]

        monkeypatch.setattr(plots, "save_diagnostic_plots", capture_plot_inputs)
        _generate_plots(report, matcher, measurements, tmp_path / "diagnostic.png")

        assert captured["reference_start_year"] == 1700
        offset = best.proposed_start_year - captured["reference_start_year"]
        plotted_sample = np.asarray(captured["sample"])
        plotted_reference = np.asarray(captured["reference"])[offset:offset + len(plotted_sample)]
        plotted_r = float(np.corrcoef(plotted_sample, plotted_reference)[0, 1])
        # The plotted pair reproduces the correlation printed next to it.
        assert plotted_r == pytest.approx(best.correlation, abs=1e-9)


class TestSegmentPlotIsReadableWithoutRedGreenVision:
    """Segment quality must be encoded by pattern as well as colour."""

    def _bars(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex
        from dendro.visualization.plots import plot_segment_analysis

        fig, ax = plt.subplots()
        try:
            # One segment per quality class: strong, good, marginal, weak.
            segments = [(0, 0.70, 7.0), (25, 0.55, 5.0), (50, 0.45, 3.7), (75, 0.10, 1.0)]
            plot_segment_analysis(segments, sample_start_year=1700, ax=ax)
            bars = [(patch.get_hatch(), to_hex(patch.get_facecolor())) for patch in ax.patches]
            legend = ax.get_legend()
            legend_entries = [
                (handle.get_hatch(), text.get_text())
                for handle, text in zip(legend.get_patches(), legend.get_texts())
            ]
        finally:
            plt.close(fig)
        return bars, legend_entries

    def test_each_quality_class_has_its_own_fill_pattern(self):
        bars, _ = self._bars()
        hatches = [hatch for hatch, _ in bars]
        assert len(bars) == 4
        assert len(set(hatches)) == 4

    def test_no_class_is_signalled_by_pure_red_or_green(self):
        bars, _ = self._bars()
        colors = {color for _, color in bars}
        assert not colors & {"#008000", "#ff0000", "#00ff00"}

    def test_legend_shows_the_same_patterns_as_the_bars(self):
        bars, legend_entries = self._bars()
        assert [hatch for hatch, _ in legend_entries] == [hatch for hatch, _ in bars]
        assert [label.split()[0] for _, label in legend_entries] == ["Strong", "Good", "Marginal", "Weak"]
