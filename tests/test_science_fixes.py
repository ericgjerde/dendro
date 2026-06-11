"""
Targeted regression tests for the cross-dating correctness fixes:
- Cook & Peters spline frequency response
- Gleichlaeufigkeit significance
- missing/false ring detection
- Tucson writer/parser round-trip
"""

import tempfile
from pathlib import Path

import numpy as np

from dendro.crossdating.correlator import (
    calculate_gleichlauf,
    detect_missing_ring,
    gleichlauf_significance,
)
from dendro.crossdating.detrend import _fit_cubic_spline, standardize
from dendro.imaging.path_sampler import widths_to_tucson
from dendro.reference.tucson_parser import parse_rwl_file


class TestSplineFrequencyResponse:
    def _response(self, period, wavelength, n=600):
        t = np.arange(n)
        y = 10.0 + np.sin(2 * np.pi * t / wavelength)
        curve = _fit_cubic_spline(t, y, np.ones(n, bool), period) - 10.0
        k = int(wavelength)
        sl = slice(k, n - k)
        s = np.sin(2 * np.pi * t / wavelength)
        c = np.cos(2 * np.pi * t / wavelength)
        a = 2 * np.mean(curve[sl] * s[sl])
        b = 2 * np.mean(curve[sl] * c[sl])
        return np.hypot(a, b)

    def test_fifty_percent_at_cutoff(self):
        # By definition the response is 0.5 at the cutoff wavelength.
        assert abs(self._response(50.0, 50.0) - 0.5) < 0.03

    def test_long_wavelengths_retained(self):
        assert self._response(50.0, 200.0) > 0.9

    def test_short_wavelengths_removed(self):
        assert self._response(50.0, 12.5) < 0.1


class TestGleichlaufSignificance:
    def test_chance_level_not_significant(self):
        # 50% GLK over a long series is not significant.
        assert gleichlauf_significance(50.0, 100) > 0.5

    def test_high_glk_significant(self):
        assert gleichlauf_significance(80.0, 100) < 0.01

    def test_partial_agreement_value(self):
        s1 = np.array([1, 2, 3, 2, 3])
        s2 = np.array([1, 2, 1, 2, 3])
        assert 40 <= calculate_gleichlauf(s1, s2) <= 60


class TestMissingRing:
    def test_locates_deleted_ring(self):
        rng = np.random.default_rng(7)
        ref = standardize(rng.normal(0, 1, 200) + np.sin(np.linspace(0, 20 * np.pi, 200)))
        true = ref[40:141].copy()
        sample = np.delete(true, 50)[:100]
        hint = detect_missing_ring(sample, ref[40:160])
        assert hint is not None
        assert hint.kind == "missing"
        assert abs(hint.index - 50) <= 2
        assert hint.gain > 0.1

    def test_clean_sample_no_hint(self):
        rng = np.random.default_rng(8)
        ref = standardize(rng.normal(0, 1, 200) + np.sin(np.linspace(0, 20 * np.pi, 200)))
        clean = ref[40:140].copy()
        assert detect_missing_ring(clean, ref[40:160]) is None


class TestTucsonRoundTrip:
    def test_round_trip_widths(self):
        rng = np.random.default_rng(1)
        widths_mm = np.round(rng.uniform(0.5, 3.0, 60), 2)
        text = widths_to_tucson(widths_mm, "RT01", end_year=1850)
        with tempfile.NamedTemporaryFile("w", suffix=".rwl", delete=False) as f:
            f.write(text)
            path = f.name
        try:
            rwl = parse_rwl_file(path)
            series = next(iter(rwl.series.values()))
            got = series.values / 100.0  # 0.01 mm -> mm
            expect = widths_mm[::-1]  # writer stores oldest-first
            assert series.end_year == 1850
            assert len(got) == len(expect)
            assert np.allclose(got, expect, atol=1e-6)
        finally:
            Path(path).unlink()
