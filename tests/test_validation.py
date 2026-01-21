"""
Validation tests using real ITRDB data.

These tests verify the cross-dating algorithm can correctly recover
known dates from actual tree ring data.
"""

import pytest
import numpy as np
from pathlib import Path

# Skip if reference data not downloaded
REF_DIR = Path(__file__).parent.parent / "data" / "reference"
SKIP_IF_NO_DATA = pytest.mark.skipif(
    not REF_DIR.exists() or not any(REF_DIR.rglob("*.rwl")),
    reason="Reference data not downloaded. Run 'dendro download' first."
)


@pytest.fixture
def reference_dir():
    return REF_DIR


@SKIP_IF_NO_DATA
class TestRealDataValidation:
    """Validation tests using actual ITRDB chronologies."""

    def _build_site_master(self, rwl, exclude_series_id):
        """Build master chronology from all series except one."""
        from dendro.crossdating.detrend import detrend_series, standardize, DetrendMethod

        other_series = [(sid, s) for sid, s in rwl.series.items()
                       if sid != exclude_series_id and s.length >= 50]

        if len(other_series) < 3:
            return None, None, None

        min_yr = min(s.start_year for _, s in other_series)
        max_yr = max(s.end_year for _, s in other_series)
        n_years = max_yr - min_yr + 1

        matrix = np.full((n_years, len(other_series)), np.nan)

        for i, (sid, s) in enumerate(other_series):
            try:
                dt, _ = detrend_series(s.values, DetrendMethod.SPLINE)
                st = standardize(dt)
                for j, yr in enumerate(range(s.start_year, s.end_year + 1)):
                    matrix[yr - min_yr, i] = st[j]
            except Exception:
                continue

        master = np.nanmean(matrix, axis=1)
        return master, min_yr, max_yr

    def _crossdate_within_site(self, rwl_path):
        """Test cross-dating one core against others from same site."""
        from dendro.reference.tucson_parser import parse_rwl_file
        from dendro.crossdating.correlator import find_best_match
        from dendro.crossdating.detrend import detrend_series, standardize, DetrendMethod

        rwl = parse_rwl_file(rwl_path)

        # Find series with good overlap in late 1700s
        candidates = [(sid, s) for sid, s in rwl.series.items()
                     if s.start_year <= 1780 and s.end_year >= 1800 and s.length >= 50]

        if len(candidates) < 4:
            pytest.skip(f"Not enough overlapping series in {rwl_path.name}")

        # Test first candidate
        test_id, test_series = candidates[0]
        true_end = test_series.end_year

        # Build master from others
        master, min_yr, max_yr = self._build_site_master(rwl, test_id)
        if master is None:
            pytest.skip("Could not build master chronology")

        # Detrend test series
        test_dt, _ = detrend_series(test_series.values, DetrendMethod.SPLINE)
        test_std = standardize(test_dt)

        # Cross-date
        results = find_best_match(test_std, master, min_yr, min_overlap=40, n_best=1)

        assert len(results) > 0, "No matches found"

        best = results[0]
        recovered_end = best.position + len(test_std) - 1
        error = abs(recovered_end - true_end)

        return error, best.correlation, best.t_value

    def test_nh001_red_spruce(self, reference_dir):
        """Test cross-dating with NH Red Spruce (Nancy Brook)."""
        rwl_path = reference_dir / "nh" / "nh001.rwl"
        if not rwl_path.exists():
            pytest.skip("nh001.rwl not found")

        error, corr, t_val = self._crossdate_within_site(rwl_path)

        assert error <= 2, f"Date recovery error {error} years exceeds threshold"
        assert t_val > 3.0, f"T-value {t_val} below significance threshold"

    def test_nh002_hemlock(self, reference_dir):
        """Test cross-dating with NH Eastern Hemlock (Gibb's Brook)."""
        rwl_path = reference_dir / "nh" / "nh002.rwl"
        if not rwl_path.exists():
            pytest.skip("nh002.rwl not found")

        error, corr, t_val = self._crossdate_within_site(rwl_path)

        assert error <= 2, f"Date recovery error {error} years exceeds threshold"
        assert t_val > 3.0, f"T-value {t_val} below significance threshold"

    def test_nh003_red_spruce(self, reference_dir):
        """Test cross-dating with NH Red Spruce (Nancy Brook Recollection)."""
        rwl_path = reference_dir / "nh" / "nh003.rwl"
        if not rwl_path.exists():
            pytest.skip("nh003.rwl not found")

        error, corr, t_val = self._crossdate_within_site(rwl_path)

        assert error <= 2, f"Date recovery error {error} years exceeds threshold"
        assert t_val > 3.0, f"T-value {t_val} below significance threshold"

    def test_ma015_historic_buildings(self, reference_dir):
        """Test cross-dating with MA historic building samples."""
        rwl_path = reference_dir / "ma" / "ma015.rwl"
        if not rwl_path.exists():
            pytest.skip("ma015.rwl not found")

        error, corr, t_val = self._crossdate_within_site(rwl_path)

        # MA015 appears to be from historic buildings - a relevant test case
        assert error <= 5, f"Date recovery error {error} years exceeds threshold"
        assert t_val > 2.5, f"T-value {t_val} below significance threshold"
