"""Regression tests for imaging ring detection, Tucson export, and index integrity."""

from pathlib import Path
import numpy as np
import pytest

from dendro.imaging import path_sampler, ring_detector
from dendro.imaging.path_sampler import ProfileSample, SamplePath, profile_to_ring_positions, widths_to_tucson
from dendro.imaging.ring_detector import _calculate_gradient, _enhance_contrast, _find_boundaries
from dendro.reference.chronology_index import ChronologyIndex
from dendro.reference.tucson_parser import parse_rwl_file


def _synthetic_ring_profile(
    n_rings: int = 12,
    ring_px: int = 40,
    high: float = 220.0,
    low: float = 40.0,
) -> np.ndarray:
    """Create a realistic sawtooth ring intensity profile: gradual earlywood->latewood, sharp boundary."""
    segments = []
    for _ in range(n_rings):
        # Gradual darkening from bright earlywood to dark latewood, then sharp jump to next ring.
        segments.append(np.linspace(high, low, ring_px, endpoint=False))
    return np.concatenate(segments)


def _latewood_band_profile(n_rings: int = 12) -> np.ndarray:
    """60 px rings with a two-edged dark latewood band.

    Bright earlywood (30 px), a soft 10 px earlywood->latewood ramp, a dark
    latewood band (20 px), then the sharp ring boundary back to earlywood.
    Both edges of the band are strong, so abs(gradient) sees two per ring.
    """
    ring = np.concatenate([
        np.full(30, 200.0),
        np.linspace(200.0, 60.0, 10, endpoint=False),
        np.full(20, 60.0),
    ])
    return np.tile(ring, n_rings)


def _graded_boundary_profile(jumps: list[float], ring_px: int = 40) -> np.ndarray:
    """Sawtooth whose sharp boundary after ring i has height jumps[i] (the last one is the profile end)."""
    return np.concatenate([np.linspace(200.0, 200.0 - j, ring_px, endpoint=False) for j in jumps])


def _profile_sample(intensities: np.ndarray, dpi: int = 1200) -> ProfileSample:
    mm_per_px = 25.4 / dpi
    return ProfileSample(
        intensities=intensities,
        positions_px=np.arange(len(intensities)),
        positions_mm=np.arange(len(intensities)) * mm_per_px,
        path=SamplePath.from_points([(0, 0), (len(intensities), 0)]),
        dpi=dpi,
    )


def _assert_near_multiples(positions_px, ring_px: int, tol: float = 6.0):
    # A step smoothed by a k-wide box has a flat-topped gradient; the reported peak can sit
    # anywhere on that plateau, so allow half the widest kernel (11 px) plus one.
    positions_px = np.asarray(positions_px, dtype=np.float64)
    offsets = np.abs(positions_px - np.round(positions_px / ring_px) * ring_px)
    assert np.all(offsets <= tol), f"boundaries off the true ring boundaries: {positions_px.tolist()}"


def test_directional_gradient_does_not_double_count_rings():
    profile = _latewood_band_profile(n_rings=12)
    grad = _calculate_gradient(profile)
    peaks = _find_boundaries(grad, min_distance=15, max_distance=100, threshold=0.5)

    # 12 rings have 11 internal sharp ring boundaries. abs(gradient) also marks the
    # 12 earlywood->latewood ramps (23 peaks), which this must not do.
    assert len(peaks) == 11
    _assert_near_multiples(peaks, 60)


def test_profile_to_ring_positions_does_not_double_count():
    sample = _profile_sample(_latewood_band_profile(n_rings=12))
    boundaries_mm, widths_mm = profile_to_ring_positions(sample, sensitivity=0.5, min_width_mm=0.2)

    assert len(boundaries_mm) == 11
    _assert_near_multiples(boundaries_mm / (25.4 / sample.dpi), 60)
    assert len(widths_mm) == 10
    np.testing.assert_allclose(widths_mm, 60 * 25.4 / sample.dpi, atol=2 * 25.4 / sample.dpi)


def test_sensitivity_increases_detected_boundaries():
    # Strong boundaries (160) alternate with weak ones (40, a quarter of the strongest).
    # 10 rings -> 9 internal boundaries: 5 strong, 4 weak.
    profile = _graded_boundary_profile([160.0, 40.0] * 5)
    grad = _calculate_gradient(profile)

    low_peaks = _find_boundaries(grad, min_distance=15, max_distance=100, threshold=0.2)
    high_peaks = _find_boundaries(grad, min_distance=15, max_distance=100, threshold=0.85)

    assert len(low_peaks) == 5  # cutoff 0.45 of max: strong boundaries only
    assert len(high_peaks) == 9  # cutoff 0.125 of max: strong and weak
    assert len(high_peaks) > len(low_peaks)
    assert set(low_peaks) <= set(high_peaks)

    sample = _profile_sample(profile)
    low_mm, _ = profile_to_ring_positions(sample, sensitivity=0.2, min_width_mm=0.2)
    high_mm, _ = profile_to_ring_positions(sample, sensitivity=0.85, min_width_mm=0.2)
    assert (len(low_mm), len(high_mm)) == (5, 9)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("ring_px", [40, 100, 200, 300, 500, 800, 1000])
def test_boundary_polarity_does_not_depend_on_ring_width(ring_px: int, reverse: bool):
    # Sharp boundaries cover a smaller share of the profile as rings get wider; the
    # polarity choice must still follow the sharp boundaries, not the gradual slope.
    profile = _synthetic_ring_profile(n_rings=12, ring_px=ring_px)
    if reverse:
        profile = profile[::-1].copy()

    grad = _calculate_gradient(profile)
    peaks = _find_boundaries(grad, min_distance=15, max_distance=2 * ring_px, threshold=0.5)
    assert len(peaks) == 11
    _assert_near_multiples(peaks, ring_px)

    boundaries_mm, _ = profile_to_ring_positions(_profile_sample(profile), sensitivity=0.5, min_width_mm=0.2)
    assert len(boundaries_mm) == 11
    _assert_near_multiples(boundaries_mm / (25.4 / 1200), ring_px)


def test_smoothing_keeps_a_constant_profile_constant():
    flat = np.full(50, 200.0)
    np.testing.assert_allclose(path_sampler._smooth_profile(flat), flat)
    np.testing.assert_allclose(path_sampler._smooth_profile(flat, window=11), flat)
    assert float(np.max(_calculate_gradient(flat))) == pytest.approx(0.0, abs=1e-9)


def test_profile_ends_do_not_outweigh_real_boundaries():
    # Bright, low-contrast wood: zero padding at the ends would look like a step of
    # ~150 next to real boundaries of 40 and push the real ones under the cutoff.
    profile = _synthetic_ring_profile(n_rings=12, ring_px=40, high=170.0, low=130.0)

    for candidate in (profile, profile[::-1].copy()):
        grad = _calculate_gradient(candidate)
        assert 30 <= int(np.argmax(grad)) <= len(candidate) - 30
        peaks = _find_boundaries(grad, min_distance=15, max_distance=100, threshold=0.5)
        assert len(peaks) == 11
        _assert_near_multiples(peaks, 40)

        boundaries_mm, _ = profile_to_ring_positions(_profile_sample(candidate), sensitivity=0.5, min_width_mm=0.2)
        assert len(boundaries_mm) == 11
        _assert_near_multiples(boundaries_mm / (25.4 / 1200), 40)


def test_viewer_auto_detect_cutoff_is_a_fifth_of_the_strongest_boundary():
    from dendro.imaging.viewer import MeasurementSession, MeasurementViewer

    # Gradient peaks relative to the strongest: 1.0, 0.24 and 0.16. The viewer's
    # cutoff is 0.2 of the strongest (sensitivity 0.7), so it keeps the first two
    # classes only. Sensitivity 0.6 would lose the 0.24 class; 0.8 would add the 0.16 class.
    jumps = [160.0, 40.0, 30.0] * 4
    profile = _graded_boundary_profile(jumps)

    viewer = MeasurementViewer.__new__(MeasurementViewer)
    viewer.profile = profile
    viewer.dpi = 1200
    viewer.session = MeasurementSession(image_path="synthetic", dpi=1200, image_shape=(1, len(profile)))
    viewer._update_ring_display = lambda: None
    viewer._auto_detect_rings()

    expected = [40 * (i + 1) for i, jump in enumerate(jumps[:-1]) if jump >= 40.0]
    found = viewer.session.ring_boundaries_px
    assert len(found) == len(expected) == 8
    np.testing.assert_allclose(found, expected, atol=2)


def test_to_uint8_scales_16bit_by_value_range():
    to_uint8 = ring_detector._to_uint8

    # uint8 input is returned unchanged.
    gray_u8 = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert to_uint8(gray_u8) is gray_u8

    # 12-bit data in a uint16 container: two levels 127 counts apart span the full output range.
    narrow = np.tile(np.array([2048, 2175], dtype=np.uint16), (64, 32))
    out = to_uint8(narrow)
    assert out.dtype == np.uint8
    assert np.array_equal(out, np.tile(np.array([0, 255], dtype=np.uint8), (64, 32)))

    # A single hot pixel must not flatten everything else.
    hot = narrow.copy()
    hot[0, 0] = 65535
    out_hot = to_uint8(hot)
    assert out_hot[0, 0] == 255
    assert np.array_equal(out_hot.ravel()[1:], out.ravel()[1:])

    # A 12-bit ramp keeps all 256 output levels (a plain >> 8 keeps 16).
    ramp_12 = np.tile(np.arange(4096, dtype=np.uint16), (4, 1))
    out_12 = to_uint8(ramp_12)
    assert (out_12[0, 0], out_12[0, -1]) == (0, 255)
    assert np.all(np.diff(out_12[0].astype(int)) >= 0)
    assert len(np.unique(out_12)) == 256
    assert np.max(np.abs(out_12[0].astype(float) - np.arange(4096) / 4095 * 255)) <= 2.0

    # Full-range 16-bit data still maps monotonically onto 0..255.
    full = np.tile(np.linspace(0, 65535, 4096).astype(np.uint16), (4, 1))
    out_full = to_uint8(full)
    assert (out_full[0, 0], out_full[0, -1]) == (0, 255)
    assert np.all(np.diff(out_full[0].astype(int)) >= 0)
    assert np.max(np.abs(out_full[0].astype(float) - full[0] / 65535 * 255)) <= 2.0


def test_enhance_contrast_handles_16bit_without_wraparound():
    # 1000 and 3000 wrap to 232 and 184 under a bare uint8 cast, which inverts the image.
    two_level = np.empty((64, 64), dtype=np.uint16)
    two_level[:, :32] = 1000
    two_level[:, 32:] = 3000
    enhanced = _enhance_contrast(two_level)
    assert enhanced.dtype == np.uint8
    assert enhanced[:, 40:].mean() > enhanced[:, :24].mean()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -0.5])
def test_widths_to_tucson_rejects_non_finite_or_negative_widths(bad: float):
    with pytest.raises(ValueError):
        widths_to_tucson(np.array([1.0, bad, 2.0]), series_id="BAD01", end_year=1902)


def test_widths_to_tucson_rounds_and_handles_999_marker(tmp_path: Path):
    # 1.236 mm should round to 124 (not truncate to 123).
    out1 = tmp_path / "rounded.rwl"
    widths_to_tucson(np.array([1.236, 1.234, 2.505]), series_id="RND01", end_year=1902, output_path=out1)
    rwl1 = parse_rwl_file(out1)
    assert list(rwl1.series["RND01"].values) == [124.0, 123.0, 250.0]

    # 9.99 mm would collide with 999 stop marker in 0.01mm precision -> must auto-switch to -9999.
    out2 = tmp_path / "nine99.rwl"
    widths_to_tucson(np.array([1.50, 9.99, 2.10]), series_id="BIG01", end_year=1902, output_path=out2)
    rwl2 = parse_rwl_file(out2)
    assert rwl2.series["BIG01"].start_year == 1900
    assert rwl2.series["BIG01"].end_year == 1902
    assert list(rwl2.series["BIG01"].values) == [1500.0, 9990.0, 2100.0]


def test_chronology_index_preserves_internal_gap_and_detects_deleted_files(tmp_path: Path):
    rwl_a = tmp_path / "a.rwl"
    rwl_b = tmp_path / "b.rwl"
    # Two non-overlapping 12-year series leaving an internal gap (1812..1819)
    rwl_a.write_text(
        "SER1    1800   100   120   110   130   105   115   125   140   135   120\n"
        "SER1    1810   110   125   999\n"
        "SER2    1820   200   210   190   220   205   215   195   230   225   210\n"
        "SER2    1830   200   215   999\n"
    )
    rwl_b.write_text(
        "SER3    1800   100   110   120   130   140   130   120   110   100   115\n"
        "SER3    1810   125   135   999\n"
    )

    idx = ChronologyIndex(tmp_path)
    assert len(idx) == 2
    entry_a = next(e for e in idx.entries if Path(e.filepath).name == "a.rwl")
    assert entry_a.master is not None
    # Span 1800..1831 = 32 years, with 8 internal NaN gap years preserved!
    assert entry_a.master.start_year == 1800
    assert entry_a.master.end_year == 1831
    assert len(entry_a.master.values) == 32
    arr = entry_a.master_values
    assert int(np.isnan(arr).sum()) == 8

    # Deleting rwl_b must invalidate the manifest on next load_or_build!
    rwl_b.unlink()
    idx2 = ChronologyIndex(tmp_path)
    assert len(idx2) == 1
