"""Regression tests for imaging ring detection, Tucson export, and index integrity."""

from pathlib import Path
import numpy as np

from dendro.imaging.path_sampler import ProfileSample, SamplePath, profile_to_ring_positions, widths_to_tucson
from dendro.imaging.ring_detector import _calculate_gradient, _enhance_contrast, _find_boundaries
from dendro.reference.chronology_index import ChronologyIndex
from dendro.reference.tucson_parser import parse_rwl_file


def _synthetic_ring_profile(n_rings: int = 12, ring_px: int = 40) -> np.ndarray:
    """Create a realistic sawtooth ring intensity profile: gradual earlywood->latewood, sharp boundary."""
    segments = []
    for _ in range(n_rings):
        # Gradual darkening from bright earlywood (220) to dark latewood (40), then sharp jump to next ring.
        segments.append(np.linspace(220.0, 40.0, ring_px, endpoint=False))
    return np.concatenate(segments)


def test_directional_gradient_does_not_double_count_rings():
    profile = _synthetic_ring_profile(n_rings=12, ring_px=40)
    grad = _calculate_gradient(profile)
    peaks = _find_boundaries(grad, min_distance=15, max_distance=100, threshold=0.5)

    # 12 rings have 11 internal sharp latewood->earlywood boundaries (must NOT double-count ~22).
    assert 9 <= len(peaks) <= 12


def test_sensitivity_increases_detected_boundaries():
    rng = np.random.default_rng(5)
    profile = _synthetic_ring_profile(n_rings=10, ring_px=40)
    # Attenuate half of the rings so sensitivity controls how many are detected.
    for i in range(0, 10, 2):
        profile[i * 40 : (i + 1) * 40] = np.linspace(150.0, 100.0, 40, endpoint=False)
    profile += rng.normal(0.0, 2.0, size=len(profile))

    grad = _calculate_gradient(profile)
    low_peaks = _find_boundaries(grad, min_distance=15, max_distance=100, threshold=0.2)
    high_peaks = _find_boundaries(grad, min_distance=15, max_distance=100, threshold=0.85)
    assert len(high_peaks) >= len(low_peaks)


def test_profile_to_ring_positions_does_not_double_count():
    intensities = _synthetic_ring_profile(n_rings=10, ring_px=40)
    path = SamplePath.from_points([(0, 0), (len(intensities), 0)])
    dpi = 1200
    mm_per_px = 25.4 / dpi
    sample = ProfileSample(
        intensities=intensities,
        positions_px=np.arange(len(intensities)),
        positions_mm=np.arange(len(intensities)) * mm_per_px,
        path=path,
        dpi=dpi,
    )
    boundaries_mm, widths_mm = profile_to_ring_positions(sample, sensitivity=0.5, min_width_mm=0.2)
    assert 7 <= len(boundaries_mm) <= 10
    assert len(widths_mm) == len(boundaries_mm) - 1


def test_enhance_contrast_handles_16bit_without_wraparound():
    # A monotonic 16-bit ramp should stay monotonic (no uint8 modulo wraparound).
    ramp_16 = np.tile(np.linspace(0, 65535, 64, dtype=np.uint16), (64, 1))
    enhanced = _enhance_contrast(ramp_16)
    assert enhanced.dtype == np.uint8
    assert enhanced[32, -1] > enhanced[32, 0]


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
