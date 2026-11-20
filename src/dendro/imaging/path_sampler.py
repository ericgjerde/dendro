"""
Path-based sampling for ring width extraction.

This module provides tools for sampling image intensity along
a user-defined measurement path and converting the results
to ring width measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SamplePath:
    """A measurement path through a wood sample image."""

    points: np.ndarray  # Nx2 array of (x, y) coordinates
    total_length_px: float
    bark_end: int = 0  # Index of bark end (usually 0)
    pith_end: int = -1  # Index of pith end (usually -1)

    @classmethod
    def from_points(cls, points: list[tuple[int, int]]) -> "SamplePath":
        """Create path from list of (x, y) points."""
        pts = np.array(points, dtype=np.float64)

        # Calculate total length
        diffs = np.diff(pts, axis=0)
        lengths = np.sqrt(np.sum(diffs**2, axis=1))
        total_length = np.sum(lengths)

        return cls(
            points=pts,
            total_length_px=total_length,
            bark_end=0,
            pith_end=len(pts) - 1,
        )

    def interpolate(self, n_samples: int) -> np.ndarray:
        """
        Interpolate path to get evenly-spaced sample points.

        Returns:
            Nx2 array of (x, y) coordinates.
        """
        # Calculate cumulative distance
        diffs = np.diff(self.points, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        cum_dist = np.concatenate([[0], np.cumsum(segment_lengths)])

        # Normalize to 0-1
        cum_dist_norm = cum_dist / cum_dist[-1]

        # Target positions
        t = np.linspace(0, 1, n_samples)

        # Interpolate x and y
        x_interp = np.interp(t, cum_dist_norm, self.points[:, 0])
        y_interp = np.interp(t, cum_dist_norm, self.points[:, 1])

        return np.column_stack([x_interp, y_interp])

    def to_mm(self, dpi: int) -> float:
        """Convert path length to millimeters."""
        return self.total_length_px * 25.4 / dpi


@dataclass
class ProfileSample:
    """Intensity profile sampled along a path."""

    intensities: np.ndarray  # 1D array of intensity values
    positions_px: np.ndarray  # Pixel positions along path
    positions_mm: np.ndarray  # Millimeter positions along path
    path: SamplePath
    dpi: int


def sample_along_path(
    image: np.ndarray,
    path: SamplePath | list[tuple[int, int]],
    dpi: int = 1200,
    sample_width: int = 5,
) -> ProfileSample:
    """
    Sample image intensity along a measurement path.

    Args:
        image: Grayscale image array.
        path: SamplePath object or list of (x, y) points.
        dpi: Image resolution in dots per inch.
        sample_width: Width of sampling band perpendicular to path.

    Returns:
        ProfileSample with intensity values and position info.
    """
    # Convert to SamplePath if needed
    if isinstance(path, list):
        path = SamplePath.from_points(path)

    # Number of samples (1 per pixel along path)
    n_samples = int(path.total_length_px)
    if n_samples < 10:
        raise ValueError("Path too short for sampling")

    # Get interpolated sample positions
    sample_points = path.interpolate(n_samples)

    # Sample intensity at each point
    intensities = np.zeros(n_samples)
    height, width = image.shape[:2]

    for i, (x, y) in enumerate(sample_points):
        x_int, y_int = int(x), int(y)

        # Clamp to image bounds
        x_int = max(0, min(x_int, width - 1))
        y_int = max(0, min(y_int, height - 1))

        # Average over perpendicular band
        half_width = sample_width // 2
        y1 = max(0, y_int - half_width)
        y2 = min(height, y_int + half_width + 1)
        x1 = max(0, x_int - half_width)
        x2 = min(width, x_int + half_width + 1)

        intensities[i] = np.mean(image[y1:y2, x1:x2])

    # Calculate positions
    mm_per_pixel = 25.4 / dpi
    positions_px = np.arange(n_samples)
    positions_mm = positions_px * mm_per_pixel

    return ProfileSample(
        intensities=intensities,
        positions_px=positions_px,
        positions_mm=positions_mm,
        path=path,
        dpi=dpi,
    )


def profile_to_ring_positions(
    profile: ProfileSample,
    sensitivity: float = 0.5,
    min_width_mm: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect ring boundaries from intensity profile.

    Args:
        profile: ProfileSample from sample_along_path.
        sensitivity: Detection sensitivity (0-1). Higher = more boundaries.
        min_width_mm: Minimum ring width to detect.

    Returns:
        Tuple of (boundary_positions_mm, ring_widths_mm).
    """
    from scipy.signal import find_peaks

    smoothed = _smooth_profile(profile.intensities)
    grad = np.gradient(smoothed)
    pos = np.maximum(grad, 0.0)
    neg = np.maximum(-grad, 0.0)
    gradient = neg if float(np.percentile(neg, 98)) > float(np.percentile(pos, 98)) else pos

    grad_max = float(np.max(gradient)) if len(gradient) else 0.0
    if grad_max <= 0.0:
        return np.array([]), np.array([])

    min_distance_px = max(3, int(round(min_width_mm / (25.4 / profile.dpi))))
    sens = float(np.clip(sensitivity, 0.05, 0.99))
    height_cutoff = ((1.0 - sens) * 0.5 + 0.05) * grad_max

    boundaries_px, _ = find_peaks(
        gradient,
        height=height_cutoff,
        prominence=height_cutoff * 0.4,
        distance=min_distance_px,
    )

    mm_per_pixel = 25.4 / profile.dpi
    boundaries_mm = np.asarray(boundaries_px, dtype=np.float64) * mm_per_pixel

    if len(boundaries_mm) >= 2:
        widths_mm = np.diff(boundaries_mm)
    else:
        widths_mm = np.array([])

    return boundaries_mm, widths_mm


def _smooth_profile(profile: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply smoothing to intensity profile."""
    if window < 3:
        return profile

    kernel = np.ones(window) / window
    return np.convolve(profile, kernel, mode='same')


def _find_peaks(
    signal: np.ndarray,
    min_distance: int,
    threshold: float,
) -> list[int]:
    """Find peaks in a signal."""
    peaks = []

    for i in range(min_distance, len(signal) - min_distance):
        # Local maximum check
        if signal[i] < threshold:
            continue

        window = signal[max(0, i - min_distance):min(len(signal), i + min_distance + 1)]
        if signal[i] == np.max(window):
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)

    return peaks


def manual_ring_widths(
    boundary_positions_mm: list[float],
) -> np.ndarray:
    """
    Calculate ring widths from manually marked boundary positions.

    Args:
        boundary_positions_mm: List of boundary positions in mm from bark.

    Returns:
        Array of ring widths in mm.
    """
    positions = np.array(sorted(boundary_positions_mm))

    if len(positions) < 2:
        return np.array([])

    return np.diff(positions)


def widths_to_tucson(
    widths_mm: np.ndarray,
    series_id: str,
    end_year: int,
    orientation: str = "oldest_to_newest",
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Convert ring widths to Tucson format for export.

    Args:
        widths_mm: Ring widths in mm.
        series_id: Series identifier (max 8 characters).
        end_year: Year of the outermost ring (bark).
        orientation: Input orientation, one of "oldest_to_newest" or "bark_to_pith".
        output_path: Optional path to write the file.

    Returns:
        Tucson format string.
    """
    # Convert mm to 0.01mm units (standard Tucson, rounded rather than truncated).
    # If any rounded measurement equals 999 (the 0.01mm stop marker), switch to
    # 0.001mm precision with -9999 stop marker so the file round-trips losslessly.
    widths_arr = np.asarray(widths_mm, dtype=np.float64)
    widths_001mm = np.rint(widths_arr * 100.0).astype(int)
    if np.any(widths_001mm == 999):
        widths_int = np.rint(widths_arr * 1000.0).astype(int)
        stop_marker_str = " -9999"
    else:
        widths_int = widths_001mm
        stop_marker_str = "   999"

    if orientation == "bark_to_pith":
        widths_int = widths_int[::-1]

    start_year = end_year - len(widths_int) + 1

    # Format series ID
    series_id = series_id[:8].ljust(8)

    # Build output lines
    lines = []

    # Process in decades
    year = start_year
    idx = 0

    while idx < len(widths_int):
        # How many values fit in this decade
        values_in_decade = min(10 - (year % 10), len(widths_int) - idx)

        # Build line
        line = f"{series_id}{year:4d}"

        for i in range(values_in_decade):
            val = widths_int[idx + i]
            line += f"{val:6d}"

        # Add stop marker at end
        if idx + values_in_decade >= len(widths_int):
            line += stop_marker_str

        lines.append(line)

        idx += values_in_decade
        year += values_in_decade

    result = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(result)

    return result


def widths_to_csv(
    widths_mm: np.ndarray,
    end_year: Optional[int] = None,
    orientation: str = "oldest_to_newest",
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Export ring widths to CSV format.

    Args:
        widths_mm: Ring widths in mm.
        end_year: Optional year of outermost ring.
        orientation: Input orientation, one of "oldest_to_newest" or "bark_to_pith".
        output_path: Optional path to write the file.

    Returns:
        CSV format string.
    """
    export_widths = np.asarray(widths_mm, dtype=np.float64)
    if orientation == "bark_to_pith":
        export_widths = export_widths[::-1]

    lines = ["ring_index,width_mm"]

    if end_year is not None:
        start_year = end_year - len(export_widths) + 1
        lines[0] = "ring_index,calendar_year,width_mm"
        for i, w in enumerate(export_widths):
            lines.append(f"{i + 1},{start_year + i},{w:.3f}")
    else:
        for i, w in enumerate(export_widths):
            lines.append(f"{i + 1},{w:.3f}")

    result = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(result)

    return result
