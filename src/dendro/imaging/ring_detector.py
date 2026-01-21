"""
Ring boundary detection from scanned wood cross-sections.

This module detects annual ring boundaries in high-resolution scans of
wood samples. Ring detection works by:

1. Converting to grayscale and enhancing contrast
2. Applying gradient filters to detect transitions (latewood/earlywood)
3. Finding peaks in the gradient profile along a measurement path
4. Refining peak positions for accurate ring boundary placement

For best results:
- Scan at 1200+ DPI (2400+ preferred)
- Sand surface to 400+ grit
- Ensure good lighting and no reflections
- Mark a clear radial path from pith to bark
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class RingBoundary:
    """A detected ring boundary position."""

    position: float  # Position along the measurement path (0-1 normalized)
    pixel_position: int  # Pixel position along path
    confidence: float  # Detection confidence (0-1)
    width_before: Optional[float] = None  # Width of ring before this boundary (mm)


@dataclass
class DetectionResult:
    """Result of ring boundary detection."""

    boundaries: list[RingBoundary]
    ring_widths: np.ndarray  # Ring widths in mm
    profile: np.ndarray  # Intensity profile along path
    gradient: np.ndarray  # Gradient profile used for detection
    path_length_mm: float  # Total path length in mm
    dpi: int
    warnings: list[str]


def detect_rings(
    image_path: str | Path,
    path_points: list[tuple[int, int]],
    dpi: int = 1200,
    min_ring_width_mm: float = 0.1,
    max_ring_width_mm: float = 10.0,
    sensitivity: float = 0.5,
) -> DetectionResult:
    """
    Detect ring boundaries along a specified measurement path.

    Args:
        image_path: Path to the scanned image (TIFF, PNG, JPG).
        path_points: List of (x, y) points defining the measurement path
                    from bark (outer) to pith (inner).
        dpi: Scanner resolution in dots per inch.
        min_ring_width_mm: Minimum expected ring width.
        max_ring_width_mm: Maximum expected ring width.
        sensitivity: Detection sensitivity (0-1). Higher = more boundaries.

    Returns:
        DetectionResult with boundaries, widths, and diagnostics.
    """
    if cv2 is None:
        raise ImportError("OpenCV (cv2) required for image processing. "
                         "Install with: pip install opencv-python")

    image_path = Path(image_path)

    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Enhance contrast
    gray = _enhance_contrast(gray)

    # Sample intensity along the path
    profile, path_length_px = _sample_path(gray, path_points)

    # Calculate path length in mm
    mm_per_pixel = 25.4 / dpi
    path_length_mm = path_length_px * mm_per_pixel

    # Calculate gradient (detect transitions)
    gradient = _calculate_gradient(profile)

    # Find ring boundaries (peaks in gradient)
    min_ring_px = int(min_ring_width_mm / mm_per_pixel)
    max_ring_px = int(max_ring_width_mm / mm_per_pixel)

    boundary_positions = _find_boundaries(
        gradient,
        min_distance=min_ring_px,
        max_distance=max_ring_px,
        threshold=sensitivity,
    )

    # Convert to RingBoundary objects
    boundaries = []
    for i, pos in enumerate(boundary_positions):
        boundaries.append(RingBoundary(
            position=pos / len(profile),
            pixel_position=pos,
            confidence=_boundary_confidence(gradient, pos),
        ))

    # Calculate ring widths
    ring_widths = _calculate_widths(boundary_positions, mm_per_pixel)

    # Assign widths to boundaries
    for i, boundary in enumerate(boundaries[:-1]):
        boundary.width_before = ring_widths[i]

    # Generate warnings
    warnings = []
    if len(ring_widths) < 10:
        warnings.append(f"Only {len(ring_widths)} rings detected. "
                       "Consider adjusting sensitivity or path.")

    narrow_rings = np.sum(ring_widths < 0.2)
    if narrow_rings > len(ring_widths) * 0.3:
        warnings.append(f"{narrow_rings} very narrow rings detected. "
                       "May indicate false positives or need higher DPI.")

    return DetectionResult(
        boundaries=boundaries,
        ring_widths=ring_widths,
        profile=profile,
        gradient=gradient,
        path_length_mm=path_length_mm,
        dpi=dpi,
        warnings=warnings,
    )


def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Apply contrast enhancement for better ring visibility."""
    if cv2 is None:
        return gray

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray.astype(np.uint8))

    return enhanced


def _sample_path(
    image: np.ndarray,
    path_points: list[tuple[int, int]],
    sample_width: int = 5,
) -> tuple[np.ndarray, float]:
    """
    Sample image intensity along a path with averaging.

    Args:
        image: Grayscale image array.
        path_points: List of (x, y) points defining the path.
        sample_width: Width of sampling band (perpendicular to path).

    Returns:
        Tuple of (intensity profile, path length in pixels).
    """
    if len(path_points) < 2:
        raise ValueError("Path must have at least 2 points")

    # Interpolate path to get dense sampling
    points = np.array(path_points, dtype=np.float64)

    # Calculate cumulative distance along path
    diffs = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    total_length = np.sum(segment_lengths)

    if total_length < 10:
        raise ValueError("Path too short for meaningful analysis")

    # Resample path at regular intervals (1 pixel spacing)
    n_samples = int(total_length)
    t_values = np.linspace(0, 1, n_samples)

    # Cumulative distance normalized
    cum_dist = np.concatenate([[0], np.cumsum(segment_lengths)]) / total_length

    # Interpolate x and y coordinates
    x_interp = np.interp(t_values, cum_dist, points[:, 0])
    y_interp = np.interp(t_values, cum_dist, points[:, 1])

    # Sample intensity at each point (with perpendicular averaging)
    profile = np.zeros(n_samples)
    height, width = image.shape[:2]

    for i in range(n_samples):
        x, y = int(x_interp[i]), int(y_interp[i])

        # Ensure we're within image bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))

        # Average over a small window for noise reduction
        y1 = max(0, y - sample_width // 2)
        y2 = min(height, y + sample_width // 2 + 1)
        x1 = max(0, x - sample_width // 2)
        x2 = min(width, x + sample_width // 2 + 1)

        profile[i] = np.mean(image[y1:y2, x1:x2])

    return profile, total_length


def _calculate_gradient(profile: np.ndarray) -> np.ndarray:
    """
    Calculate gradient profile for ring detection.

    Ring boundaries appear as transitions from light (earlywood)
    to dark (latewood) or vice versa.
    """
    # Smooth to reduce noise
    kernel_size = max(3, len(profile) // 100)
    if kernel_size % 2 == 0:
        kernel_size += 1

    smoothed = np.convolve(
        profile,
        np.ones(kernel_size) / kernel_size,
        mode='same'
    )

    # Calculate gradient (first derivative)
    gradient = np.gradient(smoothed)

    # We want to detect transitions from light to dark (ring boundaries)
    # Take absolute value to catch both directions
    gradient = np.abs(gradient)

    return gradient


def _find_boundaries(
    gradient: np.ndarray,
    min_distance: int,
    max_distance: int,
    threshold: float,
) -> list[int]:
    """
    Find ring boundary positions as peaks in gradient.

    Args:
        gradient: Gradient profile.
        min_distance: Minimum pixels between boundaries.
        max_distance: Maximum pixels between boundaries.
        threshold: Detection threshold (0-1, fraction of max gradient).

    Returns:
        List of boundary positions (pixel indices).
    """
    # Normalize gradient
    grad_max = np.max(gradient)
    if grad_max == 0:
        return []

    grad_norm = gradient / grad_max

    # Find peaks above threshold
    threshold_value = threshold * 0.5  # Scale threshold

    # Simple peak finding
    peaks = []
    for i in range(min_distance, len(gradient) - min_distance):
        # Check if this is a local maximum
        window = gradient[max(0, i - min_distance // 2):
                         min(len(gradient), i + min_distance // 2 + 1)]

        if gradient[i] == np.max(window) and grad_norm[i] > threshold_value:
            # Check distance from last peak
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)

    return peaks


def _boundary_confidence(gradient: np.ndarray, position: int) -> float:
    """Calculate confidence score for a boundary detection."""
    if position < 0 or position >= len(gradient):
        return 0.0

    # Confidence based on gradient magnitude relative to local context
    window_size = min(50, len(gradient) // 10)
    start = max(0, position - window_size)
    end = min(len(gradient), position + window_size)

    local_max = np.max(gradient[start:end])
    if local_max == 0:
        return 0.0

    return float(gradient[position] / local_max)


def _calculate_widths(
    boundary_positions: list[int],
    mm_per_pixel: float,
) -> np.ndarray:
    """Calculate ring widths from boundary positions."""
    if len(boundary_positions) < 2:
        return np.array([])

    positions = np.array(boundary_positions)
    widths_px = np.diff(positions)
    widths_mm = widths_px * mm_per_pixel

    return widths_mm


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file and return as numpy array."""
    if cv2 is None:
        raise ImportError("OpenCV required. Install with: pip install opencv-python")

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {path}")

    return img


def estimate_dpi(image_path: str | Path) -> Optional[int]:
    """
    Try to read DPI from image metadata.

    Returns None if DPI cannot be determined.
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            dpi = img.info.get('dpi')
            if dpi:
                return int(dpi[0])
    except Exception:
        pass

    return None


def preprocess_for_rings(
    image: np.ndarray,
    enhance_contrast: bool = True,
    denoise: bool = True,
) -> np.ndarray:
    """
    Preprocess image for optimal ring detection.

    Args:
        image: Input image (color or grayscale).
        enhance_contrast: Apply contrast enhancement.
        denoise: Apply noise reduction.

    Returns:
        Preprocessed grayscale image.
    """
    if cv2 is None:
        raise ImportError("OpenCV required")

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Denoise
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Enhance contrast
    if enhance_contrast:
        gray = _enhance_contrast(gray)

    return gray
