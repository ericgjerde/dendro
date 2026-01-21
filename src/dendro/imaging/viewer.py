"""
Interactive measurement viewer for ring width extraction.

This module provides a matplotlib-based interactive tool for:
- Viewing scanned wood samples
- Drawing measurement paths
- Manually marking or adjusting ring boundaries
- Previewing and exporting measurements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider
    from matplotlib.backend_bases import MouseEvent, KeyEvent
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class MeasurementSession:
    """State for an interactive measurement session."""

    image_path: Path
    dpi: int
    path_points: list[tuple[int, int]] = field(default_factory=list)
    ring_boundaries: list[int] = field(default_factory=list)  # Pixel positions
    ring_widths_mm: np.ndarray = field(default_factory=lambda: np.array([]))
    is_finalized: bool = False


class MeasurementViewer:
    """
    Interactive viewer for measuring ring widths from scanned images.

    Usage:
        viewer = MeasurementViewer("sample.tiff", dpi=1200)
        viewer.show()
        # User interactively marks path and rings
        widths = viewer.get_widths()
    """

    def __init__(
        self,
        image_path: str | Path,
        dpi: int = 1200,
        on_complete: Optional[Callable[[np.ndarray], None]] = None,
    ):
        """
        Initialize the measurement viewer.

        Args:
            image_path: Path to scanned image.
            dpi: Scanner resolution.
            on_complete: Callback when measurement is finalized.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required. Install with: pip install matplotlib")

        self.image_path = Path(image_path)
        self.dpi = dpi
        self.on_complete = on_complete

        self.session = MeasurementSession(
            image_path=self.image_path,
            dpi=dpi,
        )

        # UI state
        self.mode = "path"  # "path", "rings", "adjust"
        self.fig = None
        self.ax_image = None
        self.ax_profile = None

        # Plot elements
        self.image_display = None
        self.path_line = None
        self.ring_markers = []
        self.profile_line = None

    def show(self):
        """Display the interactive viewer."""
        self._load_image()
        self._setup_figure()
        self._connect_events()

        plt.show()

    def _load_image(self):
        """Load the image file."""
        try:
            import cv2
            img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load: {self.image_path}")
            # Convert BGR to RGB for matplotlib
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except ImportError:
            # Fallback to PIL
            from PIL import Image
            self.image = np.array(Image.open(self.image_path))

    def _setup_figure(self):
        """Set up the matplotlib figure and axes."""
        self.fig = plt.figure(figsize=(14, 8))

        # Main image view
        self.ax_image = self.fig.add_axes([0.05, 0.25, 0.6, 0.7])
        self.ax_image.set_title("Click to mark path (bark → pith), then press ENTER")

        self.image_display = self.ax_image.imshow(self.image)
        self.ax_image.axis('off')

        # Profile view
        self.ax_profile = self.fig.add_axes([0.7, 0.25, 0.25, 0.7])
        self.ax_profile.set_title("Intensity Profile")
        self.ax_profile.set_xlabel("Position (mm)")
        self.ax_profile.set_ylabel("Intensity")

        # Instructions
        self.ax_instructions = self.fig.add_axes([0.05, 0.02, 0.9, 0.15])
        self.ax_instructions.axis('off')
        self._update_instructions()

        # Path line
        self.path_line, = self.ax_image.plot([], [], 'r-', linewidth=2)

        # Ring marker container
        self.ring_markers = []

    def _update_instructions(self):
        """Update instruction text based on current mode."""
        instructions = {
            "path": (
                "PATH MODE\n"
                "• Left-click: Add path points (start at bark, end at pith)\n"
                "• Right-click: Remove last point\n"
                "• ENTER: Finish path and switch to ring marking mode\n"
                "• ESC: Cancel and close"
            ),
            "rings": (
                "RING MODE\n"
                "• Left-click on image: Mark ring boundary\n"
                "• Right-click: Remove nearest boundary\n"
                "• A: Auto-detect rings\n"
                "• ENTER: Finalize and save measurements\n"
                "• BACKSPACE: Return to path mode"
            ),
            "adjust": (
                "ADJUST MODE\n"
                "• Drag boundaries to adjust position\n"
                "• ENTER: Accept adjustments\n"
                "• BACKSPACE: Return to ring mode"
            ),
        }

        self.ax_instructions.clear()
        self.ax_instructions.axis('off')
        self.ax_instructions.text(
            0.5, 0.5, instructions.get(self.mode, ""),
            ha='center', va='center',
            fontfamily='monospace',
            fontsize=9,
            transform=self.ax_instructions.transAxes,
        )
        self.fig.canvas.draw_idle()

    def _connect_events(self):
        """Connect matplotlib event handlers."""
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_click(self, event: "MouseEvent"):
        """Handle mouse click events."""
        if event.inaxes != self.ax_image:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.mode == "path":
            if event.button == 1:  # Left click - add point
                self.session.path_points.append((x, y))
                self._update_path_display()
            elif event.button == 3:  # Right click - remove last
                if self.session.path_points:
                    self.session.path_points.pop()
                    self._update_path_display()

        elif self.mode == "rings":
            if event.button == 1:  # Left click - add boundary
                # Convert click to position along path
                pos = self._click_to_path_position(x, y)
                if pos is not None:
                    self.session.ring_boundaries.append(pos)
                    self.session.ring_boundaries.sort()
                    self._update_ring_display()
            elif event.button == 3:  # Right click - remove nearest
                if self.session.ring_boundaries:
                    pos = self._click_to_path_position(x, y)
                    if pos is not None:
                        nearest = min(
                            self.session.ring_boundaries,
                            key=lambda b: abs(b - pos)
                        )
                        self.session.ring_boundaries.remove(nearest)
                        self._update_ring_display()

    def _on_key(self, event: "KeyEvent"):
        """Handle keyboard events."""
        if event.key == 'escape':
            plt.close(self.fig)
            return

        if self.mode == "path":
            if event.key == 'enter' and len(self.session.path_points) >= 2:
                self.mode = "rings"
                self._compute_profile()
                self._update_instructions()
                self.ax_image.set_title("Click to mark ring boundaries")
                self.fig.canvas.draw_idle()

        elif self.mode == "rings":
            if event.key == 'backspace':
                self.mode = "path"
                self._update_instructions()
                self.ax_image.set_title("Click to mark path (bark → pith)")
                self.fig.canvas.draw_idle()

            elif event.key == 'a':  # Auto-detect
                self._auto_detect_rings()

            elif event.key == 'enter':
                self._finalize()

    def _update_path_display(self):
        """Update the path line on the image."""
        if self.session.path_points:
            xs = [p[0] for p in self.session.path_points]
            ys = [p[1] for p in self.session.path_points]
            self.path_line.set_data(xs, ys)

            # Also draw markers at each point
            self.ax_image.plot(xs, ys, 'ro', markersize=4)
        else:
            self.path_line.set_data([], [])

        self.fig.canvas.draw_idle()

    def _update_ring_display(self):
        """Update ring boundary markers."""
        # Clear old markers
        for marker in self.ring_markers:
            marker.remove()
        self.ring_markers = []

        if not self.session.ring_boundaries or not hasattr(self, 'path_coords'):
            return

        # Draw boundary lines perpendicular to path
        for pos in self.session.ring_boundaries:
            if 0 <= pos < len(self.path_coords):
                x, y = self.path_coords[pos]
                marker = self.ax_image.axvline(x=x, color='yellow', alpha=0.7, linewidth=1)
                self.ring_markers.append(marker)

        # Update profile plot
        if hasattr(self, 'profile') and len(self.profile) > 0:
            self.ax_profile.clear()
            mm_per_px = 25.4 / self.dpi
            positions_mm = np.arange(len(self.profile)) * mm_per_px

            self.ax_profile.plot(positions_mm, self.profile, 'b-', linewidth=0.5)
            self.ax_profile.set_xlabel("Position (mm)")
            self.ax_profile.set_ylabel("Intensity")

            # Mark boundaries on profile
            for pos in self.session.ring_boundaries:
                pos_mm = pos * mm_per_px
                self.ax_profile.axvline(x=pos_mm, color='red', alpha=0.5)

            # Show ring count
            n_rings = len(self.session.ring_boundaries) - 1
            self.ax_profile.set_title(f"Profile ({n_rings} rings)")

        self.fig.canvas.draw_idle()

    def _compute_profile(self):
        """Compute intensity profile along the path."""
        from .path_sampler import SamplePath, sample_along_path
        from .ring_detector import preprocess_for_rings

        try:
            import cv2
            gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            gray = preprocess_for_rings(gray)
        except ImportError:
            gray = np.mean(self.image, axis=2).astype(np.uint8)

        path = SamplePath.from_points(self.session.path_points)
        sample = sample_along_path(gray, path, self.dpi)

        self.profile = sample.intensities
        self.path_coords = path.interpolate(len(self.profile))

        # Initial profile display
        self._update_ring_display()

    def _click_to_path_position(self, x: int, y: int) -> Optional[int]:
        """Convert a click position to the nearest path position."""
        if not hasattr(self, 'path_coords') or len(self.path_coords) == 0:
            return None

        # Find nearest point on path
        dists = np.sqrt(
            (self.path_coords[:, 0] - x)**2 +
            (self.path_coords[:, 1] - y)**2
        )
        return int(np.argmin(dists))

    def _auto_detect_rings(self):
        """Auto-detect ring boundaries."""
        if not hasattr(self, 'profile'):
            return

        from .ring_detector import _calculate_gradient, _find_boundaries

        gradient = _calculate_gradient(self.profile)

        mm_per_px = 25.4 / self.dpi
        min_ring_px = int(0.2 / mm_per_px)  # 0.2mm minimum

        boundaries = _find_boundaries(
            gradient,
            min_distance=min_ring_px,
            max_distance=int(10.0 / mm_per_px),
            threshold=0.4,
        )

        self.session.ring_boundaries = sorted(boundaries)
        self._update_ring_display()

    def _finalize(self):
        """Finalize measurements and calculate ring widths."""
        if len(self.session.ring_boundaries) < 2:
            print("Need at least 2 boundaries to calculate ring widths")
            return

        mm_per_px = 25.4 / self.dpi
        boundaries_mm = np.array(self.session.ring_boundaries) * mm_per_px
        widths = np.diff(boundaries_mm)

        self.session.ring_widths_mm = widths
        self.session.is_finalized = True

        print(f"\nMeasurement complete!")
        print(f"Number of rings: {len(widths)}")
        print(f"Total span: {boundaries_mm[-1] - boundaries_mm[0]:.1f} mm")
        print(f"Mean ring width: {np.mean(widths):.2f} mm")
        print(f"Min/Max: {np.min(widths):.2f} / {np.max(widths):.2f} mm")

        if self.on_complete:
            self.on_complete(widths)

        plt.close(self.fig)

    def get_widths(self) -> np.ndarray:
        """Get the measured ring widths."""
        return self.session.ring_widths_mm

    def get_session(self) -> MeasurementSession:
        """Get the full measurement session data."""
        return self.session


def interactive_measure(
    image_path: str | Path,
    dpi: int = 1200,
) -> np.ndarray:
    """
    Convenience function for interactive measurement.

    Args:
        image_path: Path to scanned image.
        dpi: Scanner resolution.

    Returns:
        Array of ring widths in mm.
    """
    result = []

    def on_complete(widths):
        result.append(widths)

    viewer = MeasurementViewer(image_path, dpi, on_complete)
    viewer.show()

    if result:
        return result[0]
    return np.array([])
