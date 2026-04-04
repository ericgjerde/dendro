"""
Interactive measurement viewer with durable session export.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent, MouseEvent

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class MeasurementSession:
    """Serializable measurement session for scan review and reuse."""

    image_path: str
    dpi: int
    image_shape: tuple[int, int]
    path_points: list[tuple[int, int]] = field(default_factory=list)
    ring_boundaries_px: list[int] = field(default_factory=list)
    ring_widths_mm_bark_to_pith: list[float] = field(default_factory=list)
    exported_widths_mm_oldest_to_newest: list[float] = field(default_factory=list)
    measured_orientation: str = "bark_to_pith"
    export_orientation: str = "oldest_to_newest"
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    is_finalized: bool = False

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["image_shape"] = list(self.image_shape)
        return payload

    def save(self, output_path: str | Path):
        Path(output_path).write_text(json.dumps(self.to_dict(), indent=2))


class MeasurementViewer:
    """Interactive viewer for building a measurement session."""

    def __init__(
        self,
        image_path: str | Path,
        dpi: int = 1200,
        on_complete: Optional[Callable[[np.ndarray], None]] = None,
        session_output: Optional[str | Path] = None,
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required. Install with: pip install matplotlib")

        self.image_path = Path(image_path)
        self.dpi = dpi
        self.on_complete = on_complete
        self.session_output = Path(session_output) if session_output else None
        self.mode = "path"
        self.fig = None
        self.ax_image = None
        self.ax_profile = None
        self.ax_instructions = None
        self.image_display = None
        self.path_line = None
        self.path_markers = None
        self.ring_marker_artists: list = []
        self.profile = np.array([])
        self.path_coords = np.empty((0, 2))

    def show(self):
        self._check_interactive_backend()
        self._load_image()
        self.session = MeasurementSession(
            image_path=str(self.image_path),
            dpi=self.dpi,
            image_shape=tuple(self.image.shape[:2]),
        )
        self._setup_figure()
        self._connect_events()
        plt.show()

    def _check_interactive_backend(self):
        backend = matplotlib.get_backend().lower()
        if backend == "agg":
            raise RuntimeError("Interactive measurement requires a GUI matplotlib backend.")
        if os.name != "nt" and "linux" in os.sys.platform and not os.environ.get("DISPLAY"):
            raise RuntimeError("Interactive measurement requires a display server on Linux.")

    def _load_image(self):
        try:
            import cv2

            img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image: {self.image_path}")
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except ImportError:
            from PIL import Image

            self.image = np.array(Image.open(self.image_path))

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.ax_image = self.fig.add_axes([0.05, 0.25, 0.6, 0.7])
        self.ax_image.set_title("Path mode: click bark to pith, then press ENTER")
        self.image_display = self.ax_image.imshow(self.image)
        self.ax_image.axis("off")

        self.ax_profile = self.fig.add_axes([0.7, 0.25, 0.25, 0.7])
        self.ax_profile.set_title("Intensity profile")
        self.ax_profile.set_xlabel("Position (mm)")
        self.ax_profile.set_ylabel("Intensity")

        self.ax_instructions = self.fig.add_axes([0.05, 0.02, 0.9, 0.15])
        self.ax_instructions.axis("off")

        self.path_line, = self.ax_image.plot([], [], "r-", linewidth=2)
        self.path_markers = self.ax_image.scatter([], [], c="red", s=18)
        self._update_instructions()

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _update_instructions(self):
        instructions = {
            "path": (
                "PATH MODE\n"
                "• Left-click adds path points from bark to pith\n"
                "• Right-click removes the last path point\n"
                "• ENTER starts ring review\n"
                "• ESC closes without saving"
            ),
            "rings": (
                "RING REVIEW MODE\n"
                "• Left-click near the path adds a boundary\n"
                "• Right-click removes the nearest boundary\n"
                "• A auto-detects boundaries from the sampled profile\n"
                "• ENTER finalizes, saves the session, and exports widths\n"
                "• BACKSPACE returns to path mode"
            ),
        }
        self.ax_instructions.clear()
        self.ax_instructions.axis("off")
        self.ax_instructions.text(
            0.5,
            0.65,
            instructions[self.mode],
            ha="center",
            va="center",
            fontsize=9,
            fontfamily="monospace",
            transform=self.ax_instructions.transAxes,
        )
        if self.session.warnings:
            self.ax_instructions.text(
                0.5,
                0.15,
                "QC: " + " | ".join(self.session.warnings),
                ha="center",
                va="center",
                fontsize=8,
                color="darkorange",
                transform=self.ax_instructions.transAxes,
            )
        self.fig.canvas.draw_idle()

    def _on_click(self, event: "MouseEvent"):
        if event.inaxes != self.ax_image or event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)
        if self.mode == "path":
            if event.button == 1:
                self.session.path_points.append((x, y))
            elif event.button == 3 and self.session.path_points:
                self.session.path_points.pop()
            self._update_path_display()
            return

        if self.mode == "rings":
            pos = self._click_to_path_position(x, y)
            if pos is None:
                return
            if event.button == 1:
                self.session.ring_boundaries_px.append(pos)
                self.session.ring_boundaries_px = sorted(set(self.session.ring_boundaries_px))
            elif event.button == 3 and self.session.ring_boundaries_px:
                nearest = min(self.session.ring_boundaries_px, key=lambda value: abs(value - pos))
                self.session.ring_boundaries_px.remove(nearest)
            self._update_ring_display()

    def _on_key(self, event: "KeyEvent"):
        if event.key == "escape":
            plt.close(self.fig)
            return

        if self.mode == "path":
            if event.key == "enter" and len(self.session.path_points) >= 2:
                self.mode = "rings"
                self.ax_image.set_title("Ring review mode: adjust boundaries, then press ENTER")
                self._compute_profile()
                self._update_instructions()
            return

        if self.mode == "rings":
            if event.key == "backspace":
                self.mode = "path"
                self.session.ring_boundaries_px = []
                self.ax_image.set_title("Path mode: click bark to pith, then press ENTER")
                self._update_ring_display()
                self._update_instructions()
                return
            if event.key == "a":
                self._auto_detect_rings()
                return
            if event.key == "enter":
                self._finalize()

    def _update_path_display(self):
        if self.session.path_points:
            xs = [point[0] for point in self.session.path_points]
            ys = [point[1] for point in self.session.path_points]
            self.path_line.set_data(xs, ys)
            self.path_markers.set_offsets(np.column_stack([xs, ys]))
        else:
            self.path_line.set_data([], [])
            self.path_markers.set_offsets(np.empty((0, 2)))
        self.fig.canvas.draw_idle()

    def _update_ring_display(self):
        for artist in self.ring_marker_artists:
            artist.remove()
        self.ring_marker_artists = []

        if self.path_coords.size and self.session.ring_boundaries_px:
            points = []
            for pos in self.session.ring_boundaries_px:
                if 0 <= pos < len(self.path_coords):
                    x, y = self.path_coords[pos]
                    points.append([x, y])
            if points:
                offsets = np.asarray(points)
                scatter = self.ax_image.scatter(
                    offsets[:, 0],
                    offsets[:, 1],
                    c="yellow",
                    edgecolors="black",
                    s=32,
                    zorder=5,
                )
                self.ring_marker_artists.append(scatter)

        self._update_profile_plot()
        self.fig.canvas.draw_idle()

    def _compute_profile(self):
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
        self.session.warnings = self._profile_warnings(sample.intensities, path.to_mm(self.dpi))
        self._update_ring_display()
        self._update_instructions()

    def _update_profile_plot(self):
        self.ax_profile.clear()
        if len(self.profile) == 0:
            self.ax_profile.set_title("Intensity profile")
            self.ax_profile.set_xlabel("Position (mm)")
            self.ax_profile.set_ylabel("Intensity")
            return

        mm_per_px = 25.4 / self.dpi
        positions_mm = np.arange(len(self.profile)) * mm_per_px
        self.ax_profile.plot(positions_mm, self.profile, "b-", linewidth=0.6)
        for pos in self.session.ring_boundaries_px:
            if 0 <= pos < len(self.profile):
                self.ax_profile.axvline(pos * mm_per_px, color="red", alpha=0.5)
        ring_count = max(0, len(self.session.ring_boundaries_px) - 1)
        self.ax_profile.set_title(f"Intensity profile ({ring_count} rings)")
        self.ax_profile.set_xlabel("Position (mm)")
        self.ax_profile.set_ylabel("Intensity")

    def _click_to_path_position(self, x: int, y: int) -> Optional[int]:
        if not self.path_coords.size:
            return None
        distances = np.sqrt((self.path_coords[:, 0] - x) ** 2 + (self.path_coords[:, 1] - y) ** 2)
        return int(np.argmin(distances))

    def _auto_detect_rings(self):
        if len(self.profile) == 0:
            return
        from .ring_detector import _calculate_gradient, _find_boundaries

        gradient = _calculate_gradient(self.profile)
        mm_per_px = 25.4 / self.dpi
        min_ring_px = max(3, int(0.2 / mm_per_px))
        boundaries = _find_boundaries(
            gradient,
            min_distance=min_ring_px,
            max_distance=max(min_ring_px + 1, int(10.0 / mm_per_px)),
            threshold=0.4,
        )
        self.session.ring_boundaries_px = sorted(set(int(value) for value in boundaries))
        self._update_ring_display()

    def _profile_warnings(self, profile: np.ndarray, path_length_mm: float) -> list[str]:
        warnings: list[str] = []
        if self.dpi < 1200:
            warnings.append("DPI below the recommended 1200 minimum")
        if path_length_mm < 20:
            warnings.append("Measurement path is short; dating confidence may be poor")
        if np.nanstd(profile) < 5:
            warnings.append("Low profile contrast may reduce ring detection quality")
        return warnings

    def _finalize(self):
        if len(self.session.ring_boundaries_px) < 2:
            self.session.warnings.append("Need at least 2 boundaries to export measurements.")
            self._update_instructions()
            return

        boundaries_px = np.asarray(self.session.ring_boundaries_px, dtype=np.float64)
        mm_per_px = 25.4 / self.dpi
        boundaries_mm = boundaries_px * mm_per_px
        widths_bark_to_pith = np.diff(boundaries_mm)
        exported_widths = widths_bark_to_pith[::-1]

        qc_warnings = list(self.session.warnings)
        if len(widths_bark_to_pith) < 30:
            qc_warnings.append("Fewer than 30 rings exported; analysis may be inconclusive")
        if np.any(widths_bark_to_pith <= 0):
            qc_warnings.append("Non-positive ring widths detected")
        if np.nanmax(widths_bark_to_pith) > 10:
            qc_warnings.append("Very wide rings detected; verify boundary placement")

        self.session.ring_widths_mm_bark_to_pith = widths_bark_to_pith.tolist()
        self.session.exported_widths_mm_oldest_to_newest = exported_widths.tolist()
        self.session.warnings = qc_warnings
        self.session.is_finalized = True

        if self.session_output:
            self.session.save(self.session_output)

        if self.on_complete:
            self.on_complete(exported_widths)

        plt.close(self.fig)

    def get_widths(self) -> np.ndarray:
        return np.asarray(self.session.exported_widths_mm_oldest_to_newest, dtype=np.float64)

    def get_session(self) -> MeasurementSession:
        return self.session


def interactive_measure(
    image_path: str | Path,
    dpi: int = 1200,
    session_output: Optional[str | Path] = None,
) -> np.ndarray:
    result: list[np.ndarray] = []

    def on_complete(widths: np.ndarray):
        result.append(widths)

    viewer = MeasurementViewer(image_path, dpi, on_complete=on_complete, session_output=session_output)
    viewer.show()
    return result[0] if result else np.array([])
