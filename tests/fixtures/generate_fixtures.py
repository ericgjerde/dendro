"""
Deterministic synthetic ITRDB-style fixtures for testing the dating pipeline.

NOAA's ITRDB archive is not reachable from CI, and redistributing real
chronologies raises licensing questions, so we generate realistic,
*deterministic* tree-ring data instead. The generated series share a common
"climate" signal (an AR(1) process plus sharp marker years), each modulated by
an individual age-related growth trend and measurement noise -- exactly the
structure real cross-dating relies on.

Running this module rewrites the committed fixtures under
``tests/fixtures/reference/synth`` and a known-date sample CSV. The output is
fully determined by the seeds below, so regenerating must produce identical
files (this is asserted by ``tests/test_fixtures.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).parent
REF_DIR = FIXTURE_DIR / "reference" / "synth"
SAMPLE_DIR = FIXTURE_DIR / "samples"

# Calendar span of the synthetic "climate"
CLIMATE_START = 1500
CLIMATE_END = 2000

# Known felling year of the committed dating sample (with bark edge).
SAMPLE_FELLING_YEAR = 1789
SAMPLE_N_RINGS = 90


def make_climate_signal(seed: int = 20240101) -> np.ndarray:
    """An AR(1) climate signal with a few sharp marker years."""
    rng = np.random.default_rng(seed)
    n = CLIMATE_END - CLIMATE_START + 1
    phi = 0.3
    c = np.zeros(n)
    for i in range(1, n):
        c[i] = phi * c[i - 1] + rng.normal(0, 1)
    c = (c - c.mean()) / c.std()
    # Sharp marker years (e.g. 1816 "Year Without Summer"): very narrow rings.
    for year in (1816, 1709, 1780, 1601):
        idx = year - CLIMATE_START
        if 0 <= idx < n:
            c[idx] = -3.0
    return c


def _neg_exp_trend(n: int, a: float, b: float, c: float) -> np.ndarray:
    """Age-related growth trend: large when young, decaying with age."""
    t = np.arange(n)
    return a * np.exp(-b * t) + c


def synth_core(
    climate: np.ndarray,
    start_year: int,
    n_rings: int,
    rng: np.random.Generator,
    sensitivity: float = 0.45,
) -> np.ndarray:
    """Generate one raw ring-width series (in mm) from the shared climate."""
    s = start_year - CLIMATE_START
    sig = climate[s : s + n_rings]
    trend = _neg_exp_trend(
        n_rings,
        a=rng.uniform(1.5, 3.0),
        b=rng.uniform(0.01, 0.03),
        c=rng.uniform(0.4, 0.8),
    )
    noise = rng.normal(0, 0.12, n_rings)
    widths_mm = trend * (1.0 + sensitivity * sig + noise)
    return np.clip(widths_mm, 0.05, None)


def widths_to_tucson_lines(series_id: str, start_year: int, widths_mm: np.ndarray) -> list[str]:
    """Format a series as standard Tucson .rwl lines (0.01 mm units, 999 stop)."""
    vals = np.rint(widths_mm * 100).astype(int)
    vals = np.clip(vals, 1, 9989)  # keep away from the 999/9990 sentinels
    sid = series_id[:8].ljust(8)
    lines: list[str] = []
    year = start_year
    idx = 0
    n = len(vals)
    while idx < n:
        count = min(10 - (year % 10), n - idx)
        line = f"{sid}{year:4d}"
        for i in range(count):
            line += f"{int(vals[idx + i]):6d}"
        idx += count
        year += count
        if idx >= n:
            line += "   999"
        lines.append(line)
    return lines


def write_site(
    path: Path,
    site_code: str,
    species: str,
    climate: np.ndarray,
    cores: list[tuple[str, int, int]],
    seed: int,
) -> None:
    """Write one .rwl site file with a header and several cores."""
    rng = np.random.default_rng(seed)
    lines = [
        f"{site_code} 1 Synthetic Site {site_code} {species}",
        f"{site_code} 2 Test Region  {species}  45 00 N  72 00 W  300M",
        f"{site_code} 3 Synthetic generator  1500 2000",
    ]
    for core_id, start, length in cores:
        widths = synth_core(climate, start, length, rng)
        lines.extend(widths_to_tucson_lines(core_id, start, widths))
    path.write_text("\n".join(lines) + "\n")


def generate_all() -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    climate = make_climate_signal()

    # Two synthetic sites of eastern-hemlock-like and white-pine-like cores.
    write_site(
        REF_DIR / "synth01.rwl",
        site_code="SYNTH01",
        species="TSCA",
        climate=climate,
        cores=[
            ("SY01A", 1600, 300),
            ("SY01B", 1620, 280),
            ("SY01C", 1650, 250),
            ("SY01D", 1580, 320),
            ("SY01E", 1700, 200),
            ("SY01F", 1660, 240),
            ("SY01G", 1550, 350),
            ("SY01H", 1690, 210),
        ],
        seed=111,
    )
    write_site(
        REF_DIR / "synth02.rwl",
        site_code="SYNTH02",
        species="PIST",
        climate=climate,
        cores=[
            ("SY02A", 1610, 290),
            ("SY02B", 1630, 270),
            ("SY02C", 1640, 260),
            ("SY02D", 1600, 300),
            ("SY02E", 1680, 220),
            ("SY02F", 1655, 245),
        ],
        seed=222,
    )

    # A known-date dating sample: a core that ends (bark edge) at the felling year.
    rng = np.random.default_rng(999)
    sample_start = SAMPLE_FELLING_YEAR - SAMPLE_N_RINGS + 1
    sample_widths = synth_core(climate, sample_start, SAMPLE_N_RINGS, rng)
    # CSV is written oldest -> newest (canonical / pith -> bark orientation).
    csv_lines = ["year,width_mm"]
    for i, w in enumerate(sample_widths):
        csv_lines.append(f"{sample_start + i},{w:.3f}")
    (SAMPLE_DIR / "known_1789.csv").write_text("\n".join(csv_lines) + "\n")


if __name__ == "__main__":
    generate_all()
    print(f"Wrote fixtures to {REF_DIR} and {SAMPLE_DIR}")
