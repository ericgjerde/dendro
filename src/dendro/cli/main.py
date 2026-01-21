"""
Command-line interface for the dendrochronology dating tool.

Commands:
    dendro download  - Download reference chronologies from ITRDB
    dendro measure   - Extract ring widths from scanned image
    dendro date      - Cross-date a sample against references
    dendro info      - Show information about downloaded references
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np


# Default data directory
DEFAULT_DATA_DIR = Path.cwd() / "data"


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    Dendrochronology dating tool for historic timber analysis.

    Cross-date wood samples against ITRDB reference chronologies
    to determine felling years.
    """
    pass


@cli.command()
@click.option(
    "--states", "-s",
    default="me,nh,vt,ma,ct,ri,ny",
    help="State codes to download (comma-separated)."
)
@click.option(
    "--species", "-p",
    default="PIST,TSCA,QUAL,QURU",
    help="Species codes to download (comma-separated)."
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output directory for downloaded files."
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Overwrite existing files."
)
def download(states: str, species: str, output: Optional[str], overwrite: bool):
    """
    Download reference chronologies from ITRDB.

    Downloads ring-width measurements and site chronologies for
    specified states and species from NOAA's ITRDB archive.

    Example:
        dendro download --states=nh,vt,ma --species=PIST,TSCA
    """
    from ..reference.downloader import download_chronologies

    output_dir = Path(output) if output else DEFAULT_DATA_DIR / "reference"

    state_list = [s.strip().lower() for s in states.split(",")]
    species_list = [s.strip().upper() for s in species.split(",")]

    click.echo(f"Downloading chronologies for:")
    click.echo(f"  States: {', '.join(s.upper() for s in state_list)}")
    click.echo(f"  Species: {', '.join(species_list)}")
    click.echo(f"  Output: {output_dir}")
    click.echo()

    try:
        files = download_chronologies(
            output_dir=output_dir,
            states=state_list,
            species=species_list,
            overwrite=overwrite,
        )
        click.echo(f"\nDownloaded {len(files)} files successfully.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("image", type=click.Path(exists=True))
@click.option(
    "--dpi", "-d",
    type=int,
    default=1200,
    help="Scanner resolution in DPI."
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output file for measurements (CSV or RWL)."
)
@click.option(
    "--auto/--manual",
    default=False,
    help="Use automatic ring detection (default: manual)."
)
def measure(image: str, dpi: int, output: Optional[str], auto: bool):
    """
    Extract ring widths from a scanned wood sample.

    Opens an interactive viewer to mark the measurement path
    and ring boundaries. Exports measurements to CSV or Tucson format.

    Example:
        dendro measure sample.tiff --dpi=2400 --output=sample.csv
    """
    from ..imaging.viewer import MeasurementViewer
    from ..imaging.path_sampler import widths_to_csv

    image_path = Path(image)

    if output:
        output_path = Path(output)
    else:
        output_path = image_path.with_suffix(".csv")

    click.echo(f"Opening {image_path.name} for measurement...")
    click.echo(f"DPI: {dpi}")
    click.echo()
    click.echo("Instructions:")
    click.echo("  1. Click to mark path from bark (outer) to pith (center)")
    click.echo("  2. Press ENTER to switch to ring marking mode")
    click.echo("  3. Click to mark ring boundaries (or press 'A' for auto-detect)")
    click.echo("  4. Press ENTER to save and close")
    click.echo()

    widths = None

    def on_complete(w):
        nonlocal widths
        widths = w

    try:
        viewer = MeasurementViewer(image_path, dpi, on_complete)
        viewer.show()

        if widths is not None and len(widths) > 0:
            csv_output = widths_to_csv(widths, output_path=output_path)
            click.echo(f"\nSaved {len(widths)} ring measurements to {output_path}")
        else:
            click.echo("No measurements recorded.")

    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Install required packages: pip install matplotlib opencv-python")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("measurements", type=click.Path(exists=True), nargs=-1, required=True)
@click.option(
    "--reference", "-r",
    type=click.Path(exists=True),
    default=None,
    help="Directory containing reference chronologies."
)
@click.option(
    "--era-start",
    type=int,
    default=1600,
    help="Earliest possible felling year."
)
@click.option(
    "--era-end",
    type=int,
    default=1900,
    help="Latest possible felling year."
)
@click.option(
    "--species", "-p",
    default=None,
    help="Species codes to match against (comma-separated)."
)
@click.option(
    "--states", "-s",
    default=None,
    help="State codes to match against (comma-separated)."
)
@click.option(
    "--bark-edge/--no-bark-edge",
    default=True,
    help="Sample includes bark edge (for exact felling year)."
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output file for results (JSON)."
)
@click.option(
    "--top", "-n",
    type=int,
    default=10,
    help="Number of top matches to display."
)
@click.option(
    "--plot", "-P",
    is_flag=True,
    default=False,
    help="Generate diagnostic plots (saves PNG alongside output)."
)
@click.option(
    "--cross-verify", "-X",
    is_flag=True,
    default=False,
    help="Cross-verify multiple samples for consistency."
)
@click.option(
    "--segments/--no-segments",
    default=True,
    help="Show segment-by-segment correlation analysis."
)
@click.option(
    "--markers/--no-markers",
    default=True,
    help="Detect and display marker year matches."
)
def date(
    measurements: tuple[str, ...],
    reference: Optional[str],
    era_start: int,
    era_end: int,
    species: Optional[str],
    states: Optional[str],
    bark_edge: bool,
    output: Optional[str],
    top: int,
    plot: bool,
    cross_verify: bool,
    segments: bool,
    markers: bool,
):
    """
    Cross-date sample(s) against reference chronologies.

    Takes ring width measurements (CSV with 'width' column) and
    matches against downloaded ITRDB chronologies to determine
    the felling year.

    Single sample mode:
        dendro date sample.csv --era-start=1750 --era-end=1850

    Multi-sample verification mode (recommended for confidence):
        dendro date beam1.csv beam2.csv beam3.csv --cross-verify
    """
    import pandas as pd
    from ..crossdating.matcher import CrossdateMatcher
    from ..crossdating.detrend import DetrendMethod, detrend_series, standardize
    from ..crossdating.correlator import sliding_correlation

    reference_dir = Path(reference) if reference else DEFAULT_DATA_DIR / "reference"

    # Parse filters
    species_filter = [s.strip().upper() for s in species.split(",")] if species else None
    state_filter = [s.strip().upper() for s in states.split(",")] if states else None

    # Load all measurement files
    all_samples = []
    for mpath in measurements:
        measurements_path = Path(mpath)
        try:
            if measurements_path.suffix.lower() == ".csv":
                df = pd.read_csv(measurements_path)
                if "width" in df.columns:
                    values = df["width"].values
                elif "width_mm" in df.columns:
                    values = df["width_mm"].values
                else:
                    values = df.select_dtypes(include=[np.number]).iloc[:, -1].values
            else:
                values = np.loadtxt(measurements_path)
            all_samples.append((measurements_path, values))
        except Exception as e:
            click.echo(f"Error loading {measurements_path}: {e}", err=True)
            sys.exit(1)

    # Build matcher
    try:
        matcher = CrossdateMatcher(reference_dir=reference_dir)
        if len(matcher.index) == 0:
            click.echo("No reference chronologies found. Run 'dendro download' first.", err=True)
            sys.exit(1)
        click.echo(f"Loaded {len(matcher.index)} reference chronologies")
    except Exception as e:
        click.echo(f"Error loading references: {e}", err=True)
        sys.exit(1)

    # Multi-sample cross-verification mode
    if cross_verify and len(all_samples) > 1:
        _run_cross_verify(
            samples=all_samples,
            matcher=matcher,
            species_filter=species_filter,
            state_filter=state_filter,
            era_start=era_start,
            era_end=era_end,
            bark_edge=bark_edge,
            output=output,
            plot=plot,
        )
        return

    # Single sample mode (or first sample if not cross-verify)
    measurements_path, values = all_samples[0]

    click.echo(f"\nLoaded {len(values)} ring measurements from {measurements_path.name}")
    click.echo(f"Reference directory: {reference_dir}")
    click.echo(f"Search era: {era_start}-{era_end}")
    click.echo(f"Bark edge: {'Yes' if bark_edge else 'No'}")
    click.echo()

    click.echo("Cross-dating...")

    report = matcher.date_sample(
        values=values,
        sample_name=measurements_path.stem,
        has_bark_edge=bark_edge,
        species_filter=species_filter,
        state_filter=state_filter,
        era_start=era_start,
        era_end=era_end,
    )

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("CROSS-DATING RESULTS")
    click.echo("=" * 60)
    click.echo(f"Sample: {report.sample_name}")
    click.echo(f"Length: {report.sample_length} rings")
    click.echo(f"Bark edge: {'Yes' if report.has_bark_edge else 'No'}")
    click.echo()

    if report.consensus_year:
        click.echo(f"PROPOSED FELLING YEAR: {report.consensus_year}")
        click.echo(f"Confidence: {report.consensus_confidence}")
    else:
        click.echo("No confident date could be determined.")

    # Add stricter warnings for borderline t-values
    if report.matches:
        best = report.matches[0]
        if 4.0 <= best.t_value < 5.0:
            report.warnings.append(
                f"CAUTION: Best t-value ({best.t_value:.1f}) is in borderline range (4-5). "
                "This commonly produces spurious matches. Seek additional verification."
            )
        if best.t_value < 6.0 and report.consensus_confidence == "MEDIUM":
            report.warnings.append(
                "Professional standards recommend t≥6 for publication. "
                "Consider longer sample or additional references."
            )

    if report.warnings:
        click.echo("\nWarnings:")
        for w in report.warnings:
            click.echo(f"  ⚠ {w}")

    if report.matches:
        click.echo(f"\nTop {min(top, len(report.matches))} matches:")
        click.echo("-" * 60)

        for i, m in enumerate(report.matches[:top], 1):
            click.echo(f"{i}. {m.reference_name} ({m.reference_species}, {m.reference_state})")
            click.echo(f"   Felling year: {m.felling_year}")
            click.echo(f"   Correlation: {m.correlation:.3f}, T-value: {m.t_value:.1f}")
            click.echo(f"   Overlap: {m.overlap} years, Confidence: {m.confidence}")
            click.echo()

        # Show segment analysis for best match
        if segments and report.matches[0].segment_correlations:
            _display_segment_analysis(report.matches[0], report.consensus_year)

        # Detect and display marker years
        if markers and report.consensus_year:
            _display_marker_years(values, report.consensus_year)

    # Save results if requested
    if output:
        output_path = Path(output)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        click.echo(f"\nResults saved to {output_path}")

    # Generate plots if requested
    if plot and report.matches:
        _generate_plots(
            report=report,
            values=values,
            matcher=matcher,
            output_path=Path(output) if output else measurements_path.with_suffix(".png"),
        )


@cli.command()
@click.option(
    "--reference", "-r",
    type=click.Path(exists=True),
    default=None,
    help="Directory containing reference chronologies."
)
def info(reference: Optional[str]):
    """
    Show information about downloaded reference chronologies.

    Displays statistics about available chronologies including
    species, states, and time coverage.

    Example:
        dendro info
    """
    from ..reference.chronology_index import ChronologyIndex

    reference_dir = Path(reference) if reference else DEFAULT_DATA_DIR / "reference"

    if not reference_dir.exists():
        click.echo(f"Reference directory not found: {reference_dir}")
        click.echo("Run 'dendro download' to download chronologies.")
        return

    click.echo(f"Scanning {reference_dir}...")
    index = ChronologyIndex(reference_dir)

    if len(index) == 0:
        click.echo("No chronologies found.")
        click.echo("Run 'dendro download' to download chronologies.")
        return

    click.echo(f"\nFound {len(index)} chronology files")
    click.echo()

    # Species breakdown
    species = index.get_species()
    click.echo("Species:")
    for sp in species:
        count = len([e for e in index.entries if e.species == sp])
        click.echo(f"  {sp}: {count} files")

    # State breakdown
    states = index.get_states()
    click.echo("\nStates:")
    for st in states:
        count = len([e for e in index.entries if e.state == st])
        click.echo(f"  {st}: {count} files")

    # Time coverage
    all_starts = [e.start_year for e in index.entries if e.start_year > 0]
    all_ends = [e.end_year for e in index.entries if e.end_year > 0]

    if all_starts and all_ends:
        click.echo(f"\nTime coverage: {min(all_starts)} - {max(all_ends)}")

    # Late 1700s coverage (relevant for historic NH houses)
    late_1700s = [
        e for e in index.entries
        if e.start_year <= 1780 and e.end_year >= 1800
    ]
    click.echo(f"\nChronologies covering 1780-1800: {len(late_1700s)}")


@cli.command()
@click.argument("rwl_file", type=click.Path(exists=True))
def parse(rwl_file: str):
    """
    Parse and display contents of a Tucson format file.

    Useful for inspecting downloaded reference chronologies.

    Example:
        dendro parse data/reference/nh/nh001.rwl
    """
    from ..reference.tucson_parser import parse_rwl_file, parse_crn_file

    filepath = Path(rwl_file)

    try:
        if ".crn" in filepath.name.lower() or "-crn-" in filepath.name.lower():
            chron = parse_crn_file(filepath)
            if chron:
                click.echo(f"Site: {chron.site_id}")
                click.echo(f"Species: {chron.species}")
                click.echo(f"Years: {chron.start_year} - {chron.end_year} ({chron.length} years)")
                if chron.latitude and chron.longitude:
                    click.echo(f"Location: {chron.latitude:.2f}°N, {abs(chron.longitude):.2f}°W")
            else:
                click.echo("Could not parse CRN file.")
        else:
            rwl = parse_rwl_file(filepath)
            click.echo(f"File: {filepath.name}")
            click.echo(f"Series: {len(rwl.series)}")
            click.echo()

            for series_id, series in list(rwl.series.items())[:10]:
                click.echo(f"  {series_id}: {series.start_year}-{series.end_year} "
                          f"({series.length} years)")

            if len(rwl.series) > 10:
                click.echo(f"  ... and {len(rwl.series) - 10} more series")

    except Exception as e:
        click.echo(f"Error parsing file: {e}", err=True)
        sys.exit(1)


def _display_segment_analysis(match, consensus_year: int):
    """Display segment-by-segment correlation analysis."""
    click.echo("\nSegment Analysis (50-year windows):")
    click.echo("-" * 60)

    seg_corrs = match.segment_correlations
    if not seg_corrs:
        click.echo("  (Sample too short for segment analysis)")
        return

    weak_segments = []
    for start_idx, r, t in seg_corrs:
        year = match.proposed_start_year + start_idx
        year_end = year + 50

        # Determine quality indicator
        if t >= 6.0:
            quality = "+++"
        elif t >= 4.0:
            quality = "++ "
        elif t >= 3.5:
            quality = "+  "
        else:
            quality = "!  "
            weak_segments.append((year, year_end, r, t))

        click.echo(f"  {quality} {year}-{year_end}: r={r:.3f}, t={t:.1f}")

    if weak_segments:
        click.echo("\n  ⚠ Weak segments detected (t<3.5):")
        for year, year_end, r, t in weak_segments:
            click.echo(f"    {year}-{year_end}: Check measurements in this region")


def _display_marker_years(values: np.ndarray, proposed_start_year: int):
    """Detect and display potential marker year matches."""
    from ..visualization.plots import detect_marker_years, identify_known_markers, MARKER_YEARS

    click.echo("\nMarker Year Analysis:")
    click.echo("-" * 60)

    # Detect anomalous rings
    anomalies = detect_marker_years(values, threshold_sigma=2.0)

    if not anomalies:
        click.echo("  No strongly anomalous rings detected.")
        return

    # Check against known marker years
    matches = identify_known_markers(anomalies, proposed_start_year)

    if matches:
        click.echo("  Known climate events matched:")
        for year, sample_desc, known_event in matches:
            click.echo(f"  ✓ {year}: {known_event}")
            click.echo(f"    (Sample shows {sample_desc})")
        click.echo("\n  Marker year alignment strengthens confidence in dating.")
    else:
        click.echo("  Anomalous rings detected but no known markers matched:")
        for idx, z_score, desc in anomalies[:5]:
            year = proposed_start_year + idx
            click.echo(f"    Ring at year {year}: {desc} (z={z_score:.1f})")

    # Also show which marker years should be in the sample
    sample_end = proposed_start_year + len(values) - 1
    expected_markers = [
        (y, desc) for y, desc in MARKER_YEARS.items()
        if proposed_start_year <= y <= sample_end
    ]

    if expected_markers:
        click.echo(f"\n  Expected marker years in sample range ({proposed_start_year}-{sample_end}):")
        for year, desc in sorted(expected_markers):
            idx = year - proposed_start_year
            if 0 <= idx < len(values):
                ring_val = values[idx]
                mean = np.mean(values)
                std = np.std(values)
                z = (ring_val - mean) / std if std > 0 else 0
                status = "✓ narrow" if z < -1.5 else "? not anomalous"
                click.echo(f"    {year}: {desc}")
                click.echo(f"       Ring value z-score: {z:.1f} ({status})")


def _run_cross_verify(
    samples: list[tuple[Path, np.ndarray]],
    matcher,
    species_filter,
    state_filter,
    era_start: int,
    era_end: int,
    bark_edge: bool,
    output: Optional[str],
    plot: bool,
):
    """Run cross-verification across multiple samples."""
    click.echo("\n" + "=" * 60)
    click.echo("MULTI-SAMPLE CROSS-VERIFICATION")
    click.echo("=" * 60)
    click.echo(f"Analyzing {len(samples)} samples for consistency...")
    click.echo()

    results = []
    for path, values in samples:
        report = matcher.date_sample(
            values=values,
            sample_name=path.stem,
            has_bark_edge=bark_edge,
            species_filter=species_filter,
            state_filter=state_filter,
            era_start=era_start,
            era_end=era_end,
        )
        results.append((path, values, report))

        # Brief summary per sample
        if report.consensus_year:
            click.echo(f"  {path.name}: {report.consensus_year} ({report.consensus_confidence})")
        else:
            click.echo(f"  {path.name}: No confident date")

    click.echo()

    # Check for consensus across samples
    dated_results = [(p, v, r) for p, v, r in results if r.consensus_year]

    if len(dated_results) < 2:
        click.echo("⚠ Insufficient samples with confident dates for cross-verification.")
        return

    # Extract years from each sample
    years = [r.consensus_year for _, _, r in dated_results]
    unique_years = set(years)

    click.echo("CROSS-VERIFICATION RESULTS")
    click.echo("-" * 60)

    if len(unique_years) == 1:
        consensus_year = years[0]
        click.echo(f"✓ STRONG AGREEMENT: All {len(dated_results)} samples date to {consensus_year}")
        click.echo(f"  This significantly increases confidence in the dating.")

        # Calculate combined statistics
        total_t = sum(r.matches[0].t_value if r.matches else 0 for _, _, r in dated_results)
        avg_t = total_t / len(dated_results)
        click.echo(f"\n  Combined statistics:")
        click.echo(f"    Samples agreeing: {len(dated_results)}/{len(results)}")
        click.echo(f"    Average t-value: {avg_t:.1f}")

    else:
        click.echo("⚠ DISAGREEMENT DETECTED")
        click.echo("  Samples propose different felling years:")
        for year in sorted(unique_years):
            agreeing = [p.name for p, _, r in dated_results if r.consensus_year == year]
            click.echo(f"    {year}: {', '.join(agreeing)}")

        click.echo("\n  Possible causes:")
        click.echo("    - Samples from different trees/buildings")
        click.echo("    - Missing rings in one or more samples")
        click.echo("    - Measurement errors")
        click.echo("    - One or more spurious matches")

    # Geographic diversity check
    if dated_results:
        all_states = set()
        for _, _, r in dated_results:
            if r.matches:
                for m in r.matches[:3]:
                    all_states.add(m.reference_state)

        if len(all_states) >= 3:
            click.echo(f"\n✓ Geographic diversity: References from {len(all_states)} states agree")
            click.echo(f"  States: {', '.join(sorted(all_states))}")

    # Save combined results if requested
    if output:
        combined = {
            "samples": [
                {
                    "name": p.name,
                    "length": len(v),
                    "consensus_year": r.consensus_year,
                    "confidence": r.consensus_confidence,
                    "top_match_t": r.matches[0].t_value if r.matches else None,
                }
                for p, v, r in results
            ],
            "cross_verification": {
                "samples_agreeing": len([y for y in years if y == max(set(years), key=years.count)]),
                "total_samples": len(results),
                "unique_years_proposed": list(unique_years),
                "agreement": len(unique_years) == 1,
            }
        }
        output_path = Path(output)
        with open(output_path, "w") as f:
            json.dump(combined, f, indent=2)
        click.echo(f"\nCombined results saved to {output_path}")


def _generate_plots(report, values: np.ndarray, matcher, output_path: Path):
    """Generate and save diagnostic plots."""
    from ..crossdating.detrend import detrend_series, standardize
    from ..crossdating.correlator import sliding_correlation
    from ..visualization.plots import save_diagnostic_plots

    if not report.matches:
        click.echo("No matches to plot.")
        return

    best_match = report.matches[0]

    # Get the reference data
    candidates = matcher.index.search(
        species=None,
        states=None,
        min_year=best_match.proposed_start_year - 100,
        max_year=best_match.proposed_end_year + 100,
        min_overlap=30,
    )

    # Find the matching reference
    ref_data = None
    ref_start = None

    for meta in candidates:
        if meta.site_name == best_match.reference_name:
            data = matcher.index.load_chronology(meta)
            if data is not None:
                from ..reference.tucson_parser import Chronology, RWLFile

                if isinstance(data, Chronology):
                    ref_values = data.values
                    ref_start = data.start_year
                    if np.mean(ref_values) > 500:
                        ref_data = ref_values / 1000.0
                    else:
                        ref_data = standardize(ref_values)
                elif isinstance(data, RWLFile):
                    df = data.to_dataframe()
                    ref_values = df.mean(axis=1).values
                    ref_start = int(df.index.min())
                    try:
                        detrended, _ = detrend_series(ref_values)
                        ref_data = standardize(detrended)
                    except Exception:
                        ref_data = standardize(ref_values)
                break

    if ref_data is None or ref_start is None:
        click.echo("Could not load reference for plotting.")
        return

    # Standardize sample
    try:
        detrended, _ = detrend_series(values)
        sample_std = standardize(detrended)
    except Exception:
        sample_std = standardize(values)

    # Get correlation profile
    all_correlations = sliding_correlation(
        sample_std, ref_data, ref_start, min_overlap=30
    )

    # Generate plots
    plot_path = output_path.with_suffix(".png")

    save_diagnostic_plots(
        report=report,
        sample=sample_std,
        reference=ref_data,
        reference_start_year=ref_start,
        output_path=plot_path,
        all_correlations=all_correlations,
    )

    click.echo(f"Diagnostic plots saved to {plot_path}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
