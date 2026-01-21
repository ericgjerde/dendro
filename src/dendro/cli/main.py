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
@click.argument("measurements", type=click.Path(exists=True))
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
def date(
    measurements: str,
    reference: Optional[str],
    era_start: int,
    era_end: int,
    species: Optional[str],
    states: Optional[str],
    bark_edge: bool,
    output: Optional[str],
    top: int,
):
    """
    Cross-date a sample against reference chronologies.

    Takes ring width measurements (CSV with 'width' column) and
    matches against downloaded ITRDB chronologies to determine
    the felling year.

    Example:
        dendro date sample.csv --era-start=1750 --era-end=1850 --bark-edge
    """
    import pandas as pd
    from ..crossdating.matcher import CrossdateMatcher
    from ..crossdating.detrend import DetrendMethod

    measurements_path = Path(measurements)
    reference_dir = Path(reference) if reference else DEFAULT_DATA_DIR / "reference"

    # Load measurements
    try:
        if measurements_path.suffix.lower() == ".csv":
            df = pd.read_csv(measurements_path)
            if "width" in df.columns:
                values = df["width"].values
            elif "width_mm" in df.columns:
                values = df["width_mm"].values
            else:
                # Try last numeric column
                values = df.select_dtypes(include=[np.number]).iloc[:, -1].values
        else:
            # Try to parse as raw values
            values = np.loadtxt(measurements_path)
    except Exception as e:
        click.echo(f"Error loading measurements: {e}", err=True)
        sys.exit(1)

    click.echo(f"Loaded {len(values)} ring measurements")
    click.echo(f"Reference directory: {reference_dir}")
    click.echo(f"Search era: {era_start}-{era_end}")
    click.echo(f"Bark edge: {'Yes' if bark_edge else 'No'}")
    click.echo()

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

    # Parse filters
    species_filter = [s.strip().upper() for s in species.split(",")] if species else None
    state_filter = [s.strip().upper() for s in states.split(",")] if states else None

    # Run cross-dating
    click.echo("\nCross-dating...")

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

    if report.warnings:
        click.echo("\nWarnings:")
        for w in report.warnings:
            click.echo(f"  - {w}")

    if report.matches:
        click.echo(f"\nTop {min(top, len(report.matches))} matches:")
        click.echo("-" * 60)

        for i, m in enumerate(report.matches[:top], 1):
            click.echo(f"{i}. {m.reference_name} ({m.reference_species}, {m.reference_state})")
            click.echo(f"   Felling year: {m.felling_year}")
            click.echo(f"   Correlation: {m.correlation:.3f}, T-value: {m.t_value:.1f}")
            click.echo(f"   Overlap: {m.overlap} years, Confidence: {m.confidence}")
            click.echo()

    # Save results if requested
    if output:
        output_path = Path(output)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        click.echo(f"Results saved to {output_path}")


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


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
