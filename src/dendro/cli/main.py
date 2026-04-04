"""
Public CLI for Northeast-first assisted dendrochronology workflows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd

from ..crossdating.matcher import CrossdateMatcher, DatingCandidate, DatingReport
from ..reference.chronology_index import ChronologyIndex
from ..reference.downloader import download_chronologies
from ..reference.tucson_parser import (
    load_measurement_session,
    load_measurements_csv,
    parse_crn_file,
    parse_rwl_file,
)


DEFAULT_DATA_DIR = Path.cwd() / "data"
ORIENTATION_CHOICES = click.Choice(["auto", "oldest_to_newest", "bark_to_pith"], case_sensitive=False)


@click.group()
@click.version_option(version="0.2.0")
def cli():
    """
    Assisted Northeast dendrochronology CLI.

    This tool ranks plausible outer-ring calendar-year alignments against ITRDB
    references. It recommends a candidate only when the current policy gates are
    cleared; otherwise results remain ranked or inconclusive.
    """


@cli.command()
@click.option("--states", "-s", default="me,nh,vt,ma,ct,ri,ny", help="State codes to download.")
@click.option("--species", "-p", default=None, help="Species codes to download.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output directory.")
@click.option(
    "--file-types",
    default="rwl,crn",
    help="Comma-separated file types to fetch. Supported values: rwl, crn.",
)
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite existing files.")
def download(states: str, species: Optional[str], output: Optional[str], file_types: str, overwrite: bool):
    """Download Northeast references and NOAA sidecar metadata."""
    output_dir = Path(output) if output else DEFAULT_DATA_DIR / "reference"
    state_list = [value.strip().lower() for value in states.split(",") if value.strip()]
    species_list = [value.strip().upper() for value in species.split(",")] if species else None
    requested_file_types = [value.strip().lower() for value in file_types.split(",") if value.strip()]

    try:
        downloaded = download_chronologies(
            output_dir=output_dir,
            states=state_list,
            species=species_list,
            file_types=requested_file_types,
            overwrite=overwrite,
        )
        index = ChronologyIndex(output_dir)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1)

    click.echo(f"Downloaded {len(downloaded)} artifacts into {output_dir}")
    click.echo(f"Indexed {len(index)} reference chronologies")
    if index.get_species():
        click.echo("Species: " + ", ".join(index.get_species()))
    if index.get_states():
        click.echo("States: " + ", ".join(index.get_states()))


@cli.command()
@click.argument("image", type=click.Path(exists=True))
@click.option("--dpi", "-d", type=int, default=1200, help="Scanner resolution in DPI.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output CSV path.")
@click.option("--session-output", type=click.Path(), default=None, help="Session JSON path.")
def measure(image: str, dpi: int, output: Optional[str], session_output: Optional[str]):
    """Measure ring widths from a scan and persist a reviewable session."""
    from ..imaging.path_sampler import widths_to_csv
    from ..imaging.viewer import MeasurementViewer

    image_path = Path(image)
    output_path = Path(output) if output else image_path.with_suffix(".measurements.csv")
    session_path = Path(session_output) if session_output else output_path.with_suffix(".session.json")

    measured_widths: list[np.ndarray] = []

    def on_complete(widths: np.ndarray):
        measured_widths.append(widths)

    click.echo("Opening interactive measurement workflow...")
    click.echo("Measurement export orientation: oldest_to_newest")
    click.echo(f"CSV output: {output_path}")
    click.echo(f"Session output: {session_path}")

    try:
        viewer = MeasurementViewer(
            image_path=image_path,
            dpi=dpi,
            on_complete=on_complete,
            session_output=session_path,
        )
        viewer.show()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1)

    if not measured_widths:
        click.echo("No measurements were exported.")
        return

    widths_to_csv(
        measured_widths[0],
        orientation="oldest_to_newest",
        output_path=output_path,
    )

    warnings = []
    try:
        payload = json.loads(session_path.read_text())
        warnings = payload.get("warnings", [])
    except Exception:
        pass

    click.echo(f"Saved measurements to {output_path}")
    click.echo(f"Saved session to {session_path}")
    if warnings:
        click.echo("QC warnings:")
        for warning in warnings:
            click.echo(f"  - {warning}")


@cli.command()
@click.argument("measurements", type=click.Path(exists=True), nargs=-1, required=True)
@click.option("--reference", "-r", type=click.Path(exists=True), default=None, help="Reference directory.")
@click.option("--era-start", type=int, default=1600, help="Earliest plausible outer-ring year.")
@click.option("--era-end", type=int, default=1900, help="Latest plausible outer-ring year.")
@click.option("--species", "-p", default=None, help="Restrict ranking to species codes.")
@click.option("--states", "-s", default=None, help="Restrict ranking to state codes.")
@click.option("--bark-edge/--no-bark-edge", default=True, help="Sample includes the bark edge.")
@click.option("--orientation", type=ORIENTATION_CHOICES, default="auto", help="Input measurement orientation.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Write JSON report to this path.")
@click.option("--json", "json_output", is_flag=True, default=False, help="Print JSON report to stdout.")
@click.option("--top", "-n", type=int, default=10, help="Maximum ranked candidates to return.")
@click.option("--plot", is_flag=True, default=False, help="Save diagnostic plots alongside the JSON report.")
@click.option("--cross-verify", is_flag=True, default=False, help="Aggregate multiple sample reports.")
def date(
    measurements: tuple[str, ...],
    reference: Optional[str],
    era_start: int,
    era_end: int,
    species: Optional[str],
    states: Optional[str],
    bark_edge: bool,
    orientation: str,
    output: Optional[str],
    json_output: bool,
    top: int,
    plot: bool,
    cross_verify: bool,
):
    """Rank candidate outer-ring calendar years against Northeast references."""
    reference_dir = Path(reference) if reference else DEFAULT_DATA_DIR / "reference"
    species_filter = [value.strip().upper() for value in species.split(",")] if species else None
    state_filter = [value.strip().upper() for value in states.split(",")] if states else None

    try:
        matcher = CrossdateMatcher(reference_dir=reference_dir, allow_remote_metadata=True)
    except Exception as exc:
        click.echo(f"Error loading references: {exc}", err=True)
        raise SystemExit(1)

    if len(matcher.index) == 0:
        click.echo("No reference chronologies found. Run 'dendro download' first.", err=True)
        raise SystemExit(1)

    reports: list[DatingReport] = []
    loaded_samples: list[np.ndarray] = []
    for measurement_path in measurements:
        path = Path(measurement_path)
        try:
            values = _load_measurements(path)
        except Exception as exc:
            click.echo(f"Error loading {path}: {exc}", err=True)
            raise SystemExit(1)

        reports.append(
            matcher.date_sample(
                values=values,
                sample_name=path.stem,
                has_bark_edge=bark_edge,
                orientation=orientation,
                species_filter=species_filter,
                state_filter=state_filter,
                era_start=era_start,
                era_end=era_end,
                top_n=top,
            )
        )
        loaded_samples.append(values)

    if cross_verify and len(reports) > 1:
        payload = _cross_verify_payload(reports)
        if output:
            Path(output).write_text(json.dumps(payload, indent=2))
        if json_output:
            click.echo(json.dumps(payload, indent=2))
        else:
            _print_cross_verify(payload)
        return

    report = reports[0]
    payload = report.to_dict()

    if output:
        output_path = Path(output)
        output_path.write_text(json.dumps(payload, indent=2))
        if plot:
            _generate_plots(report, matcher, loaded_samples[0], output_path.with_suffix(".png"))
    elif plot and len(measurements) == 1:
        _generate_plots(
            report,
            matcher,
            loaded_samples[0],
            Path(measurements[0]).with_suffix(".diagnostic.png"),
        )

    if json_output:
        click.echo(json.dumps(payload, indent=2))
        return

    _print_report(report)


@cli.command()
@click.option("--reference", "-r", type=click.Path(exists=True), default=None, help="Reference directory.")
@click.option("--json", "json_output", is_flag=True, default=False, help="Print manifest summary as JSON.")
def info(reference: Optional[str], json_output: bool):
    """Show the indexed reference inventory and metadata coverage."""
    reference_dir = Path(reference) if reference else DEFAULT_DATA_DIR / "reference"
    if not reference_dir.exists():
        click.echo(f"Reference directory not found: {reference_dir}", err=True)
        raise SystemExit(1)

    index = ChronologyIndex(reference_dir, allow_remote_metadata=True)
    payload = {
        "reference_dir": str(reference_dir),
        "entries": len(index),
        "species": {species: len([entry for entry in index.entries if entry.species == species]) for species in index.get_species()},
        "states": {state: len([entry for entry in index.entries if entry.state == state]) for state in index.get_states()},
        "file_types": {
            file_type: len([entry for entry in index.entries if entry.file_type == file_type])
            for file_type in sorted({entry.file_type for entry in index.entries})
        },
        "missing_species": len([entry for entry in index.entries if not entry.species]),
    }

    if json_output:
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo(f"Reference directory: {reference_dir}")
    click.echo(f"Indexed entries: {payload['entries']}")
    click.echo("File types:")
    for file_type, count in payload["file_types"].items():
        click.echo(f"  {file_type}: {count}")
    click.echo("Species:")
    if payload["species"]:
        for species_name, count in payload["species"].items():
            click.echo(f"  {species_name}: {count}")
    else:
        click.echo("  (none indexed)")
    click.echo("States:")
    for state_name, count in payload["states"].items():
        click.echo(f"  {state_name}: {count}")
    click.echo(f"Entries missing species metadata: {payload['missing_species']}")


@cli.command()
@click.argument("reference_file", type=click.Path(exists=True))
def parse(reference_file: str):
    """Parse a Tucson or measurement file and print a concise summary."""
    filepath = Path(reference_file)
    try:
        if filepath.suffix.lower() == ".crn":
            chronology = parse_crn_file(filepath)
            if chronology is None:
                click.echo("Could not parse chronology file.", err=True)
                raise SystemExit(1)
            click.echo(f"Site: {chronology.site_id}")
            click.echo(f"Species: {chronology.species or '(unknown)'}")
            click.echo(f"Years: {chronology.start_year}-{chronology.end_year}")
            click.echo(f"Depth entries: {len(chronology.sample_depth)}")
            return

        if filepath.suffix.lower() == ".rwl":
            rwl = parse_rwl_file(filepath)
            click.echo(f"File: {filepath.name}")
            click.echo(f"Series: {len(rwl.series)}")
            for series_id, series in list(rwl.series.items())[:10]:
                click.echo(f"  {series_id}: {series.start_year}-{series.end_year} ({series.length} rings)")
            if len(rwl.series) > 10:
                click.echo(f"  ... and {len(rwl.series) - 10} more series")
            return

        if filepath.suffix.lower() == ".json":
            df = load_measurement_session(filepath)
        else:
            df = load_measurements_csv(filepath)
        click.echo(f"Rows: {len(df)}")
        click.echo(f"Columns: {', '.join(df.columns)}")
        click.echo(df.head(5).to_string(index=False))
    except Exception as exc:
        click.echo(f"Error parsing file: {exc}", err=True)
        raise SystemExit(1)


def _load_measurements(filepath: Path) -> np.ndarray:
    if filepath.suffix.lower() == ".json":
        df = load_measurement_session(filepath)
    elif filepath.suffix.lower() == ".csv":
        df = load_measurements_csv(filepath)
    else:
        values = np.loadtxt(filepath)
        return np.asarray(values, dtype=np.float64)

    for preferred in ("width", "width_mm"):
        if preferred in df.columns:
            return np.asarray(df[preferred].values, dtype=np.float64)
    raise ValueError("Could not locate a width column in the measurement file")


def _print_report(report: DatingReport):
    click.echo("=" * 60)
    click.echo("ASSISTED DATING REPORT")
    click.echo("=" * 60)
    click.echo(f"Sample: {report.sample_name}")
    click.echo(f"Rings: {report.sample_length}")
    click.echo(f"Status: {report.status.upper()}")
    click.echo(f"Chosen input orientation: {report.chosen_orientation}")
    click.echo(f"Policy: {report.policy_version}")
    click.echo()

    if report.best_candidate is None:
        click.echo("No candidate alignments were produced.")
    else:
        best = report.best_candidate
        if report.status == "recommended":
            label = "Recommended possible felling year" if report.bark_edge else "Recommended outer-ring year"
        else:
            label = "Top candidate outer-ring year"
        click.echo(f"{label}: {best.outer_ring_year}")
        click.echo(f"Reference: {best.reference_name} ({best.reference_species or 'unknown species'}, {best.reference_state or 'unknown state'})")
        click.echo(f"Correlation: {best.correlation:.3f}")
        click.echo(f"T-value: {best.t_value:.2f}")
        click.echo(f"Composite score: {best.composite_score:.3f}")
        click.echo()

    if report.warnings:
        click.echo("Warnings:")
        for warning in report.warnings:
            click.echo(f"  - {warning}")
        click.echo()

    if report.candidates:
        click.echo("Ranked candidates:")
        for index, candidate in enumerate(report.candidates, 1):
            click.echo(
                f"{index}. {candidate.reference_name} "
                f"[{candidate.reference_species or 'unknown'} {candidate.reference_state or '??'}] "
                f"outer={candidate.outer_ring_year} "
                f"score={candidate.composite_score:.3f} "
                f"r={candidate.correlation:.3f} "
                f"t={candidate.t_value:.2f}"
            )


def _cross_verify_payload(reports: list[DatingReport]) -> dict:
    statuses = [report.status for report in reports]
    best_years = [report.best_candidate.outer_ring_year for report in reports if report.best_candidate]
    agreement = len(set(best_years)) == 1 if best_years else False
    return {
        "policy_version": reports[0].policy_version if reports else "",
        "samples": [report.to_dict() for report in reports],
        "summary": {
            "statuses": statuses,
            "best_outer_ring_years": best_years,
            "agreement": agreement,
        },
    }


def _print_cross_verify(payload: dict):
    click.echo("=" * 60)
    click.echo("CROSS-VERIFICATION SUMMARY")
    click.echo("=" * 60)
    for sample in payload["samples"]:
        best = sample.get("best_candidate")
        top_year = best["outer_ring_year"] if best else "n/a"
        click.echo(f"{sample['sample']['name']}: {sample['status']} (top outer-ring year: {top_year})")
    click.echo(f"Agreement: {payload['summary']['agreement']}")


def _generate_plots(report: DatingReport, matcher: CrossdateMatcher, sample_values: np.ndarray, output_path: Path):
    from ..crossdating.detrend import detrend_series, standardize
    from ..visualization.plots import save_diagnostic_plots

    best = report.best_candidate
    if best is None:
        return

    entry = next((entry for entry in matcher.index.entries if entry.site_id == best.reference_id and entry.file_type == best.reference_file_type), None)
    if entry is None or entry.master is None:
        return

    if report.chosen_orientation == "bark_to_pith":
        sample_values = sample_values[::-1]

    detrended, _ = detrend_series(sample_values)
    sample_std = standardize(detrended)

    save_diagnostic_plots(
        report=report,
        sample=sample_std,
        reference=entry.master_values,
        reference_start_year=entry.master_start_year,
        output_path=output_path,
    )


def main():
    cli()


if __name__ == "__main__":
    main()
