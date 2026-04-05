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
from ..materials.catalog import material_group_display_name, material_group_species, normalize_material_group
from ..materials.inference import MaterialInferenceContext, MaterialInferenceEngine, parse_built_year_range
from ..reference.chronology_index import ChronologyIndex
from ..reference.curated import parse_curated_chronology_file
from ..reference.downloader import download_chronologies
from ..reference.tucson_parser import (
    load_measurement_session,
    load_measurements_csv,
    parse_crn_file,
    parse_rwl_file,
)
from ..validation.intake import (
    DEFAULT_LOCAL_VALIDATION_BUILT_YEAR_RANGE,
    EXPECTED_POLICY_OUTCOMES,
    SAMPLE_ORIGINS,
    add_local_validation_case,
    init_local_validation_suite,
    summarize_local_validation_suite,
)


DEFAULT_DATA_DIR = Path.cwd() / "data"
ORIENTATION_CHOICES = click.Choice(["auto", "oldest_to_newest", "bark_to_pith"], case_sensitive=False)
MEASUREMENT_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
EXPECTED_POLICY_OUTCOME_CHOICES = click.Choice(list(EXPECTED_POLICY_OUTCOMES), case_sensitive=False)
SAMPLE_ORIGIN_CHOICES = click.Choice(list(SAMPLE_ORIGINS), case_sensitive=False)


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
@click.option("--material-group", default=None, help="Restrict ranking to a Walpole material group.")
@click.option("--auto-material", is_flag=True, default=False, help="Infer a Walpole material group first, then date only that group if recommended.")
@click.option("--town", default=None, help="Context town for material inference.")
@click.option("--state", "context_state", default=None, help="Context state for material inference.")
@click.option("--built-year-range", default=None, help="Context built-year range as START:END.")
@click.option("--member-type", default="unknown", help="House-member type for Walpole inference.")
@click.option("--context-profile", default=None, help="Explicit context profile id. Walpole mode currently supports walpole_nh_late_1700s_house.")
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
    material_group: Optional[str],
    auto_material: bool,
    town: Optional[str],
    context_state: Optional[str],
    built_year_range: Optional[str],
    member_type: str,
    context_profile: Optional[str],
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
    built_year_range_value = _parse_built_year_range_or_exit(built_year_range)
    selected_material_group = normalize_material_group(material_group) if material_group else None

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
    inference_engine = MaterialInferenceEngine(matcher=matcher)

    for measurement_path in measurements:
        path = Path(measurement_path)
        try:
            values = _load_measurements(path)
        except Exception as exc:
            click.echo(f"Error loading {path}: {exc}", err=True)
            raise SystemExit(1)

        inference_report = None
        if auto_material or selected_material_group or town or context_state or built_year_range_value or member_type != "unknown" or context_profile:
            inference_report = inference_engine.infer(
                values=values,
                sample_name=path.stem,
                context=MaterialInferenceContext(
                    town=town,
                    state=context_state,
                    built_year_range=built_year_range_value,
                    member_type=member_type,
                    profile_id=context_profile,
                ),
                has_bark_edge=bark_edge,
                orientation=orientation,
                era_start=era_start,
                era_end=era_end,
                top_n=top,
            )

        active_material_group = selected_material_group
        if auto_material:
            active_material_group = inference_report.recommended_material if inference_report is not None else None
            if active_material_group is None:
                reports.append(
                    _build_inconclusive_date_report(
                        sample_name=path.stem,
                        values=values,
                        bark_edge=bark_edge,
                        orientation=orientation,
                        warnings=(
                            list(inference_report.warnings)
                            if inference_report is not None
                            else ["Material inference did not recommend a supported Walpole material group."]
                        ),
                        diagnostics={
                            "reference_count": int(len(matcher.index)),
                            "combined_reference_count": int(len(matcher.index)),
                            "detrend_method": "spline",
                            "material_auto_selection_failed": True,
                        },
                        material_inference=inference_report.to_dict() if inference_report is not None else None,
                    )
                )
                loaded_samples.append(values)
                continue

        active_species_filter = list(species_filter) if species_filter else None
        if active_material_group:
            active_species_filter = list(material_group_species(active_material_group))

        report = matcher.date_sample(
            values=values,
            sample_name=path.stem,
            has_bark_edge=bark_edge,
            orientation=orientation,
            species_filter=active_species_filter,
            state_filter=state_filter,
            era_start=era_start,
            era_end=era_end,
            top_n=top,
        )
        report.material_inference = inference_report.to_dict() if inference_report is not None else None
        reports.append(report)
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


@cli.command("infer-materials")
@click.argument("sample", type=click.Path(exists=True))
@click.option("--reference", "-r", type=click.Path(exists=True), default=None, help="Reference directory.")
@click.option("--era-start", type=int, default=1600, help="Earliest plausible outer-ring year.")
@click.option("--era-end", type=int, default=1900, help="Latest plausible outer-ring year.")
@click.option("--town", default=None, help="Context town for material inference.")
@click.option("--state", "context_state", default=None, help="Context state for material inference.")
@click.option("--built-year-range", default=None, help="Context built-year range as START:END.")
@click.option("--member-type", default="unknown", help="House-member type for Walpole inference.")
@click.option("--context-profile", default=None, help="Explicit context profile id. Walpole mode currently supports walpole_nh_late_1700s_house.")
@click.option("--bark-edge/--no-bark-edge", default=True, help="Sample includes the bark edge.")
@click.option("--orientation", type=ORIENTATION_CHOICES, default="auto", help="Input measurement orientation.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Write JSON report to this path.")
@click.option("--json", "json_output", is_flag=True, default=False, help="Print JSON report to stdout.")
@click.option("--top", "-n", type=int, default=5, help="Maximum ranked material candidates to return.")
def infer_materials(
    sample: str,
    reference: Optional[str],
    era_start: int,
    era_end: int,
    town: Optional[str],
    context_state: Optional[str],
    built_year_range: Optional[str],
    member_type: str,
    context_profile: Optional[str],
    bark_edge: bool,
    orientation: str,
    output: Optional[str],
    json_output: bool,
    top: int,
):
    """Infer likely Walpole material groups from a scan, session, or measurement file."""
    reference_dir = Path(reference) if reference else DEFAULT_DATA_DIR / "reference"
    built_year_range_value = _parse_built_year_range_or_exit(built_year_range)

    try:
        matcher = CrossdateMatcher(reference_dir=reference_dir, allow_remote_metadata=True)
    except Exception as exc:
        click.echo(f"Error loading references: {exc}", err=True)
        raise SystemExit(1)

    if len(matcher.index) == 0:
        click.echo("No reference chronologies found. Run 'dendro download' first.", err=True)
        raise SystemExit(1)

    sample_path = Path(sample)
    try:
        values = _load_measurements(sample_path)
    except Exception as exc:
        click.echo(f"Error loading {sample_path}: {exc}", err=True)
        raise SystemExit(1)

    engine = MaterialInferenceEngine(matcher=matcher)
    report = engine.infer(
        values=values,
        sample_name=sample_path.stem,
        context=MaterialInferenceContext(
            town=town,
            state=context_state,
            built_year_range=built_year_range_value,
            member_type=member_type,
            profile_id=context_profile,
        ),
        has_bark_edge=bark_edge,
        orientation=orientation,
        era_start=era_start,
        era_end=era_end,
        top_n=top,
    )
    payload = report.to_dict()

    if output:
        Path(output).write_text(json.dumps(payload, indent=2))
    if json_output:
        click.echo(json.dumps(payload, indent=2))
        return

    _print_material_inference(report)


@cli.command("init-validation-suite")
@click.argument("suite_dir", type=click.Path())
@click.option("--suite-id", default=None, help="Stable suite identifier.")
@click.option("--name", default=None, help="Human-readable suite name.")
@click.option("--scope", default="walpole_nh_late_1700s_house", help="Context scope id.")
@click.option("--town", default="Walpole", help="Default town for new cases.")
@click.option("--state", "context_state", default="NH", help="Default state for new cases.")
@click.option("--built-year-range", default="1760:1800", help="Default built-year range as START:END.")
@click.option("--force/--no-force", default=False, help="Overwrite the suite manifest if it already exists.")
def init_validation_suite(
    suite_dir: str,
    suite_id: Optional[str],
    name: Optional[str],
    scope: str,
    town: str,
    context_state: str,
    built_year_range: str,
    force: bool,
):
    """Create a local known-date scan validation suite scaffold."""
    built_year_range_value = _parse_built_year_range_or_exit(built_year_range) or DEFAULT_LOCAL_VALIDATION_BUILT_YEAR_RANGE
    try:
        suite_file = init_local_validation_suite(
            suite_dir,
            suite_id=suite_id,
            name=name,
            scope=scope,
            town=town,
            state=context_state,
            built_year_range=built_year_range_value,
            force=force,
        )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1)

    click.echo(f"Created validation suite at {suite_file}")
    click.echo(f"Add cases with: dendro add-validation-case {Path(suite_dir)} <case-id> ...")


@cli.command("add-validation-case")
@click.argument("suite_dir", type=click.Path(exists=True))
@click.argument("case_id")
@click.option("--label", default=None, help="Human-readable case label.")
@click.option("--species-name", required=True, help="Trusted species or wood name.")
@click.option("--species-code", default=None, help="Optional species code such as TSCA.")
@click.option("--material-group", default=None, help="Known supported material group when applicable.")
@click.option("--expected-policy-outcome", type=EXPECTED_POLICY_OUTCOME_CHOICES, default="supported_dateable", help="Expected policy behavior for this case.")
@click.option("--sample-origin", type=SAMPLE_ORIGIN_CHOICES, default="firewood", help="Where the sample came from.")
@click.option("--member-type", default="unknown", help="House-member type or unknown.")
@click.option("--true-outer-ring-year", type=int, required=True, help="Trusted outer-ring year for the sample.")
@click.option("--cut-date", default=None, help="Optional full cut date for provenance.")
@click.option("--town", default=None, help="Override town for this case.")
@click.option("--state", "context_state", default=None, help="Override state for this case.")
@click.option("--built-year-range", default=None, help="Override built-year range as START:END.")
@click.option("--bark-edge/--no-bark-edge", default=True, help="Whether the scanned sample includes the bark edge.")
@click.option("--scan-dpi", type=int, default=None, help="Known scan DPI if already planned.")
@click.option("--scale-included/--no-scale-included", default=False, help="Scale or ruler will appear in the scan.")
@click.option("--note", "notes", multiple=True, help="Additional case note. Repeat to add more.")
@click.option("--force/--no-force", default=False, help="Overwrite an existing case scaffold.")
def add_validation_case(
    suite_dir: str,
    case_id: str,
    label: Optional[str],
    species_name: str,
    species_code: Optional[str],
    material_group: Optional[str],
    expected_policy_outcome: str,
    sample_origin: str,
    member_type: str,
    true_outer_ring_year: int,
    cut_date: Optional[str],
    town: Optional[str],
    context_state: Optional[str],
    built_year_range: Optional[str],
    bark_edge: bool,
    scan_dpi: Optional[int],
    scale_included: bool,
    notes: tuple[str, ...],
    force: bool,
):
    """Add a local known-date scan case scaffold."""
    built_year_range_value = _parse_built_year_range_or_exit(built_year_range) if built_year_range else None
    try:
        case_file = add_local_validation_case(
            suite_dir,
            case_id=case_id,
            label=label,
            sample_origin=sample_origin,
            expected_policy_outcome=expected_policy_outcome,
            known_species_name=species_name,
            known_species_code=species_code,
            known_material_group=material_group,
            member_type=member_type,
            true_outer_ring_year=true_outer_ring_year,
            cut_date=cut_date,
            town=town,
            state=context_state,
            built_year_range=built_year_range_value,
            has_bark_edge=bark_edge,
            scan_dpi=scan_dpi,
            scale_included=scale_included,
            notes=tuple(notes),
            force=force,
        )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1)

    click.echo(f"Created validation case at {case_file}")
    click.echo(f"Drop artifacts under {Path(case_file).parent / 'artifacts'}")


@cli.command("validation-info")
@click.argument("suite_path", type=click.Path(exists=True))
@click.option("--json", "json_output", is_flag=True, default=False, help="Print suite readiness as JSON.")
def validation_info(suite_path: str, json_output: bool):
    """Summarize local known-date scan validation readiness."""
    try:
        payload = summarize_local_validation_suite(suite_path)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1)

    if json_output:
        click.echo(json.dumps(payload, indent=2))
        return

    suite = payload["suite"]
    summary = payload["summary"]
    click.echo(f"Validation suite: {suite['name']}")
    click.echo(f"Root: {Path(suite_path).resolve()}")
    click.echo(f"Scope: {suite['scope']}")
    click.echo(f"Cases: {summary['case_count']}")
    click.echo("Expected policy outcomes:")
    for key, count in summary["expected_policy_outcomes"].items():
        click.echo(f"  {key}: {count}")
    click.echo("Readiness:")
    for key in (
        "metadata_ready",
        "measurement_track_ready",
        "scan_session_track_ready",
        "ready_for_full_validation",
        "ready_for_measurement_validation",
        "awaiting_artifacts",
        "metadata_incomplete",
    ):
        click.echo(f"  {key}: {summary['readiness'].get(key, 0)}")
    click.echo("Cases:")
    for case in payload["cases"]:
        missing = ", ".join(case["missing_artifacts"]) if case["missing_artifacts"] else "none"
        click.echo(
            f"  {case['case_id']}: {case['status']} "
            f"[species={case['known_species_name']}; material={case['known_material_group'] or 'none'}; missing={missing}]"
        )
        if case["missing_metadata"]:
            click.echo("    missing metadata: " + ", ".join(case["missing_metadata"]))
        if case["warnings"]:
            click.echo("    warnings: " + " | ".join(case["warnings"]))


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
        "material_groups": {group: len([entry for entry in index.entries if entry.material_group == group]) for group in index.get_material_groups()},
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
    click.echo("Material groups:")
    if payload["material_groups"]:
        for material_group, count in payload["material_groups"].items():
            click.echo(f"  {material_group}: {count}")
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

        if filepath.name.lower().endswith(".curated.json") or filepath.name.lower().endswith(".chronology.json"):
            chronology = parse_curated_chronology_file(filepath)
            if chronology is None:
                click.echo("Could not parse curated chronology file.", err=True)
                raise SystemExit(1)
            click.echo(f"Site: {chronology.site_id}")
            click.echo(f"Species: {chronology.species or '(unknown)'}")
            click.echo(f"Material group: {chronology.material_group or '(unknown)'}")
            click.echo(f"Years: {chronology.start_year}-{chronology.end_year}")
            click.echo(f"Length: {chronology.length}")
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
    if filepath.suffix.lower() in MEASUREMENT_IMAGE_SUFFIXES:
        resolved = _resolve_scan_measurement_artifact(filepath)
        if resolved is None:
            raise ValueError(
                "Scan input requires a sibling .session.json or .measurements.csv artifact produced by 'dendro measure'."
            )
        filepath = resolved

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


def _resolve_scan_measurement_artifact(filepath: Path) -> Optional[Path]:
    stem = filepath.with_suffix("")
    candidates = [
        stem.with_suffix(".session.json"),
        stem.with_suffix(".measurements.csv"),
        filepath.parent / f"{filepath.stem}.session.json",
        filepath.parent / f"{filepath.stem}.measurements.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _parse_built_year_range_or_exit(raw_value: Optional[str]) -> Optional[tuple[int, int]]:
    try:
        return parse_built_year_range(raw_value)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1)


def _build_inconclusive_date_report(
    *,
    sample_name: str,
    values: np.ndarray,
    bark_edge: bool,
    orientation: str,
    warnings: list[str],
    diagnostics: dict,
    material_inference: Optional[dict],
) -> DatingReport:
    chosen_orientation = "oldest_to_newest" if orientation == "auto" else orientation
    return DatingReport(
        sample_name=sample_name,
        sample_length=len(values),
        bark_edge=bark_edge,
        requested_orientation=orientation,
        chosen_orientation=chosen_orientation,
        analysis_orientation="oldest_to_newest",
        status="inconclusive",
        policy_version="2026.04-assisted-ranking-v1",
        candidates=[],
        material_inference=material_inference,
        warnings=warnings,
        diagnostics=diagnostics,
    )


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

    if report.material_inference:
        material_inference = report.material_inference
        click.echo("Material inference:")
        click.echo(f"  Status: {material_inference['status'].upper()}")
        click.echo(f"  Support: {material_inference['support_status']}")
        if material_inference.get("recommended_material"):
            click.echo(
                "  Recommended: "
                + material_group_display_name(material_inference["recommended_material"])
            )
        top_materials = material_inference.get("material_candidates", [])[:3]
        for index, candidate in enumerate(top_materials, start=1):
            click.echo(
                f"  {index}. {candidate['display_name']} "
                f"score={candidate['score']:.3f} "
                f"support={candidate['support_status']}"
            )
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
        click.echo(
            f"Reference: {best.reference_name} "
            f"({best.reference_species or 'unknown species'}, {best.reference_state or 'unknown state'})"
        )
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
                f"[{candidate.reference_species or 'unknown'} {candidate.reference_state or '??'} {candidate.reference_material_group or 'unknown'}] "
                f"outer={candidate.outer_ring_year} "
                f"score={candidate.composite_score:.3f} "
                f"r={candidate.correlation:.3f} "
                f"t={candidate.t_value:.2f}"
            )


def _print_material_inference(report):
    click.echo("=" * 60)
    click.echo("MATERIAL INFERENCE REPORT")
    click.echo("=" * 60)
    click.echo(f"Status: {report.status.upper()}")
    if report.context_profile is not None:
        click.echo(f"Profile: {report.context_profile.profile_id} v{report.context_profile.version}")
    if report.recommended_material:
        click.echo("Recommended material: " + material_group_display_name(report.recommended_material))
    click.echo(f"Support status: {report.support_status}")
    click.echo()

    if report.warnings:
        click.echo("Warnings:")
        for warning in report.warnings:
            click.echo(f"  - {warning}")
        click.echo()

    if report.candidates:
        click.echo("Ranked material groups:")
        for index, candidate in enumerate(report.candidates, start=1):
            click.echo(
                f"{index}. {candidate.display_name} "
                f"score={candidate.score:.3f} "
                f"support={candidate.support_status} "
                f"best_outer={candidate.best_outer_ring_year or 'n/a'}"
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
