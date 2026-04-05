"""
Scaffolding and inventory helpers for local known-date scan validation suites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Optional

from ..materials.catalog import SUPPORTED_MATERIAL_GROUPS, normalize_material_group


LOCAL_VALIDATION_SCHEMA_VERSION = 1
DEFAULT_LOCAL_VALIDATION_SCOPE = "walpole_nh_late_1700s_house"
DEFAULT_LOCAL_VALIDATION_TOWN = "Walpole"
DEFAULT_LOCAL_VALIDATION_STATE = "NH"
DEFAULT_LOCAL_VALIDATION_BUILT_YEAR_RANGE = (1760, 1800)
EXPECTED_POLICY_OUTCOMES = ("supported_dateable", "unsupported_inconclusive")
SAMPLE_ORIGINS = ("firewood", "structure", "tree_slice", "log_round", "core", "unknown", "other")
ARTIFACT_KEYS = ("scan_image", "measurement_session", "measurement_csv")


def _normalize_year_range(raw_value: object) -> tuple[int, int]:
    if isinstance(raw_value, (list, tuple)) and len(raw_value) >= 2:
        return int(raw_value[0]), int(raw_value[1])
    if isinstance(raw_value, str) and ":" in raw_value:
        start, end = raw_value.split(":", 1)
        return int(start.strip()), int(end.strip())
    return DEFAULT_LOCAL_VALIDATION_BUILT_YEAR_RANGE


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "local_known_scans"


def _suite_file_for(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_dir():
        return path_obj / "suite.json"
    return path_obj


def _normalize_expected_policy_outcome(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in EXPECTED_POLICY_OUTCOMES:
        raise ValueError(
            "expected_policy_outcome must be one of: "
            + ", ".join(EXPECTED_POLICY_OUTCOMES)
        )
    return normalized


@dataclass(frozen=True)
class LocalValidationCase:
    """Single local known-date validation case."""

    suite_root: str
    case_dir: str
    case_file: str
    case_id: str
    label: str
    sample_origin: str
    expected_policy_outcome: str
    known_species_name: str
    known_species_code: str = ""
    known_material_group: str = ""
    member_type: str = "unknown"
    town: str = DEFAULT_LOCAL_VALIDATION_TOWN
    state: str = DEFAULT_LOCAL_VALIDATION_STATE
    built_year_range: tuple[int, int] = DEFAULT_LOCAL_VALIDATION_BUILT_YEAR_RANGE
    true_outer_ring_year: Optional[int] = None
    cut_date: str = ""
    has_bark_edge: bool = True
    scan_dpi: Optional[int] = None
    scale_included: bool = False
    notes: tuple[str, ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)

    def artifact_path(self, key: str) -> Path:
        return Path(self.case_dir) / self.artifacts[key]

    def to_dict(self) -> dict:
        return {
            "schema_version": LOCAL_VALIDATION_SCHEMA_VERSION,
            "case_id": self.case_id,
            "label": self.label,
            "sample_origin": self.sample_origin,
            "expected_policy_outcome": self.expected_policy_outcome,
            "known_species_name": self.known_species_name,
            "known_species_code": self.known_species_code,
            "known_material_group": self.known_material_group or None,
            "member_type": self.member_type,
            "town": self.town,
            "state": self.state,
            "built_year_range": [int(self.built_year_range[0]), int(self.built_year_range[1])],
            "true_outer_ring_year": int(self.true_outer_ring_year) if self.true_outer_ring_year is not None else None,
            "cut_date": self.cut_date or None,
            "has_bark_edge": bool(self.has_bark_edge),
            "scan_dpi": int(self.scan_dpi) if self.scan_dpi is not None else None,
            "scale_included": bool(self.scale_included),
            "notes": list(self.notes),
            "artifacts": dict(self.artifacts),
        }

    def readiness(self) -> dict:
        missing_metadata: list[str] = []
        warnings: list[str] = []

        if not self.case_id.strip():
            missing_metadata.append("case_id")
        if not self.label.strip():
            missing_metadata.append("label")
        if not self.known_species_name.strip():
            missing_metadata.append("known_species_name")
        if self.true_outer_ring_year is None:
            missing_metadata.append("true_outer_ring_year")
        if self.expected_policy_outcome not in EXPECTED_POLICY_OUTCOMES:
            missing_metadata.append("expected_policy_outcome")
        if self.expected_policy_outcome == "supported_dateable" and not self.known_material_group:
            missing_metadata.append("known_material_group")
        if self.expected_policy_outcome == "supported_dateable" and self.known_material_group:
            if self.known_material_group not in SUPPORTED_MATERIAL_GROUPS:
                missing_metadata.append("known_material_group")
        if self.scan_dpi is None and not self.scale_included:
            warnings.append("Provide either scan_dpi or an in-frame scale before capture.")

        artifact_status = {
            key: self.artifact_path(key).exists()
            for key in ARTIFACT_KEYS
        }
        missing_artifacts = [key for key, exists in artifact_status.items() if not exists]
        metadata_ready = len(missing_metadata) == 0
        measurement_track_ready = metadata_ready and artifact_status["measurement_csv"]
        scan_session_track_ready = (
            metadata_ready
            and artifact_status["scan_image"]
            and artifact_status["measurement_session"]
        )

        status = "metadata_incomplete"
        if metadata_ready and scan_session_track_ready and measurement_track_ready:
            status = "ready_for_full_validation"
        elif metadata_ready and measurement_track_ready:
            status = "ready_for_measurement_validation"
        elif metadata_ready:
            status = "awaiting_artifacts"

        return {
            "case_id": self.case_id,
            "label": self.label,
            "status": status,
            "expected_policy_outcome": self.expected_policy_outcome,
            "known_species_name": self.known_species_name,
            "known_species_code": self.known_species_code,
            "known_material_group": self.known_material_group or None,
            "sample_origin": self.sample_origin,
            "member_type": self.member_type,
            "town": self.town,
            "state": self.state,
            "built_year_range": [int(self.built_year_range[0]), int(self.built_year_range[1])],
            "true_outer_ring_year": int(self.true_outer_ring_year) if self.true_outer_ring_year is not None else None,
            "cut_date": self.cut_date or None,
            "has_bark_edge": bool(self.has_bark_edge),
            "scan_dpi": int(self.scan_dpi) if self.scan_dpi is not None else None,
            "scale_included": bool(self.scale_included),
            "metadata_ready": metadata_ready,
            "measurement_track_ready": measurement_track_ready,
            "scan_session_track_ready": scan_session_track_ready,
            "missing_metadata": missing_metadata,
            "artifact_status": artifact_status,
            "missing_artifacts": missing_artifacts,
            "warnings": warnings,
            "notes": list(self.notes),
            "case_dir": self.case_dir,
            "case_file": self.case_file,
        }


@dataclass(frozen=True)
class LocalValidationSuite:
    """Loaded local validation suite manifest."""

    root_dir: str
    suite_file: str
    suite_id: str
    name: str
    scope: str
    town: str
    state: str
    built_year_range: tuple[int, int]
    cases_dir: str
    notes: tuple[str, ...]
    cases: tuple[LocalValidationCase, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": LOCAL_VALIDATION_SCHEMA_VERSION,
            "suite_id": self.suite_id,
            "name": self.name,
            "scope": self.scope,
            "town": self.town,
            "state": self.state,
            "built_year_range": [int(self.built_year_range[0]), int(self.built_year_range[1])],
            "cases_dir": self.cases_dir,
            "case_ids": [case.case_id for case in self.cases],
            "notes": list(self.notes),
        }


def init_local_validation_suite(
    root_dir: str | Path,
    *,
    suite_id: Optional[str] = None,
    name: Optional[str] = None,
    scope: str = DEFAULT_LOCAL_VALIDATION_SCOPE,
    town: str = DEFAULT_LOCAL_VALIDATION_TOWN,
    state: str = DEFAULT_LOCAL_VALIDATION_STATE,
    built_year_range: tuple[int, int] = DEFAULT_LOCAL_VALIDATION_BUILT_YEAR_RANGE,
    force: bool = False,
) -> Path:
    """Create a local known-date scan validation suite scaffold."""
    root_path = Path(root_dir)
    suite_file = root_path / "suite.json"
    if suite_file.exists() and not force:
        raise FileExistsError(f"Suite already exists at {suite_file}")

    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "cases").mkdir(parents=True, exist_ok=True)

    suite_payload = {
        "schema_version": LOCAL_VALIDATION_SCHEMA_VERSION,
        "suite_id": suite_id or _slugify(root_path.name),
        "name": name or "Local Known-Date Scan Suite",
        "scope": scope,
        "town": town,
        "state": state.upper(),
        "built_year_range": [int(built_year_range[0]), int(built_year_range[1])],
        "cases_dir": "cases",
        "case_ids": [],
        "notes": [
            "Use this suite for real local scans with trusted provenance.",
            "Each case should record known species/material, true outer-ring year, and bark-edge status.",
            "Cases can support either supported_dateable checks or unsupported_inconclusive checks.",
        ],
    }
    suite_file.write_text(json.dumps(suite_payload, indent=2))

    readme_path = root_path / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Local Known-Date Scan Suite",
                "",
                "This directory is for real local validation cases such as firewood slices or known house timbers.",
                "",
                "Recommended workflow:",
                "1. Add a case with `dendro add-validation-case ...`.",
                "2. Drop the scan image into `cases/<case_id>/artifacts/scan.tif` or update the artifact path in `case.json`.",
                "3. Run `dendro measure` to create the session and measurement CSV.",
                "4. Use `dendro validation-info <suite_dir>` to check readiness before running benchmark work.",
                "",
                "Required provenance per case:",
                "- known species or material group",
                "- trusted true outer-ring year",
                "- bark-edge present or absent",
                "- scan DPI or an in-frame scale",
                "",
                "If you want to test unsupported species such as ash, set",
                "`expected_policy_outcome` to `unsupported_inconclusive`.",
            ]
        )
        + "\n"
    )
    return suite_file


def add_local_validation_case(
    suite_path: str | Path,
    *,
    case_id: str,
    label: Optional[str],
    sample_origin: str,
    expected_policy_outcome: str,
    known_species_name: str,
    known_species_code: Optional[str] = None,
    known_material_group: Optional[str] = None,
    member_type: str = "unknown",
    true_outer_ring_year: int,
    cut_date: Optional[str] = None,
    town: Optional[str] = None,
    state: Optional[str] = None,
    built_year_range: Optional[tuple[int, int]] = None,
    has_bark_edge: bool = True,
    scan_dpi: Optional[int] = None,
    scale_included: bool = False,
    notes: tuple[str, ...] = (),
    force: bool = False,
) -> Path:
    """Add a case scaffold to a local validation suite."""
    suite = load_local_validation_suite(suite_path)
    normalized_case_id = _slugify(case_id)
    if sample_origin not in SAMPLE_ORIGINS:
        raise ValueError("sample_origin must be one of: " + ", ".join(SAMPLE_ORIGINS))
    expected_policy_outcome = _normalize_expected_policy_outcome(expected_policy_outcome)
    normalized_material_group = normalize_material_group(known_material_group) if known_material_group else ""
    if expected_policy_outcome == "supported_dateable" and not normalized_material_group:
        raise ValueError("supported_dateable cases require --material-group")

    suite_file = Path(suite.suite_file)
    cases_dir = Path(suite.root_dir) / suite.cases_dir
    case_dir = cases_dir / normalized_case_id
    case_file = case_dir / "case.json"
    if case_file.exists() and not force:
        raise FileExistsError(f"Case already exists at {case_file}")

    artifacts_dir = case_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / ".gitkeep").write_text("")

    case_payload = {
        "schema_version": LOCAL_VALIDATION_SCHEMA_VERSION,
        "case_id": normalized_case_id,
        "label": label or normalized_case_id.replace("_", " ").title(),
        "sample_origin": sample_origin,
        "expected_policy_outcome": expected_policy_outcome,
        "known_species_name": known_species_name,
        "known_species_code": (known_species_code or "").upper(),
        "known_material_group": normalized_material_group or None,
        "member_type": member_type.strip().lower(),
        "town": town or suite.town,
        "state": (state or suite.state).upper(),
        "built_year_range": list(built_year_range or suite.built_year_range),
        "true_outer_ring_year": int(true_outer_ring_year),
        "cut_date": cut_date or None,
        "has_bark_edge": bool(has_bark_edge),
        "scan_dpi": int(scan_dpi) if scan_dpi is not None else None,
        "scale_included": bool(scale_included),
        "notes": list(notes),
        "artifacts": {
            "scan_image": "artifacts/scan.tif",
            "measurement_session": "artifacts/measurement.session.json",
            "measurement_csv": "artifacts/measurements.csv",
        },
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    case_file.write_text(json.dumps(case_payload, indent=2))

    (case_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {case_payload['label']}",
                "",
                f"- Case ID: `{normalized_case_id}`",
                f"- Species: `{known_species_name}`",
                f"- Material group: `{normalized_material_group or 'none / unsupported'}`",
                f"- Expected policy outcome: `{expected_policy_outcome}`",
                f"- True outer-ring year: `{true_outer_ring_year}`",
                "",
                "Place artifacts here:",
                f"- `{case_payload['artifacts']['scan_image']}`",
                f"- `{case_payload['artifacts']['measurement_session']}`",
                f"- `{case_payload['artifacts']['measurement_csv']}`",
                "",
                "Capture checklist:",
                f"- Bark edge present: `{bool(has_bark_edge)}`",
                f"- Scan DPI: `{scan_dpi if scan_dpi is not None else 'not yet recorded'}`",
                f"- Scale included: `{bool(scale_included)}`",
                "",
                "Update `case.json` if the artifact filenames differ.",
            ]
        )
        + "\n"
    )

    suite_payload = json.loads(suite_file.read_text())
    case_ids = list(suite_payload.get("case_ids", []))
    if normalized_case_id not in case_ids:
        case_ids.append(normalized_case_id)
    suite_payload["case_ids"] = sorted(case_ids)
    suite_file.write_text(json.dumps(suite_payload, indent=2))
    return case_file


def load_local_validation_suite(path: str | Path) -> LocalValidationSuite:
    """Load a local validation suite from a directory or suite file."""
    suite_file = _suite_file_for(path).resolve()
    if not suite_file.exists():
        raise FileNotFoundError(f"Suite file not found: {suite_file}")

    root_dir = suite_file.parent
    payload = json.loads(suite_file.read_text())
    cases_dir_name = str(payload.get("cases_dir", "cases"))
    cases_dir = root_dir / cases_dir_name
    case_ids = list(payload.get("case_ids", []))

    if not case_ids and cases_dir.exists():
        case_ids = sorted(
            path.parent.name
            for path in cases_dir.rglob("case.json")
        )

    cases: list[LocalValidationCase] = []
    for case_id in case_ids:
        case_file = cases_dir / case_id / "case.json"
        if not case_file.exists():
            continue
        case_payload = json.loads(case_file.read_text())
        material_group = ""
        if case_payload.get("known_material_group"):
            material_group = normalize_material_group(str(case_payload["known_material_group"]))
        artifacts = {
            key: str(case_payload.get("artifacts", {}).get(key, f"artifacts/{key}"))
            for key in ARTIFACT_KEYS
        }
        cases.append(
            LocalValidationCase(
                suite_root=str(root_dir),
                case_dir=str(case_file.parent),
                case_file=str(case_file),
                case_id=str(case_payload.get("case_id", case_id)),
                label=str(case_payload.get("label", case_id)),
                sample_origin=str(case_payload.get("sample_origin", "unknown")).strip().lower(),
                expected_policy_outcome=_normalize_expected_policy_outcome(
                    str(case_payload.get("expected_policy_outcome", "supported_dateable"))
                ),
                known_species_name=str(case_payload.get("known_species_name", "")),
                known_species_code=str(case_payload.get("known_species_code", "")).upper(),
                known_material_group=material_group,
                member_type=str(case_payload.get("member_type", "unknown")).strip().lower(),
                town=str(case_payload.get("town", payload.get("town", DEFAULT_LOCAL_VALIDATION_TOWN))),
                state=str(case_payload.get("state", payload.get("state", DEFAULT_LOCAL_VALIDATION_STATE))).upper(),
                built_year_range=_normalize_year_range(
                    case_payload.get("built_year_range", payload.get("built_year_range", DEFAULT_LOCAL_VALIDATION_BUILT_YEAR_RANGE))
                ),
                true_outer_ring_year=int(case_payload["true_outer_ring_year"]) if case_payload.get("true_outer_ring_year") is not None else None,
                cut_date=str(case_payload.get("cut_date", "") or ""),
                has_bark_edge=bool(case_payload.get("has_bark_edge", True)),
                scan_dpi=int(case_payload["scan_dpi"]) if case_payload.get("scan_dpi") is not None else None,
                scale_included=bool(case_payload.get("scale_included", False)),
                notes=tuple(str(item) for item in case_payload.get("notes", [])),
                artifacts=artifacts,
            )
        )

    return LocalValidationSuite(
        root_dir=str(root_dir),
        suite_file=str(suite_file),
        suite_id=str(payload.get("suite_id", _slugify(root_dir.name))),
        name=str(payload.get("name", "Local Known-Date Scan Suite")),
        scope=str(payload.get("scope", DEFAULT_LOCAL_VALIDATION_SCOPE)),
        town=str(payload.get("town", DEFAULT_LOCAL_VALIDATION_TOWN)),
        state=str(payload.get("state", DEFAULT_LOCAL_VALIDATION_STATE)).upper(),
        built_year_range=_normalize_year_range(payload.get("built_year_range", DEFAULT_LOCAL_VALIDATION_BUILT_YEAR_RANGE)),
        cases_dir=cases_dir_name,
        notes=tuple(str(item) for item in payload.get("notes", [])),
        cases=tuple(cases),
    )


def summarize_local_validation_suite(path: str | Path) -> dict:
    """Summarize case readiness for a local validation suite."""
    suite = load_local_validation_suite(path)
    case_summaries = [case.readiness() for case in suite.cases]
    expected_counts: dict[str, int] = {key: 0 for key in EXPECTED_POLICY_OUTCOMES}
    material_counts: dict[str, int] = {}
    readiness_counts = {
        "metadata_ready": 0,
        "measurement_track_ready": 0,
        "scan_session_track_ready": 0,
        "ready_for_full_validation": 0,
        "ready_for_measurement_validation": 0,
        "awaiting_artifacts": 0,
        "metadata_incomplete": 0,
    }

    for case_summary in case_summaries:
        expected_counts[case_summary["expected_policy_outcome"]] += 1
        material_label = case_summary["known_material_group"] or "(unsupported_or_unspecified)"
        material_counts[material_label] = material_counts.get(material_label, 0) + 1
        if case_summary["metadata_ready"]:
            readiness_counts["metadata_ready"] += 1
        if case_summary["measurement_track_ready"]:
            readiness_counts["measurement_track_ready"] += 1
        if case_summary["scan_session_track_ready"]:
            readiness_counts["scan_session_track_ready"] += 1
        readiness_counts[case_summary["status"]] = readiness_counts.get(case_summary["status"], 0) + 1

    return {
        "schema_version": LOCAL_VALIDATION_SCHEMA_VERSION,
        "suite": suite.to_dict(),
        "summary": {
            "case_count": len(case_summaries),
            "expected_policy_outcomes": expected_counts,
            "material_groups": dict(sorted(material_counts.items())),
            "readiness": readiness_counts,
        },
        "cases": case_summaries,
    }
