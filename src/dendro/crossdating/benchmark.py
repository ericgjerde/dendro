"""
Corpus-scale benchmark helpers for assisted ranking validation.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Optional

from .matcher import CrossdateMatcher
from ..reference.chronology_index import ChronologyIndex
from ..reference.tucson_parser import parse_rwl_file


PASS_TOLERANCE_YEARS = 2
NEAR_MISS_TOLERANCE_YEARS = 5
DEFAULT_BENCHMARK_SLICES_PATH = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "benchmark_slices_v1.json"
_WORKER_BASE_INDEX: Optional[ChronologyIndex] = None


@dataclass(frozen=True)
class BenchmarkCase:
    """Single known-date validation case from the bundled corpus."""

    test_file: str
    series_id: str
    state: str
    species: str
    site_id: str
    site_name: str
    true_start_year: int
    true_end_year: int
    length: int
    curated_suite_case: bool = False

    def to_dict(self) -> dict:
        return {
            "test_file": self.test_file,
            "series_id": self.series_id,
            "state": self.state,
            "species": self.species,
            "site_id": self.site_id,
            "site_name": self.site_name,
            "true_start_year": int(self.true_start_year),
            "true_end_year": int(self.true_end_year),
            "length": int(self.length),
            "curated_suite_case": bool(self.curated_suite_case),
        }


@dataclass
class BenchmarkCaseResult:
    """Structured outcome for a single benchmark case."""

    case: BenchmarkCase
    status: str
    reference_count: int
    candidate_count: int
    top1_outer_ring_year: Optional[int]
    top1_error: Optional[int]
    top1_reference_name: str = ""
    top1_reference_state: str = ""
    top1_reference_species: str = ""
    top1_score: Optional[float] = None
    top1_correlation: Optional[float] = None
    top1_t_value: Optional[float] = None
    correct_year_rank: Optional[int] = None
    correct_year_in_top5: bool = False
    correct_year_in_top10: bool = False
    category: str = ""
    likely_causes: list[str] = field(default_factory=list)
    warning_count: int = 0
    primary_slice_id: str = "unclassified"
    overlay_ids: list[str] = field(default_factory=list)
    primary_slice_matched: bool = False

    @property
    def top1_abs_error(self) -> Optional[int]:
        return abs(self.top1_error) if self.top1_error is not None else None

    @property
    def passed_top1(self) -> bool:
        return self.top1_abs_error is not None and self.top1_abs_error <= PASS_TOLERANCE_YEARS

    def to_dict(self) -> dict:
        return {
            "case": self.case.to_dict(),
            "status": self.status,
            "reference_count": int(self.reference_count),
            "candidate_count": int(self.candidate_count),
            "top1_outer_ring_year": int(self.top1_outer_ring_year) if self.top1_outer_ring_year is not None else None,
            "top1_error": int(self.top1_error) if self.top1_error is not None else None,
            "top1_reference_name": self.top1_reference_name,
            "top1_reference_state": self.top1_reference_state,
            "top1_reference_species": self.top1_reference_species,
            "top1_score": round(float(self.top1_score), 4) if self.top1_score is not None else None,
            "top1_correlation": round(float(self.top1_correlation), 4) if self.top1_correlation is not None else None,
            "top1_t_value": round(float(self.top1_t_value), 3) if self.top1_t_value is not None else None,
            "correct_year_rank": int(self.correct_year_rank) if self.correct_year_rank is not None else None,
            "correct_year_in_top5": bool(self.correct_year_in_top5),
            "correct_year_in_top10": bool(self.correct_year_in_top10),
            "category": self.category,
            "likely_causes": list(self.likely_causes),
            "warning_count": int(self.warning_count),
            "benchmark_slices": {
                "primary_slice_id": self.primary_slice_id,
                "overlay_ids": list(self.overlay_ids),
                "primary_slice_matched": bool(self.primary_slice_matched),
            },
        }


@dataclass(frozen=True)
class BenchmarkSliceDefinition:
    """Primary slice or overlay definition loaded from fixture metadata."""

    id: str
    description: str
    inclusion: dict
    targets: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkSliceCatalog:
    """Loaded benchmark slice taxonomy."""

    source_path: str
    primary_slices: tuple[BenchmarkSliceDefinition, ...]
    overlays: tuple[BenchmarkSliceDefinition, ...]
    taxonomy_order: tuple[str, ...]

    @property
    def primary_by_id(self) -> dict[str, BenchmarkSliceDefinition]:
        return {slice_def.id: slice_def for slice_def in self.primary_slices}

    @property
    def overlay_by_id(self) -> dict[str, BenchmarkSliceDefinition]:
        return {slice_def.id: slice_def for slice_def in self.overlays}


def _default_slice_catalog_path() -> Path:
    return DEFAULT_BENCHMARK_SLICES_PATH


@lru_cache(maxsize=8)
def _load_benchmark_slice_catalog_cached(path_str: str) -> BenchmarkSliceCatalog:
    path = Path(path_str)
    if not path.exists():
        return BenchmarkSliceCatalog(
            source_path=str(path),
            primary_slices=(),
            overlays=(),
            taxonomy_order=(),
        )

    payload = json.loads(path.read_text())
    primary_slices = tuple(
        BenchmarkSliceDefinition(
            id=str(item["id"]),
            description=str(item.get("description", "")),
            inclusion=dict(item.get("inclusion", {})),
            targets=dict(item.get("targets", {})),
        )
        for item in payload.get("primary_slices", [])
    )
    overlays = tuple(
        BenchmarkSliceDefinition(
            id=str(item["id"]),
            description=str(item.get("description", "")),
            inclusion=dict(item.get("inclusion", {})),
            targets=dict(item.get("targets", {})),
        )
        for item in payload.get("overlays", [])
    )
    taxonomy_order = tuple(str(item) for item in payload.get("taxonomy_order", []))
    if not taxonomy_order:
        taxonomy_order = tuple(slice_def.id for slice_def in primary_slices)

    return BenchmarkSliceCatalog(
        source_path=str(path),
        primary_slices=primary_slices,
        overlays=overlays,
        taxonomy_order=taxonomy_order,
    )


def load_benchmark_slice_catalog(slice_config_path: Optional[str | Path] = None) -> BenchmarkSliceCatalog:
    """Load the benchmark slice taxonomy, defaulting to the shipped fixture."""
    if slice_config_path is None:
        slice_config_path = _default_slice_catalog_path()
    return _load_benchmark_slice_catalog_cached(str(Path(slice_config_path).resolve()))


def _normalize_rule_values(values: object) -> set:
    if values is None:
        return set()
    if isinstance(values, str):
        return {values.upper()}
    return {str(value).upper() for value in values}


def _matches_benchmark_slice(
    slice_def: BenchmarkSliceDefinition,
    result: BenchmarkCaseResult,
    *,
    primary_slice_id: str | None = None,
    respect_length: bool = True,
) -> bool:
    inclusion = slice_def.inclusion or {}
    case = result.case
    species = (case.species or "").upper()
    state = (case.state or "").upper()
    reference_count = int(result.reference_count)

    if "exclude_slice_ids" in inclusion and primary_slice_id in _normalize_rule_values(inclusion.get("exclude_slice_ids")):
        return False

    allowlist = inclusion.get("species_state_allowlist")
    if allowlist:
        normalized_allowlist = {
            (str(item[0]).upper(), str(item[1]).upper())
            for item in allowlist
            if isinstance(item, (list, tuple)) and len(item) >= 2
        }
        if (species, state) not in normalized_allowlist:
            return False

    species_any_of = _normalize_rule_values(inclusion.get("species_any_of"))
    if species_any_of and species not in species_any_of:
        return False

    states_any_of = _normalize_rule_values(inclusion.get("states_any_of"))
    if states_any_of and state not in states_any_of:
        return False

    min_length = inclusion.get("min_length")
    if respect_length and min_length is not None and case.length < int(min_length):
        return False

    max_length = inclusion.get("max_length")
    if max_length is not None and case.length > int(max_length):
        return False

    min_reference_count = inclusion.get("min_reference_count_after_exclusion")
    if min_reference_count is not None and reference_count < int(min_reference_count):
        return False

    max_reference_count = inclusion.get("max_reference_count_after_exclusion")
    if max_reference_count is not None and reference_count > int(max_reference_count):
        return False

    return True


def assign_benchmark_slices(
    result: BenchmarkCaseResult,
    *,
    slice_config_path: Optional[str | Path] = None,
) -> BenchmarkCaseResult:
    """Assign a primary slice and overlays to a benchmark result."""
    catalog = load_benchmark_slice_catalog(slice_config_path)

    primary_slice_id = "unclassified"
    primary_slice_matched = False
    ordered_primary_ids = [
        slice_id
        for slice_id in catalog.taxonomy_order
        if slice_id in catalog.primary_by_id
    ]
    ordered_primary_ids.extend(
        slice_def.id
        for slice_def in catalog.primary_slices
        if slice_def.id not in ordered_primary_ids
    )

    for slice_id in ordered_primary_ids:
        slice_def = catalog.primary_by_id[slice_id]
        if _matches_benchmark_slice(
            slice_def,
            result,
            primary_slice_id=slice_id,
            respect_length=False,
        ):
            primary_slice_id = slice_id
            primary_slice_matched = True
            break

    overlay_ids = [
        slice_def.id
        for slice_def in catalog.overlays
        if _matches_benchmark_slice(slice_def, result, primary_slice_id=primary_slice_id)
    ]

    result.primary_slice_id = primary_slice_id
    result.primary_slice_matched = primary_slice_matched
    result.overlay_ids = overlay_ids
    return result


def _result_group_summary(grouped_results: list[BenchmarkCaseResult]) -> dict:
    count = len(grouped_results)
    if count == 0:
        return {"count": 0}

    top1_passes = sum(1 for result in grouped_results if result.passed_top1)
    top5_hits = sum(1 for result in grouped_results if result.correct_year_in_top5)
    recommended = sum(1 for result in grouped_results if result.category == "correct_recommended")
    recommended_outputs = sum(1 for result in grouped_results if result.status == "recommended")
    no_matches = sum(1 for result in grouped_results if result.category == "no_match")
    long_offsets = sum(1 for result in grouped_results if result.category == "long_offset_false_positive")
    ranking_misses = sum(1 for result in grouped_results if result.category == "ranking_miss")
    recommended_precision = (
        round(recommended / recommended_outputs, 4)
        if recommended_outputs > 0
        else None
    )

    return {
        "count": count,
        "top1_within_2_years": top1_passes,
        "top1_within_2_years_rate": round(top1_passes / count, 4),
        "top5_within_2_years": top5_hits,
        "top5_within_2_years_rate": round(top5_hits / count, 4),
        "recommended": recommended,
        "recommended_rate": round(recommended / count, 4),
        "recommended_outputs": recommended_outputs,
        "recommended_outputs_rate": round(recommended_outputs / count, 4),
        "recommended_precision": recommended_precision,
        "no_match": no_matches,
        "no_match_rate": round(no_matches / count, 4),
        "ranking_miss": ranking_misses,
        "ranking_miss_rate": round(ranking_misses / count, 4),
        "long_offset_false_positive": long_offsets,
        "long_offset_false_positive_rate": round(long_offsets / count, 4),
    }


def _evaluate_target(summary: dict, target_key: str, target_value: float) -> dict:
    metric_map = {
        "top1_within_2_years_rate": ("top1_within_2_years_rate", "gte"),
        "top5_within_2_years_rate": ("top5_within_2_years_rate", "gte"),
        "long_offset_false_positive_rate": ("long_offset_false_positive_rate", "lte"),
        "no_match_rate": ("no_match_rate", "lte"),
        "recommended_precision": ("recommended_precision", "gte"),
        "recommended_rate_max": ("recommended_outputs_rate", "lte"),
    }
    actual_key, comparison = metric_map.get(target_key, (target_key, "gte"))
    actual_value = summary.get(actual_key)
    passed = False
    delta = None

    if actual_value is not None:
        actual_float = float(actual_value)
        target_float = float(target_value)
        if comparison == "lte":
            passed = actual_float <= target_float
            delta = round(target_float - actual_float, 4)
        else:
            passed = actual_float >= target_float
            delta = round(actual_float - target_float, 4)

    return {
        "metric": actual_key,
        "comparison": comparison,
        "target": float(target_value),
        "actual": actual_value,
        "passed": bool(passed),
        "delta": delta,
    }


def _slice_summary(
    slice_def: BenchmarkSliceDefinition,
    grouped_results: list[BenchmarkCaseResult],
    *,
    kind: str,
) -> dict:
    summary = _result_group_summary(grouped_results)
    target_checks = {
        target_key: _evaluate_target(summary, target_key, target_value)
        for target_key, target_value in slice_def.targets.items()
    }
    return {
        "id": slice_def.id,
        "kind": kind,
        "description": slice_def.description,
        "count": summary["count"],
        "summary": summary,
        "targets": dict(slice_def.targets),
        "target_checks": target_checks,
        "targets_met": all(check["passed"] for check in target_checks.values()) if target_checks else None,
        "unmet_targets": [key for key, check in target_checks.items() if not check["passed"]],
    }


def load_curated_suite_cases(suite_file: Optional[str | Path]) -> set[tuple[str, str]]:
    """Load curated suite pairs of (test_file, series_id)."""
    if suite_file is None:
        return set()

    suite_path = Path(suite_file)
    payload = json.loads(suite_path.read_text())
    cases = set()
    for item in payload.get("cases", []):
        raw_path = item.get("test_file")
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        cases.add((str(path), str(item["series_id"])))
    return cases


def discover_benchmark_cases(
    reference_dir: str | Path,
    *,
    min_length: int = 50,
    max_start_year: int = 1780,
    min_end_year: int = 1800,
    scope: str = "all",
    curated_suite_file: Optional[str | Path] = None,
) -> list[BenchmarkCase]:
    """Discover eligible benchmark cases from RWL files."""
    reference_dir = Path(reference_dir)
    base_index = ChronologyIndex(reference_dir)
    curated_cases = load_curated_suite_cases(curated_suite_file)
    metadata_by_file = {
        str(Path(entry.filepath).resolve()): entry
        for entry in base_index.entries
        if entry.file_type == "rwl"
    }

    cases: list[BenchmarkCase] = []
    for rwl_path in sorted(reference_dir.rglob("*.rwl")):
        try:
            rwl = parse_rwl_file(rwl_path)
        except Exception:
            continue

        metadata = metadata_by_file.get(str(rwl_path.resolve()))
        eligible: list[BenchmarkCase] = []
        for series_id, series in sorted(rwl.series.items()):
            if series.length < min_length:
                continue
            if series.start_year > max_start_year:
                continue
            if series.end_year < min_end_year:
                continue

            eligible.append(
                BenchmarkCase(
                    test_file=str(rwl_path.resolve()),
                    series_id=series_id,
                    state=rwl_path.parent.name.upper(),
                    species=metadata.species if metadata is not None else "",
                    site_id=metadata.site_id if metadata is not None else rwl_path.stem.upper(),
                    site_name=metadata.site_name if metadata is not None else rwl_path.stem,
                    true_start_year=int(series.start_year),
                    true_end_year=int(series.end_year),
                    length=int(series.length),
                    curated_suite_case=(str(rwl_path.resolve()), series_id) in curated_cases,
                )
            )

        if scope == "representative" and eligible:
            cases.append(eligible[0])
        else:
            cases.extend(eligible)

    return cases


def build_subset_index(base_index: ChronologyIndex, excluded_file: str) -> ChronologyIndex:
    """Clone the cached index while excluding a single source file."""
    subset = ChronologyIndex()
    subset.entries = [
        entry for entry in base_index.entries
        if Path(entry.filepath).name != excluded_file
    ]
    subset._by_species = defaultdict(list)
    subset._by_state = defaultdict(list)
    for entry in subset.entries:
        if entry.species:
            subset._by_species[entry.species].append(entry)
        if entry.state:
            subset._by_state[entry.state].append(entry)
    return subset


def classify_case_result(result: BenchmarkCaseResult) -> tuple[str, list[str]]:
    """Assign a failure bucket and likely-cause labels."""
    case = result.case
    causes: list[str] = []

    if result.top1_error is None:
        category = "no_match"
        if result.reference_count <= 3:
            causes.append("sparse_filtered_reference_coverage")
        if case.length < 120:
            causes.append("short_or_low_signal_series")
        causes.append("candidate_search_returned_no_usable_alignment")
        return category, causes

    if result.top1_abs_error is not None and result.top1_abs_error <= PASS_TOLERANCE_YEARS:
        if result.status == "recommended":
            return "correct_recommended", []
        return "correct_ranked", ["conservative_recommendation_policy"]

    if result.correct_year_rank is not None:
        category = "ranking_miss"
        causes.append("correct_year_survives_but_scores_below_false_peak")
        if result.top1_abs_error is not None and result.top1_abs_error >= 50:
            causes.append("long_offset_aliasing_not_penalized_enough")
        if result.reference_count >= 15:
            causes.append("species_filter_still_leaves_broad_search_space")
        return category, causes

    if result.top1_abs_error is not None and result.top1_abs_error <= NEAR_MISS_TOLERANCE_YEARS:
        return "near_miss", ["boundary_placement_or_outer_ring_instability"]

    if result.top1_abs_error is not None and result.top1_abs_error >= 50:
        category = "long_offset_false_positive"
        causes.append("false_high_correlation_peak_dominates_ranking")
        if result.top1_correlation is not None and result.top1_correlation >= 0.45:
            causes.append("cross_site_periodicity_or_aliasing")
        if result.reference_count >= 15:
            causes.append("ranking_features_lack_calendar_context_penalties")
        return category, causes

    category = "search_miss"
    if result.reference_count <= 5:
        causes.append("thin_reference_coverage_after_exclusion")
    else:
        causes.append("correct_year_not_surfaced_near_top_candidates")
    return category, causes


def run_benchmark_case(
    case: BenchmarkCase,
    *,
    base_index: ChronologyIndex,
    matcher_cache: dict[str, CrossdateMatcher],
    rwl_cache: dict[str, object],
    top_n: int = 20,
    min_overlap: int = 30,
    slice_config_path: Optional[str | Path] = None,
) -> BenchmarkCaseResult:
    """Run a single leave-one-file-out benchmark case."""
    excluded_name = Path(case.test_file).name
    matcher = matcher_cache.get(excluded_name)
    if matcher is None:
        subset_index = build_subset_index(base_index, excluded_name)
        matcher = CrossdateMatcher(index=subset_index)
        matcher_cache[excluded_name] = matcher

    rwl = rwl_cache.get(case.test_file)
    if rwl is None:
        rwl = parse_rwl_file(case.test_file)
        rwl_cache[case.test_file] = rwl
    series = rwl.series[case.series_id]

    report = matcher.date_sample(
        values=series.values,
        sample_name=case.series_id,
        has_bark_edge=True,
        species_filter=[case.species] if case.species else None,
        era_start=case.true_start_year - 50,
        era_end=case.true_end_year + 50,
        min_overlap=min_overlap,
        top_n=top_n,
    )

    top1 = report.best_candidate
    correct_year_rank = None
    for rank, candidate in enumerate(report.candidates, start=1):
        if abs(candidate.outer_ring_year - case.true_end_year) <= PASS_TOLERANCE_YEARS:
            correct_year_rank = rank
            break

    result = BenchmarkCaseResult(
        case=case,
        status=report.status,
        reference_count=int(report.diagnostics.get("reference_count", 0)),
        candidate_count=len(report.candidates),
        top1_outer_ring_year=top1.outer_ring_year if top1 is not None else None,
        top1_error=(top1.outer_ring_year - case.true_end_year) if top1 is not None else None,
        top1_reference_name=top1.reference_name if top1 is not None else "",
        top1_reference_state=top1.reference_state if top1 is not None else "",
        top1_reference_species=top1.reference_species if top1 is not None else "",
        top1_score=top1.composite_score if top1 is not None else None,
        top1_correlation=top1.correlation if top1 is not None else None,
        top1_t_value=top1.t_value if top1 is not None else None,
        correct_year_rank=correct_year_rank,
        correct_year_in_top5=(correct_year_rank is not None and correct_year_rank <= 5),
        correct_year_in_top10=(correct_year_rank is not None and correct_year_rank <= 10),
        warning_count=len(report.warnings),
    )
    result.category, result.likely_causes = classify_case_result(result)
    return assign_benchmark_slices(result, slice_config_path=slice_config_path)


def sweep_corpus(
    reference_dir: str | Path,
    *,
    scope: str = "all",
    curated_suite_file: Optional[str | Path] = None,
    top_n: int = 20,
    min_overlap: int = 30,
    progress_every: int = 0,
    max_workers: int = 1,
    slice_config_path: Optional[str | Path] = None,
) -> list[BenchmarkCaseResult]:
    """Run a leave-one-file-out sweep across the corpus."""
    reference_dir = Path(reference_dir)
    cases = discover_benchmark_cases(
        reference_dir,
        scope=scope,
        curated_suite_file=curated_suite_file,
    )

    if max_workers <= 1:
        base_index = ChronologyIndex(reference_dir)
        matcher_cache: dict[str, CrossdateMatcher] = {}
        rwl_cache: dict[str, object] = {}
        results: list[BenchmarkCaseResult] = []
        total = len(cases)
        for index, case in enumerate(cases, start=1):
            results.append(
                run_benchmark_case(
                    case,
                    base_index=base_index,
                    matcher_cache=matcher_cache,
                    rwl_cache=rwl_cache,
                    top_n=top_n,
                    min_overlap=min_overlap,
                    slice_config_path=slice_config_path,
                )
            )
            if progress_every and index % progress_every == 0:
                print(f"Processed {index}/{total} cases...", flush=True)
        return results

    grouped_cases: dict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        grouped_cases[case.test_file].append(case)

    max_workers = max(1, min(max_workers, os.cpu_count() or 1))
    results: list[BenchmarkCaseResult] = []
    total = len(cases)
    processed = 0

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(str(reference_dir),),
    ) as executor:
        futures = [
            executor.submit(
                _run_file_cases,
                file_path,
                file_cases,
                top_n=top_n,
                min_overlap=min_overlap,
                slice_config_path=str(slice_config_path) if slice_config_path is not None else None,
            )
            for file_path, file_cases in grouped_cases.items()
        ]
        for future in as_completed(futures):
            file_results = future.result()
            results.extend(file_results)
            processed += len(file_results)
            if progress_every:
                print(f"Processed {processed}/{total} cases...", flush=True)

    results.sort(key=lambda result: (result.case.test_file, result.case.series_id))
    return results


def summarize_corpus_results(
    results: list[BenchmarkCaseResult],
    *,
    slice_config_path: Optional[str | Path] = None,
) -> dict:
    """Aggregate corpus sweep results into actionable buckets."""
    category_counts = Counter(result.category for result in results)
    status_counts = Counter(result.status for result in results)
    likely_cause_counts = Counter(
        cause
        for result in results
        for cause in result.likely_causes
    )

    def length_bucket(length: int) -> str:
        if length < 100:
            return "050-099"
        if length < 150:
            return "100-149"
        if length < 250:
            return "150-249"
        return "250+"

    by_state: dict[str, list[BenchmarkCaseResult]] = defaultdict(list)
    by_species: dict[str, list[BenchmarkCaseResult]] = defaultdict(list)
    by_length_bucket: dict[str, list[BenchmarkCaseResult]] = defaultdict(list)
    by_primary_slice: dict[str, list[BenchmarkCaseResult]] = defaultdict(list)
    by_overlay: dict[str, list[BenchmarkCaseResult]] = defaultdict(list)

    for result in results:
        by_state[result.case.state].append(result)
        by_species[result.case.species or "UNKNOWN"].append(result)
        by_length_bucket[length_bucket(result.case.length)].append(result)
        by_primary_slice[result.primary_slice_id or "unclassified"].append(result)
        for overlay_id in result.overlay_ids:
            by_overlay[overlay_id].append(result)

    representative_failures = []
    for result in sorted(
        [result for result in results if not result.passed_top1],
        key=lambda result: (
            result.top1_abs_error is None,
            -(result.top1_abs_error or 0),
            result.case.length,
        ),
    )[:20]:
        representative_failures.append(
            {
                "test_file": result.case.test_file,
                "series_id": result.case.series_id,
                "state": result.case.state,
                "species": result.case.species,
                "length": result.case.length,
                "true_end_year": result.case.true_end_year,
                "top1_outer_ring_year": result.top1_outer_ring_year,
                "top1_error": result.top1_error,
                "top1_reference_name": result.top1_reference_name,
                "category": result.category,
                "likely_causes": result.likely_causes,
                "correct_year_rank": result.correct_year_rank,
            }
        )

    benchmark_slice_catalog = load_benchmark_slice_catalog(slice_config_path)
    primary_summaries = []
    for slice_def in benchmark_slice_catalog.primary_slices:
        primary_summaries.append(
            _slice_summary(
                slice_def,
                by_primary_slice.get(slice_def.id, []),
                kind="primary",
            )
        )
    if "unclassified" in by_primary_slice:
        primary_summaries.append(
            {
                "id": "unclassified",
                "kind": "primary",
                "description": "Cases that did not match any configured primary benchmark slice.",
                "count": len(by_primary_slice["unclassified"]),
                "summary": _result_group_summary(by_primary_slice["unclassified"]),
                "targets": {},
                "target_checks": {},
                "targets_met": None,
                "unmet_targets": [],
            }
        )

    overlay_summaries = [
        _slice_summary(
            slice_def,
            by_overlay.get(slice_def.id, []),
            kind="overlay",
        )
        for slice_def in benchmark_slice_catalog.overlays
    ]

    return {
        "overall": _result_group_summary(results),
        "category_counts": dict(category_counts),
        "status_counts": dict(status_counts),
        "likely_cause_counts": dict(likely_cause_counts),
        "state_summary": {
            key: _result_group_summary(value)
            for key, value in sorted(by_state.items())
        },
        "species_summary": {
            key: _result_group_summary(value)
            for key, value in sorted(by_species.items())
        },
        "length_bucket_summary": {
            key: _result_group_summary(value)
            for key, value in sorted(by_length_bucket.items())
        },
        "benchmark_slices": {
            "config_path": benchmark_slice_catalog.source_path,
            "taxonomy_order": list(benchmark_slice_catalog.taxonomy_order),
            "primary_slices": primary_summaries,
            "overlays": overlay_summaries,
        },
        "representative_failures": representative_failures,
    }


def _init_worker(reference_dir: str):
    global _WORKER_BASE_INDEX
    _WORKER_BASE_INDEX = ChronologyIndex(reference_dir)


def _run_file_cases(
    file_path: str,
    file_cases: list[BenchmarkCase],
    *,
    top_n: int,
    min_overlap: int,
    slice_config_path: Optional[str | Path] = None,
) -> list[BenchmarkCaseResult]:
    global _WORKER_BASE_INDEX
    if _WORKER_BASE_INDEX is None:
        raise RuntimeError("Worker index was not initialized")

    subset_index = build_subset_index(_WORKER_BASE_INDEX, Path(file_path).name)
    matcher = CrossdateMatcher(index=subset_index)
    rwl = parse_rwl_file(file_path)
    results: list[BenchmarkCaseResult] = []

    for case in file_cases:
        series = rwl.series[case.series_id]
        report = matcher.date_sample(
            values=series.values,
            sample_name=case.series_id,
            has_bark_edge=True,
            species_filter=[case.species] if case.species else None,
            era_start=case.true_start_year - 50,
            era_end=case.true_end_year + 50,
            min_overlap=min_overlap,
            top_n=top_n,
        )

        top1 = report.best_candidate
        correct_year_rank = None
        for rank, candidate in enumerate(report.candidates, start=1):
            if abs(candidate.outer_ring_year - case.true_end_year) <= PASS_TOLERANCE_YEARS:
                correct_year_rank = rank
                break

        result = BenchmarkCaseResult(
            case=case,
            status=report.status,
            reference_count=int(report.diagnostics.get("reference_count", 0)),
            candidate_count=len(report.candidates),
            top1_outer_ring_year=top1.outer_ring_year if top1 is not None else None,
            top1_error=(top1.outer_ring_year - case.true_end_year) if top1 is not None else None,
            top1_reference_name=top1.reference_name if top1 is not None else "",
            top1_reference_state=top1.reference_state if top1 is not None else "",
            top1_reference_species=top1.reference_species if top1 is not None else "",
            top1_score=top1.composite_score if top1 is not None else None,
            top1_correlation=top1.correlation if top1 is not None else None,
            top1_t_value=top1.t_value if top1 is not None else None,
            correct_year_rank=correct_year_rank,
            correct_year_in_top5=(correct_year_rank is not None and correct_year_rank <= 5),
            correct_year_in_top10=(correct_year_rank is not None and correct_year_rank <= 10),
            warning_count=len(report.warnings),
        )
        result.category, result.likely_causes = classify_case_result(result)
        results.append(assign_benchmark_slices(result, slice_config_path=slice_config_path))

    return results
