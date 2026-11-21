import json
import math
from pathlib import Path

import numpy as np
import pytest

import dendro.crossdating.matcher as matcher_module
from dendro.crossdating.correlator import CorrelationResult
from dendro.crossdating.detrend import DetrendMethod
from dendro.crossdating.matcher import RECOMMEND_MIN_T_VALUE, CrossdateMatcher
from dendro.reference.chronology_index import ChronologyIndex, MasterChronology, ReferenceManifestEntry
from dendro.reference.tucson_parser import parse_rwl_file


REF_DIR = Path(__file__).parent.parent / "data" / "reference"
SKIP_IF_NO_DATA = pytest.mark.skipif(
    not REF_DIR.exists() or not any(REF_DIR.rglob("*.rwl")),
    reason="Reference data not downloaded. Run 'dendro download' first.",
)


def _synthetic_entry(site_id, site_name, start_year, values, state="NH"):
    """Reference entry whose master is used by the matcher exactly as given."""
    values = [float(value) for value in values]
    end_year = start_year + len(values) - 1
    master = MasterChronology(
        start_year=start_year,
        end_year=end_year,
        values=values,
        sample_depth=[10] * len(values),
        build_method="rwl-log-hp20-ar1-biweight",
        detrend_method="spline",
    )
    return ReferenceManifestEntry(
        filepath=f"/nonexistent/{site_id}.rwl",
        site_id=site_id,
        site_name=site_name,
        species="TSCA",
        state=state,
        start_year=start_year,
        end_year=end_year,
        num_years=len(values),
        num_series=10,
        file_type="rwl",
        material_group="hemlock",
        master=master,
    )


def _matcher_for(*entries):
    index = ChronologyIndex()
    for entry in entries:
        index._add_entry(entry)
    return CrossdateMatcher(index=index)


def _two_agreeing_sites(seed=99):
    """Two independent sites sharing one signal; the sample dates to 1859."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0, 1.0, size=200)
    sample = signal[60:160].copy()
    entries = [
        _synthetic_entry(f"SITE{i}", f"Independent Site {i}", 1700, signal + rng.normal(0.0, 0.1, size=200))
        for i in range(2)
    ]
    return _matcher_for(*entries), sample


@SKIP_IF_NO_DATA
def test_known_series_report_is_serializable_and_ranked():
    matcher = CrossdateMatcher(reference_dir=REF_DIR)
    rwl = parse_rwl_file(REF_DIR / "nh" / "nh001.rwl")
    series = rwl.series["297031"]

    report = matcher.date_sample(
        series.values,
        sample_name="297031",
        era_start=1500,
        era_end=2000,
        top_n=5,
    )
    payload = report.to_dict()

    assert report.status in {"recommended", "ranked"}
    assert payload["best_candidate"] is not None
    assert payload["best_candidate"]["outer_ring_year"] == series.end_year
    assert payload["sample"]["chosen_orientation"] == "oldest_to_newest"
    assert payload["candidates"][0]["composite_score"] >= payload["candidates"][1]["composite_score"]

    json.dumps(payload)


@SKIP_IF_NO_DATA
def test_auto_orientation_recovers_reversed_measurements():
    matcher = CrossdateMatcher(reference_dir=REF_DIR)
    rwl = parse_rwl_file(REF_DIR / "nh" / "nh001.rwl")
    series = rwl.series["297031"]

    report = matcher.date_sample(
        series.values[::-1],
        sample_name="297031-reversed",
        orientation="auto",
        era_start=1500,
        era_end=2000,
        top_n=3,
    )

    assert report.best_candidate is not None
    assert report.best_candidate.outer_ring_year == series.end_year
    assert report.chosen_orientation == "bark_to_pith"


def test_year_consensus_boost_prefers_dense_year_cluster():
    """Two sites agreeing on a year outrank a stronger single-site peak at another year."""
    rng = np.random.default_rng(22)
    signal = np.exp(rng.normal(0.0, 0.3, size=300))  # regional signal, 1700..1999
    sample = signal[150:250].copy()  # true rings 1850..1949
    cluster = [
        _synthetic_entry(f"CORRECT{i}", name, 1700, signal * np.exp(rng.normal(0.0, 0.33, size=300)), state=state)
        for i, (name, state) in enumerate([("Correct Cluster A", "NY"), ("Correct Cluster B", "NH")])
    ]
    # An unrelated site holding a cleaner copy of the sample at the wrong date (1720..1819).
    wrong_values = np.exp(rng.normal(0.0, 0.3, size=300))
    wrong_values[20:120] = sample * np.exp(rng.normal(0.0, 0.22, size=100))
    wrong = _synthetic_entry("WRONG1", "Wrong Peak", 1700, wrong_values, state="NY")

    report = _matcher_for(*cluster, wrong).date_sample(
        sample,
        sample_name="consensus-sample",
        orientation="oldest_to_newest",
        era_start=1700,
        era_end=1999,
        min_overlap=30,
    )

    def without_bonus(candidate):
        return candidate.composite_score - candidate.year_cluster_bonus

    wrong_peak = next(
        candidate
        for candidate in report.candidates
        if candidate.reference_id == "WRONG1" and candidate.outer_ring_year == 1819
    )
    correct = [
        candidate
        for candidate in report.candidates
        if candidate.reference_id.startswith("CORRECT") and candidate.outer_ring_year == 1949
    ]
    assert len(correct) == 2
    # Premise: on its own evidence the wrong peak is the strongest alignment.
    assert without_bonus(wrong_peak) > max(without_bonus(candidate) for candidate in correct)
    assert wrong_peak.t_value > max(candidate.t_value for candidate in correct)

    best = report.candidates[0]
    assert best.outer_ring_year == 1949
    assert best.year_cluster_support == 2
    assert wrong_peak.year_cluster_support == 1
    assert best.year_cluster_bonus > wrong_peak.year_cluster_bonus
    assert "year-consensus=2" in best.rationale
    assert "isolated-year candidate" in wrong_peak.rationale
    assert not any("year-consensus=" in rationale for rationale in wrong_peak.rationale)


@pytest.mark.parametrize("detrend_method", [DetrendMethod.NONE, DetrendMethod.SPLINE], ids=["none", "spline"])
def test_era_filter_finds_in_era_match_even_with_stronger_out_of_era_peak(detrend_method):
    """Era bounds must restrict the lag search rather than dropping the whole reference.

    The ``none`` case compares the planted values as constructed, so it isolates the era
    handling from sample conditioning; ``spline`` covers the default conditioning.
    """
    rng = np.random.default_rng(42)
    ref_vals = np.exp(rng.normal(0.0, 0.3, size=300))  # ring-width-like, 1700..1999
    # A noisy match inside the era (rings 1770..1829) ...
    sample = ref_vals[70:130] * np.exp(rng.normal(0.0, 0.35, size=60))
    # ... and a near-exact copy of the sample outside it (rings 1920..1979).
    ref_vals[220:280] = sample * np.exp(rng.normal(0.0, 0.03, size=60))
    matcher = _matcher_for(_synthetic_entry("ERATEST", "Era Test Site", 1700, ref_vals))

    def run(era_start, era_end):
        return matcher.date_sample(
            sample,
            sample_name="era-sample",
            orientation="oldest_to_newest",
            era_start=era_start,
            era_end=era_end,
            detrend_method=detrend_method,
            min_overlap=30,
        )

    # Premise: over the whole reference the out-of-era copy is the best alignment.
    unrestricted = run(1750, 1999).best_candidate
    assert unrestricted is not None
    assert unrestricted.outer_ring_year == 1979

    restricted = run(1750, 1860).best_candidate
    assert restricted is not None
    assert restricted.outer_ring_year == 1829
    assert restricted.t_value < unrestricted.t_value


def test_partial_overlap_computes_segment_consistency():
    """Partial edge overlap must still compute segment correlations instead of returning 0."""
    import numpy as np
    from dendro.reference.chronology_index import ChronologyIndex, MasterChronology, ReferenceManifestEntry

    rng = np.random.default_rng(77)
    ref_vals = rng.normal(0.0, 1.0, size=120)  # 1700..1819
    # Sample spans 1750..1849 (70-year overlap with ref end, 30 years past ref end)
    sample = np.concatenate([ref_vals[50:120], rng.normal(0.0, 1.0, size=30)])

    master = MasterChronology(
        start_year=1700,
        end_year=1819,
        values=ref_vals.tolist(),
        sample_depth=[8] * 120,
        build_method="rwl-log-hp20-ar1-biweight",
        detrend_method="spline",
    )
    entry = ReferenceManifestEntry(
        filepath="/tmp/partial.rwl",
        site_id="PARTIAL",
        site_name="Partial Overlap Site",
        species="TSCA",
        state="VT",
        start_year=1700,
        end_year=1819,
        num_years=120,
        num_series=8,
        file_type="rwl",
        material_group="hemlock",
        master=master,
    )
    index = ChronologyIndex()
    index._add_entry(entry)
    matcher = CrossdateMatcher(index=index)

    report = matcher.date_sample(
        sample,
        sample_name="partial-sample",
        orientation="oldest_to_newest",
        era_start=1800,
        era_end=1870,
        min_overlap=30,
    )
    assert report.best_candidate is not None
    assert report.best_candidate.outer_ring_year == 1849
    assert len(report.best_candidate.segment_correlations) > 0
    assert report.best_candidate.segment_consistency > 0.5


def test_distinct_year_score_gap_does_not_penalize_multi_site_agreement():
    """Two sites agreeing on the same year must not collapse score_gap_to_next to 0."""
    import numpy as np
    from dendro.reference.chronology_index import ChronologyIndex, MasterChronology, ReferenceManifestEntry

    rng = np.random.default_rng(99)
    signal = rng.normal(0.0, 1.0, size=200)
    sample = signal[60:160].copy()

    index = ChronologyIndex()
    for i in range(2):
        noisy = signal + rng.normal(0.0, 0.1, size=200)
        master = MasterChronology(
            start_year=1700,
            end_year=1899,
            values=noisy.tolist(),
            sample_depth=[10] * 200,
            build_method="rwl-log-hp20-ar1-biweight",
            detrend_method="spline",
        )
        index._add_entry(
            ReferenceManifestEntry(
                filepath=f"/tmp/site_{i}.rwl",
                site_id=f"SITE{i}",
                site_name=f"Independent Site {i}",
                species="TSCA",
                state="NH",
                start_year=1700,
                end_year=1899,
                num_years=200,
                num_series=10,
                file_type="rwl",
                material_group="hemlock",
                master=master,
            )
        )

    matcher = CrossdateMatcher(index=index)
    report = matcher.date_sample(
        sample,
        sample_name="consensus-sample",
        orientation="oldest_to_newest",
        era_start=1750,
        era_end=1900,
        min_overlap=30,
    )
    assert report.best_candidate is not None
    assert report.best_candidate.outer_ring_year == 1859
    assert report.best_candidate.score_gap_to_next > 0.2
    assert report.status == "recommended"


def _score(matcher, entry, correlation, t_value):
    return matcher._score_candidate(
        result=CorrelationResult(
            position=1800,
            correlation=correlation,
            t_value=t_value,
            p_value=0.0,
            overlap=60,
            gleichlauf=50.0,
        ),
        segment_consistency=0.0,
        entry=entry,
        species_filter=None,
        state_filter=None,
        sample_length=60,
    )


def test_only_positive_infinite_t_earns_the_capped_t_score():
    """t = -inf (r = -1) and t = NaN must score like no t-evidence, not like a perfect match."""
    entry = _synthetic_entry("TSCORE", "T Score Site", 1800, np.arange(1.0, 61.0))
    matcher = _matcher_for(entry)

    no_evidence = _score(matcher, entry, -1.0, 0.0)
    assert _score(matcher, entry, -1.0, float("-inf")) == pytest.approx(no_evidence)
    assert _score(matcher, entry, -1.0, float("nan")) == pytest.approx(no_evidence)
    assert _score(matcher, entry, 1.0, float("inf")) == pytest.approx(_score(matcher, entry, 1.0, 15.0))
    assert _score(matcher, entry, 1.0, float("inf")) > _score(matcher, entry, 1.0, 0.0)


def test_perfect_anticorrelation_is_not_ranked_as_a_match():
    """A reference that is the exact negation of the sample (r = -1, t = -inf) is no match."""
    sample = np.arange(1.0, 61.0)
    matcher = _matcher_for(_synthetic_entry("NEGATED", "Negated Site", 1800, -sample))

    report = matcher.date_sample(
        sample,
        sample_name="anticorrelated",
        orientation="oldest_to_newest",
        era_start=1800,
        era_end=1900,
        detrend_method=DetrendMethod.NONE,
        min_overlap=30,
    )

    best = report.best_candidate
    assert best is not None
    assert best.correlation == pytest.approx(-1.0)
    assert best.t_value == float("-inf")
    assert best.composite_score < 0.25
    assert report.status == "inconclusive"


def test_nan_and_negative_infinite_t_never_win_a_reference_peak(monkeypatch):
    """Peak selection must prefer a real positive alignment over t = NaN and t = -inf lags."""

    def fake_sliding_stats(sample, reference, reference_start_year, **kwargs):
        return (
            np.array([1800, 1810, 1820], dtype=np.int64),
            np.array([-1.0, np.nan, 0.5]),
            np.array([-np.inf, np.nan, 4.4]),
            np.array([0.0, np.nan, 1e-4]),
            np.array([60, 60, 60], dtype=np.int64),
            np.array([0.0, 50.0, 65.0]),
        )

    monkeypatch.setattr(matcher_module, "_compute_sliding_stats", fake_sliding_stats)
    rng = np.random.default_rng(5)
    matcher = _matcher_for(_synthetic_entry("PEAKS", "Peak Site", 1780, np.exp(rng.normal(0.0, 0.3, size=150))))

    report = matcher.date_sample(
        np.exp(rng.normal(0.0, 0.3, size=60)),
        sample_name="peaks",
        orientation="oldest_to_newest",
        era_start=1800,
        era_end=1900,
        min_overlap=30,
    )

    assert [candidate.proposed_start_year for candidate in report.candidates] == [1820]
    assert all(math.isfinite(candidate.t_value) for candidate in report.candidates)


def test_segment_indices_are_relative_to_the_sample_for_partial_overlap():
    """Segment start indices count sample rings, also when the overlap begins mid-sample."""
    rng = np.random.default_rng(7)
    ref_vals = np.exp(rng.normal(0.0, 0.3, size=180))  # 1720..1899
    # 80-ring sample starting 1700: its first 20 rings pre-date the reference.
    sample = np.concatenate([np.exp(rng.normal(0.0, 0.3, size=20)), ref_vals[:60]])
    matcher = _matcher_for(_synthetic_entry("LATEREF", "Late Reference Site", 1720, ref_vals))

    report = matcher.date_sample(
        sample,
        sample_name="early-sample",
        orientation="oldest_to_newest",
        era_start=1750,
        era_end=1850,
        min_overlap=30,
    )

    best = report.best_candidate
    assert best is not None
    assert best.proposed_start_year == 1700
    assert best.overlap == 60
    starts = [start for start, _, _ in best.segment_correlations]
    assert starts == [20, 35, 50]
    # The calendar year a plot or report derives for the first segment is the reference start.
    assert best.proposed_start_year + starts[0] == 1720
    assert [segment["start_index"] for segment in best.to_dict()["segment_correlations"]] == starts


def test_segment_indices_start_at_zero_for_full_overlap():
    matcher, sample = _two_agreeing_sites()

    report = matcher.date_sample(
        sample,
        sample_name="full-overlap",
        orientation="oldest_to_newest",
        era_start=1750,
        era_end=1900,
        min_overlap=30,
    )

    assert report.best_candidate.overlap == len(sample)
    assert report.best_candidate.segment_correlations[0][0] == 0


@pytest.mark.parametrize("top_n", [0, -3])
def test_top_n_below_one_still_returns_the_best_candidate(top_n):
    """A recommended report must never come back with an empty candidate list."""
    matcher, sample = _two_agreeing_sites()

    report = matcher.date_sample(
        sample,
        sample_name="top-n",
        orientation="oldest_to_newest",
        era_start=1750,
        era_end=1900,
        min_overlap=30,
        top_n=top_n,
    )

    assert report.status == "recommended"
    assert len(report.candidates) == 1
    assert report.best_candidate.outer_ring_year == 1859
    assert report.to_dict()["best_candidate"] is not None


def _t_rationale(t_value):
    entry = _synthetic_entry("RATIONALE", "Rationale Site", 1800, np.arange(1.0, 61.0))
    matcher = _matcher_for(entry)
    rationale = matcher._candidate_rationale(
        CorrelationResult(
            position=1800,
            correlation=0.6,
            t_value=t_value,
            p_value=0.0,
            overlap=60,
            gleichlauf=70.0,
        ),
        1.0,
        True,
        entry,
    )
    return [text for text in rationale if "t-value" in text]


def test_t_value_rationale_follows_the_recommendation_gate():
    """Only a t-value that clears RECOMMEND_MIN_T_VALUE may be described as high."""
    assert _t_rationale(RECOMMEND_MIN_T_VALUE) == ["high t-value"]
    assert _t_rationale(RECOMMEND_MIN_T_VALUE + 5.0) == ["high t-value"]
    assert _t_rationale(float("inf")) == ["high t-value"]
    assert _t_rationale(RECOMMEND_MIN_T_VALUE - 0.01) == ["useful but not secure t-value"]
    assert _t_rationale(4.0) == ["useful but not secure t-value"]
    assert _t_rationale(3.99) == ["borderline t-value"]


def test_recommended_candidate_rationale_never_says_not_secure():
    matcher, sample = _two_agreeing_sites()

    report = matcher.date_sample(
        sample,
        sample_name="rationale",
        orientation="oldest_to_newest",
        era_start=1750,
        era_end=1900,
        min_overlap=30,
    )

    assert report.status == "recommended"
    assert report.best_candidate.t_value >= RECOMMEND_MIN_T_VALUE
    assert "high t-value" in report.best_candidate.rationale
    assert not any("not secure" in text for text in report.best_candidate.rationale)
    for candidate in report.candidates:
        if any("not secure" in text for text in candidate.rationale):
            assert candidate.t_value < RECOMMEND_MIN_T_VALUE
