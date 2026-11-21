"""Tests for cross-dating correlation algorithms."""

import numpy as np
import pytest

from scipy import stats

from dendro.crossdating import correlator as correlator_module
from dendro.crossdating.correlator import (
    sliding_correlation,
    calculate_tvalue,
    calculate_gleichlauf,
    find_best_match,
    segment_correlation,
    dating_confidence,
    CorrelationResult,
)


def _naive_gleichlauf(sample, reference, lag):
    """Independent GLK reference: plain loop over calendar-consecutive year pairs.

    A year-to-year change is counted only when both of its years lie inside the
    overlap and are non-NaN in both series. A zero change in either series
    scores 0.5. Returns (glk_percent, n_counted_changes); glk is 0.0 when no
    change can be counted.
    """
    score = 0.0
    counted = 0
    for i in range(len(sample) - 1):
        j = i + lag
        if j < 0 or j + 1 >= len(reference):
            continue
        quad = (sample[i], sample[i + 1], reference[j], reference[j + 1])
        if any(np.isnan(v) for v in quad):
            continue
        counted += 1
        d_sample = sample[i + 1] - sample[i]
        d_reference = reference[j + 1] - reference[j]
        if d_sample == 0 or d_reference == 0:
            score += 0.5
        elif (d_sample > 0) == (d_reference > 0):
            score += 1.0
    if counted == 0:
        return 0.0, 0
    return 100.0 * score / counted, counted


def _brute_force_sliding(sample, reference, start_year, min_overlap,
                         min_position=None, max_position=None):
    """Independent per-lag reference: scipy pearsonr on each window, naive GLK.

    Returns {position: (r, t, p, overlap, glk)}. Windows with fewer than
    ``min_overlap`` jointly valid years, or that are constant in either series
    (where Pearson is undefined), are skipped.
    """
    sample = np.asarray(sample, dtype=float)
    reference = np.asarray(reference, dtype=float)
    expected = {}
    for lag in range(-len(sample) + min_overlap, len(reference) - min_overlap + 1):
        position = start_year + lag
        if min_position is not None and position < min_position:
            continue
        if max_position is not None and position > max_position:
            continue
        xs, ys = [], []
        for i in range(len(sample)):
            j = i + lag
            if 0 <= j < len(reference) and not np.isnan(sample[i]) and not np.isnan(reference[j]):
                xs.append(sample[i])
                ys.append(reference[j])
        n = len(xs)
        if n < min_overlap or len(set(xs)) < 2 or len(set(ys)) < 2:
            continue
        r, p = stats.pearsonr(xs, ys)
        if n < 3:
            t, p = 0.0, 1.0
        elif abs(r) >= 1.0:
            t = float("inf") if r > 0 else float("-inf")
        else:
            t = r * np.sqrt((n - 2) / (1.0 - r * r))
        glk, _ = _naive_gleichlauf(sample, reference, lag)
        expected[position] = (float(r), float(t), float(p), n, glk)
    return expected


def _assert_matches_brute_force(sample, reference, start_year, min_overlap, **bounds):
    expected = _brute_force_sliding(sample, reference, start_year, min_overlap, **bounds)
    results = sliding_correlation(sample, reference, start_year, min_overlap=min_overlap, **bounds)
    assert [r.position for r in results] == sorted(expected)
    assert len(results) > 0
    for got in results:
        exp_r, exp_t, exp_p, exp_n, exp_glk = expected[got.position]
        where = f"position {got.position}"
        assert got.overlap == exp_n, where
        assert abs(got.correlation - exp_r) < 1e-12, where
        np.testing.assert_allclose(got.t_value, exp_t, rtol=1e-10, atol=1e-12, err_msg=where)
        np.testing.assert_allclose(got.p_value, exp_p, rtol=1e-8, atol=0.0, err_msg=where)
        assert abs(got.gleichlauf - exp_glk) < 1e-9, where
    return results, expected


class TestCalculateTvalue:
    """Tests for t-value calculation."""

    def test_perfect_correlation(self):
        """T-value for r=1 should be infinite."""
        t = calculate_tvalue(1.0, 50)
        assert t == float('inf')

    def test_zero_correlation(self):
        """T-value for r=0 should be 0."""
        t = calculate_tvalue(0.0, 50)
        assert t == 0.0

    def test_moderate_correlation(self):
        """Test t-value for moderate correlation."""
        t = calculate_tvalue(0.5, 50)
        # t = 0.5 * sqrt(48 / 0.75) = 0.5 * sqrt(64) = 0.5 * 8 = 4.0
        assert abs(t - 4.0) < 0.01

    def test_negative_correlation(self):
        """T-value for negative correlation should be negative."""
        t = calculate_tvalue(-0.5, 50)
        assert t < 0

    def test_small_sample(self):
        """Test with small sample size."""
        t = calculate_tvalue(0.5, 10)
        # Should still compute but with fewer degrees of freedom
        assert t > 0


class TestCalculateGleichlauf:
    """Tests for Gleichläufigkeit calculation."""

    def test_perfect_agreement(self):
        """GLK for perfectly correlated series should be 100%."""
        s1 = np.array([1, 2, 3, 4, 5])
        s2 = np.array([10, 20, 30, 40, 50])

        glk = calculate_gleichlauf(s1, s2)
        assert glk == 100.0

    def test_perfect_disagreement(self):
        """GLK for inverse series should be 0%."""
        s1 = np.array([1, 2, 3, 4, 5])
        s2 = np.array([50, 40, 30, 20, 10])

        glk = calculate_gleichlauf(s1, s2)
        assert glk == 0.0

    def test_partial_agreement(self):
        """Test partial agreement."""
        s1 = np.array([1, 2, 3, 2, 3])  # +, +, -, +
        s2 = np.array([1, 2, 1, 2, 3])  # +, -, +, +

        glk = calculate_gleichlauf(s1, s2)
        # Agreement on 2 out of 4 changes = 50%
        assert 40 <= glk <= 60


class TestSlidingCorrelation:
    """Tests for sliding correlation."""

    def test_perfect_match(self):
        """Test finding a perfect match."""
        # Reference: known signal starting at year 1800
        reference = np.sin(np.linspace(0, 4 * np.pi, 100))

        # Sample: same signal (subset)
        sample = reference[30:60].copy()

        results = sliding_correlation(
            sample=sample,
            reference=reference,
            reference_start_year=1800,
            min_overlap=20,
        )

        assert len(results) > 0

        # Best match should be at position 1830 (1800 + 30)
        best = max(results, key=lambda r: r.correlation)
        assert abs(best.position - 1830) <= 2
        assert best.correlation > 0.95

    def test_no_match_too_short(self):
        """Test that short series returns empty results."""
        reference = np.random.randn(50)
        sample = np.random.randn(10)

        results = sliding_correlation(sample, reference, 1800, min_overlap=30)
        assert len(results) == 0

    def test_results_have_all_fields(self):
        """Test that results have all required fields."""
        reference = np.random.randn(100)
        sample = np.random.randn(40)

        results = sliding_correlation(sample, reference, 1800, min_overlap=30)

        if results:
            r = results[0]
            assert isinstance(r.position, (int, np.integer))
            assert isinstance(r.correlation, (float, np.floating))
            assert isinstance(r.t_value, (float, np.floating))
            assert isinstance(r.p_value, (float, np.floating))
            assert isinstance(r.overlap, (int, np.integer))
            assert isinstance(r.gleichlauf, (float, np.floating))


class TestFindBestMatch:
    """Tests for best match finding."""

    def test_returns_sorted_results(self):
        """Test that results are sorted by t-value."""
        reference = np.random.randn(200) + np.sin(np.linspace(0, 10 * np.pi, 200))
        sample = reference[50:100] + np.random.randn(50) * 0.1

        results = find_best_match(sample, reference, 1800, min_overlap=30, n_best=5)

        assert len(results) <= 5

        # Should be sorted by t-value descending
        for i in range(len(results) - 1):
            assert results[i].t_value >= results[i + 1].t_value


class TestDatingConfidence:
    """Tests for confidence assessment."""

    def test_high_confidence(self):
        """Test HIGH confidence criteria."""
        result = CorrelationResult(
            position=1800,
            correlation=0.65,
            t_value=7.0,
            p_value=0.001,
            overlap=60,
            gleichlauf=75,
        )
        assert dating_confidence(result) == "HIGH"

    def test_medium_confidence(self):
        """Test MEDIUM confidence criteria."""
        result = CorrelationResult(
            position=1800,
            correlation=0.50,
            t_value=4.5,
            p_value=0.01,
            overlap=40,
            gleichlauf=65,
        )
        assert dating_confidence(result) == "MEDIUM"

    def test_low_confidence(self):
        """Test LOW confidence criteria."""
        result = CorrelationResult(
            position=1800,
            correlation=0.30,
            t_value=2.0,
            p_value=0.1,
            overlap=25,
            gleichlauf=55,
        )
        assert dating_confidence(result) == "LOW"


class TestVectorizedSlidingEquivalence:
    """Verify vectorized sliding_correlation matches exact scipy pearsonr + naive GLK."""

    def test_exact_equivalence_and_nan_handling(self):
        rng = np.random.default_rng(123)
        ref = rng.normal(0.0, 1.0, size=100)
        sample = ref[25:65] + rng.normal(0.0, 0.15, size=40)
        ref[40] = np.nan
        sample[10] = np.nan

        results = sliding_correlation(sample, ref, reference_start_year=1800, min_overlap=25)
        by_pos = {r.position: r for r in results}
        assert 1825 in by_pos

        for pos in [1805, 1825, 1850]:
            lag = pos - 1800
            s0, s1 = max(0, -lag), min(len(sample), len(ref) - lag)
            r0, r1 = max(0, lag), min(len(ref), lag + len(sample))
            s_seg = sample[s0:s1]
            r_seg = ref[r0:r1]
            mask = ~(np.isnan(s_seg) | np.isnan(r_seg))
            expected_r, expected_p = stats.pearsonr(s_seg[mask], r_seg[mask])
            expected_glk, _ = _naive_gleichlauf(sample, ref, lag)
            got = by_pos[pos]
            assert got.overlap == int(np.sum(mask))
            assert abs(got.correlation - expected_r) < 1e-10
            assert abs(got.p_value - expected_p) < 1e-10
            assert abs(got.gleichlauf - expected_glk) < 1e-10

    def test_position_bounds_filter_cleanly(self):
        rng = np.random.default_rng(9)
        ref = rng.normal(0.0, 1.0, size=200)
        sample = ref[120:170].copy()

        bounded = find_best_match(
            sample,
            ref,
            reference_start_year=1700,
            min_overlap=30,
            n_best=1,
            min_position=1700,
            max_position=1760,
        )
        assert len(bounded) == 1
        assert 1700 <= bounded[0].position <= 1760

    def test_constant_input_emits_no_warnings(self):
        import warnings

        ref = np.ones(100)
        sample = np.linspace(0.0, 1.0, 40)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = sliding_correlation(sample, ref, reference_start_year=1800, min_overlap=30)
        assert results == []
        assert len(caught) == 0



def _series_pair(seed, n_reference=90, n_sample=35, offset=30, decimals=1):
    """Noisy sample cut from a reference; rounding creates exact ties (zero changes)."""
    rng = np.random.default_rng(seed)
    reference = np.round(rng.normal(0.0, 1.0, size=n_reference), decimals)
    sample = np.round(
        reference[offset:offset + n_sample] + rng.normal(0.0, 0.6, size=n_sample), decimals
    )
    return sample, reference


class TestBruteForceEquivalence:
    """r, t, p, overlap, position and GLK against scipy.pearsonr + a naive GLK loop.

    min_overlap is far below the sample length, so every case includes partial
    overlaps hanging off both ends of the reference.
    """

    def test_nan_free(self):
        sample, reference = _series_pair(101)
        results, _ = _assert_matches_brute_force(sample, reference, 1800, min_overlap=12)
        overlaps = [r.overlap for r in results]
        assert results[0].position == 1800 - 35 + 12  # hangs off the start
        assert results[-1].position == 1800 + 90 - 12  # hangs off the end
        assert min(overlaps) == 12 and max(overlaps) == 35
        # Ties are present, and GLK is not a constant.
        assert np.any(np.diff(sample) == 0) and np.any(np.diff(reference) == 0)
        assert len({round(r.gleichlauf, 6) for r in results}) > 10

    def test_nan_gap_in_sample(self):
        sample, reference = _series_pair(102)
        sample[[0, 9, 10, 11, 20]] = np.nan
        _assert_matches_brute_force(sample, reference, 1800, min_overlap=12)

    def test_nan_gap_in_reference(self):
        sample, reference = _series_pair(103)
        reference[[3, 40, 41, 42, 43, 70, 89]] = np.nan
        _assert_matches_brute_force(sample, reference, 1800, min_overlap=12)

    def test_nan_gaps_in_both(self):
        sample, reference = _series_pair(104)
        sample[[5, 6, 18, 34]] = np.nan
        reference[[0, 1, 33, 50, 51, 52, 77]] = np.nan
        results, expected = _assert_matches_brute_force(sample, reference, 1800, min_overlap=12)
        # Gaps really do change the answer: overlap is below the window length
        # everywhere, and GLK differs from a 50.0 placeholder.
        assert all(r.overlap < 35 for r in results)
        assert any(abs(r.gleichlauf - 50.0) > 5.0 for r in results)

    def test_position_bounds_with_gaps(self):
        sample, reference = _series_pair(105)
        sample[12] = np.nan
        reference[[20, 21, 60]] = np.nan
        results, _ = _assert_matches_brute_force(
            sample, reference, 1800, min_overlap=12, min_position=1790, max_position=1850
        )
        assert results[0].position == 1790 and results[-1].position == 1850

    def test_longer_series_take_the_same_answers(self):
        # Large enough that scipy may switch correlation method (direct vs FFT).
        sample, reference = _series_pair(106, n_reference=700, n_sample=160, offset=300, decimals=2)
        _assert_matches_brute_force(sample, reference, 1200, min_overlap=40)
        sample[[4, 77, 78, 150]] = np.nan
        reference[[10, 350, 351, 352, 699]] = np.nan
        _assert_matches_brute_force(sample, reference, 1200, min_overlap=40)


class TestGleichlaufWithGaps:
    """A year-to-year change counts only when both years are valid in both series."""

    def test_scalar_does_not_count_the_change_across_a_gap(self):
        s1 = np.array([1.0, 2.0, np.nan, 3.0, 2.0])
        s2 = np.array([1.0, 2.0, np.nan, 1.0, 0.0])
        # Observed consecutive-year changes: (1->2, 1->2) and (3->2, 1->0). Both agree.
        assert calculate_gleichlauf(s1, s2) == 100.0

    def test_sliding_does_not_count_the_change_across_a_gap(self):
        s1 = np.array([1.0, 2.0, np.nan, 3.0, 2.0])
        s2 = np.array([1.0, 2.0, np.nan, 1.0, 0.0])
        by_pos = {r.position: r for r in sliding_correlation(s1, s2, 1800, min_overlap=4)}
        assert by_pos[1800].overlap == 4
        assert by_pos[1800].gleichlauf == 100.0

    def test_gap_in_one_series_only(self):
        s1 = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 5.0])
        s2 = np.array([1.0, 3.0, np.nan, 2.0, 1.0, 0.0])
        # Counted changes: years 0-1 (agree), 3-4 (agree), 4-5 (disagree).
        assert calculate_gleichlauf(s1, s2) == pytest.approx(100.0 * 2 / 3)
        by_pos = {r.position: r for r in sliding_correlation(s1, s2, 1800, min_overlap=5)}
        assert by_pos[1800].gleichlauf == pytest.approx(100.0 * 2 / 3)

    def test_scalar_and_sliding_agree_at_every_lag(self):
        sample, reference = _series_pair(107)
        sample[[7, 8, 22]] = np.nan
        reference[[15, 44, 45, 46]] = np.nan
        results = sliding_correlation(sample, reference, 1800, min_overlap=12)
        assert len(results) > 50
        for got in results:
            lag = got.position - 1800
            s0, s1 = max(0, -lag), min(len(sample), len(reference) - lag)
            r0, r1 = max(0, lag), min(len(reference), lag + len(sample))
            scalar = calculate_gleichlauf(sample[s0:s1], reference[r0:r1])
            assert got.gleichlauf == pytest.approx(scalar, abs=1e-12)

    def test_zero_change_counts_half_with_gaps(self):
        s1 = np.array([1.0, 1.0, 2.0, np.nan, 5.0, 4.0])
        s2 = np.array([3.0, 4.0, 5.0, np.nan, 2.0, 2.0])
        # Counted: 0-1 (tie in s1: 0.5), 1-2 (agree: 1), 4-5 (tie in s2: 0.5) -> 2/3.
        assert calculate_gleichlauf(s1, s2) == pytest.approx(100.0 * 2 / 3)
        by_pos = {r.position: r for r in sliding_correlation(s1, s2, 1800, min_overlap=5)}
        assert by_pos[1800].gleichlauf == pytest.approx(100.0 * 2 / 3)

    def test_no_countable_change_reports_zero_in_both_paths(self):
        s1 = np.array([1.0, np.nan, 3.0, np.nan, 2.0, np.nan, 5.0, np.nan, 4.0])
        s2 = np.array([2.0, 1.0, 4.0, 3.0, 1.0, 0.0, 6.0, 2.0, 3.0])
        assert calculate_gleichlauf(s1, s2) == 0.0
        by_pos = {r.position: r for r in sliding_correlation(s1, s2, 1800, min_overlap=5)}
        assert by_pos[1800].overlap == 5
        assert by_pos[1800].gleichlauf == 0.0

    def test_nan_free_scalar_unchanged(self):
        s1 = np.array([1.0, 2.0, 2.0, 1.0, 3.0, 2.0])
        s2 = np.array([0.0, 1.0, 3.0, 2.0, 2.0, 3.0])
        # agree, tie(0.5), agree, tie(0.5), disagree -> 3 / 5
        assert calculate_gleichlauf(s1, s2) == pytest.approx(60.0)

    def test_masked_path_does_not_fall_back_to_a_per_lag_python_loop(self, monkeypatch):
        """Old behaviour: any NaN sent GLK through calculate_gleichlauf once per lag."""
        calls = []
        original = correlator_module.calculate_gleichlauf

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(correlator_module, "calculate_gleichlauf", counting)
        rng = np.random.default_rng(5)
        reference = rng.normal(size=600)
        sample = reference[200:350] + rng.normal(scale=0.5, size=150)
        sample[70] = np.nan
        results = sliding_correlation(sample, reference, 1400, min_overlap=30)
        assert len(results) > 500
        assert calls == []


class TestVarianceTolerance:
    """Constant windows are excluded; rescaling the data changes nothing."""

    @pytest.mark.parametrize("with_gap", [False, True])
    def test_window_constant_in_reference_is_excluded(self, with_gap):
        rng = np.random.default_rng(5)
        reference = rng.normal(size=400)
        reference[150:260] = 0.37
        sample = rng.normal(size=60)
        min_overlap = 60
        if with_gap:
            sample[7] = np.nan
            min_overlap = 59
        results, expected = _assert_matches_brute_force(
            sample, reference, 1000, min_overlap=min_overlap
        )
        positions = {r.position for r in results}
        # Every window lying wholly inside the flat stretch has no Pearson r.
        flat = set(range(1150, 1201))
        assert flat.isdisjoint(positions)
        assert len(positions) == 341 - len(flat)

    def test_window_constant_in_sample_is_excluded(self):
        rng = np.random.default_rng(22)
        reference = rng.normal(size=120)
        sample = rng.normal(size=50)
        sample[:20] = -1.3
        # Lags at the far right overlap only the flat head of the sample.
        _, expected = _assert_matches_brute_force(sample, reference, 1000, min_overlap=10)
        assert 1000 + 120 - 10 not in expected
        assert 1000 + 120 - 21 in expected

    @pytest.mark.parametrize("scale", [1e-10, 1e-6, 1e6, 1e10])
    @pytest.mark.parametrize("with_gap", [False, True])
    def test_rescaling_both_series_keeps_positions_and_r(self, scale, with_gap):
        rng = np.random.default_rng(23)
        reference = rng.normal(size=600)
        sample = reference[200:350] + rng.normal(scale=0.5, size=150)
        if with_gap:
            sample[70] = np.nan
            reference[[10, 400]] = np.nan
        base = sliding_correlation(sample, reference, 1400, min_overlap=30)
        scaled = sliding_correlation(sample * scale, reference * scale, 1400, min_overlap=30)
        # 691 lags; with the gaps, the first lag has only 29 jointly valid years.
        assert len(base) == (690 if with_gap else 691)
        assert [r.position for r in scaled] == [r.position for r in base]
        for a, b in zip(base, scaled):
            assert b.overlap == a.overlap
            assert abs(b.correlation - a.correlation) < 1e-10
            assert b.gleichlauf == pytest.approx(a.gleichlauf, abs=1e-9)

    def test_tiny_scale_reference_still_matches(self):
        x = np.arange(1.0, 31.0)
        results = sliding_correlation(x, x * 1e-12, 1800, min_overlap=10)
        assert len(results) == 41
        by_pos = {r.position: r for r in results}
        assert by_pos[1800].overlap == 30
        assert by_pos[1800].correlation == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("scale", [1e-100, 1e100])
    def test_absurd_scale_gives_no_result_instead_of_garbage(self, scale):
        """Sums of squares under/overflow here; r cannot be computed, so skip quietly."""
        import warnings

        rng = np.random.default_rng(26)
        reference = rng.normal(size=120)
        sample = reference[40:80] + rng.normal(scale=0.5, size=40)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = sliding_correlation(sample * scale, reference * scale, 1800, min_overlap=30)
            segments = segment_correlation(
                reference[:100] * scale, reference[:100] * scale, segment_length=50, lag=25
            )
        assert results == []
        assert segments == []

    def test_segment_correlation_is_scale_invariant(self):
        rng = np.random.default_rng(24)
        reference = rng.normal(size=200)
        sample = reference + rng.normal(scale=0.7, size=200)
        base = segment_correlation(sample, reference, segment_length=50, lag=25)
        assert len(base) == 7
        for scale in (1e-10, 1e-12, 1e8):
            scaled = segment_correlation(sample * scale, reference * scale, segment_length=50, lag=25)
            assert [s[0] for s in scaled] == [s[0] for s in base]
            for (_, r_a, t_a), (_, r_b, t_b) in zip(base, scaled):
                assert abs(r_a - r_b) < 1e-12
                assert t_b == pytest.approx(t_a, rel=1e-9)

    @pytest.mark.parametrize("flat_value", [0.37, 0.1, 3.3, 0.0, 1e-9])
    def test_segment_correlation_skips_constant_segments(self, flat_value):
        rng = np.random.default_rng(25)
        reference = rng.normal(size=100)
        sample = rng.normal(size=100)
        sample[0:50] = flat_value
        starts = [s[0] for s in segment_correlation(sample, reference, segment_length=50, lag=25)]
        assert starts == [25, 50]
        starts = [s[0] for s in segment_correlation(reference, sample, segment_length=50, lag=25)]
        assert starts == [25, 50]


class TestDegenerateOverlap:
    """Fewer than 3 overlapping points: Pearson has no degrees of freedom."""

    def test_two_point_overlap_reports_p_one_and_t_zero(self):
        results = sliding_correlation([1.0, 2.0], [3.0, 4.0], 1800, min_overlap=2)
        assert len(results) == 1
        only = results[0]
        assert only.position == 1800 and only.overlap == 2
        assert only.correlation == 1.0
        assert only.p_value == 1.0
        assert only.t_value == 0.0
        # scipy agrees on the p-value.
        assert stats.pearsonr([1.0, 2.0], [3.0, 4.0])[1] == 1.0

    def test_two_point_partial_overlaps_next_to_a_three_point_overlap(self):
        results = sliding_correlation([1.0, 3.0, 2.0], [2.0, 5.0, 1.0], 1800, min_overlap=2)
        by_pos = {r.position: r for r in results}
        assert sorted(by_pos) == [1799, 1800, 1801]
        for pos in (1799, 1801):
            assert by_pos[pos].overlap == 2
            assert by_pos[pos].p_value == 1.0
            assert by_pos[pos].t_value == 0.0
        exp_r, exp_p = stats.pearsonr([1.0, 3.0, 2.0], [2.0, 5.0, 1.0])
        assert by_pos[1800].overlap == 3
        assert by_pos[1800].correlation == pytest.approx(exp_r, abs=1e-12)
        assert by_pos[1800].p_value == pytest.approx(exp_p, rel=1e-9)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_min_overlap_raises_value_error(self, bad):
        with pytest.raises(ValueError):
            sliding_correlation([1.0, 2.0, 4.0], [3.0, 4.0, 1.0, 2.0], 1800, min_overlap=bad)
        with pytest.raises(ValueError):
            find_best_match([1.0, 2.0, 4.0], [3.0, 4.0, 1.0, 2.0], 1800, min_overlap=bad)


class TestTieOrder:
    """Exact t-value ties come back earliest position first."""

    PATTERN = [0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 2.0]

    def test_exact_ties_are_earliest_first(self):
        reference = np.tile(self.PATTERN, 6)
        # Small integers with a mean exactly representable in binary: all sums are exact,
        # so the five aligned positions tie exactly.
        sample = np.array(self.PATTERN + self.PATTERN, dtype=float)
        sample[7] = 3.0
        results = find_best_match(sample, reference, 1000, min_overlap=16, n_best=5)
        assert len({r.t_value for r in results}) == 1
        assert np.isfinite(results[0].t_value)
        assert [r.position for r in results] == [1000, 1008, 1016, 1024, 1032]

    def test_infinite_t_ties_are_earliest_first(self):
        reference = np.tile(self.PATTERN, 6)
        sample = np.array(self.PATTERN + self.PATTERN, dtype=float)
        results = find_best_match(sample, reference, 1000, min_overlap=16, n_best=6)
        assert [r.t_value for r in results[:5]] == [float("inf")] * 5
        assert [r.position for r in results[:5]] == [1000, 1008, 1016, 1024, 1032]
        assert np.isfinite(results[5].t_value)

    def test_ties_with_peak_separation_keep_earliest(self):
        reference = np.tile(self.PATTERN, 6)
        sample = np.array(self.PATTERN + self.PATTERN, dtype=float)
        results = find_best_match(
            sample, reference, 1000, min_overlap=16, n_best=3, min_peak_separation=12
        )
        assert [r.position for r in results] == [1000, 1016, 1032]

    def test_descending_order_without_ties_unchanged(self):
        rng = np.random.default_rng(31)
        reference = rng.normal(size=300)
        sample = reference[100:160] + rng.normal(scale=0.4, size=60)
        results = find_best_match(sample, reference, 1500, min_overlap=30, n_best=25)
        t_values = [r.t_value for r in results]
        assert t_values == sorted(t_values, reverse=True)
        assert results[0].position == 1600
        everything = sliding_correlation(sample, reference, 1500, min_overlap=30)
        assert t_values == sorted((r.t_value for r in everything), reverse=True)[:25]
