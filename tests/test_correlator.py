"""Tests for cross-dating correlation algorithms."""

import numpy as np
import pytest

from dendro.crossdating.correlator import (
    sliding_correlation,
    calculate_tvalue,
    calculate_gleichlauf,
    find_best_match,
    dating_confidence,
    CorrelationResult,
)


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
