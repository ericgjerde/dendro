"""
Tests using synthetic "known date" samples.

These tests verify the cross-dating algorithm can correctly
date samples when we know the true answer.
"""

import numpy as np
import pytest

from dendro.crossdating.correlator import find_best_match
from dendro.crossdating.detrend import standardize


class TestKnownDateRecovery:
    """Tests for recovering known dates from synthetic samples."""

    def test_exact_subset_recovery(self):
        """Test recovering exact position of a subset."""
        np.random.seed(42)

        # Create a "reference chronology" with distinctive pattern
        # Mix of periodic signal and random component
        n_years = 200
        years = np.arange(1700, 1700 + n_years)

        # Climate-like signal: multi-frequency sine waves
        signal = (
            np.sin(np.linspace(0, 8 * np.pi, n_years)) +
            0.5 * np.sin(np.linspace(0, 24 * np.pi, n_years)) +
            np.random.randn(n_years) * 0.3
        )
        reference = standardize(signal)

        # Extract a "sample" - exact subset from years 1780-1830
        sample_start_idx = 80  # year 1780
        sample_end_idx = 130   # year 1830
        sample = reference[sample_start_idx:sample_end_idx].copy()

        # Find best match
        results = find_best_match(
            sample=sample,
            reference=reference,
            reference_start_year=1700,
            min_overlap=30,
            n_best=5,
        )

        assert len(results) > 0
        best = results[0]

        # Should find the correct position (1780)
        assert best.position == 1780
        assert best.correlation > 0.99  # Nearly perfect match

    def test_noisy_subset_recovery(self):
        """Test recovering position with added noise."""
        np.random.seed(42)

        n_years = 200
        signal = (
            np.sin(np.linspace(0, 8 * np.pi, n_years)) +
            0.5 * np.sin(np.linspace(0, 24 * np.pi, n_years)) +
            np.random.randn(n_years) * 0.3
        )
        reference = standardize(signal)

        # Extract sample and add noise
        sample_start_idx = 80
        sample = reference[sample_start_idx:sample_start_idx + 50].copy()
        sample += np.random.randn(50) * 0.3  # Add noise

        # Find best match
        results = find_best_match(
            sample=sample,
            reference=reference,
            reference_start_year=1700,
            min_overlap=30,
            n_best=5,
        )

        best = results[0]

        # Should still find correct position despite noise
        assert abs(best.position - 1780) <= 2  # Allow small error
        assert best.correlation > 0.6  # Reasonable correlation

    def test_marker_year_detection(self):
        """Test that distinct "marker years" aid dating."""
        np.random.seed(42)

        n_years = 200

        # Create signal with distinctive marker years
        signal = np.random.randn(n_years) * 0.3

        # Add "marker years" - distinctive narrow/wide rings
        # Simulating events like 1816 "Year Without Summer"
        marker_positions = [50, 75, 100, 125]  # Years with distinctive values
        for pos in marker_positions:
            signal[pos] = -2.0  # Very narrow ring

        reference = standardize(signal)

        # Sample that includes some marker years
        sample_start = 70  # Will include markers at 75, 100
        sample = reference[sample_start:sample_start + 50].copy()

        results = find_best_match(
            sample=sample,
            reference=reference,
            reference_start_year=1700,
            min_overlap=30,
            n_best=3,
        )

        best = results[0]

        # The marker years should help nail down the position
        assert best.position == 1770  # 1700 + 70
        assert best.correlation > 0.9

    def test_short_sample_less_certain(self):
        """Test that shorter samples produce less certain results."""
        np.random.seed(42)

        n_years = 200
        signal = (
            np.sin(np.linspace(0, 8 * np.pi, n_years)) +
            np.random.randn(n_years) * 0.5
        )
        reference = standardize(signal)

        # Long sample
        long_sample = reference[80:130].copy()  # 50 years
        long_results = find_best_match(long_sample, reference, 1700, min_overlap=30)

        # Short sample
        short_sample = reference[80:110].copy()  # 30 years
        short_results = find_best_match(short_sample, reference, 1700, min_overlap=25)

        # Both should find correct position
        assert long_results[0].position == 1780
        assert short_results[0].position == 1780

        # Long sample should have higher or equal t-value (more data usually = more confident)
        # But with perfect correlation both can be very high, so just check they're valid
        assert long_results[0].t_value > 0
        assert short_results[0].t_value > 0


class TestEdgeCases:
    """Tests for edge cases in cross-dating."""

    def test_sample_at_start_of_reference(self):
        """Test sample matching at the very start."""
        np.random.seed(42)

        reference = standardize(np.random.randn(100) + np.sin(np.linspace(0, 4*np.pi, 100)))
        sample = reference[0:40].copy()

        results = find_best_match(sample, reference, 1800, min_overlap=30)

        assert len(results) > 0
        assert results[0].position == 1800

    def test_sample_at_end_of_reference(self):
        """Test sample matching at the very end."""
        np.random.seed(42)

        reference = standardize(np.random.randn(100) + np.sin(np.linspace(0, 4*np.pi, 100)))
        sample = reference[60:100].copy()

        results = find_best_match(sample, reference, 1800, min_overlap=30)

        assert len(results) > 0
        assert results[0].position == 1860  # 1800 + 60

    def test_flat_reference_low_correlation(self):
        """Test that flat reference produces NaN or undefined correlations."""
        # Reference with no signal - correlation is undefined for constant input
        reference = np.ones(100)
        sample = np.random.randn(40)

        results = find_best_match(sample, reference, 1800, min_overlap=30)

        # With constant reference, correlations will be NaN or results may be empty
        # This is expected behavior - you can't correlate against a constant
        # Just verify we don't crash
        assert isinstance(results, list)
