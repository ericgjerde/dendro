"""
Visualization module for dendrochronology cross-dating results.

Provides diagnostic plots for validating dating results:
- Sample vs. reference overlay at best-match position
- Sliding correlation profile showing match quality at each position
- Segment correlation analysis showing regional consistency
- Marker year highlighting for climate event verification
"""

from .plots import (
    plot_crossdate_results,
    plot_correlation_profile,
    plot_segment_analysis,
    plot_sample_reference_overlay,
    save_diagnostic_plots,
)

__all__ = [
    "plot_crossdate_results",
    "plot_correlation_profile",
    "plot_segment_analysis",
    "plot_sample_reference_overlay",
    "save_diagnostic_plots",
]
