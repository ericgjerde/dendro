"""
Diagnostic visualization plots for cross-dating results.

These plots help users visually verify dating matches and build confidence
in results through multiple lines of evidence.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ..crossdating.matcher import CrossdateReport, MatchResult
from ..crossdating.correlator import sliding_correlation, CorrelationResult


# Known marker years for New England region
MARKER_YEARS = {
    1816: "Year Without Summer (Tambora volcanic winter)",
    1709: "Great Frost of 1709",
    1780: "Dark Day of 1780 / severe drought",
    1741: "Great Frost / severe winter",
    1747: "Severe drought",
    1762: "Severe drought",
    1772: "Severe winter",
}


def plot_sample_reference_overlay(
    sample: np.ndarray,
    reference: np.ndarray,
    sample_start_year: int,
    reference_start_year: int,
    reference_name: str = "Reference",
    correlation: float = 0.0,
    t_value: float = 0.0,
    ax: Optional[plt.Axes] = None,
    marker_years: Optional[dict[int, str]] = None,
) -> plt.Axes:
    """
    Plot sample and reference series overlaid at the matched position.

    This is the primary visual verification: both series should show
    similar patterns if the dating is correct.

    Args:
        sample: Sample ring width series (standardized).
        reference: Reference chronology (standardized).
        sample_start_year: Proposed calendar year of first sample ring.
        reference_start_year: Calendar year of first reference value.
        reference_name: Name of reference chronology for label.
        correlation: Correlation value to display.
        t_value: T-value to display.
        ax: Matplotlib axes to plot on (creates new if None).
        marker_years: Dict of year -> description for known events.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))

    # Create year arrays
    sample_years = np.arange(sample_start_year, sample_start_year + len(sample))
    ref_years = np.arange(reference_start_year, reference_start_year + len(reference))

    # Plot reference
    ax.plot(ref_years, reference, 'b-', alpha=0.6, linewidth=1, label=reference_name)

    # Plot sample
    ax.plot(sample_years, sample, 'r-', linewidth=1.5, label="Sample")

    # Highlight overlap region
    overlap_start = max(sample_start_year, reference_start_year)
    overlap_end = min(sample_start_year + len(sample), reference_start_year + len(reference))
    ax.axvspan(overlap_start, overlap_end, alpha=0.1, color='green', label='Overlap')

    # Mark known climate events if within range
    if marker_years is None:
        marker_years = MARKER_YEARS

    for year, description in marker_years.items():
        if overlap_start <= year <= overlap_end:
            ax.axvline(year, color='orange', linestyle='--', alpha=0.7, linewidth=1)
            ax.annotate(
                str(year),
                xy=(year, ax.get_ylim()[1]),
                xytext=(0, 5),
                textcoords='offset points',
                fontsize=8,
                ha='center',
                rotation=90,
            )

    ax.set_xlabel("Year")
    ax.set_ylabel("Standardized Ring Width Index")
    ax.set_title(
        f"Sample vs. {reference_name}\n"
        f"Proposed date: {sample_start_year}-{sample_start_year + len(sample) - 1} | "
        f"r={correlation:.3f}, t={t_value:.1f}"
    )
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    return ax


def plot_correlation_profile(
    correlations: list[CorrelationResult],
    best_position: int,
    reference_name: str = "Reference",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot sliding correlation values across all tested positions.

    This shows how distinctive the best match is compared to other positions.
    A clear peak at the best position indicates reliable dating.

    Args:
        correlations: List of CorrelationResult from sliding_correlation.
        best_position: The selected best match position (year).
        reference_name: Name of reference for label.
        ax: Matplotlib axes to plot on.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    if not correlations:
        ax.text(0.5, 0.5, "No correlation data available",
                transform=ax.transAxes, ha='center', va='center')
        return ax

    years = [c.position for c in correlations]
    t_values = [c.t_value for c in correlations]

    # Plot correlation profile
    ax.fill_between(years, 0, t_values, alpha=0.3, color='blue')
    ax.plot(years, t_values, 'b-', linewidth=1)

    # Mark best position
    ax.axvline(best_position, color='red', linewidth=2, linestyle='-', label=f'Best match: {best_position}')

    # Add threshold lines
    ax.axhline(6.0, color='green', linestyle='--', alpha=0.7, label='HIGH threshold (t=6)')
    ax.axhline(4.0, color='orange', linestyle='--', alpha=0.7, label='MEDIUM threshold (t=4)')
    ax.axhline(3.5, color='red', linestyle=':', alpha=0.5, label='Significance threshold (t=3.5)')

    ax.set_xlabel("Sample Start Year")
    ax.set_ylabel("T-value")
    ax.set_title(f"Correlation Profile vs. {reference_name}")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    return ax


def plot_segment_analysis(
    segment_correlations: list[tuple[int, float, float]],
    sample_start_year: int,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot per-segment correlations (COFECHA-style analysis).

    Consistent correlation across all segments indicates reliable dating.
    Weak segments may indicate measurement errors or local anomalies.

    Args:
        segment_correlations: List of (start_index, r, t) tuples.
        sample_start_year: Calendar year of first sample ring.
        ax: Matplotlib axes to plot on.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    if not segment_correlations:
        ax.text(0.5, 0.5, "No segment data available\n(sample may be too short)",
                transform=ax.transAxes, ha='center', va='center')
        return ax

    # Convert to years
    years = [sample_start_year + seg[0] for seg in segment_correlations]
    correlations = [seg[1] for seg in segment_correlations]
    t_values = [seg[2] for seg in segment_correlations]

    # Color bars by quality
    colors = []
    for t in t_values:
        if t >= 6.0:
            colors.append('green')
        elif t >= 4.0:
            colors.append('orange')
        elif t >= 3.5:
            colors.append('yellow')
        else:
            colors.append('red')

    # Plot as bars
    width = (years[1] - years[0]) * 0.8 if len(years) > 1 else 20
    bars = ax.bar(years, correlations, width=width, color=colors, alpha=0.7, edgecolor='black')

    # Add threshold lines
    ax.axhline(0.55, color='green', linestyle='--', alpha=0.7, label='HIGH (r=0.55)')
    ax.axhline(0.45, color='orange', linestyle='--', alpha=0.7, label='MEDIUM (r=0.45)')
    ax.axhline(0.0, color='black', linestyle='-', alpha=0.3)

    ax.set_xlabel("Segment Start Year")
    ax.set_ylabel("Correlation (r)")
    ax.set_title("Segment-by-Segment Correlation Analysis")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Add custom legend for colors
    legend_elements = [
        mpatches.Patch(facecolor='green', alpha=0.7, label='Strong (t≥6)'),
        mpatches.Patch(facecolor='orange', alpha=0.7, label='Good (t≥4)'),
        mpatches.Patch(facecolor='yellow', alpha=0.7, label='Marginal (t≥3.5)'),
        mpatches.Patch(facecolor='red', alpha=0.7, label='Weak (t<3.5)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    return ax


def plot_crossdate_results(
    report: CrossdateReport,
    sample: np.ndarray,
    reference: np.ndarray,
    reference_start_year: int,
    all_correlations: Optional[list[CorrelationResult]] = None,
    figsize: tuple[float, float] = (14, 12),
) -> plt.Figure:
    """
    Create a comprehensive diagnostic figure for cross-dating results.

    Includes:
    1. Sample vs. reference overlay at best position
    2. Correlation profile showing match quality at each position
    3. Segment analysis showing per-segment correlations
    4. Match summary table

    Args:
        report: CrossdateReport from matcher.
        sample: Sample ring width series (standardized).
        reference: Reference chronology for best match (standardized).
        reference_start_year: Calendar year of first reference value.
        all_correlations: Optional list of correlations from sliding_correlation.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # Create grid for subplots
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1])

    if report.matches:
        best_match = report.matches[0]
        sample_start = best_match.proposed_start_year

        # 1. Sample/Reference overlay (spans full width)
        ax1 = fig.add_subplot(gs[0, :])
        plot_sample_reference_overlay(
            sample=sample,
            reference=reference,
            sample_start_year=sample_start,
            reference_start_year=reference_start_year,
            reference_name=best_match.reference_name,
            correlation=best_match.correlation,
            t_value=best_match.t_value,
            ax=ax1,
        )

        # 2. Correlation profile
        ax2 = fig.add_subplot(gs[1, :])
        if all_correlations:
            plot_correlation_profile(
                correlations=all_correlations,
                best_position=sample_start,
                reference_name=best_match.reference_name,
                ax=ax2,
            )
        else:
            ax2.text(0.5, 0.5, "Correlation profile not available",
                    transform=ax2.transAxes, ha='center', va='center')

        # 3. Segment analysis
        ax3 = fig.add_subplot(gs[2, 0])
        plot_segment_analysis(
            segment_correlations=best_match.segment_correlations,
            sample_start_year=sample_start,
            ax=ax3,
        )

        # 4. Summary table
        ax4 = fig.add_subplot(gs[2, 1])
        _plot_summary_table(report, ax4)
    else:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No matches found",
                transform=ax.transAxes, ha='center', va='center', fontsize=16)

    fig.suptitle(
        f"Cross-Dating Results: {report.sample_name}\n"
        f"Proposed Year: {report.consensus_year or 'N/A'} | "
        f"Confidence: {report.consensus_confidence}",
        fontsize=14, fontweight='bold'
    )

    return fig


def _plot_summary_table(report: CrossdateReport, ax: plt.Axes):
    """Plot a summary table of top matches."""
    ax.axis('off')

    if not report.matches:
        return

    # Build table data
    headers = ['Reference', 'Year', 'r', 't', 'Conf.']
    rows = []

    for m in report.matches[:5]:
        rows.append([
            m.reference_name[:20],  # Truncate long names
            str(m.felling_year),
            f"{m.correlation:.3f}",
            f"{m.t_value:.1f}",
            m.confidence,
        ])

    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['lightblue'] * 5,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    ax.set_title("Top 5 Matches", pad=20)


def save_diagnostic_plots(
    report: CrossdateReport,
    sample: np.ndarray,
    reference: np.ndarray,
    reference_start_year: int,
    output_path: str | Path,
    all_correlations: Optional[list[CorrelationResult]] = None,
    format: str = "png",
    dpi: int = 150,
) -> Path:
    """
    Generate and save diagnostic plots to a file.

    Args:
        report: CrossdateReport from matcher.
        sample: Sample ring width series.
        reference: Reference chronology for best match.
        reference_start_year: Calendar year of first reference value.
        output_path: Path to save the plot.
        all_correlations: Optional correlations for profile plot.
        format: Output format (png, pdf, svg).
        dpi: Resolution for raster formats.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)

    fig = plot_crossdate_results(
        report=report,
        sample=sample,
        reference=reference,
        reference_start_year=reference_start_year,
        all_correlations=all_correlations,
    )

    fig.savefig(output_path, format=format, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return output_path


def detect_marker_years(
    values: np.ndarray,
    threshold_sigma: float = 2.0,
) -> list[tuple[int, float, str]]:
    """
    Detect potential marker years (anomalously narrow or wide rings).

    Args:
        values: Ring width values (not standardized).
        threshold_sigma: Standard deviations from mean to flag.

    Returns:
        List of (index, z-score, description) for anomalous years.
    """
    values = np.asarray(values, dtype=np.float64)

    # Calculate z-scores
    mean = np.nanmean(values)
    std = np.nanstd(values)

    if std == 0:
        return []

    z_scores = (values - mean) / std

    anomalies = []
    for i, z in enumerate(z_scores):
        if z < -threshold_sigma:
            anomalies.append((i, z, "extremely narrow (drought/cold?)"))
        elif z > threshold_sigma:
            anomalies.append((i, z, "extremely wide (good growing season)"))

    return anomalies


def identify_known_markers(
    anomalies: list[tuple[int, float, str]],
    proposed_start_year: int,
) -> list[tuple[int, str, str]]:
    """
    Match detected anomalies against known marker years.

    Args:
        anomalies: List of (index, z_score, description) from detect_marker_years.
        proposed_start_year: Proposed calendar year of first ring.

    Returns:
        List of (year, sample_description, known_event) for matches.
    """
    matches = []

    for idx, z_score, desc in anomalies:
        year = proposed_start_year + idx

        # Check if this matches a known marker
        if year in MARKER_YEARS:
            matches.append((year, desc, MARKER_YEARS[year]))
        # Check adjacent years (dating could be off by 1)
        elif year - 1 in MARKER_YEARS:
            matches.append((year, desc, f"Near {year-1}: {MARKER_YEARS[year-1]}"))
        elif year + 1 in MARKER_YEARS:
            matches.append((year, desc, f"Near {year+1}: {MARKER_YEARS[year+1]}"))

    return matches
