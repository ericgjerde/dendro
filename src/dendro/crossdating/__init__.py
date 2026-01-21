"""Cross-dating algorithms for matching samples to reference chronologies."""

from .detrend import detrend_series, standardize
from .correlator import sliding_correlation, calculate_tvalue
from .matcher import CrossdateMatcher, MatchResult

__all__ = [
    "detrend_series",
    "standardize",
    "sliding_correlation",
    "calculate_tvalue",
    "CrossdateMatcher",
    "MatchResult",
]
