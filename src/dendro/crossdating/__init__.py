"""Cross-dating algorithms for matching samples to reference chronologies."""

from .correlator import calculate_tvalue, sliding_correlation
from .detrend import detrend_series, standardize
from .matcher import CrossdateMatcher, MatchResult

__all__ = [
    "detrend_series",
    "standardize",
    "sliding_correlation",
    "calculate_tvalue",
    "CrossdateMatcher",
    "MatchResult",
]
