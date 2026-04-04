"""Cross-dating package exports."""

from .detrend import build_chronology, detrend_series, standardize
from .correlator import calculate_tvalue, sliding_correlation

__all__ = [
    "build_chronology",
    "detrend_series",
    "standardize",
    "sliding_correlation",
    "calculate_tvalue",
    "CrossdateMatcher",
    "DatingCandidate",
    "DatingReport",
    "MatchResult",
]


def __getattr__(name):
    if name in {"CrossdateMatcher", "DatingCandidate", "DatingReport", "MatchResult"}:
        from .matcher import CrossdateMatcher, DatingCandidate, DatingReport, MatchResult

        mapping = {
            "CrossdateMatcher": CrossdateMatcher,
            "DatingCandidate": DatingCandidate,
            "DatingReport": DatingReport,
            "MatchResult": MatchResult,
        }
        return mapping[name]
    raise AttributeError(name)
