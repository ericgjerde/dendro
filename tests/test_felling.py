"""Tests for felling-year interpretation."""

from dendro.crossdating.felling import FellingType, estimate_felling


def test_bark_edge_exact():
    fe = estimate_felling(1789, has_bark_edge=True)
    assert fe.felling_type is FellingType.EXACT
    assert fe.earliest_felling == 1789
    assert fe.latest_felling == 1789


def test_no_bark_no_sapwood_is_after():
    fe = estimate_felling(1789, has_bark_edge=False)
    assert fe.felling_type is FellingType.AFTER
    assert fe.earliest_felling > 1789
    assert fe.latest_felling is None


def test_sapwood_present_gives_range():
    fe = estimate_felling(
        1789, has_bark_edge=False, has_sapwood=True, sapwood_count=5, species="QUAL"
    )
    assert fe.felling_type is FellingType.RANGE
    assert fe.earliest_felling >= 1789
    assert fe.latest_felling is not None
    assert fe.latest_felling >= fe.earliest_felling


def test_oak_uses_oak_sapwood_model():
    oak = estimate_felling(1800, has_bark_edge=False, has_sapwood=True, species="QURU")
    other = estimate_felling(1800, has_bark_edge=False, has_sapwood=True, species="PIST")
    # The two species groups should not produce identical ranges in general.
    assert (oak.earliest_felling, oak.latest_felling) != (
        other.earliest_felling, other.latest_felling
    ) or oak.note != other.note


def test_summary_and_dict_roundtrip():
    fe = estimate_felling(1789, has_bark_edge=True)
    assert "1789" in fe.summary()
    d = fe.to_dict()
    assert d["felling_type"] == "exact"
    assert d["earliest_felling"] == 1789
