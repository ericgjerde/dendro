"""Tests for Tucson format parser."""

import numpy as np
import pytest
from io import StringIO
from pathlib import Path
import tempfile

from dendro.reference.tucson_parser import (
    parse_rwl_file,
    _parse_rwl_line,
    RingWidthSeries,
    RWLFile,
)


class TestParseRWLLine:
    """Tests for individual RWL line parsing."""

    def test_standard_line(self):
        """Test parsing a standard Tucson format line."""
        # Tucson format: cols 1-8 = series ID, cols 9-12 = year, then 6-char value blocks
        line = "SAMPLE011850   120   135   142   158   167   145   132   128   119   125"
        result = _parse_rwl_line(line)

        assert result is not None
        series_id, year, values, is_end = result

        assert series_id == "SAMPLE01"
        assert year == 1850
        assert len(values) == 10
        assert values[0] == 120
        assert values[-1] == 125
        assert not is_end

    def test_line_with_stop_marker(self):
        """Test parsing line with 999 stop marker."""
        # Tucson format: cols 1-8 = series ID, cols 9-12 = year
        line = "SAMPLE011860   110   105   999"
        result = _parse_rwl_line(line)

        assert result is not None
        series_id, year, values, is_end = result

        assert series_id == "SAMPLE01"
        assert year == 1860
        assert 999 in values
        assert is_end

    def test_short_line_rejected(self):
        """Test that too-short lines are rejected."""
        result = _parse_rwl_line("SHORT")
        assert result is None

    def test_empty_line_rejected(self):
        """Test that empty lines are rejected."""
        result = _parse_rwl_line("")
        assert result is None

    def test_padded_series_id(self):
        """Test series ID with trailing spaces."""
        # Tucson format: cols 1-8 = series ID (space padded), cols 9-12 = year
        line = "ABC     1900   100   100   100"
        result = _parse_rwl_line(line)

        assert result is not None
        assert result[0] == "ABC"
        assert result[1] == 1900


class TestRingWidthSeries:
    """Tests for RingWidthSeries data class."""

    def test_properties(self):
        """Test computed properties."""
        series = RingWidthSeries(
            series_id="TEST",
            start_year=1800,
            end_year=1809,
            values=np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190]),
        )

        assert series.length == 10
        assert len(series.years) == 10
        assert series.years[0] == 1800
        assert series.years[-1] == 1809

    def test_to_series(self):
        """Test conversion to pandas Series."""
        series = RingWidthSeries(
            series_id="TEST",
            start_year=1800,
            end_year=1802,
            values=np.array([100, 110, 120]),
        )

        ps = series.to_series()
        assert ps.name == "TEST"
        assert ps.index[0] == 1800
        assert ps[1801] == 110


class TestRWLFile:
    """Tests for RWL file parsing."""

    def test_parse_simple_rwl(self):
        """Test parsing a simple RWL file."""
        # Tucson format: cols 1-8 = series ID, cols 9-12 = year, then 6-char value blocks
        content = """SAMPLE011850   120   135   142   158   167   145   132   128   119   125
SAMPLE011860   130   128   135   140   999
SAMPLE021855   200   210   220   230   240   999
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rwl', delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            rwl = parse_rwl_file(filepath)

            assert len(rwl.series) == 2
            assert "SAMPLE01" in rwl.series
            assert "SAMPLE02" in rwl.series

            s1 = rwl.series["SAMPLE01"]
            assert s1.start_year == 1850
            # Values before stop marker
            assert s1.length >= 10

            s2 = rwl.series["SAMPLE02"]
            assert s2.start_year == 1855
        finally:
            Path(filepath).unlink()

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        rwl = RWLFile()
        rwl.series["A"] = RingWidthSeries("A", 1800, 1802, np.array([100, 110, 120]))
        rwl.series["B"] = RingWidthSeries("B", 1801, 1803, np.array([200, 210, 220]))

        df = rwl.to_dataframe()

        assert "A" in df.columns
        assert "B" in df.columns
        assert df.index.name == "year"
        # A has values for 1800-1802, B has values for 1801-1803
        assert 1800 in df.index
        assert 1803 in df.index


class TestMasterChronology:
    """Tests for master chronology calculation."""

    def test_get_master_chronology(self):
        """Test calculating mean chronology."""
        rwl = RWLFile()
        rwl.series["A"] = RingWidthSeries("A", 1800, 1802, np.array([100., 100., 100.]))
        rwl.series["B"] = RingWidthSeries("B", 1800, 1802, np.array([200., 200., 200.]))

        master = rwl.get_master_chronology()

        # Mean should be 150 for overlapping years
        assert master[1800] == 150
        assert master[1801] == 150
        assert master[1802] == 150
