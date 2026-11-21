"""Tests for Tucson format parser."""

import numpy as np
import pytest
from io import StringIO
from pathlib import Path
import tempfile

from dendro.reference.tucson_parser import (
    parse_crn_file,
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


class TestTucsonParserRegressions:
    """Regression tests for precision stop markers, internal gaps, and CRN format."""

    def test_neg9999_rwl_preserves_mid_series_999_width(self, tmp_path: Path):
        """A 0.999 mm measurement (999) in a -9999 file must NOT terminate the decade."""
        content = (
            "CORE01  1950   934  1084   999  1762  1223  1489  1606  1098  2077  1069\n"
            "CORE01  1960  1582  1774 -9999\n"
        )
        rwl_path = tmp_path / "neg9999.rwl"
        rwl_path.write_text(content)

        rwl = parse_rwl_file(rwl_path)
        series = rwl.series["CORE01"]
        assert series.start_year == 1950
        assert series.end_year == 1961
        assert series.length == 12
        assert series.values[2] == 999.0
        assert series.values[10] == 1582.0
        assert series.values[11] == 1774.0

    def test_mixed_precision_file_keeps_999_stop_marker_per_series(self, tmp_path: Path):
        """A -9999 series elsewhere in the file must not turn another series' 999 stop marker into a ring."""
        content = (
            "AAA01   1960   158   177   999\n"
            "BBB01   1960  1582  1774 -9999\n"
        )
        rwl_path = tmp_path / "mixed.rwl"
        rwl_path.write_text(content)

        rwl = parse_rwl_file(rwl_path)
        coarse = rwl.series["AAA01"]
        assert coarse.end_year == 1961
        assert list(coarse.values) == [158.0, 177.0]
        fine = rwl.series["BBB01"]
        assert fine.end_year == 1961
        assert list(fine.values) == [1582.0, 1774.0]

    def test_rwl_internal_gap_inserts_nans_without_shifting_years(self, tmp_path: Path):
        """A break between segments must produce NaNs rather than shifting later years."""
        content = (
            "GAP01   1920  1000  1100  1200 -9999\n"
            "GAP01   1930  1300  1400 -9999\n"
        )
        rwl_path = tmp_path / "gap.rwl"
        rwl_path.write_text(content)

        rwl = parse_rwl_file(rwl_path)
        series = rwl.series["GAP01"]
        assert series.start_year == 1920
        assert series.end_year == 1931
        assert series.length == 12
        assert series.values[0] == 1000.0
        assert np.all(np.isnan(series.values[3:10]))
        assert series.values[10] == 1300.0
        assert series.values[11] == 1400.0

    def test_parse_standard_6char_crn_with_padding_and_trailer(self, tmp_path: Path):
        """Standard ITRDB 6-char CRN format with 9990 padding and decimal trailer."""
        content = (
            "PORSTD 1 Porters Creek                                       CADN\n"
            "PORSTD 2 Tennessee    Chestnut          1000M  3540-08223    __    1703 1712\n"
            "PORSTD 3 D.Stahle\n"
            "STNDRD17009990  09990  09990  01100  21050  2 980  31020  3 990  31010  41040  4\n"
            "STNDRD1710 970  41030  41060  39990  09990  09990  09990  09990  09990  09990  0\n"
            "PORSTD  10  .238  .110  .1061.000\n"
        )
        crn_path = tmp_path / "test.crn"
        crn_path.write_text(content)

        chron = parse_crn_file(crn_path)
        assert chron is not None
        # Site ID comes from the header lines; "STNDRD" on data lines is the chronology type.
        assert chron.site_id == "PORSTD"
        assert chron.start_year == 1703
        assert chron.end_year == 1712
        assert chron.length == 10
        assert chron.values[0] == 1100.0
        assert chron.values[-1] == 1060.0
        assert chron.sample_depth[0] == 2
        assert chron.sample_depth[-1] == 3



NOAA_DIR = Path(__file__).resolve().parents[1] / "data" / "walpole_open_sources" / "noaa"


def _crn_line(site_id: str, year: int, pairs, id_width: int = 6, pad_to: int = 10) -> str:
    """Build one fixed-width CRN data line: ID, 4-char year, (4-char value, 3-char depth) pairs."""
    cells = list(pairs) + [(9990, 0)] * max(0, pad_to - len(pairs))
    return f"{site_id:<{id_width}}{year:>4}" + "".join(f"{v:>4}{d:>3}" for v, d in cells)


def _crn_header(site_id: str) -> list[str]:
    return [
        f"{site_id:<6} 1 Synthetic Site                                      CADN",
        f"{site_id:<6} 2 Tennessee    Chestnut          1000M  3540-08223    __",
        f"{site_id:<6} 3 A.Tester",
    ]


class TestCRNFirstDecade:
    """Defect 1: a first line whose year field is the true (mid-decade) first year."""

    def test_mid_decade_year_field_with_decade_aligned_slots(self, tmp_path: Path):
        """Year field 1641, slot 0 is 9990 padding for 1640: values belong to 1641..1649."""
        lines = _crn_header("SYNSTD") + [
            _crn_line("STNDRD", 1641, [(9990, 0)] + [(1001 + i, 1) for i in range(9)]),
            _crn_line("STNDRD", 1650, [(2000 + i, 2) for i in range(10)]),
            _crn_line("STNDRD", 1660, [(3000, 3), (3001, 3), (3002, 3)]),
        ]
        crn_path = tmp_path / "middecade.crn"
        crn_path.write_text("\n".join(lines) + "\n")

        chron = parse_crn_file(crn_path)
        assert chron is not None
        assert chron.start_year == 1641
        assert chron.end_year == 1662
        assert chron.length == 22
        assert not np.isnan(chron.values).any()
        series = chron.to_dataframe()["value"]
        assert series[1641] == 1001.0
        assert series[1649] == 1009.0
        assert series[1650] == 2000.0
        assert series[1659] == 2009.0
        assert series[1662] == 3002.0
        assert chron.sample_depth[0] == 1
        assert chron.sample_depth[-1] == 3

    def test_mid_decade_year_field_with_values_from_slot_zero(self, tmp_path: Path):
        """Year field 1703 with a real value in slot 0: slot 0 is 1703 (RWL-style first line)."""
        lines = _crn_header("SYNSTD") + [
            _crn_line("STNDRD", 1703, [(1100 + i, 2) for i in range(7)], pad_to=7),
            _crn_line("STNDRD", 1710, [(1200, 3), (1201, 3)]),
        ]
        crn_path = tmp_path / "slotzero.crn"
        crn_path.write_text("\n".join(lines) + "\n")

        chron = parse_crn_file(crn_path)
        assert chron is not None
        assert chron.start_year == 1703
        assert chron.end_year == 1711
        assert chron.length == 9
        assert list(chron.values[:2]) == [1100.0, 1101.0]
        assert list(chron.values[-2:]) == [1200.0, 1201.0]

    def test_first_line_padded_with_blanks_instead_of_9990(self, tmp_path: Path):
        """Blank leading slots hold their positions: the first value belongs to 1703."""
        first = "STNDRD1700" + " " * 21 + "".join(f"{1100 + i:>4}{2:>3}" for i in range(7))
        lines = _crn_header("SYNSTD") + [first, _crn_line("STNDRD", 1710, [(1200, 3), (1201, 3)])]
        crn_path = tmp_path / "blankpad.crn"
        crn_path.write_text("\n".join(lines) + "\n")

        chron = parse_crn_file(crn_path)
        assert chron is not None
        assert chron.start_year == 1703
        assert chron.end_year == 1711
        assert chron.length == 9
        assert chron.values[0] == 1100.0
        assert chron.values[6] == 1106.0
        assert chron.values[7] == 1200.0

    def test_mid_decade_negative_year(self, tmp_path: Path):
        """Decades of negative years floor downward: -205 sits in the -210..-201 line."""
        lines = _crn_header("SYNSTD") + [
            _crn_line("STNDRD", -205, [(9990, 0)] * 5 + [(1000 + i, 1) for i in range(5)]),
            _crn_line("STNDRD", -200, [(2000 + i, 1) for i in range(10)]),
        ]
        crn_path = tmp_path / "negative.crn"
        crn_path.write_text("\n".join(lines) + "\n")

        chron = parse_crn_file(crn_path)
        assert chron is not None
        assert chron.start_year == -205
        assert chron.end_year == -191
        assert chron.length == 15
        assert chron.values[0] == 1000.0
        assert chron.values[4] == 1004.0
        assert chron.values[5] == 2000.0
        assert chron.values[-1] == 2009.0


def _real_crn_header_facts(path: Path) -> tuple[str, int, int, int]:
    """Read (header ID, first year, last year, year count) straight from a real CRN file."""
    lines = [ln.rstrip("\r\n") for ln in path.read_text().splitlines() if ln.strip()]
    header_two = next(ln for ln in lines if ln[6:9] == " 2 ")
    first_year, last_year = (int(tok) for tok in header_two.split()[-2:])
    header_id = header_two[:6].strip()
    trailer = next(ln for ln in reversed(lines) if ln.startswith(header_id) and "." in ln)
    year_count = int(trailer[6:10])
    return header_id, first_year, last_year, year_count


class TestRealCRNFiles:
    """The two tracked NOAA chronologies must match what their own headers state."""

    @pytest.mark.parametrize(
        "filename, first_year, last_year, year_count",
        [("tn013.crn", 1641, 1930, 290), ("tn012.crn", 1700, 1924, 225)],
    )
    def test_span_matches_header_and_trailer(self, filename, first_year, last_year, year_count):
        path = NOAA_DIR / filename
        _, hdr_first, hdr_last, hdr_count = _real_crn_header_facts(path)
        # Guard the test's own header reading against the known file contents.
        assert (hdr_first, hdr_last, hdr_count) == (first_year, last_year, year_count)

        chron = parse_crn_file(path)
        assert chron is not None
        assert chron.start_year == hdr_first
        assert chron.end_year == hdr_last
        assert chron.length == hdr_count
        assert int(np.count_nonzero(~np.isnan(chron.values))) == hdr_count
        assert (chron.sample_depth > 0).all()

    def test_tn013_first_decade_values(self):
        """Raw line: STNDRD16419990  01270  11010  1 723  1 ... 914  4 / STNDRD1650 850  4."""
        chron = parse_crn_file(NOAA_DIR / "tn013.crn")
        series = chron.to_dataframe()["value"]
        assert series[1641] == 1270.0
        assert series[1642] == 1010.0
        assert series[1649] == 914.0
        assert series[1650] == 850.0
        assert series[1930] == 893.0

    @pytest.mark.parametrize("filename, header_id", [("tn013.crn", "GBRSTD"), ("tn012.crn", "PORSTD")])
    def test_site_id_comes_from_header_lines(self, filename, header_id):
        path = NOAA_DIR / filename
        assert _real_crn_header_facts(path)[0] == header_id
        chron = parse_crn_file(path)
        assert chron.site_id == header_id


class TestCRNEightCharLayout:
    """Defect 2: 8-character IDs that end in digits must not be read as the 6-character layout."""

    def test_single_short_line_id_ending_in_digits(self, tmp_path: Path):
        line = _crn_line("SITE0012", 1900, [(1050, 1), (1060, 1), (1040, 1)], id_width=8, pad_to=3)
        assert line == "SITE001219001050  11060  11040  1"
        crn_path = tmp_path / "eight_short.crn"
        crn_path.write_text(line + "\n")

        chron = parse_crn_file(crn_path)
        assert chron is not None
        assert chron.site_id == "SITE0012"
        assert chron.start_year == 1900
        assert chron.end_year == 1902
        assert list(chron.values) == [1050.0, 1060.0, 1040.0]
        assert list(chron.sample_depth) == [1, 1, 1]

    def test_single_full_decade_line_all_digit_suffix(self, tmp_path: Path):
        line = _crn_line("ABC12345", 1900, [(1000 + i, 4) for i in range(10)], id_width=8)
        crn_path = tmp_path / "eight_full.crn"
        crn_path.write_text(line + "\n")

        chron = parse_crn_file(crn_path)
        assert chron is not None
        assert chron.site_id == "ABC12345"
        assert chron.start_year == 1900
        assert chron.end_year == 1909
        assert list(chron.values) == [1000.0 + i for i in range(10)]
        assert list(chron.sample_depth) == [4] * 10

    def test_multi_line_id_ending_in_digits(self, tmp_path: Path):
        lines = [
            _crn_line("SITE0012", 1887, [(9990, 0)] * 7 + [(900, 2), (910, 2), (920, 2)], id_width=8),
            _crn_line("SITE0012", 1890, [(1000 + i, 3) for i in range(10)], id_width=8),
            _crn_line("SITE0012", 1900, [(1100 + i, 12) for i in range(10)], id_width=8),
            _crn_line("SITE0012", 1910, [(1200, 5), (1210, 5)], id_width=8),
        ]
        crn_path = tmp_path / "eight_multi.crn"
        crn_path.write_text("\n".join(lines) + "\n")

        chron = parse_crn_file(crn_path)
        assert chron is not None
        assert chron.site_id == "SITE0012"
        assert chron.start_year == 1887
        assert chron.end_year == 1911
        assert chron.length == 25
        assert chron.values[0] == 900.0
        assert chron.values[-1] == 1210.0
        assert chron.sample_depth[13] == 12

    def test_six_char_layout_still_preferred_for_standard_file(self, tmp_path: Path):
        """A 6-char file whose year digits could be mistaken for ID digits stays 6-char."""
        lines = _crn_header("SYNSTD") + [
            _crn_line("STNDRD", 1890, [(1000 + i, 3) for i in range(10)]),
            _crn_line("STNDRD", 1900, [(1100 + i, 12) for i in range(10)]),
        ]
        crn_path = tmp_path / "six.crn"
        crn_path.write_text("\n".join(lines) + "\n")

        chron = parse_crn_file(crn_path)
        assert chron is not None
        assert chron.start_year == 1890
        assert chron.end_year == 1909
        assert chron.length == 20


class TestCRNEarlyYears:
    """Defect 3: years below 400 (and below 100) are data, not headers."""

    def test_years_80_to_419_with_three_digit_first_values(self, tmp_path: Path):
        lines = _crn_header("SYNSTD")
        for decade in range(80, 420, 10):
            lines.append(_crn_line("STNDRD", decade, [(900 + (decade + i) % 90, 4) for i in range(10)]))
        crn_path = tmp_path / "early.crn"
        crn_path.write_text("\n".join(lines) + "\n")

        chron = parse_crn_file(crn_path)
        assert chron is not None
        assert chron.start_year == 80
        assert chron.end_year == 419
        assert chron.length == 340
        assert not np.isnan(chron.values).any()
        series = chron.to_dataframe()["value"]
        for year in (80, 99, 100, 199, 200, 399, 400, 419):
            assert series[year] == 900.0 + year % 90
        assert chron.site_id == "SYNSTD"

    def test_years_0_to_29(self, tmp_path: Path):
        lines = _crn_header("SYNSTD") + [
            _crn_line("STNDRD", 0, [(950 + i, 2) for i in range(10)]),
            _crn_line("STNDRD", 10, [(960 + i, 2) for i in range(10)]),
            _crn_line("STNDRD", 20, [(970 + i, 2) for i in range(10)]),
        ]
        crn_path = tmp_path / "year_zero.crn"
        crn_path.write_text("\n".join(lines) + "\n")

        chron = parse_crn_file(crn_path)
        assert chron is not None
        assert chron.start_year == 0
        assert chron.end_year == 29
        assert chron.length == 30
        assert chron.values[0] == 950.0
        assert chron.values[-1] == 979.0


class TestRWLSlotPositions:
    """Defects 5 and 6 plus missing-value codes: RWL values keep their column's year."""

    def test_blank_mid_line_field_leaves_gap(self, tmp_path: Path):
        line = "CORE01  1950" + f"{120:>6}" + " " * 6 + f"{130:>6}" + f"{999:>6}"
        parsed = _parse_rwl_line(line)
        assert parsed is not None
        assert parsed[2] == [120, None, 130, 999]

        rwl_path = tmp_path / "blank.rwl"
        rwl_path.write_text(line + "\n")
        series = parse_rwl_file(rwl_path).series["CORE01"]
        assert series.start_year == 1950
        assert series.end_year == 1952
        assert series.values[0] == 120.0
        assert np.isnan(series.values[1])
        assert series.values[2] == 130.0

    def test_non_numeric_mid_line_field_leaves_gap(self, tmp_path: Path):
        line = "CORE02  1950" + f"{120:>6}" + f"{'n/a':>6}" + f"{130:>6}" + f"{140:>6}" + f"{-9999:>6}"
        rwl_path = tmp_path / "nonnumeric.rwl"
        rwl_path.write_text(line + "\n")
        series = parse_rwl_file(rwl_path).series["CORE02"]
        assert series.start_year == 1950
        assert series.end_year == 1953
        assert series.values[0] == 120.0
        assert np.isnan(series.values[1])
        assert list(series.values[2:]) == [130.0, 140.0]

    def test_missing_value_codes_become_nan(self, tmp_path: Path):
        """-999 and -8888 inside a series are missing-value codes, not widths."""
        content = (
            "MISS01  1950   120  -999   130   140   999\n"
            "MISS02  1950  1200 -8888  1300  1400 -9999\n"
        )
        rwl_path = tmp_path / "missing.rwl"
        rwl_path.write_text(content)
        rwl = parse_rwl_file(rwl_path)

        coarse = rwl.series["MISS01"]
        assert (coarse.start_year, coarse.end_year, coarse.length) == (1950, 1953, 4)
        assert np.isnan(coarse.values[1])
        assert [coarse.values[0], coarse.values[2], coarse.values[3]] == [120.0, 130.0, 140.0]

        fine = rwl.series["MISS02"]
        assert (fine.start_year, fine.end_year, fine.length) == (1950, 1953, 4)
        assert np.isnan(fine.values[1])
        assert [fine.values[0], fine.values[2], fine.values[3]] == [1200.0, 1300.0, 1400.0]

    def test_five_char_negative_year(self, tmp_path: Path):
        """A 5-character negative year borrows column 8 from the ID field."""
        content = (
            "ABC01  -1005   120   130   140   150   160\n"
            "ABC01  -1000   170   180   190   200   210   220   230   240   250   260\n"
            "ABC01   -990   270   280   999\n"
        )
        assert _parse_rwl_line(content.splitlines()[0])[:2] == ("ABC01", -1005)

        rwl_path = tmp_path / "negative.rwl"
        rwl_path.write_text(content)
        rwl = parse_rwl_file(rwl_path)
        assert list(rwl.series) == ["ABC01"]
        series = rwl.series["ABC01"]
        assert series.start_year == -1005
        assert series.end_year == -989
        assert series.length == 17
        assert series.values[0] == 120.0
        assert series.values[5] == 170.0
        assert series.values[-1] == 280.0
        assert not np.isnan(series.values).any()
