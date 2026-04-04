from types import SimpleNamespace

from dendro.reference.downloader import list_available_files


MEASUREMENTS_HTML = """
<html>
  <body>
    <a href="nh001.rwl">nh001.rwl</a>
    <a href="nh001-rwl-noaa.txt">nh001-rwl-noaa.txt</a>
    <a href="nh001-noaa.rwl">nh001-noaa.rwl</a>
  </body>
</html>
"""

CHRONOLOGIES_HTML = """
<html>
  <body>
    <a href="nh001.crn">nh001.crn</a>
    <a href="nh001-crn-noaa.txt">nh001-crn-noaa.txt</a>
    <a href="nh001-noaa.crn">nh001-noaa.crn</a>
  </body>
</html>
"""

SIDECAR_TEXT = """
# Fritts - Nancy Brook - PCRU - ITRDB NH001
# NOAA_Landing_Page: https://example.invalid/study/3274
"""


def test_list_available_files_discovers_rwl_and_crn(monkeypatch):
    def fake_get(url, timeout=60):
        if url.endswith("/measurements/northamerica/usa/"):
            return SimpleNamespace(text=MEASUREMENTS_HTML, status_code=200, raise_for_status=lambda: None)
        if url.endswith("/chronologies/northamerica/usa/"):
            return SimpleNamespace(text=CHRONOLOGIES_HTML, status_code=200, raise_for_status=lambda: None)
        if url.endswith("nh001-rwl-noaa.txt") or url.endswith("nh001-crn-noaa.txt"):
            return SimpleNamespace(text=SIDECAR_TEXT, status_code=200, raise_for_status=lambda: None)
        raise AssertionError(f"Unexpected URL {url}")

    monkeypatch.setattr("dendro.reference.downloader.requests.get", fake_get)

    rwl_files = list_available_files(states=["nh"], species=["PCRU"], file_type="rwl")
    crn_files = list_available_files(states=["nh"], species=["PCRU"], file_type="crn")

    assert len(rwl_files) == 1
    assert rwl_files[0].filename == "nh001.rwl"
    assert rwl_files[0].sidecar_filename == "nh001-rwl-noaa.txt"
    assert rwl_files[0].species == "PCRU"

    assert len(crn_files) == 1
    assert crn_files[0].filename == "nh001.crn"
    assert crn_files[0].sidecar_filename == "nh001-crn-noaa.txt"
    assert crn_files[0].species == "PCRU"
