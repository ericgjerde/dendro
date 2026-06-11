"""Tests for the FastAPI dating service."""

import io
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from dendro.api import app as api_app  # noqa: E402
from dendro.crossdating.matcher import CrossdateMatcher  # noqa: E402
from dendro.reference.chronology_index import ChronologyIndex  # noqa: E402

FIX = Path(__file__).parent / "fixtures"
SAMPLE = FIX / "samples" / "known_1789.csv"


@pytest.fixture
def client(monkeypatch):
    # Point the service at the committed synthetic reference fixtures.
    matcher = CrossdateMatcher(index=ChronologyIndex(FIX / "reference"))
    api_app.state.matcher = matcher
    api_app.state.reference_dir = FIX / "reference"
    with TestClient(api_app.app) as c:
        # TestClient lifespan reloads the matcher from DENDRO_REFERENCE_DIR;
        # re-pin it to the fixtures afterwards.
        api_app.state.matcher = matcher
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_references(client):
    r = client.get("/references")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    # Synthetic fixture filenames don't encode species/state (as with many real
    # ITRDB files), so just assert the coverage envelope is reported.
    assert body["coverage"]["start"] is not None
    assert body["coverage"]["end"] is not None


def test_date_json_recovers_year(client):
    widths = pd.read_csv(SAMPLE)["width_mm"].tolist()
    r = client.post(
        "/date",
        json={
            "widths": widths,
            "has_bark_edge": True,
            "era_start": 1700,
            "era_end": 1850,
            "min_overlap": 40,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["consensus_year"] == 1789
    assert body["felling_estimate"]["felling_type"] == "exact"


def test_date_csv_upload(client):
    raw = SAMPLE.read_bytes()
    r = client.post(
        "/date/csv?era_start=1700&era_end=1850",
        files={"file": ("known_1789.csv", io.BytesIO(raw), "text/csv")},
    )
    assert r.status_code == 200
    assert r.json()["consensus_year"] == 1789


def test_date_rejects_negative_widths(client):
    r = client.post("/date", json={"widths": [1.0, -2.0, 3.0] * 20})
    assert r.status_code == 422


def test_parse_upload(client):
    raw = (FIX / "reference" / "synth" / "synth01.rwl").read_bytes()
    r = client.post(
        "/parse",
        files={"file": ("synth01.rwl", io.BytesIO(raw), "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["series_count"] == 8
