"""
HTTP API for the dendrochronology dating engine.

This is the cloud-facing layer. It wraps the existing cross-dating engine as a
library (it does **not** shell out to the CLI) and exposes stateless endpoints
suitable for running behind a load balancer:

    GET  /health        liveness/readiness probe
    GET  /references    summary of the loaded reference chronologies
    POST /date          date a ring-width series supplied as JSON
    POST /date/csv      date a ring-width series uploaded as a CSV file
    POST /parse         summarize an uploaded Tucson .rwl file

The reference index is parsed once at startup and cached in memory (its master
chronologies are also cached lazily by the matcher), so requests do not re-read
the reference corpus. The reference directory is taken from the
``DENDRO_REFERENCE_DIR`` environment variable, defaulting to ``data/reference``.

Run locally with::

    dendro-api               # or: uvicorn dendro.api.app:app --reload
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..crossdating.matcher import CrossdateMatcher
from ..reference.chronology_index import ChronologyIndex

# Guardrails for untrusted input.
MAX_RINGS = 5000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


def _reference_dir() -> Path:
    return Path(os.environ.get("DENDRO_REFERENCE_DIR", "data/reference"))


class _State:
    """Process-wide, lazily-loaded reference state."""

    matcher: Optional[CrossdateMatcher] = None
    reference_dir: Optional[Path] = None


state = _State()


def _load_matcher() -> CrossdateMatcher:
    ref_dir = _reference_dir()
    state.reference_dir = ref_dir
    index = ChronologyIndex(ref_dir) if ref_dir.exists() else ChronologyIndex()
    state.matcher = CrossdateMatcher(index=index)
    return state.matcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_matcher()
    yield


app = FastAPI(
    title="Dendro Dating API",
    version="0.2.0",
    summary="Cross-date tree-ring width series against reference chronologies.",
    lifespan=lifespan,
)


class DateRequest(BaseModel):
    widths: list[float] = Field(..., min_length=1, description="Ring widths.")
    sample_name: str = "sample"
    has_bark_edge: bool = True
    orientation: str = Field("pith_to_bark", pattern="^(pith_to_bark|bark_to_pith)$")
    has_sapwood: bool = False
    sapwood_count: Optional[int] = None
    era_start: int = 1600
    era_end: int = 1900
    species: Optional[list[str]] = None
    states: Optional[list[str]] = None
    min_overlap: int = Field(30, ge=10, le=500)
    top: int = Field(10, ge=1, le=100)


def _require_matcher() -> CrossdateMatcher:
    if state.matcher is None:
        _load_matcher()
    assert state.matcher is not None
    if len(state.matcher.index) == 0:
        raise HTTPException(
            status_code=503,
            detail=(
                "No reference chronologies are loaded. Set DENDRO_REFERENCE_DIR "
                "to a directory of ITRDB .rwl/.crn files."
            ),
        )
    return state.matcher


def _validate_widths(widths: list[float]) -> np.ndarray:
    if len(widths) > MAX_RINGS:
        raise HTTPException(status_code=413, detail=f"Too many rings (> {MAX_RINGS}).")
    arr = np.asarray(widths, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise HTTPException(status_code=422, detail="Widths must all be finite numbers.")
    if np.any(arr < 0):
        raise HTTPException(status_code=422, detail="Widths must be non-negative.")
    return arr


def _run_dating(matcher: CrossdateMatcher, req: DateRequest, widths: np.ndarray) -> dict:
    report = matcher.date_sample(
        values=widths,
        sample_name=req.sample_name,
        has_bark_edge=req.has_bark_edge,
        species_filter=[s.upper() for s in req.species] if req.species else None,
        state_filter=[s.upper() for s in req.states] if req.states else None,
        era_start=req.era_start,
        era_end=req.era_end,
        min_overlap=req.min_overlap,
        orientation=req.orientation,
        has_sapwood=req.has_sapwood,
        sapwood_count=req.sapwood_count,
    )
    payload = report.to_dict()
    payload["matches"] = payload["matches"][: req.top]
    return payload


@app.get("/health")
def health() -> dict:
    n = len(state.matcher.index) if state.matcher else 0
    return {
        "status": "ok",
        "reference_dir": str(state.reference_dir) if state.reference_dir else None,
        "references_loaded": n,
    }


@app.get("/references")
def references() -> dict:
    matcher = state.matcher or _load_matcher()
    index = matcher.index
    starts = [e.start_year for e in index.entries if e.start_year > 0]
    ends = [e.end_year for e in index.entries if e.end_year > 0]
    return {
        "count": len(index),
        "species": index.get_species(),
        "states": index.get_states(),
        "coverage": {
            "start": min(starts) if starts else None,
            "end": max(ends) if ends else None,
        },
    }


@app.post("/date")
def date(req: DateRequest) -> dict:
    matcher = _require_matcher()
    widths = _validate_widths(req.widths)
    return _run_dating(matcher, req, widths)


@app.post("/date/csv")
async def date_csv(
    file: UploadFile = File(...),
    has_bark_edge: bool = True,
    orientation: str = "pith_to_bark",
    era_start: int = 1600,
    era_end: int = 1900,
) -> dict:
    matcher = _require_matcher()
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file too large.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 - surface parse errors to the client
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    df.columns = [str(c).lower().strip() for c in df.columns]
    if "width_mm" in df.columns:
        widths = df["width_mm"].to_numpy(dtype=float)
    elif "width" in df.columns:
        widths = df["width"].to_numpy(dtype=float)
    else:
        numeric = df.select_dtypes("number")
        if numeric.shape[1] == 0:
            raise HTTPException(status_code=422, detail="No numeric width column found.")
        widths = numeric.iloc[:, -1].to_numpy(dtype=float)

    req = DateRequest(
        widths=widths.tolist(),
        sample_name=Path(file.filename or "sample").stem,
        has_bark_edge=has_bark_edge,
        orientation=orientation,
        era_start=era_start,
        era_end=era_end,
    )
    return _run_dating(matcher, req, _validate_widths(req.widths))


@app.post("/parse")
async def parse(file: UploadFile = File(...)) -> dict:
    from ..reference.tucson_parser import parse_rwl_file

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file too large.")
    tmp = Path("/tmp") / f"upload_{os.getpid()}_{file.filename or 'series.rwl'}"
    tmp.write_bytes(raw)
    try:
        rwl = parse_rwl_file(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    return {
        "filename": file.filename,
        "series_count": len(rwl.series),
        "series": [
            {"id": sid, "start": s.start_year, "end": s.end_year, "length": s.length}
            for sid, s in list(rwl.series.items())[:50]
        ],
    }


def run() -> None:
    """Console-script entry point: launch uvicorn."""
    import uvicorn

    uvicorn.run(
        "dendro.api.app:app",
        host=os.environ.get("DENDRO_HOST", "0.0.0.0"),
        port=int(os.environ.get("DENDRO_PORT", "8000")),
    )


if __name__ == "__main__":
    run()
