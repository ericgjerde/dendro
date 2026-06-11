# Dendro: Technical Review & Plan for a Cloud Dendrochronology Platform

**Status:** Draft for discussion
**Scope:** Whole codebase (`src/dendro`, `tests`, `scripts`, packaging)
**Goal:** Turn the current local CLI prototype into *actually functional* dendrochronology
software that runs as a service in the cloud.

---

## Implementation status (this branch)

A first implementation pass has landed the science-correctness fixes, the
testing/CI foundation, and a deployable service layer. What is **done**:

- **§3.1 Orientation** — canonical oldest→newest enforced end-to-end via
  `normalize_orientation`; `--orientation` / API field; reversed input recovers
  the same date (tested).
- **§3.2 Felling logic** — new `crossdating/felling.py`; reports distinguish
  exact felling year (bark edge), estimated range (sapwood), and *terminus post
  quem* (tested).
- **§3.3 RWL master chronology** — now detrend-per-series → biweight mean of
  indices, cached. On the synthetic fixture this changed the result from a
  *confidently wrong* 1739 (r=0.89) to the correct **1789 (r=0.97, HIGH)**.
- **§3.4 Gleichläufigkeit** — formula corrected, significance test added, folded
  into the confidence model (tested).
- **§3.5 Missing rings** — first-cut `detect_missing_ring` locates a likely
  missing/false ring and the matcher surfaces it as a warning (tested). Full
  auto-realignment search remains future work.
- **§3.6 Spline** — replaced with the Cook & Peters cubic smoothing spline;
  frequency response verified to be 0.5 at the cutoff wavelength (tested).
- **§3.7/3.8 Parsers/cleanup** — Tucson writer rounding fixed; a writer↔parser
  round-trip test added; era filtering rationalized to the felling-year window.
- **§4.2 Imaging** — directional gradient stops double-counting ring boundaries.
- **Phase 0** — committed deterministic fixtures, CI (ruff + pytest + Docker
  smoke test), Dockerfile, lockfile, ruff config, SessionStart hook. 79 tests
  pass offline; the 4 real-ITRDB tests skip without downloaded data.
- **Phase 2 (spine)** — FastAPI service (`api/app.py`) wrapping the engine as a
  library, with input validation, a cached reference index, and a Docker image.

Still **outstanding** (tracked below): full missing-ring re-alignment search,
the browser measurement workflow (Phase 3), a persistent project/database model
and job queue (the rest of Phase 2), real-corpus reference data management, and
validation against dplR.

---

## 1. Executive summary

The repository is a well-organized **prototype** with the right module boundaries
(reference data, imaging, cross-dating, visualization, CLI) and good documentation. It
demonstrably recovers known dates on *clean, same-site* ITRDB series in its synthetic and
within-site validation tests.

However, it is **not yet scientifically reliable** and **cannot run in the cloud as-is**.
The blockers fall into three groups:

1. **Correctness bugs in the dating science** that will silently produce wrong or
   weakened results on real-world samples (sample orientation is unenforced, the
   bark-edge flag is ignored when reporting a felling year, the RWL master chronology is
   built by averaging *raw* widths instead of detrended indices, the Gleichläufigkeit
   formula is a no-op, and there is no missing-ring handling).
2. **A measurement front-end that requires a local GUI** (matplotlib `plt.show()`),
   plus ring detection that is too rudimentary for real hardwood scans.
3. **No service architecture** — no HTTP API, no persistence, no job processing, no
   container, no CI, and reference data that is scraped from NOAA HTML at runtime and
   re-parsed on every invocation.

This document inventories the specific issues and proposes a phased plan. The
recommended sequencing is: **fix the science first** (Phase 1), because a cloud service
that returns confidently wrong dates is worse than no service; then **build the service
spine** (Phase 2); then **replace the measurement GUI with a browser workflow** (Phase 3);
then **harden and operationalize** (Phase 4).

---

## 2. What works today (keep this)

- Clean package layout under `src/dendro` with sensible separation of concerns.
- Tucson `.rwl` reading is good enough for the common ITRDB layout and is the backbone
  of the working validation tests.
- The sliding-correlation core (`crossdating/correlator.py`) is straightforward and
  correct for the rigid-alignment case; t-value computation is right.
- Diagnostic plotting (`visualization/plots.py`) is genuinely useful and is the kind of
  "show your evidence" output dendrochronologists expect (overlay, correlation profile,
  segment bars, marker years).
- The validation harness concept (`scripts/validate_crossdating.py`, `tests/test_validation.py`)
  — "forget the date, recover it" — is exactly the right way to test this domain.

---

## 3. Correctness issues in the dating science (highest priority)

These are ordered by how badly they corrupt real-world results.

### 3.1 Sample orientation is ambiguous and unenforced — *critical*
`CrossdateMatcher.date_sample` documents `values` as *"from bark to pith, outer to inner"*
(newest→oldest), but reference chronologies are stored **oldest→newest**, and
`sliding_correlation` compares the two arrays element-by-element. If the sample is in the
opposite temporal order from the reference, every correlation is computed on a
time-reversed series and dating silently fails.

Meanwhile the measurement export path (`path_sampler.widths_to_csv` /
`widths_to_tucson`) *reverses* widths to oldest-first — the opposite of what the matcher
docstring asks for. So the tool's own output and its dating engine disagree about
orientation. Depending on how a CSV was produced, results are either correct or garbage,
with no error surfaced.

**Fix:** Define one canonical internal orientation (recommend **pith→bark / oldest→newest**,
matching ITRDB), convert at every boundary (CSV in, RWL in, measurement out), label the
orientation explicitly in file metadata, and add a guard that runs the correlation both
ways during development to detect regressions.

### 3.2 The `bark_edge` flag never affects the reported felling year — *critical*
`MatchResult.felling_year` simply returns `proposed_end_year`. The `has_bark_edge`
boolean is threaded through the entire stack but is **never used** in the computation.
For a sample *without* bark/waney edge, the last measured ring is **not** the felling
year — there are unknown missing sapwood (and possibly heartwood) rings. Reporting
`proposed_end_year` as "PROPOSED FELLING YEAR" in that case is scientifically wrong and is
the single most consequential output of the whole tool.

**Fix:**
- With bark edge → felling year = last ring year (state precision: exact, ± nothing).
- Without bark edge but with sapwood → "felling after" (*terminus post quem*) plus a
  sapwood-estimate range (e.g. Hollstein / Baillie & Pilcher sapwood models for oak;
  species-specific). Report a range, not a point.
- Heartwood-only → report only "felling after last heartwood ring + minimum sapwood."
- Make the report type distinguish *last-ring date*, *felling year*, and *felling-year
  range* as separate fields rather than overloading one number.

### 3.3 RWL master chronologies are built from raw widths — *high*
In `matcher._match_against_reference`, the RWL branch does:
```python
ref_values = df.mean(axis=1).values   # mean of RAW ring widths across cores
detrended, _ = detrend_series(ref_values)
```
Averaging *raw* widths across cores of different ages/growth rates, then detrending the
average, is methodologically wrong. The standard is to **detrend each series to a
dimensionless index first, then average the indices** (optionally with a robust/biweight
mean). The repo already does it correctly in the test helper
`tests/test_validation.py::_build_site_master` and has `detrend.build_chronology` — but the
production matcher doesn't use either. Every RWL-based match is weakened by this.

**Fix:** Route all RWL references through a single, correct chronology builder
(detrend-per-series → biweight robust mean → standardized master), cache the result.

### 3.4 Gleichläufigkeit is computed incorrectly and then ignored — *medium*
In `calculate_gleichlauf`:
```python
glk = 100 * (agreements - zeros * 0.5 + zeros * 0.5) / n
```
`- zeros*0.5 + zeros*0.5` cancels to zero, so the function returns `100 * agreements / n`,
which is not GLK (it also double-counts zero/zero pairs as agreements). The intended
formula is `GLK = (agreements + 0.5 * semi_agreements) / (n-1)`. Worse, the value is never
used in confidence scoring — GLK is a standard, robust cross-dating statistic and should
be. There's also no GLK significance test (Gsl/`G_sl`).

**Fix:** Implement GLK correctly, add its significance test, and incorporate it (alongside
t and overlap) into the confidence model.

### 3.5 No missing/false-ring handling — *high (defines "real" vs "toy")*
The core difficulty of dating field samples is **locally absent rings** (a year with no
measurable growth) and **false/double rings**. These shift everything inward of the error
and destroy a single rigid alignment. The current engine performs exactly one rigid slide
and cannot detect, locate, or compensate for a missing ring. Segment correlation
(`segment_correlation`) can *reveal* that something is wrong but offers no remedy.

**Fix (phased):** Start with COFECHA-style segmented re-correlation that tests segment
offsets of ±1, ±2 years to flag the likely location of a missing/extra ring; later add a
search that evaluates candidate insertions/deletions and re-scores. This is a substantial
algorithmic addition and should be its own milestone.

### 3.6 Spline detrending is an ad-hoc approximation — *medium*
`_fit_cubic_spline` maps a target wavelength to scipy's `UnivariateSpline` smoothing
factor via `s = n/(period/2)` then `s*var(y)`. This is not the standard dendro cubic
smoothing spline (Cook & Peters 1981), which is defined by a 50%-frequency-response
cutoff at a given wavelength. Results won't match dplR/standard tooling and aren't
reproducible across series of different length.

**Fix:** Implement the Cook & Peters cubic smoothing spline (or adopt a vetted
implementation), parameterized by cutoff wavelength as a fraction of series length, and
make the detrending method/curve configurable and recorded in the report.

### 3.7 Tucson parser robustness — *medium*
- `999` is treated as a stop marker, but `999` in 0.01 mm units is a legitimate 9.99 mm
  ring; conflating them can truncate or corrupt wide-ringed series. Tucson's true
  terminator is position/format-dependent (and `-9999`/`9990` are the missing/end
  sentinels in several variants).
- The CRN reader assumes a rigid 7-char value+depth layout and a hardcoded species
  regex; ITRDB has multiple CRN and "-noaa" tabular variants it won't parse.
- Header metadata extraction (lat/long/species/state) is brittle regex over the first 20
  lines.

**Fix:** Replace with a rigorous, format-aware parser (evaluate adopting an existing
maintained RWL/CRN reader rather than hand-rolling), add a corpus of real-file fixtures,
and parse the structured ITRDB metadata/NOAA template where available instead of regex
scraping.

### 3.8 Smaller correctness/typing items
- Era filtering in `_match_against_reference` (`era_start - len(sample)`, `+ 50` slop) is
  arbitrary and can both admit and reject valid dates inconsistently — replace with an
  explicit, documented window on the **felling year**.
- `cross_correlation` (FFT path) is defined but unused and its normalization is suspect;
  either fix and use it for speed or remove it.
- `prewhiten` is implemented but never used by the pipeline, and its AR estimation is
  questionable; decide whether residual chronologies are in scope and wire it in properly
  if so.
- Packaging says Python ≥3.9 but `setup_env.sh` checks for 3.10 and some modules rely on
  PEP 604 (`X | Y`) annotations; pin one floor (recommend 3.11) and make it consistent.

---

## 4. Imaging / measurement issues

### 4.1 The interactive viewer cannot run headless — *critical for cloud*
`MeasurementViewer.show()` calls `plt.show()` and depends on a GUI backend and mouse
events. This is the **single biggest cloud blocker** for the measurement feature: it
fundamentally cannot run on a server. Measurement must move to a browser-based workflow
(image served to a canvas; clicks/paths posted back; server samples the profile and
detects rings) or a headless batch detector with a human review step.

### 4.2 Ring detection is too naive for real samples — *high*
`detect_rings` uses CLAHE + smoothed gradient magnitude + a hand-rolled local-max scan.
Because it uses `abs(gradient)`, it fires on **both** the earlywood→latewood and the
latewood→earlywood transitions, tending to double-count boundaries. It has no model for
ring curvature, ring-porous hardwood vessels (oak earlywood pores devastate gradient
methods), knots, rays, or reflectance gradients. It will not produce trustworthy widths on
the oak/hemlock the README targets.

**Fix:** Treat automated detection as *assistive*, always human-reviewed. Improve the
signal model (directional latewood-density gradient, per-species tuning, perpendicular
band integration along the true path normal rather than vertical `axvline`s), and validate
against hand-measured fixtures. Consider established techniques/tools for ring detection as
a reference baseline.

### 4.3 Minor
- `_update_ring_display` draws vertical lines at the path x-coordinate; for diagonal or
  curved paths these mislocate boundaries.
- DPI is user-supplied and trusted; `estimate_dpi` exists but isn't wired into `measure`.

---

## 5. Architecture & cloud-readiness gaps

### 5.1 No service layer
Today it's a CLI plus an interactive GUI. To "run in the cloud" we need:
- **HTTP API** (recommend **FastAPI**): endpoints for upload, measurement, dating jobs,
  results, reference catalog.
- **Async job processing** for cross-dating runs (a worker queue — e.g. RQ/Celery/Arq —
  since matching against many references is not instantaneous and shouldn't block requests).
- **Object storage** for uploaded scans and generated plots (S3-compatible).
- **A browser UI** for measurement and for reviewing dating evidence.
- Keep the existing engine as a library the API calls — do **not** shell out to the CLI.

### 5.2 Reference data is scraped and re-parsed every run
`downloader.py` regex-scrapes NOAA's HTML directory listing at runtime;
`ChronologyIndex` re-parses **every** file on every `CrossdateMatcher` construction, and
`load_chronology` re-parses again during matching. `save_index`/`load_index` exist but are
unused. In a service this is slow, fragile (NOAA layout/rate limits/blocked egress), and
wasteful.

**Fix:** Curate a **versioned reference dataset** as a build/deploy artifact (or a managed
object-store mirror), parse it **once** into a persisted, serialized index + cached
per-site master chronologies, and load that at service start. Make reference-set version
part of every result's provenance.

### 5.3 No persistence / project model
A real platform needs durable domain objects: **Project → Structure/Building → Timber/Beam
→ Sample → Measurement series → Dating run → Result**, with provenance (who, when, scanner,
DPI, operator notes) and an audit trail. Today everything is loose CSV/JSON on disk.
**Fix:** Introduce a relational store (Postgres) and a schema for the above, with results
linked to the exact reference-set version and algorithm parameters used.

### 5.4 No container, CI, lockfile, or reproducible build
No Dockerfile, no CI, `uv.lock` is gitignored, deps unpinned. The project couldn't be
installed reproducibly in a fresh environment during this review. **Fix:** pin deps with a
committed lockfile, add a Dockerfile, add CI (lint + type-check + tests on every PR), and
add a SessionStart hook so web sessions can run tests/linters.

### 5.5 Input safety
Uploaded CSVs go straight into `pd.read_csv`/`np.loadtxt` with no size/shape validation;
downloaded filenames come from remote HTML with only a `startswith("..")` check (weak
path-traversal guard). **Fix:** validate and bound all inputs; sanitize any
externally-derived filenames; cap upload sizes; never trust scraped paths.

---

## 6. Testing & validation gaps

- Most unit tests correlate a sample that is an **exact subset** of the reference, so
  `r ≈ 1.0` trivially — they exercise plumbing, not dating difficulty.
- The realistic tests (`test_validation.py`) **skip** unless reference data is downloaded,
  and no reference data is committed, so in CI the only meaningful tests don't run.
- No fixtures for: missing-ring samples, no-bark-edge samples, hardwood scans, malformed
  Tucson files.

**Fix:** Commit a **small, license-clear fixture set** (a few real RWL files + a couple of
hand-measured sample CSVs with known answers, including at least one missing-ring and one
no-bark-edge case). Make the recover-the-date validation run in CI against those fixtures.
Add property tests for the parsers.

---

## 7. Proposed phased plan

Each phase is independently shippable and leaves the project in a better state.

### Phase 0 — Foundations (1 short iteration)
- Commit a lockfile; pin deps; settle the Python floor (3.11).
- Add CI (ruff + mypy + pytest) and a Dockerfile.
- Commit a small real-data fixture set; wire the recover-the-date validation into CI.
- Add a SessionStart hook for web sessions.
**Exit:** `pytest` green in CI on a clean checkout, including ≥3 real-data dating
validations.

### Phase 1 — Make the science correct (the core of "actually functional")
- **1a** Enforce a single canonical series orientation end-to-end; add guards (§3.1).
- **1b** Make `bark_edge` real: distinct last-ring / felling-year / felling-range outputs,
  with species sapwood models (§3.2).
- **1c** Correct RWL master-chronology construction (detrend-then-average, cached) (§3.3).
- **1d** Fix Gleichläufigkeit + significance; fold GLK into confidence (§3.4).
- **1e** Replace ad-hoc spline with Cook & Peters spline; record detrending in reports (§3.6).
- **1f** Harden the Tucson/CRN parsers against real-file variants + fixtures (§3.7).
- **1g** Rationalize era filtering; remove/repair dead code paths (§3.8).
**Exit:** Documented, reproducible dating on the fixture corpus; outputs distinguish
felling-year certainty levels; results match a reference implementation (e.g. dplR) within
tolerance on shared inputs.

### Phase 2 — Service spine
- FastAPI app wrapping the engine as a library; async job queue; Postgres project model;
  object storage; serialized reference index + cached masters loaded at startup;
  versioned reference dataset as a deploy artifact (§5.1–5.3).
- Input validation and upload limits (§5.5).
**Exit:** Upload a measurement CSV via API → dating job runs → results + diagnostic plot
retrievable; everything persisted with provenance and reference-set version.

### Phase 3 — Browser measurement workflow
- Replace the matplotlib GUI with a web canvas: serve the scan, capture path + ring
  marks, server-side profile sampling and assistive detection, human review/adjust, export
  to the canonical series format (§4.1–4.3).
- Improve assistive ring detection and validate against hand-measured fixtures.
**Exit:** A user can upload a scan, measure rings in the browser, and date the result
without touching a CLI or local GUI.

### Phase 4 — Algorithmic depth & hardening
- Missing/false-ring detection and assisted re-alignment (§3.5).
- Multi-sample replication / site master building in-app; cross-verification UI.
- Calibrated, species-aware confidence; configurable thresholds with documented provenance.
- Observability (structured logs, metrics), authn/authz, rate limiting, backups.
**Exit:** Handles real field samples with missing rings; defensible confidence reporting;
production-operable.

---

## 8. Build vs. reuse

Dendrochronology has a de-facto standard implementation in **dplR (R)**. We should not
blindly reimplement decades of refined methods. Recommended posture:
- **Reuse the *algorithms* faithfully** (Cook & Peters spline, biweight robust mean,
  standard cross-dating statistics) and **validate our outputs against dplR** on shared
  inputs as an acceptance gate.
- **Evaluate adopting a maintained Tucson/RWL parser** rather than hand-rolling format
  handling, given how many ITRDB variants exist.
- Keep our own thin engine for the parts we need to control (orientation, sapwood/felling
  logic, the service API), so we own the product surface while standing on validated math.

A short spike at the start of Phase 1 should confirm which external pieces we adopt vs.
reimplement; this plan assumes we reimplement the spline and chronology math (small,
well-specified) and lean on existing parsing/validation references where practical.

---

## 9. Risks & open questions

- **Reference coverage** for the target region/species/era ultimately bounds dating
  success more than any code change; curating the right reference set is a project in its
  own right.
- **Missing-ring handling (§3.5)** is the hardest algorithmic work and the difference
  between "demo" and "usable on field timber"; it deserves its own design doc.
- **Legal/licensing** of any committed reference fixtures and of derived chronologies
  needs checking before redistribution.
- **Scope of measurement-in-browser** (full annotation UI vs. assisted batch) is a product
  decision that changes Phase 3 size substantially.
</content>
</invoke>
