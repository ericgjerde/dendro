# Dendrochronology Assisted Ranking CLI

`dendro` is a public Northeast-first CLI for **assisted dendrochronology dating** on macOS and Linux.

It helps you:
- measure ring widths from scanned samples,
- download and index Northeast ITRDB references,
- rank plausible outer-ring calendar-year alignments,
- decide whether a result is recommended, merely ranked, or inconclusive.

It does **not** claim publication-grade or fully automated secure dating. The default product contract is assisted ranking with explicit diagnostics.

## Supported Scope

- Region: Northeast United States (`CT`, `MA`, `ME`, `NH`, `NY`, `RI`, `VT`)
- Platforms: macOS and Linux
- Primary workflow: CLI first, with interactive measurement for scan review
- Output: stable JSON report schema and human-readable summaries
- Walpole mode: late-1700s Walpole, NH house-timber material-group inference for `hemlock`, `white_pine`, `hard_pine`, `oak`, and `chestnut`

Walpole mode is intentionally conservative:
- it ranks and recommends **material groups**, not exact botanical species,
- it uses historical context plus dating evidence, not image-only wood anatomy,
- it returns `inconclusive` when the requested context is unsupported or a material group lacks adequate reference coverage.

## Status Model

`dendro date` returns one of three report states:

- `recommended`: one candidate alignment cleared the current policy gates
- `ranked`: candidates exist, but none is strong enough to recommend
- `inconclusive`: the sample, references, or statistics are too weak to support ranking

When bark edge is supplied and the result is `recommended`, the top candidate may be treated as a **possible felling year**. Otherwise the tool reports candidate **outer-ring years** only.

## Installation

```bash
cd dendrochronology
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.9+.

## Quick Start

```bash
source .venv/bin/activate

# Download Northeast references plus NOAA sidecar metadata
dendro download --states=me,nh,vt,ma,ct,ri,ny

# Inspect the indexed manifest
dendro info --json

# Measure a scanned sample; writes both CSV and session JSON
dendro measure sample.tiff --dpi=1200 --output sample.measurements.csv

# Rank candidate alignments
dendro date sample.measurements.csv \
  --reference data/reference \
  --era-start 1750 \
  --era-end 1850 \
  --json

# Infer likely Walpole material groups before dating
dendro infer-materials sample.measurements.csv \
  --reference data/reference \
  --town Walpole \
  --state NH \
  --built-year-range 1760:1800 \
  --member-type frame \
  --json

# Let Walpole mode auto-select a supported material group, then date it
dendro date sample.measurements.csv \
  --reference data/reference \
  --town Walpole \
  --state NH \
  --built-year-range 1760:1800 \
  --member-type frame \
  --auto-material \
  --json
```

## CLI Overview

### `dendro download`

Downloads `rwl` and/or `crn` files from NOAA NCEI together with NOAA sidecar metadata, then builds a persisted local manifest.

```bash
dendro download --states=nh,vt,ma --species=PIST,TSCA,QUAL,QURU
```

### `dendro info`

Summarizes the indexed reference inventory, including file types, species coverage, states, and missing metadata.

```bash
dendro info --reference data/reference --json
```

### `dendro measure`

Opens an interactive measurement workflow for scans.

Artifacts:
- canonical CSV export in `oldest_to_newest` order
- persisted session JSON with path, boundaries, QC warnings, and exported widths

```bash
dendro measure beam-end.tiff --dpi=1200 \
  --output beam-end.measurements.csv \
  --session-output beam-end.session.json
```

### `dendro date`

Ranks candidate outer-ring years against the indexed Northeast references.

```bash
dendro date beam-end.measurements.csv \
  --reference data/reference \
  --orientation auto \
  --era-start 1750 \
  --era-end 1850 \
  --top 5 \
  --json
```

Key options:
- `--orientation auto|oldest_to_newest|bark_to_pith`
- `--species PIST,TSCA,...`
- `--states NH,VT,...`
- `--bark-edge/--no-bark-edge`
- `--output report.json`
- `--json`

When you know the timber species, pass `--species`. Species-aware ranking is materially more reliable than unconstrained matching.

Walpole-specific options:
- `--material-group hemlock|white_pine|hard_pine|oak|chestnut`
- `--auto-material`
- `--town Walpole`
- `--state NH`
- `--built-year-range 1760:1800`
- `--member-type frame|brace|sill|joist|rafter|board|unknown`
- `--context-profile walpole_nh_late_1700s_house`

### `dendro infer-materials`

Ranks likely Walpole material groups from a scan, measurement session, or measurement CSV.

```bash
dendro infer-materials beam-end.measurements.csv \
  --reference data/reference \
  --town Walpole \
  --state NH \
  --built-year-range 1760:1800 \
  --member-type frame \
  --json
```

### `dendro parse`

Summarizes a Tucson file, measurement CSV, or saved measurement session JSON.

```bash
dendro parse data/reference/nh/nh001.rwl
```

## Measurement Workflow

1. In path mode, click from bark to pith.
2. Press `ENTER` to switch into ring review mode.
3. Add or remove boundaries manually, or press `A` for auto-detect.
4. Press `ENTER` to export the canonical CSV and session JSON.

The session file stores:
- image path and DPI,
- path points,
- boundary positions,
- bark-to-pith widths,
- exported `oldest_to_newest` widths,
- QC warnings.

## JSON Report Schema

`dendro date --json` emits:

```json
{
  "status": "recommended|ranked|inconclusive",
  "policy_version": "2026.04-assisted-ranking-v1",
  "sample": {
    "name": "sample",
    "length": 80,
    "bark_edge": true,
    "requested_orientation": "auto",
    "chosen_orientation": "bark_to_pith",
    "analysis_orientation": "oldest_to_newest"
  },
  "best_candidate": {},
  "candidates": [],
  "material_inference": null,
  "diagnostics": {},
  "warnings": []
}
```

## Validation and Quality Gates

The repository ships with:
- unit and contract tests,
- real-data validation against included Northeast references,
- CLI smoke coverage,
- packaging and install checks in GitHub Actions.

Run the main local gates with:

```bash
source .venv/bin/activate
pytest -q
python scripts/validate_crossdating.py --reference-dir data/reference --suite-file tests/fixtures/validated_northeast_v1.json
python scripts/validate_crossdating.py --reference-dir data/reference --walpole-suite tests/fixtures/walpole_benchmark_v1.json
```

The curated v1 benchmark suite lives at `tests/fixtures/validated_northeast_v1.json`.
It is intentionally deterministic and represents the Northeast cases that are
currently validated for release gating. The exploratory sampler remains
available through `--num-tests` for ad hoc evaluation, but it is not the public
release gate.

The repo also includes a smoke-test measurement fixture and expected output:

```bash
python -m dendro.cli.main date tests/fixtures/known_samples/nh001_297031.csv \
  --reference data/reference \
  --json \
  --era-start 1500 \
  --era-end 2000

python -m dendro.cli.main infer-materials tests/fixtures/walpole/bp7s_measurements.csv \
  --reference data/reference \
  --town Walpole \
  --state NH \
  --built-year-range 1760:1800 \
  --member-type frame \
  --json
```

## Known Limits

- The validated scope is Northeast-first; broader ITRDB support is future work.
- Walpole mode is specific to late-1700s Walpole, NH house timbers. It is not a generic species-identification claim.
- Chestnut is carried as a required-coverage material group. Until the repo ships a real chestnut reference corpus, chestnut recommendations should remain conservative or inconclusive.
- The curated benchmark suite is intentionally smaller than the full reference
  corpus. Expanding that suite remains ongoing accuracy work.
- Imaging is interactive and review-driven, not a fully automated vision pipeline.
- A `ranked` result is useful for investigation, not a secure date claim.
- Samples shorter than ~30 rings often end up inconclusive.

## Repository Layout

```text
src/dendro/
  cli/           Public CLI contract
  crossdating/   Detrending, correlation, ranking, policy
  imaging/       Scan measurement and session export
  reference/     Downloader, metadata resolution, persisted manifest
tests/           Unit, contract, and real-data validation
scripts/         Validation and release-support scripts
```
