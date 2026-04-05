# Local Known-Date Scan Validation

This workflow exists to close the main remaining blocker in the repo: real `scan -> measure -> infer/date` validation with trusted local samples.

## What This Is For

Use it when you have a real slice, log round, core, or timber sample and you know the true outer-ring year or cut date with confidence.

Examples:
- a hemlock firewood round cut this year
- a white pine slice from a felled tree with a known cut year
- an ash sample used to verify that unsupported materials stay `inconclusive`

## Suite Workflow

Create a suite once:

```bash
dendro init-validation-suite data/local_validation/walpole_firewood \
  --name "Walpole firewood known-date scans" \
  --town Walpole \
  --state NH \
  --built-year-range 1760:1800
```

Add a supported case:

```bash
dendro add-validation-case data/local_validation/walpole_firewood hemlock_round_2026_001 \
  --species-name "Eastern Hemlock" \
  --species-code TSCA \
  --material-group hemlock \
  --true-outer-ring-year 2026 \
  --sample-origin firewood \
  --bark-edge \
  --scan-dpi 1200
```

Add an unsupported-policy case such as ash:

```bash
dendro add-validation-case data/local_validation/walpole_firewood ash_round_2026_001 \
  --species-name "White Ash" \
  --species-code FRAM \
  --expected-policy-outcome unsupported_inconclusive \
  --true-outer-ring-year 2026 \
  --sample-origin firewood \
  --scale-included
```

Check readiness at any time:

```bash
dendro validation-info data/local_validation/walpole_firewood
dendro validation-info data/local_validation/walpole_firewood --json
```

## Per-Case Files

Each case lives under `cases/<case_id>/` and includes:
- `case.json`: required metadata and expected policy outcome
- `README.md`: capture checklist and artifact paths
- `artifacts/scan.tif`: the future raw scan image
- `artifacts/measurement.session.json`: saved review session from `dendro measure`
- `artifacts/measurements.csv`: canonical exported widths

## Minimum Metadata

Every case should record:
- trusted species name
- true outer-ring year
- whether bark edge is present
- expected policy outcome:
  - `supported_dateable`
  - `unsupported_inconclusive`

Recommended:
- species code
- supported material group when applicable
- scan DPI or an in-frame scale
- cut date
- sample origin such as `firewood` or `structure`

## Why Unsupported Cases Matter

Ash is useful even though it is not part of the Walpole shipping claim. It helps validate that the pipeline does not produce false confident material/date recommendations on out-of-scope material.
