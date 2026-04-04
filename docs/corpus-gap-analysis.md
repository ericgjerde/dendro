# Corpus Gap Analysis

Snapshot date: 2026-04-04

Command used:

```bash
python scripts/validate_crossdating.py \
  --reference-dir data/reference \
  --corpus-sweep \
  --scope all \
  --suite-file tests/fixtures/validated_northeast_v1.json \
  --output-json /tmp/dendro-corpus-sweep.json \
  --top-n 20 \
  --workers 4
```

## Sweep Scope

- Eligible files: 91
- Eligible series: 1,121
- Leave-one-file-out evaluation: yes
- Species filter: yes, using the indexed species for each source file
- Era window: true sample span with +/- 50 years

## Overall Results

- Top-1 within +/- 2 years: 427 / 1,121 (38.1%)
- Top-5 contains the correct year: 577 / 1,121 (51.5%)
- Correct and `recommended`: 123 / 1,121 (11.0%)
- Correct but only `ranked`: 304 / 1,121 (27.1%)
- `ranking_miss`: 237 / 1,121 (21.1%)
- `long_offset_false_positive`: 298 / 1,121 (26.6%)
- `search_miss`: 101 / 1,121 (9.0%)
- `no_match`: 54 / 1,121 (4.8%)

The dominant failure mode is not total absence of candidates. The system usually
finds something, but it often promotes the wrong era to the top of the ranking.

## Main Failure Areas

### 1. Long-offset false positives dominate the common-species corpus

- `long_offset_false_positive`: 298 cases
- `ranking_miss`: 237 cases
- Wrong top-1 alignments skew strongly earlier than the truth:
  - earlier: 555
  - later: 81
  - mean top-1 error across failed ranked cases: about -131 years

This is most visible in the large eastern hemlock and red spruce networks:

- `TSCA`: 423 cases total, 47.8% top-1 pass, 27.9% ranking miss, 18.7% long-offset false positive
- `PCRU`: 385 cases total, 48.3% top-1 pass, 26.0% ranking miss, 20.8% long-offset false positive

Observed pattern:

- many wrong winners still have respectable `r` and `t` values,
- many failures are older calendar assignments with strong local similarity,
- the correct year is often still present in the candidate list.

Evidence that this is primarily a ranking problem:

- among `ranking_miss` cases, the correct year rank distribution starts with:
  - rank 2: 74 cases
  - rank 3: 31 cases
  - rank 4: 25 cases
  - rank 5: 20 cases

Likely cause:

- the composite score is still too willing to reward local correlation peaks that
  occur far earlier in the chronology,
- long-offset aliases are not penalized enough,
- species-only filtering still leaves a large enough search space that strong
  analog chronologies can outrank the right year.

### 2. Rare-species failures are mostly coverage failures, not ranking failures

The `no_match` bucket is concentrated in a small set of species:

- `JUVI`: 16 no-match cases
- `PIPA`: 15 no-match cases
- `THOC`: 6 no-match cases
- `QUST`: 5 no-match cases
- `PIRE`: 4 no-match cases

Representative examples:

- `me022.rwl` (`THOC`): reference count after exclusion = 1
- `nh005.rwl` (`PIRE`): reference count after exclusion = 1
- `nh006.rwl` (`JUVI`): reference count after exclusion = 1

Likely cause:

- source-file exclusion leaves only one or very few species-matching references,
- the current species-filtered search has no fallback strategy when the coverage
  floor collapses.

### 3. State-level weakness is driven by species/domain mix

Top-1 pass rate by state:

- `CT`: 84.9%
- `VT`: 73.8%
- `NH`: 52.2%
- `ME`: 37.1%
- `NY`: 26.8%
- `MA`: 17.5%

This is not just geography. The weak states are loaded with harder domains:

- `MA`: mostly `TSCA`, `PCRU`, and `PIRI`, with heavy ranking-miss / long-offset behavior
- `NY`: `TSCA` is mixed with weak hardwood and historical-building groups such as `QUAL`, `QUPR`, `PIPA`, and `JUVI`
- `NH`: strong for `PCRU`, weak for `TSCA`, `JUVI`, and `PIRE`

Likely cause:

- the current benchmark pool mixes high-quality cross-site ecology chronologies
  with historical/hardwood/sparse-domain material,
- one global scoring policy is being asked to solve materially different dating problems.

### 4. Short series are fragile, but long series still fail for ranking reasons

Length-bucket top-1 pass rate:

- 50-99 rings: 4.0%
- 100-149 rings: 18.4%
- 150-249 rings: 39.8%
- 250+ rings: 39.5%

This means length matters, but it is not the main blocker after about 150 rings.
Long series still fail because of wrong-era promotion:

- `250+` bucket long-offset false positives: 169 cases
- `250+` bucket ranking misses: 144 cases

Likely cause:

- weak samples need better gating,
- but improving minimum-length thresholds alone will not fix the dominant error mode.

### 5. Recommendation policy is conservative relative to actual recovery

- curated suite: 5 / 5 correct, 0 / 5 `recommended`
- full sweep: 427 correct top-1, but only 123 `recommended`

This is partly intentional, but it still creates a product tension:

- there are many correct recoveries that never clear the recommendation policy,
- the current release gate verifies recovery, but not recommendation behavior.

Likely cause:

- the recommendation thresholds are conservative,
- the gap-to-next and segment-consistency gates are tripped even when the top year is correct,
- calibration has not yet been done against the broader corpus.

## Remediation Priorities

### P1. Redesign ranking features to suppress earlier false peaks

Focus areas:

- penalize long-offset aliases directly,
- use stronger candidate-separation features,
- incorporate year-neighborhood consistency instead of relying so heavily on a
  single best local peak,
- consider collapsing near-duplicate references or chronology families before ranking.

Success signal:

- reduce `ranking_miss` + `long_offset_false_positive` materially in `TSCA` and `PCRU`
- raise full-sweep top-1 pass rate before changing recommendation policy

### P1. Fix sparse-species coverage and fallback behavior

Focus areas:

- bundle more references for `JUVI`, `PIPA`, `THOC`, `PIRE`, and `QUST`,
- when filtered reference count is too low, fall back to an explicitly labeled
  broader search instead of returning no match silently.

Success signal:

- reduce `no_match` cases driven by reference count <= 1

### P1. Split validation by domain, not just by region

Focus areas:

- track separate benchmark slices for:
  - strong conifer ecology chronologies,
  - sparse-species chronologies,
  - hardwood / historical-building material,
- stop treating one acceptance target as sufficient for all of them.

Success signal:

- per-domain acceptance targets that reflect actual difficulty and intended product scope

### P2. Calibrate recommendation policy after ranking improves

Focus areas:

- separate "top-1 is right" from "safe to recommend publicly",
- tune policy thresholds on the expanded corpus rather than the curated five-case suite,
- check how many correct top-1 cases fail because of score-gap and segment thresholds.

Success signal:

- recommendation precision stays high while recommendation recall improves on the validated domain

## Immediate Follow-Up Work

The analysis suggests four concrete follow-up tracks:

1. ranking overhaul for long-offset false positives in `TSCA` and `PCRU`
2. coverage/fallback work for sparse species
3. domain-specific benchmark slices and acceptance targets
4. recommendation calibration after the ranking changes land
