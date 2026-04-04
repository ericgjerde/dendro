# Release Checklist

Before tagging a public release:

1. Run `pytest -q`.
2. Run `python scripts/validate_crossdating.py --reference-dir data/reference --suite-file tests/fixtures/validated_northeast_v1.json`.
3. Run CLI smoke checks:
   - `python -m dendro.cli.main --help`
   - `python -m dendro.cli.main measure --help`
   - `python -m dendro.cli.main info --reference data/reference --json`
   - `python -m dendro.cli.main date tests/fixtures/known_samples/nh001_297031.csv --reference data/reference --json --era-start 1500 --era-end 2000`
4. Confirm `data/reference/.dendro-reference-manifest.json` is up to date.
5. Confirm README examples match current CLI output and option names.
6. Build distribution artifacts with `python -m build`.
7. Verify the release notes describe the product as assisted ranking, not secure dating.
