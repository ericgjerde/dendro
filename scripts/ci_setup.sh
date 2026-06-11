#!/usr/bin/env bash
# Idempotent environment setup for CI and Claude Code web sessions.
# Ensures the package and its dev/api dependencies are importable so that
# `pytest` and `ruff` work. Safe to run repeatedly; never fails the session.
set -u

cd "$(dirname "$0")/.." || exit 0

# Prefer an existing virtualenv; otherwise install into the active interpreter.
if [ -d venv ]; then
    # shellcheck disable=SC1091
    . venv/bin/activate 2>/dev/null || true
fi

if ! python -c "import dendro" >/dev/null 2>&1; then
    python -m pip install --upgrade pip >/dev/null 2>&1 || true
    pip install -e ".[dev,api]" >/dev/null 2>&1 || true
fi

python -c "import dendro" >/dev/null 2>&1 \
    && echo "dendro environment ready" \
    || echo "warning: dendro not importable; run 'pip install -e .[dev,api]'"

exit 0
