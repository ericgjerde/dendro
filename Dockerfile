# Dendro dating API container.
#
# Builds an image that serves the cross-dating engine over HTTP. By default it
# ships the small synthetic reference fixtures so the API is functional out of
# the box for a demo; mount or copy a real ITRDB corpus and point
# DENDRO_REFERENCE_DIR at it for production use:
#
#   docker run -p 8000:8000 \
#     -e DENDRO_REFERENCE_DIR=/data/reference \
#     -v /path/to/itrdb:/data/reference dendro
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DENDRO_REFERENCE_DIR=/app/data/reference

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml setup.py README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[api]"

# Provide the synthetic fixtures as a default (demo) reference set.
COPY tests/fixtures/reference/synth /app/data/reference/synth

EXPOSE 8000

# Simple container healthcheck against the API.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["dendro-api"]
