# =============================================================================
# DeepArticle - Multi-Agent Academic Paper Analysis System
# Container image: runs the FastAPI + React web UI.
# =============================================================================
FROM python:3.11-slim AS base

# Faster, cleaner Python in containers.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # The UI/agents print emoji; keep stdout UTF-8 inside the container too.
    PYTHONIOENCODING=utf-8

WORKDIR /app

# System deps: build tools occasionally needed by wheels (e.g. PyMuPDF on some
# platforms). Kept minimal and removed after install to keep the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so Docker can cache this layer across code changes.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the application source.
COPY . .

# Application code lives under src/ (flat layout) — make it importable.
ENV PYTHONPATH=/app/src

# Persisted API cache lives here (see utils/cache.py). Declared as a volume so
# repeated searches survive container restarts.
ENV DEEPARTICLE_CACHE_DIR=/app/.cache
VOLUME ["/app/.cache"]

EXPOSE 8000

# Simple healthcheck against the config endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/config || exit 1

# Bind to 0.0.0.0 so the server is reachable from outside the container.
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
