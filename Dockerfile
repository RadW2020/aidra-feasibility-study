# AIDRA - Artificial Intelligence In-orbit Data pRocessing Assessment
# Python 3.11 slim — CPU-only (no CUDA) for ARM64

FROM python:3.11-slim-bookworm

# System deps for GDAL, rasterio, OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    libffi-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV GDAL_CONFIG=/usr/bin/gdal-config

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU-only FIRST (avoids pulling ~2GB CUDA libs)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY models/ ./models/
# Second copy of the cards OUTSIDE the /app/models volume: the volume shadows
# ./models at runtime and was only seeded on the first deploy, so git changes
# to the cards never reached the server. Synced at startup (cards_sync).
COPY models/cards/ ./models_dist/cards/
COPY scripts/ ./scripts/

# Install project dependencies, then swap opencv for headless
RUN pip install --no-cache-dir ".[all]" && \
    pip uninstall -y opencv-python 2>/dev/null || true && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless

# Bake commit SHA into the image so I-TRACE-4 (commit_sha in execution_log)
# works even when /.git is not present at runtime.
ARG AIDRA_COMMIT_SHA=unknown
ENV AIDRA_COMMIT_SHA=${AIDRA_COMMIT_SHA}
# The same SHA under a name no runtime config sets: the compose file and
# Coolify override AIDRA_COMMIT_SHA / SOURCE_COMMIT with the *pushed* commit,
# which a deploy can run before its image exists. This one always names the
# code inside the image (get_commit_sha prefers it).
ENV AIDRA_IMAGE_COMMIT=${AIDRA_COMMIT_SHA}

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
