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
COPY scripts/ ./scripts/

# Install project dependencies, then swap opencv for headless
RUN pip install --no-cache-dir ".[all]" && \
    pip uninstall -y opencv-python 2>/dev/null || true && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless

# Bake commit SHA into the image so I-TRACE-4 (commit_sha in execution_log)
# works even when /.git is not present at runtime.
ARG AIDRA_COMMIT_SHA=unknown
ENV AIDRA_COMMIT_SHA=${AIDRA_COMMIT_SHA}

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
