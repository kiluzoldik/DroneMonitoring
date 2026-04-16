FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_VERSION=1.8.3 \
    PORT=8000

# System deps for geopandas/fiona/rtree/osmnx
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    libgeos-dev \
    libproj-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install "poetry==${POETRY_VERSION}" && \
    poetry install --only main --no-ansi

COPY . .

RUN mkdir -p /app/data /app/cache /app/cache_data && \
    useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Default Redis URL can be overridden at runtime
ENV REDIS_URL=redis://redis:6379/0

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
