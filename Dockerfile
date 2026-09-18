# =============================================================================
# Astro-Stack — Multi-stage Docker build
# Base: NVIDIA CUDA 12.8.1 + cuDNN 9 + Ubuntu 24.04
# Aligned with host driver (≥ 570.x announces CUDA 12.8). CUDA guarantees
# forward compatibility within the 12.x major, but matching base/runtime/driver
# eliminates a variable when diagnosing GPU init failures (notably
# `cudaSetDevice ... initialization error` from ORT 1.25 + CUDA fork).
# =============================================================================

# ── Stage 1: System dependencies & build tools ─────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS system-base

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3.12 python3.12-dev python3-pip python3.12-venv \
    # Build tools
    build-essential cmake pkg-config git wget curl \
    # LibRaw (for DSLR RAW conversion)
    libraw-dev \
    # Siril dependencies
    libgtk-3-dev libglib2.0-dev libfftw3-dev \
    libcfitsio-dev libgsl-dev libopencv-dev \
    wcslib-dev libjpeg-dev libpng-dev libtiff-dev \
    libheif-dev libexiv2-dev \
    # Misc
    ca-certificates libgomp1 zstd \
    && rm -rf /var/lib/apt/lists/*

# Create Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# ── Stage 2: Siril headless ────────────────────────────────────────────────
FROM system-base AS siril-build

# Install Siril 1.4.x from the official Siril PPA (ppa:lock042/siril).
# This PPA ships Siril 1.4.x for Ubuntu 24.04 (Noble) and is the distribution
# channel maintained by the Siril team for all Ubuntu releases.
# The previous AppImage approach required FUSE and failed silently in Docker
# (masked by || true), causing the fallback to install the outdated Ubuntu
# archive version (~1.0.x) that does not support the -noout register flag.
# The trailing siril-cli --version line ensures the build fails immediately
# if Siril is not correctly installed — no more silent regressions.
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:lock042/siril \
    && apt-get update \
    && apt-get install -y --no-install-recommends siril \
    && rm -rf /var/lib/apt/lists/* \
    && siril-cli --version

# ── Stage 3: ASTAP plate solver ───────────────────────────────────────────
FROM siril-build AS astap-install

# ASTAP plate solver — use the standalone `astap_cli` binary published on
# SourceForge (project: astap-program / linux_installer). This is the
# barebone command-line solver: no GTK, no display, no Xvfb wrapper needed.
# It supports our exact usage (-f -r -wcs -d) and is purpose-built for
# headless environments. The previous approach (install the GUI .deb and
# wrap it with xvfb-run) pulled libgtk2.0-0 + xvfb (~30+ MB) and added
# 100-300ms startup overhead per solve for nothing.
#
# The star catalogue (.290 / .1476 files in /opt/astap/stars) is downloaded
# separately by scripts/init-models.sh into a persisted volume — it is not
# bundled with the binary, so we just create the mount point here.
#
# Use wget with retries — SourceForge CDN can return 504 on first try.
RUN apt-get update \
    && apt-get install -y --no-install-recommends unzip \
    && wget --tries=5 --waitretry=15 --timeout=120 -q \
        "https://downloads.sourceforge.net/project/astap-program/linux_installer/astap_command-line_version_Linux_amd64.zip" \
        -O /tmp/astap_cli.zip \
    && unzip -j /tmp/astap_cli.zip astap_cli -d /usr/local/bin/ \
    && chmod +x /usr/local/bin/astap_cli \
    && rm /tmp/astap_cli.zip \
    && apt-get purge -y unzip \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/astap/stars \
    && /usr/local/bin/astap_cli -h > /dev/null 2>&1 || true

# ── Stage 4: Python venv & dependencies ───────────────────────────────────
FROM astap-install AS python-deps

WORKDIR /build

# Create virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA support (large download, cached separately)
# cu128 wheels match the 12.8.1 base above. PyTorch 2.7+ ships cu128 builds.
RUN pip install --upgrade pip setuptools wheel \
    && pip install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# Install all application dependencies BEFORE copying source
# This ensures a clean, reproducible pip install
# Note: [prod] extra does not exist in pyproject.toml — install deps explicitly
RUN pip install \
        fastapi "uvicorn[standard]" websockets \
        sqlmodel asyncpg alembic greenlet \
        arq "redis[hiredis]" \
        "pydantic[email]" pydantic-settings email-validator \
        "python-jose[cryptography]" "passlib[bcrypt]" \
        httpx astropy astroquery rawpy numpy Pillow ExifRead \
        skyfield timezonefinder \
        watchdog python-multipart aiofiles anyio structlog \
        # Live-stacking pipeline (frame alignment + hot-pixel rejection)
        astroalign scipy \
        # Cosmic Clarity & GraXpert AI inference runtime (CUDA 12.x build)
        onnxruntime-gpu

# ── Stage 5: Cosmic Clarity (via SetiAstroSuitePro) & GraXpert sources ────
FROM python-deps AS ai-tools

# Cosmic Clarity engines — formerly a standalone script bundle at
# github.com/setiastro/cosmicclarity (archived 2026-09-16, read-only).
# Replaced by SetiAstroSuitePro (SASpro), the actively maintained successor
# (GPL-3.0, github.com/setiastro/setiastrosuitepro), which bundles the same
# denoise/sharpen/superres/darkstar engines plus a new satellite-trail
# remover and a built-in aberration corrector, behind a real headless CLI:
# https://github.com/setiastro/setiastrosuitepro/wiki/CLI:-Command-Line-Interface
#
# `cosmicclarity` is the pip-installed entry point (see cosmic_adapter.py).
# SASpro manages its own model weights (no manual .pth download step here,
# unlike the old bundle) — verify on first headless run whether that means
# bundled-in-the-wheel or an on-demand download; if the latter, models must
# be pre-warmed into a persisted volume before an air-gapped deploy.
RUN pip install setiastrosuitepro \
    && cosmicclarity --help > /dev/null \
    && cosmicclarity denoise --help > /dev/null \
    && cosmicclarity sharpen --help > /dev/null \
    && cosmicclarity superres --help > /dev/null \
    && cosmicclarity satellite --help > /dev/null

# sirilpy — Siril's Python scripting module (SirilPyAdapter, see
# app/pipeline/adapters/siril_pyadapter.py). Not published on PyPI; it ships
# as source inside the `siril` apt package installed in the siril-build
# stage above, at /usr/share/siril/python_module (valid pyproject.toml).
# Installing it here (rather than depending on it being importable from the
# system Python) makes it available inside our own venv at /opt/venv.
RUN pip install /usr/share/siril/python_module \
    && python -c "import sirilpy" 

# GraXpert — GPLv3 gradient removal
# GraXpert uses MinIO S3 to download AI models
# We need to fetch the s3_secrets.py file which contains credentials/endpoint
RUN pip install graxpert[cuda] \
    || ( \
        git clone --depth=1 https://github.com/Steffenhir/GraXpert.git /opt/graxpert && \
        pip install -r /opt/graxpert/requirements.txt && \
        # Download s3_secrets.py which contains the MinIO endpoint for model downloads
        curl -sL "https://raw.githubusercontent.com/Steffenhir/GraXpert/main/graxpert/s3_secrets.py" \
            -o /opt/venv/lib/python3.12/site-packages/graxpert/s3_secrets.py \
        ) \
    || echo "WARNING: GraXpert install failed — mount sources manually"

# ── Stage 6: Final application image ──────────────────────────────────────
FROM ai-tools AS final

LABEL org.opencontainers.image.title="AstroStack" \
      org.opencontainers.image.description="Automated astrophotography processing pipeline with event-driven API" \
      org.opencontainers.image.url="https://github.com/x42en/astro-stack" \
      org.opencontainers.image.source="https://github.com/x42en/astro-stack" \
      org.opencontainers.image.licenses="GPL-3.0-or-later"

# Create non-root user for the application
RUN useradd -m -s /bin/bash astro \
    && mkdir -p /inbox /sessions /output /models /opt/ephemerides \
    && chown -R astro:astro /inbox /sessions /output /models /opt/ephemerides

# Skyfield ephemeris (DE440s, ~32 MB, covers 1849-2150) — bundled to keep
# planning offline. Supersedes the older DE421 with improved planetary
# positions; the small (\"s\") variant trims the time span to limit size.
# Network failure is non-fatal at build time; planner_service will warn at boot.
# A zero-byte file is *worse* than a missing one (jplephem crashes with a
# cryptic error), so we explicitly delete the artefact when the download
# fails or produces an empty file.
RUN set -eu; \
    if wget -q --tries=3 --timeout=120 \
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp" \
            -O /opt/ephemerides/de440s.bsp \
        && [ -s /opt/ephemerides/de440s.bsp ] \
        && [ "$(stat -c %s /opt/ephemerides/de440s.bsp)" -gt 1024 ]; then \
        chown astro:astro /opt/ephemerides/de440s.bsp; \
        echo "DE440s ephemeris downloaded ($(stat -c %s /opt/ephemerides/de440s.bsp) bytes)"; \
    else \
        rm -f /opt/ephemerides/de440s.bsp; \
        echo "WARNING: DE440s ephemeris download failed — planner will be disabled until file is provided"; \
    fi

# venv is already present (FROM ai-tools), just set PATH
ENV PATH="/opt/venv/bin:$PATH" \
    QT_QPA_PLATFORM=offscreen \
    XDG_DATA_HOME=/models
# XDG_DATA_HOME=/models — GraXpert (via appdirs.user_data_dir("GraXpert")) reads
# its model bundles from $XDG_DATA_HOME/GraXpert/{bge,denoise,deconvolution-*}-ai-models.
# Without this override appdirs falls back to /root/.local/share/GraXpert/ where
# only models GraXpert downloaded itself at runtime would live — bge / deconv-*
# bundles populated by scripts/init-models.sh into the /models volume would be
# silently invisible, forcing GraXpert to re-download them on every fresh
# container or (worse) skip the AI step entirely.
# Note: PYTHONPATH must NOT be set to /app here — it would shadow the installed
# alembic package with /app/alembic/ (our migrations dir) causing ImportError.
# The editable install (pip install -e .) creates a .pth file in site-packages
# that appends /app to sys.path AFTER site-packages, avoiding the conflict.

WORKDIR /app

# Copy application source
COPY --chown=astro:astro app/ ./app/
COPY --chown=astro:astro pyproject.toml .

# Install only the app package itself (all deps already in venv from python-deps stage)
# --no-deps avoids re-resolving dependencies and touching already-installed packages
RUN pip install --no-deps -e .

# Copy alembic configuration
COPY --chown=astro:astro alembic.ini ./alembic.ini
COPY --chown=astro:astro alembic/ ./alembic/
RUN mkdir -p alembic/versions

# Copy helper scripts (init-models.sh, probe.py, ...)
COPY --chown=astro:astro scripts/ ./scripts/
RUN chmod +x /app/scripts/init-models.sh /app/scripts/probe.py 2>/dev/null || true

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command: start the FastAPI API server
# Workers are started separately (see docker-compose.yml)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--loop", "uvloop"]