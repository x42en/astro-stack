# =============================================================================
# Astro-Stack — Multi-stage Docker build
# Base: NVIDIA CUDA 12.6.3 + cuDNN + Ubuntu 24.04
# =============================================================================

# ── Stage 1: System dependencies & build tools ─────────────────────────────
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS system-base

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
    ca-certificates xvfb libgomp1 zstd \
    && rm -rf /var/lib/apt/lists/*

# Create Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# ── Stage 2: Siril headless ────────────────────────────────────────────────
FROM system-base AS siril-build

# Install Siril 1.4.2 from AppImage (simplest cross-distro approach)
RUN wget -q "https://siril.org/download/siril-1.4.2-x86_64.AppImage" \
        -O /opt/siril.AppImage \
    && chmod +x /opt/siril.AppImage \
    && /opt/siril.AppImage --appimage-extract \
    && mv squashfs-root /opt/siril-extracted \
    && ln -s /opt/siril-extracted/usr/bin/siril-cli /usr/local/bin/siril-cli \
    && ln -s /opt/siril-extracted/usr/bin/siril /usr/local/bin/siril \
    || true

# Fallback: install Siril from PPA if AppImage fails
RUN command -v siril-cli || \
    (apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:lock042/siril \
    && apt-get update \
    && apt-get install -y --no-install-recommends siril \
    && rm -rf /var/lib/apt/lists/*)

# ── Stage 3: ASTAP plate solver ───────────────────────────────────────────
FROM siril-build AS astap-install

# ASTAP plate solver — install via apt-get (not dpkg) so that declared
# Depends: (Qt5, libGL, …) are resolved automatically.
# libgtk2.0-0 (→ libgdk-x11-2.0 / libgtk-x11-2.0) is NOT listed in the
# .deb Depends but is required at runtime by the ASTAP binary; install it
# explicitly before the package so the linker finds it.
# Use wget with retries — curl + SourceForge redirect CDN can return 504.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgtk2.0-0 \
    && wget --tries=5 --waitretry=15 --timeout=120 -q \
        "https://downloads.sourceforge.net/project/astap-program/linux_installer/astap_amd64.deb" \
        -O /tmp/astap.deb \
    && apt-get install -y --no-install-recommends /tmp/astap.deb \
    && rm /tmp/astap.deb \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 4: Python venv & dependencies ───────────────────────────────────
FROM astap-install AS python-deps

WORKDIR /build

# Create virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA support (large download, cached separately)
RUN pip install --upgrade pip setuptools wheel \
    && pip install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126

# Install all application dependencies BEFORE copying source
# This ensures a clean, reproducible pip install
# Note: [prod] extra does not exist in pyproject.toml — install deps explicitly
RUN pip install \
        fastapi "uvicorn[standard]" websockets \
        sqlmodel asyncpg alembic greenlet \
        arq "redis[hiredis]" \
        pydantic pydantic-settings \
        "python-jose[cryptography]" "passlib[bcrypt]" \
        httpx astropy rawpy numpy Pillow \
        watchdog python-multipart aiofiles anyio structlog \
        # Cosmic Clarity & GraXpert AI inference runtime (CUDA 12.x build)
        onnxruntime-gpu

# ── Stage 5: Cosmic Clarity & GraXpert sources ────────────────────────────
FROM python-deps AS ai-tools

# Cosmic Clarity — MIT licence Python scripts from setiastro
# Clone first; install requirements separately so a pip failure is fatal
# (not silently swallowed by the || fallback on the clone).
# onnxruntime-gpu is already installed above; the requirements.txt may pin
# additional helpers (tifffile, etc.) that we still want.
RUN git clone --depth=1 https://github.com/setiastro/cosmicclarity.git \
        /opt/cosmic-clarity \
    || echo "WARNING: Cosmic Clarity clone failed — mount sources manually"
RUN test -f /opt/cosmic-clarity/requirements.txt \
    && pip install -r /opt/cosmic-clarity/requirements.txt \
    || true
# sharpen, super-resolution and darkstar scripts import PyQt6 unconditionally
# at the top of their module — even headless CLI invocations need the package.
# QT_QPA_PLATFORM=offscreen (set below) lets Qt run without a real display.
RUN pip install PyQt6 || true

# GraXpert — GPLv3 gradient removal
# GraXpert uses MinIO S3 to download AI models
# We need to fetch the s3_secrets.py file which contains credentials/endpoint
RUN pip install graxpert \
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

# Create non-root user for the application
RUN useradd -m -s /bin/bash astro \
    && mkdir -p /inbox /sessions /output /models \
    && chown -R astro:astro /inbox /sessions /output /models

# venv is already present (FROM ai-tools), just set PATH
ENV PATH="/opt/venv/bin:$PATH" \
    QT_QPA_PLATFORM=offscreen
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