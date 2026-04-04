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
    ca-certificates xvfb libgomp1 \
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

RUN mkdir -p /opt/astap \
    && wget -q "https://sourceforge.net/projects/astap-program/files/linux_installer/astap_amd64.tar.gz/download" \
        -O /opt/astap/astap.tar.gz \
    && tar -xzf /opt/astap/astap.tar.gz -C /opt/astap --strip-components=1 \
    && rm /opt/astap/astap.tar.gz \
    && chmod +x /opt/astap/astap \
    && ln -s /opt/astap/astap /usr/local/bin/astap

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

# Copy and install application dependencies
COPY pyproject.toml .
RUN pip install -e ".[prod]" || \
    pip install \
        fastapi uvicorn[standard] websockets \
        sqlmodel asyncpg alembic greenlet \
        arq redis[hiredis] \
        pydantic pydantic-settings \
        python-jose[cryptography] passlib[bcrypt] \
        httpx astropy rawpy numpy Pillow \
        watchdog python-multipart aiofiles anyio structlog

# ── Stage 5: Cosmic Clarity & GraXpert sources ────────────────────────────
FROM python-deps AS ai-tools

# Cosmic Clarity — MIT licence Python scripts from setiastro
RUN git clone --depth=1 https://github.com/setiastro/cosmicclarity.git \
        /opt/cosmic-clarity \
    && pip install -r /opt/cosmic-clarity/requirements.txt \
    || echo "WARNING: Cosmic Clarity clone failed — mount sources manually"

# GraXpert — GPLv3 gradient removal
RUN pip install graxpert \
    || git clone --depth=1 https://github.com/Steffenhir/GraXpert.git \
        /opt/graxpert \
    && pip install -r /opt/graxpert/requirements.txt \
    || echo "WARNING: GraXpert install failed — mount sources manually"

# ── Stage 6: Final application image ──────────────────────────────────────
FROM ai-tools AS final

# Create non-root user for the application
RUN useradd -m -s /bin/bash astro \
    && mkdir -p /inbox /sessions /output /models \
    && chown -R astro:astro /inbox /sessions /output /models

# Copy venv and set PATH
COPY --from=ai-tools /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application source
COPY --chown=astro:astro app/ ./app/
COPY --chown=astro:astro pyproject.toml .

# Install application in development mode
RUN pip install -e .

# Create Alembic directory
RUN mkdir -p alembic/versions

# Copy init-models script
COPY --chown=astro:astro scripts/ ./scripts/
RUN chmod +x /app/scripts/init-models.sh 2>/dev/null || true

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command: start the FastAPI API server
# Workers are started separately (see docker-compose.yml)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--loop", "uvloop"]