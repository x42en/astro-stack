#!/usr/bin/env bash
# =============================================================================
# init-models.sh — Download AI model weight files to the /models volume
#
# Run this ONCE before starting the application:
#   docker compose --profile init up astro-init-models
#
# Or locally:
#   docker run --rm -v ./data/models:/models astro-stack:latest /app/scripts/init-models.sh
#
# Models are downloaded from GitHub releases and stored in /models/.
# Subsequent container builds do NOT re-download models.
# =============================================================================

set -euo pipefail

MODELS_DIR="${MODELS_PATH:-/models}"
COSMIC_DIR="${MODELS_DIR}"
GRAXPERT_DIR="${MODELS_DIR}/graxpert"
COSMIC_SRC="${COSMIC_CLARITY_SOURCE_PATH:-/opt/cosmic-clarity}"

mkdir -p "${COSMIC_DIR}" "${GRAXPERT_DIR}"

echo "=== Astro-Stack Model Initialiser ==="
echo "Models directory: ${MODELS_DIR}"

# ── Cosmic Clarity models ─────────────────────────────────────────────────────
echo ""
echo "--- Cosmic Clarity models ---"

# Models are co-located in the Cosmic Clarity source repo
# Copy from the cloned source if available, else download from GitHub releases
if [ -d "${COSMIC_SRC}" ]; then
    echo "Copying models from Cosmic Clarity source at ${COSMIC_SRC}..."
    for model_file in \
        "denoise_cnn.pth" \
        "deep_denoise_cnn.pth" \
        "deep_sharp_stellar_cnn.pth" \
        "deep_nonstellar_sharp_cnn_radius_1.pth" \
        "deep_nonstellar_sharp_cnn_radius_2.pth" \
        "deep_nonstellar_sharp_cnn_radius_4.pth" \
        "deep_nonstellar_sharp_cnn_radius_8.pth" \
        "satelliteremoval128featuremaps.pth" \
        "nonstellar_sharp_cnn_radius_1.pth" \
        "nonstellar_sharp_cnn_radius_2.pth" \
        "nonstellar_sharp_cnn_radius_4.pth" \
        "nonstellar_sharp_cnn_radius_8.pth" \
        "sharp_cnn_radius_1.pth"; do
        src="${COSMIC_SRC}/${model_file}"
        dst="${COSMIC_DIR}/${model_file}"
        if [ -f "${src}" ] && [ ! -f "${dst}" ]; then
            cp "${src}" "${dst}"
            echo "  Copied: ${model_file}"
        elif [ -f "${dst}" ]; then
            echo "  Already present: ${model_file}"
        else
            echo "  WARNING: ${model_file} not found in source"
        fi
    done
else
    echo "Cosmic Clarity source not found at ${COSMIC_SRC}."
    echo "Downloading from GitHub releases..."
    COSMIC_RELEASE_URL="https://github.com/setiastro/cosmicclarity/raw/main"

    for model_file in \
        "denoise_cnn.pth" \
        "deep_denoise_cnn.pth" \
        "deep_sharp_stellar_cnn.pth" \
        "deep_nonstellar_sharp_cnn_radius_1.pth" \
        "deep_nonstellar_sharp_cnn_radius_2.pth" \
        "deep_nonstellar_sharp_cnn_radius_4.pth"; do
        dst="${COSMIC_DIR}/${model_file}"
        if [ ! -f "${dst}" ]; then
            echo "  Downloading: ${model_file}..."
            wget -q "${COSMIC_RELEASE_URL}/${model_file}" -O "${dst}" \
                || echo "  WARNING: Failed to download ${model_file}"
        else
            echo "  Already present: ${model_file}"
        fi
    done
fi

# ── GraXpert AI model ─────────────────────────────────────────────────────────
echo ""
echo "--- GraXpert AI model ---"

GRAXPERT_MODEL="GraXpert-AI-1.0.0.pth"
GRAXPERT_MODEL_PATH="${GRAXPERT_DIR}/${GRAXPERT_MODEL}"

if [ ! -f "${GRAXPERT_MODEL_PATH}" ]; then
    echo "  GraXpert AI model is optional for gradient removal."
    echo "  The model will be downloaded automatically on first use via the GUI."
    echo "  For headless usage, you can manually download from:"
    echo "    https://github.com/Steffenhir/GraXpert/releases"
    # Note: GraXpert's download_model() is only available in GUI mode
    # The model is loaded lazily when first needed
    echo "  Skipping automatic download (requires GUI)."
else
    echo "  Already present: ${GRAXPERT_MODEL}"
fi

# ── ASTAP star catalogue ──────────────────────────────────────────────────────
echo ""
echo "--- ASTAP star catalogue ---"
ASTAP_DB_DIR="${ASTAP_STAR_DB_PATH:-/opt/astap/stars}"
mkdir -p "${ASTAP_DB_DIR}"

# D50 catalogue (~500MB) — covers most deep sky objects
# Download from SourceForge (new location, hnsky.org is deprecated)
ASTAP_DB_FILE="${ASTAP_DB_DIR}/d50.1476"
if [ ! -f "${ASTAP_DB_FILE}" ]; then
    echo "  Downloading ASTAP D50 star catalogue (~500MB)..."
    wget -q "https://sourceforge.net/projects/astap-program/files/star_databases/d50_star_database.deb/download" -O /tmp/d50.deb \
        && dpkg -x /tmp/d50.deb /tmp/d50_extract \
        && mv /tmp/d50_extract/usr/share/astap/star_database/d50* "${ASTAP_DB_DIR}/" 2>/dev/null \
        || mv /tmp/d50_extract/usr/share/astap/d50* "${ASTAP_DB_DIR}/" 2>/dev/null \
        && rm -rf /tmp/d50.deb /tmp/d50_extract \
        && echo "  ASTAP D50 catalogue installed." \
        || echo "  WARNING: ASTAP D50 download failed — plate solving will not work."
else
    echo "  Already present: ASTAP D50 catalogue"
fi

echo ""
echo "=== Model initialisation complete ==="
echo "Models directory contents:"
ls -lah "${MODELS_DIR}"
