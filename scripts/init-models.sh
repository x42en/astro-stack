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
    echo "  Downloading GraXpert AI model..."
    # GraXpert downloads its own models via its built-in mechanism
    # Try to trigger the download via Python
    python3 -c "
import sys
sys.path.insert(0, '/opt/graxpert')
try:
    from graxpert.ai_model_handling import download_model
    download_model('GraXpert-AI-1.0.0', models_dir='${GRAXPERT_DIR}')
    print('  GraXpert model downloaded successfully.')
except Exception as e:
    print(f'  WARNING: GraXpert model download failed: {e}')
    print('  You may need to run GraXpert once to trigger automatic download.')
" 2>&1 || echo "  WARNING: GraXpert model auto-download unavailable"
else
    echo "  Already present: ${GRAXPERT_MODEL}"
fi

# ── ASTAP star catalogue ──────────────────────────────────────────────────────
echo ""
echo "--- ASTAP star catalogue ---"
ASTAP_DB_DIR="${ASTAP_STAR_DB_PATH:-/opt/astap/stars}"
mkdir -p "${ASTAP_DB_DIR}"

# D50 catalogue (~500MB) — covers most deep sky objects
ASTAP_DB_FILE="${ASTAP_DB_DIR}/d50.1476"
if [ ! -f "${ASTAP_DB_FILE}" ]; then
    echo "  Downloading ASTAP D50 star catalogue (~500MB)..."
    wget -q "https://www.hnsky.org/d50.zip" -O /tmp/d50.zip \
        && unzip -q /tmp/d50.zip -d "${ASTAP_DB_DIR}" \
        && rm /tmp/d50.zip \
        && echo "  ASTAP D50 catalogue installed." \
        || echo "  WARNING: ASTAP D50 download failed — plate solving will not work."
else
    echo "  Already present: ASTAP D50 catalogue"
fi

echo ""
echo "=== Model initialisation complete ==="
echo "Models directory contents:"
ls -lah "${MODELS_DIR}"
