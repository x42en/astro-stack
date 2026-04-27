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

# Cosmic Clarity scripts load models from their own directory (exe_dir =
# /opt/cosmic-clarity/).  The Dockerfile downloads them at build time, but
# this section serves as a runtime fallback in case the build was done
# without network access (e.g. air-gapped CI).
#
# New model names (current scripts, AI3.x series):
#   denoise       → deep_denoise_cnn_AI3_6.pth
#   sharpen       → deep_sharp_stellar_cnn_AI3_5s.pth
#                   deep_nonstellar_sharp_cnn_radius_{1,2,4,8}AI3_5s.pth
#   super-res     → superres_{2,3,4}x.pth
#   darkstar      → darkstar_v2.1.pth  darkstar_v2.1c.pth

CC_RELEASE="https://github.com/setiastro/cosmicclarity/releases/download/Linux"

if [ -d "${COSMIC_SRC}" ]; then
    MISSING=0
    for model_file in \
        "deep_denoise_cnn_AI3_6.pth" \
        "deep_sharp_stellar_cnn_AI3_5s.pth" \
        "deep_nonstellar_sharp_cnn_radius_1AI3_5s.pth" \
        "deep_nonstellar_sharp_cnn_radius_2AI3_5s.pth" \
        "deep_nonstellar_sharp_cnn_radius_4AI3_5s.pth" \
        "deep_nonstellar_sharp_cnn_radius_8AI3_5s.pth" \
        "superres_2x.pth" \
        "superres_3x.pth" \
        "superres_4x.pth" \
        "darkstar_v2.1.pth" \
        "darkstar_v2.1c.pth"; do
        target="${COSMIC_SRC}/${model_file}"
        if [ -f "${target}" ]; then
            echo "  OK: ${model_file}"
        else
            echo "  Downloading missing model: ${model_file}..."
            wget --tries=3 --timeout=120 -q \
                "${CC_RELEASE}/${model_file}" -O "${target}" \
            && echo "  Downloaded: ${model_file}" \
            || { echo "  WARNING: download failed for ${model_file}"; rm -f "${target}"; MISSING=$((MISSING+1)); }
        fi
    done
    if [ "${MISSING}" -eq 0 ]; then
        echo "  All Cosmic Clarity models present in ${COSMIC_SRC}"
    else
        echo "  WARNING: ${MISSING} model(s) could not be downloaded — check network access."
    fi
else
    echo "  WARNING: Cosmic Clarity source not found at ${COSMIC_SRC}."
    echo "  Models are loaded from that directory at runtime; ensure the image"
    echo "  was built with network access so the Dockerfile download step ran."
fi

# ── GraXpert AI model ─────────────────────────────────────────────────────────
echo ""
echo "--- GraXpert AI model ---"

# GraXpert 3.x stores models under XDG_DATA_HOME/GraXpert/ (capital G+X)
GRAXPERT_DIR="${MODELS_DIR}/GraXpert"
mkdir -p "${GRAXPERT_DIR}"

# GraXpert downloads its AI models on first use.
# Pre-download all four models now so they are available without network
# access at job runtime.  We invoke graxpert with /dev/null as input — it
# fetches the model first, then fails on the invalid file (which is fine).
if command -v graxpert >/dev/null 2>&1; then
    # graxpert stores models under XDG_DATA_HOME; redirect to our volume
    # so they survive container restarts.
    export XDG_DATA_HOME="${MODELS_DIR}"

    # GraXpert needs a valid image file to proceed past file-open.
    # Deconvolution models require at least 512x512 pixels (window_size constraint).
    # Use 512x512 for all model types to be safe.
    DUMMY_FITS="/tmp/graxpert_dummy.fits"
    python3 - <<'PYEOF'
import numpy as np
from astropy.io import fits
hdu = fits.PrimaryHDU(np.random.rand(512, 512).astype(np.float32))
hdu.writeto("/tmp/graxpert_dummy.fits", overwrite=True)
PYEOF

    for GRAXPERT_CMD in background-extraction denoising deconv-obj deconv-stellar; do
        echo "  Pre-downloading GraXpert model: ${GRAXPERT_CMD}..."
        # GraXpert CLI uses single-dash flags; filename is positional.
        # -gpu false: model download doesn't need GPU.
        graxpert \
            -cli \
            -cmd "${GRAXPERT_CMD}" \
            -gpu false \
            -output "${GRAXPERT_DIR}/dummy_${GRAXPERT_CMD}.fits" \
            "${DUMMY_FITS}" \
            2>&1 || true
        rm -f "${GRAXPERT_DIR}/dummy_${GRAXPERT_CMD}.fits"
    done
    rm -f "${DUMMY_FITS}"

    MODEL_COUNT=$(find "${GRAXPERT_DIR}" -type f | wc -l)
    if [ "${MODEL_COUNT}" -gt 0 ]; then
        echo "  GraXpert models ready: ${MODEL_COUNT} file(s) in ${GRAXPERT_DIR}"
    else
        echo "  WARNING: GraXpert model download produced no files (network issue?)."
        echo "  Pipeline will attempt to download on first use."
    fi
else
    echo "  WARNING: graxpert CLI not found — skipping model pre-download."
fi

# ── ASTAP star catalogue ──────────────────────────────────────────────────────
echo ""
echo "--- ASTAP star catalogue ---"
ASTAP_DB_DIR="${ASTAP_STAR_DB_PATH:-/opt/astap/stars}"
mkdir -p "${ASTAP_DB_DIR}"

# D50 catalogue (~900MB) — covers most deep-sky objects to mag 17
# The .pkg.tar.zst is an Arch Linux package; the actual catalogue file
# lives at usr/share/astap/ inside the archive, NOT at the top level.
# We must extract to a temp dir and then find + copy the catalogue files.

# Remove any stale archive that a previous failed run may have left here
rm -f "${ASTAP_DB_DIR}"/*.tar.zst "${ASTAP_DB_DIR}"/*.deb 2>/dev/null || true

ASTAP_DB_FILE="${ASTAP_DB_DIR}/d50.1476"
if [ ! -f "${ASTAP_DB_FILE}" ]; then
    echo "  Downloading ASTAP D50 star catalogue (~900MB)..."

    # Primary: direct CDN URL (no redirect page, more reliable than /download)
    ASTAP_ZST_URL="https://downloads.sourceforge.net/project/astap-program/star_databases/d50_star_database.pkg.tar.zst"
    wget --tries=3 --waitretry=15 --timeout=300 -q --show-progress \
        "${ASTAP_ZST_URL}" -O /tmp/d50.tar.zst \
    || wget --tries=2 --waitretry=15 --timeout=300 -q --show-progress \
        "https://sourceforge.net/projects/astap-program/files/star_databases/d50_star_database.pkg.tar.zst/download" \
        -O /tmp/d50.tar.zst \
    || { echo "  WARNING: ASTAP D50 download failed — plate solving will not work."; rm -f /tmp/d50.tar.zst; }

    if [ -f "/tmp/d50.tar.zst" ]; then
        echo "  Extracting D50 catalogue..."
        mkdir -p /tmp/d50_extract
        if tar -I zstd -xf /tmp/d50.tar.zst -C /tmp/d50_extract 2>&1; then
            # The .pkg.tar.zst lays files out as: usr/share/astap/d50.1476
            # Use find to locate all catalogue files regardless of sub-path.
            FOUND=0
            while IFS= read -r -d '' f; do
                cp "$f" "${ASTAP_DB_DIR}/"
                echo "  Installed: $(basename "$f")"
                FOUND=$((FOUND + 1))
            done < <(find /tmp/d50_extract -type f \( -name "d50*" -o -name "*.1476" -o -name "*.290" -o -name "*.bin" \) -print0)
            if [ "${FOUND}" -gt 0 ]; then
                echo "  ASTAP D50 catalogue installed (${FOUND} file(s))."
            else
                echo "  WARNING: tar extracted but no catalogue files found — archive layout may have changed."
                echo "  Archive contents (first 20 lines):"
                tar -I zstd -tf /tmp/d50.tar.zst 2>/dev/null | head -20 || true
            fi
        else
            echo "  WARNING: ASTAP D50 extraction failed — plate solving will not work."
        fi
        rm -rf /tmp/d50.tar.zst /tmp/d50_extract
    fi
else
    echo "  Already present: ASTAP D50 catalogue"
fi

echo ""
echo "=== Model initialisation complete ==="
echo "Models directory contents:"
ls -lah "${MODELS_DIR}"
