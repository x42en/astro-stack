"""AI model weight file management.

Verifies that all required model files are present in the ``/models/`` volume
at application startup. Model files are never downloaded during container
builds; they must be pre-populated via the ``init-models.sh`` script.

Example:
    >>> store = ModelStore(settings)
    >>> await store.verify_all()
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.errors import AstroStackException, ErrorCode
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelDescriptor:
    """Metadata for a single AI model weight file.

    Attributes:
        name: Human-readable name used in log messages.
        filename: File name within the models directory.
        sha256: Expected SHA-256 hex digest (empty string = skip checksum).
        required: If ``True``, absence of this file is a fatal startup error.
    """

    name: str
    filename: str
    sha256: str = ""
    required: bool = True


# ── Known model definitions ───────────────────────────────────────────────────

COSMIC_CLARITY_MODELS: list[ModelDescriptor] = [
    ModelDescriptor(
        name="Cosmic Clarity Denoise (standard)",
        filename="denoise_cnn.pth",
        required=True,
    ),
    ModelDescriptor(
        name="Cosmic Clarity Denoise (deep)",
        filename="deep_denoise_cnn.pth",
        required=True,
    ),
    ModelDescriptor(
        name="Cosmic Clarity Sharpen stellar",
        filename="deep_sharp_stellar_cnn.pth",
        required=True,
    ),
    ModelDescriptor(
        name="Cosmic Clarity Sharpen non-stellar r1",
        filename="deep_nonstellar_sharp_cnn_radius_1.pth",
        required=True,
    ),
    ModelDescriptor(
        name="Cosmic Clarity Sharpen non-stellar r2",
        filename="deep_nonstellar_sharp_cnn_radius_2.pth",
        required=True,
    ),
    ModelDescriptor(
        name="Cosmic Clarity Sharpen non-stellar r4",
        filename="deep_nonstellar_sharp_cnn_radius_4.pth",
        required=True,
    ),
    ModelDescriptor(
        name="Cosmic Clarity Satellite Removal",
        filename="satelliteremoval128featuremaps.pth",
        required=False,
    ),
    ModelDescriptor(
        name="Cosmic Clarity Dark Star",
        filename="darkstar_model.pth",
        required=False,
    ),
]

GRAXPERT_MODELS: list[ModelDescriptor] = [
    ModelDescriptor(
        name="GraXpert AI gradient model",
        filename="graxpert/GraXpert-AI-1.0.0.pth",
        required=True,
    ),
]

ALL_MODELS: list[ModelDescriptor] = COSMIC_CLARITY_MODELS + GRAXPERT_MODELS


class ModelStore:
    """Validates and resolves AI model weight files from the models volume.

    Attributes:
        models_root: Root path of the ``/models/`` volume.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the store with the models directory path.

        Args:
            settings: Optional settings override; defaults to the app singleton.
        """
        cfg = settings or get_settings()
        self.models_root = Path(cfg.models_path)

    def model_path(self, filename: str) -> Path:
        """Return the absolute path for a model file.

        Args:
            filename: Relative filename within the models root.

        Returns:
            Absolute :class:`~pathlib.Path` to the model file.
        """
        return self.models_root / filename

    async def verify_all(self) -> dict[str, bool]:
        """Check that all registered model files are present.

        Missing **required** models raise :class:`~app.core.errors.AstroStackException`.
        Missing *optional* models emit warnings and return ``False`` for that entry.

        Returns:
            Dict mapping model filename → ``True`` (present) / ``False`` (missing).

        Raises:
            AstroStackException: With code :attr:`~app.core.errors.ErrorCode.MODEL_NOT_FOUND`
                if a required model is absent.
        """
        results: dict[str, bool] = {}

        for descriptor in ALL_MODELS:
            path = self.model_path(descriptor.filename)
            present = path.exists() and path.is_file()
            results[descriptor.filename] = present

            if not present:
                if descriptor.required:
                    logger.error(
                        "required_model_missing",
                        model=descriptor.name,
                        path=str(path),
                    )
                    raise AstroStackException(
                        ErrorCode.MODEL_NOT_FOUND,
                        f"Required model '{descriptor.name}' not found at {path}. "
                        "Run init-models.sh to download model files.",
                        status_code=500,
                        details={"model": descriptor.name, "path": str(path)},
                    )
                else:
                    logger.warning(
                        "optional_model_missing",
                        model=descriptor.name,
                        path=str(path),
                    )
            else:
                logger.debug("model_verified", model=descriptor.name)

                if descriptor.sha256:
                    await self._verify_checksum(descriptor)

        return results

    async def _verify_checksum(self, descriptor: ModelDescriptor) -> None:
        """Verify the SHA-256 checksum of a model file.

        Args:
            descriptor: The model descriptor containing the expected hash.

        Raises:
            AstroStackException: With code :attr:`~app.core.errors.ErrorCode.MODEL_CHECKSUM_MISMATCH`
                if the actual hash does not match.
        """
        path = self.model_path(descriptor.filename)
        sha256 = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha256.update(chunk)
        actual = sha256.hexdigest()

        if actual != descriptor.sha256:
            raise AstroStackException(
                ErrorCode.MODEL_CHECKSUM_MISMATCH,
                f"Checksum mismatch for model '{descriptor.name}'.",
                status_code=500,
                details={
                    "model": descriptor.name,
                    "expected": descriptor.sha256,
                    "actual": actual,
                },
            )
        logger.debug("model_checksum_ok", model=descriptor.name)
