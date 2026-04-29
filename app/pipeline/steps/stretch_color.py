"""Siril stretch and colour calibration pipeline step."""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.domain.profile import ProcessingProfileConfig
from app.domain.ws_event import LogEvent, LogLevel, LogSource
from app.infrastructure.queue.events_bus import EventBus
from app.pipeline.adapters.siril_adapter import SirilAdapter, SirilEventType
from app.pipeline.adapters.siril_script_builder import SirilScriptBuilder
from app.pipeline.base_step import PipelineContext, PipelineStep, StepResult
from app.pipeline.utils.preview import save_step_preview

logger = get_logger(__name__)


def _fits_stats(path: Any) -> dict[str, Any] | None:
    """Return min/median/p997/max of a FITS file for diagnostic logging."""
    try:
        from pathlib import Path  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415
        from astropy.io import fits as _fits  # noqa: PLC0415

        p = Path(path)
        if not p.exists():
            return None
        with _fits.open(str(p)) as hdul:
            for hdu in hdul:
                if hdu.data is None:
                    continue
                arr = np.asarray(hdu.data, dtype=np.float32)
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    return {"shape": list(arr.shape), "empty": True}
                return {
                    "shape": list(arr.shape),
                    "min": float(np.min(finite)),
                    "median": float(np.median(finite)),
                    "p997": float(np.percentile(finite, 99.7)),
                    "max": float(np.max(finite)),
                }
        return None
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


# The override table now lives in ``app/pipeline/utils/object_type.py``
# (``ADAPTIVE_PROFILE_OVERRIDES_BY_TYPE``) and may tune several profile
# fields per object type, not only the stretch strength.


class StretchColorStep(PipelineStep):
    """Applies histogram stretch and photometric colour calibration using Siril.

    Operates on the background-removed image (or stacked image if gradient
    removal was skipped). Outputs a colour-calibrated, stretched FITS file.

    Attributes:
        event_bus: Redis pub/sub bus for forwarding Siril log events.
    """

    name = "stretch_color"
    display_name = "Stretch & Colour Calibration (Siril)"

    def __init__(self, event_bus: EventBus) -> None:
        """Initialise the step.

        Args:
            event_bus: Event bus for real-time log forwarding.
        """
        self.event_bus = event_bus

    async def execute(
        self,
        context: PipelineContext,
        config: dict[str, Any],
    ) -> StepResult:
        """Run Siril stretch and colour calibration.

        Args:
            context: Pipeline context with ``background_removed_path`` or
                ``stacked_fits_path``.
            config: Profile config dict with ``stretch_*`` and
                ``color_calibration_*`` fields.

        Returns:
            StepResult with ``stretched_fits_path`` in metadata.
        """
        input_path = context.background_removed_path or context.stacked_fits_path
        if input_path is None:
            return StepResult(
                success=True, skipped=True, message="No input FITS available for stretch."
            )

        # Copy input to Siril work dir so it knows the file location
        siril_input = context.work_dir / "output" / "for_stretch.fits"
        if input_path != siril_input:
            import shutil  # noqa: PLC0415

            shutil.copy2(str(input_path), str(siril_input))

        profile_config = ProcessingProfileConfig(**config)

        # Adaptive profile overrides (galaxy → softer stretch, etc.) are
        # applied centrally by the orchestrator before any step runs, so
        # ``profile_config`` already reflects them here.  We just emit a
        # diagnostic log for visibility.
        object_type = context.metadata.get("object_type")
        if object_type is not None:
            logger.info(
                "stretch_color_object_type",
                object_type=object_type,
                stretch_method=profile_config.stretch_method,
                stretch_strength=profile_config.stretch_strength,
            )

        builder = SirilScriptBuilder(
            config=profile_config,
            frames={},
            work_dir=context.work_dir / "output",
        )

        # Photometric Colour Calibration (PCC) — runs only if plate-solving
        # produced WCS headers in the FITS. Tolerated as best-effort: a PCC
        # failure (missing catalogue, no internet, low star count) must NOT
        # break the post-processing chain.
        wcs_solved = bool(context.metadata.get("solved", False))
        pcc_ran = False
        if wcs_solved and profile_config.photometric_calibration_enabled:
            # Diagnostic: inspect what's actually in the FITS Siril is about
            # to load — confirms whether the WCS chain (ASTAP → GraXpert
            # output → for_stretch.fits) preserved the plate-solve headers.
            try:
                from astropy.io import fits as _fits  # noqa: PLC0415

                with _fits.open(siril_input) as _hdul:
                    _hdr = _hdul[0].header
                    _wcs_present = sorted(
                        k for k in _hdr
                        if k in {"CRPIX1", "CRVAL1", "CTYPE1", "CD1_1", "PC1_1",
                                 "RADESYS", "PLTSOLVD"}
                    )
                    logger.info(
                        "stretch_color_pcc_input_header",
                        path=str(siril_input),
                        n_hdu=len(_hdul),
                        hdu0_shape=getattr(_hdul[0].data, "shape", None),
                        wcs_keys_present=_wcs_present,
                        ctype1=_hdr.get("CTYPE1"),
                        ctype2=_hdr.get("CTYPE2"),
                        crval1=_hdr.get("CRVAL1"),
                        crval2=_hdr.get("CRVAL2"),
                    )
            except Exception as _exc:  # noqa: BLE001
                logger.warning("stretch_color_pcc_input_header_failed", error=str(_exc))
            try:
                async with SirilAdapter(
                    work_dir=context.work_dir / "output",
                    pipe_dir=context.work_dir / "pipes_pcc",
                ) as siril:
                    for command in builder.build_pcc_commands():
                        if context.cancelled:
                            break
                        await siril.run_command(command, timeout=180.0)
                # PCC saves back to for_stretch.fit; promote to .fits so the
                # stretch script that follows reloads the calibrated image.
                import os  # noqa: PLC0415

                pcc_out = context.work_dir / "output" / "for_stretch.fit"
                if pcc_out.exists():
                    os.replace(str(pcc_out), str(siril_input))
                pcc_ran = True
                logger.info("siril_pcc_done")
            except Exception as exc:  # noqa: BLE001
                # Surface the underlying Siril console output so the actual
                # cause (missing WCS, catalogue download error, low star
                # count, ...) is visible in the worker logs instead of the
                # opaque "status: error pcc" wrapper message.
                siril_log = getattr(exc, "details", {}).get("siril_log", []) if hasattr(exc, "details") else []
                logger.warning(
                    "siril_pcc_failed",
                    error=str(exc),
                    siril_log=siril_log[-20:] if siril_log else [],
                )

        commands = builder.build_postprocessing_commands(pcc_already_ran=pcc_ran)

        if not commands:
            context.stretched_fits_path = input_path
            return StepResult(
                success=True, skipped=True, message="No stretch commands for this profile."
            )

        logger.info(
            "stretch_color_input_stats",
            stats=_fits_stats(siril_input),
            commands=commands,
        )

        async with SirilAdapter(
            work_dir=context.work_dir / "output",
            pipe_dir=context.work_dir / "pipes_stretch",
        ) as siril:
            for command in commands:
                if context.cancelled:
                    break

                events = await siril.run_command(command, timeout=300.0)
                for event in events:
                    if event.event_type == SirilEventType.LOG:
                        ws_log = LogEvent(
                            job_id=context.job_id,
                            session_id=context.session_id,
                            level=LogLevel.INFO,
                            source=LogSource.SIRIL,
                            message=event.message,
                        )
                        await self.event_bus.publish_job_event(context.job_id, ws_log)

        # Siril's `save` command writes <name>.fit (without the trailing 's').
        # Rename the output to .fits so subsequent steps can consume it correctly.
        import os  # noqa: PLC0415

        siril_out_fit = context.work_dir / "output" / "for_stretch.fit"
        if siril_out_fit.exists():
            os.replace(str(siril_out_fit), str(siril_input))

        stretched_path = context.work_dir / "output" / "for_stretch.fits"
        context.stretched_fits_path = stretched_path
        logger.info(
            "stretch_color_done",
            output=str(stretched_path),
            output_stats=_fits_stats(stretched_path),
        )

        # Generate a JPEG preview from the stretched image. Non-critical.
        preview_url: str | None = None
        try:
            preview_path = context.output_dir / "previews" / "stretch_color.jpg"
            await save_step_preview(
                stretched_path,
                preview_path,
                camera_defiltered=bool(config.get("camera_defiltered", True)),
            )
            preview_url = f"/api/v1/sessions/{context.session_id}/step-preview/stretch_color"
        except Exception:  # noqa: BLE001
            logger.warning("stretch_color_preview_failed")

        return StepResult(
            success=True,
            metadata={
                "stretched_fits_path": str(stretched_path),
                **({"preview_url": preview_url} if preview_url else {}),
            },
            message="Stretch and colour calibration complete.",
        )
