"""ASTAP plate-solving adapter.

Wraps the ASTAP command-line tool to perform astrometric plate solving on a
stacked FITS image. On success, ASTAP writes WCS header entries back into the
FITS file and produces a ``.wcs`` sidecar file.

ASTAP CLI reference: https://www.hnsky.org/astap.htm

Example:
    >>> adapter = AstapAdapter()
    >>> result = await adapter.solve(
    ...     fits_path=Path("/sessions/abc/stack_result.fits"),
    ...     search_radius_deg=30.0,
    ... )
    >>> result["object_name"]
    'M42 - Orion Nebula'
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings
from app.core.errors import ErrorCode, PipelineStepException
from app.core.logging import get_logger

logger = get_logger(__name__)

_RE_RA = re.compile(r"CRVAL1\s*=\s*([\d.+-]+)")
_RE_DEC = re.compile(r"CRVAL2\s*(=|:)\s*([\d.+-]+)")


class AstapAdapter:
    """Adapter for the ASTAP astrometric plate solver.

    Runs ASTAP as an async subprocess and parses the WCS output to extract
    sky coordinates and attempt to identify the target object.

    Attributes:
        binary: Path to the ``astap`` executable.
        star_db_path: Path to the ASTAP star catalogue directory.
    """

    def __init__(
        self,
        binary: Optional[str] = None,
        star_db_path: Optional[str] = None,
    ) -> None:
        """Initialise the adapter.

        Args:
            binary: Optional path to the ASTAP binary; defaults to settings.
            star_db_path: Optional star catalogue path; defaults to settings.
        """
        settings = get_settings()
        self.binary = binary or settings.astap_binary
        self.star_db_path = star_db_path or settings.astap_star_db_path

    async def solve(
        self,
        fits_path: Path,
        search_radius_deg: float = 180.0,
        speed: str = "auto",
        timeout: float = 120.0,
        target_ra_deg: Optional[float] = None,
        target_dec_deg: Optional[float] = None,
        hint_radius_deg: float = 5.0,
    ) -> dict[str, Any]:
        """Run ASTAP plate solving on a FITS file.

        Modifies the FITS file in-place by writing WCS headers. Also writes a
        ``<filename>.wcs`` sidecar used for result extraction.

        Args:
            fits_path: Path to the stacked FITS file to solve.
            search_radius_deg: Search radius in degrees around the image centre,
                used as the **fallback** blind-solve radius when no target
                hint is supplied. ASTAP starts from RA=0/Dec=0, so this needs
                to be 180° to cover the whole sky.
            speed: ASTAP solver speed: ``auto``, ``fast``, or ``slow``.
            timeout: Maximum execution time in seconds.
            target_ra_deg: Optional user-supplied right ascension (J2000
                decimal degrees) used as the search centre. When provided the
                solver runs in *targeted* mode with ``hint_radius_deg``
                instead of a 180° blind solve, which is far faster and far
                more reliable on noisy DSLR stacks.
            target_dec_deg: Optional user-supplied declination (J2000 decimal
                degrees), companion of ``target_ra_deg``.
            hint_radius_deg: Search radius applied around the user hint.
                5° is generous enough to absorb GoTo / framing errors while
                still being orders of magnitude faster than a blind solve.

        Returns:
            Dict with ``ra`` (float, degrees), ``dec`` (float, degrees), and
            ``object_name`` (str, best-effort Simbad lookup — may be empty).

        Raises:
            PipelineStepException: If ASTAP fails or times out.
        """
        if not fits_path.exists():
            raise PipelineStepException(
                ErrorCode.PIPE_PLATE_SOLVE_FAILED,
                f"Input FITS file not found: {fits_path}",
                step_name="plate_solving",
                retryable=False,
            )

        cmd = self._build_command(
            fits_path,
            search_radius_deg,
            speed,
            target_ra_deg=target_ra_deg,
            target_dec_deg=target_dec_deg,
            hint_radius_deg=hint_radius_deg,
        )
        logger.info("astap_solving", fits=str(fits_path), cmd=" ".join(cmd))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise PipelineStepException(
                ErrorCode.PIPE_PLATE_SOLVE_FAILED,
                f"ASTAP timed out after {timeout}s.",
                step_name="plate_solving",
                retryable=True,
            ) from exc
        except (FileNotFoundError, PermissionError) as exc:
            raise PipelineStepException(
                ErrorCode.SYS_EXTERNAL_TOOL_MISSING,
                f"ASTAP binary not executable: {self.binary} — {exc}",
                step_name="plate_solving",
                retryable=False,
            ) from exc

        output_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        logger.debug("astap_output", stdout=output_text[:500], stderr=stderr_text[:500])

        if proc.returncode != 0:
            # Exit code 1 = no solution found — ASTAP ran correctly but could not
            # match stars (catalog missing, FOV unknown, insufficient stars, etc.).
            # This is NOT a pipeline-breaking error; the job continues without WCS.
            if proc.returncode == 1:
                diag = (stderr_text or output_text)[:300].strip()
                logger.warning(
                    "astap_no_solution",
                    fits=str(fits_path),
                    detail=diag or "(no output)",
                )
                return {"ra": None, "dec": None, "object_name": "", "solved": False}
            # Any other non-zero code is a genuine binary/system error.
            diag = (stderr_text or output_text)[:200]
            raise PipelineStepException(
                ErrorCode.PIPE_PLATE_SOLVE_FAILED,
                f"ASTAP exited with code {proc.returncode}: {diag}",
                step_name="plate_solving",
                retryable=False,
            )

        result = self._parse_result(fits_path, output_text)
        result["solved"] = True
        # Promote the .wcs sidecar into the FITS header so downstream tools
        # that don't read sidecars (GraXpert, then Siril ``pcc``) still see a
        # plate-solved image.
        try:
            self._inject_wcs_into_fits(fits_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("astap_wcs_inject_failed", error=str(exc))
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_command(
        self,
        fits_path: Path,
        search_radius_deg: float,
        speed: str,
        *,
        target_ra_deg: Optional[float] = None,
        target_dec_deg: Optional[float] = None,
        hint_radius_deg: float = 5.0,
    ) -> list[str]:
        """Build the ASTAP command-line invocation.

        When ``target_ra_deg`` and ``target_dec_deg`` are both provided, ASTAP
        is invoked in *targeted* mode using its ``-ra`` (RA in **hours**) and
        ``-spd`` (south polar distance, i.e. ``Dec + 90`` in degrees) options
        plus a small search radius. Otherwise it falls back to the blind-solve
        radius supplied as ``search_radius_deg``.

        Args:
            fits_path: Input FITS file path.
            search_radius_deg: Blind-solve search radius in degrees.
            speed: Solver speed mode.
            target_ra_deg: Optional user RA hint (J2000 decimal degrees).
            target_dec_deg: Optional user Dec hint (J2000 decimal degrees).
            hint_radius_deg: Radius used when a target hint is supplied.

        Returns:
            List of command tokens ready for :func:`asyncio.create_subprocess_exec`.
        """
        cmd = [
            self.binary,
            "-f",
            str(fits_path),
            "-wcs",
        ]
        if target_ra_deg is not None and target_dec_deg is not None:
            # ASTAP CLI: -ra in HOURS (0..24), -spd = Dec + 90 in DEGREES (0..180)
            ra_hours = float(target_ra_deg) / 15.0
            spd_deg = float(target_dec_deg) + 90.0
            cmd += [
                "-ra", f"{ra_hours:.6f}",
                "-spd", f"{spd_deg:.6f}",
                "-r", str(int(round(hint_radius_deg))),
            ]
        else:
            cmd += ["-r", str(int(search_radius_deg))]
        if speed != "auto":
            cmd += ["-speed", speed]
        if self.star_db_path:
            cmd += ["-d", self.star_db_path]
        return cmd

    def _parse_result(
        self,
        fits_path: Path,
        output_text: str,
    ) -> dict[str, Any]:
        """Extract RA/Dec coordinates from ASTAP's output.

        Attempts to read the ``.wcs`` sidecar file, then falls back to
        parsing stdout for embedded CRVAL1/CRVAL2 values.

        Args:
            fits_path: Original input FITS path (used to locate sidecar).
            output_text: ASTAP stdout text.

        Returns:
            Dict with ``ra``, ``dec``, and ``object_name`` keys.
        """
        result: dict[str, Any] = {"ra": None, "dec": None, "object_name": "", "solved": True}

        # Try .wcs sidecar first
        wcs_path = fits_path.with_suffix(".wcs")
        if wcs_path.exists():
            wcs_text = wcs_path.read_text(errors="replace")
            ra_m = _RE_RA.search(wcs_text)
            dec_m = _RE_DEC.search(wcs_text)
            if ra_m:
                result["ra"] = float(ra_m.group(1))
            if dec_m:
                result["dec"] = float(dec_m.group(2))
        else:
            # Fallback: parse stdout
            ra_m = _RE_RA.search(output_text)
            dec_m = _RE_DEC.search(output_text)
            if ra_m:
                result["ra"] = float(ra_m.group(1))
            if dec_m:
                result["dec"] = float(dec_m.group(2))

        logger.info(
            "plate_solve_result",
            ra=result["ra"],
            dec=result["dec"],
        )
        return result

    @staticmethod
    def _inject_wcs_into_fits(fits_path: Path) -> None:
        """Merge the ``.wcs`` sidecar produced by ASTAP into the FITS header.

        ASTAP's ``-wcs`` flag writes a sidecar file in FITS-card text format
        (one ``KEYWORD = VALUE / comment`` per line) instead of updating the
        FITS header in place. Tools that only read the FITS header
        (GraXpert, Siril's headless ``pcc`` command) therefore see a
        non-platesolved image.

        This helper reads the sidecar (if present) and copies its
        WCS-related numeric keywords into the FITS primary header. We
        deliberately whitelist keys and parse line-by-line so malformed
        ``CONTINUE`` / long-string cards (which astropy refuses in strict
        mode) don't abort the merge.
        """
        from astropy.io import fits  # noqa: PLC0415

        wcs_path = fits_path.with_suffix(".wcs")
        if not wcs_path.exists():
            return

        # Whitelist of header keys we actually need for plate-solving
        # consumers (Siril ``pcc``). Anything outside this list is ignored
        # to keep the merge safe.
        wcs_prefixes = ("CRPIX", "CRVAL", "CTYPE", "CUNIT", "CDELT",
                        "CROTA", "CD", "PC", "A_", "B_", "AP_", "BP_")
        wcs_keys = {
            "WCSAXES", "RADESYS", "EQUINOX", "LONPOLE", "LATPOLE",
            "PLTSOLVD", "OBJCTRA", "OBJCTDEC", "RA", "DEC",
        }

        merged: list[tuple[str, Any, str]] = []
        for raw in wcs_path.read_text(errors="replace").splitlines():
            line = raw.rstrip()
            if not line or line.startswith(("END", "COMMENT", "HISTORY", "CONTINUE")):
                continue
            # FITS card layout: 8-char keyword, "= ", value [/ comment]
            if "=" not in line:
                continue
            key = line[:8].strip().upper()
            if not key:
                continue
            if key not in wcs_keys and not key.startswith(wcs_prefixes):
                continue
            try:
                card = fits.Card.fromstring(line.ljust(80))
                if card.keyword:
                    merged.append((card.keyword, card.value, card.comment))
            except Exception:  # noqa: BLE001
                continue

        if not merged:
            logger.warning(
                "astap_wcs_inject_empty",
                sidecar=str(wcs_path),
                first_lines=wcs_path.read_text(errors="replace").splitlines()[:6],
            )
            return

        with fits.open(str(fits_path), mode="update") as hdul:
            tgt_hdr = hdul[0].header
            for key, value, comment in merged:
                tgt_hdr[key] = (value, comment)
            hdul.flush()
        logger.info(
            "astap_wcs_injected",
            count=len(merged),
            keys=sorted({k for k, _, _ in merged})[:20],
        )
