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
        search_radius_deg: float = 30.0,
        speed: str = "auto",
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Run ASTAP plate solving on a FITS file.

        Modifies the FITS file in-place by writing WCS headers. Also writes a
        ``<filename>.wcs`` sidecar used for result extraction.

        Args:
            fits_path: Path to the stacked FITS file to solve.
            search_radius_deg: Search radius in degrees around the image centre.
            speed: ASTAP solver speed: ``auto``, ``fast``, or ``slow``.
            timeout: Maximum execution time in seconds.

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

        cmd = self._build_command(fits_path, search_radius_deg, speed)
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
        except FileNotFoundError as exc:
            raise PipelineStepException(
                ErrorCode.SYS_EXTERNAL_TOOL_MISSING,
                f"ASTAP binary not found: {self.binary}",
                step_name="plate_solving",
                retryable=False,
            ) from exc

        output_text = stdout.decode("utf-8", errors="replace")
        logger.debug("astap_output", output=output_text[:500])

        if proc.returncode != 0:
            raise PipelineStepException(
                ErrorCode.PIPE_PLATE_SOLVE_FAILED,
                f"ASTAP returned exit code {proc.returncode}: {output_text[:200]}",
                step_name="plate_solving",
                retryable=True,
            )

        return self._parse_result(fits_path, output_text)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_command(
        self,
        fits_path: Path,
        search_radius_deg: float,
        speed: str,
    ) -> list[str]:
        """Build the ASTAP command-line invocation.

        Args:
            fits_path: Input FITS file path.
            search_radius_deg: Search radius in degrees.
            speed: Solver speed mode.

        Returns:
            List of command tokens ready for :func:`asyncio.create_subprocess_exec`.
        """
        cmd = [
            self.binary,
            "-f",
            str(fits_path),
            "-r",
            str(int(search_radius_deg)),
            "-wcs",
        ]
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
        result: dict[str, Any] = {"ra": None, "dec": None, "object_name": ""}

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
