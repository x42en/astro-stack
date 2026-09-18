"""Siril adapter driving real Python scripts via the ``pyscript`` command.

This is a *second*, independent Siril adapter alongside
:class:`~app.pipeline.adapters.siril_adapter.SirilAdapter` (named-pipe text
protocol). It exists specifically for cases needing **structured** round-trip
data (image statistics, star PSF data, selections) that the text/log pipe
protocol cannot express — the primitives the Phase 2 adaptive critic loop
will need. It is not a drop-in replacement for the pipe adapter's real-time
per-command progress streaming and is not (yet) wired into any pipeline step.

Mechanism (validated hands-on against Siril 1.4.4 / sirilpy 1.0.25, see repo
memory notes for the full investigation):

* ``sirilpy`` is **not on PyPI**. It ships as source inside the ``siril`` apt
  package at ``/usr/share/siril/python_module/`` and must be installed from
  there (``pip install /usr/share/siril/python_module``) into the app venv.
* There is no "long-running Siril instance that external clients reconnect
  to". ``sirilpy.SirilInterface.connect()`` reads a socket path from the
  ``MY_SOCKET`` environment variable, which **Siril itself sets** when it
  spawns a Python script — so the script must be launched *by* Siril.
* The only way to make Siril launch a Python script headlessly is the
  scriptable ``pyscript <file.py>`` command, itself invoked from a plain
  ``.ssf`` script run via ``siril-cli -s wrapper.ssf``. Every call to
  :meth:`SirilPyAdapter.run_script` therefore spawns one ``siril-cli``
  process (~450ms fixed overhead observed), runs to completion, and exits —
  matching the existing per-step process-per-invocation model, just with a
  real Python script instead of static ``.ssf`` text.
* Results are returned by having the script ``print()`` a single marker
  line; Siril forwards script stdout through its own log stream prefixed
  with ``log: ``, so the adapter greps stdout for ``log: <marker> <json>``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings
from app.core.errors import ErrorCode, PipelineStepException
from app.core.logging import get_logger

logger = get_logger(__name__)

_RESULT_MARKER = "SIRILPY_RESULT"

# get_image_stats() returns None until a stats-computing command (``stat``)
# has run on the loaded image at least once — discovered by hitting an
# AttributeError on a fresh load. Bundled here so every helper script gets it
# for free instead of every caller having to remember it.
_STATS_SCRIPT_HEADER = """\
import json
import sirilpy as s

siril = s.SirilInterface()
siril.connect()
"""


class SirilPyAdapter:
    """Runs ad-hoc ``sirilpy`` Python scripts through Siril's ``pyscript`` command.

    Attributes:
        work_dir: Working directory passed to Siril via ``-d`` (also where
            relative FITS paths used inside scripts are resolved from).
        siril_binary: Name or path of the ``siril-cli`` executable.
        min_version: Value passed to the ``requires`` command in the
            generated ``.ssf`` wrapper.
    """

    def __init__(
        self,
        work_dir: Path,
        siril_binary: Optional[str] = None,
        min_version: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.work_dir = work_dir
        self.siril_binary = siril_binary or settings.siril_binary
        self.min_version = min_version or settings.siril_min_version

    # ── Public API ────────────────────────────────────────────────────────────

    async def run_script(
        self,
        python_source: str,
        *,
        timeout: float = 300.0,
        step_name: str = "sirilpy_script",
    ) -> dict[str, Any]:
        """Run a Python script through Siril and return its structured result.

        The script is expected to print exactly one line of the form
        ``f"{_RESULT_MARKER} {json.dumps(payload)}"``; that payload is
        returned. Scripts that raise will surface as a non-zero Siril exit
        code (Siril prints the Python traceback to its log and exits with a
        script-failure status), which this method turns into a
        :class:`PipelineStepException`.

        Args:
            python_source: Full contents of the ``.py`` file to run.
            timeout: Maximum execution time in seconds.
            step_name: Used in error messages/details only.

        Returns:
            The JSON-decoded payload printed by the script.

        Raises:
            PipelineStepException: On missing binary, timeout, non-zero exit,
                or a missing/malformed result marker.
        """
        run_dir = self.work_dir / "_sirilpy" / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)
        script_path = run_dir / "script.py"
        wrapper_path = run_dir / "wrapper.ssf"
        script_path.write_text(python_source)
        wrapper_path.write_text(
            f"requires {self.min_version}\npyscript {script_path}\n"
        )

        cmd = [self.siril_binary, "-d", str(self.work_dir), "-s", str(wrapper_path)]
        logger.debug("sirilpy_running", step=step_name, cmd=" ".join(cmd[:3]))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except FileNotFoundError as exc:
            raise PipelineStepException(
                ErrorCode.SYS_EXTERNAL_TOOL_MISSING,
                f"siril-cli binary not found: {self.siril_binary}",
                step_name=step_name,
                retryable=False,
            ) from exc
        except asyncio.TimeoutError as exc:
            raise PipelineStepException(
                ErrorCode.PIPE_SIRILPY_SCRIPT_FAILED,
                f"Siril pyscript '{step_name}' timed out after {timeout}s.",
                step_name=step_name,
                retryable=True,
            ) from exc
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

        stdout_text = stdout.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")[:800]
            raise PipelineStepException(
                ErrorCode.PIPE_SIRILPY_SCRIPT_FAILED,
                f"Siril pyscript '{step_name}' failed (exit {proc.returncode}): {stderr_text}",
                step_name=step_name,
                retryable=True,
                details={"returncode": proc.returncode, "stdout_tail": stdout_text[-800:]},
            )

        for line in stdout_text.splitlines():
            line = line.strip()
            if line.startswith(f"log: {_RESULT_MARKER} "):
                payload = line[len(f"log: {_RESULT_MARKER} "):]
                try:
                    return json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise PipelineStepException(
                        ErrorCode.PIPE_SIRILPY_SCRIPT_FAILED,
                        f"Siril pyscript '{step_name}' produced an unparsable result: {exc}",
                        step_name=step_name,
                        retryable=False,
                        details={"payload": payload},
                    ) from exc

        raise PipelineStepException(
            ErrorCode.PIPE_SIRILPY_SCRIPT_FAILED,
            f"Siril pyscript '{step_name}' produced no result marker.",
            step_name=step_name,
            retryable=False,
            details={"stdout_tail": stdout_text[-800:]},
        )

    async def get_image_stats(
        self,
        fits_path: Path,
        channel: int = 0,
        timeout: float = 120.0,
    ) -> dict[str, float]:
        """Load a FITS file in Siril and return its channel statistics.

        Args:
            fits_path: Path to the FITS file (loaded via Siril's ``load``
                command, so relative to ``work_dir`` or absolute).
            channel: 0=red/mono, 1=green, 2=blue.
            timeout: Maximum execution time in seconds.

        Returns:
            Dict with ``mean``, ``median``, ``sigma``, ``bgnoise``, ``min``, ``max``.
        """
        script = _STATS_SCRIPT_HEADER + f"""\
siril.cmd("load", {str(fits_path)!r})
siril.cmd("stat")
stats = siril.get_image_stats({channel})
print("{_RESULT_MARKER} " + json.dumps({{
    "mean": stats.mean, "median": stats.median, "sigma": stats.sigma,
    "bgnoise": stats.bgnoise, "min": stats.min, "max": stats.max,
}}))
"""
        return await self.run_script(script, timeout=timeout, step_name="get_image_stats")

    async def get_image_stars(
        self,
        fits_path: Path,
        channel: Optional[int] = None,
        timeout: float = 180.0,
    ) -> list[dict[str, float]]:
        """Load a FITS file, detect stars, and return their PSF measurements.

        Args:
            fits_path: Path to the FITS file.
            channel: 0=red/mono, 1=green, 2=blue; omitted defaults to green
                for colour images (Siril's own default).
            timeout: Maximum execution time in seconds.

        Returns:
            List of dicts with ``fwhm``, ``snr``, ``roundness``, ``mag`` per detected star.
        """
        channel_arg = "None" if channel is None else str(channel)
        script = _STATS_SCRIPT_HEADER + f"""\
siril.cmd("load", {str(fits_path)!r})
stars = siril.get_image_stars(channel={channel_arg})
print("{_RESULT_MARKER} " + json.dumps([
    {{"fwhm": st.fwhmx, "snr": st.SNR, "roundness": st.sy / st.sx if st.sx else 0.0, "mag": st.mag}}
    for st in (stars or [])
]))
"""
        result = await self.run_script(script, timeout=timeout, step_name="get_image_stars")
        return result if isinstance(result, list) else []
