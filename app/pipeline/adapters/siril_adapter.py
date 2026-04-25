"""Siril headless adapter using named pipes for real-time communication.

Manages a Siril process running in pipe mode (``siril-cli -p``) and provides
methods to send commands and receive structured output events including
progress percentages, log messages and step status notifications.

Siril pipe protocol::

    siril_command.in  <- commands written by this adapter
    siril_command.out -> stream parsed as:
        ready
        log: <message>
        status: starting|success|error <command>
        progress: <percent>%

Reference: https://siril.readthedocs.io/en/stable/Headless.html

Example:
    >>> async with SirilAdapter(work_dir=Path("/sessions/abc")) as siril:
    ...     await siril.send_command("cd lights")
    ...     async for event in siril.stream_output():
    ...         print(event)
"""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.errors import ErrorCode, PipelineStepException
from app.core.logging import get_logger

logger = get_logger(__name__)

# Regex patterns for parsing Siril pipe output
_RE_LOG = re.compile(r"^log:\s*(.+)$")
_RE_PROGRESS = re.compile(r"^progress:\s*([\d.]+)%$")
_RE_STATUS = re.compile(r"^status:\s*(\w+)\s*(.*)$")


class SirilEventType(str, Enum):
    """Types of events emitted by the Siril pipe output parser.

    Attributes:
        READY: Siril has started and is ready to accept commands.
        LOG: A log message was emitted.
        PROGRESS: A progress percentage update.
        STATUS: A command status change (starting/success/error).
        UNKNOWN: Unrecognised output line.
    """

    READY = "ready"
    LOG = "log"
    PROGRESS = "progress"
    STATUS = "status"
    UNKNOWN = "unknown"


@dataclass
class SirilEvent:
    """A single parsed event from the Siril pipe output stream.

    Attributes:
        event_type: Type of the event.
        message: Text content (log message or status description).
        percent: Progress percentage (0–100), only for PROGRESS events.
        status_verb: One of ``starting``, ``success``, ``error`` for STATUS events.
        command_name: Command name associated with a STATUS event.
    """

    event_type: SirilEventType
    message: str = ""
    percent: float = 0.0
    status_verb: str = ""
    command_name: str = ""


class SirilAdapter:
    """Manages a Siril headless process and provides async pipe communication.

    The adapter spawns ``siril-cli -p -r <in_pipe> -w <out_pipe>`` and opens
    the named pipes for async I/O. Commands are sent one at a time; the caller
    awaits :meth:`run_command` which resolves when Siril emits a
    ``status: success <cmd>`` or raises on ``status: error <cmd>``.

    Attributes:
        work_dir: Working directory passed to Siril via ``-d``.
        pipe_dir: Directory containing the named pipes (default: ``work_dir``).
        gpu_device: CUDA device hint (currently informational, not passed to Siril).
    """

    def __init__(
        self,
        work_dir: Path,
        pipe_dir: Optional[Path] = None,
        gpu_device: str = "cuda:0",
    ) -> None:
        """Initialise the adapter.

        Args:
            work_dir: Working directory for Siril (``-d`` argument).
            pipe_dir: Directory where named pipes are created.
                Defaults to ``work_dir``.
            gpu_device: CUDA device hint (not forwarded to Siril).
        """
        self.work_dir = work_dir
        self.pipe_dir = pipe_dir or work_dir
        self.gpu_device = gpu_device
        self._process: Optional[asyncio.subprocess.Process] = None
        self._in_pipe: Optional[str] = None
        self._out_pipe: Optional[str] = None
        self._out_file: Optional[asyncio.StreamReader] = None
        self._in_transport: Optional[asyncio.WriteTransport] = None
        self._settings = get_settings()

    async def __aenter__(self) -> "SirilAdapter":
        """Start Siril and open the communication pipes.

        Returns:
            This adapter instance.
        """
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Terminate Siril and close pipes on context exit.

        Args:
            exc_type: Exception type (ignored).
            exc_val: Exception value (ignored).
            exc_tb: Exception traceback (ignored).
        """
        await self.stop()

    async def start(self) -> None:
        """Spawn the Siril process and establish named pipe communication.

        Creates two named pipes in ``pipe_dir``, starts ``siril-cli -p``, and
        waits for the initial ``ready`` message before returning.

        Raises:
            PipelineStepException: If Siril fails to start or does not emit
                ``ready`` within the timeout period.
        """
        pipe_dir = self.pipe_dir
        pipe_dir.mkdir(parents=True, exist_ok=True)

        self._in_pipe = str(pipe_dir / "siril_command.in")
        self._out_pipe = str(pipe_dir / "siril_command.out")

        for pipe_path in (self._in_pipe, self._out_pipe):
            if not os.path.exists(pipe_path):
                os.mkfifo(pipe_path)

        siril_bin = self._settings.siril_binary
        cmd = [
            siril_bin,
            "-d",
            str(self.work_dir),
            "-p",
            "-r",
            self._in_pipe,
            "-w",
            self._out_pipe,
        ]

        logger.info("siril_starting", cmd=" ".join(cmd))
        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise PipelineStepException(
                ErrorCode.PIPE_SIRIL_INIT_FAILED,
                f"Siril binary not found: {siril_bin}",
                step_name="siril_init",
                retryable=False,
            ) from exc

        # Open out pipe for reading (non-blocking)
        self._out_file = await asyncio.wait_for(
            self._open_pipe_reader(self._out_pipe),
            timeout=15.0,
        )
        # Open in pipe for writing — returns the write transport directly
        self._in_transport = await asyncio.wait_for(
            self._open_pipe_writer(self._in_pipe),
            timeout=15.0,
        )

        # Wait for the initial "ready" signal
        await asyncio.wait_for(self._wait_for_ready(), timeout=30.0)
        logger.info("siril_ready")

    async def stop(self) -> None:
        """Send the ``exit`` command and terminate the Siril process.

        Closes named pipes and cleans up the process handle.
        """
        try:
            if self._in_transport is not None:
                try:
                    await self.send_command("exit")
                except Exception:  # noqa: BLE001
                    pass
                self._in_transport.close()
                self._in_transport = None
        finally:
            if self._process is not None:
                try:
                    self._process.terminate()
                    await asyncio.wait_for(self._process.wait(), timeout=10.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    self._process.kill()
                self._process = None
            self._cleanup_pipes()
        logger.info("siril_stopped")

    async def send_command(self, command: str) -> None:
        """Write a single command to the Siril input pipe.

        Args:
            command: A valid Siril command string (without trailing newline).

        Raises:
            RuntimeError: If the adapter has not been started.
        """
        if self._in_transport is None:
            raise RuntimeError("SirilAdapter not started.")
        self._in_transport.write(f"{command}\n".encode())
        logger.debug("siril_command_sent", command=command)

    async def run_command(
        self,
        command: str,
        timeout: float = 300.0,
    ) -> list[SirilEvent]:
        """Send a command and wait for its completion status.

        Collects all events emitted between sending the command and receiving
        ``status: success|error <command>``.

        Args:
            command: Siril command to execute.
            timeout: Maximum seconds to wait for the status response.

        Returns:
            List of all :class:`SirilEvent` objects emitted during execution.

        Raises:
            PipelineStepException: If the command results in an error status.
            asyncio.TimeoutError: If no status is received within ``timeout``.
        """
        await self.send_command(command)
        collected: list[SirilEvent] = []
        cmd_name = command.split()[0].lower()

        async def _collect() -> None:
            async for event in self.stream_output():
                collected.append(event)
                if event.event_type == SirilEventType.STATUS:
                    if event.command_name.lower() == cmd_name or event.status_verb in (
                        "success",
                        "error",
                    ):
                        if event.status_verb == "error":
                            raise PipelineStepException(
                                ErrorCode.PIPE_SIRIL_COMMAND_ERROR,
                                f"Siril command '{command}' failed: {event.message}",
                                step_name=cmd_name,
                                retryable=True,
                                details={"command": command, "siril_message": event.message},
                            )
                        return

        await asyncio.wait_for(_collect(), timeout=timeout)
        return collected

    async def stream_output(self) -> AsyncGenerator[SirilEvent, None]:
        """Yield parsed events from the Siril output pipe until EOF.

        This is a raw streaming generator. Prefer :meth:`run_command` for
        command-response interactions.

        Yields:
            :class:`SirilEvent` parsed from each output line.

        Raises:
            RuntimeError: If the adapter has not been started.
        """
        if self._out_file is None:
            raise RuntimeError("SirilAdapter not started.")

        while True:
            try:
                raw = await self._out_file.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                yield _parse_siril_line(line)
            except (ConnectionResetError, BrokenPipeError):
                break

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _wait_for_ready(self) -> None:
        """Block until Siril emits the initial ``ready`` message.

        Raises:
            PipelineStepException: If EOF is reached without a ``ready`` message.
        """
        async for event in self.stream_output():
            if event.event_type == SirilEventType.READY:
                return
        raise PipelineStepException(
            ErrorCode.PIPE_SIRIL_INIT_FAILED,
            "Siril process closed before emitting 'ready'.",
            step_name="siril_init",
            retryable=False,
        )

    @staticmethod
    async def _open_pipe_reader(path: str) -> asyncio.StreamReader:
        """Open a named pipe for async reading.

        Args:
            path: Absolute path to the named pipe.

        Returns:
            An :class:`asyncio.StreamReader` bound to the pipe.
        """
        loop = asyncio.get_event_loop()
        fd = await loop.run_in_executor(None, lambda: os.open(path, os.O_RDONLY | os.O_NONBLOCK))
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        transport, _ = await loop.connect_read_pipe(lambda: protocol, os.fdopen(fd, "rb", 0))
        return reader

    @staticmethod
    async def _open_pipe_writer(path: str) -> asyncio.WriteTransport:
        """Open a named pipe for async writing.

        Returns the raw write transport — avoids the asyncio.StreamWriter
        dependency on ``_drain_helper`` which is absent on BaseProtocol.

        Args:
            path: Absolute path to the named pipe.

        Returns:
            An :class:`asyncio.WriteTransport` bound to the pipe.
        """
        loop = asyncio.get_event_loop()
        fd = await loop.run_in_executor(None, lambda: os.open(path, os.O_WRONLY))

        class _NullProtocol(asyncio.BaseProtocol):
            def connection_made(self, transport: asyncio.BaseTransport) -> None:  # noqa: D401
                pass

            def connection_lost(self, exc: Optional[Exception]) -> None:  # noqa: D401
                pass

        transport, _ = await loop.connect_write_pipe(_NullProtocol, os.fdopen(fd, "wb", 0))
        return transport  # type: ignore[return-value]

    def _cleanup_pipes(self) -> None:
        """Remove the named pipe files if they exist."""
        for pipe_path in (self._in_pipe, self._out_pipe):
            if pipe_path and os.path.exists(pipe_path):
                try:
                    os.unlink(pipe_path)
                except OSError:
                    pass


def _parse_siril_line(line: str) -> SirilEvent:
    """Parse a single line of Siril pipe output into a :class:`SirilEvent`.

    Args:
        line: Raw text line from the Siril output pipe (stripped).

    Returns:
        The corresponding :class:`SirilEvent` instance.
    """
    if line.strip() == "ready":
        return SirilEvent(event_type=SirilEventType.READY, message="ready")

    m = _RE_LOG.match(line)
    if m:
        return SirilEvent(event_type=SirilEventType.LOG, message=m.group(1))

    m = _RE_PROGRESS.match(line)
    if m:
        return SirilEvent(event_type=SirilEventType.PROGRESS, percent=float(m.group(1)))

    m = _RE_STATUS.match(line)
    if m:
        verb, rest = m.group(1), m.group(2).strip()
        return SirilEvent(
            event_type=SirilEventType.STATUS,
            status_verb=verb,
            command_name=rest,
            message=f"status: {verb} {rest}",
        )

    return SirilEvent(event_type=SirilEventType.UNKNOWN, message=line)
