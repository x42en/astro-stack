#!/usr/bin/env python3
"""AstroStack worker environment probe.

Validates that every external tool and Python dependency required by the
processing pipeline is correctly installed and functional.  Run this inside
a worker container to diagnose problems without having to submit a real job:

    docker compose exec astro-worker-gpu0 python /app/scripts/probe.py
    docker compose exec astro-worker-gpu1 python /app/scripts/probe.py

Exit code:
    0  -- all CRITICAL checks passed (warnings allowed)
    1  -- one or more CRITICAL checks failed
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_IS_TTY = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text


def _ok(msg: str)   -> str: return _c(f"  [OK]  {msg}", "32")
def _fail(msg: str) -> str: return _c(f"  [!!]  {msg}", "31")
def _warn(msg: str) -> str: return _c(f"  [??]  {msg}", "33")
def _info(msg: str) -> str: return _c(f"  [..]  {msg}", "36")


def _section(title: str) -> str:
    bar = "-" * 60
    return _c(f"\n{bar}\n  {title}\n{bar}", "1;34")


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------


class Level(str, Enum):
    OK       = "ok"
    WARNING  = "warning"
    CRITICAL = "critical"


@dataclass
class ProbeResult:
    name:   str
    level:  Level
    detail: str = ""


@dataclass
class Report:
    results: list[ProbeResult] = field(default_factory=list)

    def add(self, name: str, level: Level, detail: str = "") -> None:
        self.results.append(ProbeResult(name, level, detail))

    def print_summary(self) -> None:
        print(_section("SUMMARY"))
        for r in self.results:
            line = f"{r.name}: {r.detail}" if r.detail else r.name
            if r.level == Level.OK:
                print(_ok(line))
            elif r.level == Level.WARNING:
                print(_warn(line))
            else:
                print(_fail(line))
        n_crit = sum(1 for r in self.results if r.level == Level.CRITICAL)
        n_warn = sum(1 for r in self.results if r.level == Level.WARNING)
        n_ok   = sum(1 for r in self.results if r.level == Level.OK)
        print()
        status = _c("PASS", "32") if n_crit == 0 else _c("FAIL", "31")
        print(f"  {status}  {n_ok} ok / {n_warn} warning / {n_crit} critical\n")

    def has_critical(self) -> bool:
        return any(r.level == Level.CRITICAL for r in self.results)


report = Report()


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _run_cmd(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    """Run a subprocess synchronously; return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return -1, "", f"binary not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -2, "", f"timed out after {timeout}s"
    except PermissionError as exc:
        return -3, "", str(exc)


def _try_import(module: str) -> tuple[bool, str]:
    """Attempt to import a top-level module; return (success, version_or_error)."""
    try:
        mod = __import__(module.split(".")[0])
        version = getattr(mod, "__version__", "?")
        return True, version
    except ImportError as exc:
        return False, str(exc)


def _create_dummy_fits(path: Path, size: int = 512) -> None:
    """Write a *size*×*size* float32 FITS to *path* using astropy.

    Uses random (non-zero) values so tools that normalise by median/MAD do
    not divide by zero on a blank frame.
    """
    import numpy as np
    from astropy.io import fits as pyfits

    data = (np.random.rand(size, size) * 65535).astype(np.float32)
    hdu = pyfits.PrimaryHDU(data)
    hdu.writeto(str(path), overwrite=True)


async def _run_async(
    cmd: list[str],
    timeout: int = 60,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run a subprocess asynchronously; return (returncode, stdout, stderr)."""
    env = {**os.environ, **(extra_env or {})}
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
        return -2, "", f"timed out after {timeout}s"
    except FileNotFoundError:
        return -1, "", f"binary not found: {cmd[0]}"
    except Exception as exc:
        return -99, "", str(exc)


# ---------------------------------------------------------------------------
# Section 1 -- Filesystem paths
# ---------------------------------------------------------------------------


def check_paths() -> None:
    print(_section("1  Filesystem paths"))

    critical_paths = [
        ("/sessions", "session work dir (runtime volume mount)"),
        ("/inbox",    "inbox dir (runtime volume mount)"),
        ("/output",   "output dir (runtime volume mount)"),
        ("/models",   "AI model weights root"),
    ]
    advisory_paths = [
        ("/opt/astap/stars",    "ASTAP star catalogue"),
        ("/opt/cosmic-clarity", "Cosmic Clarity scripts"),
    ]

    for path, label in critical_paths:
        if Path(path).exists():
            print(_ok(f"{path}  ({label})"))
            report.add(f"path:{path}", Level.OK)
        else:
            print(_fail(f"{path}  MISSING  ({label})"))
            report.add(f"path:{path}", Level.CRITICAL, "missing")

    for path, label in advisory_paths:
        p = Path(path)
        if p.exists():
            n = sum(1 for _ in p.iterdir()) if p.is_dir() else 1
            print(_ok(f"{path}  ({label})  [{n} item(s)]"))
            report.add(f"path:{path}", Level.OK)
        else:
            print(_warn(f"{path}  not found  ({label})"))
            report.add(f"path:{path}", Level.WARNING, "missing")

    # ASTAP star catalogue must contain catalogue files to solve anything.
    # Known extensions: .290 (G17/H17/…), .1476 (D50), .bin
    star_db = Path("/opt/astap/stars")
    if star_db.exists():
        cats = (
            list(star_db.glob("*.290"))
            + list(star_db.glob("*.1476"))
            + list(star_db.glob("*.bin"))
        )
        if cats:
            print(_ok(f"  ASTAP star catalogue: {len(cats)} catalogue file(s)"))
            report.add("astap:catalogue", Level.OK, f"{len(cats)} files")
        else:
            # Show what IS in the directory to aid diagnosis
            existing = list(star_db.iterdir())
            extras = ", ".join(f.name for f in existing[:5]) if existing else "(empty)"
            print(_warn(
                f"  ASTAP star catalogue has no .290/.1476/.bin files (found: {extras})\n"
                "       -> plate-solving will fail. Run: scripts/init-models.sh"
            ))
            report.add("astap:catalogue", Level.WARNING, f"no catalogue files -- run init-models.sh")

    # Cosmic Clarity model weights (.pth files in /models)
    models_dir = Path("/models")
    if models_dir.exists():
        pth = list(models_dir.glob("*.pth"))
        if pth:
            print(_ok(f"  Cosmic Clarity model weights: {len(pth)} .pth file(s)"))
            report.add("cosmic:models", Level.OK, f"{len(pth)} files")
        else:
            print(_warn("  No .pth files in /models -- Cosmic Clarity steps will fail"))
            report.add("cosmic:models", Level.WARNING, "empty -- run init-models.sh")


# ---------------------------------------------------------------------------
# Section 2 -- Python imports
# ---------------------------------------------------------------------------


def check_python_imports() -> None:
    print(_section("2  Python package imports"))

    # (module_name, is_critical, friendly_name)
    packages: list[tuple[str, bool, str]] = [
        ("astropy",     True,  "astropy"),
        ("rawpy",       True,  "rawpy (LibRaw bindings)"),
        ("numpy",       True,  "numpy"),
        ("PIL",         True,  "Pillow"),
        ("fastapi",     True,  "fastapi"),
        ("sqlmodel",    True,  "sqlmodel"),
        ("arq",         True,  "arq (task queue)"),
        ("redis",       True,  "redis-py"),
        ("structlog",   True,  "structlog"),
        ("torch",       True,  "PyTorch"),
        ("onnxruntime", True,  "onnxruntime"),
        ("tifffile",    False, "tifffile"),
        ("graxpert",    False, "graxpert"),
    ]

    for module, critical, name in packages:
        success, detail = _try_import(module)
        if success:
            print(_ok(f"{name}  v{detail}"))
            report.add(f"import:{module}", Level.OK, f"v{detail}")
        else:
            lvl = Level.CRITICAL if critical else Level.WARNING
            fn  = _fail if critical else _warn
            print(fn(f"{name}  -- {detail}"))
            report.add(f"import:{module}", lvl, detail[:120])


# ---------------------------------------------------------------------------
# Section 3 -- CUDA / GPU
# ---------------------------------------------------------------------------


def check_cuda() -> None:
    print(_section("3  CUDA / GPU"))

    ok_flag, _ = _try_import("torch")
    if not ok_flag:
        print(_fail("torch not importable -- skipping CUDA checks"))
        report.add("cuda:torch", Level.CRITICAL, "torch missing")
        return

    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory // (1024 ** 3)
            print(_ok(f"cuda:{i}  {props.name}  ({mem_gb} GiB)"))
        report.add("cuda:devices", Level.OK, f"{n} GPU(s)")
    else:
        print(_fail("CUDA not available -- GPU pipeline steps will fail"))
        report.add("cuda:devices", Level.CRITICAL, "not available")

    # onnxruntime CUDA execution provider
    imp_ok, _ = _try_import("onnxruntime")
    if imp_ok:
        import onnxruntime as ort  # noqa: PLC0415

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            print(_ok("onnxruntime CUDAExecutionProvider available"))
            report.add("onnxruntime:cuda", Level.OK)
        else:
            print(_warn(f"onnxruntime CUDAExecutionProvider missing; got: {providers}"))
            report.add("onnxruntime:cuda", Level.WARNING, str(providers)[:100])


# ---------------------------------------------------------------------------
# Section 4 -- Siril
# ---------------------------------------------------------------------------


async def check_siril() -> None:
    print(_section("4  Siril headless (siril-cli)"))

    binary = shutil.which("siril-cli") or "siril-cli"
    rc, stdout, stderr = _run_cmd([binary, "--version"], timeout=10)

    if rc == -1:
        print(_fail("siril-cli not found in PATH"))
        report.add("siril:binary", Level.CRITICAL, "not found")
        return
    if rc == 127:
        print(_fail("siril-cli exits 127 -- missing shared libraries"))
        report.add("siril:binary", Level.CRITICAL, "exit 127 -- missing libs")
        return

    version_line = (stdout + stderr).strip().splitlines()[0] if (stdout or stderr) else "?"
    print(_ok(f"siril-cli: {version_line[:80]}"))
    report.add("siril:binary", Level.OK, version_line[:60])

    # Smoke-test the named-pipe interface ----------------------------------------
    print(_info("Testing named-pipe interface (requires 1.2.0)..."))
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        pipe_in  = td / "siril.in"
        pipe_out = td / "siril.out"
        os.mkfifo(pipe_in)
        os.mkfifo(pipe_out)

        cmd = [binary, "-d", str(td), "-p", "-r", str(pipe_in), "-w", str(pipe_out)]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            loop = asyncio.get_running_loop()

            async def _open_pipes() -> tuple[int, int]:
                out_fd = await loop.run_in_executor(
                    None,
                    lambda: os.open(str(pipe_out), os.O_RDONLY | os.O_NONBLOCK),
                )
                in_fd = await loop.run_in_executor(
                    None,
                    lambda: os.open(str(pipe_in), os.O_WRONLY),
                )
                return out_fd, in_fd

            out_fd, in_fd = await asyncio.wait_for(_open_pipes(), timeout=8.0)

            reader = asyncio.StreamReader()
            proto  = asyncio.StreamReaderProtocol(reader)
            transport, _ = await loop.connect_read_pipe(
                lambda: proto, os.fdopen(out_fd, "rb", 0)
            )

            ready_line = await asyncio.wait_for(reader.readline(), timeout=8.0)
            if b"ready" not in ready_line:
                print(_warn(f"  Expected 'ready', got: {ready_line!r}"))
                report.add("siril:pipe", Level.WARNING, f"unexpected: {ready_line!r}")
            else:
                os.write(in_fd, b"requires 1.2.0\n")
                resp = await asyncio.wait_for(reader.readline(), timeout=8.0)
                if b"error" in resp.lower():
                    print(_fail(f"  requires command failed: {resp!r}"))
                    report.add("siril:pipe", Level.CRITICAL, f"requires failed: {resp!r}")
                else:
                    print(_ok(f"  Pipe interface OK: ready -> cmd -> {resp.strip()!r}"))
                    report.add("siril:pipe", Level.OK)

            transport.close()
            os.close(in_fd)

        except asyncio.TimeoutError:
            print(_warn("  Pipe interface timed out -- siril may be slow to start"))
            report.add("siril:pipe", Level.WARNING, "timeout")
        except Exception as exc:
            print(_warn(f"  Pipe smoke-test skipped: {exc}"))
            report.add("siril:pipe", Level.WARNING, str(exc)[:80])
        finally:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Section 5 -- ASTAP
# ---------------------------------------------------------------------------


def check_astap() -> None:
    print(_section("5  ASTAP plate solver"))

    binary_path = shutil.which("astap")
    binary      = binary_path or "astap"
    rc, stdout, stderr = _run_cmd([binary, "-h"], timeout=10)

    if rc == -1:
        print(_fail("astap not found in PATH"))
        report.add("astap:binary", Level.CRITICAL, "not found")
        return
    if rc == -3:
        print(_fail("astap: Permission denied -- binary is not executable"))
        report.add("astap:binary", Level.CRITICAL, "permission denied")
        return
    if rc == 127:
        # Exit 127 from execve = dynamic linker failed (missing .so files)
        print(_fail("astap exits 127 -- dynamic linker error (missing shared libs)"))
        if binary_path:
            ldd_rc, ldd_out, _ = _run_cmd(["ldd", binary_path], timeout=5)
            missing = [line for line in ldd_out.splitlines() if "not found" in line]
            if missing:
                print(_info("  Missing libraries (ldd):"))
                for m in missing:
                    print(_info(f"    {m.strip()}"))
            else:
                print(_info("  ldd output (no 'not found' lines found):"))
                for line in ldd_out.splitlines()[:10]:
                    print(_info(f"    {line}"))
        report.add("astap:binary", Level.CRITICAL, "exit 127 -- missing shared libs")
        return

    # exit 0 or 1 both indicate a working binary (-h may not be a valid flag)
    version_line = (stdout + stderr).strip().splitlines()[0] if (stdout or stderr) else "?"
    print(_ok(f"astap binary functional: {version_line[:80]}"))
    report.add("astap:binary", Level.OK, version_line[:60])


# ---------------------------------------------------------------------------
# Section 6 -- Cosmic Clarity
# ---------------------------------------------------------------------------


def check_cosmic_clarity() -> None:
    print(_section("6  Cosmic Clarity (SetiAstro)"))

    base = Path("/opt/cosmic-clarity")
    if not base.exists():
        print(_fail(f"{base} directory missing"))
        report.add("cosmic:dir", Level.CRITICAL, "missing")
        return

    report.add("cosmic:dir", Level.OK)
    scripts: dict[str, Path] = {
        "denoise":          base / "setiastrocosmicclarity_denoise.py",
        "sharpen":          base / "SetiAstroCosmicClarity.py",
        "super_resolution": base / "SetiAstroCosmicClarity_SuperRes.py",
        "star_removal":     base / "setiastrocosmicclarity_darkstar.py",
    }

    for name, path in scripts.items():
        if path.exists():
            print(_ok(f"  {name}: {path.name}"))
            report.add(f"cosmic:script:{name}", Level.OK)
        else:
            print(_fail(f"  {name}: {path.name}  NOT FOUND"))
            report.add(f"cosmic:script:{name}", Level.CRITICAL, "missing")

    # Probe the imports inside the denoise script without actually running it.
    # We parse the file with ast, extract only the import statements, and exec
    # them in an isolated namespace so we catch ImportError without side-effects.
    denoise = scripts["denoise"]
    if not denoise.exists():
        return

    print(_info("  Probing denoise script imports..."))
    probe_code = textwrap.dedent(f"""\
        import ast, sys, types
        with open({str(denoise)!r}) as f:
            src = f.read()
        tree = ast.parse(src)
        import_nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mod = types.ModuleType('_probe')
        try:
            exec(
                compile(ast.Module(body=import_nodes, type_ignores=[]), '<probe>', 'exec'),
                mod.__dict__,
            )
            print('IMPORTS_OK')
        except ImportError as e:
            print(f'IMPORT_ERROR:{{e}}')
        except Exception as e:
            print(f'OTHER_ERROR:{{e}}')
    """)
    rc, stdout, stderr = _run_cmd([sys.executable, "-c", probe_code], timeout=20)
    output = (stdout + stderr).strip()

    if "IMPORTS_OK" in output:
        print(_ok("  All denoise script imports satisfied"))
        report.add("cosmic:imports", Level.OK)
    elif "IMPORT_ERROR" in output:
        err = output.replace("IMPORT_ERROR:", "")
        print(_fail(f"  Import error in denoise script: {err}"))
        report.add("cosmic:imports", Level.CRITICAL, err[:120])
    else:
        print(_warn(f"  Import probe inconclusive: {output[:120]}"))
        report.add("cosmic:imports", Level.WARNING, output[:80])


# ---------------------------------------------------------------------------
# Section 7 -- GraXpert
# ---------------------------------------------------------------------------


def check_graxpert() -> None:
    print(_section("7  GraXpert (background gradient removal)"))

    installed, ver = _try_import("graxpert")
    source = Path("/opt/graxpert")

    if installed:
        print(_ok(f"graxpert package installed: v{ver}"))
        report.add("graxpert:install", Level.OK, f"v{ver}")
    elif source.exists():
        main = source / "GraXpert.py"
        if main.exists():
            print(_ok(f"GraXpert source: {source}/GraXpert.py"))
            report.add("graxpert:install", Level.OK, "source install")
        else:
            print(_warn(f"{source} exists but GraXpert.py not found"))
            report.add("graxpert:install", Level.WARNING, "no GraXpert.py")
    else:
        print(_warn("graxpert not installed as package and /opt/graxpert not found"))
        report.add("graxpert:install", Level.WARNING, "not installed")

    # GraXpert 3.x uses XDG_DATA_HOME/GraXpert/ (capital G+X) with
    # per-command subdirs: bge-ai-models/, denoise-ai-models/, etc.
    models_dir = Path("/models/GraXpert")
    if models_dir.exists():
        model_files = list(models_dir.rglob("*.onnx")) + list(models_dir.rglob("*.pth"))
        if model_files:
            print(_ok(f"  GraXpert models: {len(model_files)} file(s) in /models/GraXpert"))
            report.add("graxpert:models", Level.OK, f"{len(model_files)} files")
        else:
            # Subdirs exist but are empty — models will download on first use
            subdirs = [d for d in models_dir.iterdir() if d.is_dir()]
            if subdirs:
                print(_warn(f"  /models/GraXpert has {len(subdirs)} subdir(s) but no .onnx files -- will download on first use"))
                report.add("graxpert:models", Level.WARNING, "dirs present but empty -- downloads on first use")
            else:
                print(_warn("  /models/GraXpert exists but is empty -- AI mode will download on first use"))
                report.add("graxpert:models", Level.WARNING, "empty -- downloads on first use")
    else:
        print(_warn("  /models/GraXpert not found -- GraXpert AI mode will download models on first use"))
        report.add("graxpert:models", Level.WARNING, "missing -- downloads on first use")


# ---------------------------------------------------------------------------
# Section 8 -- Environment variables
# ---------------------------------------------------------------------------


def check_env() -> None:
    print(_section("8  Environment variables"))

    required = [
        "DATABASE_URL",
        "REDIS_URL",
        "SESSIONS_PATH",
        "INBOX_PATH",
        "MODELS_PATH",
    ]
    optional = [
        "SIRIL_BINARY",
        "ASTAP_BINARY",
        "ASTAP_STAR_DB_PATH",
        "COSMIC_CLARITY_SOURCE_PATH",
        "GPU_DEVICES",
        "LOG_LEVEL",
    ]

    for var in required:
        val = os.environ.get(var)
        if val:
            # Redact credentials: for DSN-style URLs show only host:port/db
            display = val.split("@")[-1] if "@" in val else val
            print(_ok(f"{var} = {display}"))
            report.add(f"env:{var}", Level.OK)
        else:
            print(_warn(f"{var} not set (will fall back to default)"))
            report.add(f"env:{var}", Level.WARNING, "not set")

    for var in optional:
        display = os.environ.get(var, "(not set / default)")
        print(_info(f"{var} = {display}"))


# ---------------------------------------------------------------------------
# Section 9 -- Tool smoke tests (live CLI invocations)
# ---------------------------------------------------------------------------


async def check_tool_smoke_tests() -> None:
    """Run one end-to-end invocation of every pipeline CLI tool.

    Uses a 512×512 random float32 FITS so tools that normalise by
    median/MAD do not crash on a blank image.  Results are classified as:
    - OK      : binary ran and produced expected output / benign failure
    - WARNING : ran but with unexpected exit or missing output file
    - CRITICAL: binary not found or hard crash (exit 127)
    """
    print(_section("9  Tool smoke tests  (live CLI invocations)"))
    print(_info("  Creating 512×512 dummy FITS for smoke tests..."))

    tmpdir = Path(tempfile.mkdtemp(prefix="probe_smoke_"))
    dummy = tmpdir / "probe_input.fits"

    try:
        try:
            _create_dummy_fits(dummy)
        except Exception as exc:
            print(_fail(f"  Cannot create dummy FITS: {exc}"))
            report.add("smoke:setup", Level.CRITICAL, str(exc)[:80])
            return

        # ── 1. Siril: load + close via script mode ────────────────────────────
        print(_info("  [Siril] load + close (script mode)..."))
        siril_script = tmpdir / "probe.ssf"
        # Siril load expects filename without extension when working dir is set.
        siril_script.write_text("requires 1.2.0\nload probe_input\nclose\n")
        rc, out, err = await _run_async(
            ["siril-cli", "-d", str(tmpdir), "-s", str(siril_script)],
            timeout=30,
        )
        if rc == -1:
            print(_fail("  Siril smoke: binary not found"))
            report.add("smoke:siril", Level.CRITICAL, "not found")
        elif rc == 0:
            print(_ok("  Siril smoke: load+close OK"))
            report.add("smoke:siril", Level.OK)
        else:
            snippet = (out + err).strip().splitlines()[-1][:80] if (out + err).strip() else ""
            print(_warn(f"  Siril smoke: exit {rc} — {snippet}"))
            report.add("smoke:siril", Level.WARNING, f"exit {rc}")

        # ── 2. ASTAP: plate-solve (no solution expected on dummy) ─────────────
        print(_info("  [ASTAP] plate-solve attempt (no-solution expected on dummy)..."))
        astap_dummy = tmpdir / "astap_probe.fits"
        shutil.copy(dummy, astap_dummy)
        astap_bin = shutil.which("astap") or "astap"
        rc, out, err = await _run_async(
            [astap_bin, "-f", str(astap_dummy), "-r", "30", "-wcs",
             "-d", "/opt/astap/stars"],
            timeout=60,
        )
        if rc == -1:
            print(_fail("  ASTAP smoke: binary not found"))
            report.add("smoke:astap", Level.CRITICAL, "not found")
        elif rc == 127:
            print(_fail("  ASTAP smoke: exit 127 — missing shared libs"))
            report.add("smoke:astap", Level.CRITICAL, "exit 127")
        else:
            # Non-zero is expected ("no solution" on a random image) — still OK
            print(_ok(f"  ASTAP smoke: binary executed cleanly (exit {rc}, no solution expected)"))
            report.add("smoke:astap", Level.OK, f"exit {rc}")

        # ── 3. GraXpert: background-extraction (the only command used in pipeline)
        print(_info("  [GraXpert] background-extraction..."))
        graxpert_out = tmpdir / "probe_graxpert.fits"
        rc, out, err = await _run_async(
            ["graxpert",
             "-cli", "-cmd", "background-extraction",
             "-gpu", "false",
             "-output", str(graxpert_out),
             str(dummy)],
            timeout=90,
            extra_env={"XDG_DATA_HOME": "/models"},
        )
        if rc == -1:
            print(_fail("  GraXpert smoke: binary not found"))
            report.add("smoke:graxpert", Level.CRITICAL, "not found")
        elif rc != 0:
            snippet = (out + err).strip().splitlines()[-1][:80] if (out + err).strip() else ""
            print(_warn(f"  GraXpert smoke: exit {rc} — {snippet}"))
            report.add("smoke:graxpert", Level.WARNING, f"exit {rc}")
        elif not graxpert_out.exists():
            print(_warn("  GraXpert smoke: exit 0 but output file not written"))
            report.add("smoke:graxpert", Level.WARNING, "no output file")
        else:
            print(_ok("  GraXpert smoke: background-extraction OK"))
            report.add("smoke:graxpert", Level.OK)

        # ── 4. Cosmic Clarity: all four pipeline scripts ───────────────────────
        cosmic_src = Path("/opt/cosmic-clarity")
        cc_input_dir  = cosmic_src / "input"
        cc_output_dir = cosmic_src / "output"
        # PyQt6 scripts (sharpen, super_res, darkstar) need an offscreen Qt
        # platform; inherit from the container env or force the value here.
        cc_qt_env = {"QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM", "offscreen")}

        async def _cc_dir_test(
            label: str,
            script: Path,
            cmd: list[str],
            expected_name: str,
            timeout: int,
        ) -> None:
            """Copy dummy into cc_input_dir, run cmd, verify output in cc_output_dir."""
            if not script.exists():
                print(_warn(f"  Cosmic smoke [{label}]: script not found — skipped"))
                report.add(f"smoke:cosmic:{label}", Level.WARNING, "script missing")
                return
            print(_info(f"  [Cosmic Clarity] {label}..."))
            cc_input_dir.mkdir(parents=True, exist_ok=True)
            cc_output_dir.mkdir(parents=True, exist_ok=True)
            for f in cc_input_dir.glob("*"):
                f.unlink(missing_ok=True)
            for f in cc_output_dir.glob("*"):
                f.unlink(missing_ok=True)
            shutil.copy(dummy, cc_input_dir / dummy.name)
            rc, out, err = await _run_async(cmd, timeout=timeout, extra_env=cc_qt_env)
            expected = cc_output_dir / expected_name
            for f in cc_input_dir.glob("*"):
                f.unlink(missing_ok=True)
            if rc == -2:
                print(_warn(f"  Cosmic smoke [{label}]: timed out after {timeout}s"))
                report.add(f"smoke:cosmic:{label}", Level.WARNING, "timeout")
            elif rc == -1:
                print(_fail(f"  Cosmic smoke [{label}]: Python interpreter not found"))
                report.add(f"smoke:cosmic:{label}", Level.CRITICAL, "python not found")
            elif rc != 0:
                snippet = (out + err).strip().splitlines()[-1][:80] if (out + err).strip() else ""
                print(_warn(f"  Cosmic smoke [{label}]: exit {rc} — {snippet}"))
                report.add(f"smoke:cosmic:{label}", Level.WARNING, f"exit {rc}")
            elif not expected.exists():
                print(_warn(f"  Cosmic smoke [{label}]: exit 0 but output not found"))
                report.add(f"smoke:cosmic:{label}", Level.WARNING, "no output file")
            else:
                print(_ok(f"  Cosmic smoke [{label}]: OK"))
                report.add(f"smoke:cosmic:{label}", Level.OK)
            for f in cc_output_dir.glob("*"):
                f.unlink(missing_ok=True)

        # 4a. Denoise — uses tkinter for GUI; CLI is fully headless
        await _cc_dir_test(
            "denoise",
            cosmic_src / "setiastrocosmicclarity_denoise.py",
            [
                sys.executable,
                str(cosmic_src / "setiastrocosmicclarity_denoise.py"),
                "--denoise_strength", "0.5",
                "--denoise_mode", "luminance",
                "--disable_gpu",
            ],
            f"{dummy.stem}_denoised{dummy.suffix}",
            180,
        )

        # 4b. Sharpen — imports PyQt6 unconditionally; QT_QPA_PLATFORM=offscreen required.
        # ALL four of --sharpening_mode, --nonstellar_strength, --stellar_amount and
        # --nonstellar_amount must be supplied, otherwise the script falls back to the
        # PyQt6 QDialog which blocks indefinitely even with QT_QPA_PLATFORM=offscreen.
        await _cc_dir_test(
            "sharpen",
            cosmic_src / "SetiAstroCosmicClarity.py",
            [
                sys.executable,
                str(cosmic_src / "SetiAstroCosmicClarity.py"),
                "--sharpening_mode", "Both",
                "--nonstellar_strength", "3.0",
                "--stellar_amount", "0.5",
                "--nonstellar_amount", "0.5",
                "--disable_gpu",
            ],
            f"{dummy.stem}_sharpened{dummy.suffix}",
            180,
        )

        # 4c. Super-resolution — accepts --input/--output_dir/--scale/--model_dir;
        #     output is always written as {stem}_upscaled{scale}x.fit in output_dir
        sr_script = cosmic_src / "SetiAstroCosmicClarity_SuperRes.py"
        if not sr_script.exists():
            print(_warn("  Cosmic smoke [super_resolution]: script not found — skipped"))
            report.add("smoke:cosmic:super_resolution", Level.WARNING, "script missing")
        else:
            print(_info("  [Cosmic Clarity] super_resolution..."))
            rc, out, err = await _run_async(
                [
                    sys.executable, str(sr_script),
                    "--input",      str(dummy),
                    "--output_dir", str(tmpdir),
                    "--scale",      "2",
                    "--model_dir",  str(cosmic_src),
                ],
                timeout=300,
                extra_env=cc_qt_env,
            )
            sr_expected = tmpdir / f"{dummy.stem}_upscaled2x.fit"
            if rc == -2:
                print(_warn("  Cosmic smoke [super_resolution]: timed out after 300s"))
                report.add("smoke:cosmic:super_resolution", Level.WARNING, "timeout")
            elif rc == -1:
                print(_fail("  Cosmic smoke [super_resolution]: Python interpreter not found"))
                report.add("smoke:cosmic:super_resolution", Level.CRITICAL, "python not found")
            elif rc != 0:
                snippet = (out + err).strip().splitlines()[-1][:80] if (out + err).strip() else ""
                print(_warn(f"  Cosmic smoke [super_resolution]: exit {rc} — {snippet}"))
                report.add("smoke:cosmic:super_resolution", Level.WARNING, f"exit {rc}")
            elif not sr_expected.exists():
                print(_warn("  Cosmic smoke [super_resolution]: exit 0 but output not found"))
                report.add("smoke:cosmic:super_resolution", Level.WARNING, "no output file")
            else:
                print(_ok("  Cosmic smoke [super_resolution]: OK"))
                report.add("smoke:cosmic:super_resolution", Level.OK)

        # 4d. Star removal — imports PyQt6; dir-based I/O like denoise/sharpen
        await _cc_dir_test(
            "star_removal",
            cosmic_src / "setiastrocosmicclarity_darkstar.py",
            [
                sys.executable,
                str(cosmic_src / "setiastrocosmicclarity_darkstar.py"),
                "--disable_gpu",
            ],
            f"{dummy.stem}_starless{dummy.suffix}",
            180,
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> int:
    quick = "--quick" in sys.argv

    print(_c("\n  AstroStack Worker Environment Probe", "1;37"))
    print(_info(f"Python {sys.version.split()[0]}  |  pid {os.getpid()}"))
    if quick:
        print(_info("  --quick: section 9 smoke tests skipped"))

    check_paths()
    check_python_imports()
    check_cuda()
    await check_siril()
    check_astap()
    check_cosmic_clarity()
    check_graxpert()
    check_env()

    if not quick:
        await check_tool_smoke_tests()

    report.print_summary()
    return 1 if report.has_critical() else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
