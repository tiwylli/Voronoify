"""Backend discovery and subprocess orchestration for the Voronoify UI."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BackendSpec:
    identifier: str
    label: str
    available: bool
    unavailable_reason: str = ""
    executable: Path | None = None


@dataclass(frozen=True)
class RunConfig:
    image_path: Path | str
    backend: str
    cells: int
    jitter: float
    seed: int


@dataclass(frozen=True)
class RunResult:
    image: Image.Image
    backend: BackendSpec
    elapsed_seconds: float
    output: str


class RunnerError(RuntimeError):
    """Base exception for user-facing runner errors."""


class JobCancelled(RunnerError):
    """Raised when a running backend process is cancelled."""


class BackendExecutionError(RunnerError):
    """Raised when a backend subprocess exits unsuccessfully."""


class JobManager:
    """Thread-safe registry of active subprocesses keyed by browser session."""

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()

    def begin(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._processes:
                raise RunnerError("A job is already running for this browser session.")
            self._cancelled.discard(session_id)

    def register(self, session_id: str, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._processes[session_id] = process
            cancelled = session_id in self._cancelled
        if cancelled:
            process.terminate()

    def was_cancelled(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._cancelled

    def finish(self, session_id: str, process: subprocess.Popen[str] | None = None) -> None:
        with self._lock:
            current = self._processes.get(session_id)
            if process is None or current is process:
                self._processes.pop(session_id, None)
            self._cancelled.discard(session_id)

    def cancel(self, session_id: str, timeout: float = 2.0) -> bool:
        with self._lock:
            self._cancelled.add(session_id)
            process = self._processes.get(session_id)
        if process is None or process.poll() is not None:
            return False

        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return True


JOB_MANAGER = JobManager()


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _probe_cupy(python_executable: str, timeout: float = 10.0) -> tuple[bool, str]:
    probe = (
        "import cupy as cp; "
        "count = cp.cuda.runtime.getDeviceCount(); "
        "assert count > 0, 'no CUDA devices detected'"
    )
    try:
        completed = subprocess.run(
            [python_executable, "-c", probe],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"CuPy probe failed: {exc}"

    if completed.returncode == 0:
        return True, ""
    detail = _process_detail(completed.stderr, completed.stdout)
    return False, f"CuPy cannot access a CUDA device: {detail}"


def _first_executable(paths: Sequence[Path]) -> Path | None:
    return next((path for path in paths if path.is_file() and os.access(path, os.X_OK)), None)


def detect_backends(
    repo_root: Path = REPO_ROOT,
    python_executable: str = sys.executable,
    *,
    probe_cupy: bool = True,
) -> list[BackendSpec]:
    """Return every supported backend in display order with availability details."""
    python_dir = repo_root / "python"
    rust_dir = repo_root / "rust"
    required_modules = [name for name in ("PIL", "numpy", "scipy") if not _module_exists(name)]
    python_reason = f"Missing Python dependencies: {', '.join(required_modules)}" if required_modules else ""

    fast_exists = (python_dir / "voronoify_image_fast.py").is_file()
    slow_exists = (python_dir / "voronoify_image.py").is_file()
    backends = [
        BackendSpec(
            "fast",
            "Python fast (SciPy)",
            fast_exists and not python_reason,
            python_reason or ("Backend module is missing." if not fast_exists else ""),
        ),
        BackendSpec(
            "slow",
            "Python slow (reference)",
            slow_exists and not python_reason,
            python_reason or ("Backend module is missing." if not slow_exists else ""),
        ),
    ]

    cupy_module_exists = (python_dir / "voronoify_cupy.py").is_file()
    if not cupy_module_exists:
        cupy_available, cupy_reason = False, "Backend module is missing."
    elif not _module_exists("cupy"):
        cupy_available, cupy_reason = False, "CuPy is not installed."
    elif probe_cupy:
        cupy_available, cupy_reason = _probe_cupy(python_executable)
    else:
        cupy_available, cupy_reason = True, ""
    backends.append(BackendSpec("cupy", "CuPy (CUDA)", cupy_available, cupy_reason))

    native = _first_executable([repo_root / "bin" / "voronoify_native"])
    backends.append(
        BackendSpec(
            "native",
            "Native CUDA",
            native is not None,
            "Build bin/voronoify_native with `make -C cuda`." if native is None else "",
            native,
        )
    )

    rust = _first_executable(
        [
            rust_dir / "target" / "release" / "voronoify_parallel",
            rust_dir / "target" / "debug" / "voronoify_parallel",
            rust_dir / "target" / "release" / "voronoify-rs",
            rust_dir / "target" / "debug" / "voronoify-rs",
        ]
    )
    backends.append(
        BackendSpec(
            "rust",
            "Rust",
            rust is not None,
            "Build the Rust binaries with `cargo build --release --manifest-path rust/Cargo.toml`."
            if rust is None
            else "",
            rust,
        )
    )
    return backends


def validate_config(config: RunConfig, backends: Mapping[str, BackendSpec]) -> RunConfig:
    image_path = Path(config.image_path) if config.image_path else None
    if image_path is None or not image_path.is_file():
        raise RunnerError("Upload an input image before generating.")
    if config.backend not in backends:
        raise RunnerError("Select a recognized backend.")
    if not backends[config.backend].available:
        reason = backends[config.backend].unavailable_reason
        raise RunnerError(f"{backends[config.backend].label} is unavailable: {reason}")
    if isinstance(config.cells, bool) or not isinstance(config.cells, int) or config.cells <= 0:
        raise RunnerError("Cells must be a positive integer.")
    if not 0.0 <= config.jitter <= 1.0:
        raise RunnerError("Jitter must be between 0 and 1.")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int) or config.seed < 0:
        raise RunnerError("Seed must be a non-negative integer.")

    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as exc:
        raise RunnerError(f"The uploaded file is not a readable image: {exc}") from exc
    return config


def build_command(
    config: RunConfig,
    backend: BackendSpec,
    input_path: Path,
    output_path: Path,
    python_executable: str = sys.executable,
) -> list[str]:
    common = ["--out", str(output_path), "--cells", str(config.cells), "--jitter", str(config.jitter)]
    if backend.identifier == "fast":
        return [
            python_executable,
            "-m",
            "python.voronoify_image_fast",
            str(input_path),
            *common,
            "--edge-thickness",
            "0",
            "--seed",
            str(config.seed),
        ]
    if backend.identifier == "slow":
        return [
            python_executable,
            "-m",
            "python.voronoify_image",
            str(input_path),
            *common,
            "--edge-thickness",
            "0",
            "--seed",
            str(config.seed),
        ]
    if backend.identifier == "cupy":
        return [
            python_executable,
            "-m",
            "python.voronoify_cupy",
            str(input_path),
            *common,
            "--seed",
            str(config.seed),
        ]
    if backend.identifier == "native" and backend.executable:
        return [
            str(backend.executable),
            str(input_path),
            str(output_path),
            str(config.cells),
            str(config.jitter),
            str(config.seed),
        ]
    if backend.identifier == "rust" and backend.executable:
        return [
            str(backend.executable),
            str(input_path),
            *common,
            "--edge-thickness",
            "0",
            "--seed",
            str(config.seed),
        ]
    raise RunnerError(f"Backend {backend.label} has no runnable command.")


def _process_detail(stderr: str, stdout: str, limit: int = 2000) -> str:
    detail = (stderr or stdout).strip()
    if not detail:
        return "the process did not provide an error message"
    return detail[-limit:]


def _load_result(output_path: Path, png_path: Path | None = None) -> Image.Image:
    try:
        with Image.open(output_path) as image:
            result = image.convert("RGB")
            if png_path is not None:
                result.save(png_path, format="PNG")
            result.load()
            return result.copy()
    except Exception as exc:
        raise BackendExecutionError(f"The backend did not produce a readable image: {exc}") from exc


PopenFactory = Callable[..., subprocess.Popen[str]]


def run_job(
    config: RunConfig,
    session_id: str,
    *,
    backends: Sequence[BackendSpec] | None = None,
    manager: JobManager = JOB_MANAGER,
    popen_factory: PopenFactory = subprocess.Popen,
    python_executable: str = sys.executable,
) -> RunResult:
    """Run one backend process and return a detached in-memory result image."""
    available_backends = (
        detect_backends(python_executable=python_executable) if backends is None else backends
    )
    backend_map = {backend.identifier: backend for backend in available_backends}
    validate_config(config, backend_map)
    backend = backend_map[config.backend]
    source_path = Path(config.image_path)
    process: subprocess.Popen[str] | None = None
    manager.begin(session_id)
    started = time.monotonic()

    try:
        with tempfile.TemporaryDirectory(prefix="voronoify-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            if backend.identifier == "native":
                input_path = temp_dir / "input.ppm"
                output_path = temp_dir / "output.ppm"
                with Image.open(source_path) as source:
                    source.convert("RGB").save(input_path, format="PPM")
            else:
                input_path = source_path
                output_path = temp_dir / "output.png"

            command = build_command(config, backend, input_path, output_path, python_executable)
            process = popen_factory(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            manager.register(session_id, process)
            stdout, stderr = process.communicate()
            cancelled = manager.was_cancelled(session_id)
            if cancelled:
                raise JobCancelled("Generation cancelled.")
            if process.returncode != 0:
                detail = _process_detail(stderr, stdout)
                raise BackendExecutionError(f"{backend.label} failed: {detail}")
            if not output_path.is_file():
                raise BackendExecutionError(f"{backend.label} completed without creating an output image.")

            png_path = temp_dir / "output.png" if backend.identifier == "native" else None
            image = _load_result(output_path, png_path)
            return RunResult(image, backend, time.monotonic() - started, stdout.strip())
    finally:
        manager.finish(session_id, process)


def cancel_job(session_id: str, *, manager: JobManager = JOB_MANAGER, timeout: float = 2.0) -> bool:
    """Cancel the active process for a browser session, if one exists."""
    return manager.cancel(session_id, timeout=timeout)
