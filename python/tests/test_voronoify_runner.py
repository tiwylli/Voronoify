from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from python.voronoify_runner import (
    BackendExecutionError,
    BackendSpec,
    JobCancelled,
    JobManager,
    RunConfig,
    RunnerError,
    build_command,
    cancel_job,
    detect_backends,
    run_job,
    validate_config,
)


def make_image(path: Path, size: tuple[int, int] = (24, 16)) -> Path:
    Image.new("RGB", size, (30, 90, 180)).save(path)
    return path


def test_detect_backends_reports_all_methods(tmp_path: Path) -> None:
    python_dir = tmp_path / "python"
    python_dir.mkdir()
    (python_dir / "voronoify_image_fast.py").touch()
    (python_dir / "voronoify_image.py").touch()
    (python_dir / "voronoify_cupy.py").touch()

    backends = detect_backends(tmp_path, probe_cupy=False)

    assert [backend.identifier for backend in backends] == ["fast", "slow", "cupy", "native", "rust"]
    assert backends[0].available
    assert not next(backend for backend in backends if backend.identifier == "native").available
    assert "make -C cuda" in next(
        backend.unavailable_reason for backend in backends if backend.identifier == "native"
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (RunConfig("", "fast", 10, 0.5, 0), "Upload an input image"),
        (RunConfig("placeholder", "fast", 0, 0.5, 0), "Upload an input image"),
    ],
)
def test_validate_config_rejects_missing_image(config: RunConfig, message: str) -> None:
    with pytest.raises(RunnerError, match=message):
        validate_config(config, {"fast": BackendSpec("fast", "Fast", True)})


@pytest.mark.parametrize(
    ("cells", "jitter", "seed", "message"),
    [
        (0, 0.5, 0, "Cells"),
        (100, 1.1, 0, "Jitter"),
        (100, 0.5, -1, "Seed"),
    ],
)
def test_validate_config_rejects_invalid_values(
    tmp_path: Path, cells: int, jitter: float, seed: int, message: str
) -> None:
    image_path = make_image(tmp_path / "input.png")
    with pytest.raises(RunnerError, match=message):
        validate_config(
            RunConfig(image_path, "fast", cells, jitter, seed),
            {"fast": BackendSpec("fast", "Fast", True)},
        )


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (BackendSpec("fast", "Fast", True), ["python.voronoify_image_fast", "--edge-thickness", "--seed"]),
        (BackendSpec("slow", "Slow", True), ["python.voronoify_image", "--edge-thickness", "--seed"]),
        (BackendSpec("cupy", "CuPy", True), ["python.voronoify_cupy", "--seed"]),
        (BackendSpec("native", "Native", True, executable=Path("/native")), ["/native", "1200", "0.5", "42"]),
        (
            BackendSpec("rust", "Rust", True, executable=Path("/rust")),
            ["/rust", "--edge-thickness", "--seed"],
        ),
    ],
)
def test_build_command_passes_common_controls(backend: BackendSpec, expected: list[str]) -> None:
    command = build_command(
        RunConfig("input.png", backend.identifier, 1200, 0.5, 42),
        backend,
        Path("input.png"),
        Path("output.png"),
        python_executable="python",
    )
    for item in expected:
        assert item in command


def test_fast_backend_integration_uses_repository_image(tmp_path: Path) -> None:
    source_path = Path(__file__).resolve().parents[2] / "img" / "wave.jpg"
    input_path = tmp_path / "wave-thumbnail.png"
    with Image.open(source_path) as source:
        source.resize((96, 54)).save(input_path)
    backend = BackendSpec("fast", "Python fast (SciPy)", True)

    result = run_job(
        RunConfig(input_path, "fast", 40, 0.5, 7),
        "integration",
        backends=[backend],
    )

    assert result.image.size == (96, 54)
    assert result.image.mode == "RGB"


def test_native_backend_converts_ppm_output_to_rgb(tmp_path: Path) -> None:
    input_path = make_image(tmp_path / "input.png", (12, 8))
    script = tmp_path / "fake-native.py"
    script.write_text(
        "from PIL import Image\n"
        "import sys\n"
        "with Image.open(sys.argv[1]) as image:\n"
        "    image.save(sys.argv[2], format='PPM')\n",
        encoding="utf-8",
    )
    command_backend = BackendSpec("native", "Native CUDA", True, executable=Path(sys.executable))

    def popen_factory(command, **kwargs):
        rewritten = [command[0], str(script), *command[1:]]
        return subprocess.Popen(rewritten, **kwargs)

    result = run_job(
        RunConfig(input_path, "native", 10, 0.5, 0),
        "native-test",
        backends=[command_backend],
        popen_factory=popen_factory,
    )

    assert result.image.size == (12, 8)
    assert result.image.mode == "RGB"


def test_run_job_removes_temporary_output_directory(tmp_path: Path) -> None:
    input_path = make_image(tmp_path / "input.png")
    backend = BackendSpec("fast", "Python fast (SciPy)", True)
    output_paths: list[Path] = []

    def popen_factory(command, **kwargs):
        output_path = Path(command[command.index("--out") + 1])
        output_paths.append(output_path)
        return subprocess.Popen(command, **kwargs)

    run_job(
        RunConfig(input_path, "fast", 10, 0.5, 0),
        "cleanup-test",
        backends=[backend],
        popen_factory=popen_factory,
    )

    assert len(output_paths) == 1
    assert not output_paths[0].parent.exists()


def test_backend_failure_surfaces_stderr(tmp_path: Path) -> None:
    input_path = make_image(tmp_path / "input.png")
    backend = BackendSpec("native", "Native CUDA", True, executable=Path(sys.executable))

    def popen_factory(_command, **kwargs):
        return subprocess.Popen(
            [sys.executable, "-c", "import sys; print('CUDA unavailable', file=sys.stderr); sys.exit(2)"],
            **kwargs,
        )

    with pytest.raises(BackendExecutionError, match="CUDA unavailable"):
        run_job(
            RunConfig(input_path, "native", 10, 0.5, 0),
            "failure-test",
            backends=[backend],
            popen_factory=popen_factory,
        )


def test_cancel_job_terminates_active_process(tmp_path: Path) -> None:
    input_path = make_image(tmp_path / "input.png")
    backend = BackendSpec("native", "Native CUDA", True, executable=Path(sys.executable))
    manager = JobManager()
    caught: list[Exception] = []

    def popen_factory(_command, **kwargs):
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

    def worker() -> None:
        try:
            run_job(
                RunConfig(input_path, "native", 10, 0.5, 0),
                "cancel-test",
                backends=[backend],
                manager=manager,
                popen_factory=popen_factory,
            )
        except Exception as exc:  # assertion inspects the cross-thread exception
            caught.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    deadline = time.monotonic() + 5
    while not manager._processes and time.monotonic() < deadline:
        time.sleep(0.01)

    assert cancel_job("cancel-test", manager=manager)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(caught) == 1
    assert isinstance(caught[0], JobCancelled)


def test_cancel_job_kills_process_that_ignores_terminate() -> None:
    class StubbornProcess:
        returncode = None

        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.wait_calls = 0

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def wait(self, timeout=None) -> int:
            self.wait_calls += 1
            if timeout is not None:
                raise subprocess.TimeoutExpired("fake", timeout)
            return self.returncode

    manager = JobManager()
    process = StubbornProcess()
    manager.begin("stubborn")
    manager.register("stubborn", process)  # type: ignore[arg-type]

    assert cancel_job("stubborn", manager=manager, timeout=0.01)
    assert process.terminated
    assert process.killed
    assert process.wait_calls == 2
