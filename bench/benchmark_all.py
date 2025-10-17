#!/usr/bin/env python3
"""Run all available Voronoify implementations and record timings to CSV.

Runs:
- Python fast (`voronoify_image_fast`)
- CuPy (`voronoify_cupy`) if importable
- Native CUDA (`./voronoify_native`) if present
- Rust binaries in `rust` (cargo build or existing binaries)

Writes `bench/results.csv` with columns: implementation,size,cells,time_s,output
"""

import argparse
import csv
import subprocess
import time
import sys
from pathlib import Path
from PIL import Image
import numpy as np


def make_test_image(path: Path, size=(512, 512)):
    W, H = size
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            arr[y, x] = [(x * 255) // max(1, W - 1), (y * 255) // max(1, H - 1), ((x + y) * 255) // (W + H - 2)]
    Image.fromarray(arr, mode="RGB").save(path)


def run_python_fast(inp, out, cells, jitter):
    # Run the fast Python implementation in a separate process to avoid
    # import-time binary/incompat issues in the current interpreter.
    cmd = [
        sys.executable,
        "-m",
        "python.voronoify_image_fast",
        str(inp),
        "--out",
        str(out),
        "--cells",
        str(cells),
        "--jitter",
        str(jitter),
    ]
    t0 = time.time()
    subprocess.check_call(cmd)
    return time.time() - t0


def run_cupy(inp, out, cells, jitter):
    # invoke the CuPy script as a separate process; if cupy isn't installed the subprocess will fail
    cmd = [
        sys.executable,
        "python/voronoify_cupy.py",
        str(inp),
        "--out",
        str(out),
        "--cells",
        str(cells),
        "--jitter",
        str(jitter),
    ]
    t0 = time.time()
    try:
        subprocess.check_call(cmd)
        return time.time() - t0
    except FileNotFoundError:
        # script missing
        return None
    except subprocess.CalledProcessError:
        # script ran but failed (e.g., cupy not available); signal caller
        return None


def run_native(inp, out, cells, jitter):
    exe = Path.cwd() / "bin" / "voronoify_native"
    if not exe.exists():
        return None
    t0 = time.time()
    subprocess.check_call([str(exe), str(inp), str(out), str(cells), str(jitter), "42"])
    return time.time() - t0


def build_rust(project_dir: Path):
    if not (project_dir / "Cargo.toml").exists():
        return False
    print("Building Rust release binaries (this may take a while)...")
    subprocess.check_call(["cargo", "build", "--release"], cwd=str(project_dir))
    return True


def run_rust_binary(bin_name: str, inp: Path, out: Path, cells: int, jitter: float, project_dir: Path):
    # Attempt to find built binary in target/release or target/debug
    candidates = [project_dir / "target" / "release" / bin_name, project_dir / "target" / "debug" / bin_name]
    exe = None
    for c in candidates:
        if c.exists():
            exe = c
            break
    if exe is None:
        # try calling via cargo run (slow)
        try:
            t0 = time.time()
            subprocess.check_call(
                [
                    "cargo",
                    "run",
                    "--release",
                    "--bin",
                    bin_name,
                    "--",
                    str(inp),
                    "--out",
                    str(out),
                    "--cells",
                    str(cells),
                    "--jitter",
                    str(jitter),
                ],
                cwd=str(project_dir),
            )
            return time.time() - t0
        except Exception:
            return None
    t0 = time.time()
    subprocess.check_call([str(exe), str(inp), "--out", str(out), "--cells", str(cells), "--jitter", str(jitter)])
    return time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--cells", type=int, default=1024)
    p.add_argument("--jitter", type=float, default=0.6)
    p.add_argument("--outdir", default="bench/bench_out")
    p.add_argument("--build-rust", action="store_true", help="Build Rust binaries before running")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    inp = outdir / "bench_in.png"
    make_test_image(inp, size=(args.size, args.size))

    results_csv = outdir / "results.csv"
    with open(results_csv, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["implementation", "size", "cells", "time_s", "output"])

        # Python fast
        py_out = outdir / "py_fast_out.png"
        try:
            t = run_python_fast(inp, py_out, args.cells, args.jitter)
            writer.writerow(["python_fast", f"{args.size}x{args.size}", args.cells, f"{t:.6f}", str(py_out)])
            print(f"python_fast: {t:.3f}s")
        except Exception as e:
            print("python_fast failed:", e)

        # CuPy
        cupy_out = outdir / "cupy_out.png"
        try:
            t = run_cupy(inp, cupy_out, args.cells, args.jitter)
            if t is not None:
                writer.writerow(["cupy", f"{args.size}x{args.size}", args.cells, f"{t:.6f}", str(cupy_out)])
                print(f"cupy: {t:.3f}s")
            else:
                print("cupy not available; skipped")
        except Exception as e:
            print("cupy failed:", e)

        # Native
        native_in = outdir / "bench_in.ppm"
        native_out = outdir / "native_out.ppm"
        Image.open(inp).save(native_in)
        try:
            t = run_native(native_in, native_out, args.cells, args.jitter)
            if t is not None:
                writer.writerow(["native_cuda", f"{args.size}x{args.size}", args.cells, f"{t:.6f}", str(native_out)])
                # transform ppm to png for easier viewing
                img = Image.open(native_out)
                png_out = outdir / "native_out.png"
                img.save(png_out)
                print(f"native_cuda: {t:.3f}s")
            else:
                print("native_cuda not built; skipped")
        except Exception as e:
            print("native_cuda failed:", e)

        # Rust binaries
        rust_proj = Path.cwd() / "rust"
        if args.build_rust:
            try:
                build_rust(rust_proj)
            except Exception as e:
                print("Rust build failed:", e)

        # try parallel binary name first
        try:
            t = run_rust_binary(
                "voronoify_parallel",
                inp,
                outdir / "rs_parallel_out.png",
                args.cells,
                args.jitter,
                rust_proj,
            )
            if t is not None:
                writer.writerow(
                    [
                        "rust_parallel",
                        f"{args.size}x{args.size}",
                        args.cells,
                        f"{t:.6f}",
                        str(Path.cwd() / "img" / "rs_parallel_out.png"),
                    ]
                )
                print(f"rust_parallel: {t:.3f}s")
            else:
                print("rust_parallel not found; skipped")
        except Exception as e:
            print("rust_parallel failed:", e)

        # try single-thread binary
        try:
            t = run_rust_binary("voronoify-rs", inp, outdir / "rs_single_out.png", args.cells, args.jitter, rust_proj)
            if t is not None:
                writer.writerow(
                    [
                        "rust_single",
                        f"{args.size}x{args.size}",
                        args.cells,
                        f"{t:.6f}",
                        str(outdir / "rs_single_out.png"),
                    ]
                )
                print(f"rust_single: {t:.3f}s")
            else:
                print("rust_single not found; skipped")
        except Exception as e:
            print("rust_single failed:", e)

    print(f"Results written to {results_csv}")


if __name__ == "__main__":
    main()
