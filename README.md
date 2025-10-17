
# Voronoify

Voronoify creates Voronoi-style mosaics of images. This repository contains multiple implementations and a small benchmark/test harness so you can compare CPU, Rust, and GPU approaches.

Top-level layout (relevant paths)

- `python/` — canonical Python sources (fast KD-tree/vectorized and CuPy prototype)
- `voronoify_image_fast.py`, `voronoify_cupy.py` — thin top-level wrappers that call into `python/`
- `cuda/` — native CUDA source and Makefile (builds into `bin/`)
- `bin/` — build outputs (native binary lives here: `bin/voronoify_native`)
- `img/` — default image outputs used by benches/tests
- `voronoify-rs/` — Rust implementations (single-threaded + Rayon-parallel)
- `bench/` — benchmark harnesses
- `tests/` — small smoke tests (pytest)

Why this repo has many implementations

- Fast Python: easy to iterate and inspect (KD-tree + vectorized aggregations)
- CuPy: prototype GPU label computation using Jump Flooding Algorithm (JFA)
- Native CUDA: production-like native JFA host + kernels (built with nvcc)
- Rust: performant CPU implementation (single-threaded and Rayon-parallel)

Requirements

- Linux (CUDA only required when building/running native CUDA)
- Python 3.8+ (3.10+ tested)
- `pip` and virtualenv recommended
- Optional: CUDA toolkit & nvcc to build the native binary
- Optional: CuPy (wheel matched to your CUDA version) for the CuPy prototype

Install Python deps (recommended)

Create and activate a virtualenv, then install dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

If you plan to run the CuPy prototype, install the correct CuPy wheel for your CUDA version (see CuPy docs). Example (CUDA 11.8):

```bash
# example; pick the wheel that matches your CUDA and platform
pip install cupy-cuda118
```

Build the native CUDA binary

The CUDA host+kernels live in `cuda/`. Build the binary into `bin/` with:

```bash
# build using the Makefile in cuda/ (produces bin/voronoify_native)
make -C cuda
# or override GPU architecture if needed, e.g. for Ampere:
make -C cuda NVCC_ARCH=sm_86
```

Build and run the Rust implementations

The Rust code lives in `rust/`. There are two binaries produced by the Cargo package:

- `target/release/voronoify-rs` — the single-threaded binary (from `src/main.rs`)
- `target/release/voronoify_parallel` — the Rayon-parallel binary (from `src/bin/voronoify_parallel.rs`)

Build both release binaries with a single command and run the one you want:

```bash
# from repository root
cd rust
# build release binaries
cargo build --release

# single-threaded:
target/release/voronoify-rs ../img/input.jpg --out ../img/rust_out.png --cells 1200 --jitter 0.6

# parallel (Rayon):
target/release/voronoify_parallel ../img/input.jpg --out ../img/rust_out_parallel.png --cells 1200 --jitter 0.6
```

For iterative development use `cargo run --release -- <args>` (release-mode) or `cargo run -- <args>` (debug-mode) and pass the desired args after `--`.


Run (examples)

Python (fast KD-tree implementation):

```bash
# top-level wrapper; implementation lives at python/voronoify_image_fast.py
python voronoify_image_fast.py input.jpg --out img/py_fast_out.png --cells 1200 --jitter 0.6
```

CuPy prototype (if you installed CuPy):

```bash
python voronoify_cupy.py input.jpg --out img/cupy_out.png --cells 2000 --jitter 0.6
```

Native CUDA binary (built into `bin/`):

```bash
# native binary expects PPM input/output (the Makefile and tests use this).
# Convert PNG to PPM for viewing (ImageMagick / magick)
magick img/native_out.png img/native_out.ppm
# Build first: make -C cuda
bin/voronoify_native input.ppm img/native_out.ppm 2000 0.6 42
# Convert PPM to PNG for viewing (ImageMagick / magick)
magick img/native_out.ppm img/native_out.png
```

Converting images for the native binary

ImageMagick can convert JPG/PNG to/from PPM which the native binary uses:

```bash
magick input.jpg input.ppm
magick img/native_out.ppm img/native_out.png
```

Benchmarking

There is a small benchmark harness at `bench/benchmark_all.py`. It generates a synthetic image and runs each implementation it can detect, writing images into `img/` and results into an output directory.

Example:

```bash
python bench/benchmark_all.py --size 512 --cells 512 --outdir bench/out_small
# Larger run (example):
python bench/benchmark_all.py --size 2048 --cells 4096 --outdir bench/out_big
```

Notes, troubleshooting, and recommendations

- The Python implementations are the simplest to run and debug. The canonical Python sources live in `python/`.
- For true end-to-end GPU speedups implement per-site color reduction on the GPU (atomics or tiled reduction). The native CUDA host currently performs the final color averaging on the host.