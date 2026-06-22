
# Voronoify

Voronoify creates Voronoi-style mosaics of images. This repository contains multiple implementations and a small benchmark/test harness so you can compare CPU, Rust, and GPU approaches.
<!-- Insert source and voronoify image using /img/wave.jpg and native_out.png-->
| Source | Voronoify (native) |
|---:|:---|
| ![Source image](/img/wave.jpg) | ![Voronoify output](/img/native_out.png) |

Input (left) and native CUDA output (right); files are under img/.

Requirements

- Linux (CUDA only required when building/running native CUDA)
- Python 3.10+
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

## Run (examples)

### Local web UI

Voronoify includes a Gradio web interface that runs entirely on your machine. It binds to `127.0.0.1`, does not create a public share link, and opens in your default browser.

After installing the Python dependencies, run:

```bash
python python/voronoify_gui.py
```

The UI provides:

- Drag-and-drop image upload with PNG preview and download
- Bold, Balanced, and Fine presets plus editable cells, jitter, and seed controls
- Python, CuPy, native CUDA, and Rust backend selection when available
- A backend-status panel explaining why optional methods are unavailable
- A single-job queue and a Cancel button that terminates the active backend process

Notes

- Generated images are temporary; use the download control in the output preview to keep a result.
- The fast Python backend is selected by default. All backends run as subprocesses so failures and cancellation remain isolated from the web server.
- CuPy is enabled only when its matching wheel is installed and it can access a CUDA device.
- Native CUDA requires `bin/voronoify_native`; build it with `make -C cuda`.
- Rust requires a debug or release binary; build it with `cargo build --release --manifest-path rust/Cargo.toml`.
- Cell-edge controls remain available through the backend command-line interfaces, but are not exposed in the common web UI.

### Benchmarking

There is a small benchmark harness at `bench/benchmark_all.py`. It generates a synthetic image and runs each implementation it can detect, writing images into `img/` and results into an output directory.

Example:

```bash
python bench/benchmark_all.py --size 512 --cells 512 --outdir bench/out_small
# Larger run (example):
python bench/benchmark_all.py --size 2048 --cells 4096 --outdir bench/out_big
```


### Python (fast KD-tree implementation):

```bash
# top-level wrapper; implementation lives at python/voronoify_image_fast.py
python voronoify_image_fast.py input.jpg --out img/py_fast_out.png --cells 1200 --jitter 0.6
```

CuPy prototype (if you installed CuPy):

```bash
python voronoify_cupy.py input.jpg --out img/cupy_out.png --cells 2000 --jitter 0.6
```

### Native CUDA binary (built into `bin/`):

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
