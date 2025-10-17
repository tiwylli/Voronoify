#!/usr/bin/env python3
"""CuPy-based Voronoify prototype (packaged copy).

This file is a copy of the top-level `voronoify_cupy.py` moved into the `python/`
package so it can be invoked as `python -m python.voronoify_cupy` if desired.
"""

import argparse
import math
import numpy as np
from PIL import Image
import cupy as cp

JFA_KERNEL = r"""
extern "C" __global__ void jfa_seed(const int *seed_x, const int *seed_y, const int *seed_id, int seed_count, int *labels, int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seed_count) return;
    int sx = seed_x[idx];
    int sy = seed_y[idx];
    if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
        labels[sy * w + sx] = seed_id[idx];
    }
}

// read labels from src (row-major) and write into dst using JFA step size 'step'
extern "C" __global__ void jfa_step_rw(const int *src, int *dst, const int *seed_x, const int *seed_y, const int *seed_id, int seed_count, int step, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int best = src[y * w + x];
    float bestd = 1e30f;
    if (best >= 0) {
        int bsx = seed_x[best];
        int bsy = seed_y[best];
        float dx = (float)bsx - (float)x;
        float dy = (float)bsy - (float)y;
        bestd = dx*dx + dy*dy;
    }
    for (int oy=-step; oy<=step; oy+=step) {
        for (int ox=-step; ox<=step; ox+=step) {
            int nx = x + ox;
            int ny = y + oy;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            int nid = src[ny * w + nx];
            if (nid < 0) continue;
            int sx = seed_x[nid];
            int sy = seed_y[nid];
            float dx = (float)sx - (float)x;
            float dy = (float)sy - (float)y;
            float d2 = dx*dx + dy*dy;
            if (d2 < bestd) {
                bestd = d2;
                best = nid;
            }
        }
    }
    dst[y * w + x] = best;
}
"""


def voronoi_cupy_jfa(image_path, out_path="voronoi_cupy.png", n_cells=1200, jitter=0.5, seed=0, debug=False):
    rng = np.random.default_rng(seed)
    im = Image.open(image_path).convert("RGB")
    H, W = im.size[1], im.size[0]
    if debug:
        im = im.resize((256, 256))
        H, W = im.size[1], im.size[0]
    img_np = np.array(im, dtype=np.uint8)

    aspect = W / H
    gy = int(np.sqrt(n_cells / aspect))
    gy = max(1, gy)
    gx = max(1, int(round(n_cells / gy)))
    tile_w = W / gx
    tile_h = H / gy
    xs = (np.arange(gx) + 0.5) * tile_w
    ys = (np.arange(gy) + 0.5) * tile_h
    X, Y = np.meshgrid(xs, ys)
    jx = (rng.random((gy, gx)) - 0.5) * 2.0 * jitter * (tile_w * 0.5)
    jy = (rng.random((gy, gx)) - 0.5) * 2.0 * jitter * (tile_h * 0.5)
    sites_x = (X + jx).clip(0, W - 1).round().astype(np.int32).ravel()
    sites_y = (Y + jy).clip(0, H - 1).round().astype(np.int32).ravel()
    N = sites_x.size
    site_ids = np.arange(N, dtype=np.int32)

    seed_x = cp.asarray(sites_x)
    seed_y = cp.asarray(sites_y)
    seed_id = cp.asarray(site_ids)

    mod = cp.RawModule(code=JFA_KERNEL, options=("--std=c++11",), name_expressions=["jfa_seed", "jfa_step_rw"])
    k_seed = mod.get_function("jfa_seed")
    k_step = mod.get_function("jfa_step_rw")

    labels_a = cp.full((H * W,), -1, dtype=cp.int32)
    labels_b = cp.full((H * W,), -1, dtype=cp.int32)

    threads = 256
    blocks = (N + threads - 1) // threads
    k_seed((blocks,), (threads,), (seed_x, seed_y, seed_id, N, labels_a, W, H))
    cp.cuda.Device().synchronize()

    maxdim = max(H, W)
    step = 1 << (maxdim.bit_length() - 1)
    block2d = (16, 16)
    gridx = (W + block2d[0] - 1) // block2d[0]
    gridy = (H + block2d[1] - 1) // block2d[1]
    grid = (gridx, gridy)
    src = labels_a
    dst = labels_b
    passno = 0
    while step >= 1:
        passno += 1
        if debug:
            print(f"JFA pass {passno} step={step} grid={grid} blocks={(blocks,)}")
        k_step(grid, block2d, (src, dst, seed_x, seed_y, seed_id, N, step, W, H))
        cp.cuda.Device().synchronize()
        src, dst = dst, src
        step >>= 1

    labels = cp.asnumpy(src).reshape((H, W))

    flat_labels = labels.ravel()
    flat_img = img_np.reshape(-1, 3).astype(np.float64)
    sums = np.zeros((N, 3), dtype=np.float64)
    counts = np.bincount(flat_labels, minlength=N).astype(np.int64)
    for ch in range(3):
        sums[:, ch] = np.bincount(flat_labels, weights=flat_img[:, ch], minlength=N)
    nonzero = counts > 0
    means = np.zeros_like(sums, dtype=np.uint8)
    means[nonzero] = np.clip((sums[nonzero] / counts[nonzero, None]), 0, 255).astype(np.uint8)

    out = means[flat_labels].reshape((H, W, 3)).astype(np.uint8)
    Image.fromarray(out, mode="RGB").save(out_path)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--out", default="voronoi_cupy.png")
    p.add_argument("--cells", type=int, default=1200)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    out = voronoi_cupy_jfa(
        args.image, out_path=args.out, n_cells=args.cells, jitter=args.jitter, seed=args.seed, debug=args.debug
    )
    print("saved:", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--out", default="voronoi_cupy.png")
    p.add_argument("--cells", type=int, default=1200)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    out = voronoi_cupy_jfa(
        args.image, out_path=args.out, n_cells=args.cells, jitter=args.jitter, seed=args.seed, debug=args.debug
    )
    print("saved:", out)
