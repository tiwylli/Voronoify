#!/usr/bin/env python3
"""Faster Voronoi-style image mosaic (packaged copy).

This file is a copy of the top-level script moved into the `python/` package
so other tooling can import it reliably.
"""

import argparse
from typing import Tuple

from PIL import Image
import numpy as np
from scipy.spatial import cKDTree


def _mean_color_per_label(img_np: np.ndarray, labels: np.ndarray, n_labels: int) -> np.ndarray:
    H, W, C = img_np.shape
    flat_labels = labels.ravel()
    flat_img = img_np.reshape(-1, C).astype(np.float64)

    sums = np.zeros((n_labels, C), dtype=np.float64)
    counts = np.bincount(flat_labels, minlength=n_labels).astype(np.int64)

    for ch in range(C):
        sums[:, ch] = np.bincount(flat_labels, weights=flat_img[:, ch], minlength=n_labels)

    nonzero = counts > 0
    means = np.zeros_like(sums, dtype=np.float32)
    means[nonzero] = (sums[nonzero] / counts[nonzero, None]).astype(np.float32)
    return means


def voronoi_bitmap_colorize_fast(
    image_path: str,
    out_path: str = "voronoi_bitmap_fast.png",
    n_cells: int = 1200,
    jitter: float = 0.5,
    edge_thickness: int = 1,
    edge_color: Tuple[int, int, int] = (0, 0, 0),
    seed: int = 0,
    chunk_pixels: int = 4_000_000,
):
    rng = np.random.default_rng(seed)

    im = Image.open(image_path).convert("RGB")
    H, W = im.size[1], im.size[0]
    img_np = np.array(im, dtype=np.uint8)

    aspect = W / H
    gy = int(np.sqrt(n_cells / aspect))
    gy = max(1, gy)
    gx = max(1, int(np.round(n_cells / gy)))

    tile_w = W / gx
    tile_h = H / gy
    xs = (np.arange(gx) + 0.5) * tile_w
    ys = (np.arange(gy) + 0.5) * tile_h
    X, Y = np.meshgrid(xs, ys)

    jx = (rng.random((gy, gx)) - 0.5) * 2.0 * jitter * (tile_w * 0.5)
    jy = (rng.random((gy, gx)) - 0.5) * 2.0 * jitter * (tile_h * 0.5)

    sites_x = (X + jx).clip(0, W - 1)
    sites_y = (Y + jy).clip(0, H - 1)
    sites = np.stack([sites_x.ravel(), sites_y.ravel()], axis=1)
    n_sites = sites.shape[0]

    tree = cKDTree(sites)

    ys_full, xs_full = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.stack([xs_full.ravel(), ys_full.ravel()], axis=1)
    n_pixels = coords.shape[0]

    labels = np.empty(n_pixels, dtype=np.int32)
    chunk = max(1, min(n_pixels, chunk_pixels))
    for start in range(0, n_pixels, chunk):
        end = min(n_pixels, start + chunk)
        _, idx = tree.query(coords[start:end], k=1, workers=-1)
        labels[start:end] = idx

    labels = labels.reshape(H, W)

    site_colors = _mean_color_per_label(img_np, labels, n_sites)

    out = site_colors[labels]
    out = np.clip(out, 0, 255).astype(np.uint8)

    if edge_thickness > 0:
        edge = np.zeros((H, W), dtype=bool)
        edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
        edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
        if edge_thickness > 1:
            from scipy.ndimage import binary_dilation

            edge = binary_dilation(edge, iterations=edge_thickness - 1)
        out[edge] = np.array(edge_color, dtype=np.uint8)

    Image.fromarray(out, mode="RGB").save(out_path)
    return out_path


def _parse_args():
    p = argparse.ArgumentParser(description="Fast Voronoi image mosaic")
    p.add_argument("image", help="input image path")
    p.add_argument("--out", default="voronoi_bitmap_fast.png")
    p.add_argument("--cells", type=int, default=1200)
    p.add_argument("--jitter", type=float, default=0.7)
    p.add_argument("--edge-thickness", type=int, default=0)
    p.add_argument("--edge-color", type=int, nargs=3, default=(0, 0, 0))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = voronoi_bitmap_colorize_fast(
        image_path=args.image,
        out_path=args.out,
        n_cells=args.cells,
        jitter=args.jitter,
        edge_thickness=args.edge_thickness,
        edge_color=tuple(args.edge_color),
        seed=args.seed,
    )
    print("saved:", out)


def main():
    """Console entrypoint for the packaged fast implementation."""
    args = _parse_args()
    out = voronoi_bitmap_colorize_fast(
        image_path=args.image,
        out_path=args.out,
        n_cells=args.cells,
        jitter=args.jitter,
        edge_thickness=args.edge_thickness,
        edge_color=tuple(args.edge_color),
        seed=args.seed,
    )
    print("saved:", out)
