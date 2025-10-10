# requirements:
#   pip install pillow numpy scipy

from PIL import Image
import numpy as np
from scipy.spatial import cKDTree


def voronoi_bitmap_colorize(
    image_path: str,
    # palette_path: str,
    out_path: str = "voronoi_bitmap.png",
    n_cells: int = 1200,
    jitter: float = 0.5,
    edge_thickness: int = 1,
    edge_color=(0, 0, 0),
    seed: int = 0,
):
    """
    Create a Voronoi-style mosaic of an input image where each cell's color is sampled
    from a separate "palette"/bitmap image at the seed's (u,v) position.

    Args
    ----
    image_path: path to input image (only used for size & optional edges)
    palette_path: path to bitmap used as color source (sampled by seed UV)
    out_path: where to save the result
    n_cells: approximate number of Voronoi sites
    jitter: 0..1 random offset inside each stratified tile (0=center, 1=full tile)
    edge_thickness: 0 for none; >=1 draws borders between cells
    edge_color: RGB tuple for borders
    seed: RNG seed for reproducibility
    """

    rng = np.random.default_rng(seed)

    # --- load images (we only need input size; colors come from palette) ---
    im = Image.open(image_path).convert("RGB")
    H, W = im.size[1], im.size[0]

    # palette = Image.open(palette_path).convert("RGB")
    # P_h, P_w = palette.size[1], palette.size[0]
    # palette_np = np.array(palette, dtype=np.uint8)

    # --- generate stratified sites (approx n_cells) with jitter ---
    # choose grid so grid_x * grid_y ~= n_cells, matching aspect ratio
    aspect = W / H
    gy = int(np.sqrt(n_cells / aspect))
    gy = max(1, gy)
    gx = max(1, int(np.round(n_cells / gy)))

    tile_w = W / gx
    tile_h = H / gy

    # centers with jitter inside each tile
    xs = (np.arange(gx) + 0.5) * tile_w
    ys = (np.arange(gy) + 0.5) * tile_h
    X, Y = np.meshgrid(xs, ys)  # shape (gy, gx)

    jx = (rng.random((gy, gx)) - 0.5) * 2.0 * jitter * (tile_w * 0.5)
    jy = (rng.random((gy, gx)) - 0.5) * 2.0 * jitter * (tile_h * 0.5)

    sites_x = (X + jx).clip(0, W - 1)
    sites_y = (Y + jy).clip(0, H - 1)
    sites = np.stack([sites_x.ravel(), sites_y.ravel()], axis=1)  # (N, 2) in (x, y)

    # --- build KD-tree over sites for nearest-site labeling ---
    tree = cKDTree(sites)

    # query per-pixel nearest site in manageable chunks
    # (flattened coords in (x, y) to match 'sites')
    ys_full, xs_full = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords_xy = np.stack([xs_full.ravel(), ys_full.ravel()], axis=1)

    labels = np.empty(coords_xy.shape[0], dtype=np.int32)
    batch = max(1, (H * W))  # chunking to keep mem reasonable
    for start in range(0, coords_xy.shape[0], batch):
        end = min(coords_xy.shape[0], start + batch)
        _, idx = tree.query(coords_xy[start:end], k=1, workers=-1)
        labels[start:end] = idx

    labels = labels.reshape(H, W)

    # # --- assign a color to each site from the palette bitmap ---
    # # map seed (x,y) -> UV in [0,1]^2 using normalized position in input image
    # # then sample palette at nearest pixel
    # sites_u = sites[:, 0] / (W - 1 + 1e-9)
    # sites_v = sites[:, 1] / (H - 1 + 1e-9)

    # px = np.clip((sites_u * (P_w - 1)).round().astype(int), 0, P_w - 1)
    # py = np.clip((sites_v * (P_h - 1)).round().astype(int), 0, P_h - 1)
    # site_colors = palette_np[py, px]  # (N, 3)

    # # paint output by label lookup
    # out = site_colors[labels]  # (H, W, 3), uint8

    # after 'labels' is computed, replace the color assignment block with:
    out = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(sites.shape[0]):
        mask = labels == i
        if not np.any(mask):
            continue
        mean_col = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)[mask].mean(axis=0)
        out[mask] = mean_col
    out = np.clip(out, 0, 255).astype(np.uint8)

    # --- optional edges between different labels ---
    if edge_thickness > 0:
        # find boundaries by comparing neighbor labels
        edge = np.zeros((H, W), dtype=bool)
        edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
        edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]

        if edge_thickness > 1:
            # simple dilation for thicker edges
            from scipy.ndimage import binary_dilation

            edge = binary_dilation(edge, iterations=edge_thickness - 1)

        # draw edges
        out[edge] = np.array(edge_color, dtype=np.uint8)

    Image.fromarray(out, mode="RGB").save(out_path)
    return out_path


# -----------------------------
# example usage:
# -----------------------------
if __name__ == "__main__":
    # input.png defines the size / geometry.
    # palette.png is ANY bitmap whose colors you want to “stamp” into Voronoi cells.
    result_path = voronoi_bitmap_colorize(
        image_path="wave.jpg",
        # palette_path="palette.png",  # e.g., a gradient, texture, or photo
        out_path="wave_bitmap10000.png",
        n_cells=10000,
        jitter=0.7,
        edge_thickness=0,
        edge_color=(10, 10, 10),
        seed=42,
    )
    print("saved:", result_path)
