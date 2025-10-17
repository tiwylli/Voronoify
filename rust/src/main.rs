use clap::Parser;
use image::{io::Reader as ImageReader, ImageBuffer, Rgb};
use kdtree::KdTree;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[derive(Parser, Debug)]
#[command(author, version, about = "Voronoify (single-threaded Rust port)")]
struct Args {
    /// input image path
    image: String,
    /// output path
    #[arg(long, default_value = "voronoi_rs_parallel.png")]
    out: String,
    /// approximate number of cells
    #[arg(long, default_value_t = 1200)]
    cells: usize,
    /// jitter 0..1
    #[arg(long, default_value_t = 0.5)]
    jitter: f32,
    /// edge thickness
    #[arg(long, default_value_t = 1)]
    edge_thickness: u32,
    /// seed
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let img = ImageReader::open(&args.image)?.decode()?.to_rgb8();
    let (W, H) = img.dimensions();

    // stratified sites
    let aspect = W as f32 / H as f32;
    let mut gy = ((args.cells as f32 / aspect).sqrt()).floor() as usize;
    gy = std::cmp::max(1, gy);
    let gx = std::cmp::max(1, ((args.cells as f32 / gy as f32).round()) as usize);

    let tile_w = W as f32 / gx as f32;
    let tile_h = H as f32 / gy as f32;

    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);

    let mut sites: Vec<[f32; 2]> = Vec::with_capacity(gx * gy);
    for iy in 0..gy {
        for ix in 0..gx {
            let cx = (ix as f32 + 0.5) * tile_w;
            let cy = (iy as f32 + 0.5) * tile_h;
            let jx = (rng.gen::<f32>() - 0.5) * 2.0 * args.jitter * (tile_w * 0.5);
            let jy = (rng.gen::<f32>() - 0.5) * 2.0 * args.jitter * (tile_h * 0.5);
            let sx = (cx + jx).clamp(0.0, W as f32 - 1.0);
            let sy = (cy + jy).clamp(0.0, H as f32 - 1.0);
            sites.push([sx, sy]);
        }
    }

    let n_sites = sites.len();

    // build kdtree (2D points)
    let mut kdt: KdTree<f32, usize, [f32; 2]> = KdTree::new(2);
    for (i, s) in sites.iter().enumerate() {
        kdt.add(*s, i).unwrap();
    }

    // labels per pixel
    let n_pixels = (W as usize) * (H as usize);
    let mut labels: Vec<usize> = vec![0; n_pixels];

    // query per pixel (single-threaded). Chunk over rows to keep memory behavior similar
    let chunk_rows = 256_usize.min(H as usize);
    for row_start in (0..H as usize).step_by(chunk_rows) {
        let row_end = std::cmp::min(H as usize, row_start + chunk_rows);
        for y in row_start..row_end {
            for x in 0..W as usize {
                let px = x as f32;
                let py = y as f32;
                let nearest = kdt.nearest(&[px, py], 1, &squared_euclid).unwrap();
                let idx = *nearest[0].1;
                labels[y * W as usize + x] = idx;
            }
        }
    }

    // aggregate sums and counts per site
    let mut sums: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; n_sites];
    let mut counts: Vec<u64> = vec![0; n_sites];

    for y in 0..H as usize {
        for x in 0..W as usize {
            let idx = labels[y * W as usize + x];
            let p = img.get_pixel(x as u32, y as u32);
            let r = p[0] as f64;
            let g = p[1] as f64;
            let b = p[2] as f64;
            sums[idx][0] += r;
            sums[idx][1] += g;
            sums[idx][2] += b;
            counts[idx] += 1;
        }
    }

    // compute means
    let mut means: Vec<[u8; 3]> = vec![[0, 0, 0]; n_sites];
    for i in 0..n_sites {
        if counts[i] > 0 {
            means[i][0] = (sums[i][0] / counts[i] as f64).round().clamp(0.0, 255.0) as u8;
            means[i][1] = (sums[i][1] / counts[i] as f64).round().clamp(0.0, 255.0) as u8;
            means[i][2] = (sums[i][2] / counts[i] as f64).round().clamp(0.0, 255.0) as u8;
        }
    }

    // create output image
    let mut out_img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(W, H);
    for y in 0..H as usize {
        for x in 0..W as usize {
            let idx = labels[y * W as usize + x];
            let c = means[idx];
            out_img.put_pixel(x as u32, y as u32, Rgb([c[0], c[1], c[2]]));
        }
    }

    out_img.save(&args.out)?;
    println!("saved: {}", args.out);
    Ok(())
}

fn squared_euclid(a: &[f32], b: &[f32]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}
