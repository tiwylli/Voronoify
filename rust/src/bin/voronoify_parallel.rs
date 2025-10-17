use clap::Parser;
use image::{io::Reader as ImageReader, ImageBuffer, Rgb};
use kdtree::KdTree;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

#[derive(Parser, Debug)]
#[command(author, version, about = "Voronoify parallel (Rayon)")]
struct Args {
    image: String,
    #[arg(long, default_value = "voronoi_rs_parallel.png")]
    out: String,
    #[arg(long, default_value_t = 1200)]
    cells: usize,
    #[arg(long, default_value_t = 0.5)]
    jitter: f32,
    #[arg(long, default_value_t = 1)]
    edge_thickness: u32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let img = ImageReader::open(&args.image)?.decode()?.to_rgb8();
    let (w, h) = img.dimensions();

    // stratified sites
    let aspect = w as f32 / h as f32;
    let mut gy = ((args.cells as f32 / aspect).sqrt()).floor() as usize;
    gy = std::cmp::max(1, gy);
    let gx = std::cmp::max(1, ((args.cells as f32 / gy as f32).round()) as usize);

    let tile_w = w as f32 / gx as f32;
    let tile_h = h as f32 / gy as f32;

    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let mut sites: Vec<[f32; 2]> = Vec::with_capacity(gx * gy);
    for iy in 0..gy {
        for ix in 0..gx {
            let cx = (ix as f32 + 0.5) * tile_w;
            let cy = (iy as f32 + 0.5) * tile_h;
            let jx = (rng.gen::<f32>() - 0.5) * 2.0 * args.jitter * (tile_w * 0.5);
            let jy = (rng.gen::<f32>() - 0.5) * 2.0 * args.jitter * (tile_h * 0.5);
            let sx = (cx + jx).clamp(0.0, w as f32 - 1.0);
            let sy = (cy + jy).clamp(0.0, h as f32 - 1.0);
            sites.push([sx, sy]);
        }
    }
    let n_sites = sites.len();

    // build kdtree
    let mut kdt: KdTree<f32, usize, [f32; 2]> = KdTree::new(2);
    for (i, s) in sites.iter().enumerate() {
        kdt.add(*s, i).unwrap();
    }

    // Partition rows into chunks for work distribution
    let chunk_rows = 64_usize.min(h as usize);
    let row_ranges: Vec<(usize, usize)> = (0..h as usize)
        .step_by(chunk_rows)
        .map(|start| (start, std::cmp::min(h as usize, start + chunk_rows)))
        .collect();

    // Each chunk computes local labels, sums, counts
    let partials: Vec<(Vec<usize>, Vec<[f64; 3]>, Vec<u64>)> = row_ranges
        .par_iter()
        .map(|(row_start, row_end)| {
            let mut local_labels: Vec<usize> =
                Vec::with_capacity((row_end - row_start) * w as usize);
            let mut local_sums: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; n_sites];
            let mut local_counts: Vec<u64> = vec![0; n_sites];

            for y in *row_start..*row_end {
                for x in 0..w as usize {
                    let px = x as f32;
                    let py = y as f32;
                    let nearest = kdt.nearest(&[px, py], 1, &squared_euclid).unwrap();
                    let idx = *nearest[0].1;
                    local_labels.push(idx);
                    let p = img.get_pixel(x as u32, y as u32);
                    local_sums[idx][0] += p[0] as f64;
                    local_sums[idx][1] += p[1] as f64;
                    local_sums[idx][2] += p[2] as f64;
                    local_counts[idx] += 1;
                }
            }

            (local_labels, local_sums, local_counts)
        })
        .collect();

    // Reduce partials
    let mut sums: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; n_sites];
    let mut counts: Vec<u64> = vec![0; n_sites];
    let mut labels: Vec<usize> = Vec::with_capacity((w * h) as usize);

    for partial in partials.iter() {
        let (ref local_labels, ref local_sums, ref local_counts) = partial;
        labels.extend(local_labels.iter().copied());
        for i in 0..n_sites {
            sums[i][0] += local_sums[i][0];
            sums[i][1] += local_sums[i][1];
            sums[i][2] += local_sums[i][2];
            counts[i] += local_counts[i];
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

    // paint output image
    let mut out_img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(w, h);
    let mut li = 0usize;
    for _ in 0..row_ranges.len() {
        // each partial contains row chunk labels sequentially
        // iterate rows in chunk
        // we simply pop-through labels sequentially
        for _ in 0..(chunk_rows * w as usize) {
            if li >= labels.len() {
                break;
            }
            let idx = labels[li];
            // compute x,y from li
            let y = (li / w as usize) as u32;
            let x = (li % w as usize) as u32;
            let c = means[idx];
            out_img.put_pixel(x, y, Rgb([c[0], c[1], c[2]]));
            li += 1;
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
