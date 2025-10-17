// Native CUDA implementation for Voronoify (JFA) with simple PPM I/O.
// Moved into cuda/ directory; build with: make -C cuda

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <stdint.h>

// Read/write simple binary PPM (P6) to avoid external image deps.
static bool read_ppm(const char *path, std::vector<uint8_t> &out, int &w, int &h) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    char header[3] = {0};
    if (fscanf(f, "%2s", header) != 1) { fclose(f); return false; }
    if (strcmp(header, "P6") != 0) { fclose(f); return false; }
    // skip comments
    int c = getc(f);
    while (c == '\n' || c == '\r' || c == '#') {
        if (c == '#') {
            // skip line
            while ((c = getc(f)) != '\n' && c != EOF) ;
        }
        c = getc(f);
    }
    ungetc(c, f);
    int maxv = 0;
    if (fscanf(f, "%d %d %d", &w, &h, &maxv) != 3) { fclose(f); return false; }
    if (maxv != 255) { fclose(f); return false; }
    // consume single whitespace
    fgetc(f);
    size_t sz = (size_t)w * (size_t)h * 3;
    out.resize(sz);
    if (fread(out.data(), 1, sz, f) != sz) { fclose(f); return false; }
    fclose(f);
    return true;
}

static bool write_ppm(const char *path, const uint8_t *data, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    size_t sz = (size_t)w * (size_t)h * 3;
    if (fwrite(data, 1, sz, f) != sz) { fclose(f); return false; }
    fclose(f);
    return true;
}

// CUDA kernels
extern "C" __global__ void init_labels_kernel(int *labels, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    labels[y * w + x] = -1;
}

extern "C" __global__ void seed_kernel(const int *seed_x, const int *seed_y, const int *seed_id, int seed_count, int *labels, int w, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seed_count) return;
    int sx = seed_x[i];
    int sy = seed_y[i];
    if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
        labels[sy * w + sx] = seed_id[i];
    }
}

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
    for (int oy = -step; oy <= step; oy += step) {
        for (int ox = -step; ox <= step; ox += step) {
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

int next_pow2_floor(int v) {
    if (v <= 0) return 0;
    int p = 1 << (31 - __builtin_clz(v));
    return p;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.ppm output.ppm [cells] [jitter] [seed]\n", argv[0]);
        return 1;
    }
    const char *inpath = argv[1];
    const char *outpath = argv[2];
    int n_cells = argc > 3 ? atoi(argv[3]) : 1200;
    float jitter = argc > 4 ? atof(argv[4]) : 0.5f;
    unsigned seed = argc > 5 ? (unsigned)atoi(argv[5]) : 0u;

    std::vector<uint8_t> img;
    int W, H;
    if (!read_ppm(inpath, img, W, H)) {
        fprintf(stderr, "failed to read PPM %s\n", inpath);
        return 1;
    }
    fprintf(stderr, "loaded %s (%dx%d)\n", inpath, W, H);

    // generate stratified seeds
    float aspect = (float)W / (float)H;
    int gy = (int)std::sqrt(n_cells / aspect);
    gy = std::max(1, gy);
    int gx = std::max(1, (int)std::round((float)n_cells / gy));
    float tile_w = (float)W / (float)gx;
    float tile_h = (float)H / (float)gy;

    std::vector<int> seed_x; seed_x.reserve(gx*gy);
    std::vector<int> seed_y; seed_y.reserve(gx*gy);
    std::vector<int> seed_id; seed_id.reserve(gx*gy);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ur(-0.5f, 0.5f);
    for (int iy=0; iy<gy; ++iy) {
        for (int ix=0; ix<gx; ++ix) {
            float cx = (ix + 0.5f) * tile_w;
            float cy = (iy + 0.5f) * tile_h;
            float jx = ur(rng) * 2.0f * jitter * (tile_w * 0.5f);
            float jy = ur(rng) * 2.0f * jitter * (tile_h * 0.5f);
            int sx = std::lroundf(std::min(std::max(cx + jx, 0.0f), (float)W - 1.0f));
            int sy = std::lroundf(std::min(std::max(cy + jy, 0.0f), (float)H - 1.0f));
            seed_x.push_back(sx);
            seed_y.push_back(sy);
            seed_id.push_back((int)seed_id.size());
        }
    }
    int N = (int)seed_x.size();
    fprintf(stderr, "generated %d seeds (grid %dx%d)\n", N, gx, gy);

    // device allocations
    int *d_seed_x = nullptr, *d_seed_y = nullptr, *d_seed_id = nullptr;
    int *d_labels = nullptr, *d_labels2 = nullptr;
    size_t img_pixels = (size_t)W * (size_t)H;
    cudaError_t cerr;
    cerr = cudaMalloc((void**)&d_seed_x, N * sizeof(int)); if (cerr != cudaSuccess) { fprintf(stderr, "cudaMalloc seed_x failed: %s\n", cudaGetErrorString(cerr)); return 1; }
    cerr = cudaMalloc((void**)&d_seed_y, N * sizeof(int)); if (cerr != cudaSuccess) { fprintf(stderr, "cudaMalloc seed_y failed: %s\n", cudaGetErrorString(cerr)); return 1; }
    cerr = cudaMalloc((void**)&d_seed_id, N * sizeof(int)); if (cerr != cudaSuccess) { fprintf(stderr, "cudaMalloc seed_id failed: %s\n", cudaGetErrorString(cerr)); return 1; }
    cerr = cudaMalloc((void**)&d_labels, img_pixels * sizeof(int)); if (cerr != cudaSuccess) { fprintf(stderr, "cudaMalloc labels failed: %s\n", cudaGetErrorString(cerr)); return 1; }
    cerr = cudaMalloc((void**)&d_labels2, img_pixels * sizeof(int)); if (cerr != cudaSuccess) { fprintf(stderr, "cudaMalloc labels2 failed: %s\n", cudaGetErrorString(cerr)); return 1; }

    // copy seeds
    cudaMemcpy(d_seed_x, seed_x.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seed_y, seed_y.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seed_id, seed_id.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // init labels to -1
    dim3 block(16,16);
    dim3 grid((W + block.x-1)/block.x, (H + block.y-1)/block.y);
    init_labels_kernel<<<grid, block>>>(d_labels, W, H);
    cudaDeviceSynchronize();

    // seed kernel (1D)
    int threads1d = 256;
    int blocks1d = (N + threads1d - 1) / threads1d;
    seed_kernel<<<blocks1d, threads1d>>>(d_seed_x, d_seed_y, d_seed_id, N, d_labels, W, H);
    cudaDeviceSynchronize();

    // JFA ping-pong
    int maxdim = std::max(W, H);
    int step = next_pow2_floor(maxdim);
    int pass = 0;
    int *src = d_labels;
    int *dst = d_labels2;
    while (step >= 1) {
        pass++;
        jfa_step_rw<<<grid, block>>>(src, dst, d_seed_x, d_seed_y, d_seed_id, N, step, W, H);
        cudaDeviceSynchronize();
        // swap
        int *tmp = src; src = dst; dst = tmp;
        step >>= 1;
    }

    // download final labels
    std::vector<int> labels(img_pixels);
    cudaMemcpy(labels.data(), src, img_pixels * sizeof(int), cudaMemcpyDeviceToHost);

    // compute mean color per label on host
    std::vector<uint64_t> counts(N, 0);
    std::vector<double> sums(N * 3, 0.0);
    for (size_t i = 0; i < img_pixels; ++i) {
        int lbl = labels[i];
        if (lbl < 0 || lbl >= N) continue;
        counts[lbl]++;
        sums[lbl*3 + 0] += img[i*3 + 0];
        sums[lbl*3 + 1] += img[i*3 + 1];
        sums[lbl*3 + 2] += img[i*3 + 2];
    }
    std::vector<uint8_t> out(img_pixels * 3);
    for (size_t i = 0; i < img_pixels; ++i) {
        int lbl = labels[i];
        if (lbl < 0 || lbl >= N) { out[i*3+0]=0; out[i*3+1]=0; out[i*3+2]=0; continue; }
        if (counts[lbl] == 0) { out[i*3+0]=0; out[i*3+1]=0; out[i*3+2]=0; continue; }
        out[i*3+0] = (uint8_t)std::round(sums[lbl*3 + 0] / counts[lbl]);
        out[i*3+1] = (uint8_t)std::round(sums[lbl*3 + 1] / counts[lbl]);
        out[i*3+2] = (uint8_t)std::round(sums[lbl*3 + 2] / counts[lbl]);
    }

    if (!write_ppm(outpath, out.data(), W, H)) {
        fprintf(stderr, "failed to write %s\n", outpath);
    } else {
        fprintf(stderr, "wrote %s\n", outpath);
    }

    // free
    cudaFree(d_seed_x); cudaFree(d_seed_y); cudaFree(d_seed_id);
    cudaFree(d_labels); cudaFree(d_labels2);
    return 0;
}
