// EXP13 diagnostic probe: which term binds the tiled f32 GEMM?
//
// Registered discrimination table (experiments/LEDGER.md EXP13, fixed before
// measurement) — this probe prints the quantities that decide it:
//   grid parallelism -> TFLOPS tracks the reported grid block count at fixed
//                       TM/TN (small-N cells sensitive, LM head insensitive)
//   occupancy        -> TFLOPS tracks reported blocks/SM x threads
//   shared bandwidth -> TFLOPS tracks loads/FMA = (TM+TN)/(TM*TN)
//   barrier cost     -> BK 16 -> 32 (halved barriers) moves time >= 10%
//
// Measurement: flushed only (256 MB memset between timed launches) so every
// cell is DRAM-honest; 5 warmup + 8 samples, median reported.

#include "gemm_f32_tmpl.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using trail::check_cuda;

namespace {

constexpr int kMaxThreadsPerSm = 2048;

struct Cell {
    const char* name;
    int m;
    int n;
    int k;
};

struct Result {
    int bm = 0, bn = 0, bk = 0, tm = 0, tn = 0;
    int threads = 0;
    int blocks_per_sm = 0;
    int occupancy_pct = 0;
    double loads_per_fma = 0.0;
    // One entry per measurement cell, in Cell order.
    double us[3] = {0.0, 0.0, 0.0};
    long long grid[3] = {0, 0, 0};
    double tflops[3] = {0.0, 0.0, 0.0};
};

char* g_flush = nullptr;
constexpr std::size_t kFlushBytes = 256ull << 20;

template <typename G>
double time_cell(const float* x, const float* w, float* y, const Cell& c) {
    constexpr int kWarmup = 5;
    constexpr int kSamples = 8;
    for (int i = 0; i < kWarmup; ++i) {
        G::launch(x, w, y, c.m, c.n, c.k);
    }
    check_cuda(cudaDeviceSynchronize(), "warmup");
    std::vector<float> ts(kSamples);
    cudaEvent_t a = nullptr, b = nullptr;
    check_cuda(cudaEventCreate(&a), "ev");
    check_cuda(cudaEventCreate(&b), "ev");
    for (int s = 0; s < kSamples; ++s) {
        check_cuda(cudaMemset(g_flush, 0, kFlushBytes), "flush");
        check_cuda(cudaEventRecord(a), "rec");
        G::launch(x, w, y, c.m, c.n, c.k);
        check_cuda(cudaEventRecord(b), "rec");
        check_cuda(cudaEventSynchronize(b), "sync");
        float ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&ms, a, b), "elapsed");
        ts[s] = ms * 1000.0F;
    }
    std::sort(ts.begin(), ts.end());
    check_cuda(cudaEventDestroy(a), "ev");
    check_cuda(cudaEventDestroy(b), "ev");
    return ts[kSamples / 2];
}

// Explicit variant runner (template args spelled out at each call site).
template <int BM, int BN, int BK, int TM, int TN>
void measure(std::vector<Result>& out, const float* x, const float* w, float* y,
             const Cell (&cells)[3]) {
    using G = trail::Geometry<BM, BN, BK, TM, TN>;
    Result r;
    r.bm = BM;
    r.bn = BN;
    r.bk = BK;
    r.tm = TM;
    r.tn = TN;
    r.threads = G::kThreads;
    r.blocks_per_sm = G::blocks_per_sm();
    r.occupancy_pct =
        (r.blocks_per_sm * r.threads * 100) / kMaxThreadsPerSm;
    r.loads_per_fma = static_cast<double>(TM + TN) / static_cast<double>(TM * TN);
    for (int i = 0; i < 3; ++i) {
        r.us[i] = time_cell<G>(x, w, y, cells[i]);
        r.grid[i] = G::grid_blocks(cells[i].m, cells[i].n);
        const double flops = 2.0 * cells[i].m * cells[i].n * cells[i].k;
        r.tflops[i] = flops / (r.us[i] * 1e-6) / 1e12;
    }
    out.push_back(r);
}

void print_header(const Cell (&cells)[3]) {
    std::printf("cells:");
    for (const Cell& c : cells) {
        std::printf("  %s(M=%d,N=%d,K=%d)", c.name, c.m, c.n, c.k);
    }
    std::printf("\n");
    std::printf("%-22s %8s %6s %7s %7s %6s | %11s %11s %11s | %s\n",
                "geometry BM,BN,BK", "TMxTN", "thr", "blk/SM", "occ%",
                "lds/FMA", "cell0", "cell1", "cell2", "grids (blk:TFLOPS)");
}

void print_row(const Result& r, const Cell (&cells)[3]) {
    char geo[64];
    std::snprintf(geo, sizeof(geo), "%d,%d,%d", r.bm, r.bn, r.bk);
    char tile[32];
    std::snprintf(tile, sizeof(tile), "%dx%d", r.tm, r.tn);
    std::printf("%-22s %8s %6d %7d %6d%% %6.2f |", geo, tile, r.threads,
                r.blocks_per_sm, r.occupancy_pct, r.loads_per_fma);
    for (int i = 0; i < 3; ++i) {
        std::printf(" %6.0fus/%5.1f", r.us[i], r.tflops[i]);
    }
    std::printf(" |");
    for (int i = 0; i < 3; ++i) {
        std::printf(" %lld:%5.1f", r.grid[i], r.tflops[i]);
    }
    std::printf("\n");
    (void)cells;
}

}  // namespace

int main() {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&g_flush), kFlushBytes),
               "malloc flush");

    const Cell cells[3] = {
        {"QKV-M64", 64, 4096, 2048},
        {"QKV-M512", 512, 4096, 2048},
        {"LMhead-M512", 512, 151936, 2048},
    };

    // Allocate the largest operand set once: LM head W = 1244.7 MB.
    const std::size_t max_w = static_cast<std::size_t>(151936) * 2048;
    const std::size_t max_x = static_cast<std::size_t>(512) * 2048;
    const std::size_t max_y = static_cast<std::size_t>(512) * 151936;
    std::vector<float> w(max_w, 0.5F), x(max_x, 0.25F);
    float *d_w = nullptr, *d_x = nullptr, *d_y = nullptr;
    check_cuda(cudaMalloc(&d_w, w.size() * sizeof(float)), "mw");
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "mx");
    check_cuda(cudaMalloc(&d_y, max_y * sizeof(float)), "my");
    check_cuda(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "h2d w");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "h2d x");

    std::vector<Result> results;

    // Group A: TM=TN=8 (loads/FMA = 0.25), BK=16, block count varies 4x.
    measure<64, 64, 16, 8, 8>(results, d_x, d_w, d_y, cells);
    measure<128, 64, 16, 8, 8>(results, d_x, d_w, d_y, cells);
    measure<128, 128, 16, 8, 8>(results, d_x, d_w, d_y, cells);
    measure<256, 64, 16, 8, 8>(results, d_x, d_w, d_y, cells);
    measure<256, 128, 16, 8, 8>(results, d_x, d_w, d_y, cells);

    // Group B: TM=TN=4 (loads/FMA = 0.5), BK=32, same geometry ladder.
    measure<64, 32, 32, 4, 4>(results, d_x, d_w, d_y, cells);
    measure<128, 32, 32, 4, 4>(results, d_x, d_w, d_y, cells);
    measure<128, 64, 32, 4, 4>(results, d_x, d_w, d_y, cells);
    measure<128, 128, 32, 4, 4>(results, d_x, d_w, d_y, cells);

    // Barrier probe: same geometry, BK varies so barriers per K fall.
    // NOTE: static __shared__ is capped at 48 KB, which is why the probe runs
    // at (256,64) and (64,64) rather than (256,128) — the wide tile with
    // BK >= 32 needs dynamic shared + cudaFuncSetAttribute (a recorded
    // design constraint, not a bug).
    measure<256, 64, 32, 8, 8>(results, d_x, d_w, d_y, cells);
    measure<64, 64, 32, 8, 8>(results, d_x, d_w, d_y, cells);
    measure<64, 64, 64, 8, 8>(results, d_x, d_w, d_y, cells);

    print_header(cells);
    for (const Result& r : results) {
        print_row(r, cells);
    }

    check_cuda(cudaFree(d_w), "fw");
    check_cuda(cudaFree(d_x), "fx");
    check_cuda(cudaFree(d_y), "fy");
    check_cuda(cudaFree(g_flush), "ff");
    return EXIT_SUCCESS;
}