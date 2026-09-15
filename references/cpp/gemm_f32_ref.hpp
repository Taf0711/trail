#pragma once

// EXP9 / M2 Rung 0: f32 GEMM CPU reference.
//
// Weight-stationary convention (same W layout as reference::gemv_f32: W
// rows are output features):
//     Y[M,N] = X[M,K] . W[N,K]^T
// i.e. y[m*N + n] = sum_k x[m*K + k] * w[n*K + k].
//
// Accumulation is sequential-k with std::fmaf — the exact order and
// operation the naive device kernel uses, so the differential gate is
// BITWISE (zero tolerance; E0003 precedent).

#include <cmath>
#include <cstddef>

namespace trail::reference {

inline void gemm_f32(const float* x, const float* w, int m_rows, int n_cols,
                     int k_dim, float* y) {
    for (int m = 0; m < m_rows; ++m) {
        const float* xr = x + static_cast<std::size_t>(m) * k_dim;
        for (int n = 0; n < n_cols; ++n) {
            const float* wr = w + static_cast<std::size_t>(n) * k_dim;
            float acc = 0.0F;
            for (int k = 0; k < k_dim; ++k) {
                acc = std::fmaf(xr[k], wr[k], acc);
            }
            y[static_cast<std::size_t>(m) * n_cols + n] = acc;
        }
    }
}

}  // namespace trail::reference
