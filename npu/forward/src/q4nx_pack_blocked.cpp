// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// The production weight packer, extracted from I8Ctx so it can be tested without XRT (engine #1416).
//
// `I8Ctx::packB_blocked` lived inside npu_engine_i8ctx_inc.h, which pulls the device runtime, so nothing could check it
// against the layout contract or against the row-major reference in q4nx_pack.c. The traversal question this issue turns
// on -- CB=64 stripes through a row-major matrix whose rows are 69 KB apart -- could only be *reasoned* about, and a
// reasoned change to the packer the device runs on every weight is exactly what should not happen.
//
// This is the same code with `bpos()` and the class's geometry passed in, so the extraction is behaviour-preserving by
// construction and the traversal can be changed next to a test.

#include "q4nx_pack.h"
#include "q4nx_weight_layout.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

inline size_t bpos(size_t i, size_t j, int KD, int ND, int blocked) {
    return blocked ? q4nx_layout::blocked_index(i, j, KD, ND)
                   : q4nx_layout::rowmajor_index(i, j, KD, ND);
}

}  // namespace

extern "C" int q4nx_pack_B_int8_blocked(int8_t* Bm, const float* w, int K, int N, int KD, int ND, int blocked,
                                        float* col_scales, double* ssum_out) {
    if (!Bm || !w || K <= 0 || N <= 0 || KD <= 0 || ND <= 0) return -1;
    if (!col_scales || !ssum_out) return -1;
    // ROW-major traversal (engine #1416). The CB=64 stripes this replaced touched 256 B of a row, then jumped 69 KB to the
    // next at 27B shapes, so every row opened fresh cache lines: measured 928 ms serial against 299 ms for the row-major
    // reference (q4nx_pack_B_int8), 384 vs 1194 MB/s. Both phases now stream whole rows, and the blocked write lands in
    // 128-column runs inside a tile, which is contiguous (`q4nx_layout::blocked_index`), so neither side of the pass is
    // strided.
    //
    // The scale sum is accumulated in a serial loop over j rather than inside a `reduction(+:ssum)` parallel loop. That is
    // deliberate: the mean column scale becomes order-independent, where the parallel reduction left its last bits to the
    // schedule. Codes and scales are unchanged (the amax of a column is a max over rows, and the rounding is per element).
    std::vector<float> amax((size_t)N, 0.0f);
    std::vector<float> tis((size_t)N, 0.0f);
    for (int i = 0; i < K; i++) {
        const float* row = w + (size_t)i * (size_t)N;
        for (int j = 0; j < N; j++) {
            const float a = std::fabs(row[j]);
            if (std::isfinite(a) && a > amax[(size_t)j]) amax[(size_t)j] = a;
        }
    }
    double ssum = 0.0;
    for (int j = 0; j < N; j++) {
        float am = amax[(size_t)j];
        if (am < 1e-12f) am = 1.0f;
        col_scales[j] = am / 127.0f;
        tis[(size_t)j] = 127.0f / am;
        ssum += (double)(am / 127.0f);
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < K; i++) {
        const float* row = w + (size_t)i * (size_t)N;
        for (int j = 0; j < N; j++) {
            float v = row[j];
            if (!std::isfinite(v)) v = 0.0f;
            int x = (int)std::roundf(v * tis[(size_t)j]);
            if (x > 127) x = 127;
            else if (x < -127) x = -127;
            Bm[bpos((size_t)i, (size_t)j, KD, ND, blocked)] = (int8_t)x;
        }
    }
    *ssum_out = ssum;
    return 0;
}
