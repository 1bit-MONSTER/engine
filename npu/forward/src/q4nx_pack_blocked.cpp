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
    const int CB = 64;
    double ssum = 0;
#pragma omp parallel for reduction(+ : ssum) schedule(static)
    for (int jb = 0; jb < N; jb += CB) {
        const int jn = (N - jb < CB) ? (N - jb) : CB;
        float amax[64], tis[64];
        for (int b = 0; b < jn; b++) amax[b] = 0.0f;
        for (int i = 0; i < K; i++) {
            const float* row = w + (size_t)i * (size_t)N + (size_t)jb;
            for (int b = 0; b < jn; b++) {
                const float a = std::fabs(row[b]);
                if (std::isfinite(a) && a > amax[b]) amax[b] = a;
            }
        }
        for (int b = 0; b < jn; b++) {
            float am = amax[b];
            if (am < 1e-12f) am = 1.0f;
            col_scales[jb + b] = am / 127.0f;
            tis[b] = 127.0f / am;
            ssum += (double)(am / 127.0f);
        }
        for (int i = 0; i < K; i++) {
            const float* row = w + (size_t)i * (size_t)N + (size_t)jb;
            for (int b = 0; b < jn; b++) {
                float v = row[b];
                if (!std::isfinite(v)) v = 0.0f;
                int x = (int)std::roundf(v * tis[b]);
                if (x > 127) x = 127;
                else if (x < -127) x = -127;
                Bm[bpos((size_t)i, (size_t)(jb + b), KD, ND, blocked)] = (int8_t)x;
            }
        }
    }
    *ssum_out = ssum;
    return 0;
}
