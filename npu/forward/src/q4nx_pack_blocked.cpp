// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

// One code: roundf(v * tis) clamped to [-127, 127], a non-finite v as 0 -- the expression this packer has always used,
// written so the loop vectorises (engine#1783: std::roundf and std::isfinite kept it scalar, ~3 ns per element, and a
// 1536 x 1536 activation is packed on every attention call). Exact, not approximate: y = v * tis is finite with
// |y| <= 127 (v is bounded by its column's amax, or zeroed), so the truncation t fits an int and y - t is exact; rounding
// half away from zero is then t + 1 for a fraction >= 0.5 and t - 1 for one <= -0.5. `v - v == 0` holds for finite v only.
inline int8_t quant_code(float v, float tis) {
    const float y = (v - v == 0.0f) ? v * tis : 0.0f;
    int x = (int)y;
    const float f = y - (float)x;
    x += (f >= 0.5f) - (f <= -0.5f);
    x = x > 127 ? 127 : x;
    x = x < -127 ? -127 : x;
    return (int8_t)x;
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
    // Column amax in parallel over row chunks, each thread into its own array, merged after: a max is exact and
    // order-independent, so the scales are the serial loop's bit for bit (engine#1783: the serial pass was most of a
    // 1536 x 1536 pack once the write below was parallel).
    std::vector<float> amax((size_t)N, 0.0f);
    std::vector<float> tis((size_t)N, 0.0f);
#pragma omp parallel
    {
        std::vector<float> mine((size_t)N, 0.0f);
#pragma omp for schedule(static) nowait
        for (int i = 0; i < K; i++) {
            const float* row = w + (size_t)i * (size_t)N;
            for (int j = 0; j < N; j++) {
                const float a = std::fabs(row[j]);
                if (std::isfinite(a) && a > mine[(size_t)j]) mine[(size_t)j] = a;
            }
        }
#pragma omp critical(q4nx_pack_amax)
        for (int j = 0; j < N; j++)
            if (mine[(size_t)j] > amax[(size_t)j]) amax[(size_t)j] = mine[(size_t)j];
    }
    double ssum = 0.0;
    for (int j = 0; j < N; j++) {
        float am = amax[(size_t)j];
        if (am < 1e-12f) am = 1.0f;
        col_scales[j] = am / 127.0f;
        tis[(size_t)j] = 127.0f / am;
        ssum += (double)(am / 127.0f);
    }
    // The write, one address per 8-column run. Inside the blocked layout, columns j0..j0+7 of a row with j0 % 8 == 0 sit
    // in one 8x8 microtile row, so they are 8 consecutive bytes from blocked_index(i, j0) (q4nx_weight_layout.h, the
    // image_index/copy note); row-major is consecutive anyway. The element arithmetic is unchanged.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < K; i++) {
        const float* row = w + (size_t)i * (size_t)N;
        for (int j0 = 0; j0 < N; j0 += 8) {
            int8_t* dst = Bm + bpos((size_t)i, (size_t)j0, KD, ND, blocked);
            const int jn = N - j0 < 8 ? N - j0 : 8;
            for (int c = 0; c < jn; c++) dst[c] = quant_code(row[j0 + c], tis[(size_t)(j0 + c)]);
        }
    }
    *ssum_out = ssum;
    return 0;
}
