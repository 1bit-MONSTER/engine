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

}  // namespace

// ── 4-bit twin (i4 plan, 2026-11-12) ──
// Same traversal and the same per-column scale contract as the int8 packer, so
// the existing read-back (Cm * group_scales[layer][n]) needs no change. Two
// nibbles per byte, in the SAME element order that unpack_i4_b_plain expands:
// element e = byte e/2, even e = low nibble.
extern "C" int q4nx_pack_B_int4_blocked(uint8_t* Bm, const float* w, int K, int N, int KD, int ND,
                                        float* col_scales, double* ssum) {
    if (!Bm || !w || K <= 0 || N <= 0) return -1;
    const float qmax = 7.0f;
    double acc = 0;
    for (int j = 0; j < N; ++j) {
        float amax = 0;
        for (int i = 0; i < K; ++i) amax = std::max(amax, std::fabs(w[(size_t)i * N + j]));
        float s = amax > 0 ? amax / qmax : 1.0f;
        if (col_scales) col_scales[j] = s;
        acc += s;
        for (int i = 0; i < K; ++i) {
            float v = w[(size_t)i * N + j] / s;
            int q = (int)std::lrint(std::max(-8.0f, std::min(qmax, v)));
            size_t e = (size_t)q4nx_layout::blocked_index((size_t)i, (size_t)j, KD, ND);
            const uint8_t nib = (uint8_t)(q & 0x0F);
            // CONVENTION UNRESOLVED (see plan 9 item 3): three candidates --
            // consecutive (e, e+1), de-interleaved (e, e+32 per 64-lane group,
            // measured: different output, still wrong), and odd-low.  The
            // consecutive form is restored because NPU_I4_EMU proved it correct
            // through the int8 path; settle it with a ramp probe, not by
            // guessing.  aiecc/simulator cannot run this builtin, so the probe
            // has to be on-device.
            // Lane-permutation probe: the aie2p unpack builtin does not put
            // nibble e in lane e for every convention.  NPU_I4_PERM selects the
            // candidate mapping so all of them can be measured from ONE binary
            // (0 = consecutive even-low, the form NPU_I4_EMU proved correct).
            static const int perm = getenv("NPU_I4_PERM") ? atoi(getenv("NPU_I4_PERM")) : 0;
            size_t byte_i; bool low;
            switch (perm) {
                case 1: byte_i = e >> 1; low = (e & 1) != 0; break;
                case 2: { size_t g = e & ~(size_t)63; byte_i = (g >> 1) + ((e - g) & 31); low = (e - g) < 32; break; }
                case 3: { size_t g = e & ~(size_t)63; byte_i = (g >> 1) + ((e - g) & 31); low = (e - g) >= 32; break; }
                case 4: { size_t g = e & ~(size_t)31; byte_i = (g >> 1) + ((e - g) & 15); low = (e - g) < 16; break; }
                case 5: { size_t g = e & ~(size_t)31; byte_i = (g >> 1) + ((e - g) & 15); low = (e - g) >= 16; break; }
                default: byte_i = e >> 1; low = (e & 1) == 0; break;
            }
            uint8_t& byte = Bm[byte_i];
            byte = low ? (uint8_t)((byte & 0xF0) | nib) : (uint8_t)((byte & 0x0F) | (nib << 4));
        }
    }
    if (ssum) *ssum = acc;
    return 0;
}

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
