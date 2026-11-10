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

// npu_pack_test [<model-dir> <bo-dump-dir>]
//
// Without arguments (CI): the tile reorder is a permutation for even and odd
// group sizes, and bf16 rounding is round-to-nearest-even.
// With arguments: the packed weight, i5 and i6 buffers equal, byte for byte, the
// ones the reference lane fed the device (its RT_DUMP_BOS dump: w_<L>.bin,
// i5_<L>.bin, i6_<L>.bin for layers 0 and 2).
#include "full_elf.h"
#include "pack.h"
#include "q4nx_pack.h"
#include "batch_split.h"
#include "batch_width.h"

#include "golden/host_gemm_i8_fixture.h"

#include <cmath>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace onebit::npu;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

/* engine#1839: an int4 code and its scale are only correct as a PAIR. The defect this pins: both sites that quantise
 * float weights into the container's int4 codes used `amax / 127` (an INT8 scale) with the int4 clamp [-8, 7], so every
 * code but the very largest saturated and the dequantised weight was ~18x too small — in the host arm the device gate
 * compares against. The arms below are a known input -> known code (not a property any pairing satisfies), the no-
 * saturation property at full range, and the two degenerate cases the floor exists for. */
static void quant_int4_arms() {
    const float amax = 7.0f;
    check(q4nx_quant_int4_scale(7.0f) == 1.0f, "int4 pair: amax 7 takes scale 1 (so the codes ARE the values)");
    check(q4nx_quant_int4_code(7.0f, amax) == 7, "int4 pair: +amax quantises to +7");
    check(q4nx_quant_int4_code(-7.0f, amax) == -7, "int4 pair: -amax quantises to -7");
    check(q4nx_quant_int4_code(0.0f, amax) == 0, "int4 pair: zero is zero");
    check(q4nx_quant_int4_code(3.5f, amax) == 4, "int4 pair: half of amax rounds to 4 (round-half-away)");
    /* The arm the old pairing fails: at amax/127 a mid-range value saturates. */
    check(q4nx_quant_int4_code(0.1f, 1.0f) == 1, "int4 pair: a small value is 1, not saturated — the 127 scale gives 13");
    /* THE LADDER: the sevenths of amax must come back as their own code number. This is the arm that can fail for the
     * wrong scale -- under amax/127 every one of these saturates to +-7 (checked: v=0.1 at amax=1 gives 7 where int4
     * expects 1), so it distinguishes the pairings rather than asserting a property the clamp makes unfailable. */
    {
        const float amax_l = 3.25f;
        int wrong = 0;
        for (int k = -7; k <= 7; k++) {
            const float v = (float)k / 7.0f * amax_l;
            if (q4nx_quant_int4_code(v, amax_l) != k) wrong++;
        }
        check(wrong == 0, "int4 pair: each seventh of amax comes back as its own code (k/7*amax -> k, k=-7..7)");
    }
    /* Degenerate groups: the floor keeps the scale usable and the code zero. */
    check(q4nx_quant_int4_scale(0.0f) == 1.0f, "int4 pair: an all-zero group floors the scale at 1");
    check(q4nx_quant_int4_code(0.0f, 0.0f) == 0, "int4 pair: and its code is 0, not a division by zero");
    check(q4nx_quant_int4_code(NAN, 1.0f) == 0, "int4 pair: a NaN value quantises to 0 rather than an undefined code");
    /* The dequantised value round-trips inside half a step: that is the accuracy the kernel's epilogue assumes. */
    int worst_ok = 1;
    for (int i = -127; i <= 127; i++) {
        const float v = (float)i / 127.0f * 2.0f;     /* amax 2.0 */
        const float s = q4nx_quant_int4_scale(2.0f);
        const float back = (float)q4nx_quant_int4_code(v, 2.0f) * s;
        if (std::fabs(back - v) > s / 2.0f + 1e-6f) worst_ok = 0;
    }
    check(worst_ok, "int4 pair: every value round-trips within half a step of amax/7");
    std::printf("  int4 pair: 11 arms (known codes, ladder, floor, NaN, round-trip)\n");
}

/* engine#1900: the batch width is the chunk width, and the activation BO holds exactly that many rows -- an OVER-wide
 * answer reads past the BO (engine#1372, memory corruption rather than a slow path). The old code returned the first
 * populated design's width, so a mixed set answered whichever role was index 0.
 *
 * These arms pin the RULE (minimise, skip empty slots, take a fallback), not a measurement: no pack on this box has been
 * shown to adopt mixed widths, so nothing here claims the old answer was wrong on a real pack -- only that it would be
 * the dangerous way round if one ever did. The sealed route's width is #1904's, read from the plan and measured on the
 * device. */
static void batch_width_arms() {
    using onebit::npu::batch_width_min;
    const int mixed[] = {16, 16, 128};
    check(batch_width_min(mixed, 3, 1) == 16, "batch width: 16/16/128 answers 16, not the 128 at index 2");
    const int wide_first[] = {128, 16, 16};
    check(batch_width_min(wide_first, 3, 1) == 16,
          "batch width: 128 at index 0 does not win (the old code returned the FIRST populated entry)");
    const int slots[] = {0, 0, 16, 0};
    check(batch_width_min(slots, 4, 1) == 16, "batch width: unloaded slots are skipped, not treated as a width");
    const int none[] = {0, 0, 0};
    check(batch_width_min(none, 3, 1) == 1, "batch width: no design at all falls back to 1 (a single-row launch)");
    check(batch_width_min(none, 3, 16) == 16, "batch width: the fallback is the loaded design's own M when the caller has it");
    check(batch_width_min(nullptr, 0, 128) == 128, "batch width: a null array reads as empty and takes the fallback");
    check(batch_width_min(mixed, 0, 4) == 4, "batch width: an n of 0 reads nothing and takes the fallback");
    std::printf("  batch width: 7 arms (min not first, empty slots, fallback)\n");
}

/* engine#1959: a 17-row chunk ran the 128-row design and paid for 128 rows (6992 ms against 1073 ms for 16 rows on an
 * imported 7B pack). batch_split_rows() decides from the two MEASURED launch costs whether a chunk is walked at the
 * twin's width. These arms pin the decision, including the direction that must not change anything: one family only,
 * an unmeasured side, and a family where the wide launch is the cheaper one. */
static void batch_split_arms() {
    using onebit::npu::batch_split_rows;
    const double n7b = 7.6, w7b = 56.3;   // per-launch device wait measured on the 7B b8 family (ms)
    check(batch_split_rows(17, 16, 128, n7b, w7b) == 16, "batch split: 17 rows run as 2 narrow launches (15.2 ms against 56.3)");
    check(batch_split_rows(96, 16, 128, n7b, w7b) == 16, "batch split: 96 rows run as 6 narrow launches (45.6 ms against 56.3)");
    check(batch_split_rows(97, 16, 128, n7b, w7b) == 0,
          "batch split: 97 rows stay one wide launch (7 x 7.6 = 53.2 is not a tenth below 56.3)");
    check(batch_split_rows(128, 16, 128, n7b, w7b) == 0, "batch split: a full 128-row chunk stays one wide launch (60.8 > 56.3)");
    check(batch_split_rows(16, 16, 128, n7b, w7b) == 0 && batch_split_rows(1, 16, 128, n7b, w7b) == 0,
          "batch split: a chunk the twin already takes is not split");
    check(batch_split_rows(129, 16, 128, n7b, w7b) == 0, "batch split: a chunk wider than the wide design is the block walk's, not this rule's");
    check(batch_split_rows(17, 16, 128, 0.0, w7b) == 0 && batch_split_rows(17, 16, 128, n7b, 0.0) == 0,
          "batch split: an unmeasured side keeps the wide launch (that launch is how it gets measured)");
    check(batch_split_rows(17, 16, 16, n7b, w7b) == 0 && batch_split_rows(17, 128, 128, n7b, w7b) == 0 &&
          batch_split_rows(17, 0, 128, n7b, w7b) == 0 && batch_split_rows(17, 1, 128, n7b, w7b) == 0,
          "batch split: one family only (no twin, or twin == wide) never splits");
    // a family whose wide launch costs twice a narrow one: the wide launch wins from 33 rows, and 17..32 is level
    check(batch_split_rows(17, 16, 128, 0.59, 1.20) == 0, "batch split: 2 x 0.59 = 1.18 is not a tenth below 1.20, so 17 rows stay wide there");
    check(batch_split_rows(40, 16, 128, 0.59, 1.20) == 0, "batch split: 40 rows stay wide on a family where wide = 2 x narrow");
    check(batch_split_rows(17, 16, 128, 0.50, 1.20) == 16, "batch split: 2 x 0.50 = 1.00 is a tenth below 1.20, so it splits");
    std::printf("  batch split: 11 arms (crossover, unmeasured, one family, a family where wide wins)\n");
}

int main(int argc, char** argv) {
    batch_width_arms();
    batch_split_arms();
    quant_int4_arms();
    for (int G : {8, 9, 16}) {
        for (int n : {G, 3 * G}) {  // whole groups: a ragged tail is not a permutation
            // Tile i holds the value i in its first word; after the reorder every
            // source tile must appear exactly once.
            std::vector<uint8_t> src(size_t(n) * kTileBytes, 0), dst(src.size(), 0);
            for (int i = 0; i < n; ++i) std::memcpy(&src[size_t(i) * kTileBytes], &i, 4);
            reorder_tiles(dst.data(), src.data(), n, G);
            std::vector<int> seen;
            for (int o = 0; o < n; ++o) {
                int v;
                std::memcpy(&v, &dst[size_t(o) * kTileBytes], 4);
                seen.push_back(v);
            }
            std::sort(seen.begin(), seen.end());
            bool perm = true;
            for (int i = 0; i < n; ++i) perm &= seen[size_t(i)] == i;
            check(perm, "reorder is a permutation, G=" + std::to_string(G) + " n=" + std::to_string(n));
        }
    }
    check(f32_to_bf16(1.0f) == 0x3F80, "bf16(1.0)");
    check(f32_to_bf16(1.00390625f) == 0x3F80, "bf16 ties to even (down)");
    check(f32_to_bf16(1.01171875f) == 0x3F82, "bf16 ties to even (up)");
    // The integer host arm (engine#1808/#1809): "bit-exact against the host reference" needs a host path that
    // runs the SAME integer arithmetic as the kernel, and f32 cannot be bit-exact with int8 x int4 by
    // construction. These checks live here, next to the pack/tile checks they belong with, because ci.yml
    // builds this binary and a new ctest target would need a workflow edit this lane's token cannot make.
    {
        int32_t got[kHostI8N] = {0, 0, 0, 0};
        q4nx_host_gemm_i8(kHostI8A, kHostI8K, &kHostI8Codes[0][0], kHostI8N, got);
        bool same = true;
        for (int n = 0; n < kHostI8N; n++) same &= got[n] == kHostI8Expected[n];
        check(same, "int8 x int4 -> int32 matches the independent Python fixture (golden/host_gemm_i8_fixture.h)");
        bool in_range = true;
        for (int k = 0; k < kHostI8K; k++)
            for (int n = 0; n < kHostI8N; n++)
                if (kHostI8Codes[k][n] < -8 || kHostI8Codes[k][n] > 7) in_range = false;
        check(in_range, "every weight code is an int4 value in [-8, 7]");
        // a -128 activation is outside the quantiser's range but inside int8: it must accumulate exactly, and
        // the product bound for the largest K in use (14336) is 1.47e7, so int32 saturation would be a bug
        static int8_t a8[256], codes2[512];
        for (int k = 0; k < 256; k++) { a8[k] = (k == 0) ? -128 : 127; codes2[k * 2] = -8; codes2[k * 2 + 1] = 7; }
        int32_t edge[2] = {0, 0};
        q4nx_host_gemm_i8(a8, 256, codes2, 2, edge);
        check(edge[0] == -258056 && edge[1] == 225799,
              "a -128 activation accumulates exactly (no int8 intermediate, no int32 saturation)");
        // the reusable grouped form: inner sum int32, per-128-K multipliers applied outside it, K=130 so the
        // group boundary matters
        const int KK = 130, NN = 2, groups = 2;
        std::vector<float> A(KK), B((size_t)KK * NN);
        std::vector<int8_t> codes((size_t)KK * NN, 0);
        std::vector<float> d((size_t)groups * NN, 0.0f), want((size_t)NN, 0.0f);
        for (int k = 0; k < KK; k++) A[k] = float(k % 13) - 6.0f;
        for (int k = 0; k < KK; k++)
            for (int n = 0; n < NN; n++) B[(size_t)k * NN + n] = float((k * 3 + n) % 7) - 3.0f;
        float amax_all = 0.0f;
        for (int k = 0; k < KK; k++) amax_all = std::max(amax_all, std::fabs(A[k]));
        const float ascale = amax_all / 127.0f, tis = 1.0f / ascale;
        std::vector<int8_t> a8q(KK);
        for (int k = 0; k < KK; k++) {
            int q = int(std::round(A[k] * tis));
            a8q[k] = int8_t(q > 127 ? 127 : (q < -127 ? -127 : q));
        }
        for (int g = 0; g < groups; g++) {
            const int k0 = g * 128, k1 = (k0 + 128 < KK) ? k0 + 128 : KK;
            for (int n = 0; n < NN; n++) {
                float amax = 0.0f;
                for (int k = k0; k < k1; k++) amax = std::max(amax, std::fabs(B[(size_t)k * NN + n]));
                if (amax < 1e-12f) amax = 1.0f;
                d[(size_t)g * NN + n] = amax / 127.0f;
                for (int k = k0; k < k1; k++) {
                    int q = int(std::round(B[(size_t)k * NN + n] * (127.0f / amax)));
                    codes[(size_t)k * NN + n] = int8_t(q > 7 ? 7 : (q < -8 ? -8 : q));
                }
            }
        }
        for (int g = 0; g < groups; g++) {
            const int k0 = g * 128, k1 = (k0 + 128 < KK) ? k0 + 128 : KK;
            for (int n = 0; n < NN; n++) {
                double acc = 0;
                for (int k = k0; k < k1; k++) acc += double(codes[(size_t)k * NN + n]) * double(a8q[k]);
                want[(size_t)n] += float(d[(size_t)g * NN + n] * double(ascale * float(acc)));
            }
        }
        std::vector<float> outv((size_t)NN, 0.0f);
        q4nx_host_gemm_i8_grouped(a8q.data(), KK, codes.data(), d.data(), groups, NN, ascale, outv.data());
        bool ok = true;
        for (int n = 0; n < NN; n++)
            ok &= std::fabs(outv[(size_t)n] - want[(size_t)n]) <= 1e-5f * std::fabs(want[(size_t)n]) + 1e-6f;
        check(ok, "grouped form: int32 inner sum + per-128-K multipliers match an independent computation");
    }

    if (argc == 1) {
        std::printf("%s\n", failures ? "FAILED" : "ok (pass <model-dir> <bo-dump-dir> for the model checks)");
        return failures ? 1 : 0;
    }
    if (argc != 3) {
        std::fprintf(stderr, "usage: npu_pack_test [<model-dir> <bo-dump-dir>]\n");
        return 2;
    }
    const Model m(argv[1]);
    const std::string dump = argv[2];
    for (int L : {0, 2}) {
        char suffix[16];
        std::snprintf(suffix, sizeof suffix, "_%02d.bin", L);
        const Bytes w_ref = read_file(dump + "/w" + suffix), i5_ref = read_file(dump + "/i5" + suffix),
                    i6_ref = read_file(dump + "/i6" + suffix);
        std::vector<uint8_t> w(layer_weight_bytes(m)), i5(i5_ref.size(), 0), i6(i6_ref.size(), 0);
        pack_layer_weights(m, L, w.data());
        fill_i5(m, L, i5.data());
        fill_i6(m, L, i6.data());
        check(w == w_ref, "layer " + std::to_string(L) + " weights (" + std::to_string(w.size()) + " B)");
        check(i5 == i5_ref, "layer " + std::to_string(L) + " i5");
        check(i6 == i6_ref, "layer " + std::to_string(L) + " i6");
    }
    // At position 0 the RoPE half is cos = 1, sin = 0.
    uint16_t rope[128];
    fill_rope(rope, 0, 1000000.0f);
    check(rope[0] == 0x3F80 && rope[64] == 0, "rope pos 0");
    std::printf("%s\n", failures ? "FAILED" : "ok: weights, i5, i6 identical to the reference lane's buffers");
    return failures ? 1 : 0;
}
