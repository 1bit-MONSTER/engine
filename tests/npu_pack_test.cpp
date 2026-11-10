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

    /* engine#1839: the quantiser's known answer. A vector whose amax is 127 must come back as int8 codes in the INT8
     * range -- 127 stays 127 and -128 stays -128 -- because `codes` holds one int8 per element and the scale is amax/127.
     * The bug this pins: the clamp was the int4 range (-8..7), so 127 -> 7 and -128 -> -8, an 18x error on the largest
     * entries before the first multiply, in the arm the engine#1809 gate grades the device against. */
    {
        float   col[129];
        int8_t  codes[129];
        col[0] = 127.0f;  col[1] = -128.0f;  col[2] = 0.5f;  col[3] = -0.5f;
        for (int k = 4; k < 129; k++) col[k] = float((k % 17) - 8);      /* small values, still inside int8 */
        float d = 0.0f;
        q4nx_quantise_i8_group(col, 129, 1, &d, codes, 1);
        check(codes[0] == 127, "engine#1839: an amax element quantises to 127, not the int4 clamp 7");
        check(codes[1] == -126, "engine#1839: a large negative element quantises to -126, not the int4 clamp -8");
        check(codes[2] == 1, "engine#1839: 0.5 rounds half away to 1");
        check(codes[3] == -1, "engine#1839: -0.5 rounds half away to -1");
        check(fabsf(d - 1.0f) < 1e-6f, "engine#1839: the scale is amax/127 = 1.0 for an amax of 127");
    }

int main(int argc, char** argv) {
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
