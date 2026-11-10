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
        // the column form that matches packB_blocked(): ONE multiplier per weight column, amax over the whole
        // K, int8 codes, inner sum exact int32. Checked against an independent double computation.
        const int KC = 130, NC = 3;
        std::vector<float> A2(KC), B2((size_t)KC * NC), want2((size_t)NC, 0.0f);
        std::vector<int8_t> codes3((size_t)KC * NC, 0);
        std::vector<float> d2((size_t)NC, 0.0f);
        for (int k = 0; k < KC; k++) A2[k] = float((k * 7) % 11) - 5.0f;
        for (int k = 0; k < KC; k++)
            for (int n = 0; n < NC; n++) B2[(size_t)k * NC + n] = float((k * 5 + n * 13) % 17) / 3.0f;
        float amaxA = 0.0f;
        for (int k = 0; k < KC; k++) amaxA = std::max(amaxA, std::fabs(A2[k]));
        const float ascale2 = amaxA / 127.0f, tis2 = 1.0f / ascale2;
        std::vector<int8_t> a8c(KC);
        for (int k = 0; k < KC; k++) {
            int q = int(std::round(A2[k] * tis2));
            a8c[k] = int8_t(q > 127 ? 127 : (q < -127 ? -127 : q));
        }
        for (int n = 0; n < NC; n++) {
            float amax = 0.0f;
            for (int k = 0; k < KC; k++) amax = std::max(amax, std::fabs(B2[(size_t)k * NC + n]));
            if (amax < 1e-12f) amax = 1.0f;
            d2[(size_t)n] = amax / 127.0f;
            for (int k = 0; k < KC; k++) {
                int q = int(std::round(B2[(size_t)k * NC + n] * (127.0f / amax)));
                codes3[(size_t)k * NC + n] = int8_t(q > 127 ? 127 : (q < -127 ? -127 : q));
            }
            double acc = 0;
            for (int k = 0; k < KC; k++) acc += double(codes3[(size_t)k * NC + n]) * double(a8c[k]);
            want2[(size_t)n] = float(d2[(size_t)n] * double(ascale2 * float(acc)));
        }
        std::vector<float> got2((size_t)NC, 0.0f);
        q4nx_host_gemm_i8_col(a8c.data(), KC, codes3.data(), d2.data(), NC, ascale2, got2.data());
        bool ok2 = true;
        for (int n = 0; n < NC; n++)
            ok2 &= std::fabs(got2[(size_t)n] - want2[(size_t)n]) <= 1e-5f * std::fabs(want2[(size_t)n]) + 1e-6f;
        check(ok2, "column form (our packing rule): one multiplier per column, int32 inner sum, matches an independent computation");
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
