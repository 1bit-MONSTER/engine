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
