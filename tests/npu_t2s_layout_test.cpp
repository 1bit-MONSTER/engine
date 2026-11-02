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

// npu_t2s_layout_test -- the t2s weight stream contract (q4nx_weight_layout "t2s"), no device.
//
// The engine copies a t2s sidecar into a weight BO unchanged and the kernel expands the 2-bit codes
// to the mmul's int4 nibbles as it reads them, so the host's only job is to hand it a file of the
// right size.  That makes exactly two things worth pinning here, and both are cross-checks against
// numbers the tool measured rather than against this header restated:
//
//   1. the tile is 2,176 B = 2,048 B of codes + 128 B of int16 multipliers, i.e. 2.125 bits per
//      weight, and `bits_per_weight()` says so;
//   2. for every 27B role, w_bytes() is the tile count times that tile -- and, per tensor, exactly
//      2176/4608 of the pb16i image's size.  The measured leg is the gate/up tensor
//      (K=5120, N=17408): 50,135,040 B of pb16i image -> 23,674,880 B of stream, which is what the
//      real-image run in tools/q4nx_t2s.py produced (0.4722x, 14.44 GB -> 6.82 GB over the set).
//
// Nothing here re-derives the code order: that is the kernel's tap and tools/q4nx_t2s.py's
// expand(), and tests/q4nx_t2s_test.py gates it byte for byte.
#include "q4nx_pb16i.h"
#include "q4nx_t2s.h"

#include <cstdio>

namespace {

int failures = 0;
void check(bool ok, const char* what, int K, int N) {
    if (!ok) {
        std::printf("FAIL %s (K=%d N=%d)\n", what, K, N);
        failures++;
    }
}

struct geom {
    int K, N;
};

// The 27B decode roles (the shapes the arena and the container use), plus boundary shapes.
const geom GEOMS[] = {
    {5120, 17408},   // GU: gate / up (the measured one)
    {17408, 5120},   // D
    {5120, 5120},    // q | gate half, o
    {5120, 256},     // k, v (NKV*HD = 2*128)
    {6144, 5120},    // z (GDN)
    {5120, 32},      // one tile's N: the smallest stream with a column
    {256, 32},       // one tile
};

void test_tile() {
    if (q4nx_t2s::CODE_BYTES != 2048 || q4nx_t2s::MS_BYTES != 128 || q4nx_t2s::TILE != 2176) {
        std::printf("FAIL tile geometry: codes %d ms %d tile %d\n", q4nx_t2s::CODE_BYTES,
                    q4nx_t2s::MS_BYTES, q4nx_t2s::TILE);
        failures++;
    }
    check(q4nx_t2s::TILE == q4nx_t2s::CODE_BYTES + q4nx_t2s::MS_BYTES, "tile = codes + ms", 0, 0);
    check(q4nx_t2s::bits_per_weight() == 2.125, "2,176 B per 8,192 weights is 2.125 bits", 0, 0);
    // The multiplier is stored once per 128-K block per column, i.e. a quarter of pb16i's four
    // duplicated copies per tile: exactly one 128-B side area per 2,048-B code area.
    check(q4nx_t2s::MS_BYTES == 2 * (q4nx_t2s::KC / 128) * q4nx_t2s::NW, "one int16 per 128 K", 0, 0);
    check(q4nx_pb16i::TILE == 4608 && q4nx_pb16i::CODE_BYTES == 4096, "pb16i tile is the 4.5 b/w one",
          0, 0);
}

void test_sizes() {
    for (const geom& g : GEOMS) {
        check(q4nx_t2s::fits(g.K, g.N), "geometry has a stream", g.K, g.N);
        const size_t t = q4nx_t2s::tiles(g.K, g.N);
        check(t == (size_t)(g.K / q4nx_t2s::KC) * (size_t)(g.N / q4nx_t2s::NW), "tile count", g.K, g.N);
        check(q4nx_t2s::w_bytes(g.K, g.N) == t * (size_t)q4nx_t2s::TILE, "size = tiles x 2,176",
              g.K, g.N);
        // Same weights, same tiles, a smaller tile: the stream is 2176/4608 of the pb16i image,
        // tensor for tensor, with no rounding anywhere in the relation.
        check(q4nx_t2s::w_bytes(g.K, g.N) * (size_t)q4nx_pb16i::TILE ==
                  q4nx_pb16i::w_bytes(g.K, g.N) * (size_t)q4nx_t2s::TILE,
              "size ratio is exactly 2176/4608 of pb16i", g.K, g.N);
    }
    // The measured tensor: blk_0_ffn_gate_weight.pb16i is 50,135,040 B on disk.
    check(q4nx_pb16i::w_bytes(5120, 17408) == 50135040u, "gate/up pb16i image is 50,135,040 B", 5120,
          17408);
    check(q4nx_t2s::w_bytes(5120, 17408) == 23674880u, "gate/up t2s stream is 23,674,880 B", 5120,
          17408);
}

void test_refusals() {
    check(!q4nx_t2s::fits(5120, 24), "N must be whole 32-column tiles", 5120, 24);
    check(!q4nx_t2s::fits(5000, 17408), "K must be whole 256-row tiles", 5000, 17408);
    check(!q4nx_t2s::fits(0, 17408) && !q4nx_t2s::fits(5120, 0), "no empty geometry", 0, 0);
    // N%32 is what t2s allows that pb16i does not: its tile owns 32 columns, not 1,024 -- that is
    // why a k/v projection of 256 columns has a stream here.
    check(!q4nx_pb16i::fits(5120, 256) && q4nx_t2s::fits(5120, 256), "t2s N unit is 32, not 1,024",
          5120, 256);
}

}  // namespace

int main() {
    test_tile();
    test_sizes();
    test_refusals();
    if (failures) {
        std::printf("npu_t2s_layout_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("npu_t2s_layout_test: all checks passed\n");
    return 0;
}
