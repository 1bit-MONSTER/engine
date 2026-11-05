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

// npu_blocked_layout_test -- the device weight layout contract (q4nx_weight_layout).
//
// The engine packs weights into a buffer the kernels read with a fixed tap, so the
// mapping between "weight element (i,j)" and "byte in the device buffer" is a
// contract between three places: this header, the packer (packB_blocked ->
// bpos()), and the microkernel's B indexing.  A mistake in it is silent: the GEMM
// simply multiplies the wrong pairs.  This test pins the contract without a device:
//
//   1. the mapping is a bijection onto [0, KD*ND) for every geometry,
//   2. each 64x128 tile occupies one contiguous 8192-byte window, and the tiles are
//      laid out k-fastest (n-tile-major), which is what makes a core's k-loop a
//      forward walk,
//   3. inside a tile, element (i,j) sits where the int8 microkernel reads it,
//      b_in[((kb*(N/8) + nb)*8 + r)*8 + c] with kb = (i%64)/8, r = i%8,
//      nb = (j%128)/8, c = j%8 -- the independent restatement of
//      mm_kernel_reference.cc's "B LAYOUT (silicon-verified)" note,
//   4. geometries that are not a whole number of 64x128 tiles are refused,
//   5. the row-major layout is unchanged (it must keep serving every container
//      built before the blocked one).
#include "q4nx_weight_layout.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what, int ki = 0, int ni = 0) {
    if (!ok) {
        std::printf("FAIL %s (K=%d N=%d)\n", what, ki, ni);
        failures++;
    }
}

struct Geom { int K, N; };
constexpr std::array<Geom, 6> kGeoms = {{{64, 128}, {128, 256}, {192, 384}, {512, 1280},
                                         {1088, 512}, {17408, 5120}}};

void test_bijection(Geom g) {
    std::vector<char> seen((size_t)g.K * (size_t)g.N, 0);
    for (int i = 0; i < g.K; i++)
        for (int j = 0; j < g.N; j++) {
            const size_t p = q4nx_layout::blocked_index((size_t)i, (size_t)j, g.K, g.N);
            if (p >= seen.size()) { check(false, "index outside the buffer", g.K, g.N); return; }
            if (seen[p]) { check(false, "two elements share a byte", g.K, g.N); return; }
            seen[p] = 1;
        }
    for (char c : seen) if (!c) { check(false, "a byte is never written", g.K, g.N); return; }
}

void test_tile_windows(Geom g) {
    const int KT = g.K / 64;
    // every element of tile (nt,ki) must land in [tile*8192, tile*8192 + 8192)
    for (int nt = 0; nt < g.N / 128; nt++)
        for (int ki = 0; ki < KT; ki++) {
            const size_t tile = (size_t)nt * KT + ki;
            for (int r = 0; r < 64; r += 7)
                for (int c = 0; c < 128; c += 13) {
                    const size_t i = (size_t)ki * 64 + r, j = (size_t)nt * 128 + c;
                    const size_t p = q4nx_layout::blocked_index(i, j, g.K, g.N);
                    if (p / 8192 != tile || p % 8192 >= 8192) {
                        check(false, "element outside its tile window", g.K, g.N);
                        return;
                    }
                }
        }
    // tiles are k-fastest: tile index grows by 1 as ki grows, and jumps by KT
    // between consecutive n-tiles
    const size_t a = q4nx_layout::blocked_index(0, 0, g.K, g.N) / 8192;
    const size_t b = q4nx_layout::blocked_index(64, 0, g.K, g.N) / 8192;
    const size_t c = q4nx_layout::blocked_index(0, 128, g.K, g.N) / 8192;
    check(b == a + 1, "k-tiles are not consecutive", g.K, g.N);
    check(c == a + (size_t)KT, "n-tile stride is not KT tiles", g.K, g.N);
}

void test_microkernel_order(Geom g) {
    // The independent restatement of the microkernel's own indexing.
    for (int i = 0; i < g.K; i += 5)
        for (int j = 0; j < g.N; j += 11) {
            const size_t kb = (size_t)(i % 64) / 8, r = (size_t)i % 8;
            const size_t nb = (size_t)(j % 128) / 8, c = (size_t)j % 8;
            const size_t tile = (size_t)(j / 128) * (g.K / 64) + (size_t)(i / 64);
            const size_t want = tile * 8192 + ((kb * 16 + nb) * 8 + r) * 8 + c;
            if (q4nx_layout::blocked_index((size_t)i, (size_t)j, g.K, g.N) != want) {
                check(false, "microkernel tile order does not match", g.K, g.N);
                return;
            }
        }
}

void test_refusals() {
    check(!q4nx_layout::blocked_fits(100, 128), "K=100 is not a whole number of 64-row tiles");
    check(!q4nx_layout::blocked_fits(64, 100), "N=100 is not a whole number of 128-col tiles");
    check(q4nx_layout::blocked_fits(17408, 5120), "the 27B D geometry should be packable");
    check(q4nx_layout::blocked_fits(5120, 17408), "the 27B GU geometry should be packable");
}

void test_row_major_unchanged() {
    // The pre-blocked layout every earlier container uses: element (i,j) at
    // i*ND + j, tiles irrelevant.
    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 128; j++)
            check(q4nx_layout::rowmajor_index((size_t)i, (size_t)j, 64, 128) ==
                      (size_t)i * 128 + (size_t)j, "row-major index moved");
}

// pack_int8_image() writes a whole weight image at microtile granularity (it is what
// onnx/ort/onebit_npu_gemm.cpp re-runs on every call whose weight is not the one cached
// on that kernel, so its cost is on the decode path).  Its contract is the scatter:
// the same bytes as writing one element at a time through the index function, and zero
// everywhere the logical K x N matrix does not reach.  Compared byte for byte, with
// ragged K/N edges and with KD/ND padded past them, in both layouts.
void test_pack_matches_scatter(int K, int N, int KD, int ND) {
    std::vector<int8_t> src((size_t)K * (size_t)N);
    for (size_t x = 0; x < src.size(); x++)  // mixed signs, zeros and +-127: a misplaced byte shows
        src[x] = (int8_t)((int)((x * 37u + 11u) % 255u) - 127);
    std::vector<int8_t> want((size_t)KD * (size_t)ND, 0), got((size_t)KD * (size_t)ND, 0);

    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            want[q4nx_layout::blocked_index((size_t)i, (size_t)j, KD, ND)] = src[(size_t)i * N + j];
    q4nx_layout::pack_int8_image(got.data(), src.data(), K, N, KD, ND, /*blocked=*/true);
    for (size_t x = 0; x < want.size(); x++)
        if (want[x] != got[x]) {
            std::printf("FAIL blocked pack differs from the scatter at byte %zu (K=%d N=%d KD=%d ND=%d): "
                        "%d, scatter says %d\n",
                        x, K, N, KD, ND, (int)got[x], (int)want[x]);
            failures++;
            return;
        }

    std::fill(want.begin(), want.end(), 0);
    std::fill(got.begin(), got.end(), 0);
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            want[q4nx_layout::rowmajor_index((size_t)i, (size_t)j, KD, ND)] = src[(size_t)i * N + j];
    q4nx_layout::pack_int8_image(got.data(), src.data(), K, N, KD, ND, /*blocked=*/false);
    for (size_t x = 0; x < want.size(); x++)
        if (want[x] != got[x]) {
            std::printf("FAIL row-major pack differs from the scatter at byte %zu (K=%d N=%d KD=%d ND=%d): "
                        "%d, scatter says %d\n",
                        x, K, N, KD, ND, (int)got[x], (int)want[x]);
            failures++;
            return;
        }
}

void test_pack_geometries() {
    // Whole tiles (the shapes the EP configures), padded past the logical matrix, and
    // ragged on both axes (the 8- and 64/128-tile edges that the fast path skips over).
    const int cases[][4] = {{64, 128, 64, 128},   {192, 384, 192, 384}, {1024, 2048, 1024, 2048},
                            {6, 6, 64, 128},      {100, 200, 128, 256}, {40, 7, 64, 128},
                            {1024, 1024, 1024, 1024}};
    for (const auto& c : cases) test_pack_matches_scatter(c[0], c[1], c[2], c[3]);
}

}  // namespace

int main() {
    for (Geom g : kGeoms) {
        test_bijection(g);
        test_tile_windows(g);
        test_microkernel_order(g);
    }
    test_refusals();
    test_row_major_unchanged();
    test_pack_geometries();
    if (failures) {
        std::printf("npu_blocked_layout_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("npu_blocked_layout_test: OK (bijection, tile windows, microkernel order, "
                "refusals, row-major, pack == scatter)\n");
    return 0;
}
