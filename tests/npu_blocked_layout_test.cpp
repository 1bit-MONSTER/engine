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
#include "q4nx_pack.h"
#include "q4nx_host_head.h"
#include "q4nx_weight_layout.h"

#include <algorithm>
#include <array>
#include <cmath>
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


// ── the production packer, now that it is reachable (engine #1416) ────────────────────────────────────────────────
// q4nx_pack_B_int8_blocked is the body I8Ctx::packB_blocked runs on every weight. Two things are asserted here, neither of
// which any test could check while it lived inside a header that pulls XRT: that its scales and codes equal the row-major
// reference's (q4nx_pack_B_int8), and that its blocked output is exactly the row-major output permuted by bpos -- which is
// the layout contract the microkernel indexes through.
void test_production_packer() {
    const int K = 64, N = 256, KD = 64, ND = 256;
    std::vector<float> w((size_t)K * N);
    for (size_t i = 0; i < w.size(); ++i) w[i] = float((i * 2654435761u) % 2000) * 1e-3f - 1.0f;
    w[(size_t)3 * N + 7] = 0.0f / 0.0f;                  // a NaN, which both paths must zero
    for (int j = 100; j < 164; ++j) w[(size_t)5 * N + (size_t)j] = 0.0f;   // an all-zero column block: the amax floor

    std::vector<int8_t> rowmaj((size_t)K * N), blocked((size_t)K * N);
    std::vector<float> sc_ref((size_t)N), sc_blk((size_t)N);
    const float mean_ref = q4nx_pack_B_int8(w.data(), K, N, rowmaj.data(), N, sc_ref.data());
    double ssum = 0;
    const int rc = q4nx_pack_B_int8_blocked(blocked.data(), w.data(), K, N, KD, ND, /*blocked=*/1, sc_blk.data(), &ssum);
    check(rc == 0, "production packer: runs");
    check(sc_blk == sc_ref, "production packer: every column scale equals the row-major reference");
    check(std::fabs(float(ssum) - mean_ref * N) < 1e-3f, "production packer: the scale sum is the reference's mean x N");

    size_t mismatched = 0;
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j) {
            const size_t dst = q4nx_layout::blocked_index((size_t)i, (size_t)j, KD, ND);
            if (blocked[dst] != rowmaj[(size_t)i * N + (size_t)j]) mismatched++;
        }
    check(mismatched == 0, "production packer: the blocked image is the row-major image permuted by bpos");
    std::printf("  production packer: %zu codes, %d scales, blocked == bpos(rowmajor)\n", (size_t)K * N, N);
}

// engine#1783: the production packer's rounding is written so it vectorises (truncate, then step on the fraction), and
// that has to be roundf's half-away-from-zero exactly. Columns whose amax is 127 make the scale exactly 1, so each value
// below reaches the rounding unchanged: every k + 0.5 and k - 0.5 tie, the floats one ulp either side of a tie, the
// +-127 clamp, and NaN / +-inf (zeroed). The row-major reference (q4nx_pack_B_int8, roundf) is the oracle.
void test_packer_rounding_edges() {
    const int K = 64, N = 128;
    std::vector<float> vals;
    for (int k = -127; k <= 126; ++k) {
        const float tie = float(k) + 0.5f;
        vals.push_back(tie);
        vals.push_back(std::nextafter(tie, 0.0f));
        vals.push_back(std::nextafter(tie, tie > 0 ? 1e9f : -1e9f));
    }
    for (float v : {0.0f, -0.0f, 0.49999997f, -0.49999997f, 127.0f, -127.0f, 126.99999f, -126.99999f, 1e-30f, -1e-30f})
        vals.push_back(v);
    vals.push_back(0.0f / 0.0f);
    vals.push_back(1.0f / 0.0f);
    vals.push_back(-1.0f / 0.0f);
    std::vector<float> w((size_t)K * N, 0.0f);
    for (int j = 0; j < N; ++j) w[(size_t)j] = (j % 2) ? 127.0f : -127.0f;   // row 0: amax 127 in every column
    size_t at = 0;
    for (int i = 1; i < K && at < vals.size(); ++i)
        for (int j = 0; j < N && at < vals.size(); ++j) w[(size_t)i * N + (size_t)j] = vals[at++];
    check(at == vals.size(), "packer rounding: every edge value placed");
    std::vector<int8_t> rowmaj((size_t)K * N), blocked((size_t)K * N);
    std::vector<float> sc_ref((size_t)N), sc_blk((size_t)N);
    q4nx_pack_B_int8(w.data(), K, N, rowmaj.data(), N, sc_ref.data());
    double ssum = 0;
    check(q4nx_pack_B_int8_blocked(blocked.data(), w.data(), K, N, K, N, 1, sc_blk.data(), &ssum) == 0, "packer rounding: runs");
    check(sc_blk == sc_ref, "packer rounding: scales equal the reference's");
    size_t mismatched = 0, ties_up = 0;
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j) {
            const int8_t got = blocked[q4nx_layout::blocked_index((size_t)i, (size_t)j, K, N)];
            if (got != rowmaj[(size_t)i * N + (size_t)j]) mismatched++;
            ties_up += w[(size_t)i * N + (size_t)j] == 2.5f && got == 3;
        }
    check(mismatched == 0, "packer rounding: every code equals roundf's, ties and neighbours included");
    check(ties_up == 1, "packer rounding: 2.5 rounds away from zero (3), not to even");
    std::printf("  packer rounding: %zu edge values, codes == roundf reference\n", vals.size());
}

// engine#1994: the integer host arm packs UNPADDED images (KD == K, ND == N) into K*N bytes. blocked_index() addresses
// whole 64x128 tiles, so a geometry that is not whole tiles has to be packed and read row-major, or both the packer and
// the MAC walk past the buffer. This pins the rule the arm uses -- blocked = blocked_fits(K, N), read through
// image_index() -- on the shapes that broke it, with guard bytes after the image to catch a write past K*N.
void test_unpadded_image(int K, int N, bool expect_blocked) {
    const bool blocked = q4nx_layout::blocked_fits(K, N);
    check(blocked == expect_blocked, "unpadded image: blocked only when K x N is whole tiles", K, N);
    std::vector<float> w((size_t)K * N);
    for (size_t i = 0; i < w.size(); ++i) w[i] = float((i * 2654435761u) % 2000) * 1e-3f - 1.0f;
    const size_t guard = size_t(1) << 18, img = (size_t)K * N;
    std::vector<int8_t> buf(img + guard, int8_t(0x5A)), rowmaj(img);
    std::vector<float> sc((size_t)N), sc_ref((size_t)N);
    q4nx_pack_B_int8(w.data(), K, N, rowmaj.data(), N, sc_ref.data());
    double ssum = 0;
    const int rc = q4nx_pack_B_int8_blocked(buf.data(), w.data(), K, N, K, N, blocked ? 1 : 0, sc.data(), &ssum);
    bool guard_ok = true;
    for (size_t g = img; g < img + guard; ++g) guard_ok &= buf[g] == int8_t(0x5A);
    size_t past = 0, mismatched = 0, forced_max = 0;
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j) {
            const size_t at = q4nx_layout::image_index((size_t)i, (size_t)j, K, N, blocked);
            if (at >= img) past++;
            else if (buf[at] != rowmaj[(size_t)i * N + (size_t)j]) mismatched++;
            forced_max = std::max(forced_max, q4nx_layout::blocked_index((size_t)i, (size_t)j, K, N));
        }
    check(rc == 0 && guard_ok, "unpadded image: the packer writes nothing past K*N", K, N);
    check(past == 0 && mismatched == 0, "unpadded image: every code reads back at image_index, inside K*N", K, N);
    // The mechanism, so the arm above cannot pass for the wrong reason: on an unaligned shape the blocked index the
    // arm used to force really does leave the buffer.
    if (!expect_blocked) check(forced_max >= img, "unpadded image: forced blocked_index would run past K*N", K, N);
    std::printf("  unpadded %dx%d: %s, forced blocked reaches %zu of %zu B\n", K, N, blocked ? "blocked" : "row-major",
                forced_max + 1, img);
}

// The host lm-head tile (q4nx_host_head.h, engine#1785): its sums are the element-at-a-time sums over
// image_index(), for both layouts, padded and ragged geometries, and activation codes at the int8 edges.
// The reference below shares nothing with the kernel but the index contract this file already pins.
void test_host_head_tile(int K, int N, int KD, int ND, bool blocked) {
    std::vector<int8_t> img((size_t)KD * (size_t)ND);
    uint32_t s = 0x1234567u + (uint32_t)(K * 31 + N);
    for (auto& b : img) { s = s * 1664525u + 1013904223u; b = (int8_t)(int)((s >> 24) % 255) - 127; }
    std::vector<int8_t> a((size_t)K);
    for (auto& v : a) { s = s * 1664525u + 1013904223u; v = (int8_t)((int)((s >> 24) % 255) - 127); }
    a[0] = 127; a[(size_t)K - 1] = -127;
    if (K > 2) a[1] = 0;
    std::vector<int32_t> want((size_t)N), got((size_t)N + 1, 0x5a5a5a5a);
    for (int j = 0; j < N; j++) {
        int64_t acc = 0;
        for (int k = 0; k < K; k++)
            acc += (int64_t)a[(size_t)k] * (int64_t)img[q4nx_layout::image_index((size_t)k, (size_t)j, KD, ND, blocked)];
        want[(size_t)j] = (int32_t)acc;
    }
    const bool ok = q4nx_host_head::tile(img.data(), KD, ND, blocked, a.data(), K, N, got.data());
    int bad = 0;
    for (int j = 0; j < N; j++) bad += got[(size_t)j] != want[(size_t)j];
    if (!ok || bad || got[(size_t)N] != 0x5a5a5a5a) {
        std::printf("FAIL host head tile K=%d N=%d KD=%d ND=%d %s: ok=%d, %d of %d sums differ, guard %s\n", K, N, KD, ND,
                    blocked ? "blocked" : "row-major", (int)ok, bad, N, got[(size_t)N] == 0x5a5a5a5a ? "kept" : "overwritten");
        failures++;
    }
}

void test_host_head_refusals() {
    std::vector<int8_t> img(64 * 128, 1), a(64, 1);
    std::vector<int32_t> out(128, 7);
    const bool r1 = q4nx_host_head::tile(img.data(), 64, 128, true, a.data(), 65, 128, out.data());   // K > KD
    const bool r2 = q4nx_host_head::tile(img.data(), 64, 128, true, a.data(), 64, 129, out.data());   // ncols > ND
    const bool r3 = q4nx_host_head::tile(img.data(), 60, 128, true, a.data(), 60, 128, out.data());   // not whole tiles
    if (r1 || r2 || r3 || out[0] != 7) {
        std::printf("FAIL host head tile: a geometry with no image was not refused (%d %d %d) or out was written\n", (int)r1, (int)r2, (int)r3);
        failures++;
    }
}

int main() {
    for (Geom g : kGeoms) {
        test_bijection(g);
        test_tile_windows(g);
        test_microkernel_order(g);
    }
    test_refusals();
    test_production_packer();
    test_packer_rounding_edges();
    test_row_major_unchanged();
    test_pack_geometries();
    test_unpadded_image(3072, 7488, false);   // phi-3-mini's last vocabulary block (32064 % 8192)
    test_unpadded_image(2048, 8512, false);   // a body width of 66.5 tiles
    test_unpadded_image(100, 128, false);     // K not a whole number of 64-row tiles
    test_unpadded_image(4096, 2688, true);    // the measured head block (150144 % 8192 = 21 tiles): stays blocked
    test_host_head_tile(2048, 7168, 2048, 7168, true);     // a whole blocked head tile
    test_host_head_tile(2048, 4672, 2048, 7168, true);     // the last tile of a vocabulary: fewer columns than ND
    test_host_head_tile(1000, 300, 1024, 384, true);       // padded K and N, ragged inside a microtile
    test_host_head_tile(1024, 7168, 1024, 7168, false);    // row-major
    test_host_head_tile(100, 77, 128, 130, false);         // row-major, padded and ragged
    test_host_head_refusals();
    if (failures) {
        std::printf("npu_blocked_layout_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("npu_blocked_layout_test: OK (bijection, tile windows, microkernel order, "
                "refusals, row-major, pack == scatter)\n");
    return 0;
}
