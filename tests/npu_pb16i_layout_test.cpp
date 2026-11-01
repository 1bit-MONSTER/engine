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

// npu_pb16i_layout_test -- the pb16i weight image contract (q4nx_weight_layout
// "pb16i"), checked without a device.
//
// The engine copies a pb16i sidecar into a weight BO unchanged, so nothing on the
// host re-derives the element order: if this header disagrees with the packer, the
// kernel multiplies the wrong pairs and the output is silently garbage.  This test
// is the independent restatement of the contract, taken from the kernel's own tap
// rather than from the header:
//
//   1. the image is K*N/2 + (K/32)*N*2 bytes and also tiles*4,608;
//   2. for a sample of tiles taken across the whole image, the (i,j) elements that
//      belong to a tile are exactly a 256-K x 32-N block, and they claim that tile's
//      4,096 code bytes exactly twice and its 256 ms entries exactly 32 times each --
//      i.e. the map is a bijection onto the image, with no byte shared between tiles;
//   3. inside a tile the code of (kk,nn) is at e = ((kb*2+nb)*16+rr)*16+cc, byte e/2,
//      even e in the low nibble two's complement, and its ms at 4096 + 2*((nb*8+g)*16+cc)
//      with g = kk/32;
//   4. an image built through index() reads back as code and ms, so a packer written
//      from this header and the kernel's tap agree element for element;
//   5. the side table (e_ref int16[N], mode byte) round-trips and refuses short tables
//      and unknown modes; the host's scale is one IEEE rounding;
//   6. geometries that are not whole tiles are refused.
#include "q4nx_pb16i.h"

#include <cstdio>
#include <vector>

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

// The 27B roles, plus shapes that stress each axis.
const geom GEOMS[] = {
    {5120, 17408},   // GU: gate / up
    {17408, 5120},   // D
    {5120, 10240},   // QKV (GDN fused q|k|v)
    {6144, 5120},    // O (ssm_out, attn_output)
    {5120, 6144},    // Z (attn_gate), and the full-attention q halves
    {5120, 1024},    // full-attention k / v
    {256, 1024},     // smallest legal
    {5120, 2048},
};

// Restated from the kernel tap: element e of a tile's code block.
int code_index(int kk, int nn) {
    const int kb = kk / 16, rr = kk % 16;
    const int nb = nn / 16, cc = nn % 16;
    return ((kb * 2 + nb) * 16 + rr) * 16 + cc;
}

// Check one tile exhaustively: its elements are a 256-K x 32-N block, they fill the
// tile's bytes as the tap says, and an image written through index() reads back.
void check_tile(int c, int ki, int nt, int r, int K, int N) {
    const int cw = N / q4nx_pb16i::COLS / q4nx_pb16i::ROWS;   // columns per core
    const int col0 = c * (N / q4nx_pb16i::COLS) + r * cw + nt * 32;
    const size_t tile = (((size_t)c * (size_t)(K / 256) + (size_t)ki) * (size_t)(cw / 32) +
                         (size_t)nt) * 4 + (size_t)r;
    const size_t base = tile * 4608;
    std::vector<unsigned> code_hits(4096, 0), ms_hits(256, 0);   // per byte / per int16 entry
    bool inside = true, tap_ok = true;
    for (int kk = 0; kk < 256; kk++) {
        for (int nn = 0; nn < 32; nn++) {
            const size_t i = (size_t)ki * 256 + (size_t)kk, j = (size_t)col0 + (size_t)nn;
            const q4nx_pb16i::offsets o = q4nx_pb16i::index(i, j, K, N);
            const size_t coff = o.code - base, moff = o.ms - base - 4096;
            if (o.code < base || o.code >= base + 4096) inside = false;
            if (o.ms < base + 4096 || o.ms + 1 >= base + 4608) inside = false;
            if (coff >= 4096 || moff >= 512) continue;
            if ((int)(coff * 2 + (o.high ? 1 : 0)) != code_index(kk, nn)) tap_ok = false;
            if ((int)(moff / 2) != ((nn / 16) * 8 + kk / 32) * 16 + (nn % 16)) tap_ok = false;
            code_hits[coff]++;
            ms_hits[moff / 2]++;
        }
    }
    bool bijective = true;
    for (unsigned h : code_hits) bijective = bijective && h == 2;
    for (unsigned h : ms_hits) bijective = bijective && h == 32;
    if (!inside) { check(false, "an element's bytes landed outside its tile", K, N); }
    if (!tap_ok) { check(false, "an element sits where the tap does not read it", K, N); }
    if (!bijective) { check(false, "a tile byte is not claimed by its own elements", K, N); }
}

void check_geometry(const geom& g) {
    const int K = g.K, N = g.N;
    if (!q4nx_pb16i::fits(K, N)) {
        std::printf("FAIL fits() refused a whole-tile geometry (K=%d N=%d)\n", K, N);
        failures++;
        return;
    }
    // 1
    const size_t bytes = q4nx_pb16i::w_bytes(K, N);
    check(bytes == (size_t)K * N / 2 + (size_t)(K / 32) * N * 2, "image size != codes + ms", K, N);
    check(bytes == q4nx_pb16i::tiles(K, N) * 4608, "image size != tiles * 4608", K, N);
    const size_t ntiles = (size_t)8 * (K / 256) * (size_t)((N / 32) / 32) * 4;
    check(ntiles == q4nx_pb16i::tiles(K, N), "tile count", K, N);

    // 2 + 3 + 4: the first and last tile of every core corner, plus a stride sweep.
    const int kt = K / 256, ntl = (N / 32) / 32;
    for (int c = 0; c < 8; c++)
        for (int r = 0; r < 4; r++) {
            check_tile(c, 0, 0, r, K, N);
            check_tile(c, kt - 1, ntl - 1, r, K, N);
            check_tile(c, kt / 2, ntl / 2, r, K, N);
        }

    // 5: a synthetic image over one tile, read back the way the host does.
    {
        std::vector<uint8_t> img(4608, 0);
        const int col0 = 0, ki = 1 % kt;
        for (int kk = 0; kk < 256; kk++)
            for (int nn = 0; nn < 32; nn++) {
                const size_t i = (size_t)ki * 256 + (size_t)kk, j = (size_t)col0 + (size_t)nn;
                const q4nx_pb16i::offsets o = q4nx_pb16i::index(i, j, K, N);
                const size_t coff = o.code % 4608, moff = o.ms % 4608;
                const int nib = (((kk + nn) % 3) - 1) & 0x0F;
                img[coff] = (uint8_t)(o.high ? ((img[coff] & 0x0F) | (nib << 4))
                                             : ((img[coff] & 0xF0) | nib));
                const int ms = 200 + (nn % 7) * 31;
                img[moff] = (uint8_t)(ms & 0xFF);
                img[moff + 1] = (uint8_t)(ms >> 8);
            }
        bool roundtrip = true;
        for (int kk = 0; kk < 256; kk++)
            for (int nn = 0; nn < 32; nn++) {
                const size_t i = (size_t)ki * 256 + (size_t)kk, j = (size_t)nn;
                const int want = ((kk + nn) % 3) - 1, want_ms = 200 + (nn % 7) * 31;
                const q4nx_pb16i::offsets o = q4nx_pb16i::index(i, j, K, N);
                const uint8_t b = img[o.code % 4608];
                const int got = o.high ? (b >> 4) : (b & 0x0F);
                if ((got >= 8 ? got - 16 : got) != want) roundtrip = false;
                int16_t msv;
                std::memcpy(&msv, &img[o.ms % 4608], sizeof(msv));
                if (msv != want_ms) roundtrip = false;
            }
        check(roundtrip, "image read-back != what was written", K, N);
    }

    // 5: the side table and the host's rounding.
    if (K == 5120 && N == 17408) {
        const int16_t e0 = -18, e1 = -12;
        std::vector<uint8_t> ref((size_t)N * 2 + 1, 0);
        for (int j = 0; j < N; j++) {
            const int16_t e = (j & 1) ? e1 : e0;
            ref[(size_t)j * 2] = (uint8_t)((uint16_t)e & 0xFF);
            ref[(size_t)j * 2 + 1] = (uint8_t)((uint16_t)e >> 8);
        }
        ref[(size_t)N * 2] = 1;
        std::vector<int16_t> eref;
        int mode = -1;
        const bool parsed = q4nx_pb16i::parse_ref(ref.data(), ref.size(), N, &eref, &mode);
        bool ref_ok = parsed && mode == 1 && (int)eref.size() == N;
        for (int j = 0; ref_ok && j < N; j++)
            ref_ok = eref[(size_t)j] == ((j & 1) ? e1 : e0);
        check(ref_ok, "side table round-trip", K, N);
        check(!q4nx_pb16i::parse_ref(ref.data(), ref.size() - 1, N, &eref, &mode),
              "a short side table must be refused", K, N);
        ref[(size_t)N * 2] = 2;
        check(!q4nx_pb16i::parse_ref(ref.data(), ref.size(), N, &eref, &mode),
              "an unknown mode byte must be refused", K, N);
        check(q4nx_pb16i::ref_bytes(N) == ref.size(), "side table size", K, N);

        check(q4nx_pb16i::scale_i64(3, -2) == 0.75f, "scale_i64(3,-2) != 0.75", K, N);
        check(q4nx_pb16i::scale_i64(-1, 0) == -1.0f, "scale_i64(-1,0) != -1", K, N);
        check(q4nx_pb16i::scale_i64(1, -30) == (float)std::ldexp(1.0, -30),
              "scale_i64 is not exact for a power of two", K, N);
        // The integer sums this format produces are below 2^33, so the double
        // intermediate is exact and the cast is the only rounding.
        const int32_t lo = 1234567, hi = -3;
        check(q4nx_pb16i::sum_of(&lo, &hi, 1) == (int64_t)hi * (1 << 21) + lo, "plane merge", K, N);
        check(q4nx_pb16i::sum_of(&lo, &hi, 0) == lo, "int32 mode ignores the high plane", K, N);
    }
}

}  // namespace

int main() {
    for (const geom& g : GEOMS) check_geometry(g);

    // 6: refusals.
    const geom bad[] = {{255, 1024}, {256, 1000}, {0, 1024}, {256, 0}};
    for (const geom& g : bad)
        check(!q4nx_pb16i::fits(g.K, g.N), "a non-whole-tile geometry must be refused", g.K, g.N);

    if (failures) {
        std::printf("npu_pb16i_layout_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("npu_pb16i_layout_test: ok (%zu geometries)\n", sizeof(GEOMS) / sizeof(GEOMS[0]));
    return 0;
}
