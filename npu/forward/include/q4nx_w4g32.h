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

// q4nx_w4g32.h -- the "w4g32" weight image: the large-M int4-weight prefill GEMM's contract.
//
// What it carries.  A q4_1 tile grid whose zero-points are all -8 * scale (every 2-D projection
// of bonsai2-27b-NPU, checked bit-exact on 2026-11-15) is exactly
//
//    w[k][n] = c[k][n] * s[k/32][n],   c = code - 8 in [-8, 7],  s a signed bf16 per 32 K.
//
// The image keeps c exactly and carries s as a signed int8 magnitude per (32-K group, column) times
// one float per column:
//
//    s[g][n] ~= mag[g][n] * cs[n],  cs[n] = max_g |s[g][n]| / MAGMAX(K),  mag = rint(s / cs)
//
// The device sums sum_g mag[g][n] * sum_{k in g} a[m][k] * c[k][n] exactly (int8 x int4 into acc32 per
// 32-K group, |P| <= 32512 < 2^15, then int16 x int16 into acc32, then int32 C), and the host applies
// C_f[m][n] = C[m][n] * ascale[m] * cs[n] -- the int8 path's own dequantisation with cs as the column
// scale.  MAGMAX(K) keeps |C| <= K * 127 * 8 * MAGMAX < 2^31: 127 up to K = 6144 (6,605 levels short of
// overflow at 5120), 121 at K = 17408.  The only rounding before the host's is rint(s / cs): the worst
// element error is |c| * cs / 2 <= 8 * max|s| / (2 * MAGMAX), the same bound as the int8 per-column image
// (round(w * 127 / amax), amax = 8 * max|s|) when MAGMAX = 127, and the RMS error is about half of it
// (measured 0.45-0.48x on bonsai2-27b layers 0 and 3).
//
// Image geometry (npu-kernels npu/arena/GU-w4/layout_w4.py, the b4 layout, k tile 128).  K % 128 == 0 and
// N % 128 == 0 (the design additionally needs N % 1024 == 0).  One tile per (nt = n/128, ki = k/128),
// at byte (nt * K/128 + ki) * 8704:
//
//   [0, 8192)     int4 codes, [kb 8][nb 8][16 r][16 cc]: element (k, n) with kb = (k%128)/16, r = k%16,
//                 nb = (n%128)/16, cc = n%16; e = r*16 + cc of its 16x16 block, byte block*128 + e/2,
//                 even e in the low nibble, two's complement
//   [8192, 8704)  int8 mag, [nb 8][g 4][16 cc], g = (k%128)/32
//
// Activations are int8 with one scale per row, host-packed per 32-row tile (the coalesced design's A):
//   A_packed[rt][ki][mb][kb][r][cc] = A[rt*32 + mb*4 + r][ki*128 + kb*16 + cc]
// (rt = m/32, mb = 4-row block, r = row in it, kb = 16-K block, cc = k in it).  C is int32 row-major M x N.
//
// The packer's input is the int8 path's own assembled float B (no second scale stream): factor() finds an exact
// code x per-32 scale form of it or refuses.
//
// This header is the contract between the packer, the A pack, the host dequantisation and the kernel's
// tap (mm_g32h.cc: codes at w + (kb*8+nb)*128, mags at w + 8192 + (nb*4+g)*16, A at (mb*8+kb)*64).
// tests/npu_w4g32_layout_test.cpp restates it independently and pins it without a device.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_OPENMP)
#define Q4NX_W4G32_PAR _Pragma("omp parallel for schedule(static)")
#else
#define Q4NX_W4G32_PAR
#endif

namespace q4nx_w4g32 {

constexpr int KT = 128;            // K rows per tile (one kernel call)
constexpr int NT = 128;            // N columns per tile
constexpr int GROUP = 32;          // K rows per magnitude
constexpr int CODE_BYTES = KT * NT / 2;           // 8192
constexpr int MAG_BYTES = (KT / GROUP) * NT;      // 512
constexpr int TILE = CODE_BYTES + MAG_BYTES;      // 8704
constexpr int MT = 32;             // activation rows per A tile (per core row)

inline bool fits(int K, int N) { return K > 0 && N > 0 && K % KT == 0 && N % NT == 0; }
inline size_t w_bytes(int K, int N) { return (size_t)(K / KT) * (size_t)(N / NT) * TILE; }

// Largest |mag| that keeps the device's int32 C exact for this K: K * 127 * 8 * MAGMAX < 2^31.
inline int magmax(int K) {
    if (K <= 0) return 0;
    const long long m = (long long)2147483647 / ((long long)K * 127 * 8);
    return m >= 127 ? 127 : (int)m;
}

struct offsets {
    size_t code;   // byte holding the element's nibble
    bool high;     // the HIGH nibble of that byte
    size_t mag;    // byte of its int8 magnitude
};

// Where weight element (k, n) and its group magnitude live.  Undefined unless fits(K, N).
inline offsets index(int K, int N, int k, int n) {
    (void)N;
    const size_t tile = ((size_t)(n / NT) * (size_t)(K / KT) + (size_t)(k / KT)) * TILE;
    const int kk = k % KT, nn = n % NT;
    const int kb = kk / 16, r = kk % 16, nb = nn / 16, cc = nn % 16, g = kk / GROUP;
    const int e = r * 16 + cc;
    offsets o;
    o.code = tile + (size_t)(kb * 8 + nb) * 128 + (size_t)(e / 2);
    o.high = (e & 1) != 0;
    o.mag = tile + CODE_BYTES + (size_t)(nb * 4 + g) * 16 + (size_t)cc;
    return o;
}

// Factor an ASSEMBLED design matrix B[k][n] (the float the int8 path packs) into code[k][n] in [-8, 7] and
// one scale per (32-K group, column): B = code * scale exactly, element for element.  The representation the
// kernel needs is any exact one, not the container's own: for each group with a = max |B| the candidates are
// s = +-a/q for q = 8 .. 1, and the first under which every element is an exact in-range multiple is taken (a
// q4_1 z = -8s pack, a symmetric q4_0 pack and a ternary pack all have one; a/q is exact whenever the true
// scale is a/q, because the correctly rounded quotient of q*s by q is s).  An all-zero group gets scale 0.
// Returns false -- and the tier must not run on this weight -- when some group has no such scale.
inline bool factor_group(const float* B, size_t stride, int8_t* code, float* scale_out) {
    float a = 0.0f;
    for (int i = 0; i < GROUP; i++) {
        const float v = B[(size_t)i * stride];
        if (!std::isfinite(v)) return false;
        a = std::fmax(a, std::fabs(v));
    }
    if (a == 0.0f) {
        for (int i = 0; i < GROUP; i++) code[(size_t)i * stride] = 0;
        *scale_out = 0.0f;
        return true;
    }
    for (int q = 8; q >= 1; q--)
        for (int sg = 0; sg < 2; sg++) {
            const float s = sg ? -a / (float)q : a / (float)q;
            bool ok = true;
            for (int i = 0; i < GROUP && ok; i++) {
                const float v = B[(size_t)i * stride];
                const float c = std::nearbyint(v / s);
                ok = c >= -8.0f && c <= 7.0f && c * s == v;
            }
            if (!ok) continue;
            for (int i = 0; i < GROUP; i++) code[(size_t)i * stride] = (int8_t)std::nearbyint(B[(size_t)i * stride] / s);
            *scale_out = s;
            return true;
        }
    return false;
}

inline bool factor(const float* B, int K, int N, int8_t* code, float* scale) {
    if (!B || !code || !scale || K % GROUP || N <= 0) return false;
    int bad = 0;
    Q4NX_W4G32_PAR
    for (int g = 0; g < K / GROUP; g++) {
        int b = 0;
        for (int n = 0; n < N; n++)
            b |= !factor_group(B + (size_t)g * GROUP * N + n, (size_t)N, code + (size_t)g * GROUP * N + n,
                               scale + (size_t)g * N + n);
        if (b) {
#if defined(_OPENMP)
#pragma omp atomic write
#endif
            bad = 1;
        }
    }
    return bad == 0;
}

// Build the image and the column scales from code[k][n] (in [-8, 7]) and scale[g][n].  `img` must hold
// w_bytes(K, N).  Returns false on a bad geometry or a code out of range.
inline bool pack(const int8_t* code, const float* scale, int K, int N, uint8_t* img, float* cs) {
    if (!fits(K, N) || !code || !scale || !img || !cs) return false;
    const int mm = magmax(K), G = K / GROUP, KTn = K / KT, NTn = N / NT;
    Q4NX_W4G32_PAR
    for (int n = 0; n < N; n++) {
        float amax = 0.0f;
        for (int g = 0; g < G; g++) amax = std::fmax(amax, std::fabs(scale[(size_t)g * N + n]));
        cs[n] = amax > 0.0f ? amax / (float)mm : 1.0f;
    }
    int bad = 0;
    // One tile per iteration: tile (nt, ki) owns bytes [(nt*KTn + ki) * TILE, +TILE), so iterations never share
    // a byte and the image does not depend on the schedule.
    Q4NX_W4G32_PAR
    for (int t = 0; t < NTn * KTn; t++) {
        const int nt = t / KTn, ki = t % KTn;
        uint8_t* tile = img + (size_t)t * TILE;
        std::memset(tile, 0, TILE);
        int b = 0;
        for (int g = 0; g < KT / GROUP; g++)
            for (int nn = 0; nn < NT; nn++) {
                const int n = nt * NT + nn;
                long q = std::lrint(scale[(size_t)(ki * (KT / GROUP) + g) * N + n] / cs[n]);
                if (q > mm) q = mm;
                if (q < -mm) q = -mm;
                tile[CODE_BYTES + (nn / 16 * 4 + g) * 16 + nn % 16] = (uint8_t)(int8_t)q;
            }
        for (int kk = 0; kk < KT; kk++) {
            const int8_t* row = code + (size_t)(ki * KT + kk) * N + (size_t)nt * NT;
            const int kb = kk / 16, r = kk % 16;
            for (int nn = 0; nn < NT; nn++) {
                const int c = row[nn];
                b |= (c < -8 || c > 7);
                const int e = r * 16 + nn % 16;
                uint8_t& byte = tile[(kb * 8 + nn / 16) * 128 + e / 2];
                byte |= (e & 1) ? (uint8_t)((c & 0x0F) << 4) : (uint8_t)(c & 0x0F);
            }
        }
        if (b) {
#if defined(_OPENMP)
#pragma omp atomic write
#endif
            bad = 1;
        }
    }
    return bad == 0;
}

// Read element (k, n) back as (code, mag).
inline void read(const uint8_t* img, int K, int N, int k, int n, int* c, int* mag) {
    const offsets o = index(K, N, k, n);
    const int nib = o.high ? (img[o.code] >> 4) : (img[o.code] & 0x0F);
    *c = nib >= 8 ? nib - 16 : nib;
    *mag = (int)(int8_t)img[o.mag];
}

// The design's A: int8 rows a[m][ld] (row-major, the int8 path's own quantised rows) -> the packed tile
// order above.  M % 32 == 0, K % 128 == 0; `out` holds M*K bytes.
inline bool pack_a(const int8_t* a, int ld, int M, int K, int8_t* out) {
    if (!a || !out || M % MT || K % KT || ld < K) return false;
    size_t o = 0;
    for (int rt = 0; rt < M / MT; rt++)
        for (int ki = 0; ki < K / KT; ki++)
            for (int mb = 0; mb < MT / 4; mb++)
                for (int kb = 0; kb < KT / 16; kb++)
                    for (int r = 0; r < 4; r++) {
                        std::memcpy(out + o, a + (size_t)(rt * MT + mb * 4 + r) * ld + (size_t)(ki * KT + kb * 16), 16);
                        o += 16;
                    }
    return true;
}

// The device's integer result for one (m, n), from the image: the CPU reference the gate compares against.
inline int64_t ref_dot(const int8_t* a_row, const uint8_t* img, int K, int N, int n) {
    int64_t acc = 0;
    for (int g = 0; g < K / GROUP; g++) {
        int64_t p = 0;
        int mag = 0;
        for (int kk = 0; kk < GROUP; kk++) {
            int c;
            read(img, K, N, g * GROUP + kk, n, &c, &mag);
            p += (int64_t)a_row[g * GROUP + kk] * c;
        }
        acc += p * mag;
    }
    return acc;
}

}  // namespace q4nx_w4g32
