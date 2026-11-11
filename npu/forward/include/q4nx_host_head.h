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
// q4nx_host_head.h -- one lm-head tile as exact integer sums on the host.
//
// A decode step sends one activation row through every tile of the lm head. On the device a launch
// costs the same weight pass for 1 row as for 128, so a one-row head is the most expensive way to
// use it: on a 248,320 x 2,048 head, 21 launches were 65 ms of each decode token, and the same sums
// on the host take about 6 ms (engine#1785). The sums are integers -- int8 activation codes times
// int8 weight codes into int32 -- so the host result is not an approximation of the device's: it is
// the same number, and the caller's epilogue (activation scale x per-column scale) is unchanged.
//
// The image is the one the device reads (q4nx_weight_layout.h: blocked microtiles or padded
// row-major, KD x ND bytes), so nothing is packed twice and no second copy of the head is held.
#ifndef Q4NX_HOST_HEAD_H
#define Q4NX_HOST_HEAD_H

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "q4nx_weight_layout.h"

namespace q4nx_host_head {

#if defined(__GNUC__) && !defined(__clang__) && (defined(__x86_64__) || defined(__i386__))
#define Q4NX_HOST_HEAD_CLONES __attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
#else
#define Q4NX_HOST_HEAD_CLONES
#endif

// out[j] = sum_{k < K} a8[k] * image(k, j) for j < ncols, exact int32.
//   image   KD x ND bytes in the given layout; rows k >= K and columns j >= ncols are not read
//   blocked true = the blocked microtile layout (needs q4nx_layout::blocked_fits(KD, ND))
// Returns false, writing nothing, when the geometry has no such image (K > KD, ncols > ND, or a
// blocked image that is not whole tiles).
Q4NX_HOST_HEAD_CLONES
static bool tile(const int8_t* image, int KD, int ND, bool blocked, const int8_t* a8, int K, int ncols,
                 int32_t* out) {
    if (!image || !a8 || !out || K <= 0 || ncols <= 0 || K > KD || ncols > ND) return false;
    if (blocked && !q4nx_layout::blocked_fits(KD, ND)) return false;
    if (!blocked) {
        // Row-major: one axpy per activation code over that weight row.
        std::memset(out, 0, (size_t)ncols * sizeof(int32_t));
        for (int k = 0; k < K; k++) {
            const int32_t a = a8[k];
            if (a == 0) continue;
            const int8_t* row = image + (size_t)k * (size_t)ND;
            for (int j = 0; j < ncols; j++) out[j] += a * (int32_t)row[j];
        }
        return true;
    }
    // Blocked: n-tile-major, k-tiles inside, and 8x8 microtiles (row r = k, column c = n) inside a
    // 64x128 tile. The walk below is the image's own byte order, so the head is read front to back.
    const int ktiles = KD / 64;
    for (int nt = 0; nt * 128 < ncols; nt++) {
        int32_t acc[128];
        std::memset(acc, 0, sizeof acc);
        for (int ki = 0; ki < ktiles && ki * 64 < K; ki++) {
            const int8_t* tp = image + ((size_t)nt * (size_t)ktiles + (size_t)ki) * 8192;
            for (int kb = 0; kb < 8; kb++) {
                const int k0 = ki * 64 + kb * 8;
                if (k0 >= K) break;
                int32_t a[8];
                for (int r = 0; r < 8; r++) a[r] = (k0 + r < K) ? (int32_t)a8[k0 + r] : 0;
                const int8_t* mp = tp + (size_t)kb * 16 * 64;
                for (int nb = 0; nb < 16; nb++, mp += 64) {
                    int32_t* dst = acc + nb * 8;
                    for (int r = 0; r < 8; r++) {
                        const int32_t ar = a[r];
                        const int8_t* w = mp + r * 8;
                        for (int c = 0; c < 8; c++) dst[c] += ar * (int32_t)w[c];
                    }
                }
            }
        }
        const int j0 = nt * 128;
        const int n = (ncols - j0 < 128) ? ncols - j0 : 128;
        std::memcpy(out + j0, acc, (size_t)n * sizeof(int32_t));
    }
    return true;
}

}  // namespace q4nx_host_head

#endif  // Q4NX_HOST_HEAD_H
