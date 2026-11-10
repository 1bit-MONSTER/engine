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
// q4nx_weight_layout.h -- where a weight element lives in the device buffer.
//
// Two layouts exist, chosen per container by `q4nx_weight_layout` in the model
// directory's config.json:
//
//   "rm" (default)  the padded row-major [KD][ND] int8 image every container
//                   built before this had.
//   "b8"            the blocked image: 64x128 tiles, each stored in the
//                   [kb][nb][8][8] microtile order the int8 microkernel already
//                   indexes (mm_kernel_reference.cc, "B LAYOUT silicon-verified"),
//                   with the tiles laid out n-tile-major so a core's k-loop walks
//                   forward through memory.  Each tile is then ONE contiguous
//                   8192-byte DMA burst instead of 64 runs of 128 contiguous bytes
//                   with a row stride of ND.  Measured on the 27B GEMM phases; the
//                   numbers are in the kernel records, not here.
//
// The codes, the per-column scales and the arithmetic are identical in both
// layouts: this is an address permutation, so the int32 result is bit-identical.
// That is why the layout is a container property and not a kernel change.
//
// The mapping lives here rather than inside the runtime header so it can be
// tested without a device (tests/npu_blocked_layout_test.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace q4nx_layout {

// Element (i,j) of the logical K x N matrix, in a buffer of KD x ND bytes.
// KD must be a multiple of 64 and ND a multiple of 128 (the tile shape); callers
// check that before packing, because a geometry that is not a whole number of
// tiles has no blocked mapping.
inline size_t blocked_index(size_t i, size_t j, int KD, int ND) {
    (void)ND;
    const size_t nt = j / 128, ki = i / 64;                 // which tile
    const size_t kb = (i % 64) / 8, r = i % 8;               // row inside the tile
    const size_t nb = (j % 128) / 8, c = j % 8;              // column inside the tile
    const size_t tile = nt * (size_t)(KD / 64) + ki;         // n-tile-major
    return tile * 8192 + (((kb * 16 + nb) * 8 + r) * 8 + c);  // microtile order
}

// The same element in the padded row-major image.
inline size_t rowmajor_index(size_t i, size_t j, int KD, int ND) {
    (void)KD;
    return i * (size_t)ND + j;
}

// Bytes a blocked image of KD x ND occupies: ceil(ND/128) tiles across, ceil(KD/64) rows of tiles, 8192 B per tile.
// engine#1994: a caller that allocates KD*ND and indexes with blocked_index() reads and writes past the buffer whenever
// the geometry is not 128/64-aligned -- the two expressions are each correct on their own, which is why it survived review:
// `150144 % 8192 = 2688 = 21 * 128` and `11008 = 86 * 128`, so every measured shape was aligned.
inline size_t blocked_bytes(int KD, int ND) {
    const size_t nt = ((size_t)ND + 127) / 128, kt = ((size_t)KD + 63) / 64;
    return nt * kt * 8192;
}

inline bool blocked_fits(int KD, int ND) {
    return KD > 0 && ND > 0 && KD % 64 == 0 && ND % 128 == 0;
}

// Copy an int8 K x N weight (row-major, K rows of N bytes) into a KD x ND image
// in the given layout; bytes outside the logical K x N matrix come out zero.
//
// The permutation is the one above, done at the layout's own granularity instead of
// one byte per index call: inside a blocked tile, element (i0+r, j0+c) of the 8x8
// microtile that starts at (i0, j0) sits at blocked_index(i0, j0) + r*8 + c -- both
// tile and microtile are unchanged for r,c < 8 -- so a microtile is eight 8-byte runs,
// one per source row, rather than 64 scattered bytes, and the destination address is
// computed 64x less often. The bytes written are identical to the element-at-a-time
// scatter; tests/npu_blocked_layout_test.cpp compares the two byte for byte, including
// ragged edges and padded KD/ND.
//
// This exists because a caller that cannot keep the packed weight resident re-packs on every call,
// and one byte-at-a-time store per element is how that was written. Measured on a 1024x2048 int8 weight
// (onnx/ort is such a caller: it keys its kernels by (K, N), so a model pass sends 28 weights of one
// shape through the same kernel in a row): the blocked scatter takes 2.09 ms and this takes 0.20 ms; in
// the row-major layout the EP's kernels actually use, the old loop was already a row copy the compiler
// vectorised (0.12 ms) and this takes 0.09 ms. So the win is in the container path's blocked images and
// the EP's is a rounding error -- the EP's re-pack cost is its BO writeback, not this loop.
inline void pack_int8_image(int8_t* dst, const int8_t* src, int K, int N, int KD, int ND,
                            bool blocked) {
    std::memset(dst, 0, (size_t)KD * (size_t)ND);
    if (!blocked) {
        for (int i = 0; i < K; i++)
            std::memcpy(dst + rowmajor_index((size_t)i, 0, KD, ND), src + (size_t)i * N, (size_t)N);
        return;
    }
    if (!blocked_fits(KD, ND)) return;  // no mapping exists; callers check this before packing
    for (int i0 = 0; i0 < K; i0 += 8) {
        const int rows = (K - i0 < 8) ? K - i0 : 8;
        for (int j0 = 0; j0 < N; j0 += 8) {
            const size_t base = blocked_index((size_t)i0, (size_t)j0, KD, ND);
            const int cols = (N - j0 < 8) ? N - j0 : 8;
            if (cols == 8) {
                for (int r = 0; r < rows; r++)
                    std::memcpy(dst + base + (size_t)r * 8, src + (size_t)(i0 + r) * N + j0, 8);
            } else {
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++)
                        dst[base + (size_t)r * 8 + c] = src[(size_t)(i0 + r) * N + j0 + c];
            }
        }
    }
}

}  // namespace q4nx_layout
