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
//                   with a row stride of ND, which measured 2.4-2.6x on every
//                   27B GEMM phase (3.32 -> 8.13 GB/s on the gate/up projection).
//
// The codes, the per-column scales and the arithmetic are identical in both
// layouts: this is an address permutation, so the int32 result is bit-identical.
// That is why the layout is a container property and not a kernel change.
//
// The mapping lives here rather than inside the runtime header so it can be
// tested without a device (tests/npu_blocked_layout_test.cpp).
#pragma once

#include <cstddef>

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

inline bool blocked_fits(int KD, int ND) {
    return KD > 0 && ND > 0 && KD % 64 == 0 && ND % 128 == 0;
}

}  // namespace q4nx_layout
