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

// pad_head_rows.h -- the padded-head row arithmetic, header-only so a test can drive it where CI configures.
//
// It lived as a free function in q4nx_forward.cpp, which meant the arm had to link `onebit_npu_forward` -- a library
// that only exists under `if(ONEBIT_NPU)`, so CI (which configures WITHOUT the NPU runtime) never built or ran the test.
// A regression arm nothing executes is the failure this whole file exists to avoid, so the rule moved here and the arm
// now sits outside the NPU guard.
#pragma once

#include "q4nx_dequant.h"

#include <cstdint>

// engine#1990: how many rows to dequantise for the lm_head, given the tensor's OWN shape and the container's K grid.
// A vocabulary need not be a multiple of the tile's 32 rows (CodeLlama-7b: 32,016; MiniCPM-V 4.7 1B: 248,094), so a
// container stores the head on the next 32-row boundary and this returns that PADDED row count; the caller decodes
// that many rows and keeps NV of them. Pure, so the numbers are drivable without a container
// (tests/npu_pad_head_rows_test.cpp).
//
// Three spellings of the shape exist, and each says where the row count is:
//   [tiles, tile bytes]              shape[1] is a chunk size. shape[0] counts tiles, kpad/256 of them per 32-row block.
//                                    Every imported container on the box (38 measured, review of #2026).
//   [row tiles, column tiles, bytes] three dimensions, shape[1] == kpad/256. shape[0] already IS the block count, and
//                                    need not divide by anything (FastFlowLM's own containers; MiniCPM-V 4.7 1B is
//                                    I8 [7753, 4, 8704], engine#1788).
//   [rows, kpad]                     shape[1] is the row width and not a chunk size. shape[0] is the row count itself.
// A width that is both the K grid and a chunk size (kpad 5,120) is read as tiles first, because that is what exists.
//
// One bound holds for all three and is what keeps a misread from becoming a long read: padding only ever reaches the
// next 32-row boundary, so an answer outside [NV, NV + 32) is not this tensor's row count. Reading [32032, 4096] as
// tiles gives 64,064, which fails the bound and falls through to the row spelling (review of #2026 at 63a47be).
// Anything the three spellings cannot place returns NV, and the caller's own dequant then refuses it by name.
inline bool head_chunk_bytes(int64_t b) {
    return b == Q4NX_TILE_BYTES || b == 6144 /* q4_1f, the imported-container tile */ || b == Q4NX_CHUNK_Q4K || b == Q4NX_CHUNK_Q8 || b == Q4NX_CHUNK_Q4S;
}
inline int padded_head_rows(const int64_t* shape, int ndim, int kpad, int NV) {
    if (!shape || kpad <= 0 || kpad % Q4NX_TILE_COLS != 0 || NV <= 0) return NV;
    const int64_t per_block = kpad / Q4NX_TILE_COLS;   // tiles per 32-row block
    const auto fits = [NV](int64_t rows) { return rows >= NV && rows < int64_t(NV) + Q4NX_TILE_ROWS && rows % Q4NX_TILE_ROWS == 0; };
    if (ndim == 3) {
        if (shape[0] <= 0 || shape[1] != per_block || !head_chunk_bytes(shape[2])) return NV;
        const int64_t rows = shape[0] * Q4NX_TILE_ROWS;
        return fits(rows) ? int(rows) : NV;
    }
    if (ndim != 2 || shape[0] <= 0) return NV;
    if (head_chunk_bytes(shape[1]) && shape[0] % per_block == 0) {
        const int64_t rows = (shape[0] / per_block) * Q4NX_TILE_ROWS;
        if (fits(rows)) return int(rows);
    }
    if (shape[1] == kpad && fits(shape[0])) return int(shape[0]);
    return NV;
}

