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

// engine#1990: how many rows to dequantise for the lm_head, given the tensor's OWN grid and the container's K grid.
// A container stores the head either as [rows, kpad_] or as [tiles, tile bytes]; both count TILE-ROWS in shape[0], so the
// number of 32-row blocks is shape[0] / (kpad / 256) either way. A vocabulary need not be a multiple of 32
// (CodeLlama-7b's is 32,016), so the importer pads to 32,032 and this returns the PADDED row count; the caller decodes
// that many rows and keeps NV of them, because the padded rows decode to zero and are never sampled. Pure, so the
// numbers are drivable without a container (tests/npu_pad_head_rows_test.cpp).
inline int padded_head_rows(int64_t shape0, int64_t shape1, int kpad, int NV) {
    (void)shape1;   // both spellings count tile-rows in shape[0]; the width only says which spelling it is
    if (kpad <= 0 || kpad % Q4NX_TILE_COLS != 0 || shape0 <= 0 || NV <= 0) return NV;
    const int64_t per_block = kpad / Q4NX_TILE_COLS;   // tiles per 32-row block
    if (per_block <= 0 || shape0 % per_block != 0) return NV;
    const int64_t r = (shape0 / per_block) * Q4NX_TILE_ROWS;
    if (r < NV || r % Q4NX_TILE_ROWS != 0) return NV;
    return int(r);
}

