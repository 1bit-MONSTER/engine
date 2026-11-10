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

// npu_pad_head_rows_test.cpp -- how many rows the padded-head path dequantises, driven on the numbers that broke it.
//
// engine#1990 pads the lm_head to the q4_1 tile quantum at import (a vocabulary need not be a multiple of the tile's 32
// rows: CodeLlama-7b's is 32,016 -> 32,032) and the reader then has to dequantise the TENSOR's grid while keeping only
// the vocabulary's rows. The first version gated that on `shape[1] == kpad_`, which is true for a `[rows, kpad_]` head and
// false for the `[tiles, tile bytes]` head the importer writes -- so it fell back to `NV` rows and `dequant_any()` refused a
// well-formed head with "lm_head dequant failed". CodeLlama-7b: `I8 [16016, 6144]`, kpad 4096, NV 32,016.
//
// The arithmetic is pure, so every case here runs with no container, no device and no pack.
#include "pad_head_rows.h"   // the rule, header-only: no library, so CI can build this

#include <cstdint>
#include <cstdio>
#include <cstdlib>


static int failures = 0;

static void check(bool ok, const char* what, long long got, long long want) {
    if (ok) {
        std::printf("ok    %s\n", what);
        return;
    }
    std::printf("FAIL  %s (got %lld, want %lld)\n", what, got, want);
    ++failures;
}

int main() {
    // The shape this path exists for: 16,016 tile-rows of 6,144 B, K grid 4,096, vocabulary 32,016.
    {
        const int r = padded_head_rows(16016, 6144, 4096, 32016);
        check(r == 32032, "the [tiles, tile bytes] head dequantises its PADDED grid (32,016 -> 32,032)", r, 32032);
        check(r % Q4NX_TILE_ROWS == 0, "and that grid is whole 32-row tiles", r % Q4NX_TILE_ROWS, 0);
    }
    // The row-grid spelling of the same tensor: shape[0] is again tile-rows, so the answer must not change.
    {
        const int r = padded_head_rows(16016, 4096, 4096, 32016);
        check(r == 32032, "the [rows, kpad_] spelling gives the same row count", r, 32032);
    }
    // A head that needs no padding keeps the vocabulary's own count.
    {
        const int r = padded_head_rows(1000, 4096, 4096, 32000);
        check(r == 32000, "an exactly-aligned vocabulary stays at NV", r, 32000);
    }
    // A K grid that is not a whole number of tile columns cannot be walked, so NV is the honest answer (the caller then
    // refuses, which is what happened before the fix -- and is correct for a grid this function does not understand).
    {
        const int r = padded_head_rows(16016, 6144, 4095, 32016);
        check(r == 32016, "a kpad that is not whole tile columns falls back to NV", r, 32016);
    }
    // A tensor narrower than the vocabulary is not a padded head; the caller must not be told to read past its rows.
    {
        const int r = padded_head_rows(500, 6144, 4096, 32016);
        check(r == 32016, "a grid smaller than NV falls back to NV rather than a short read", r, 32016);
    }
    // A partial block (shape[0] not a multiple of tiles-per-block) is not a grid this function can walk.
    {
        const int r = padded_head_rows(16001, 6144, 4096, 32016);
        check(r == 32016, "a partial block falls back to NV", r, 32016);
    }
    // NV <= 0 is a caller bug, and returning NV keeps the failure at the caller's check rather than here.
    {
        const int r = padded_head_rows(16016, 6144, 4096, 0);
        check(r == 0, "a non-positive vocabulary is passed through, not invented", r, 0);
    }

    if (failures == 0) std::printf("all padded-head row checks passed\n");
    return failures == 0 ? 0 : 1;
}
