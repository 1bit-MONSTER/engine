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

static int rows2(int64_t s0, int64_t s1, int kpad, int NV) {
    const int64_t shape[3] = {s0, s1, 0};
    return padded_head_rows(shape, 2, kpad, NV);
}
static int rows3(int64_t s0, int64_t s1, int64_t s2, int kpad, int NV) {
    const int64_t shape[3] = {s0, s1, s2};
    return padded_head_rows(shape, 3, kpad, NV);
}

int main() {
    // The shape this path exists for: 16,016 tiles of 6,144 B, K grid 4,096, vocabulary 32,016.
    {
        const int r = rows2(16016, 6144, 4096, 32016);
        check(r == 32032, "the [tiles, tile bytes] head dequantises its PADDED grid (32,016 -> 32,032)", r, 32032);
        check(r % Q4NX_TILE_ROWS == 0, "and that grid is whole 32-row tiles", r % Q4NX_TILE_ROWS, 0);
    }
    // The same head in the 5,120-byte tile, as codellama7b-b8 stores it.
    {
        const int r = rows2(16016, 5120, 4096, 32016);
        check(r == 32032, "the same head in 5,120-byte tiles gives the same rows", r, 32032);
    }
    // The ROW spelling of that head (review of #2026 at 63a47be): the tensor has 32,032 rows of 4,096, so shape[0] is
    // the row count itself. Read as tiles it would be 64,064, twice what the tensor holds.
    {
        const int r = rows2(32032, 4096, 4096, 32016);
        check(r == 32032, "the [rows, kpad] spelling reads shape[0] as rows, not as tiles", r, 32032);
    }
    // ... and a row-spelled head that is NOT padded to the next boundary is not a padded head.
    {
        const int r = rows2(32064, 4096, 4096, 32016);
        check(r == 32016, "a row grid more than one block past the vocabulary falls back to NV", r, 32016);
    }
    // The three-dimensional spelling, FastFlowLM's own (engine#1788): [row tiles, column tiles, tile bytes]. 7,753 is odd,
    // so no rule that divides shape[0] by the 4 column tiles can place it.
    {
        const int r = rows3(7753, 4, 8704, 1024, 248094);
        check(r == 248096, "MiniCPM-V 4.7 1B's [7753, 4, 8704] head is 248,096 stored rows for 248,094", r, 248096);
    }
    {
        const int r = rows3(7755, 8, 8704, 2048, 248144);
        check(r == 248160, "MiniCPM-V 4.7 35B-A3B's [7755, 8, 8704] head is 248,160 stored rows for 248,144", r, 248160);
    }
    // A three-dimensional head whose column tiles are not this K grid's is some other tensor: nothing is assumed.
    {
        const int r = rows3(7753, 8, 8704, 1024, 248094);
        check(r == 248094, "a 3-D head with the wrong column-tile count falls back to NV", r, 248094);
    }
    {
        const int r = rows3(7753, 4, 1000, 1024, 248094);
        check(r == 248094, "a 3-D head whose last dimension is not a tile size falls back to NV", r, 248094);
    }
    // A width that is both the K grid and a tile size (kpad 5,120, bonsai2-27b: [155200, 5120] for 248,320 rows) is tiles.
    {
        const int r = rows2(155200, 5120, 5120, 248320);
        check(r == 248320, "a 5,120-wide K grid in 5,120-byte tiles is read as tiles and needs no padding", r, 248320);
    }
    // A head that needs no padding keeps the vocabulary's own count.
    {
        const int r = rows2(16000, 6144, 4096, 32000);
        check(r == 32000, "an exactly-aligned vocabulary stays at NV", r, 32000);
    }
    // A K grid that is not a whole number of tile columns cannot be walked, so NV is the honest answer (the caller then
    // refuses, which is what happened before the fix -- and is correct for a grid this function does not understand).
    {
        const int r = rows2(16016, 6144, 4095, 32016);
        check(r == 32016, "a kpad that is not whole tile columns falls back to NV", r, 32016);
    }
    // A tensor narrower than the vocabulary is not a padded head; the caller must not be told to read past its rows.
    {
        const int r = rows2(500, 6144, 4096, 32016);
        check(r == 32016, "a grid smaller than NV falls back to NV rather than a short read", r, 32016);
    }
    // The LOWER end of the bound, in each spelling (review of #2026 at 3a1b03a). The arm above is declined because 500 is
    // not whole blocks, not because it is short: these three are whole 32-row grids, so only `rows >= NV` stands between
    // them and a dequant of 1,024 rows for a vocabulary of 32,016. Deleting that comparison fails all three.
    {
        const int r = rows2(512, 6144, 4096, 32016);
        check(r == 32016, "a whole-block tile grid short of the vocabulary is declined by the bound, not answered with 1,024", r, 32016);
    }
    {
        const int r = rows3(32, 16, 6144, 4096, 32016);
        check(r == 32016, "a 3-D grid short of the vocabulary is declined by the bound, not answered with 1,024", r, 32016);
    }
    {
        const int r = rows2(1024, 4096, 4096, 32016);
        check(r == 32016, "a row grid short of the vocabulary is declined by the bound, not answered with 1,024", r, 32016);
    }
    // A partial block (shape[0] not a multiple of tiles-per-block) is not a grid this function can walk.
    {
        const int r = rows2(16001, 6144, 4096, 32016);
        check(r == 32016, "a partial block falls back to NV", r, 32016);
    }
    // More than one block of padding is not padding: a tile grid twice the vocabulary must not be read.
    {
        const int r = rows2(32032, 6144, 4096, 32016);
        check(r == 32016, "a tile grid far past the vocabulary falls back to NV", r, 32016);
    }
    // NV <= 0 is a caller bug, and returning NV keeps the failure at the caller's check rather than here.
    {
        const int r = rows2(16016, 6144, 4096, 0);
        check(r == 0, "a non-positive vocabulary is passed through, not invented", r, 0);
    }

    if (failures == 0) std::printf("all padded-head row checks passed\n");
    return failures == 0 ? 0 : 1;
}
