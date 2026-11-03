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

// npu_dequant_bf16_test
//
// engine#875: `q4nx_dequant_tensor()` decoded EVERY 2-D weight as 5120-byte q4_1
// tiles, so a PLAIN bf16 container (`tools/safetensors_to_q4nx.py pack --dtype
// bf16`) — which is 3.2x the tile size for the same shape — was accepted by the
// old `nbytes >= need` test, had two thirds of its buffer ignored, and computed on
// noise.  This pins the behaviour that replaces it:
//
//   1. a rows*cols*2-byte buffer decodes as plain bf16, bit-exactly;
//   2. the same tensor through `q4nx_dequant_plain_bf16()` gives the same values;
//   3. plain bf16 needs no 32x256 tile alignment (a 3x5 tensor decodes);
//   4. a genuine q4_1 tile grid of that shape still decodes through the tile path,
//      so the new branch does not hijack packed containers;
//   5. a buffer that is neither bf16-sized nor exactly the tile grid is refused
//      instead of being read as tiles.
//
// Exact bf16 known answers: every test value below is an integer, so its float32
// bit pattern has a zero low half and the truncation to bf16 is lossless.
#include "q4nx_dequant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check(int ok, const char* what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

/* float -> the 16 bits a bf16 tensor stores (truncation; exact for the values used) */
static uint16_t to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    return (uint16_t)(bits >> 16);
}

static void put_bf16(uint8_t* dst, size_t i, float v) {
    const uint16_t b = to_bf16(v);
    dst[2 * i] = (uint8_t)(b & 0xFF);
    dst[2 * i + 1] = (uint8_t)(b >> 8);
}

/* A value pattern with both signs, no zeros in the middle, exact in bf16 */
static float value_at(size_t i) { return (float)((int)(i % 61) - 30); }

int main(void) {
    const int rows = 64, cols = 512;
    const size_t n = (size_t)rows * cols;

    uint8_t* bf = calloc(n * 2, 1);
    float* got = calloc(n, sizeof(float));
    float* want = calloc(n, sizeof(float));
    float* again = calloc(n, sizeof(float));
    if (!bf || !got || !want || !again) { printf("FAIL out of memory\n"); return 1; }
    for (size_t i = 0; i < n; i++) {
        want[i] = value_at(i);
        put_bf16(bf, i, want[i]);
    }

    /* 1. the bug: a bf16-sized buffer must decode as bf16, not as q4_1 tiles */
    check(q4nx_dequant_tensor(bf, n * 2, rows, cols, got) == 0,
          "a rows*cols*2-byte tensor decodes");
    size_t bad = 0;
    for (size_t i = 0; i < n; i++)
        if (got[i] != want[i]) bad++;
    printf("     %zu of %zu values differ from the bf16 source\n", bad, n);
    check(bad == 0, "every value equals the bf16 source (not a q4_1 tile read)");

    /* 2. the named entry point agrees */
    check(q4nx_dequant_plain_bf16(bf, rows, cols, again) == 0, "q4nx_dequant_plain_bf16 decodes");
    check(memcmp(got, again, n * sizeof(float)) == 0,
          "plain_bf16 and the q4nx_dequant_tensor path agree");

    /* 3. plain bf16 carries no tile geometry requirement */
    {
        const int r2 = 3, c2 = 5;
        uint8_t small[3 * 5 * 2];
        float small_out[3 * 5];
        for (size_t i = 0; i < 15; i++) put_bf16(small, i, value_at(i));
        check(q4nx_dequant_tensor(small, sizeof small, r2, c2, small_out) == 0,
              "a 3x5 bf16 tensor decodes (no 32x256 alignment needed)");
        int same = 1;
        for (size_t i = 0; i < 15; i++)
            if (small_out[i] != value_at(i)) same = 0;
        check(same, "the 3x5 values are exact");
    }

    /* 4. a real q4_1 tile grid of the same shape still takes the tile path:
     *    scale 1.0, zero point 0.0, code (byte & 0xF) / (byte >> 4) per row parity. */
    {
        const size_t ntiles = (size_t)(rows / Q4NX_TILE_ROWS) * (size_t)(cols / Q4NX_TILE_COLS);
        uint8_t* tiles = calloc(ntiles * Q4NX_TILE_BYTES, 1);
        float* tgot = calloc(n, sizeof(float));
        if (!tiles || !tgot) { printf("FAIL out of memory\n"); return 1; }
        for (size_t t = 0; t < ntiles; t++) {
            uint8_t* tile = tiles + t * Q4NX_TILE_BYTES;
            for (int i = 0; i < Q4NX_TILE_ROWS * Q4NX_GROUPS_PER_ROW; i++) {
                const uint16_t one = to_bf16(1.0f);   /* scales @[0,512) */
                tile[2 * i] = (uint8_t)(one & 0xFF);
                tile[2 * i + 1] = (uint8_t)(one >> 8);
                /* zero points @[512,1024) stay 0 */
            }
            uint8_t* packed = tile + Q4NX_TILE_ROWS * Q4NX_GROUPS_PER_ROW * 4; /* @1024 */
            for (int i = 0; i < Q4NX_TILE_ROWS * Q4NX_TILE_COLS / 2; i++) packed[i] = (uint8_t)(0x31);
        }
        check(q4nx_dequant_tensor(tiles, ntiles * Q4NX_TILE_BYTES, rows, cols, tgot) == 0,
              "a tile-sized q4_1 grid still decodes");
        /* codes are 1 (even row, low nibble) and 3 (odd row, high nibble), scale 1, zp 0 */
        check(tgot[0] == 1.0f && tgot[(size_t)cols] == 3.0f &&
                  tgot[(size_t)(Q4NX_TILE_ROWS * cols)] == 1.0f,
              "the tile path yields the packed codes (1.0 / 3.0), so bf16 did not hijack it");
        free(tiles);
        free(tgot);
    }

    /* 5. an unknown size is refused rather than read as tiles */
    check(q4nx_dequant_tensor(bf, n * 2 + 2, rows, cols, got) == -1,
          "a tile-plus-2-byte buffer is refused");
    check(q4nx_dequant_tensor(bf, n * 2 - 5120, rows, cols, got) == -1,
          "a short buffer is refused");

    free(bf);
    free(got);
    free(want);
    free(again);
    printf("%s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 1 : 0;
}
