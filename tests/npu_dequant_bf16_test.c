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

/* engine#1784: the loader no longer expands the plain int8 embedding table into RAM (0.74 GiB of int8 became 2.96 GiB
 * of floats on the 27B); it reads one row per token out of the container's mapping through q4nx_plain_i8_at(). The
 * arithmetic that must not drift is the scale group: per 32-column GROUP within a row, indexed (row * groups + col/32).
 * A row/column mix-up there produces plausible numbers, so this is pinned against the whole-table decode of the same
 * buffer, bit for bit, on a table whose scale differs per row AND per group -- symmetry cannot make a wrong index pass.
 * Host-only: no model, no device, no tile alignment (the element-wise read needs none).
 *
 * These arms live in this file rather than a target of their own because a new add_test() has to be built by
 * .github/workflows/ci.yml (the `ci_targets` test enforces it) and an edit to that file needs a token with the
 * `workflow` scope; this binary already links q4nx_dequant.c and already runs in CI, so the primitive is covered
 * without the workflow change. */
static void embed_row_arms(void) {
    const int rows = 7, cols = 64;               /* not a tile grid on purpose */
    const int groups = cols / Q4NX_GROUP_SIZE;
    uint8_t* q = malloc((size_t)rows * cols);
    float* sc = malloc((size_t)rows * groups * sizeof(float));
    float* ref = malloc((size_t)rows * cols * sizeof(float));
    if (!q || !sc || !ref) { check(0, "embed row: the arms allocate"); free(q); free(sc); free(ref); return; }
    unsigned seed = 12345u;
    for (int i = 0; i < rows * cols; i++) {
        seed = seed * 1103515245u + 12345u;
        q[i] = (uint8_t)((seed >> 16) & 0xff);   /* every bit pattern, including 0x80 (-128) and 0xff (-1) */
    }
    for (int r = 0; r < rows; r++)
        for (int g = 0; g < groups; g++) sc[r * groups + g] = 0.0005f * (float)(1 + r * groups + g);

    check(q4nx_dequant_plain_i8(q, sc, rows, cols, ref) == 0, "embed row: the reference decode succeeds");
    int bad = 0;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            const float got = q4nx_plain_i8_at(q, sc, rows, cols, r, c);
            if (memcmp(&got, &ref[(size_t)r * cols + c], sizeof(float)) != 0) bad++;
        }
    check(bad == 0, "embed row: every element reads identically to the whole-table decode (bit for bit)");

    bad = 0;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if (q4nx_plain_i8_at(q, NULL, rows, cols, r, c) != (float)((const int8_t*)q)[(size_t)r * cols + c]) bad++;
    check(bad == 0, "embed row: with no scale array the value is the raw int8 (the unscaled table)");

    check(q4nx_plain_i8_at(q, sc, rows, cols, -1, 0) == 0.0f, "embed row: row -1 reads as 0.0f");
    check(q4nx_plain_i8_at(q, sc, rows, cols, rows, 0) == 0.0f, "embed row: row == rows reads as 0.0f");
    check(q4nx_plain_i8_at(q, sc, rows, cols, 0, -1) == 0.0f, "embed row: col -1 reads as 0.0f");
    check(q4nx_plain_i8_at(q, sc, rows, cols, 0, cols) == 0.0f, "embed row: col == cols (a pad column) reads 0.0f");
    check(q4nx_plain_i8_at(NULL, sc, rows, cols, 0, 0) == 0.0f, "embed row: a null table reads as 0.0f");
    check(q4nx_plain_i8_at(q, sc, rows, cols - 1, 0, 0) == 0.0f,
          "embed row: cols that is not a multiple of the group reads 0.0f, not off the end of the scale row");
    free(q); free(sc); free(ref);
    printf("  embed row: 9 arms (bit-exact row read, null scale, 4 out-of-range, null table, wrong cols)\n");
}

/* engine#1784: one row of a CHUNKED (tiled) table, decoded without decoding the table. This is the path the 27B-class
 * containers actually take: 155,200 tiles of 5,120 B (0.625 B/element, packed 4-bit) that decode to 1.27 G elements and
 * materialise as 4.74 GiB of f32. The row gather must reproduce exactly what q4nx_dequant_tensor_chunked() puts in that
 * row -- all three chunk kinds, every row including the last one of a tile-row and the last row of the tensor -- because
 * the tiles are ordered (row-tile * ntiles_across + col-tile) and an off-by-one there reads a neighbouring tile and
 * still produces plausible numbers. */
static void chunked_row_arms(void) {
    const int rows = 64, cols = 512;                  /* 2x2 tiles: several rows per tile, several tiles per row */
    const int ntc = cols / Q4NX_TILE_COLS;
    const int kinds[3] = {Q4NX_TILE_BYTES, Q4NX_CHUNK_Q4K, Q4NX_CHUNK_Q8};
    float* whole = malloc((size_t)rows * cols * sizeof(float));
    float* row = malloc((size_t)cols * sizeof(float));
    if (!whole || !row) { check(0, "chunked row: the arms allocate"); free(whole); free(row); return; }
    for (int k = 0; k < 3; k++) {
        const int cb = kinds[k];
        const size_t nbytes = (size_t)(rows / Q4NX_TILE_ROWS) * (size_t)ntc * (size_t)cb;
        uint8_t* q = malloc(nbytes);
        if (!q) { check(0, "chunked row: the tile buffer allocates"); continue; }
        unsigned seed = 987u + (unsigned)k;
        for (size_t i = 0; i < nbytes; i++) { seed = seed * 1103515245u + 12345u; q[i] = (uint8_t)(seed >> 16); }
        check(q4nx_dequant_tensor_chunked(q, nbytes, rows, cols, cb, whole) == 0,
              "chunked row: the whole-table reference decodes");
        int bad = 0;
        for (int r = 0; r < rows; r++) {
            if (q4nx_dequant_tile_row(q, nbytes, rows, cols, cb, r, row) != 0) { bad++; continue; }
            if (memcmp(row, whole + (size_t)r * cols, (size_t)cols * sizeof(float)) != 0) bad++;
        }
        check(bad == 0, cb == Q4NX_TILE_BYTES   ? "chunked row: every row matches the whole-table decode (5,120 B tiles)"
                     : cb == Q4NX_CHUNK_Q4K    ? "chunked row: every row matches the whole-table decode (Q4K chunks)"
                                               : "chunked row: every row matches the whole-table decode (Q8 chunks)");
        check(q4nx_dequant_tile_row(q, nbytes, rows, cols, cb, rows, row) == -1, "chunked row: row == rows is refused");
        check(q4nx_dequant_tile_row(q, nbytes, rows, cols, cb, -1, row) == -1, "chunked row: row -1 is refused");
        check(q4nx_dequant_tile_row(q, nbytes - 1, rows, cols, cb, 0, row) == -1,
              "chunked row: one byte short is refused, not read past");
        check(q4nx_dequant_tile_row(q, nbytes, rows - 1, cols, cb, 0, row) == -1,
              "chunked row: a rows that is not a tile multiple is refused (a last partial row has no tile)");
        check(q4nx_dequant_tile_row(NULL, nbytes, rows, cols, cb, 0, row) == -1, "chunked row: a null table is refused");
        free(q);
    }
    free(whole); free(row);
    printf("  chunked row: 3 kinds x 64 rows bit-exact + 5 refusal arms\n");
}

/* engine#1842: the forward picks the per-row path for a chunked embedding with q4nx_tile_row_kind(). It used to pick it
 * on `esz % tiles == 0` alone (tiles = (NV/32)*(kpad/256), truncating), which three kinds of table pass while the reader
 * refuses every row of them -- and embed_at() turns that refusal into 0.0f. Each arm below is one such table: the old
 * rule accepts it, the reader refuses it, and the predicate must refuse it too, so the table stays on the whole-table
 * decode, which reports its own failure. The valid kinds must still be accepted, with the kind the reader takes. */
static int old_rule(size_t esz, int nv, int kpad) {
    const size_t tiles = (size_t)(nv / 32) * (size_t)(kpad / 256);
    return tiles > 0 && esz % tiles == 0;
}
static void tile_row_kind_arms(void) {
    const int kinds[3] = {Q4NX_TILE_BYTES, Q4NX_CHUNK_Q4K, Q4NX_CHUNK_Q8};
    for (int k = 0; k < 3; k++)
        check(q4nx_tile_row_kind((size_t)4 * (size_t)kinds[k], 64, 512) == kinds[k],
              "tile-row kind: each supported chunk kind on a 2x2 grid is accepted as itself");
    struct { size_t esz; int nv, kpad; const char* what; } bad[] = {
        {(size_t)4 * 1000, 64, 512, "a chunk of 1,000 B (no such kind) on an exact 2x2 grid"},
        {(size_t)8 * Q4NX_TILE_BYTES, 64, 512, "two tiles' worth per grid cell (esz / tiles = 10,240 B)"},
        {(size_t)2 * Q4NX_TILE_BYTES, 48, 512, "NV 48: NV / 32 truncates to 1 tile-row and the grid misses rows 32..47"},
        {(size_t)2 * Q4NX_TILE_BYTES, 64, 384, "kpad 384: kpad / 256 truncates to 1 tile-column"},
    };
    uint8_t* q = calloc(bad[1].esz, 1);
    float* row = malloc(512 * sizeof(float));
    if (!q || !row) { check(0, "tile-row kind: the arms allocate"); free(q); free(row); return; }
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        const int cb_old = (int)(bad[i].esz / ((size_t)(bad[i].nv / 32) * (size_t)(bad[i].kpad / 256)));
        char msg[256];
        snprintf(msg, sizeof msg, "tile-row kind: the old rule took %s", bad[i].what);
        check(old_rule(bad[i].esz, bad[i].nv, bad[i].kpad), msg);
        snprintf(msg, sizeof msg, "tile-row kind: the reader refuses %s", bad[i].what);
        check(q4nx_dequant_tile_row(q, bad[i].esz, bad[i].nv, bad[i].kpad, cb_old, 0, row) == -1, msg);
        snprintf(msg, sizeof msg, "tile-row kind: and so does the predicate the forward now asks (%s)", bad[i].what);
        check(q4nx_tile_row_kind(bad[i].esz, bad[i].nv, bad[i].kpad) == 0, msg);
    }
    free(q); free(row);
    printf("  tile-row kind: 3 kinds accepted, 4 tables the old rule took and the reader refuses are refused\n");
}

int main(void) {
    tile_row_kind_arms();
    chunked_row_arms();
    embed_row_arms();
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
