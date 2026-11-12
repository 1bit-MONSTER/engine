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
    const int kinds[4] = {Q4NX_TILE_BYTES, Q4NX_CHUNK_Q4K, Q4NX_CHUNK_Q8, Q4NX_CHUNK_Q4S};
    float* whole = malloc((size_t)rows * cols * sizeof(float));
    float* row = malloc((size_t)cols * sizeof(float));
    if (!whole || !row) { check(0, "chunked row: the arms allocate"); free(whole); free(row); return; }
    for (int k = 0; k < 4; k++) {
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
                     : cb == Q4NX_CHUNK_Q8     ? "chunked row: every row matches the whole-table decode (Q8 chunks)"
                                               : "chunked row: every row matches the whole-table decode (Q4S chunks, 4,608 B)");
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
    printf("  chunked row: 4 kinds x 64 rows bit-exact + 5 refusal arms\n");
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
    const int kinds[4] = {Q4NX_TILE_BYTES, Q4NX_CHUNK_Q4K, Q4NX_CHUNK_Q8, Q4NX_CHUNK_Q4S};
    for (int k = 0; k < 4; k++)
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

/* The two kinds of 5,120-byte tile. Same bytes for scales and codes; only the zero-point block differs, and that is
 * what selects the reading: with a zero-point present the code is 0..15 and the value is code*scale + zero_point;
 * with every zero-point 0 the code is a two's-complement nibble and the value is code*scale, scale carrying a sign. */
static void tile_kinds(void) {
    uint8_t* tile = (uint8_t*)calloc(1, Q4NX_TILE_BYTES);
    float* out = (float*)malloc(sizeof(float) * Q4NX_TILE_ROWS * Q4NX_TILE_COLS);
    if (!tile || !out) { check(0, "tile kinds: the buffers allocate"); free(tile); free(out); return; }
    /* scales: exactly representable in bf16, negative on odd groups; codes: every nibble value appears */
    for (int g = 0; g < Q4NX_GROUPS_PER_ROW; g++)
        for (int r = 0; r < Q4NX_TILE_ROWS; r++)
            put_bf16(tile, (size_t)(g * Q4NX_TILE_ROWS + r), (g % 2 ? -1.0f : 1.0f) * 0.25f * (float)(1 + r % 4));
    for (int i = 0; i < 4096; i++) tile[1024 + i] = (uint8_t)((i * 37 + 11) & 0xFF);
    #define CODE_AT(r, c) ((((r) % 16) % 2 == 0) ? (tile[1024 + (((r) / 16) * 8 + (c) / 32) * 256 + 8 * ((c) % 32) + ((r) % 16) / 2] & 0xF) \
                                                 : (tile[1024 + (((r) / 16) * 8 + (c) / 32) * 256 + 8 * ((c) % 32) + ((r) % 16) / 2] >> 4))
    #define SCALE_AT(r, c) (((((c) / 32) % 2) ? -1.0f : 1.0f) * 0.25f * (float)(1 + (r) % 4))

    check(q4nx_tile_kind(tile) == Q4NX_TILE_SYMMETRIC, "tile kinds: zero-points all zero and a scale below zero is the symmetric kind");
    q4nx_dequant_tile(tile, out, Q4NX_TILE_COLS);
    int bad = 0, saw_neg_code = 0;
    for (int r = 0; r < Q4NX_TILE_ROWS; r++)
        for (int c = 0; c < Q4NX_TILE_COLS; c++) {
            const int u = CODE_AT(r, c);
            const int sgn = u >= 8 ? u - 16 : u;
            if (u >= 8) saw_neg_code = 1;
            if (out[r * Q4NX_TILE_COLS + c] != (float)sgn * SCALE_AT(r, c)) bad++;
        }
    check(saw_neg_code, "tile kinds: the fixture has codes of 8 and above, where the two readings differ");
    check(bad == 0, "tile kinds: symmetric is (two's-complement nibble) * signed scale, every element");

    /* negative zero is still zero: the kind does not flip on a sign bit */
    tile[512 + 2 * 7 + 1] = 0x80;
    check(q4nx_tile_kind(tile) == Q4NX_TILE_SYMMETRIC, "tile kinds: a -0.0 zero-point is still the symmetric kind");
    tile[512 + 2 * 7 + 1] = 0;

    /* one non-zero zero-point anywhere makes it the asymmetric kind, read as it always was */
    put_bf16(tile + 512, (size_t)(3 * Q4NX_TILE_ROWS + 5), -0.5f);
    check(q4nx_tile_kind(tile) == Q4NX_TILE_ASYMMETRIC, "tile kinds: one non-zero zero-point makes the tile asymmetric");
    q4nx_dequant_tile(tile, out, Q4NX_TILE_COLS);
    bad = 0;
    for (int r = 0; r < Q4NX_TILE_ROWS; r++)
        for (int c = 0; c < Q4NX_TILE_COLS; c++) {
            const float z = (c / 32 == 3 && r == 5) ? -0.5f : 0.0f;
            if (out[r * Q4NX_TILE_COLS + c] != (float)CODE_AT(r, c) * SCALE_AT(r, c) + z) bad++;
        }
    check(bad == 0, "tile kinds: asymmetric is (code 0..15) * scale + zero_point, every element, as before");

    /* the whole-tensor and one-row readers go through the same tile decoder */
    for (int i = 0; i < 512; i++) tile[512 + i] = 0;
    float* row = (float*)malloc(sizeof(float) * Q4NX_TILE_COLS);
    float* whole = (float*)malloc(sizeof(float) * Q4NX_TILE_ROWS * Q4NX_TILE_COLS);
    if (row && whole) {
        check(q4nx_dequant_tensor_chunked(tile, Q4NX_TILE_BYTES, Q4NX_TILE_ROWS, Q4NX_TILE_COLS, Q4NX_TILE_BYTES, whole) == 0 &&
                  q4nx_dequant_tile_row(tile, Q4NX_TILE_BYTES, Q4NX_TILE_ROWS, Q4NX_TILE_COLS, Q4NX_TILE_BYTES, 9, row) == 0,
              "tile kinds: the tensor and row readers accept a symmetric tile");
        bad = 0;
        for (int c = 0; c < Q4NX_TILE_COLS; c++) {
            const int u = CODE_AT(9, c);
            const float want = (float)(u >= 8 ? u - 16 : u) * SCALE_AT(9, c);
            if (whole[9 * Q4NX_TILE_COLS + c] != want || row[c] != want) bad++;
        }
        check(bad == 0, "tile kinds: and both read it as the tile decoder does");
    } else check(0, "tile kinds: the reader buffers allocate");
    #undef CODE_AT
    #undef SCALE_AT
    free(row); free(whole); free(tile); free(out);
}

/* The kind is not decided from a zero offset alone. An asymmetric tile is allowed to have every zero-point 0 (weights
 * that are exactly code * scale: tests/npu_moe_container_test.cpp builds its fixture that way, and a first version
 * of this rule read its codes 8..15 as -8..-1). What marks the symmetric kind is the signed scale, and a tensor is
 * one kind: its undecided tiles follow it, and a tensor with both kinds is refused. */
static void tile_kind_is_per_tensor(void) {
    enum { T = 4 };   /* four tiles: one tile-row of a [32][1024] tensor */
    uint8_t* t = (uint8_t*)calloc(T, Q4NX_TILE_BYTES);
    float* out = (float*)malloc(sizeof(float) * Q4NX_TILE_ROWS * Q4NX_TILE_COLS * T);
    float* row = (float*)malloc(sizeof(float) * Q4NX_TILE_COLS * T);
    if (!t || !out || !row) { check(0, "per tensor: the buffers allocate"); free(t); free(out); free(row); return; }
    const int cols = Q4NX_TILE_COLS * T;
    /* every tile: scale 1.0, zero-point 0, every byte of codes 0xF9 (low nibble 9, high nibble 15) */
    for (int k = 0; k < T; k++) {
        uint8_t* tile = t + (size_t)k * Q4NX_TILE_BYTES;
        for (int i = 0; i < 256; i++) put_bf16(tile, (size_t)i, 1.0f);
        memset(tile + 1024, 0xF9, 4096);
    }
    check(q4nx_tile_kind(t) == Q4NX_TILE_UNDECIDED, "per tensor: zero offsets with positive scales decide nothing");
    check(q4nx_tiles_kind(t, T, Q4NX_TILE_BYTES) == Q4NX_TILE_ASYMMETRIC, "per tensor: a tensor of undecided tiles is asymmetric");
    check(q4nx_dequant_tensor_chunked(t, (size_t)T * Q4NX_TILE_BYTES, Q4NX_TILE_ROWS, cols, Q4NX_TILE_BYTES, out) == 0,
          "per tensor: it decodes");
    int bad = 0;
    for (int r = 0; r < Q4NX_TILE_ROWS; r++)
        for (int c = 0; c < cols; c++)
            if (out[r * cols + c] != ((r % 2) ? 15.0f : 9.0f)) bad++;
    check(bad == 0, "per tensor: and reads codes 9 and 15 as 9 and 15, the asymmetric reading it always had");
    q4nx_dequant_tile(t, out, Q4NX_TILE_COLS);
    check(out[0] == 9.0f && out[Q4NX_TILE_COLS] == 15.0f, "per tensor: the one-tile decoder agrees");

    /* give ONE tile a negative scale: that tile is symmetric, and the other three now follow their tensor */
    put_bf16(t + 2 * (size_t)Q4NX_TILE_BYTES, 40, -1.0f);
    check(q4nx_tile_kind(t + 2 * (size_t)Q4NX_TILE_BYTES) == Q4NX_TILE_SYMMETRIC, "per tensor: a negative scale makes a tile symmetric");
    check(q4nx_tiles_kind(t, T, Q4NX_TILE_BYTES) == Q4NX_TILE_SYMMETRIC, "per tensor: one symmetric tile makes the tensor symmetric");
    check(q4nx_dequant_tensor_chunked(t, (size_t)T * Q4NX_TILE_BYTES, Q4NX_TILE_ROWS, cols, Q4NX_TILE_BYTES, out) == 0 &&
              q4nx_dequant_tensor(t, (size_t)T * Q4NX_TILE_BYTES, Q4NX_TILE_ROWS, cols, out + 0) == 0,
          "per tensor: both whole-tensor readers decode it");
    bad = 0;
    for (int r = 0; r < Q4NX_TILE_ROWS; r++)
        for (int c = 0; c < cols; c++) {
            float want = (r % 2) ? -1.0f : -7.0f;                     /* 15 -> -1, 9 -> -7 */
            if (c / Q4NX_TILE_COLS == 2 && (c % Q4NX_TILE_COLS) / 32 == 1 && r == 8) want = -want;   /* index 40 = g 1, r 8 */
            if (out[r * cols + c] != want) bad++;
        }
    check(bad == 0, "per tensor: every tile, the undecided ones included, reads as two's complement * scale");
    check(q4nx_dequant_tile_row(t, (size_t)T * Q4NX_TILE_BYTES, Q4NX_TILE_ROWS, cols, Q4NX_TILE_BYTES, 3, row) == 0 &&
              row[0] == -1.0f && row[cols - 1] == -1.0f,
          "per tensor: the one-row reader takes the kind from the row's tiles, undecided ones following");
    /* the padded-grid reader: the model's tensor is the first tile only, the container's grid is all four, and the
     * kind is the grid's -- tile 0 is undecided on its own and symmetric as part of this tensor */
    check(q4nx_dequant_tensor_packed(t, (size_t)T * Q4NX_TILE_BYTES, Q4NX_TILE_ROWS, Q4NX_TILE_COLS, Q4NX_TILE_ROWS, cols,
                                     Q4NX_TILE_BYTES, out) == 0 && out[0] == -7.0f && out[Q4NX_TILE_COLS] == -1.0f,
          "per tensor: the padded-grid reader decodes it, taking the kind from the container's grid");
    q4nx_dequant_tile(t, out, Q4NX_TILE_COLS);
    check(out[0] == 9.0f, "per tensor: an undecided tile decoded alone, with no tensor around it, stays asymmetric");

    /* now an offset in another tile: one tensor, both kinds -> refused by every tensor-level reader */
    put_bf16(t + 512, 3, 0.5f);
    check(q4nx_tiles_kind(t, T, Q4NX_TILE_BYTES) == -1, "per tensor: a symmetric and an asymmetric tile in one tensor is neither kind");
    check(q4nx_dequant_tensor_chunked(t, (size_t)T * Q4NX_TILE_BYTES, Q4NX_TILE_ROWS, cols, Q4NX_TILE_BYTES, out) == -1,
          "per tensor: the chunked reader refuses it");
    check(q4nx_dequant_tensor(t, (size_t)T * Q4NX_TILE_BYTES, Q4NX_TILE_ROWS, cols, out) == -1,
          "per tensor: the whole-tensor reader refuses it");
    check(q4nx_dequant_tile_row(t, (size_t)T * Q4NX_TILE_BYTES, Q4NX_TILE_ROWS, cols, Q4NX_TILE_BYTES, 0, row) == -1,
          "per tensor: the one-row reader refuses it");
    check(q4nx_dequant_tensor_packed(t, (size_t)T * Q4NX_TILE_BYTES, Q4NX_TILE_ROWS, Q4NX_TILE_COLS, Q4NX_TILE_ROWS, cols,
                                     Q4NX_TILE_BYTES, out) == -1,
          "per tensor: the padded-grid reader refuses it (the kind is the container grid's, not the model's)");
    free(t); free(out); free(row);
}

/* The 4,608-byte symmetric chunk (Hy-MT2-1.8B-NPU2): 256 bf16 scales at g*32 + r, then 4,096 code bytes at
 * c*16 + r/2 (even row = low nibble), value = two's-complement nibble * scale. The expected values below index the
 * bytes from that description, not through the decoder; and each of the three ways the layout was NOT (the 5,120
 * tile's code order, row-major scales, offset-binary codes) is shown to give a different table, so the fixture can
 * tell them apart. */
static void q4s_chunk(void) {
    uint8_t* t = (uint8_t*)calloc(1, Q4NX_CHUNK_Q4S);
    float* out = (float*)malloc(sizeof(float) * Q4NX_TILE_ROWS * Q4NX_TILE_COLS);
    if (!t || !out) { check(0, "q4s: the buffers allocate"); free(t); free(out); return; }
    for (int i = 0; i < 256; i++) put_bf16(t, (size_t)i, ((i % 5 == 0) ? -1.0f : 1.0f) * 0.125f * (float)(1 + i % 7));
    for (int i = 0; i < 4096; i++) t[512 + i] = (uint8_t)((i * 113 + 29) & 0xFF);
    #define Q4S_U(r, c) (((r) % 2 == 0) ? (t[512 + (c) * 16 + (r) / 2] & 0xF) : (t[512 + (c) * 16 + (r) / 2] >> 4))
    #define Q4S_SC(i) ((((i) % 5 == 0) ? -1.0f : 1.0f) * 0.125f * (float)(1 + (i) % 7))
    q4nx_dequant_tile_q4s(t, out, Q4NX_TILE_COLS);
    int bad = 0, as_5120_order = 0, as_row_major = 0, as_offset = 0, neg = 0;
    for (int r = 0; r < Q4NX_TILE_ROWS; r++)
        for (int c = 0; c < Q4NX_TILE_COLS; c++) {
            const int g = c / 32, u = Q4S_U(r, c), sgn = u >= 8 ? u - 16 : u;
            const float got = out[r * Q4NX_TILE_COLS + c];
            if (u >= 8) neg = 1;
            if (got != (float)sgn * Q4S_SC(g * 32 + r)) bad++;
            /* the layouts it is not */
            const uint8_t b5 = t[512 + ((r / 16) * 8 + g) * 256 + 8 * (c % 32) + (r % 16) / 2];
            const int u5 = ((r % 16) % 2 == 0) ? (b5 & 0xF) : (b5 >> 4);
            if (got != (float)(u5 >= 8 ? u5 - 16 : u5) * Q4S_SC(g * 32 + r)) as_5120_order++;
            if (got != (float)sgn * Q4S_SC(r * 8 + g)) as_row_major++;
            if (got != (float)(u - 8) * Q4S_SC(g * 32 + r)) as_offset++;
        }
    check(neg, "q4s: the fixture has codes of 8 and above");
    check(bad == 0, "q4s: every element is (two's-complement nibble at c*16 + r/2) * (scale at g*32 + r)");
    check(as_5120_order > 1000, "q4s: the 5,120 tile's code order would give a different table");
    check(as_row_major > 1000, "q4s: row-major scales would give a different table");
    check(as_offset > 1000, "q4s: offset-binary codes would give a different table");
    float* whole = (float*)malloc(sizeof(float) * Q4NX_TILE_ROWS * Q4NX_TILE_COLS);
    if (whole) {
        check(q4nx_dequant_tensor_chunked(t, Q4NX_CHUNK_Q4S, Q4NX_TILE_ROWS, Q4NX_TILE_COLS, Q4NX_CHUNK_Q4S, whole) == 0 &&
                  memcmp(whole, out, sizeof(float) * Q4NX_TILE_ROWS * Q4NX_TILE_COLS) == 0,
              "q4s: the tensor reader takes the kind by its size and decodes the same values");
        check(q4nx_dequant_tensor_chunked(t, Q4NX_CHUNK_Q4S - 1, Q4NX_TILE_ROWS, Q4NX_TILE_COLS, Q4NX_CHUNK_Q4S, whole) == -1,
              "q4s: one byte short is refused");
    } else check(0, "q4s: the reader buffer allocates");
    check(q4nx_tile_row_kind((size_t)Q4NX_CHUNK_Q4S, 32, 256) == Q4NX_CHUNK_Q4S &&
              q4nx_tile_row_kind((size_t)Q4NX_TILE_BYTES, 32, 256) == Q4NX_TILE_BYTES,
          "q4s: 4,608 and 5,120 bytes per tile are told apart by size alone");
    #undef Q4S_U
    #undef Q4S_SC
    free(whole); free(t); free(out);
}

/* The 1,280-byte quarter tile (Gemma3-270M-NPU2): 16 rows x 128 columns, scales and zero-points at g*16 + r, codes at
 * c*8 + r/2. A [32 x 256] tensor of 2 x 2 chunks, read into a 384-wide buffer (the forward pads K to 256). */
static void q4h_chunks(void) {
    const int rows = 32, k = 256, cols = 384, ntc = k / Q4NX_Q4H_COLS;
    const size_t n = (size_t)(rows / Q4NX_Q4H_ROWS) * (size_t)ntc;
    uint8_t* t = (uint8_t*)calloc(n, Q4NX_CHUNK_Q4H);
    float* out = (float*)malloc(sizeof(float) * (size_t)rows * cols);
    if (!t || !out) { check(0, "q4h: the buffers allocate"); free(t); free(out); return; }
    for (size_t ci = 0; ci < n; ci++) {
        uint8_t* c = t + ci * Q4NX_CHUNK_Q4H;
        for (int i = 0; i < 64; i++) {
            put_bf16(c, (size_t)i, 0.25f * (float)(1 + (int)((i + ci) % 7)));          /* scale */
            put_bf16(c + 128, (size_t)i, -1.0f * (float)(1 + (int)((i + 3 * ci) % 5)));  /* zero-point */
        }
        for (int b = 0; b < 1024; b++) c[256 + b] = (uint8_t)((b * 7 + 3 * ci) & 0xff);
    }
    for (size_t i = 0; i < (size_t)rows * cols; i++) out[i] = 99.0f;
    check(q4nx_dequant_tensor_q4h(t, n * Q4NX_CHUNK_Q4H, rows, k, cols, out) == 0, "q4h: a 2 x 2 grid reads");
    int bad = 0, pad_bad = 0;
    for (int r = 0; r < rows; r++)
        for (int col = 0; col < cols; col++) {
            const float got = out[(size_t)r * cols + col];
            if (col >= k) { if (got != 0.0f) pad_bad++; continue; }
            const int tr = r / 16, tc = col / 128, rr = r % 16, cc = col % 128;
            const size_t ci = (size_t)tr * ntc + tc;
            const int i = (cc / 32) * 16 + rr;
            const uint8_t byte = (uint8_t)(((cc * 8 + rr / 2) * 7 + 3 * ci) & 0xff);
            const int code = (rr % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            const float s = 0.25f * (float)(1 + (int)((i + ci) % 7)), z = -1.0f * (float)(1 + (int)((i + 3 * ci) % 5));
            if (got != (float)code * s + z) bad++;
        }
    check(bad == 0, "q4h: every element is code (at c*8 + r/2) * scale + zero (at g*16 + r), grid row-major");
    check(pad_bad == 0, "q4h: the columns past the container's K are written 0");
    check(q4nx_dequant_tensor_q4h(t, n * Q4NX_CHUNK_Q4H - 1, rows, k, cols, out) == -1, "q4h: one byte short is refused");
    check(q4nx_dequant_tensor_q4h(t, n * Q4NX_CHUNK_Q4H, rows, k, k - 128, out) == -1, "q4h: a buffer narrower than K is refused");
    check(q4nx_dequant_tensor_q4h(t, n * Q4NX_CHUNK_Q4H, 24, k, cols, out) == -1, "q4h: rows off the 16-row grid are refused");
    free(t);
    free(out);
}

/* engine#1786: Gemma4-12B-IT-NPU2's 16 x 256 wide chunks (q4nx_dequant.h). Settled against google's QAT Q4_0 GGUF
 * of the model (corr 0.9986) and the pack's own bf16 embedding (the Q8W head, 1.00000); this pins the reading. */
static void wide_chunks(void) {
    const int rows = 32, k = 512, cols = 768, ntc = k / Q4NX_WIDE_COLS;
    const size_t n = (size_t)(rows / Q4NX_WIDE_ROWS) * (size_t)ntc;
    for (int bits = 4; bits <= 8; bits += 4) {
        const int cb = bits == 4 ? Q4NX_CHUNK_Q4W : Q4NX_CHUNK_Q8W, ncode = cb - 256;
        uint8_t* t = (uint8_t*)calloc(n, (size_t)cb);
        float* out = (float*)malloc(sizeof(float) * (size_t)rows * cols);
        if (!t || !out) { check(0, "wide: the buffers allocate"); free(t); free(out); return; }
        for (size_t ci = 0; ci < n; ci++) {
            uint8_t* c = t + ci * (size_t)cb;
            for (int i = 0; i < 128; i++)
                put_bf16(c, (size_t)i, (i % 3 ? 0.125f : -0.25f) * (float)(1 + (int)((i + ci) % 5)));   /* signed scale */
            for (int b = 0; b < ncode; b++) c[256 + b] = (uint8_t)((b * 13 + 5 * ci) & 0xff);
        }
        for (size_t i = 0; i < (size_t)rows * cols; i++) out[i] = 99.0f;
        check(q4nx_dequant_tensor_wide(t, n * (size_t)cb, cb, rows, k, cols, out) == 0,
              bits == 4 ? "q4w: a 2 x 2 grid reads" : "q8w: a 2 x 2 grid reads");
        int bad = 0, pad_bad = 0;
        for (int r = 0; r < rows; r++)
            for (int col = 0; col < cols; col++) {
                const float got = out[(size_t)r * cols + col];
                if (col >= k) { if (got != 0.0f) pad_bad++; continue; }
                const int tr = r / 16, tc = col / 256, rr = r % 16, cc = col % 256;
                const size_t ci = (size_t)tr * ntc + tc;
                const int i = (cc / 32) * 16 + rr;
                const float sc = (i % 3 ? 0.125f : -0.25f) * (float)(1 + (int)((i + ci) % 5));
                int code;
                if (bits == 4) {
                    const uint8_t byte = (uint8_t)(((cc * 8 + rr / 2) * 13 + 5 * ci) & 0xff);
                    const int v = (rr % 2 == 0) ? (byte & 0xF) : (byte >> 4);
                    code = v >= 8 ? v - 16 : v;
                } else {
                    code = (int8_t)(uint8_t)(((cc * 16 + rr) * 13 + 5 * ci) & 0xff);
                }
                if (got != (float)code * sc) bad++;
            }
        check(bad == 0, bits == 4 ? "q4w: code (two's complement, c*8 + r/2, even row low) * scale (g*16 + r), row-major grid"
                                  : "q8w: int8 code (c*16 + r) * scale (g*16 + r), row-major grid");
        check(pad_bad == 0, "wide: the columns past the container's K are written 0");
        check(q4nx_dequant_tensor_wide(t, n * (size_t)cb - 1, cb, rows, k, cols, out) == -1, "wide: one byte short is refused");
        check(q4nx_dequant_tensor_wide(t, n * (size_t)cb, cb, 24, k, cols, out) == -1, "wide: rows off the 16-row grid are refused");
        check(q4nx_dequant_tensor_wide(t, n * (size_t)cb, 4608, rows, k, cols, out) == -1, "wide: another chunk size is not a wide kind");
        free(t);
        free(out);
    }
    /* why the declared shape decides: a Q4W tensor is exactly as many bytes as a Q4S (4608) grid of the same size */
    check((size_t)(4096 / 16) * (3840 / 256) * Q4NX_CHUNK_Q4W == (size_t)(4096 / 32) * (3840 / 256) * 4608,
          "a 4096 x 3840 Q4W tensor has a 4608-byte Q4S grid's byte count, so only the declared shape tells them apart");
}

/* engine#1786: GPT-OSS-20B-NPU2's 2560-byte chunks (q4nx_dequant.h). Settled against ggml-org/gpt-oss-20b-GGUF
 * (D2560 corr 0.995-0.997 against its Q8_0, MX2560 1.0000 against its MXFP4); this pins the reading. */
static int d2560_code_at(const uint8_t* c, int r, int col) {
    const uint8_t b = c[512 + (r / 16) * 1024 + col * 8 + (r % 16) / 2];
    return (r & 1) ? (b >> 4) : (b & 15);
}

static void d2560_chunks(void) {
    uint8_t c[Q4NX_CHUNK_D2560];
    for (int i = 0; i < Q4NX_CHUNK_D2560; i++) c[i] = (uint8_t)((i * 29 + 7) & 0xff);
    for (int i = 0; i < 128; i++) {
        put_bf16(c, (size_t)i, 0.125f * (float)(1 + i % 9));               /* scale  at g*32 + r */
        put_bf16(c + 256, (size_t)i, -0.5f * (float)(i % 4));               /* zero   at g*32 + r */
    }
    float out[32 * 128];
    q4nx_dequant_chunk_d2560(c, out, 128);
    int bad = 0;
    for (int r = 0; r < 32; r++)
        for (int col = 0; col < 128; col++) {
            const int slot = (col / 32) * 32 + r;
            if (out[r * 128 + col] != (float)d2560_code_at(c, r, col) * 0.125f * (float)(1 + slot % 9) - 0.5f * (float)(slot % 4)) bad++;
        }
    check(bad == 0, "d2560: code (two 16-row halves, h*1024 + c*8 + (r%16)/2) * scale + zero, both at g*32 + r");

    uint8_t m[Q4NX_CHUNK_D2560];
    static const float lut[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6};
    for (int i = 0; i < Q4NX_CHUNK_D2560; i++) m[i] = (uint8_t)((i * 31 + 3) & 0xff);
    for (int i = 0; i < 128; i++) m[i] = (uint8_t)(120 + i % 11);          /* E8M0 exponents at g*32 + r */
    q4nx_dequant_chunk_mx2560(m, out, 128);
    bad = 0;
    for (int r = 0; r < 32; r++)
        for (int col = 0; col < 128; col++) {
            const int e = 120 + ((col / 32) * 32 + r) % 11;
            float p = 1.0f;
            for (int n = e - 128; n < 0; n++) p *= 0.5f;
            for (int n = e - 128; n > 0; n--) p *= 2.0f;
            if (out[r * 128 + col] != lut[d2560_code_at(m, r, col)] * p) bad++;
        }
    check(bad == 0, "mx2560: E2M1 code (same order) * 2^(e - 128), e at g*32 + r; bytes 128..511 unread");

    /* a [RB=2, CB=2, S=2, 2560] tensor: rows rb*S*32 + s*32 + r, columns cb*128 + c, clipped and padded */
    const int RB = 2, CB = 2, S = 2, rows = 100, k = 200, cols = 256;
    uint8_t* t = (uint8_t*)malloc((size_t)RB * CB * S * Q4NX_CHUNK_D2560);
    float* o = (float*)malloc(sizeof(float) * rows * cols);
    float* ref = (float*)malloc(sizeof(float) * 32 * 128);
    if (!t || !o || !ref) { check(0, "d2560: buffers allocate"); free(t); free(o); free(ref); return; }
    for (int ci = 0; ci < RB * CB * S; ci++) {
        uint8_t* q = t + (size_t)ci * Q4NX_CHUNK_D2560;
        memcpy(q, c, Q4NX_CHUNK_D2560);
        put_bf16(q + 256, 0, (float)ci);                                    /* tag each chunk through one zero */
    }
    for (int i = 0; i < rows * cols; i++) o[i] = 99.0f;
    check(q4nx_dequant_tensor_d2560(t, (size_t)RB * CB * S * Q4NX_CHUNK_D2560, RB, CB, S, rows, k, cols, o) == 0,
          "d2560: a 2 x 2 x 2 tensor reads, clipped to 100 rows and 200 columns");
    bad = 0;
    int pad_bad = 0;
    for (int row = 0; row < rows; row++) {
        const int rb = row / (S * 32), s = (row / 32) % S, r = row % 32;
        for (int col = 0; col < cols; col++) {
            const float got = o[(size_t)row * cols + col];
            if (col >= k) { if (got != 0.0f) pad_bad++; continue; }
            const int cb = col / 128;
            const uint8_t* q = t + (((size_t)rb * CB + cb) * S + s) * Q4NX_CHUNK_D2560;
            q4nx_dequant_chunk_d2560(q, ref, 128);
            if (got != ref[r * 128 + col % 128]) bad++;
        }
    }
    check(bad == 0, "d2560: tensor row rb*S*32 + s*32 + r comes from chunk [rb][cb][s]");
    check(pad_bad == 0, "d2560: columns past k are written 0");
    check(q4nx_dequant_tensor_d2560(t, (size_t)RB * CB * S * Q4NX_CHUNK_D2560 - 1, RB, CB, S, rows, k, cols, o) == -1,
          "d2560: one byte short is refused");
    check(q4nx_dequant_tensor_d2560(t, (size_t)RB * CB * S * Q4NX_CHUNK_D2560, RB, CB, S, 129, k, cols, o) == -1,
          "d2560: more rows than the grid holds is refused");
    free(t); free(o); free(ref);
}

int main(void) {
    q4s_chunk();
    q4h_chunks();
    wide_chunks();
    d2560_chunks();
    tile_kinds();
    tile_kind_is_per_tensor();
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
