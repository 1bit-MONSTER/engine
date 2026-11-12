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
// Copyright (c) 2026 bong-water-water-bong
// q4nx_dequant.c — see q4nx_dequant.h for the layout and its verification.
#include "q4nx_dequant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline float bf16_to_f32(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline uint16_t le16(const uint8_t* p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

int q4nx_tile_kind(const uint8_t* tile) {
    const uint8_t* sc_b = tile;
    const uint8_t* zp_b = tile + Q4NX_TILE_ROWS * Q4NX_GROUPS_PER_ROW * 2; /* +512 */
    for (int i = 0; i < Q4NX_TILE_ROWS * Q4NX_GROUPS_PER_ROW; i++)
        if ((le16(zp_b + 2 * i) & 0x7FFF) != 0) return Q4NX_TILE_ASYMMETRIC;   /* +0.0 and -0.0 both count as zero */
    for (int i = 0; i < Q4NX_TILE_ROWS * Q4NX_GROUPS_PER_ROW; i++) {
        const uint16_t sc = le16(sc_b + 2 * i);
        if ((sc & 0x8000) && (sc & 0x7FFF)) return Q4NX_TILE_SYMMETRIC;        /* a scale below zero */
    }
    return Q4NX_TILE_UNDECIDED;
}

int q4nx_tiles_kind(const uint8_t* data, size_t ntiles, size_t stride) {
    int sym = 0, asym = 0;
    for (size_t t = 0; t < ntiles && !(sym && asym); t++) {
        const int k = q4nx_tile_kind(data + t * stride);
        if (k == Q4NX_TILE_SYMMETRIC) sym = 1;
        else if (k == Q4NX_TILE_ASYMMETRIC) asym = 1;
    }
    if (sym && asym) return -1;
    return sym ? Q4NX_TILE_SYMMETRIC : Q4NX_TILE_ASYMMETRIC;
}

void q4nx_dequant_tile_as(const uint8_t* tile, float* out, int ld, int symmetric) {
    const uint8_t* sc_b = tile;
    const uint8_t* zp_b = tile + Q4NX_TILE_ROWS * Q4NX_GROUPS_PER_ROW * 2; /* +512  */
    const uint8_t* pk = tile + Q4NX_TILE_ROWS * Q4NX_GROUPS_PER_ROW * 4;   /* +1024 */

    for (int r = 0; r < Q4NX_TILE_ROWS; r++) {
        const int h = r / 16;
        const int rr = r % 16;
        for (int c = 0; c < Q4NX_TILE_COLS; c++) {
            const int g = c / Q4NX_GROUP_SIZE;
            const int j = c % Q4NX_GROUP_SIZE;
            const int region = h * Q4NX_GROUPS_PER_ROW + g;
            const uint8_t byte = pk[region * 256 + 8 * j + rr / 2];
            int code = (rr % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            /* scale/zero-point index is GROUP-MAJOR: i = g*32 + r */
            const int i = g * Q4NX_TILE_ROWS + r;
            const float s = bf16_to_f32(le16(sc_b + 2 * i));
            if (symmetric) {
                if (code >= 8) code -= 16;   /* two's complement nibble */
                out[(size_t)r * ld + c] = (float)code * s;
            } else {
                const float z = bf16_to_f32(le16(zp_b + 2 * i));
                out[(size_t)r * ld + c] = (float)code * s + z;
            }
        }
    }
}

void q4nx_dequant_tile(const uint8_t* tile, float* out, int ld) {
    /* One tile with nothing around it: symmetric only on its own evidence, asymmetric otherwise. */
    q4nx_dequant_tile_as(tile, out, ld, q4nx_tile_kind(tile) == Q4NX_TILE_SYMMETRIC);
}

/* The kind of a run of 5,120-byte tiles that belong to one tensor, for the tensor-level readers below: 1 symmetric,
 * 0 asymmetric, or -1 with the refusal on stderr when the tensor holds tiles of both kinds. */
static int tensor_kind_or_refuse(const uint8_t* data, size_t ntiles, size_t stride) {
    const int k = q4nx_tiles_kind(data, ntiles, stride);
    if (k < 0)
        fprintf(stderr, "q4nx: a tensor of %zu 5120-byte tiles holds symmetric tiles (zero-points all 0, a scale below "
                        "zero) and asymmetric ones (a zero-point present); one tensor is one kind, so it is refused\n",
                ntiles);
    return k < 0 ? -1 : (k == Q4NX_TILE_SYMMETRIC);
}

size_t q4nx_tensor_bytes(int rows, int cols) {
    if (rows % Q4NX_TILE_ROWS || cols % Q4NX_TILE_COLS) return 0;
    return (size_t)(rows / Q4NX_TILE_ROWS) * (size_t)(cols / Q4NX_TILE_COLS) *
           (size_t)Q4NX_TILE_BYTES;
}

void q4nx_dequant_tile_q4k(const uint8_t* t, float* out, int ld) {
    /* [0:256] uint8 scales, [256:512] uint8 mins (index g*32+r) */
    const uint8_t* scales = t;
    const uint8_t* mins = t + 256;
    const uint8_t* qs = t + 512;                 /* [256][16] */
    const uint8_t* S_b = t + 4608;               /* [32] bf16 */
    const uint8_t* M_b = t + 4672;               /* [32] bf16 */
    for (int r = 0; r < Q4NX_TILE_ROWS; r++) {
        const float S = bf16_to_f32(le16(S_b + 2 * r));
        const float M = bf16_to_f32(le16(M_b + 2 * r));
        for (int k = 0; k < Q4NX_TILE_COLS; k++) {
            const int g = k / Q4NX_GROUP_SIZE;
            const uint8_t byte = qs[k * 16 + r / 2];
            const int nib = (r % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            const float sc = (float)scales[g * Q4NX_TILE_ROWS + r];
            const float mn = (float)mins[g * Q4NX_TILE_ROWS + r];
            out[(size_t)r * ld + k] = S * sc * (float)nib + M * mn;
        }
    }
}

void q4nx_dequant_tile_q8(const uint8_t* t, float* out, int ld) {
    const uint8_t* d_b = t;                      /* [256] bf16, index g*32+r */
    const int8_t* q = (const int8_t*)(t + 512);  /* byte k*32+r */
    for (int r = 0; r < Q4NX_TILE_ROWS; r++) {
        for (int k = 0; k < Q4NX_TILE_COLS; k++) {
            const int g = k / Q4NX_GROUP_SIZE;
            const float d = bf16_to_f32(le16(d_b + 2 * (g * Q4NX_TILE_ROWS + r)));
            out[(size_t)r * ld + k] = d * (float)q[k * Q4NX_TILE_ROWS + r];
        }
    }
}

void q4nx_dequant_tile_q4s(const uint8_t* t, float* out, int ld) {
    const uint8_t* sc_b = t;          /* [256] bf16, index g*32 + r */
    const uint8_t* qs = t + 512;      /* [256][16], byte c*16 + r/2 */
    for (int r = 0; r < Q4NX_TILE_ROWS; r++) {
        for (int c = 0; c < Q4NX_TILE_COLS; c++) {
            const int g = c / Q4NX_GROUP_SIZE;
            const uint8_t byte = qs[c * 16 + r / 2];
            int code = (r % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            if (code >= 8) code -= 16;   /* two's complement nibble */
            out[(size_t)r * ld + c] = (float)code * bf16_to_f32(le16(sc_b + 2 * (g * Q4NX_TILE_ROWS + r)));
        }
    }
}

void q4nx_dequant_chunk_q4h(const uint8_t* t, float* out, int ld, int symmetric) {
    const uint8_t* sc_b = t;
    const uint8_t* zp_b = t + 128;
    const uint8_t* pk = t + 256;
    for (int r = 0; r < Q4NX_Q4H_ROWS; r++) {
        for (int c = 0; c < Q4NX_Q4H_COLS; c++) {
            const int i = (c / Q4NX_GROUP_SIZE) * Q4NX_Q4H_ROWS + r;
            const uint8_t byte = pk[c * 8 + r / 2];
            int code = (r % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            const float s = bf16_to_f32(le16(sc_b + 2 * i));
            if (symmetric) {
                if (code >= 8) code -= 16;
                out[(size_t)r * ld + c] = (float)code * s;
            } else {
                out[(size_t)r * ld + c] = (float)code * s + bf16_to_f32(le16(zp_b + 2 * i));
            }
        }
    }
}

int q4nx_dequant_tensor_q4h(const uint8_t* data, size_t nbytes, int rows, int k, int cols, float* out) {
    if (!data || !out || rows <= 0 || k <= 0 || cols < k) return -1;
    if (rows % Q4NX_Q4H_ROWS || k % Q4NX_Q4H_COLS) return -1;
    const int ntr = rows / Q4NX_Q4H_ROWS, ntc = k / Q4NX_Q4H_COLS;
    const size_t n = (size_t)ntr * (size_t)ntc;
    if (nbytes != n * (size_t)Q4NX_CHUNK_Q4H) return -1;
    /* One kind per tensor, the 5120-byte rule: symmetric when every zero-point is 0 and some scale is negative. */
    int zero_zp = 1, neg = 0;
    for (size_t i = 0; i < n && (zero_zp || !neg); i++) {
        const uint8_t* c = data + i * Q4NX_CHUNK_Q4H;
        for (int j = 0; j < 64; j++) {
            if (le16(c + 128 + 2 * j) & 0x7fff) zero_zp = 0;
            if (le16(c + 2 * j) & 0x8000) neg = 1;
        }
    }
    const int sym = zero_zp && neg;
    #pragma omp parallel for collapse(2) schedule(static) if(n >= 4)
    for (int tr = 0; tr < ntr; tr++)
        for (int tc = 0; tc < ntc; tc++)
            q4nx_dequant_chunk_q4h(data + ((size_t)tr * ntc + tc) * Q4NX_CHUNK_Q4H,
                                   out + (size_t)tr * Q4NX_Q4H_ROWS * cols + (size_t)tc * Q4NX_Q4H_COLS, cols, sym);
    if (cols > k)
        for (int r = 0; r < rows; r++)
            memset(out + (size_t)r * cols + k, 0, (size_t)(cols - k) * sizeof(float));
    return 0;
}

void q4nx_dequant_chunk_wide(const uint8_t* chunk, int bits, float* out, int ld) {
    float sc[Q4NX_WIDE_ROWS * 8];
    for (int i = 0; i < Q4NX_WIDE_ROWS * 8; i++) sc[i] = bf16_to_f32(le16(chunk + 2 * i));
    const uint8_t* codes = chunk + 256;
    for (int r = 0; r < Q4NX_WIDE_ROWS; r++)
        for (int c = 0; c < Q4NX_WIDE_COLS; c++) {
            int q;
            if (bits == 4) {
                const uint8_t b = codes[c * 8 + r / 2];
                const int v = (r & 1) ? (b >> 4) : (b & 15);
                q = v >= 8 ? v - 16 : v;
            } else {
                q = (int8_t)codes[c * 16 + r];
            }
            out[(size_t)r * ld + c] = (float)q * sc[(c / 32) * Q4NX_WIDE_ROWS + r];
        }
}

int q4nx_dequant_tensor_wide(const uint8_t* data, size_t nbytes, int chunk_bytes, int rows, int k, int cols, float* out) {
    if (!data || !out || rows <= 0 || k <= 0 || cols < k) return -1;
    const int bits = chunk_bytes == Q4NX_CHUNK_Q4W ? 4 : (chunk_bytes == Q4NX_CHUNK_Q8W ? 8 : 0);
    if (!bits || rows % Q4NX_WIDE_ROWS || k % Q4NX_WIDE_COLS) return -1;
    const int ntr = rows / Q4NX_WIDE_ROWS, ntc = k / Q4NX_WIDE_COLS;
    if (nbytes != (size_t)ntr * (size_t)ntc * (size_t)chunk_bytes) return -1;
    #pragma omp parallel for collapse(2) schedule(static) if((size_t)ntr * ntc >= 4)
    for (int tr = 0; tr < ntr; tr++)
        for (int tc = 0; tc < ntc; tc++)
            q4nx_dequant_chunk_wide(data + ((size_t)tr * ntc + tc) * (size_t)chunk_bytes, bits,
                                    out + (size_t)tr * Q4NX_WIDE_ROWS * cols + (size_t)tc * Q4NX_WIDE_COLS, cols);
    if (cols > k)
        for (int r = 0; r < rows; r++)
            memset(out + (size_t)r * cols + k, 0, (size_t)(cols - k) * sizeof(float));
    return 0;
}

static const float kE2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

/* 2^(e - 128), exact (powers of two halve and double without rounding), with no libm */
static inline float e8m0_scale(uint8_t e) {
    float p = 1.0f;
    for (int n = (int)e - 128; n > 0; n--) p *= 2.0f;
    for (int n = (int)e - 128; n < 0; n++) p *= 0.5f;
    return p;
}

static inline int d2560_code(const uint8_t* codes, int r, int c) {
    const uint8_t b = codes[(r / 16) * 1024 + c * 8 + (r % 16) / 2];
    return (r & 1) ? (b >> 4) : (b & 15);
}

void q4nx_dequant_chunk_d2560(const uint8_t* chunk, float* out, int ld) {
    const uint8_t* codes = chunk + 512;
    for (int r = 0; r < Q4NX_D2560_ROWS; r++)
        for (int c = 0; c < Q4NX_D2560_COLS; c++) {
            const int slot = (c / 32) * Q4NX_D2560_ROWS + r;
            const float s = bf16_to_f32(le16(chunk + 2 * slot)), z = bf16_to_f32(le16(chunk + 256 + 2 * slot));
            out[(size_t)r * ld + c] = (float)d2560_code(codes, r, c) * s + z;
        }
}

void q4nx_dequant_chunk_mx2560(const uint8_t* chunk, float* out, int ld) {
    const uint8_t* codes = chunk + 512;
    for (int r = 0; r < Q4NX_D2560_ROWS; r++)
        for (int c = 0; c < Q4NX_D2560_COLS; c++) {
            const float s = e8m0_scale(chunk[(c / 32) * Q4NX_D2560_ROWS + r]);
            out[(size_t)r * ld + c] = kE2M1[d2560_code(codes, r, c)] * s;
        }
}

int q4nx_dequant_tensor_d2560(const uint8_t* data, size_t nbytes, int RB, int CB, int S, int rows, int k, int cols,
                              float* out) {
    if (!data || !out || RB <= 0 || CB <= 0 || S <= 0 || rows <= 0 || k <= 0 || cols < k) return -1;
    if (nbytes != (size_t)RB * CB * S * Q4NX_CHUNK_D2560) return -1;
    const int full_rows = RB * S * Q4NX_D2560_ROWS, full_k = CB * Q4NX_D2560_COLS;
    if (rows > full_rows || k > full_k) return -1;
    float* tmp = (float*)malloc(sizeof(float) * Q4NX_D2560_ROWS * Q4NX_D2560_COLS * (size_t)CB);
    if (!tmp) return -1;
    for (int rb = 0; rb < RB; rb++)
        for (int s = 0; s < S; s++) {
            const int r0 = (rb * S + s) * Q4NX_D2560_ROWS;
            if (r0 >= rows) break;
            for (int cb = 0; cb < CB; cb++)
                q4nx_dequant_chunk_d2560(data + (((size_t)rb * CB + cb) * S + s) * Q4NX_CHUNK_D2560,
                                         tmp + (size_t)cb * Q4NX_D2560_COLS, full_k);
            for (int r = 0; r < Q4NX_D2560_ROWS && r0 + r < rows; r++) {
                float* dst = out + (size_t)(r0 + r) * cols;
                memcpy(dst, tmp + (size_t)r * full_k, sizeof(float) * (size_t)k);
                if (cols > k) memset(dst + k, 0, sizeof(float) * (size_t)(cols - k));
            }
        }
    free(tmp);
    return 0;
}

int q4nx_dequant_tensor_chunked(const uint8_t* data, size_t nbytes, int rows,
                                int cols, int chunk_bytes, float* out) {
    if (!data || !out || rows <= 0 || cols <= 0) return -1;
    if (rows % Q4NX_TILE_ROWS || cols % Q4NX_TILE_COLS) return -1;
    if (chunk_bytes != Q4NX_TILE_BYTES && chunk_bytes != Q4NX_CHUNK_Q4K &&
        chunk_bytes != Q4NX_CHUNK_Q8 && chunk_bytes != Q4NX_CHUNK_Q4S)
        return -1;
    const int ntr = rows / Q4NX_TILE_ROWS, ntc = cols / Q4NX_TILE_COLS;
    if (nbytes < (size_t)ntr * (size_t)ntc * (size_t)chunk_bytes) return -1;
    int sym = 0;
    if (chunk_bytes == Q4NX_TILE_BYTES) {
        sym = tensor_kind_or_refuse(data, (size_t)ntr * (size_t)ntc, Q4NX_TILE_BYTES);
        if (sym < 0) return -1;
    }
    // Each tile writes a disjoint output region, so the tile grid parallelizes.
    // `tmp` is declared per-tile (inside the innermost body) so it is not shared.
    #pragma omp parallel for collapse(2) schedule(static) if(ntr * ntc >= 4)
    for (int tr = 0; tr < ntr; tr++) {
        for (int tc = 0; tc < ntc; tc++) {
            float tmp[Q4NX_TILE_ROWS * Q4NX_TILE_COLS];
            const uint8_t* c = data + (size_t)(tr * ntc + tc) * (size_t)chunk_bytes;
            if (chunk_bytes == Q4NX_CHUNK_Q4K)
                q4nx_dequant_tile_q4k(c, tmp, Q4NX_TILE_COLS);
            else if (chunk_bytes == Q4NX_CHUNK_Q8)
                q4nx_dequant_tile_q8(c, tmp, Q4NX_TILE_COLS);
            else if (chunk_bytes == Q4NX_CHUNK_Q4S)
                q4nx_dequant_tile_q4s(c, tmp, Q4NX_TILE_COLS);
            else
                q4nx_dequant_tile_as(c, tmp, Q4NX_TILE_COLS, sym);
            for (int r = 0; r < Q4NX_TILE_ROWS; r++)
                memcpy(out + (size_t)(tr * Q4NX_TILE_ROWS + r) * cols +
                           (size_t)tc * Q4NX_TILE_COLS,
                       tmp + (size_t)r * Q4NX_TILE_COLS,
                       Q4NX_TILE_COLS * sizeof(float));
        }
    }
    return 0;
}

/* engine#1784: see the declaration. The tile grid and its ordering are q4nx_dequant_tensor_chunked()'s -- a row lives
 * inside one tile-row, at offset r % Q4NX_TILE_ROWS of each tile across the row -- so the two cannot disagree about
 * where a row starts. No OpenMP: this is called per token, with one row to do. */
int q4nx_dequant_tile_row(const uint8_t* data, size_t nbytes, int rows, int cols, int chunk_bytes, int r, float* out) {
    if (!data || !out || rows <= 0 || cols <= 0) return -1;
    if (r < 0 || r >= rows) return -1;
    if (rows % Q4NX_TILE_ROWS || cols % Q4NX_TILE_COLS) return -1;
    if (chunk_bytes != Q4NX_TILE_BYTES && chunk_bytes != Q4NX_CHUNK_Q4K &&
        chunk_bytes != Q4NX_CHUNK_Q8 && chunk_bytes != Q4NX_CHUNK_Q4S)
        return -1;
    const int ntr = rows / Q4NX_TILE_ROWS, ntc = cols / Q4NX_TILE_COLS;
    if (nbytes < (size_t)ntr * (size_t)ntc * (size_t)chunk_bytes) return -1;
    const int tr = r / Q4NX_TILE_ROWS, rr = r % Q4NX_TILE_ROWS;
    float tmp[Q4NX_TILE_ROWS * Q4NX_TILE_COLS];
    /* Per token, so the kind is taken from this tile-row's own tiles (ntc of them), not from the whole table. */
    int sym = 0;
    if (chunk_bytes == Q4NX_TILE_BYTES) {
        sym = tensor_kind_or_refuse(data + (size_t)tr * (size_t)ntc * Q4NX_TILE_BYTES, (size_t)ntc, Q4NX_TILE_BYTES);
        if (sym < 0) return -1;
    }
    for (int tc = 0; tc < ntc; tc++) {
        const uint8_t* c = data + (size_t)(tr * ntc + tc) * (size_t)chunk_bytes;
        if (chunk_bytes == Q4NX_CHUNK_Q4K)
            q4nx_dequant_tile_q4k(c, tmp, Q4NX_TILE_COLS);
        else if (chunk_bytes == Q4NX_CHUNK_Q8)
            q4nx_dequant_tile_q8(c, tmp, Q4NX_TILE_COLS);
        else if (chunk_bytes == Q4NX_CHUNK_Q4S)
            q4nx_dequant_tile_q4s(c, tmp, Q4NX_TILE_COLS);
        else
            q4nx_dequant_tile_as(c, tmp, Q4NX_TILE_COLS, sym);
        memcpy(out + (size_t)tc * Q4NX_TILE_COLS, tmp + (size_t)rr * Q4NX_TILE_COLS,
               Q4NX_TILE_COLS * sizeof(float));
    }
    return 0;
}

int q4nx_tile_row_kind(size_t nbytes, int rows, int cols) {
    if (rows <= 0 || cols <= 0 || rows % Q4NX_TILE_ROWS || cols % Q4NX_TILE_COLS) return 0;
    const size_t tiles = (size_t)(rows / Q4NX_TILE_ROWS) * (size_t)(cols / Q4NX_TILE_COLS);
    const int kinds[4] = {Q4NX_TILE_BYTES, Q4NX_CHUNK_Q4K, Q4NX_CHUNK_Q8, Q4NX_CHUNK_Q4S};
    for (int i = 0; i < 4; i++)
        if (nbytes == tiles * (size_t)kinds[i]) return kinds[i];
    return 0;
}

int q4nx_packed_grid(size_t nbytes, int rows, int cols, int* cb_out, int* rp_out, int* cp_out) {
    const int kinds[3] = {Q4NX_TILE_BYTES, Q4NX_CHUNK_Q4K, Q4NX_CHUNK_Q8};
    if (rows <= 0 || cols <= 0 || rows % Q4NX_TILE_ROWS || cols % Q4NX_TILE_COLS) return 0;
    const int ntr = rows / Q4NX_TILE_ROWS, ntc = cols / Q4NX_TILE_COLS;
    size_t best = 0;
    int best_cb = 0, best_rp = 0, best_cp = 0, ties = 0;
    for (int k = 0; k < 3; k++) {
        if (nbytes % (size_t)kinds[k]) continue;
        const size_t np = nbytes / (size_t)kinds[k];       /* tiles in the container */
        for (size_t d = 1; d * d <= np; d++) {
            if (np % d) continue;
            const size_t cand[2][2] = {{d, np / d}, {np / d, d}};
            for (int q = 0; q < 2; q++) {
                const size_t rpt = cand[q][0], cpt = cand[q][1];
                if (rpt < (size_t)ntr || cpt < (size_t)ntc) continue;
                const size_t excess = (rpt - (size_t)ntr) + (cpt - (size_t)ntc);
                if (best && excess > best) continue;
                if (best && excess == best &&
                    (best_rp != (int)(rpt * Q4NX_TILE_ROWS) || best_cp != (int)(cpt * Q4NX_TILE_COLS))) {
                    ties++;                                     /* a different grid, same least excess */
                    continue;
                }
                if (best && excess == best) continue;           /* the same grid seen twice */
                best = excess;
                best_cb = kinds[k];
                best_rp = (int)(rpt * Q4NX_TILE_ROWS);
                best_cp = (int)(cpt * Q4NX_TILE_COLS);
                ties = 0;
            }
        }
    }
    if (!best || ties || best > Q4NX_PACKED_MAX_EXCESS) return 0;
    if (best_rp == rows && best_cp == cols) return 0;           /* no padding: the plain decoder */
    *cb_out = best_cb;
    *rp_out = best_rp;
    *cp_out = best_cp;
    return 1;
}

int q4nx_dequant_tensor_packed(const uint8_t* data, size_t nbytes, int rows, int cols,
                               int rows_packed, int cols_packed, int chunk_bytes, float* out) {
    if (!data || !out || rows <= 0 || cols <= 0) return -1;
    if (rows % Q4NX_TILE_ROWS || cols % Q4NX_TILE_COLS) return -1;
    if (rows_packed < rows || cols_packed < cols) return -1;
    if (rows_packed % Q4NX_TILE_ROWS || cols_packed % Q4NX_TILE_COLS) return -1;
    if (chunk_bytes != Q4NX_TILE_BYTES && chunk_bytes != Q4NX_CHUNK_Q4K &&
        chunk_bytes != Q4NX_CHUNK_Q8 && chunk_bytes != Q4NX_CHUNK_Q4S)
        return -1;
    if (rows_packed == rows && cols_packed == cols)
        return q4nx_dequant_tensor_chunked(data, nbytes, rows, cols, chunk_bytes, out);
    const int ntr = rows / Q4NX_TILE_ROWS, ntc = cols / Q4NX_TILE_COLS;
    const int ntr_p = rows_packed / Q4NX_TILE_ROWS, ntc_p = cols_packed / Q4NX_TILE_COLS;
    if (nbytes < (size_t)ntr_p * (size_t)ntc_p * (size_t)chunk_bytes) return -1;
    int sym = 0;
    if (chunk_bytes == Q4NX_TILE_BYTES) {
        sym = tensor_kind_or_refuse(data, (size_t)ntr_p * (size_t)ntc_p, Q4NX_TILE_BYTES);
        if (sym < 0) return -1;
    }
    // Read geometry is the container's grid (ntc_p is the tile-row stride); write geometry is the
    // model's, so a caller's buffer and its row stride stay exactly what they were before.
    #pragma omp parallel for collapse(2) schedule(static) if(ntr * ntc >= 4)
    for (int tr = 0; tr < ntr; tr++) {
        for (int tc = 0; tc < ntc; tc++) {
            float tmp[Q4NX_TILE_ROWS * Q4NX_TILE_COLS];
            const uint8_t* c = data + (size_t)(tr * ntc_p + tc) * (size_t)chunk_bytes;
            if (chunk_bytes == Q4NX_CHUNK_Q4K)
                q4nx_dequant_tile_q4k(c, tmp, Q4NX_TILE_COLS);
            else if (chunk_bytes == Q4NX_CHUNK_Q8)
                q4nx_dequant_tile_q8(c, tmp, Q4NX_TILE_COLS);
            else if (chunk_bytes == Q4NX_CHUNK_Q4S)
                q4nx_dequant_tile_q4s(c, tmp, Q4NX_TILE_COLS);
            else
                q4nx_dequant_tile_as(c, tmp, Q4NX_TILE_COLS, sym);
            for (int r = 0; r < Q4NX_TILE_ROWS; r++)
                memcpy(out + (size_t)(tr * Q4NX_TILE_ROWS + r) * cols +
                           (size_t)tc * Q4NX_TILE_COLS,
                       tmp + (size_t)r * Q4NX_TILE_COLS,
                       Q4NX_TILE_COLS * sizeof(float));
        }
    }
    return 0;
}

int q4nx_dequant_plain_bf16(const uint8_t* data, int rows, int cols,
                           float* out) {
    if (!data || !out || rows <= 0 || cols <= 0) return -1;
    const size_t n = (size_t)rows * (size_t)cols;
    for (size_t i = 0; i < n; i++) out[i] = bf16_to_f32(le16(data + 2 * i));
    return 0;
}

int q4nx_dequant_tensor(const uint8_t* data, size_t nbytes, int rows, int cols,
                        float* out) {
    if (!data || !out || rows <= 0 || cols <= 0) return -1;
    /* A PLAIN bf16 tensor is exactly two bytes per element and carries no tile header:
     * `tools/safetensors_to_q4nx.py pack --dtype bf16` writes one.  The size test below
     * only asked for `nbytes >= need`, so such a tensor was accepted and its first
     * nbytes/5120 tiles were read as q4_1 — it loaded, computed, and answered on garbage
     * (engine#875: a bf16 container is 2/0.625 = 3.2x the tile size, so two thirds of the
     * buffer were never read and the rest was noise).  Recognise it here, before the
     * tile geometry test, because plain bf16 needs no 32x256 alignment. */
    if (nbytes == (size_t)rows * (size_t)cols * sizeof(uint16_t))
        return q4nx_dequant_plain_bf16(data, rows, cols, out);
    if (rows % Q4NX_TILE_ROWS || cols % Q4NX_TILE_COLS) return -1;
    const size_t need = q4nx_tensor_bytes(rows, cols);
    /* Exact, not `>=`: a container family whose chunk size nobody taught this decoder
     * must be refused rather than read as q4_1 tiles (engine#875). */
    if (need == 0 || nbytes != need) return -1;

    const int ntr = rows / Q4NX_TILE_ROWS;
    const int ntc = cols / Q4NX_TILE_COLS;
    const int sym = tensor_kind_or_refuse(data, (size_t)ntr * (size_t)ntc, Q4NX_TILE_BYTES);
    if (sym < 0) return -1;

    #pragma omp parallel for collapse(2) schedule(static) if(ntr * ntc >= 4)
    for (int tr = 0; tr < ntr; tr++) {
        for (int tc = 0; tc < ntc; tc++) {
            float tmp[Q4NX_TILE_ROWS * Q4NX_TILE_COLS];
            const uint8_t* tile =
                data + (size_t)(tr * ntc + tc) * (size_t)Q4NX_TILE_BYTES;
            q4nx_dequant_tile_as(tile, tmp, Q4NX_TILE_COLS, sym);
            for (int r = 0; r < Q4NX_TILE_ROWS; r++) {
                memcpy(out + (size_t)(tr * Q4NX_TILE_ROWS + r) * (size_t)cols +
                           (size_t)tc * Q4NX_TILE_COLS,
                       tmp + (size_t)r * Q4NX_TILE_COLS,
                       (size_t)Q4NX_TILE_COLS * sizeof(float));
            }
        }
    }
    return 0;
}

/* Gemma4 stores `model.embed_tokens.weight` and `per_layer_token_embd.weight` as
 * PLAIN int8 (rows*cols bytes, no 5120-byte tile header) with a sibling
 * `<name>.scale` F32 [rows, cols/32] tensor:
 *
 *     value[r][c] = int8[r][c] * scale[r][c / 32]
 *
 * Reading such a tensor through q4nx_dequant_tensor() mismatches its byte count
 * and fails, which is why the Gemma4 embedding and per-layer token table need
 * this path. */
int q4nx_dequant_plain_i8(const uint8_t* data, const float* scale, int rows,
                          int cols, float* out) {
    if (!data || !out || rows <= 0 || cols <= 0 || (cols % Q4NX_GROUP_SIZE) != 0)
        return -1;
    const int groups = cols / Q4NX_GROUP_SIZE;
    for (int r = 0; r < rows; r++) {
        const int8_t* src = (const int8_t*)(data + (size_t)r * (size_t)cols);
        const float* sc = scale ? scale + (size_t)r * (size_t)groups : NULL;
        float* dst = out + (size_t)r * (size_t)cols;
        for (int c = 0; c < cols; c++) {
            const float v = (float)src[c];
            dst[c] = sc ? v * sc[c / Q4NX_GROUP_SIZE] : v;
        }
    }
    return 0;
}

/* engine#1784: the element-wise form of the loop above, for a loader that reads one row at a time out of the
 * container's mapping instead of materialising the table. Kept beside the loop so the two cannot drift: the group
 * index and the scale row are computed the same way, and out-of-range coordinates are 0.0f (the caller's pad
 * columns read as zero, which is what the fully decoded table holds there). */
float q4nx_plain_i8_at(const uint8_t* data, const float* scale, int rows, int cols, int row, int col) {
    if (!data || rows <= 0 || cols <= 0 || (cols % Q4NX_GROUP_SIZE) != 0) return 0.0f;
    if (row < 0 || row >= rows || col < 0 || col >= cols) return 0.0f;
    const int groups = cols / Q4NX_GROUP_SIZE;
    const float v = (float)((const int8_t*)data)[(size_t)row * (size_t)cols + (size_t)col];
    return scale ? v * scale[(size_t)row * (size_t)groups + (size_t)col / Q4NX_GROUP_SIZE] : v;
}
