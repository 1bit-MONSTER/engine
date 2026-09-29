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

void q4nx_dequant_tile(const uint8_t* tile, float* out, int ld) {
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
            const int code = (rr % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            /* scale/zero-point index is GROUP-MAJOR: i = g*32 + r */
            const int i = g * Q4NX_TILE_ROWS + r;
            const float s = bf16_to_f32(le16(sc_b + 2 * i));
            const float z = bf16_to_f32(le16(zp_b + 2 * i));
            out[(size_t)r * ld + c] = (float)code * s + z;
        }
    }
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

int q4nx_dequant_tensor_chunked(const uint8_t* data, size_t nbytes, int rows,
                                int cols, int chunk_bytes, float* out) {
    if (!data || !out || rows <= 0 || cols <= 0) return -1;
    if (rows % Q4NX_TILE_ROWS || cols % Q4NX_TILE_COLS) return -1;
    if (chunk_bytes != Q4NX_TILE_BYTES && chunk_bytes != Q4NX_CHUNK_Q4K &&
        chunk_bytes != Q4NX_CHUNK_Q8)
        return -1;
    const int ntr = rows / Q4NX_TILE_ROWS, ntc = cols / Q4NX_TILE_COLS;
    if (nbytes < (size_t)ntr * (size_t)ntc * (size_t)chunk_bytes) return -1;
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
            else
                q4nx_dequant_tile(c, tmp, Q4NX_TILE_COLS);
            for (int r = 0; r < Q4NX_TILE_ROWS; r++)
                memcpy(out + (size_t)(tr * Q4NX_TILE_ROWS + r) * cols +
                           (size_t)tc * Q4NX_TILE_COLS,
                       tmp + (size_t)r * Q4NX_TILE_COLS,
                       Q4NX_TILE_COLS * sizeof(float));
        }
    }
    return 0;
}

int q4nx_dequant_tensor(const uint8_t* data, size_t nbytes, int rows, int cols,
                        float* out) {
    if (!data || !out || rows <= 0 || cols <= 0) return -1;
    if (rows % Q4NX_TILE_ROWS || cols % Q4NX_TILE_COLS) return -1;
    const size_t need = q4nx_tensor_bytes(rows, cols);
    if (need == 0 || nbytes < need) return -1;

    const int ntr = rows / Q4NX_TILE_ROWS;
    const int ntc = cols / Q4NX_TILE_COLS;

    #pragma omp parallel for collapse(2) schedule(static) if(ntr * ntc >= 4)
    for (int tr = 0; tr < ntr; tr++) {
        for (int tc = 0; tc < ntc; tc++) {
            float tmp[Q4NX_TILE_ROWS * Q4NX_TILE_COLS];
            const uint8_t* tile =
                data + (size_t)(tr * ntc + tc) * (size_t)Q4NX_TILE_BYTES;
            q4nx_dequant_tile(tile, tmp, Q4NX_TILE_COLS);
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
