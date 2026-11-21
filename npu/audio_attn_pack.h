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

// audio_attn_pack.h -- engine#1783: the host side of the whisper encoder's attention on the NPU (ONEBIT_AUDIO_NPU_ATTN=1):
// the three buffers of the sealed ELF's attention instance, packed from (and unpacked into) the f32 tensors the fused
// host path reads and writes.
//
//   q  [H][S][64]   the graph's scaled query           -> Q   bf16, times log2(e), one 8 KB block per (task, column, row)
//   kt [H][64][S]   the graph's transposed key          -> KV  bf16, per head and 32-key chunk a K block and a V^T block
//   v  [H][S][64]   the graph's value                   ->     (4 KB each)
//   O  int32 bits of f32, per (task, column, row) 64 rows x 128: columns 0..63 = the row's unnormalised output, column
//      64 = its softmax denominator                     -> out [H][S][64] = O / denominator
//
// A task is (head group hg, row group rg): the array's `rows` core rows take heads hg*rows .. +rows-1 and its `cols`
// columns take 64 query rows each, so a task covers cols*64 query rows of rows heads. Rows past S are zero and their
// outputs are never read; keys past S are zero here and masked on the device.
//
// The functions are plain loops over one unit (a task's Q, a head's K/V, a task's output); the caller spreads the units
// over threads (audio_parallel.h). Nothing here touches the device.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace onebit::audio::attn_pack {

struct Geom {
    int H = 12, S = 1500, D = 64;
    int rows = 4, cols = 8;
    int row_groups() const { return (S + cols * 64 - 1) / (cols * 64); }
    int chunks() const { return (S + 31) / 32; }
    int tasks() const { return (H / rows) * row_groups(); }
    size_t q_bytes() const { return size_t(tasks()) * cols * rows * 8192; }
    size_t kv_bytes() const { return size_t(H) * chunks() * 2 * 4096; }
    size_t o_words() const { return size_t(tasks()) * cols * rows * 64 * 128; }
    bool fits() const { return D == 64 && H > 0 && S > 0 && H % rows == 0; }
};

// f32 -> bf16, round to nearest even (NaN stays NaN)
inline uint16_t bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

constexpr float kLog2e = 1.4426950408889634f;

// Q for task `ti`, column `c`: [rows][i 8][dt 8][8][8], element (r, cc) of tile (i, dt) = q[h][row0 + i*8 + r][dt*8 + cc]
// * log2(e), h = hg*rows + j and row0 = rg*cols*64 + c*64 for core row j.
inline void pack_q_col(const Geom& g, const float* q, int ti, int c, uint16_t* Q) {
    const int nrg = g.row_groups(), hg = ti / nrg, rg = ti % nrg;
    for (int j = 0; j < g.rows; j++) {
        const int h = hg * g.rows + j, row0 = rg * g.cols * 64 + c * 64;
        uint16_t* o = Q + ((size_t(ti) * g.cols + c) * g.rows + j) * 4096;
        for (int i = 0; i < 8; i++)
            for (int r = 0; r < 8; r++) {
                const int row = row0 + i * 8 + r;
                const float* src = row < g.S ? q + (size_t(h) * g.S + row) * 64 : nullptr;
                for (int dt = 0; dt < 8; dt++)
                    for (int cc = 0; cc < 8; cc++)
                        o[((i * 8 + dt) * 8 + r) * 8 + cc] = src ? bf16(src[dt * 8 + cc] * kLog2e) : 0;
            }
    }
}
inline void pack_q_task(const Geom& g, const float* q, int ti, uint16_t* Q) {
    for (int c = 0; c < g.cols; c++) pack_q_col(g, q, ti, c, Q);
}

// The mem tile hands a 4 KB block to the cores as 8 x 8-byte microtiles; the block is written so that the core reads
// the tile order below: 8-byte unit u = (a*8 + b)*8 + r of the core's view sits at unit a*64 + r*8 + b of the block.
inline size_t a_unit(size_t u) {
    const size_t a = u / 64, b = (u / 8) % 8, r = u % 8;
    return a * 64 + r * 8 + b;
}

// K and V^T of head h, chunk ch (32 keys): [K | V^T][4 KB] at KV + ((h*chunks + ch)*2)*2048. The core's view of K =
// [kt 4][dt 8][8][8] (row = key, col = d) and of V^T = [dt 8][kt 4][8][8] (row = d, col = key). Both sources are read
// sequentially: kt by its rows (one d, 32 consecutive keys), v by its rows (one key, 64 consecutive d).
inline void pack_kv_chunk(const Geom& g, const float* kt, const float* v, int h, int ch, uint16_t* KV) {
    uint16_t want[2048];
    uint16_t* base = KV + (size_t(h) * g.chunks() + ch) * 2 * 2048;
    const int k0 = ch * 32, nk = g.S - k0 < 32 ? g.S - k0 : 32;
    for (int d = 0; d < 64; d++) {
        const float* row = kt + (size_t(h) * 64 + d) * g.S + k0;
        const int dt = d / 8, cc = d % 8;
        for (int kk = 0; kk < 32; kk++)
            want[(((kk / 8) * 8 + dt) * 8 + kk % 8) * 8 + cc] = kk < nk ? bf16(row[kk]) : 0;
    }
    for (size_t u = 0; u < 512; u++) std::memcpy(base + a_unit(u) * 4, want + u * 4, 8);
    for (int kk = 0; kk < 32; kk++) {
        const float* row = kk < nk ? v + (size_t(h) * g.S + k0 + kk) * 64 : nullptr;
        const int ktl = kk / 8, cc = kk % 8;
        for (int d = 0; d < 64; d++) want[(((d / 8) * 4 + ktl) * 8 + d % 8) * 8 + cc] = row ? bf16(row[d]) : 0;
    }
    for (size_t u = 0; u < 512; u++) std::memcpy(base + 2048 + a_unit(u) * 4, want + u * 4, 8);
}
inline void pack_kv_head(const Geom& g, const float* kt, const float* v, int h, uint16_t* KV) {
    for (int ch = 0; ch < g.chunks(); ch++) pack_kv_chunk(g, kt, v, h, ch, KV);
}

// The output of task `ti`, column `c` into out [H][S][64]: O / denominator, rows < S only.
inline void unpack_o_col(const Geom& g, const int32_t* O, int ti, int c, float* out) {
    const int nrg = g.row_groups(), hg = ti / nrg, rg = ti % nrg;
    for (int j = 0; j < g.rows; j++) {
        const int h = hg * g.rows + j, row0 = rg * g.cols * 64 + c * 64;
        const float* blk = reinterpret_cast<const float*>(O + ((size_t(ti) * g.cols + c) * g.rows + j) * 64 * 128);
        for (int r = 0; r < 64 && row0 + r < g.S; r++) {
            const float* src = blk + size_t(r) * 128;
            const float inv = 1.0f / src[64];
            float* dst = out + (size_t(h) * g.S + row0 + r) * 64;
            for (int d = 0; d < 64; d++) dst[d] = src[d] * inv;
        }
    }
}
inline void unpack_o_task(const Geom& g, const int32_t* O, int ti, float* out) {
    for (int c = 0; c < g.cols; c++) unpack_o_col(g, O, ti, c, out);
}

}  // namespace onebit::audio::attn_pack
