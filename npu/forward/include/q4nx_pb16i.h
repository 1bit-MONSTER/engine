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
//
// q4nx_pb16i.h -- the "pb16i" weight image: integer-exact ternary decode weights.
//
// A ternary (PTQ1_0) model stores every weight as code * d, code in {-1,0,+1}, with
// one fp16 d per 128 K rows per output column.  The engine's int8 path folds d into
// an int8 code and streams 2 bytes per weight (25.6 GB per 27B token).  The pb16i
// image instead streams the container's own trits, packed 2 per byte, plus one int16
// per (32 K rows, column) that carries d rescaled into that column's window:
//
//    w[k][j] = code[k][j] * ms[g][j] * 2^e_ref[j],   g = k / 32
//
// e_ref[j] = min over the column's blocks of d's exponent, ms = m * 2^(e - e_ref) is
// exactly representable in int16, and e_ref travels on the host (one int16 per
// column).  The device then sums the integer product exactly -- int32 when the
// converter's worst-case |S| bound is below 2^31, otherwise as two int32 planes
// (lo, hi) with S = hi * 2^21 + lo -- and the host applies the only rounding in the
// chain: C[i][j] = float32(S[i][j] * 2^e_ref[j]).  Nothing on the device rounds, so
// the result does not depend on the NPU's non-IEEE fp32 (OKF aie2p-fp32-add-not-ieee).
//
// Image geometry.  K % 256 == 0 and N % 1024 == 0.  32 cores = 8 columns x 4 rows;
// core (c,r) owns CW = N/32 contiguous output columns for all of K, and its stream is
// one contiguous run of the image, so a core walks forward through memory.  A tile is
// one 256-K x 32-N block of that stream, 4,608 bytes:
//
//   [0, 4096)     int4 codes, [kb 16][nb 2][16 r][16 c]; element e = r*16+c of a
//                 block in byte e/2, even e in the low nibble (two's complement)
//   [4096, 4608)  int16 ms, little-endian, [nb 2][g 8][16 c]
//
// and the tiles are ordered for c in 8, for ki in K/256, for nt in CW/32, for r in 4.
// The side table `<tensor>.ref` is e_ref int16[N] little-endian followed by one mode
// byte (0 = int32 S, 1 = int64 S as two planes).
//
// This file is the contract between the packer (tools/ and the converter), the host
// dequantisation, and the kernel's own tap: a mismatch here silently multiplies the
// wrong pairs.  tests/npu_pb16i_layout_test.cpp pins it without a device.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace q4nx_pb16i {

constexpr int COLS = 8;          // core columns
constexpr int ROWS = 4;          // core rows (memtile row split)
constexpr int KC = 256;          // K rows per tile
constexpr int NW = 32;           // N columns per tile
constexpr int CODE_BYTES = 4096; // KC*NW/2
constexpr int TILE = 4608;       // CODE_BYTES + (KC/32)*NW*2

// A geometry that has a pb16i image at all.
inline bool fits(int K, int N) {
    return K > 0 && N > 0 && K % KC == 0 && N % (COLS * ROWS * NW) == 0;
}

inline size_t tiles(int K, int N) {
    return (size_t)(K / KC) * (size_t)(N / NW);
}

// The image size, both ways of saying it: the tile count times 4,608, and the
// converter's own accounting (codes + ms).  They agree exactly when fits() holds.
inline size_t w_bytes(int K, int N) {
    return tiles(K, N) * (size_t)TILE;
}

// Where weight element (i,j) lives in the image: the byte holding its int4 code, and
// the byte of its int16 ms entry.  Undefined unless fits(K,N).
struct offsets {
    size_t code;   // byte with the element's low nibble when the code is at an even index
    bool   high;   // true when the code is the HIGH nibble of that byte
    size_t ms;     // first byte of its little-endian int16
};

inline offsets index(size_t i, size_t j, int K, int N) {
    const int cw = N / (COLS * ROWS);          // columns per core
    const size_t ki = i / KC;
    const int kk = (int)(i % KC);
    const int c = (int)(j / (size_t)(N / COLS));
    const int jr = (int)(j % (size_t)(N / COLS));
    const int r = jr / cw;
    const int jn = jr % cw;
    const int nt = jn / NW;
    const int nb = (jn % NW) / 16, cc = jn % 16;
    const size_t tile = (((size_t)c * (size_t)(K / KC) + ki) * (size_t)(cw / NW) + (size_t)nt) *
                            (size_t)ROWS + (size_t)r;
    const int e = ((kk / 16) * 2 + nb) * 256 + (kk % 16) * 16 + cc;   // code index in the tile
    offsets o;
    o.code = tile * (size_t)TILE + (size_t)(e / 2);
    o.high = (e & 1) != 0;
    o.ms = tile * (size_t)TILE + (size_t)CODE_BYTES +
           (size_t)2 * (size_t)((nb * 8 + kk / 32) * 16 + cc);
    return o;
}

// The element's code and ms as stored, read back out of an image.
inline int code_of(const uint8_t* image, size_t i, size_t j, int K, int N) {
    const offsets o = index(i, j, K, N);
    const uint8_t b = image[o.code];
    const int nib = o.high ? (b >> 4) : (b & 0x0F);
    return nib >= 8 ? nib - 16 : nib;
}

inline int16_t ms_of(const uint8_t* image, size_t i, size_t j, int K, int N) {
    const offsets o = index(i, j, K, N);
    int16_t v;
    std::memcpy(&v, image + o.ms, sizeof(v));
    return v;
}

// The side table: e_ref int16[N] little-endian, then the mode byte.
inline size_t ref_bytes(int N) { return (size_t)N * 2 + 1; }

inline bool parse_ref(const uint8_t* buf, size_t nbytes, int N,
                      std::vector<int16_t>* eref, int* mode) {
    if (!buf || !eref || !mode || nbytes != ref_bytes(N)) return false;
    eref->assign((size_t)N, 0);
    for (int j = 0; j < N; j++) {
        const uint8_t lo = buf[(size_t)j * 2], hi = buf[(size_t)j * 2 + 1];
        eref->at((size_t)j) = (int16_t)(uint16_t)((uint16_t)lo | ((uint16_t)hi << 8));
    }
    *mode = buf[(size_t)N * 2];
    return *mode == 0 || *mode == 1;
}

// The host's single rounding: the exact integer S, scaled by 2^e_ref, rounded once to
// float32.  (double holds |S| < 2^53 exactly and the power-of-two scale exactly, so
// this is one IEEE rounding, not two.)
inline float scale_i64(int64_t s, int16_t e_ref) {
    return (float)std::ldexp((double)s, (int)e_ref);
}

// Merge the device's two planes back into S (int32 mode: one plane).
inline int64_t sum_of(const int32_t* lo, const int32_t* hi, int mode) {
    if (!mode) return (int64_t)*lo;
    return ((int64_t)*hi << 21) + (int64_t)*lo;
}

// The sidecar files a container's pb16i set is keyed on: blk_<layer>_<tag>_weight.
inline std::string sidecar_stem(int layer, const char* tag) {
    return std::string("blk_") + std::to_string(layer) + "_" + tag + "_weight";
}

}  // namespace q4nx_pb16i
