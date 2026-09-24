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
#include "q4nx.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace onebit::npu::q4nx {

namespace {

float bf16(const uint8_t* p) {
    const uint32_t u = uint32_t(p[0] | (p[1] << 8)) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

uint16_t to_bf16(float f) {  // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

void put16(uint8_t* p, uint16_t v) {
    p[0] = uint8_t(v);
    p[1] = uint8_t(v >> 8);
}

// the q4_1 code byte and nibble of row r, column c
size_t q41_code_at(int r, int c) { return 1024 + size_t((r / 16) * 8 + c / kGroup) * 256 + 8 * (c % kGroup) + (r % 16) / 2; }

}  // namespace

Chunk chunk_of(const Tensor& t) {
    const int64_t last = t.shape.empty() ? 0 : t.shape.back();
    if ((t.shape.size() == 2 || t.shape.size() == 3) &&
        (last == int64_t(Chunk::Q4_1) || last == int64_t(Chunk::Q4K) || last == int64_t(Chunk::Q8)))
        return Chunk(last);
    throw std::runtime_error("not a Q4NX chunked tensor (last dimension " + std::to_string(last) + ")");
}

void dequant_tile(Chunk kind, const uint8_t* t, float* out, int ld) {
    for (int r = 0; r < kRows; r++) {
        for (int c = 0; c < kCols; c++) {
            const int g = c / kGroup, i = g * kRows + r;
            float w;
            if (kind == Chunk::Q4_1) {
                const uint8_t b = t[q41_code_at(r, c)];
                const int code = (r % 2 == 0) ? (b & 15) : (b >> 4);
                w = float(code) * bf16(t + 2 * i) + bf16(t + 512 + 2 * i);
            } else if (kind == Chunk::Q4K) {
                const uint8_t b = t[512 + c * 16 + r / 2];
                const int code = (r % 2 == 0) ? (b & 15) : (b >> 4);
                w = bf16(t + 4608 + 2 * r) * float(t[i]) * float(code) + bf16(t + 4672 + 2 * r) * float(t[256 + i]);
            } else {
                w = bf16(t + 2 * i) * float(int8_t(t[512 + c * kRows + r]));
            }
            out[size_t(r) * ld + c] = w;
        }
    }
}

void dequant(const Tensor& t, int rows, int cols, float* out) {
    const Chunk kind = chunk_of(t);
    if (rows % kRows || cols % kCols) throw std::runtime_error("Q4NX tensor size is not whole tiles");
    const int tr = rows / kRows, tc = cols / kCols;
    if (t.bytes < uint64_t(tr) * tc * uint64_t(kind)) throw std::runtime_error("Q4NX tensor is shorter than its tiles");
    for (int i = 0; i < tr; i++)
        for (int j = 0; j < tc; j++)
            dequant_tile(kind, t.data + (size_t(i) * tc + j) * size_t(kind), out + size_t(i) * kRows * cols + size_t(j) * kCols,
                         cols);
}

void q4k_to_q4_1(const uint8_t* k, uint8_t* q) {
    std::memset(q, 0, size_t(Chunk::Q4_1));
    for (int r = 0; r < kRows; r++) {
        const float S = bf16(k + 4608 + 2 * r), M = bf16(k + 4672 + 2 * r);
        for (int g = 0; g < kCols / kGroup; g++) {
            const int i = g * kRows + r;
            put16(q + 2 * i, to_bf16(S * float(k[i])));
            put16(q + 512 + 2 * i, to_bf16(M * float(k[256 + i])));
        }
        for (int c = 0; c < kCols; c++) {
            const uint8_t b = k[512 + c * 16 + r / 2];
            const uint8_t code = (r % 2 == 0) ? (b & 15) : (b >> 4);
            q[q41_code_at(r, c)] |= (r % 2 == 0) ? code : uint8_t(code << 4);
        }
    }
}

std::vector<uint8_t> as_q4_1(const Tensor& t) {
    const Chunk kind = chunk_of(t);
    const size_t n = size_t(t.bytes / uint64_t(kind));
    if (kind == Chunk::Q4_1) return std::vector<uint8_t>(t.data, t.data + n * size_t(kind));
    if (kind == Chunk::Q8) throw std::runtime_error("a Q8 tensor does not fit q4_1 (8-bit codes)");
    std::vector<uint8_t> out(n * size_t(Chunk::Q4_1));
    for (size_t i = 0; i < n; i++) q4k_to_q4_1(t.data + i * size_t(kind), out.data() + i * size_t(Chunk::Q4_1));
    return out;
}

}  // namespace onebit::npu::q4nx
