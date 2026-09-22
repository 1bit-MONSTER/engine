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
#include "dequant.h"

#include <bit>
#include <cstring>
#include <format>

namespace onebit {

float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = uint32_t{h & 0x8000u} << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // Subnormal: normalise the mantissa.
            int e = -1;
            do {
                ++e;
                mant <<= 1;
            } while ((mant & 0x400u) == 0);
            bits = sign | uint32_t(127 - 15 - e) << 23 | (mant & 0x3ffu) << 13;
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | mant << 13;  // inf / nan
    } else {
        bits = sign | (exp + (127 - 15)) << 23 | mant << 13;
    }
    return std::bit_cast<float>(bits);
}

float bf16_to_fp32(uint16_t b) { return std::bit_cast<float>(uint32_t{b} << 16); }

bool dequant_supported(GgmlType t) {
    switch (t) {
        case GgmlType::F32:
        case GgmlType::F16:
        case GgmlType::BF16:
        case GgmlType::Q8_0: return true;
        default: return false;
    }
}

namespace {

uint16_t load_u16(const std::byte* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

}  // namespace

std::expected<void, std::string> dequantize(GgmlType t, const std::byte* src, float* dst, uint64_t n) {
    switch (t) {
        case GgmlType::F32:
            std::memcpy(dst, src, n * sizeof(float));
            return {};
        case GgmlType::F16:
            for (uint64_t i = 0; i < n; ++i) dst[i] = fp16_to_fp32(load_u16(src + 2 * i));
            return {};
        case GgmlType::BF16:
            for (uint64_t i = 0; i < n; ++i) dst[i] = bf16_to_fp32(load_u16(src + 2 * i));
            return {};
        case GgmlType::Q8_0: {
            // Block of 32: fp16 scale d, then 32 int8 quants. x = d * q.
            constexpr uint64_t kBlock = 32, kBytes = 34;
            if (n % kBlock != 0) return std::unexpected(std::string("Q8_0: n is not a multiple of 32"));
            for (uint64_t b = 0; b < n / kBlock; ++b) {
                const std::byte* blk = src + b * kBytes;
                const float d = fp16_to_fp32(load_u16(blk));
                const auto* q = reinterpret_cast<const int8_t*>(blk + 2);
                for (uint64_t i = 0; i < kBlock; ++i) dst[b * kBlock + i] = d * float(q[i]);
            }
            return {};
        }
        default:
            return std::unexpected(std::format("dequantize: type {} not supported yet", ggml_type_name(t)));
    }
}

}  // namespace onebit
