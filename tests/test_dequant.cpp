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
#include <cmath>
#include <cstring>
#include <vector>

#include "check.h"
#include "dequant.h"

using namespace onebit;

static void test_fp16() {
    CHECK(fp16_to_fp32(0x3C00) == 1.0f);
    CHECK(fp16_to_fp32(0xC000) == -2.0f);
    CHECK(fp16_to_fp32(0x3555) == 0.333251953125f);
    CHECK(fp16_to_fp32(0x7BFF) == 65504.0f);
    CHECK(fp16_to_fp32(0x0001) == std::ldexp(1.0f, -24));  // smallest subnormal
    CHECK(fp16_to_fp32(0x03FF) == std::ldexp(1023.0f, -24));  // largest subnormal
    CHECK(fp16_to_fp32(0x0400) == std::ldexp(1.0f, -14));  // smallest normal
    CHECK(fp16_to_fp32(0x0000) == 0.0f);
    CHECK(std::signbit(fp16_to_fp32(0x8000)));
    CHECK(std::isinf(fp16_to_fp32(0x7C00)));
    CHECK(std::isnan(fp16_to_fp32(0x7E00)));
}

static void test_bf16() {
    CHECK(bf16_to_fp32(0x3F80) == 1.0f);
    CHECK(bf16_to_fp32(0xC040) == -3.0f);
    CHECK(bf16_to_fp32(0x3EAB) == 0.333984375f);
}

static void test_q8_0() {
    // Two blocks: scale 0.5 then scale -1.0 (fp16), quants -128..127 pattern.
    std::vector<std::byte> buf(68);
    const uint16_t scales[2] = {0x3800, 0xBC00};
    for (int b = 0; b < 2; ++b) {
        std::memcpy(buf.data() + b * 34, &scales[b], 2);
        for (int i = 0; i < 32; ++i) buf[b * 34 + 2 + i] = std::byte(uint8_t(int8_t(i * 8 - 128)));
    }
    std::vector<float> out(64);
    REQUIRE(dequantize(GgmlType::Q8_0, buf.data(), out.data(), 64).has_value());
    CHECK(out[0] == -64.0f);
    CHECK(out[31] == 0.5f * 120.0f);
    CHECK(out[32] == 128.0f);
    CHECK(out[63] == -120.0f);
    CHECK(!dequantize(GgmlType::Q8_0, buf.data(), out.data(), 40).has_value());
}

static void test_unsupported() {
    float out[256];
    std::byte in[256] = {};
    CHECK(!dequant_supported(GgmlType::Q4_K));
    CHECK(!dequantize(GgmlType::Q4_K, in, out, 256).has_value());
}

int main() {
    test_fp16();
    test_bf16();
    test_q8_0();
    test_unsupported();
    return test_result("test_dequant");
}
