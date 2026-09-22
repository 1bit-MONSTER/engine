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
// dequant.h: convert ggml tensor data to fp32.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>

#include "gguf.h"

namespace onebit {

float fp16_to_fp32(uint16_t h);
float bf16_to_fp32(uint16_t b);

// Whether dequantize() handles type t.
bool dequant_supported(GgmlType t);

// Write n fp32 values decoded from src (of type t) into dst.
// n must be a whole number of blocks for block-quantized types.
std::expected<void, std::string> dequantize(GgmlType t, const std::byte* src, float* dst, uint64_t n);

}  // namespace onebit
