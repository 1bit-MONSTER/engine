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
