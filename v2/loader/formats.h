#pragma once

// [F] Tensor formats: the ggml block formats GGUF carries, as first-class V2 tensor types
// (design 2.2 [F]). Each format has a block geometry (elements and bytes per block) for sizing
// and placement, and, for the formats the CPU reference executor reads, a straightforward
// dequantizer to f32. The dequantizers follow the published block layouts; they are the
// correctness oracle's input path, not a fast path.

#include <cstddef>
#include <cstdint>
#include <string>

namespace onebit::v2 {

// ggml type ids as GGUF stores them.
enum class Fmt : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23,
    I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, IQ1_M = 29, BF16 = 30,
    TQ1_0 = 34, TQ2_0 = 35, MXFP4 = 39,
};

struct FmtInfo {
    const char* name = nullptr;
    uint32_t block = 0;   // elements per block
    uint32_t bytes = 0;   // bytes per block
    bool cpu_dequant = false;  // the CPU reference executor can read it
};

// nullptr for an id this build does not know.
const FmtInfo* fmt_info(uint32_t id);
std::string fmt_name(uint32_t id);

// Bytes for n elements of a format (n a multiple of the block); 0 if unknown or misaligned.
uint64_t fmt_bytes(uint32_t id, uint64_t n);

// Dequantizes n elements (a multiple of the block) from src into dst. Throws
// std::runtime_error for a format without a CPU dequantizer.
void dequantize(uint32_t id, const void* src, float* dst, size_t n);

float f16_to_f32(uint16_t h);
float bf16_to_f32(uint16_t h);

}  // namespace onebit::v2
