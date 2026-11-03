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

// One row of activations, f32 -> int8 at a given 1/scale: the loop every NPU GEMM runs on the host before
// its launch (I8Ctx::quantize_async / quantize_async_rows), extracted so it can be proven bit-identical and
// made 7x cheaper.
//
// Why it is worth a file: it runs once per projection -- 288 projections, 2,310,144 A elements per decode
// token on Bonsai-2 (the shipped container's own plan.json: 48 gdn layers x K = 5120, 5120, 6144, 5120,
// 17408 and 16 attention layers x 5120, 5120, 17408) -- and the engine builds at the baseline x86-64 GCC
// targets, where roundf() is a libm call.  Measured on strixhalo replaying exactly that mix, best of 7:
// 3.96-4.05 ms/token scalar (1.72-1.76 ns/element) against 0.55 ms/token on the AVX2 path
// (0.238 ns/element), 7.2-7.4x -- about 3.4 ms of a ~415 ms decode token.  Same story as hadamard.h next
// door, and the same contract: byte-identical output, so a container's text cannot move.
//
// Bit identity is not luck here, it is the construction.  roundf is round-half-away-from-zero, and
//   t = trunc(x)            exact
//   f = x - t               exact, |f| < 1
//   x rounds away from 0 iff |f| >= 0.5, compared exactly
//   t +- 1                  exact
// so trunc + adjust is the same integer roundf produces.  The three edge cases that bite a fast path are
// modelled on what the shipped loop actually does (checked, not assumed, against the baseline build):
//   * input NaN or +-inf  -> isfinite() is false -> the loop substitutes 0
//   * input finite but |v * ais| >= 2^31 -> roundf is finite/huge, the (int) cast is INT_MIN on x86, and the
//     clamp turns it into -127 -- so the vector path must saturate the same way and must not "fix" it to 127
//   * |v * ais| just under 2^31 -> still in range, so the adjustment applies normally
// Dispatched at run time on AVX2, so a baseline x86-64 still builds and runs the scalar loop (hadamard.h's
// pattern; both exist for the same reason).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstddef>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define Q4NX_QUANT_I8_X86 1
#include <immintrin.h>
#else
#define Q4NX_QUANT_I8_X86 0
#endif

namespace q4nx_quant_i8 {

// The shipped loop, unchanged: the reference every other path is proven against.
inline void row_scalar(const float* a, int n, float ais, int8_t* out) {
    for (int k = 0; k < n; k++) {
        float v = a[k];
        if (!std::isfinite(v)) v = 0;
        int q = (int)roundf(v * ais);
        if (q > 127) q = 127;
        else if (q < -127) q = -127;
        out[k] = (int8_t)q;
    }
}

#if Q4NX_QUANT_I8_X86

__attribute__((target("avx2"))) inline void row_avx2(const float* a, int n, float ais, int8_t* out) {
    const __m256 vais = _mm256_set1_ps(ais);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 nhalf = _mm256_set1_ps(-0.5f);
    const __m256 lim = _mm256_set1_ps(2147483648.0f);          // 2^31, the (int) cast's range edge
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 pinf = _mm256_set1_ps(INFINITY), ninf = _mm256_set1_ps(-INFINITY);
    const __m256i one = _mm256_set1_epi32(1), minusone = _mm256_set1_epi32(-1);
    const __m256i vmax = _mm256_set1_epi32(127), vmin = _mm256_set1_epi32(-127);
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        // isfinite() is applied to the input, so only NaN and +-inf are zeroed -- a large finite input
        // whose product overflows to inf is NOT zeroed, and saturates through the cast like the scalar.
        __m256 in = _mm256_loadu_ps(a + k);
        in = _mm256_and_ps(in, _mm256_cmp_ps(in, in, _CMP_ORD_Q));
        __m256 isinf = _mm256_or_ps(_mm256_cmp_ps(in, pinf, _CMP_EQ_OQ),
                                    _mm256_cmp_ps(in, ninf, _CMP_EQ_OQ));
        const __m256 v = _mm256_mul_ps(_mm256_andnot_ps(isinf, in), vais);
        const __m256i t = _mm256_cvttps_epi32(v);
        const __m256 f = _mm256_sub_ps(v, _mm256_cvtepi32_ps(t));
        __m256i adj = _mm256_and_si256(
            _mm256_castps_si256(_mm256_cmp_ps(f, half, _CMP_GE_OQ)), one);
        adj = _mm256_or_si256(
            adj, _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(f, nhalf, _CMP_LE_OQ)),
                                  minusone));
        // outside +-2^31 the cast has saturated: leave INT_MIN alone so the clamp yields -127, as scalar
        const __m256i inrange =
            _mm256_castps_si256(_mm256_cmp_ps(_mm256_and_ps(v, absmask), lim, _CMP_LT_OQ));
        __m256i q = _mm256_add_epi32(t, _mm256_and_si256(adj, inrange));
        q = _mm256_min_epi32(_mm256_max_epi32(q, vmin), vmax);
        // 8 int32 -> 8 int8 in lane order (packs saturate, but every lane is already inside +-127)
        const __m128i p16 =
            _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out + k), _mm_packs_epi16(p16, p16));
    }
    if (k < n) row_scalar(a + k, n - k, ais, out + k);   // tail: the reference, unchanged
}

inline bool have_avx2() {
    static const bool ok = __builtin_cpu_supports("avx2") != 0;
    return ok;
}

#endif  // Q4NX_QUANT_I8_X86

// `ais` is 1.0f / scale, computed by the caller exactly as before: only the element loop changes here.
inline void row(const float* a, int n, float ais, int8_t* out) {
#if Q4NX_QUANT_I8_X86
    if (have_avx2()) {
        row_avx2(a, n, ais, out);
        return;
    }
#endif
    row_scalar(a, n, ais, out);
}

}  // namespace q4nx_quant_i8
