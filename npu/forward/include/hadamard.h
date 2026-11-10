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

// A <- H (A . s): the activation rotation a folded (PTQ1/Prism) container needs, because its weights were
// baked in the Hadamard basis.  H is the normalised Sylvester-Walsh Hadamard over `block` consecutive floats,
// so the transform is a fast Walsh-Hadamard butterfly, and one multiply by 1/sqrt(block) follows it.
//
// Why this file exists at all: the rotation runs ONCE PER PROJECTION on the host, i.e. 400-450 times a decode
// token for Bonsai-2's geometry, and the measured cost on this box was 14.2 us per 5120-float rotation --
// 15.8 ms/token in a quiet window and 39.0 ms/token on a loaded one, because it is pure scalar FP work and
// therefore tracks CPU contention rather than the device.  Scalar it was: 12 flops per element at ~1.7 cycles
// per flop is the shape of a loop the compiler cannot vectorise -- the butterfly's strides change every stage,
// and the 4-wide SSE2 baseline GCC targets here cannot express pairs at distance 1 or 2 without shuffles.
//
// The SIMD path below keeps the *exact* arithmetic of the scalar loop, so the rotated vector is bit-identical
// and a container's text cannot move: every output element is still one add and one subtract of the same two
// operands, and the stages still run in ascending distance order (1, 2, 4, ...), because floating-point
// addition is not associative and a different stage order would group the sums differently.  Only stages 1, 2
// and 4 need shuffles (an 8-lane in-register 8-point butterfly); every later stage is contiguous pairs of
// contiguous runs, which is a plain 8-lane add and subtract.  Dispatched at run time on AVX2 so the engine
// still builds and runs on a baseline x86-64 without it.
#pragma once

#include <cmath>
#include <cstddef>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define Q4NX_HADAMARD_X86 1
#include <immintrin.h>
#else
#define Q4NX_HADAMARD_X86 0
#endif

namespace q4nx_hadamard {

// The reference implementation, and the one every non-AVX2 host runs: the shipped scalar loop, unchanged.
inline void rotate_scalar(const float* signs, float* a, int n, int block) {
    for (int i = 0; i < n; i++) a[i] *= signs[i];
    const float sc = 1.0f / std::sqrt((float)block);
    for (int b = 0; b + block <= n; b += block) {
        float* p = a + b;
        for (int h = 1; h < block; h <<= 1)
            for (int i = 0; i < block; i += (h << 1))
                for (int j = i; j < i + h; j++) {
                    const float x = p[j], y = p[j + h];
                    p[j] = x + y;
                    p[j + h] = x - y;
                }
        for (int i = 0; i < block; i++) p[i] *= sc;
    }
}

#if Q4NX_HADAMARD_X86

// Stages 1, 2 and 4 of one block, on 8 consecutive elements: the in-register 8-point Walsh-Hadamard
// butterfly, in the scalar loop's distance order.  `idx` are the involution permutations that move each
// difference x-y from position j to its output position j+h (the same swap, applied twice, is the identity).
__attribute__((target("avx2"))) inline __m256 butterfly8(__m256 v) {
    const __m256i sw1 = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6);
    const __m256i sw2 = _mm256_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5);
    // distance 1: out[2i] = x+y (lane 2i of the sum), out[2i+1] = x-y (moved into lane 2i+1)
    {
        const __m256 s = _mm256_permutevar8x32_ps(v, sw1);
        const __m256 sum = _mm256_add_ps(v, s);
        const __m256 dif = _mm256_permutevar8x32_ps(_mm256_sub_ps(v, s), sw1);
        v = _mm256_blend_ps(sum, dif, 0xAA);      // odd lanes from the difference
    }
    // distance 2
    {
        const __m256 s = _mm256_permutevar8x32_ps(v, sw2);
        const __m256 sum = _mm256_add_ps(v, s);
        const __m256 dif = _mm256_permutevar8x32_ps(_mm256_sub_ps(v, s), sw2);
        v = _mm256_blend_ps(sum, dif, 0xCC);      // lanes 2, 3, 6, 7 from the difference
    }
    // distance 4
    {
        const __m256 s = _mm256_permute2f128_ps(v, v, 0x01);
        const __m256 sum = _mm256_add_ps(v, s);
        const __m256 d = _mm256_sub_ps(v, s);
        v = _mm256_blend_ps(sum, _mm256_permute2f128_ps(d, d, 0x01), 0xF0);   // lanes 4..7 from the difference
    }
    return v;
}

__attribute__((target("avx2"))) inline void rotate_avx2(const float* signs, float* a, int n, int block) {
    const __m256 sc8 = _mm256_set1_ps(1.0f / std::sqrt((float)block));
    // the sign vector first, over the whole vector: exact per element, and the block loop below needs it
    // applied before any stage reads across 8-lane groups
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(a + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(signs + i)));
    for (; i < n; i++) a[i] *= signs[i];

    for (int b = 0; b + block <= n; b += block) {
        float* p = a + b;
        // stages 1, 2, 4: one in-register butterfly per 8 elements
        for (int g = 0; g < block; g += 8)
            _mm256_storeu_ps(p + g, butterfly8(_mm256_loadu_ps(p + g)));
        // stages 8 .. block/2: contiguous pairs of contiguous runs, which is a plain 8-lane add/subtract
        for (int h = 8; h < block; h <<= 1)
            for (int r = 0; r < block; r += (h << 1))
                for (int j = r; j < r + h; j += 8) {
                    const __m256 x = _mm256_loadu_ps(p + j);
                    const __m256 y = _mm256_loadu_ps(p + j + h);
                    _mm256_storeu_ps(p + j, _mm256_add_ps(x, y));
                    _mm256_storeu_ps(p + j + h, _mm256_sub_ps(x, y));
                }
        // the normalisation the scalar loop applies after the stages: one multiply per element
        for (int j = 0; j < block; j += 8)
            _mm256_storeu_ps(p + j, _mm256_mul_ps(_mm256_loadu_ps(p + j), sc8));
    }
}

inline bool have_avx2() {
    static const bool ok = __builtin_cpu_supports("avx2") != 0;
    return ok;
}

#endif  // Q4NX_HADAMARD_X86

// True when the SIMD path can do this shape: a power-of-two block at least 8 wide, which is every block a
// converter writes (Bonsai-2's container ships 1024).  Anything else falls back to the scalar loop.
inline bool simd_shape(int block) {
    return Q4NX_HADAMARD_X86 && block >= 8 && (block & (block - 1)) == 0;
}

inline void rotate(const float* signs, float* a, int n, int block) {
#if Q4NX_HADAMARD_X86
    if (simd_shape(block) && have_avx2()) {
        rotate_avx2(signs, a, n, block);
        return;
    }
#endif
    rotate_scalar(signs, a, n, block);
}

// engine#1990: a batched launch's M rows -- each copied from `src` into `dst`, rotated, and given its int8 scale from
// the rotated vector (amax / 127, or 1.0 for an all-zero row). XdnaTarget::npu_gemm_rows did this one row after
// another on one thread, about 45-50 ms of a 16-row prefill block on bonsai2-27b-t2s2. The rows share nothing, so a
// thread team over them runs exactly the per-row statements of the serial loop and the result is bit-identical for
// any thread count (pinned by tests/hadamard_rotate_test.cpp). `threads` <= 1 is the serial loop.
inline void rotate_rows(const float* signs, const float* src, float* dst, int M, int K, int block, float* ascales,
                        int threads) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(threads > 1 ? threads : 1) if (threads > 1 && M > 1)
#endif
    for (int m = 0; m < M; m++) {
        float* row = dst + (size_t)m * (size_t)K;
        const float* in = src + (size_t)m * (size_t)K;
        for (int k = 0; k < K; k++) row[k] = in[k];
        rotate(signs, row, K, block);
        float am = 0.0f;
        for (int k = 0; k < K; k++) {
            const float a = std::fabs(row[k]);
            if (a > am) am = a;
        }
        ascales[m] = am > 0.0f ? am / 127.0f : 1.0f;
    }
}

}  // namespace q4nx_hadamard
