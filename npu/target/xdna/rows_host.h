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
// npu/target/xdna/rows_host.h -- the host side of a batched (M-row) int8 launch: each row's activation scale before the
// launch and the dequantisation after it (engine#1783). Header-only and device-free, so tests/npu_rows_host_test.cpp
// checks every variant against the scalar loops it replaces without a device or the private runtime.
//
// The two loops are the ones npu_gemm_rows() ran inline:
//   scale[m] = amax_k |A[m][k]| / 127 (1 when the row is all zero)    and    out[m][n] = (float)C[m][n] * scale[m] * gs[n]
// Both are row-local, so splitting rows over threads moves no bit. The AVX2 variants keep the scalar arithmetic exactly:
//  * amax: `a > amax ? a : amax` over |a| skips NaN (the comparison is false) and keeps +0 for a zero row. _mm256_max_ps(x, m)
//    returns its SECOND operand when either is NaN, so max_ps(|x|, acc) skips NaN the same way, and a max is exact whatever
//    the order, so the lane-wise reduction gives the same float as the serial walk.
//  * dequant: cvtepi32_ps rounds like the (float) cast, then two separate multiplies in the scalar order. No add follows
//    a multiply, so there is nothing an FMA could contract, whatever the TU's -mfma / -ffp-contract setting.
// What ties the *_base loops to the inline loops they replaced is the device encoder hash: the whisper-small device route
// printed the same encode-only "output fnv" before and after the move (engine#1783), so the base loops compute what those
// loops computed; tests/npu_rows_host_test.cpp then holds every other variant to the base loops byte for byte.
// ONEBIT_NPU_GLUE_AVX2=0 keeps the scalar loops; ONEBIT_NPU_GLUE_THREADS=<n> sets the thread count (1 = serial). Only
// launches of at least kMinRows rows are threaded; a decode launch (M = 1 or 16) keeps its serial loop.
#ifndef ONEBIT_NPU_TARGET_XDNA_ROWS_HOST_H
#define ONEBIT_NPU_TARGET_XDNA_ROWS_HOST_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

#include "audio_epilogue.h"   // engine#1783: the projection's folded bias / GELU / residual

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#endif

namespace npu::target::xdna::rows_host {

inline void amax_scales_base(const float* A, int m0, int m1, int K, float* scales) {
    for (int m = m0; m < m1; m++) {
        const float* row = A + (size_t)m * K;
        float amax = 0.0f;
        for (int k = 0; k < K; k++) {
            const float a = std::fabs(row[k]);
            if (a > amax) amax = a;
        }
        scales[(size_t)m] = amax > 0.0f ? amax / 127.0f : 1.0f;
    }
}

inline void dequant_base(const int32_t* C, int ldc, int m0, int m1, int N, const float* scales, const float* gs, float* out,
                         int ldo) {
    for (int m = m0; m < m1; m++) {
        const int32_t* cm = C + (size_t)m * ldc;
        const float asc = scales[(size_t)m];
        float* o = out + (size_t)m * ldo;
        for (int n = 0; n < N; n++) o[n] = (float)cm[n] * asc * gs[(size_t)n];
    }
}

#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx2"), noinline)) inline void amax_scales_avx2(const float* A, int m0, int m1, int K, float* scales) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (int m = m0; m < m1; m++) {
        const float* row = A + (size_t)m * K;
        __m256 acc = _mm256_setzero_ps();
        int k = 0;
        for (; k + 8 <= K; k += 8) acc = _mm256_max_ps(_mm256_and_ps(_mm256_loadu_ps(row + k), absmask), acc);
        alignas(32) float lane[8];
        _mm256_store_ps(lane, acc);
        float amax = 0.0f;
        for (int i = 0; i < 8; i++)
            if (lane[i] > amax) amax = lane[i];
        for (; k < K; k++) {
            const float a = std::fabs(row[k]);
            if (a > amax) amax = a;
        }
        scales[(size_t)m] = amax > 0.0f ? amax / 127.0f : 1.0f;
    }
}

__attribute__((target("avx2"), noinline)) inline void dequant_avx2(const int32_t* C, int ldc, int m0, int m1, int N,
                                                                   const float* scales, const float* gs, float* out, int ldo) {
    for (int m = m0; m < m1; m++) {
        const int32_t* cm = C + (size_t)m * ldc;
        const float asc = scales[(size_t)m];
        const __m256 vs = _mm256_set1_ps(asc);
        float* o = out + (size_t)m * ldo;
        int n = 0;
        for (; n + 8 <= N; n += 8) {
            const __m256 c = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(cm + n)));
            _mm256_storeu_ps(o + n, _mm256_mul_ps(_mm256_mul_ps(c, vs), _mm256_loadu_ps(gs + n)));
        }
        for (; n < N; n++) o[n] = (float)cm[n] * asc * gs[(size_t)n];
    }
}
#endif

inline bool use_avx2() {
#if defined(__x86_64__) && defined(__GNUC__)
    static const bool on = [] {
        const char* e = std::getenv("ONEBIT_NPU_GLUE_AVX2");
        if (e != nullptr && e[0] == '0') return false;
        return bool(__builtin_cpu_supports("avx2"));
    }();
    return on;
#else
    return false;
#endif
}

constexpr int kMinRows = 64;

inline int threads_for(int M) {
    if (M < kMinRows) return 1;
    static const int t = [] {
        if (const char* e = std::getenv("ONEBIT_NPU_GLUE_THREADS")) {
            const int v = std::atoi(e);
            return v > 0 ? v : 1;
        }
        const unsigned hw = std::thread::hardware_concurrency();
        return int(hw == 0 ? 1 : (hw > 8 ? 8 : hw));
    }();
    return t;
}

// fn(m0, m1) over [0, M) in `t` contiguous blocks.
template <class Fn>
inline void split_rows(int M, int t, Fn&& fn) {
    if (t <= 1 || M < 2) { fn(0, M); return; }
    if (t > M) t = M;
    std::vector<std::thread> pool;
    pool.reserve(size_t(t - 1));
    const int step = (M + t - 1) / t;
    for (int i = 1; i < t; i++) {
        const int a = i * step, b = std::min(M, a + step);
        if (a < b) pool.emplace_back([&fn, a, b] { fn(a, b); });
    }
    fn(0, std::min(M, step));
    for (auto& th : pool) th.join();
}

inline void amax_scales(const float* A, int M, int K, float* scales, bool vec, int threads) {
    split_rows(M, threads, [&](int a, int b) {
#if defined(__x86_64__) && defined(__GNUC__)
        if (vec) { amax_scales_avx2(A, a, b, K, scales); return; }
#endif
        (void)vec;
        amax_scales_base(A, a, b, K, scales);
    });
}

// `ep` (may be null): the projection's folded tail, applied to each block of rows right after its dequant, on the same
// thread, in the same per-element order as the nodes it replaces (audio_epilogue.h).
inline void dequant(const int32_t* C, int ldc, int M, int N, const float* scales, const float* gs, float* out, int ldo, bool vec,
                    int threads, const onebit::audio::Epilogue* ep = nullptr) {
    split_rows(M, threads, [&](int a, int b) {
#if defined(__x86_64__) && defined(__GNUC__)
        if (vec) dequant_avx2(C, ldc, a, b, N, scales, gs, out, ldo);
        else dequant_base(C, ldc, a, b, N, scales, gs, out, ldo);
#else
        (void)vec;
        dequant_base(C, ldc, a, b, N, scales, gs, out, ldo);
#endif
        if (ep != nullptr) onebit::audio::epilogue_rows(*ep, a, b, N, out, ldo);
    });
}

}  // namespace npu::target::xdna::rows_host

#endif
