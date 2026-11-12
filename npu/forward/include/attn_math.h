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

// The two float kernels the per-op forward's host attention is made of: one score dot (q.k over the head
// dimension) and one value accumulation (ctx += p * v).  Half of a prefill's attention flops each.
//
// Why they exist rather than an open-coded loop: the obvious single-accumulator `for (i) d += a[i]*b[i]`
// is one float FMA latency chain per score, and gcc may not reassociate it without -ffast-math (the
// engine sets none), so it stays scalar.  Measured that way, attention on the Phi-3.5-mini OGA pack ran
// at 8.1 GFLOP/s, which is what a 2048-token prefill spent 208-398 s on while the same prompt's GEMMs
// cost 2.1 s on the array.  Both kernels are the shape npu/forward/include/hadamard.h already uses:
// an AVX2+FMA body behind a runtime check, and a portable body for every other host.
//
// Contract: the forms differ by float ROUNDING ORDER only -- they are the same sum, not the same bits.
// tests/attn_math_test.cpp is what holds that line (each form against a double-precision direct sum, on
// every width the forward passes, including the awkward ones either body's tail loop would mishandle).
#include <cmath>
#include <cstddef>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define Q4NX_ATTN_X86 1
#include <immintrin.h>
#else
#define Q4NX_ATTN_X86 0
#endif

namespace q4nx_attn {

#if Q4NX_ATTN_X86
inline bool have_avx2_fma() {
    static const bool ok = __builtin_cpu_supports("avx2") != 0 && __builtin_cpu_supports("fma") != 0;
    return ok;
}

__attribute__((target("avx2,fma"))) inline float dot_avx2(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    for (; i + 8 <= n; i += 8)
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s0 = _mm256_add_ps(s0, s1);
    __m128 t = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    float sum = _mm_cvtss_f32(t);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

__attribute__((target("avx2,fma"))) inline void saxpy_avx2(float p, const float* x, float* y, int n) {
    const __m256 pv = _mm256_set1_ps(p);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(pv, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += p * x[i];
}
#endif  // Q4NX_ATTN_X86

// The portable bodies.  Four accumulators, so the dependency chain is a quarter as long and the compiler
// has something to vectorize at whatever width the target actually has (this tree is built for the
// baseline x86-64 ISA, i.e. 4-lane SSE2 without FMA).
inline float dot_scalar(const float* a, const float* b, int n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; i++) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

inline void saxpy_scalar(float p, const float* x, float* y, int n) {
    for (int i = 0; i < n; i++) y[i] += p * x[i];
}

// One score: sum(a[i]*b[i]) over the head dimension.
inline float dot(const float* a, const float* b, int n) {
#if Q4NX_ATTN_X86
    if (have_avx2_fma()) return dot_avx2(a, b, n);
#endif
    return dot_scalar(a, b, n);
}

// y[i] += p * x[i] over the head dimension.
inline void saxpy(float p, const float* x, float* y, int n) {
#if Q4NX_ATTN_X86
    if (have_avx2_fma()) { saxpy_avx2(p, x, y, n); return; }
#endif
    saxpy_scalar(p, x, y, n);
}

// One head of one row: softmax(q.K * scale) over cached positions [t0, T), then ctx = sum p_t * V_t.
// `kc`/`vc` are the cache planes, position-major [t][NKV][HD]; `kh` is the KV head this query head reads;
// `score` is T floats of scratch; `out` (HD floats) must be zero on entry. This is the body attention() ran
// inline before it was shared -- the same statements in the same order, so the result is the same bits.
inline void head_context(const float* qh, const float* kc, const float* vc, int NKV, int kh, int HD,
                         int t0, int T, float scale, float* score, float* out, const float* sink = nullptr) {
    float mx = sink ? *sink : -1e30f;   // GPT-OSS: a per-head logit that joins the softmax and carries no value
    for (int t = 0; t < T; t++) {
        if (t < t0) { score[t] = -1e30f; continue; }
        const float* kt = kc + ((size_t)t * NKV + kh) * HD;
        score[t] = dot(qh, kt, HD) * scale;
        if (score[t] > mx) mx = score[t];
    }
    float sum = 0.0f;
    for (int t = 0; t < T; t++) {
        if (t < t0) { score[t] = 0.0f; continue; }
        score[t] = std::exp(score[t] - mx);
        sum += score[t];
    }
    if (sink) sum += std::exp(*sink - mx);
    const float inv = sum > 0 ? 1.0f / sum : 0.0f;
    for (int t = t0; t < T; t++) saxpy(score[t] * inv, vc + ((size_t)t * NKV + kh) * HD, out, HD);
}

// M rows of attention in ONE thread team (engine#1990). A batched pass used to call attention() once per row,
// and each call opened its own team over the heads: 16 rows x 16 attention layers = 256 teams per prefill
// block, and the team count is what a loaded box pays for. Here the (row, head) pairs are one work-shared
// loop. Each pair is independent -- it reads its own q, the read-only cache, and writes its own HD floats of
// ctx -- so the split across threads cannot change a value; tests/attn_math_test.cpp compares this against a
// plain loop over head_context() bit for bit.
//
// `q` is [M][NH][HD] (already normed and rotated), `T[m]`/`t0[m]` are row m's read window, `ctx` is
// [M][NH][HD] and is zeroed here. `nt` <= 1 runs it on the caller's thread.
inline void rows_context(const float* q, const float* kc, const float* vc, int M, int NH, int NKV, int HD,
                         const int* T, const int* t0, float scale, float* ctx, int nt) {
    const int gqa = NH / NKV;
    int tmax = 0;
    for (int m = 0; m < M; m++) tmax = T[m] > tmax ? T[m] : tmax;
    for (size_t i = 0, n = (size_t)M * NH * HD; i < n; i++) ctx[i] = 0.0f;
    const long pairs = (long)M * NH;
    (void)nt;
#pragma omp parallel num_threads(nt > 1 ? nt : 1) if (nt > 1 && pairs > 1)
    {
        std::vector<float> score((size_t)(tmax > 0 ? tmax : 1));
        // static, not dynamic: on a loaded box a dynamic schedule's tail was the whole cost (a host microbench at
        // this geometry, 24 threads: p99 62 ms against 17 ms static per layer), and the rows of one block read
        // windows within M positions of each other, so there is no imbalance for dynamic to fix.
#pragma omp for schedule(static)
        for (long i = 0; i < pairs; i++) {
            const int m = (int)(i / NH), h = (int)(i % NH);
            head_context(q + ((size_t)m * NH + h) * HD, kc, vc, NKV, h / gqa, HD, t0[m], T[m], scale,
                         score.data(), ctx + ((size_t)m * NH + h) * HD);
        }
    }
}

}  // namespace q4nx_attn
