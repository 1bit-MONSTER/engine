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
// tests/npu_rows_host_test.cpp -- engine#1783: the batched int8 launch's host loops (npu/target/xdna/rows_host.h). Every
// variant -- scalar serial, scalar threaded, AVX2 serial, AVX2 threaded -- must produce the same bytes as the scalar serial
// loop the target used to run inline, on rows that include zeros, -0, NaN, +-inf, denormals and a width that is not a
// multiple of 8. Device-free.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "target/xdna/rows_host.h"

namespace rh = npu::target::xdna::rows_host;

int main() {
    int failures = 0;
    std::mt19937 rng(1783);
    std::uniform_real_distribution<float> uf(-3.0f, 3.0f);
    std::uniform_int_distribution<int32_t> ui(-(1 << 24), 1 << 24);
    for (const int K : {768, 3072, 245, 7}) {
        for (const int M : {1, 16, 63, 64, 1500}) {
            std::vector<float> A(size_t(M) * K);
            for (auto& v : A) v = uf(rng);
            if (M > 3) {
                for (int k = 0; k < K; k++) A[size_t(1) * K + k] = 0.0f;                 // an all-zero row
                for (int k = 0; k < K; k++) A[size_t(2) * K + k] = (k & 1) ? -0.0f : 0.0f;
                A[size_t(3) * K + 0] = std::numeric_limits<float>::quiet_NaN();
                A[size_t(3) * K + (K - 1)] = -std::numeric_limits<float>::infinity();
                // -inf in a VECTOR lane (index 8 = 0 mod 8), not only in the scalar tail: |x| = +inf through max_ps
                if (K >= 16) A[size_t(4) * K + 8] = -std::numeric_limits<float>::infinity();
                A[0] = 1e-42f;                                                              // denormal
            }
            std::vector<float> ref(static_cast<size_t>(M));
            rh::amax_scales_base(A.data(), 0, M, K, ref.data());
            for (const bool vec : {false, true})
                for (const int th : {1, 3, 8}) {
                    std::vector<float> got(size_t(M), -1.0f);
                    rh::amax_scales(A.data(), M, K, got.data(), vec && rh::use_avx2(), th);
                    if (std::memcmp(got.data(), ref.data(), ref.size() * 4) != 0) {
                        std::printf("FAIL amax K=%d M=%d vec=%d threads=%d\n", K, M, int(vec), th);
                        failures++;
                    }
                }
            // dequant: C is M x ldc int32 (ldc = staged width), the output keeps the first N columns at stride ldo
            const int N = K, ldc = ((K + 511) / 512) * 512, ldo = N + 3;
            std::vector<int32_t> C(size_t(M) * ldc);
            for (auto& v : C) v = ui(rng);
            if (M > 2) C[size_t(2) * ldc] = std::numeric_limits<int32_t>::min();
            std::vector<float> gs(static_cast<size_t>(ldc));
            for (auto& v : gs) v = uf(rng) * 1e-3f;
            std::vector<float> oref(size_t(M) * ldo, 7.0f);
            rh::dequant_base(C.data(), ldc, 0, M, N, ref.data(), gs.data(), oref.data(), ldo);
            for (const bool vec : {false, true})
                for (const int th : {1, 3, 8}) {
                    std::vector<float> o(size_t(M) * ldo, 7.0f);
                    rh::dequant(C.data(), ldc, M, N, ref.data(), gs.data(), o.data(), ldo, vec && rh::use_avx2(), th);
                    if (std::memcmp(o.data(), oref.data(), o.size() * 4) != 0) {
                        std::printf("FAIL dequant K=%d M=%d vec=%d threads=%d\n", K, M, int(vec), th);
                        failures++;
                    }
                }
        }
    }
    // A decode launch (M = 1 or 16) and anything under kMinRows keeps its serial loop, whatever ONEBIT_NPU_GLUE_THREADS says.
    for (const int m : {1, 16, 63})
        if (rh::threads_for(m) != 1) {
            std::printf("FAIL threads_for(%d) = %d, want 1\n", m, rh::threads_for(m));
            failures++;
        }
    // engine#1783: dequant with a folded tail == dequant, then the tail, for every variant (bias + GELU, bias + residual).
    {
        const int M = 1500, N = 768, ldc = 1024, ldo = 768;
        std::vector<int32_t> C(size_t(M) * ldc);
        for (auto& v : C) v = ui(rng);
        std::vector<float> sc(static_cast<size_t>(M)), gs(static_cast<size_t>(ldc)), bias(static_cast<size_t>(N)),
            res(size_t(M) * N);
        for (auto& v : sc) v = uf(rng) * 1e-6f;
        for (auto& v : gs) v = uf(rng) * 1e-3f;
        for (auto& v : bias) v = uf(rng);
        for (auto& v : res) v = uf(rng);
        for (int kind = 0; kind < 2; ++kind) {
            onebit::audio::Epilogue ep;
            ep.bias = bias.data();
            if (kind == 0) {
                ep.gelu = true;
                ep.gelu_div = 1.4142135381698608f;
                ep.gelu_one = 1.0f;
                ep.gelu_half = 0.5f;
            } else {
                ep.res = res.data();
                ep.res_ld = N;
            }
            std::vector<float> ref(size_t(M) * ldo);
            rh::dequant_base(C.data(), ldc, 0, M, N, sc.data(), gs.data(), ref.data(), ldo);
            onebit::audio::epilogue_rows(ep, 0, M, N, ref.data(), ldo);
            for (const bool vec : {false, true})
                for (const int th : {1, 3, 8}) {
                    std::vector<float> o(size_t(M) * ldo, 7.0f);
                    rh::dequant(C.data(), ldc, M, N, sc.data(), gs.data(), o.data(), ldo, vec && rh::use_avx2(), th, &ep);
                    if (std::memcmp(o.data(), ref.data(), o.size() * 4) != 0) {
                        std::printf("FAIL dequant+epilogue kind=%d vec=%d threads=%d\n", kind, int(vec), th);
                        failures++;
                    }
                }
        }
    }
    std::printf("npu rows host: avx2 %s, %s\n", rh::use_avx2() ? "on" : "not available (scalar arms only)",
                failures == 0 ? "all variants bit-identical" : "FAILED");
    return failures == 0 ? 0 : 1;
}
