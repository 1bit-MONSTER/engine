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
// onebit_npu_gemm_probe.cpp — one MatMul on the NPU, checked against a host reference.
//
// This is the deliverable-3 proof: it takes the device lock (through the module), runs
// C = A[M,K] @ B[K,N] on the NPU, recomputes it on the host, and prints the max |difference|.
// It is also the module's self-check, so a broken context fails here before any ORT session.
//
//   onebit_npu_gemm_probe <K> <N> <M> <repeats>
//
// The kernel is built for one (K, N), so those two must be the ELF's design shape:
//   ONEBIT_EP_NPU_ELF=/path/full_i8_O_K1024_N1024_m16.elf \
//   ONEBIT_EP_NPU_M=16 ONEBIT_EP_NPU_K=1024 ONEBIT_EP_NPU_N=1024 \
//   onebit_npu_gemm_probe 1024 1024 4 3
#include "onebit_npu_gemm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void reference(const float* a, const float* b, float* c, int64_t m, int64_t k, int64_t n) {
    std::memset(c, 0, (size_t)m * (size_t)n * sizeof(float));
    for (int64_t i = 0; i < m; ++i)
        for (int64_t p = 0; p < k; ++p) {
            const float aik = a[i * k + p];
            const float* brow = b + p * n;
            for (int64_t j = 0; j < n; ++j) c[i * n + j] += aik * brow[j];
        }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <K> <N> <M> [repeats]\n", argv[0]);
        return 2;
    }
    const int64_t k = atoll(argv[1]), n = atoll(argv[2]), m = atoll(argv[3]);
    const int repeats = argc > 4 ? atoi(argv[4]) : 1;
    if (k <= 0 || n <= 0 || m <= 0) return 2;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    std::string why;
    if (!onebit::npu::available(&why)) {
        std::printf("NPU unavailable: %s\n", why.c_str());
        return 1;
    }

    // Deterministic, sign-mixed data: a bad quantiser shows up as more than rounding noise.
    std::vector<float> a((size_t)m * k), b((size_t)k * n), c((size_t)m * n), ref((size_t)m * n);
    // Uncorrelated, zero-mean data on purpose: sin/cos give a cancellation-dominated result whose
    // magnitude is far below the int8 noise floor, so a correct kernel measures as a large relative
    // error. With ~N(0,1) inputs |ref| ~ sqrt(K) and the same noise is a third of a percent.
    uint32_t rng = 12345u;
    auto next = [&rng]() {
        rng = rng * 1664525u + 1013904223u;
        return ((double)(rng >> 8) / 8388608.0) - 1.0;  // [-1, 1)
    };
    for (size_t i = 0; i < a.size(); i++) a[i] = (float)next();
    for (size_t i = 0; i < b.size(); i++) b[i] = (float)next() * 0.5f;

    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; r++) {
        std::string err;
        if (!onebit::npu::gemm_f32(a.data(), m, k, n, b.data(), c.data(), &err)) {
            std::printf("NPU gemm failed: %s\n", err.c_str());
            return 1;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;

    reference(a.data(), b.data(), ref.data(), m, k, n);
    double max_abs = 0.0, max_rel = 0.0;
    float max_ref = 0.0f;
    for (size_t i = 0; i < c.size(); i++) max_ref = std::max(max_ref, std::fabs(ref[i]));
    for (size_t i = 0; i < c.size(); i++) {
        const double d = std::fabs((double)c[i] - (double)ref[i]);
        max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, d / std::max(1e-6, std::fabs((double)ref[i])));
    }
    std::printf("npu gemm m=%lld k=%lld n=%lld repeats=%d max_abs=%.3e max_rel=%.3e max|ref|=%.3f "
                "%.2f ms/call\n",
                (long long)m, (long long)k, (long long)n, repeats, max_abs, max_rel, max_ref, ms);

    // int8 weights+activations: the error is quantisation noise, so bound it against the
    // magnitude of the result rather than against float rounding. 2% of max|ref| is generous
    // for 7-bit activations and still catches a wrong scale, a wrong layout or a truncated K.
    // Least-squares scale of got against ref: ~1.0 rules out a scale/offset mistake and points at
    // layout, while a constant far from 1.0 means one operand's scale is being applied wrong.
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        num += (double)c[i] * (double)ref[i];
        den += (double)ref[i] * (double)ref[i];
    }
    std::printf("  fit got~scale*ref: scale=%.4f  ref[0..2]=%.4f,%.4f,%.4f  got[0..2]=%.4f,%.4f,%.4f\n",
                den > 0.0 ? num / den : 0.0, (double)ref[0], (double)ref[1], (double)ref[2],
                (double)c[0], (double)c[1], (double)c[2]);
    if (max_abs > 0.02 * (double)max_ref && max_ref > 0.0f) {
        std::printf("FAIL NPU gemm differs from the host reference by more than quantisation noise\n");
        return 1;
    }
    std::printf("PASS NPU gemm matches the host reference, max_abs=%.3e\n", max_abs);
    return 0;
}
