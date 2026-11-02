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
// onebit_npu_gemm_q8_probe.cpp — the A8W8 kernel timed on both legs, and checked exactly.
//
// The QDQ proof asserts that the *graph* agrees with the CPU EP; this measures the kernel that
// makes the device leg worth having. It runs C[m,n] = A[m,k] @ B[k,n] with int8 operands on the
// NPU (gemm_q8_raw: bytes straight into the BOs, exact int32 accumulators out) and on the host
// (the same i-k-j loop onebit_ep_kernel.cpp falls back to), times both, and requires the two
// accumulators to be identical — an exact integer sum cannot differ, so a difference here is a
// wrong layout or a truncated K, not rounding.
//
//   onebit_npu_gemm_q8_probe <K> <N> <M> [repeats]
//
// The design M is a property of the ELF, so (K, N) and M must match it:
//   ONEBIT_EP_NPU_ELFS=16x1024x2048=/path/full_i8_Q_K1024_N2048_m16.elf \
//   onebit_npu_gemm_q8_probe 1024 2048 16 50
//
// Exit: 0 the device ran the shape and matched, 1 the numbers differ, 2 bad usage,
// 3 the device could not run it (no ELF for this K x N, busy, XRT failure) — nothing measured.
#include "onebit_npu_gemm.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// The EP's host fallback loop, byte for byte (onnx/ep/onebit_ep_kernel.cpp: i-k-j, exact int32).
void host_gemm(const int8_t* a, const int8_t* b, int32_t* c, int64_t m, int64_t k, int64_t n) {
    std::vector<int32_t> acc((size_t)n);
    for (int64_t i = 0; i < m; ++i) {
        std::fill(acc.begin(), acc.end(), 0);
        for (int64_t p = 0; p < k; ++p) {
            const int32_t aik = a[i * k + p];
            if (aik == 0) continue;
            const int8_t* brow = b + p * n;
            for (int64_t j = 0; j < n; ++j) acc[(size_t)j] += aik * brow[j];
        }
        std::memcpy(c + i * n, acc.data(), (size_t)n * sizeof(int32_t));
    }
}

double ms_per_call(const std::chrono::steady_clock::time_point& t0,
                   const std::chrono::steady_clock::time_point& t1, int repeats) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <K> <N> <M> [repeats]\n", argv[0]);
        return 2;
    }
    const int64_t k = atoll(argv[1]), n = atoll(argv[2]), m = atoll(argv[3]);
    const int repeats = argc > 4 ? atoi(argv[4]) : 20;
    if (k <= 0 || n <= 0 || m <= 0 || repeats <= 0) return 2;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    // Sign-mixed int8 in the range a real A8W8 graph produces ([-127,127], never -128).
    std::vector<int8_t> a((size_t)m * k), b((size_t)k * n);
    std::vector<int32_t> got((size_t)m * n), ref((size_t)m * n);
    uint32_t rng = 12345u;
    auto next = [&rng]() {
        rng = rng * 1664525u + 1013904223u;
        return (int)((rng >> 8) % 255u) - 127;
    };
    for (auto& v : a) v = (int8_t)next();
    for (auto& v : b) v = (int8_t)next();

    std::string err;
    if (!onebit::npu::gemm_q8_raw(a.data(), m, k, n, b.data(), got.data(), &err)) {
        std::printf("NPU q8 gemm could not run: %s\n", err.c_str());
        return 3;
    }

    const auto d0 = std::chrono::steady_clock::now();
    for (int r = 1; r < repeats; r++) {
        if (!onebit::npu::gemm_q8_raw(a.data(), m, k, n, b.data(), got.data(), &err)) {
            std::printf("NPU q8 gemm failed on repeat %d: %s\n", r, err.c_str());
            return 3;
        }
    }
    const auto d1 = std::chrono::steady_clock::now();
    const double dev_ms = ms_per_call(d0, d1, repeats);

    const auto h0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; r++) host_gemm(a.data(), b.data(), ref.data(), m, k, n);
    const auto h1 = std::chrono::steady_clock::now();
    const double host_ms = ms_per_call(h0, h1, repeats);

    int64_t bad = 0, first = -1;
    int64_t worst = 0;
    for (size_t i = 0; i < got.size(); i++)
        if (got[i] != ref[i]) {
            bad++;
            if (first < 0) first = (int64_t)i;
            worst = std::max<int64_t>(worst, llabs((int64_t)got[i] - (int64_t)ref[i]));
        }

    std::printf("q8 gemm m=%lld k=%lld n=%lld repeats=%d device %.3f ms/call host %.3f ms/call "
                "speedup %.2fx\n",
                (long long)m, (long long)k, (long long)n, repeats, dev_ms, host_ms,
                dev_ms > 0.0 ? host_ms / dev_ms : 0.0);
    if (bad != 0) {
        const size_t i = (size_t)first;
        std::printf("  %lld of %lld accumulators differ, worst |d|=%lld, first at [%lld,%lld]: "
                    "device %d host %d\n",
                    (long long)bad, (long long)got.size(), (long long)worst, (long long)(i / n),
                    (long long)(i % n), got[i], ref[i]);
        std::printf("FAIL NPU q8 gemm differs from the exact host accumulators\n");
        return 1;
    }
    std::printf("PASS NPU q8 gemm matches the exact host accumulators on %lld values, device "
                "%.3f ms/call vs host %.3f ms/call\n",
                (long long)got.size(), dev_ms, host_ms);
    return 0;
}
