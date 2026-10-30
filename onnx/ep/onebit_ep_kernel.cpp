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
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// Unless required by applicable law or agreed to in writing, software
// limitations under the License.

// The one seam between the ORT plugin EP and whatever actually multiplies.
//
// Deliverable 2 ran every MatMul here on the host, so the EP ABI and the partitioning were what
// got tested with nothing moving underneath them. With ONEBIT_EP_NPU=ON the multiply goes to the
// existing NPU runtime GEMM instead (npu/ort/onebit_npu_gemm.*: I8Ctx, the forward route's
// packB + amax/127 quantisation, the device lock), and the host loop below stays as the fallback:
// a busy device, no kernel for this K x N, or any XRT failure must leave the session working,
// never fail the graph. Signature and contract are identical either way — row-major A[M,K] x
// B[K,N] -> C[M,N], C fully written — so the EP's numerics comparison stays the same test.
//
// This is deliberately not an interface or a dispatch table. One function, one call site.

#include <cstdint>
#include <cstring>

#ifdef ONEBIT_EP_NPU
#include "onebit_npu_gemm.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#endif

namespace onebit_ep {

// Returns true when the NPU did the multiply. Always writes all of C.
bool gemm_f32(const float* a, const float* b, float* c, int64_t m, int64_t k, int64_t n) {
#ifdef ONEBIT_EP_NPU
    std::string why;
    if (onebit::npu::gemm_f32(a, m, k, n, b, c, &why)) return true;
    if (const char* trace = std::getenv("ONEBIT_EP_TRACE"))
        if (FILE* f = std::fopen(trace, "a")) {
            std::fprintf(f, "npu_fallback k=%lld n=%lld why=%s\n", (long long)k, (long long)n,
                         why.c_str());
            std::fclose(f);
        }
#endif
    std::memset(c, 0, (size_t)m * (size_t)n * sizeof(float));
    // i-k-j instead of i-j-k: each B row is walked once per A row, which is what keeps this
    // within a small factor of MLAS on shapes that matter here (K,N in the hundreds).
    for (int64_t i = 0; i < m; ++i) {
        float* crow = c + i * n;
        for (int64_t p = 0; p < k; ++p) {
            const float aik = a[i * k + p];
            if (aik == 0.0f) continue;
            const float* brow = b + p * n;
            for (int64_t j = 0; j < n; ++j) crow[j] += aik * brow[j];
        }
    }
    return false;
}

}  // namespace onebit_ep
