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

#include "onebit_ep_kernel.h"

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>
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

// One output element of the folded QuantizeLinear: round half to even (nearbyint under the default
// FE_TONEAREST), add the zero point, saturate to int8. The order matters -- rounding after adding an
// integer zero point would move exact ties to the wrong neighbour.
static inline int8_t q8_quantize(float v, int32_t zp) {
    float r = std::nearbyint(v);
    // The output is int8 whatever happens next: keep the integer add in range.
    if (r > 1.0e6f) r = 1.0e6f;
    if (r < -1.0e6f) r = -1.0e6f;
    const int32_t q = zp + (int32_t)r;
    if (q > 127) return 127;
    if (q < -128) return -128;
    return (int8_t)q;
}

void gemm_q8(const int8_t* a, const int8_t* b, int8_t* y, int64_t m, int64_t k, int64_t n,
             const Q8Epilogue& e) {
    // The whole dequantise/requantise chain collapses to one scale per output element; the column half
    // of it (and 1/s_y) is folded once, up front.
    std::vector<float> wmul((size_t)n);
    const float sw = e.s_w * e.inv_s_y;
    for (int64_t j = 0; j < n; ++j) wmul[(size_t)j] = e.s_w_col ? e.s_w_col[j] * e.inv_s_y : sw;

    const bool asym = e.zp_x != 0 || e.zp_w != 0 || e.zp_x_row || e.zp_w_col || e.zp_y != 0;
    const bool per_tensor = !asym && !e.s_x_row && !e.s_w_col;
    const float scale = e.s_x * e.s_w * e.inv_s_y;  // per-tensor symmetric path, hoisted
    const bool needs_rowsum = e.zp_w != 0 || e.zp_w_col != nullptr;

    // One row of exact int32 accumulators -> int8. Shared by the device path and the host loop, so the
    // scales, zero points and rounding exist in exactly one place: a device run and a host run of the
    // same graph differ only where the accumulator itself would differ (never, it is exact integer).
    auto epilogue_row = [&](const int32_t* acc, int64_t i, int8_t* yrow) {
        if (per_tensor) {
            for (int64_t j = 0; j < n; ++j) yrow[j] = q8_quantize((float)acc[j] * scale, 0);
            return;
        }
        const float sx_row = e.s_x_row ? e.s_x_row[i] : e.s_x;
        const int64_t zx = e.zp_x_row ? e.zp_x_row[i] : e.zp_x;
        int64_t rowsum = 0;  // sum_k a_q[m][k]: only a non-zero zp_w needs it
        if (asym && needs_rowsum) {
            const int8_t* arow = a + i * k;
            for (int64_t p = 0; p < k; ++p) rowsum += arow[p];
        }
        for (int64_t j = 0; j < n; ++j) {
            int64_t c = acc[j];
            if (asym) {
                const int64_t zw = e.zp_w_col ? e.zp_w_col[j] : e.zp_w;
                if (zx) c -= zx * (int64_t)e.colsum_w[j];
                if (zw) {
                    c -= zw * rowsum;
                    c += (int64_t)k * zx * zw;
                }
            }
            yrow[j] = q8_quantize((float)c * sx_row * wmul[(size_t)j], e.zp_y);
        }
    };

#ifdef ONEBIT_EP_NPU
    // The device leg: int8 in, exact int32 accumulators out, the epilogue above on the way back. A
    // graph whose (K, N) has no ELF, or a busy device, leaves the row blocks to the host loop below.
    {
        std::vector<int32_t> acc((size_t)m * (size_t)n);
        std::string why;
        if (onebit::npu::gemm_q8_raw(a, m, k, n, b, acc.data(), &why)) {
            for (int64_t i = 0; i < m; ++i) epilogue_row(acc.data() + i * n, i, y + i * n);
            return;
        }
        if (const char* trace = std::getenv("ONEBIT_EP_TRACE"))
            if (FILE* f = std::fopen(trace, "a")) {
                std::fprintf(f, "npu_qdq_fallback k=%lld n=%lld why=%s\n", (long long)k,
                             (long long)n, why.c_str());
                std::fclose(f);
            }
    }
#endif

    std::vector<int32_t> acc((size_t)n);
    for (int64_t i = 0; i < m; ++i) {
        std::fill(acc.begin(), acc.end(), 0);
        // i-k-j: exact int32 accumulation, one B row walked once per A row.
        for (int64_t p = 0; p < k; ++p) {
            const int32_t aik = a[i * k + p];
            if (aik == 0) continue;
            const int8_t* brow = b + p * n;
            for (int64_t j = 0; j < n; ++j) acc[(size_t)j] += aik * brow[j];
        }
        epilogue_row(acc.data(), i, y + i * n);
    }
}

}  // namespace onebit_ep
