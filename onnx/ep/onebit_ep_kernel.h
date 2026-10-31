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

#pragma once

#include <cstdint>

namespace onebit_ep {

// Row-major A[m*k] x B[k*n] -> C[m*n], float32, C fully written (not accumulated into).
// Returns true when the NPU did the multiply (false = the host loop ran).
bool gemm_f32(const float* a, const float* b, float* c, int64_t m, int64_t k, int64_t n);

// The A8W8 QDQ multiply in the quantised domain: int8 x int8 -> int8, no float tensor in between.
//
//   y_q[m][n] = saturate(round( (Sum_k (a_q[m][k]-zp_x)*(w_q[k][n]-zp_w)) * s_x[m] * s_w[n] / s_y )
//                        + zp_y)
//
// which is what DequantizeLinear -> MatMul -> QuantizeLinear computes, with the dequantised
// activations and weights never written anywhere. a is [m][k] int8 row-major, b is [k][n] int8
// row-major, y is [m][n] int8. The accumulator is exact int32, the zero-point corrections are exact
// int64, and the only float work is the scale and one round-half-to-even per output element (ONNX
// QuantizeLinear's rounding, which happens *before* the output zero point is added).
//
// Asymmetric zero points do not cost a second pass. Writing the shifted product out,
//
//   Sum_k (a-zx)(w-zw) = Sum_k a*w  -  zx*Sum_k w  -  zw*Sum_k a  +  K*zx*zw
//
// the first term is what the GEMM accumulates, the second is a column sum of the weight (precomputed
// once, see `colsum_w`), and the third is a row sum of the activation (one extra add per k while the
// row is already being read).
struct Q8Epilogue {
    // Scales: per-tensor in the scalar, per-axis in the pointer (m values for the row axis, n for the
    // column axis). A null pointer means "use the scalar"; the same convention for the zero points.
    const float* s_x_row = nullptr;
    const float* s_w_col = nullptr;
    float s_x = 1.0f;
    float s_w = 1.0f;
    float inv_s_y = 1.0f;  // 1 / output scale
    const int8_t* zp_x_row = nullptr;
    const int8_t* zp_w_col = nullptr;
    int32_t zp_x = 0;
    int32_t zp_w = 0;
    int32_t zp_y = 0;  // output zero point, added after rounding
    // sum_k w_q[k][n], n values. Required when zp_x is (or may be) non-zero; may be null otherwise.
    const int32_t* colsum_w = nullptr;
};

void gemm_q8(const int8_t* a, const int8_t* b, int8_t* y, int64_t m, int64_t k, int64_t n,
             const Q8Epilogue& epilogue);

}  // namespace onebit_ep
