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
//   y_q[m][n] = clamp(round( (Sum_k a_q[m][k]*w_q[k][n]) * s_x * s_w / s_y ), -128, 127)
//
// which is what DequantizeLinear -> MatMul -> QuantizeLinear computes, with the dequantised
// activations and weights never written anywhere. a is [m][k] int8 row-major, b is [k][n] int8
// row-major, y is [m][n] int8. The accumulator is exact int32 and the only float work is one
// multiply and one round-half-to-even per output element (ONNX QuantizeLinear's rounding).
void gemm_q8(const int8_t* a, const int8_t* b, int8_t* y, int64_t m, int64_t k, int64_t n,
             float s_x, float s_w, float inv_s_y);

}  // namespace onebit_ep
