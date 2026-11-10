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
// batch_check_compare.h -- the comparator behind NPU_BATCH_CHECK (engine#1781).
//
// The oracle that guards a batched verify pass used to compare ONE argmax at ONE position: the last row's. That
// answers "did the final argmax survive batching", which is not the question a speculative decoder asks -- it needs
// every drafted position's LOGITS (the accept rule consumes the target's probability at the drafted token, and two
// passes can agree on an argmax while disagreeing on the distribution around it).
//
// Pure and header-only on purpose: the comparison is arithmetic over two float arrays, so it can be pinned by a host
// test with no device, no model and no runtime, while the device-only half stays a single call in the forward.

#ifndef ONEBIT_NPU_FORWARD_BATCH_CHECK_COMPARE_H
#define ONEBIT_NPU_FORWARD_BATCH_CHECK_COMPARE_H

#include <cmath>
#include <cstdint>
#include <cstring>

namespace onebit {
namespace npu {

struct RowMismatch {
    int row = -1;            // first row that differs, -1 when every row matched
    int index = -1;          // column of the first difference within that row
    uint32_t seq_bits = 0;   // the two values, as bits, because a float print of two near-equal values reads as equal
    uint32_t bat_bits = 0;
    bool nan_mismatch = false;  // one side NaN and the other not (bit comparison alone would call two NaNs equal)
};

// Compare `rows` x `cols` float rows BITWISE. Returns the number of differing rows (0 == identical).
//
// Bitwise, not tolerance-based: the gate is "the batched pass is the same arithmetic as the sequential one", so a
// value that differs in the last mantissa bit is a divergence to report, not a rounding to forgive. NaN is handled
// explicitly because the usual bit comparison would consider two NaN payloads equal and, more importantly, a NaN on
// one side with a finite number on the other is exactly the failure this oracle exists to name.
inline int compare_rows_bitwise(const float* seq, const float* bat, int rows, int cols, RowMismatch* first) {
    if (!seq || !bat || rows <= 0 || cols <= 0) return rows > 0 ? rows : 0;
    int differing = 0;
    for (int r = 0; r < rows; ++r) {
        const float* s = seq + (size_t)r * (size_t)cols;
        const float* b = bat + (size_t)r * (size_t)cols;
        for (int i = 0; i < cols; ++i) {
            const bool s_nan = std::isnan(s[i]), b_nan = std::isnan(b[i]);
            if (s_nan || b_nan) {
                if (s_nan && b_nan) continue;   // both not-a-number: the same non-answer, not a divergence
                ++differing;
                if (first && first->row < 0) {
                    first->row = r; first->index = i; first->nan_mismatch = true;
                    std::memcpy(&first->seq_bits, &s[i], 4);
                    std::memcpy(&first->bat_bits, &b[i], 4);
                }
                break;                          // one report per row is enough to know the row differs
            }
            uint32_t sb = 0, bb = 0;
            std::memcpy(&sb, &s[i], 4);
            std::memcpy(&bb, &b[i], 4);
            if (sb != bb) {
                ++differing;
                if (first && first->row < 0) {
                    first->row = r; first->index = i;
                    first->seq_bits = sb; first->bat_bits = bb;
                }
                break;
            }
        }
    }
    return differing;
}

}  // namespace npu
}  // namespace onebit

#endif  // ONEBIT_NPU_FORWARD_BATCH_CHECK_COMPARE_H
