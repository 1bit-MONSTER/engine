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
#include "plan/op_registry.h"

#include <cstring>

namespace onebit::npu::plan {

const char* op_name(OpId op) {
    switch (op) {
        case OpId::gemv: return "gemv";
        case OpId::gdn_step: return "gdn_step";
        case OpId::attn_step: return "attn_step";
        case OpId::lm_head: return "lm_head";
    }
    return "?";
}

// ---------------------------------------------------------------------------------------------------
// The references.  Deliberately the plainest possible statement of the contract: i-k-j accumulation in
// double, no blocking, no padding, no device.  That is what makes them usable as an oracle -- an
// implementation that shares the model path's optimisations could share its mistakes.
//
// The `a == 0` skip is kept because the model's host path has it and a caller comparing against this
// reference must not depend on whether a zero contributes: skipping an exact zero cannot change a sum of
// finite values, and the case below pins that a zero row stays zero.
bool gemv_ref(const float* a, int m, int k, int n, const float* b, float* out) {
    if (!a || !b || !out || m <= 0 || k <= 0 || n <= 0) return false;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int p = 0; p < k; p++) {
                const double av = (double)a[(size_t)i * k + p];
                if (av == 0.0) continue;
                acc += av * (double)b[(size_t)p * n + j];
            }
            out[(size_t)i * n + j] = (float)acc;
        }
    }
    return true;
}

bool lm_head_ref(const float* a, int m, int k, int n, const float* b, float* out) {
    if (m != 1) return false;   // the lm-head contract is one hidden row
    return gemv_ref(a, m, k, n, b, out);
}

// ---------------------------------------------------------------------------------------------------
// Known answers.  Every expected value below was computed by hand from the operands shown, and the `how`
// string says so -- they are not the reference's own output.
namespace {

// gemv, m=1 k=3 n=2.  A = [1, 2, 3]; B (k x n) = [[1,0],[0,1],[1,1]].
//   out[0] = 1*1 + 2*0 + 3*1 = 4
//   out[1] = 1*0 + 2*1 + 3*1 = 5
const float g1_a[] = {1.0f, 2.0f, 3.0f};
const float g1_b[] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
const float g1_want[] = {4.0f, 5.0f};

// gemv, m=2 k=2 n=2 -- a path the first case does not take: a zero row (the `a == 0` skip) and negative
// operands.  A = [[0,0],[2,-3]]; B = [[1,2],[3,4]].
//   row 0 = [0, 0]                                  (the skip must leave zeros, not garbage)
//   row 1 = [2*1 + -3*3, 2*2 + -3*4] = [-7, -8]
const float g2_a[] = {0.0f, 0.0f, 2.0f, -3.0f};
const float g2_b[] = {1.0f, 2.0f, 3.0f, 4.0f};
const float g2_want[] = {0.0f, 0.0f, -7.0f, -8.0f};

const OpCase gemv_cases[] = {
    {"identity-ish 1x3x2", "hand: 1*1+2*0+3*1=4, 1*0+2*1+3*1=5", 1, 3, 2, g1_a, g1_b, g1_want},
    {"zero row + negatives, 2x2x2", "hand: row1 = 2*1+-3*3=-7, 2*2+-3*4=-8", 2, 2, 2, g2_a, g2_b, g2_want},
};

// lm_head, m=1 k=3 n=2.  h = [0.5, -1, 2]; W (k x n) = [[1,1],[2,0],[0,3]].
//   y[0] = 0.5*1 + -1*2 + 2*0 = -1.5
//   y[1] = 0.5*1 + -1*0 + 2*3 =  6.5
const float h1_a[] = {0.5f, -1.0f, 2.0f};
const float h1_b[] = {1.0f, 1.0f, 2.0f, 0.0f, 0.0f, 3.0f};
const float h1_want[] = {-1.5f, 6.5f};

const OpCase lm_head_cases[] = {
    {"hidden 3 -> vocab 2", "hand: 0.5*1+-1*2+2*0=-1.5, 0.5*1+-1*0+2*3=6.5", 1, 3, 2, h1_a, h1_b,
     h1_want},
};

const OpEntry kEntries[] = {
    {OpId::gemv, gemv_ref, gemv_cases, sizeof(gemv_cases) / sizeof(gemv_cases[0])},
    {OpId::lm_head, lm_head_ref, lm_head_cases, sizeof(lm_head_cases) / sizeof(lm_head_cases[0])},
};

}  // namespace

size_t op_entry_count() { return sizeof(kEntries) / sizeof(kEntries[0]); }
const OpEntry& op_entry(size_t i) { return kEntries[i]; }

const OpEntry* find_op(OpId op) {
    for (const OpEntry& e : kEntries)
        if (e.op == op) return &e;
    return nullptr;
}

const OpEntry* find_op(const char* name) {
    if (!name) return nullptr;
    for (const OpEntry& e : kEntries)
        if (std::strcmp(op_name(e.op), name) == 0) return &e;
    return nullptr;
}

std::string op_coverage() {
    // The ops a plan's layers can require.  `gemv` is every layer's projection; `gdn_step` is a `gdn`
    // layer's recurrence; `attn_step` the attention step; `lm_head` the output projection.  The two that
    // are still missing are named here and the test asserts this exact string, so the gap cannot grow
    // silently and cannot be closed without editing the test.
    std::string missing;
    for (OpId op : {OpId::gemv, OpId::gdn_step, OpId::attn_step, OpId::lm_head}) {
        if (find_op(op)) continue;
        if (!missing.empty()) missing += ",";
        missing += op_name(op);
    }
    return missing;
}

}  // namespace onebit::npu::plan
