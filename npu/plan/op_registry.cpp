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

#include <cmath>
#include <cstring>
#include <vector>

namespace onebit::npu::plan {

const char* op_name(OpId op) {
    switch (op) {
        case OpId::gemv: return "gemv";
        case OpId::gdn_step: return "gdn_step";
        case OpId::attn_step: return "attn_step";
        case OpId::lm_head: return "lm_head";
        case OpId::count: break;
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
    // Every op in the vocabulary is walked, including one added tomorrow: the sentinel is what makes that
    // automatic.  `gemv` is every layer's projection, `gdn_step` a `gdn` layer's recurrence, `attn_step`
    // the attention step, `lm_head` the output projection.
    std::string missing;
    for (int i = 0; i < (int)OpId::count; i++) {
        const OpId op = (OpId)i;
        // A reference of EITHER shape counts: `attn_step`'s operands are not a matrix pair, so it is
        // registered in the call table (engine#799 rule (b) slice 2) and is no longer a gap.
        if (find_op(op) || find_op_call(op)) continue;
        if (!missing.empty()) missing += ",";
        missing += op_name(op);
    }
    return missing;
}

// ── op-shaped calls: attn_step (engine#799 rule (b), slice 2) ───────────────────────────────────────
namespace {
// One position's attention: out[d] = sum_i softmax(q . k_i / sqrt(hd))[i] * v_i[d].
constexpr float kAttnQ[] = {1.0f, 0.0f};
constexpr float kAttnK[] = {1.0f, 0.0f, 0.0f, 1.0f};   // two cached positions: (1,0), (0,1)
constexpr float kAttnV[] = {1.0f, 2.0f, 3.0f, 4.0f};   // (1,2), (3,4)
// By hand: scores = (q.k0, q.k1)/sqrt(2) = (0.7071, 0); softmax = (0.66976, 0.33024);
// out = 0.66976*(1,2) + 0.33024*(3,4) = (1.66048, 2.66048).
constexpr float kAttnWantTwo[] = {1.6604769013466862f, 2.6604769013466862f};
// With one cached position the softmax is 1 by definition, so the output is v0 exactly -- a case the
// reference cannot pass by accident, because it pins that the normaliser is the sum and not a bare exp.
constexpr float kAttnWantOne[] = {1.0f, 2.0f};

const OpCallCase kAttnCases[] = {
    {"two cached positions, hd 2",
     "by hand: scores (1/sqrt2, 0) -> softmax (0.66976, 0.33024) -> 0.66976*(1,2)+0.33024*(3,4)",
     {OpId::attn_step, kAttnQ, kAttnK, kAttnV, 2, 2, nullptr},
     kAttnWantTwo, 2},
    {"one cached position",
     "softmax over a single score is 1, so the output is v0 (an implementation that skipped the normaliser "
     "would pass the two-position case by luck but not this one)",
     {OpId::attn_step, kAttnQ, kAttnK, kAttnV, 1, 2, nullptr},
     kAttnWantOne, 2},
};
}  // namespace

bool attn_step_ref(const OpCall& call) {
    if (!call.q || !call.kcache || !call.vcache || !call.out || call.pos <= 0 || call.hd <= 0) return false;
    const double scale = 1.0 / std::sqrt(double(call.hd));
    std::vector<double> scores(static_cast<size_t>(call.pos));
    double best = -1e300;
    for (int i = 0; i < call.pos; ++i) {
        double s = 0.0;
        for (int d = 0; d < call.hd; ++d) s += double(call.q[d]) * double(call.kcache[i * call.hd + d]);
        scores[size_t(i)] = s * scale;
        if (scores[size_t(i)] > best) best = scores[size_t(i)];
    }
    double z = 0.0;
    for (auto& s : scores) {
        s = std::exp(s - best);
        z += s;
    }
    for (int d = 0; d < call.hd; ++d) {
        double acc = 0.0;
        for (int i = 0; i < call.pos; ++i) acc += scores[size_t(i)] / z * double(call.vcache[i * call.hd + d]);
        call.out[d] = float(acc);
    }
    return true;
}

namespace {
// ── gdn_step: the delta-rule core ──────────────────────────────────────────────────────────────────
namespace {
constexpr float kGdnK[] = {2.0f};
constexpr float kGdnV[] = {3.0f};
constexpr float kGdnQ[] = {1.0f};
constexpr float kGdnWantNoDecay[] = {3.0f};    // S = 1 + 2*1
constexpr float kGdnWantDecay[] = {4.5f};      // S = 0.5 + 2*2
constexpr float kGdnWantNoWrite[] = {0.5f};    // S = 0.5, the gate writes nothing
// Each case owns its state buffer (the reference advances it in place); the test resets it to the case's
// documented initial S = 1 before running, so the table stays re-runnable.
float kGdnS0[] = {1.0f}, kGdnS1[] = {1.0f}, kGdnS2[] = {1.0f};

const OpCallCase kGdnCases[] = {
    {"dk = dv = 1, no decay, gate open",
     "by hand: S 1, k 2, v 3, g 0, beta 1 -> err = 3 - 1*2 = 1, S = 1 + 2*1 = 3, out = S*q = 3",
     {OpId::gdn_step, kGdnQ, nullptr, nullptr, 0, 0, nullptr, kGdnS0, kGdnK, kGdnV, 1, 1, 1.0f, 0.0f},
     kGdnWantNoDecay, 1},
    {"dk = dv = 1, half decay, gate open",
     "by hand: S 1*exp(ln 0.5) = 0.5, err = 3 - 0.5*2 = 2, S = 0.5 + 2*2 = 4.5, out = 4.5",
     {OpId::gdn_step, kGdnQ, nullptr, nullptr, 0, 0, nullptr, kGdnS1, kGdnK, kGdnV, 1, 1, 1.0f, -0.6931471805599453f},
     kGdnWantDecay, 1},
    {"dk = dv = 1, half decay, gate closed",
     "by hand: S = 0.5 and beta = 0 writes nothing, so out = 0.5 -- pins that the gate multiplies the write "
     "rather than being added to it",
     {OpId::gdn_step, kGdnQ, nullptr, nullptr, 0, 0, nullptr, kGdnS2, kGdnK, kGdnV, 1, 1, 0.0f, -0.6931471805599453f},
     kGdnWantNoWrite, 1},
};
}  // namespace

bool gdn_delta_ref(const OpCall& call) {
    if (!call.state || !call.k || !call.v || !call.q || !call.out || call.dk <= 0 || call.dv <= 0) return false;
    const int dk = call.dk, dv = call.dv;
    std::vector<double> S(static_cast<size_t>(dk) * dv), err(static_cast<size_t>(dv));
    for (size_t i = 0; i < S.size(); ++i) S[i] = double(call.state[i]);
    const double decay = std::exp(double(call.g));
    for (auto& s : S) s *= decay;                       // 1. decay the state
    for (int j = 0; j < dv; ++j) {                      // 2. err = v - S^T k  (S is [dk][dv])
        double acc = 0.0;
        for (int i = 0; i < dk; ++i) acc += S[size_t(i) * dv + j] * double(call.k[i]);
        err[size_t(j)] = double(call.v[j]) - acc;
    }
    for (int i = 0; i < dk; ++i)                        // 3. rank-1 write: S += beta * k (x) err
        for (int j = 0; j < dv; ++j) S[size_t(i) * dv + j] += double(call.beta) * double(call.k[i]) * err[size_t(j)];
    for (int j = 0; j < dv; ++j) {                      // 4. readout: out = S^T q
        double acc = 0.0;
        for (int i = 0; i < dk; ++i) acc += S[size_t(i) * dv + j] * double(call.q[i]);
        call.out[j] = float(acc);
    }
    for (size_t i = 0; i < S.size(); ++i) call.state[i] = float(S[i]);
    return true;
}

const OpCallEntry kCallEntries[] = {
    {OpId::attn_step, &attn_step_ref, kAttnCases, sizeof(kAttnCases) / sizeof(kAttnCases[0]), nullptr},
    {OpId::gdn_step, &gdn_delta_ref, kGdnCases, sizeof(kGdnCases) / sizeof(kGdnCases[0]),
     "the delta-rule core (state update + readout); the layer's conv, q/k/v/a/b projections, per-head norms and "
     "gates are gemv-shaped and covered by the first table, and their parameters are what load_block() makes "
     "resident"},
};
}  // namespace

const OpCallEntry* find_op_call(OpId op) {
    for (const auto& e : kCallEntries)
        if (e.op == op) return &e;
    return nullptr;
}
size_t op_call_entry_count() { return sizeof(kCallEntries) / sizeof(kCallEntries[0]); }

std::string op_partial_coverage() {
    std::string partial;
    for (const auto& e : kCallEntries) {
        if (!e.scope || !*e.scope) continue;
        if (!partial.empty()) partial += ",";
        partial += op_name(e.op);
    }
    return partial;
}
const OpCallEntry& op_call_entry(size_t i) { return kCallEntries[i]; }

}  // namespace onebit::npu::plan
