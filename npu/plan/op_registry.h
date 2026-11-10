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
// npu/plan/op_registry.h -- every op a plan may name ships with a host reference, and the reference is
// checked against known answers before the op is allowed in a plan (engine#799).
//
// THE RULE, in three parts:
//   1. a plan may only name an op that has an entry here;
//   2. an entry carries a host reference (plain C, no device, no container) AND known-answer cases whose
//      expected values were computed independently of that reference -- by hand for the small shapes, or
//      by a second formulation, and the case says which;
//   3. `tests/npu_plan_op_reference` runs every entry against its cases, *including inputs the reference
//      was not written from* (negative operands, a zero row so a skip path runs, m > 1 so the row loop
//      runs), and `op_coverage()` names the ops a plan can require that still have no entry -- so the gap
//      is asserted rather than implied.
//
// WHY: this box has the failure mode the rule prevents. The fused A8W8 kernel was proven against the CPU
// EP (good); the HRX mm tiles in another lane were proven only by a benchmark emitting non-garbage tokens
// (not good); and engine#808's host fallback lost its activation pad with every gate green. A reference
// with known answers turns "the output looks right" into "the output is what an independent computation
// says it is", and it costs a table rather than a subsystem.
//
// A target implementing an op does not satisfy the rule: the reference is the engine's own, so a device
// result can be checked against arithmetic that never ran on the device.
#pragma once

#include <cstddef>
#include <string>

namespace onebit::npu::plan {

// The ops a plan may name.  These are the names `npu/target/target.h`'s `Op` uses, so a plan, a target and
// a registry entry cannot spell the same op three ways.
// `count` is a sentinel, not an op: it exists so the registry and the test can iterate the WHOLE vocabulary.
// Without it a fifth OpId added tomorrow with no entry leaves both green -- the reviewer of this PR named
// exactly that hole, and it is the hole the rule exists to close.
enum class OpId { gemv = 0, gdn_step, attn_step, lm_head, count };

// engine#799 rule 1, made a compile error rather than a test result.  The comment above says it: "Without [`count`] a
// fifth OpId added tomorrow with no entry leaves both green".  `op_coverage()` and the test do catch a missing entry --
// but only when someone runs them, which is after the enum, the table and the target's `Op` have all been edited by
// hand.  This turns the same omission into a build failure that cannot be skipped, and bumping the number IS the
// decision: whoever adds an op has to add its entry, its known-answer case and this constant in one commit, or nothing
// compiles.  When the number is bumped, `op_coverage()`'s exact-string assertion also forces the gap to be declared
// rather than left implicit.
static_assert(static_cast<int>(OpId::count) == 4,
              "a new OpId needs an entry in the registry table, a known-answer case in it, and this number bumped: "
              "bumping it is the decision, so a fifth op cannot arrive green by omission (engine#799 rule 1)");

const char* op_name(OpId op);

// One known-answer case.  `a` is m x k and `b` is k x n, both row-major; `want` is m x n.
struct OpCase {
    const char* what;   // one line: what this case pins
    const char* how;    // how `want` was derived, because a "known answer" is only worth its name if the
                        // provenance can be checked
    int m, k, n;
    const float* a;
    const float* b;
    const float* want;
};

// The arithmetic the op promises, on the host.
using OpRefFn = bool (*)(const float* a, int m, int k, int n, const float* b, float* out);

struct OpEntry {
    OpId op;
    OpRefFn ref;
    const OpCase* cases;
    size_t n_cases;
};

// nullptr for an op this build has no reference for -- that is what makes the rule bite.
const OpEntry* find_op(OpId op);
const OpEntry* find_op(const char* name);

// The test walks the table rather than naming entries, so a new entry cannot be added without being run.
size_t op_entry_count();
const OpEntry& op_entry(size_t i);

// The ops in the vocabulary that have no entry yet, comma separated ("" when none).  It iterates every
// OpId up to the `count` sentinel, so an op added to the enum is included automatically -- the test asserts
// the exact string, so the gap is a decision rather than an omission.
std::string op_coverage();

// ── op-shaped calls (engine#799 rule (b), slice 2) ──────────────────────────────────────────────────
// `OpRefFn` above is GEMM-shaped (`a, m, k, n, b, out`), so the two ops whose operands are not a matrix
// pair -- `attn_step` and `gdn_step` -- could not be registered at all, which is why `op_coverage()` named
// them as gaps. `OpCall` is the operand set they need. The GEMM table stays as it is (its entries and
// cases are the shipped contract); this is an additive second shape and `op_coverage()` now means "no
// reference of EITHER shape", which is the honest question a plan asks.
struct OpCall {
    OpId op = OpId::gemv;
    // attn_step: one query against `pos` cached keys/values of one head.
    //   q      [hd]
    //   kcache [pos][hd], vcache [pos][hd]   (row-major, one row per cached position)
    //   out    [hd] = softmax(q . k_i / sqrt(hd)) over i, applied to v
    const float* q = nullptr;
    const float* kcache = nullptr;
    const float* vcache = nullptr;
    int pos = 0, hd = 0;
    float* out = nullptr;
    // gdn_step: the delta-rule CORE of one layer step -- the stateful, order-sensitive half.
    //   state [dk][dv] in and out (the recurrence, advanced in place)
    //   k     [dk], v [dv], q [dk]   the layer's projected vectors for this position
    //   beta  the write gate, g the log decay (g = 0 means no decay; exp(g) is applied to the state)
    // The projections and norms around the core are gemv/elementwise ops covered by the first table, and
    // `npu/target/target.h`'s CONTRACT says the layer takes them from what `load_block()` made resident --
    // so a reference for the whole layer would have to be a reimplementation of it, which is not what this
    // entry claims. Scope is stated here and in every case's `how`, because a partial oracle presented as a
    // whole one is exactly the failure this registry exists to prevent.
    float* state = nullptr;
    const float* k = nullptr;
    const float* v = nullptr;
    int dk = 0, dv = 0;
    float beta = 0.0f, g = 0.0f;
};
using OpCallRefFn = bool (*)(const OpCall&);

// One known-answer case for an op-shaped call: the case owns its operands, so the test can run it without
// knowing the op's shape. `want` is `hd` floats.
struct OpCallCase {
    const char* what;
    const char* how;          // how the expectation was derived, same rule as OpCase::how
    OpCall call;              // operands; `out` is filled by the reference
    const float* want;
    int want_len;
};

struct OpCallEntry {
    OpId op;
    OpCallRefFn ref;
    const OpCallCase* cases;
    size_t n_cases;
    // Empty when the reference covers the WHOLE op as a plan names it. Non-empty says which part it covers,
    // because a partial oracle presented as a whole one is the failure this registry exists to prevent:
    // `gdn_step` as a plan names it is a whole layer (conv, four projections, per-head norms and gates, then the
    // delta rule), and only the delta rule is hand-derivable -- the rest comes from what `load_block()` made
    // resident. `op_partial_coverage()` is what keeps that a stated gap instead of an implied coverage.
    const char* scope;
};

const OpCallEntry* find_op_call(OpId op);

// The ops whose entry states a PARTIAL scope, comma separated ("" when every entry is whole). The companion
// question to op_coverage(): that one says "nothing registered", this one says "registered, and here is what it
// does not cover" -- so a plan naming such an op is known to be less guarded than one naming a whole entry.
std::string op_partial_coverage();
size_t op_call_entry_count();
const OpCallEntry& op_call_entry(size_t i);

// `attn_step`: one query, one head, causal-agnostic over the cached positions.  Hand-derived answers in the
// cases: with hd = 2 and q = (1, 0), the first score is 1/sqrt(2) and the second 0, so the softmax is
// (0.6698, 0.3302) and the output (1.6605, 2.6605); with pos = 1 the softmax is 1 and the output IS v0.
bool attn_step_ref(const OpCall& call);

// `gdn_step`'s delta-rule core: S <- exp(g) * S; err = v - S^T k; S <- S + beta * k (x) err; out = S^T q.
// Hand-derived cases (dk = dv = 1, so each one is one line of arithmetic a reader can redo):
//   S 1, k 2, v 3, g 0,  beta 1, q 1  ->  S stays 1, err = 3 - 1*2 = 1, S = 1 + 2*1 = 3, out = 3
//   same but g = ln(0.5)              ->  S = 0.5, err = 3 - 0.5*2 = 2, S = 0.5 + 2*2 = 4.5, out = 4.5
//   same but beta 0                   ->  S = 0.5 only (no write), out = 0.5
// The g = 0 case and the beta = 0 case together pin the ORDER (decay, then write) and that a zero gate
// really writes nothing; an implementation that added before decaying passes neither.
bool gdn_delta_ref(const OpCall& call);

// The two references this build implements.  `gemv_ref` is C[m][n] = A[m][k] x B[k][n]; `lm_head_ref` is
// the same product with m == 1 (the plan layer's lm-head contract), kept separate so its own cases name the
// shape it is used at.
bool gemv_ref(const float* a, int m, int k, int n, const float* b, float* out);
bool lm_head_ref(const float* a, int m, int k, int n, const float* b, float* out);

}  // namespace onebit::npu::plan
