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

// The two references this build implements.  `gemv_ref` is C[m][n] = A[m][k] x B[k][n]; `lm_head_ref` is
// the same product with m == 1 (the plan layer's lm-head contract), kept separate so its own cases name the
// shape it is used at.
bool gemv_ref(const float* a, int m, int k, int n, const float* b, float* out);
bool lm_head_ref(const float* a, int m, int k, int n, const float* b, float* out);

}  // namespace onebit::npu::plan
