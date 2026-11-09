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
// npu/model/op_dispatch.h — the model's side of the optional-op contract (engine#799 P4 / #803, step A3).
//
// `npu/target/target.h` says every op except `gemv` is optional: "the model falls back to its own host
// implementation when `supports(op)` is false". That sentence had no caller anywhere in the tree --
// `git grep 'supports(Op::'` found nothing -- so the contract was declared and unimplemented. This header is
// the missing side: one function per optional op, each of which the model calls FIRST, and which answers
// "the target did it, do not run yours" or "it did not, run yours".
//
// Two properties are deliberate, and both are pinned by tests/npu_op_dispatch_test.cpp:
//
//   1. `supports()` ALONE does not decide. A target that claims an op and then fails it (`gdn_step` returns
//      false because its state is not resident, say) must not swallow the op -- the model still has to run
//      its own implementation, or the layer silently produces garbage. So the answer is
//      `supports(op) && <op>(...)`, never `supports(op)`.
//   2. The operands are the ones the contract in target.h spells out, not the model's internal shapes. A
//      helper takes the operands; a target's state (weights, KV, recurrence) is its own, made resident by
//      `load_block()`. That is why the three helpers do not merely forward a `GemvArgs`.
//
// What is wired today and what is not, stated here rather than discovered later:
//
//   lm_head    wired in Q4nxNpuForward::run_lm_head. Its contract is complete without residency -- one
//              hidden row [H] to the vocabulary [V], the weights are the model's, and it is idempotent --
//              so a target can honour it the moment it has a head context.
//   gdn_step   NOT wired. The contract requires the layer's parameters and recurrence state to be resident
//   attn_step  ("come from the layer that load_block() made resident"), and `XdnaTarget::load_block()` is a
//              documented no-op today ("weights still fill lazily per projection"). Wiring the dispatch
//              before residency exists would hand a target an op it cannot honour and hide the reason in a
//              fallback that always fires. The helpers exist, are tested against stub targets, and are the
//              seam those two ops land on when A2c/load_block arrives.
#ifndef ONEBIT_NPU_MODEL_OP_DISPATCH_H
#define ONEBIT_NPU_MODEL_OP_DISPATCH_H

#include "target/target.h"

namespace npu::model {

// The GDN delta-rule step for one layer over M rows: x is [M][H] in, y is [M][H] out. The layer's
// parameters and recurrence state are the target's (see target.h's contract), which is why they are not
// arguments here. Not idempotent: a target that runs it advances the layer's state.
inline bool run_gdn_step(target::NpuTarget& t, int layer, int M, const float* x, float* y) {
    if (!t.supports(target::Op::gdn_step)) return false;
    return t.gdn_step(layer, M, x, y);
}

// The attention step for one layer over M rows: q is [M][NH*HD] in, o is [M][NH*HD] out. Keys and values
// come from the layer's own cache, never from the arguments (same reason as above).
inline bool run_attn_step(target::NpuTarget& t, int layer, int M, const float* q, float* o) {
    if (!t.supports(target::Op::attn_step)) return false;
    return t.attn_step(layer, M, q, o);
}

// One hidden row [H] to the vocabulary vector [V]. Idempotent: no state moves. The caller owns the
// buffers; the target writes exactly V floats into `logits`.
inline bool run_lm_head(target::NpuTarget& t, const float* h, float* logits) {
    if (!t.supports(target::Op::lm_head)) return false;
    return t.lm_head(h, logits);
}

}  // namespace npu::model

#endif  // ONEBIT_NPU_MODEL_OP_DISPATCH_H
