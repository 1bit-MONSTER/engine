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
// npu/model/npu_model.h — `NpuModel` (engine#799 P4 / #803, step A3): the half of the forward that belongs to the
// model rather than to the target.
//
// The split's contract (npu/target/target.h) is that the model owns the maths and the state, and a target owns the
// runtime, the kernels and the weight residency. A3 is the step that moves that state here; this first slice takes
// the ownership the design names first — **the target itself** — so the rest of the move becomes a mechanical
// relocation of members into a class that already exists and already holds what the model needs.
//
// Why it is safe to land before the numerical gate (the doc's A3 gate is window11 text identity, logits KLD 0 and a
// profile row, all needing device windows): **no expression moves.** `Q4nxNpuForward` keeps its members and its code
// and binds a reference to what this class owns, so the ~94 `xd_->…` call sites and every arithmetic statement stay
// byte-identical; the forward is already non-copyable (copy ctor and assignment deleted), so a reference member
// cannot change its semantics either. What changes is *who owns the target*, which is the structural claim A3 makes.
#ifndef ONEBIT_NPU_MODEL_NPU_MODEL_H
#define ONEBIT_NPU_MODEL_NPU_MODEL_H

#include <memory>

#include "target/xdna/xdna_target.h"   // npu-relative: the CI syntax target has no tree root on its path

namespace npu {

class NpuModel {
  public:
    // The target this model runs on. Null until `Q4nxNpuForward::init()` creates it with the device, exactly as the
    // forward's own member was: every use site already null-checks it (`if (xd_ && …)`).
    std::unique_ptr<npu::target::xdna::XdnaTarget> xd;
};

}  // namespace npu

#endif  // ONEBIT_NPU_MODEL_NPU_MODEL_H
