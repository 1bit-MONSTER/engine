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
// The optional-op dispatch (npu/model/op_dispatch.h, engine#803 step A3): the model asks the target first and
// runs its own implementation when the target does not do the op. Header only, no device, no model, no
// runtime library -- the targets here are stubs that record what they were asked.
//
// The cases that matter are the two ways a fallback goes wrong:
//   * a target that does not claim the op must leave it to the model (`gemv`-only stub), and
//   * a target that CLAIMS the op and then FAILS it must also leave it to the model -- otherwise the layer
//     silently produces whatever the half-executed op left behind. That second case is why the dispatch is
//     `supports() && op()` and not `supports()`, and it is the one a "does it compile" review cannot see.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "model/op_dispatch.h"
#include "target/target.h"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (ok) {
        std::printf("ok   %s\n", what);
    } else {
        std::printf("FAIL %s\n", what);
        failures++;
    }
}

// A target that implements only what the interface requires: no optional op.
class GemvOnly final : public npu::target::NpuTarget {
  public:
    const char* name() const override { return "gemv-only"; }
    bool open(const onebit::npu::plan::Plan&, const std::string&, const char*, std::string&) override {
        return true;
    }
    bool load_block(int, int, std::string& err) override {
        err = "stub";
        return false;
    }
    bool gemv(const npu::target::GemvArgs&, std::vector<float>& out, std::string&) override {
        out.assign(1, 0.0f);
        return true;
    }
    npu::target::Caps caps() const override { return {}; }
};

// A target that claims every optional op and runs it.
class ClaimsAndRuns : public npu::target::NpuTarget {
  public:
    int gdn_calls = 0, attn_calls = 0, head_calls = 0;

    const char* name() const override { return "claims-and-runs"; }
    bool open(const onebit::npu::plan::Plan&, const std::string&, const char*, std::string&) override {
        return true;
    }
    bool load_block(int, int, std::string&) override { return true; }
    bool gemv(const npu::target::GemvArgs&, std::vector<float>& out, std::string&) override {
        out.assign(1, 0.0f);
        return true;
    }
    bool gdn_step(int, int, const float*, float* y) override {
        gdn_calls++;
        if (y) y[0] = 1.0f;
        return true;
    }
    bool attn_step(int, int, const float*, float* o) override {
        attn_calls++;
        if (o) o[0] = 2.0f;
        return true;
    }
    bool lm_head(const float*, float* logits) override {
        head_calls++;
        if (logits) logits[0] = 3.0f;
        return true;
    }
    bool supports(npu::target::Op op) const override {
        return op == npu::target::Op::gemv || op == npu::target::Op::gdn_step ||
               op == npu::target::Op::attn_step || op == npu::target::Op::lm_head;
    }
    npu::target::Caps caps() const override { return {}; }
};

// A target that claims every optional op and then fails it: the case where the model must still run its own.
class ClaimsThenFails final : public ClaimsAndRuns {
  public:
    const char* name() const override { return "claims-then-fails"; }
    bool gdn_step(int, int, const float*, float*) override {
        gdn_calls++;
        return false;
    }
    bool attn_step(int, int, const float*, float*) override {
        attn_calls++;
        return false;
    }
    bool lm_head(const float*, float*) override {
        head_calls++;
        return false;
    }
};

}  // namespace

int main() {
    float x[4] = {1, 2, 3, 4};
    float y[4] = {0, 0, 0, 0};

    // 1. A target that implements only gemv leaves every optional op to the model.
    {
        GemvOnly t;
        check(!npu::model::run_gdn_step(t, 0, 1, x, y), "gemv-only target: gdn_step is left to the model");
        check(!npu::model::run_attn_step(t, 0, 1, x, y), "gemv-only target: attn_step is left to the model");
        check(!npu::model::run_lm_head(t, x, y), "gemv-only target: lm_head is left to the model");
        check(y[0] == 0.0f, "gemv-only target: nothing was written into the caller's buffer");
    }

    // 2. A target that claims and runs an op takes it, writes the contract's output, and is called once.
    {
        ClaimsAndRuns t;
        check(npu::model::run_gdn_step(t, 3, 1, x, y), "claiming target: gdn_step is taken from the target");
        check(y[0] == 1.0f, "claiming target: gdn_step's output landed in the caller's buffer");
        check(npu::model::run_attn_step(t, 3, 1, x, y), "claiming target: attn_step is taken from the target");
        check(y[0] == 2.0f, "claiming target: attn_step's output landed in the caller's buffer");
        check(npu::model::run_lm_head(t, x, y), "claiming target: lm_head is taken from the target");
        check(y[0] == 3.0f, "claiming target: lm_head's output landed in the caller's buffer");
        check(t.gdn_calls == 1 && t.attn_calls == 1 && t.head_calls == 1,
              "claiming target: each op ran exactly once (the dispatch does not double-call)");
    }

    // 3. The failure that a "supports() is enough" implementation would hide: claimed, then failed.
    {
        ClaimsThenFails t;
        check(!npu::model::run_gdn_step(t, 0, 1, x, y), "claimed-but-failed gdn_step still falls back to the model");
        check(!npu::model::run_attn_step(t, 0, 1, x, y), "claimed-but-failed attn_step still falls back to the model");
        check(!npu::model::run_lm_head(t, x, y), "claimed-but-failed lm_head still falls back to the model");
        check(t.gdn_calls == 1 && t.attn_calls == 1 && t.head_calls == 1,
              "claimed-but-failed: the target was asked exactly once per op (no retry loop in the dispatch)");
    }

    // 4. The mandatory op is not routed through this header: gemv has no fallback and no helper here, so a
    //    target that does not implement it cannot be constructed as a usable target at all. Asserted as an
    //    absence rather than a call: supports(gemv) is what the interface promises unconditionally.
    {
        GemvOnly t;
        check(t.supports(npu::target::Op::gemv), "gemv is mandatory: the interface reports it without a helper");
    }

    if (failures != 0) {
        std::printf("%d FAILED\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
