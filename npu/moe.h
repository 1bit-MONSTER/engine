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

// A mixture-of-experts model directory's routed experts, and the router that picks them
// (docs/npu.md, "Routed expert stacks"). The container stores each layer's routed stack as ONE
// tensor whose expert e is tile rows [e*rows/32, (e+1)*rows/32), and __metadata__.moe records
// that decomposition plus every expert's byte offset — so a dispatch can point at one expert
// without deriving anything, and the host can check the offsets against the tensor before a
// kernel ever sees them.
//
// The router stays on the host: it is one matvec on the hidden state, and its top-k has to be
// the model's own. router_topk is the same arithmetic as the reference (tools/moe_q4nx_ref.py),
// in float32: softmax over the bf16 router logits, the k largest, then renormalised so the k
// weights sum to 1 (Qwen3MoE `norm_topk_prob`). Expert order is descending by logit, ties broken
// by the lower id, so the host and the reference cannot disagree about which expert is which.
#pragma once

#include "model.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace onebit::npu {

struct MoeDims {
    int experts = 0;   // routed experts per layer
    int used = 0;      // top-k the router keeps
    bool norm_topk = true;
};

// One expert's weights, as the container holds them: `tiles` consecutive chunks.
struct ExpertChunks {
    // Points at this expert's first chunk inside the mapped container. The metadata's offsets are
    // measured from the data section (like data_offsets), so expert 0's offset is the tensor's
    // start and Moe subtracts it; callers see a plain pointer.
    const uint8_t* data = nullptr;
    uint64_t bytes = 0;    // == tiles * 5120 for a q4_1 stack
    int rows = 0, cols = 0, tiles = 0;
};

// model.layers.<layer>.mlp.experts.<proj>.weight — the three routed projections.
extern const char* const kExpertGate;
extern const char* const kExpertUp;
extern const char* const kExpertDown;

class Moe {
public:
    // Throws when the directory has no __metadata__.moe, or when that metadata and the tensor
    // shapes disagree (an expert count that does not divide the stack, an offset outside its
    // tensor, a byte count that is not whole chunks, a rows/cols/tiles geometry the tensor's
    // own shape contradicts). A kernel must never be handed a bad list.
    explicit Moe(const Model& model);

    const MoeDims& dims() const { return dims_; }
    // model.layers.<layer>.mlp.gate.weight, the bf16 router.
    std::string router_name(int layer) const { return "model.layers." + std::to_string(layer) + ".mlp.gate.weight"; }

    // Expert e of a routed projection. `suffix` is one of kExpertGate/kExpertUp/kExpertDown.
    // Throws on an expert id out of range.
    ExpertChunks expert(int layer, const std::string& suffix, int e) const;

    // h [hidden] -> k expert ids and their renormalised weights, descending by logit.
    // `layer` only selects the router tensor. Throws if the stack geometry is inconsistent,
    // if ids/weights are null, or if k is not in [1, used].
    void router_topk(const float* h, int layer, int k, int* ids, float* weights) const;

    // The router's raw logits and softmax probabilities, for tests and for a caller that wants
    // the full distribution. p and logits are [experts].
    void router_probs(const float* h, int layer, float* probs, float* logits_out = nullptr) const;

private:
    // One routed stack, validated against its tensor at construction: nothing is re-parsed or
    // re-checked per dispatch.
    struct Stack {
        const Tensor* tensor = nullptr;
        int experts = 0, rows = 0, cols = 0, tiles = 0;
        uint64_t bytes = 0;
        std::vector<uint64_t> offsets;
    };

    const Model& model_;
    MoeDims dims_;
    std::map<std::string, Stack> stacks_;
};

}  // namespace onebit::npu
