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

// dflash_forward.h -- the DFlash draft's forward as a host reference in f32 (engine#1780, item 2 of "supported").
//
// What a draft computes, from the interface its package states (dflash_draft.h loads it):
//
//   context   c[t] = hidden_norm(fc(taps[t]))             one row per context position; taps[t] is the target
//                                                         model's hidden states at target_layer_ids, concatenated
//   block     x[i] = noise[i]                             the target's embeddings of [last token, mask, mask, ...]
//   per layer h    = input_norm(x)
//             q    = q_norm(q_proj(h))                    the block rows only
//             k, v = k_norm(k_proj([c ; h])), v_proj([c ; h])   context rows first, then the block rows
//             x   += o_proj(attention(rope(q), rope(k), v))     every block row sees every key: not causal
//             x   += down(silu(gate(p)) * up(p)),  p = post_norm(x)
//   output    norm(x)                                     [block][hidden]; the target's own head scores it
//
// The context rows are NOT passed through a layer's input_norm and are the same rows in every layer; only the block
// rows change from layer to layer. Positions are consecutive: context position t is pos0 + t and block row i is
// pos0 + ctx + i. The rotation is relative, so the output does not depend on pos0 (the test holds it to that).
//
// This is the reference the device route will be compared with, not a fast path: plain loops, one thread, f32 sums
// accumulated in double. It has not yet been run against a target model's own hidden states (that needs the target
// pack's layer taps, the next step), so what the test establishes is the structure above and a hand-computed case,
// not agreement with a published draft's outputs.

#pragma once

#include <vector>

#include "dflash_draft.h"

namespace onebit::npu::dflash {

struct DraftLayerWeights {
    std::vector<float> q, k, v, o, gate, up, down;                 // row-major [out][in]
    std::vector<float> input_norm, post_norm, q_norm, k_norm;      // gains
};

// A draft decoded to f32. For the Bolt draft this is about 1.3 GB; the caller decides how long to keep it.
struct DraftWeights {
    int hidden = 0, intermediate = 0, heads = 0, kv_heads = 0, head_dim = 0, taps = 0;
    float rms_eps = 0.0f, rope_theta = 0.0f;
    std::vector<float> fc;            // [hidden][taps * hidden]
    std::vector<float> hidden_norm;   // [hidden]
    std::vector<float> norm;          // [hidden]
    std::vector<DraftLayerWeights> layers;
};

// Decode every tensor of a loaded package. Throws std::runtime_error if a decoded tensor has the wrong size.
DraftWeights decode_draft(const DraftModel& m);

// Throws std::runtime_error naming the first thing wrong with a DraftWeights' geometry or sizes.
void check_weights(const DraftWeights& w);

// One draft pass.
//   taps   [ctx][taps * hidden], row t = the target's hidden states at position pos0 + t, in target_layer_ids order
//   noise  [block][hidden]
//   window 0 = every key is visible; otherwise a block row sees a key only when their positions differ by less
//          than `window` (both directions, as the draft is not causal)
// Returns [block][hidden]. ctx may be 0 (the block then attends only to itself). Throws std::runtime_error on a bad
// argument or a non-finite result.
std::vector<float> draft_forward(const DraftWeights& w, const float* taps, int ctx, const float* noise, int block,
                                 long pos0 = 0, int window = 0);

}  // namespace onebit::npu::dflash
