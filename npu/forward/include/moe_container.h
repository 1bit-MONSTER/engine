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
// The two MoE container conventions the forward reads (engine#1035).
//
// tools/gguf_to_q4nx.py -- the engine's own GGUF converter -- writes a routed expert stack as ONE
// tensor `model.layers.N.mlp.experts.{gate,up,down}_proj.weight` whose per-expert slice is a plain
// [N][K] tile grid (N = the projection's output rows, K its input columns), names the router
// `model.layers.N.mlp.gate.weight`, and writes HF-shaped config keys: `num_experts`,
// `num_experts_per_tok`, `moe_intermediate_size`.
//
// FastFlowLM-style containers, which this forward was originally written against, use
// `mlp.{gate,up,down}_exps_proj.weight` and `moe_router.weight`, their writer transposes each
// expert's tiles so a slice decodes on a [K][N] grid, and their config says `n_expert` / `top_k`.
//
// Getting either half wrong is silent. The wrong names leave the tensor absent (a zero descriptor,
// which is what engine#1035's `MLP dequant : data_size 0` was); the wrong grid decodes a permutation
// of the expert -- measured on Qwen3-Coder-30B expert 0, the converter's slice correlates 0.99999
// with the GGUF on the [N][K] grid and 0.00855 on [K][N]. So both are decided from the container's
// own tensors and config, not from a model name.
#pragma once

// Global scope like the other npu/forward headers this one is included beside (model.h,
// q4nx_pack.h, q4nx_semantics.h): Q4nxNpuForward is a global-scope class.
enum class MoeFormat { Flm, EngineConverter };

// The per-layer tensor suffixes of a convention ("model.layers.%d." + these).
struct MoeNames {
    const char* gate;
    const char* up;
    const char* down;
    const char* router;
    const char* router_b;   // optional: Qwen MoE has none, some MoE architectures do
    const char* share_gate;
    const char* share_up;
    const char* share_down;
    const char* share_router;
    bool slice_nk;   // true: a per-expert slice decodes on an [N][K] grid
};

inline MoeNames moe_names(MoeFormat f) {
    if (f == MoeFormat::EngineConverter)
        return {"mlp.experts.gate_proj.weight",   "mlp.experts.up_proj.weight",
                "mlp.experts.down_proj.weight",   "mlp.gate.weight", "mlp.gate.bias",
                "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
                "mlp.shared_expert.down_proj.weight", "mlp.shared_expert_gate.weight",
                true};
    return {"mlp.gate_exps_proj.weight",   "mlp.up_exps_proj.weight",
            "mlp.down_exps_proj.weight",   "moe_router.weight", "moe_router.bias",
            "mlp.share_gate_exps_proj.weight", "mlp.share_up_exps_proj.weight",
            "mlp.share_down_exps_proj.weight", "mlp.share_router.weight",
            false};
}

// Which convention a container uses, from layer 0's routed gate tensor. Both spellings exist on
// disk; a container with neither is the FLM shape the forward assumed before engine#1035.
inline MoeFormat moe_format(bool has_flm_gate, bool has_engine_gate) {
    return (!has_flm_gate && has_engine_gate) ? MoeFormat::EngineConverter : MoeFormat::Flm;
}

// config.json's expert count / top-k: the llama.cpp-style keys first, then the HF names the engine's
// converter writes. Zero from both means "not a MoE container", which is the old behaviour.
inline int moe_expert_count(int n_expert, int num_experts) {
    return n_expert > 0 ? n_expert : (num_experts > 0 ? num_experts : 0);
}
inline int moe_top_k(int top_k, int num_experts_per_tok) {
    return top_k > 0 ? top_k : (num_experts_per_tok > 0 ? num_experts_per_tok : 0);
}

// The geometry a per-expert slice is dequantized at: [N][K] for the converter's plain grid,
// [K][N] for the FLM-style transposed one.
inline void moe_slice_geometry(MoeFormat f, int N, int K, int* rows, int* cols) {
    if (f == MoeFormat::EngineConverter) {
        *rows = N;
        *cols = K;
    } else {
        *rows = K;
        *cols = N;
    }
}
