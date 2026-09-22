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
#include "model_config.h"

#include <format>

namespace onebit {

std::expected<ModelConfig, std::string> read_model_config(const GgufFile& f) {
    ModelConfig c;
    auto arch = f.get_string("general.architecture");
    if (!arch) return std::unexpected(std::string("missing general.architecture"));
    c.arch = *arch;

    // Only architectures that have passed the golden test are accepted.
    if (c.arch == "qwen3") {
        c.rope_style = RopeStyle::Neox;
    } else {
        return std::unexpected(std::format("architecture '{}' is not verified on the CPU reference", c.arch));
    }

    auto u32 = [&](const std::string& suffix) -> std::expected<uint32_t, std::string> {
        auto v = f.get_uint(c.arch + "." + suffix);
        if (!v) return std::unexpected(std::format("missing {}.{}", c.arch, suffix));
        if (*v == 0 || *v > UINT32_MAX) return std::unexpected(std::format("{}.{} out of range", c.arch, suffix));
        return static_cast<uint32_t>(*v);
    };
    auto set = [&](uint32_t& field, const std::string& suffix) -> std::expected<void, std::string> {
        auto v = u32(suffix);
        if (!v) return std::unexpected(v.error());
        field = *v;
        return {};
    };

    for (auto [field, key] : {std::pair{&c.n_layer, "block_count"}, {&c.n_embd, "embedding_length"},
                              {&c.n_head, "attention.head_count"}, {&c.n_ff, "feed_forward_length"},
                              {&c.n_ctx_train, "context_length"}}) {
        if (auto r = set(*field, key); !r) return std::unexpected(r.error());
    }
    c.n_head_kv = u32("attention.head_count_kv").value_or(c.n_head);
    c.head_dim = u32("attention.key_length").value_or(c.n_embd / c.n_head);
    if (auto vl = f.get_uint(c.arch + ".attention.value_length"); vl && *vl != c.head_dim)
        return std::unexpected(std::string("key_length != value_length is not supported"));
    if (c.n_head % c.n_head_kv != 0) return std::unexpected(std::string("head_count is not a multiple of head_count_kv"));

    if (auto eps = f.get_float(c.arch + ".attention.layer_norm_rms_epsilon")) c.rms_eps = static_cast<float>(*eps);
    if (auto theta = f.get_float(c.arch + ".rope.freq_base")) c.rope_theta = static_cast<float>(*theta);

    const GgufTensor* emb = f.tensor("token_embd.weight");
    if (!emb || emb->ne.size() != 2 || emb->ne[0] != c.n_embd)
        return std::unexpected(std::string("token_embd.weight missing or wrong shape"));
    c.n_vocab = static_cast<uint32_t>(emb->ne[1]);
    return c;
}

}  // namespace onebit
