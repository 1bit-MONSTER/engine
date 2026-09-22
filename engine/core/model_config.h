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
// model_config.h: decoder hyper-parameters read from GGUF metadata.
#pragma once

#include <cstdint>
#include <expected>
#include <string>

#include "gguf.h"

namespace onebit {

enum class RopeStyle {
    Normal,  // rotate adjacent pairs (x[2i], x[2i+1]); llama GGUFs are permuted for this
    Neox,    // rotate halves (x[i], x[i + d/2]); qwen2/qwen3
};

struct ModelConfig {
    std::string arch;  // general.architecture, e.g. "qwen3"
    uint32_t n_layer = 0;
    uint32_t n_embd = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t n_ff = 0;
    uint32_t n_vocab = 0;
    uint32_t n_ctx_train = 0;
    float rms_eps = 1e-6f;
    float rope_theta = 10000.0f;
    RopeStyle rope_style = RopeStyle::Normal;
};

// Reads and validates the config. Fails for architectures the engine has not
// been verified on, rather than guessing their semantics.
std::expected<ModelConfig, std::string> read_model_config(const GgufFile& f);

}  // namespace onebit
