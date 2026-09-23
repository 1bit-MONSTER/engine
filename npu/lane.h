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

// The NPU fast lane (docs/npu.md): one whole-layer kernel run per decoder layer
// plus the lm head, submitted as one XRT runlist per token, from full ELFs
// generated at load time. No xclbin.
//
// A kernel directory holds the layer kernel's instruction ELFs for context
// lengths 1, 2 and 17 (layer_ctx<N>.elf), the lm head's (lmhead.elf) and the
// design PDI (layer.pdi).
//
// Two runlist slots let the host prepare the next token's runlist while the
// current one executes: prepare(slot, ctx) is host work (RoPE rows, runlist
// build), launch(slot, token) writes the token's embedding and submits.
#pragma once

#include "model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace onebit::npu {

class Lane {
public:
    Lane(const Model& model, const std::string& kernel_dir);
    ~Lane();
    Lane(const Lane&) = delete;
    Lane& operator=(const Lane&) = delete;

    // Longest context the layer kernel's KV layout holds.
    static constexpr int kMaxContext = 8192;

    // ctx is the context length including this token (1-based).
    void prepare(int slot, int ctx);
    void launch(int slot, int token);
    void wait(int slot);
    // prepare + launch + wait on slot 0.
    void step(int token, int ctx);

    // The last completed run's logits.
    int argmax();
    void logits(std::vector<float>& out);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace onebit::npu
