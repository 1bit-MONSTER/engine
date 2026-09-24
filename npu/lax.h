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

// The Qwen3.6-35B-A3B decode on the lax kernels (docs/npu-lax.md), in C++: every weight
// packed at load time from model.q4nx straight into the XRT buffers (npu/lax_pack.h), the
// 40 layers of a token as ONE runlist submit, then the final norm and the lm head.
//
// Per token: the residual `xres` (f32 [2048]) is the token's embedding; the full-attention
// stream is patched for the position; the runlist runs the 40 layers (each layer's
// arguments: pool xres consts kv|dkv act ptab state|dstate cfg; a linear layer gets the
// dummy kv, a full one the dummy state); `ln` normalizes xres into hn and `lm` writes
// f32 logits over the padded 248320 rows.
#pragma once

#include "lax_pack.h"
#include "model.h"

#include <memory>
#include <string>
#include <vector>

namespace onebit::npu::lax {

struct LoadStats {
    double buffers_ms = 0, pack_ms = 0, total_ms = 0;
    size_t packed_bytes = 0;
};

class Decoder {
public:
    // kernel_dir as classic_kernels() takes it (npu/lax_kernels.h).
    Decoder(const Model& model, const Config& config, const std::string& kernel_dir, int threads = 0);
    ~Decoder();
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    const LoadStats& load_stats() const;
    std::string kernels() const;

    // The residual for the next run: a token's embedding, or given f32 values.
    void feed(int token);
    void set_residual(const float* xres);
    // The 40 layers at cache position pos; with head, also the norm and the lm head.
    void run(size_t pos, bool head);
    // The last head's logits: argmax over the first n (first maximum wins), or all rows.
    int argmax(int n);
    void logits(std::vector<float>& out);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace onebit::npu::lax
