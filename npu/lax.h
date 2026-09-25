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

#include "lax_kernels.h"
#include "lax_pack.h"
#include "model.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace onebit::npu::lax {

struct LoadStats {
    double buffers_ms = 0, pack_ms = 0, total_ms = 0;
    size_t packed_bytes = 0;
};

// The last run(): host preparation before the submit (position, runlist), the 40-layer
// runlist (submit to completion), the norm and the lm head; ahead_ms is the part of the
// runlist's time the host spent preparing the next position (full ELFs).
struct RunTimes {
    double prep_ms = 0, layers_ms = 0, head_ms = 0, ahead_ms = 0;
};

class Decoder {
public:
    // kernel_dir as scripts/build-lax.sh's <prefix>/kernels; t picks the kernel set
    // (npu/lax_kernels.h): full ELFs, or the classic xclbin path for A/B.
    Decoder(const Model& model, const Config& config, const std::string& kernel_dir, Transport t = Transport::Elf,
            int threads = 0);
    ~Decoder();
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    const LoadStats& load_stats() const;
    const RunTimes& last_run() const;
    std::string kernels() const;

    // The residual for the next run: a token's embedding, or given f32 values.
    void feed(int token);
    void set_residual(const float* xres);
    // The 40 layers at cache position pos; with head, also the norm and the lm head.
    void run(size_t pos, bool head);
    // Start a new sequence at position 0: zero the DeltaNet state (the KV cache needs no
    // clearing). Construction leaves the decoder in this state.
    void reset();
    // The last head's logits: argmax over the first n (first maximum wins), or all rows.
    int argmax(int n);
    void logits(std::vector<float>& out);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

struct GenerateOptions {
    int max_tokens = 256;
    // As npu/generate.h: divides positive (multiplies negative) logits of the distinct ids
    // among the last penalty_window of prompt + output; 1 is plain greedy.
    float repetition_penalty = 1.0f;
    int penalty_window = 64;
    std::vector<int> stop;                   // end-of-turn ids; the one generated is returned
    std::function<bool(int token)> on_token;  // each generated token; false stops
};

struct GenerateResult {
    std::vector<int> tokens;
    double prefill_ms = 0, decode_ms = 0;
    bool stopped_at_eos = false;
    size_t reused = 0;  // prompt tokens already in the cache
};

// One conversation on the decoder: what the cache holds. A prompt that extends it
// continues from there; any other prompt starts over (Decoder::reset), because the
// DeltaNet state cannot rewind. Greedy (first maximum over the tokenizer's vocab).
class Session {
public:
    Session(Decoder& dec, int vocab) : dec_(dec), vocab_(vocab) {}
    GenerateResult generate(const std::vector<int>& prompt, const GenerateOptions& opt);
    size_t cached() const { return cache_.size(); }

private:
    int pick(const GenerateOptions& opt, const std::vector<int>& history);
    Decoder& dec_;
    int vocab_;
    std::vector<int> cache_;  // the tokens at positions 0..n-1
    std::vector<float> buf_;
};

}  // namespace onebit::npu::lax
