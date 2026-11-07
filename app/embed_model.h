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

// Sentence embeddings from a 1-bit encoder container (engine#1311, first model Cyronius/trit-embed-3l).
//
// The container (`tools/trit_embed_pack.py`) holds what the reference model's own code reads, widened to what
// the hardware executes:
//   * ternary linears: 2-bit codes {-1,0,1} widened to int8, one f32 scale per output row -> y = scale * (a @ codes^T) + bias
//   * word/position embeddings: int8 rows, one f32 scale per row
//   * LayerNorms, biases, the token-type embedding and the output projection: f32
// and the activation path is the reference's int8 emulation, because that is what makes the vectors comparable
// at all: quantise to int8 with a power-of-two scale per token (per 256-channel block when the width is a
// multiple of 256 and wider), round half to even, clamp to +-127, dequantise -- exactly `trit_embed.py`'s
// `quant_act`, applied before every ternary linear.
//
// The forward is a post-norm BERT encoder: LN(embedding sum) -> per layer {attention over q/k/v, then LN1 of the
// residual; GELU(tanh) MLP, then LN2 of the residual} -> masked mean pool -> 384x384 output projection -> L2.
// The device path (engine#1311 step 2) replaces the per-row int8 dot with the NPU's own int8 GEMM; the math
// above is what it has to reproduce, and `tests/embed_model_test.cpp` is the device-free row that says it does.
#pragma once

#include "wordpiece.h"

#include <cstdint>
#include <string>
#include <vector>

namespace onebit::embed {

struct Config {
    int vocab = 0, d = 0, heads = 0, ffn = 0, layers = 0, max_pos = 0, out_dim = 0;
    float eps = 1e-12f;
    std::string pooling = "mean";
    bool normalized = true;
    int max_tokens = 0;
};

// One ternary linear: int8 codes [N, K] and one f32 scale per output row, plus an f32 bias [N].
struct Tern {
    int n = 0, k = 0;
    std::vector<int8_t> code;
    std::vector<float> scale, bias;
};

struct Layer {
    Tern q, k, v, o, fc1, fc2;
    std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b;
};

class Model {
public:
    // Reads <dir>/model.embed and <dir>/vocab.txt. Returns false and fills `why` on any refusal.
    bool load(const std::string& dir, std::string& why);

    const Config& config() const { return cfg_; }
    int vocab_size() const { return tok_.vocab_size(); }

    // Token ids for one sentence: [CLS] ... [SEP], truncated to the container's max_tokens.
    std::vector<int> tokenize(const std::string& text) const;

    // Unit-norm embedding of one already-tokenised sentence (the reference's own contract: no padding, one
    // sentence per forward, mean over the tokens that exist).
    std::vector<float> encode_ids(const std::vector<int>& ids, std::string& why) const;

    // Convenience: tokenise + encode. The reference truncates to max_tokens, so does this.
    std::vector<float> encode(const std::string& text, std::string& why) const {
        return encode_ids(tokenize(text), why);
    }

private:
    Config cfg_;
    int tok_vocab_ = 0;
    std::vector<int8_t> word_q_, pos_q_;      // [vocab, d], [max_pos, d]
    std::vector<float> word_s_, pos_s_;       // [vocab], [max_pos]
    std::vector<float> tok_type_;             // [2, d], row 0 is used
    std::vector<float> ln_e_w_, ln_e_b_, out_w_;  // out_w_ is [out_dim, d]
    std::vector<Layer> layers_;
    wordpiece::BertTokenizer tok_;
};

}  // namespace onebit::embed
