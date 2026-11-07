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
#include <functional>
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

// One activation row set, quantised the way the reference does: int8 codes plus the power-of-two scale that
// produced them, per token and (when the width calls for it) per 256-channel block. The host path dequantises
// and multiplies; a device path sends the codes as they are and applies the scales to its int32 sums, which is
// why the scales travel with the codes instead of being folded into the floats.
struct Quantised {
    int width = 0, block = 0;          // block = 256 when the width is a multiple of it and wider, else width
    std::vector<int8_t> q;             // [rows][width]
    std::vector<float> scale;          // [rows][width/block]
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

    // Where the ternary linears run. The host dot below is the default; the NPU path (engine#1311 step 2)
    // installs a hook that runs the same int8 codes on the array's GEMM. It returns false and fills `why` on
    // failure, and must produce `y[r][n] = scale_n * sum_k q[r][k]*code[n][k] + bias_n` for the Quantised
    // activation it is handed.
    using Gemm = std::function<bool(const struct Tern&, const Quantised&, std::vector<float>& x, int rows,
                                    std::string&)>;
    void set_gemm(Gemm g) { gemm_ = std::move(g); }

    // Read-only views for a device backend that packs the same codes into its own weight buffers.
    const std::vector<Layer>& layers() const { return layers_; }
    const std::vector<int8_t>& word_q() const { return word_q_; }
    const std::vector<float>& word_s() const { return word_s_; }
    const std::vector<int8_t>& pos_q() const { return pos_q_; }
    const std::vector<float>& pos_s() const { return pos_s_; }
    const std::vector<float>& tok_type() const { return tok_type_; }
    const std::vector<float>& ln_e_w() const { return ln_e_w_; }
    const std::vector<float>& ln_e_b() const { return ln_e_b_; }
    const std::vector<float>& out_w() const { return out_w_; }

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
    Gemm gemm_;
};

}  // namespace onebit::embed
