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

// laya/scorer.h — the Laya router (PORTING.md step 4): a non-autoregressive
// System 1 decision scorer that picks where each request runs.
//
// Ports NandhaKishorM/laya (ModernBERT-large encoder + RLCD decision head) with
// zero Python at runtime: weights from model.safetensors (laya/safetensors.h),
// tokens from the Hugging Face tokenizer.json (npu::Tokenizer). One forward
// pass answers typed questions (choice / score / noul) with calibrated
// confidence. Source: 1bit-MONSTER src/laya_scorer.cpp on
// backup/laya-and-results-2026-09-22.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace onebit::laya {

// One typed question.
struct Question {
    std::string type;  // "choice" | "score" | "noul"
    std::string instructions;
    // choice: option-key -> description (label order = argmax index order)
    // score:  ordered level descriptions (index i = level i)
    std::vector<std::pair<std::string, std::string>> criteria;  // (key, desc)
};

// One answer (per question, same pass).
struct Answer {
    std::string type;
    std::string choice;                               // choice: winning key
    std::vector<std::pair<std::string, float>> probabilities;  // label -> p
    float score = 0.0f;                               // score: expected level
    float noul = 0.0f;                                // noul: p(true)
    float confidence = 0.0f;                          // 1 - H(p)/log(k)
    float act_probability = 0.0f;                     // act_head "escalate" p
};

// Raw pre-softmax outputs, for gating against the Python reference.
struct RawOutput {
    std::vector<std::vector<float>> logits;      // per question, per marker (masked = -1e4)
    std::vector<std::vector<float>> act_logits;  // per question, 2 entries
};

class Scorer {
public:
    // Loads model_dir holding model.safetensors, rl_agent_config.json and
    // tokenizer/tokenizer.json (the pinned Hugging Face checkpoint layout).
    bool load(const std::string& model_dir);

    // Evaluates every question against `state` in one forward pass. `state` is
    // free text or a JSON-serializable value already stringified, taken as
    // already NFC-normalized (as npu::Tokenizer does). When `raw` is non-null
    // the pre-softmax logits and act logits are also returned for gating.
    bool score(const std::string& state, const std::vector<Question>& questions,
               std::vector<Answer>& answers, RawOutput* raw = nullptr);

    const std::string& error() const { return err_; }

private:
    // config
    int max_len_ = 512;
    int head_max_len_ = 192;
    int nhead_ = 16;
    int head_dim_ = 64;
    int d_ = 1024;
    int n_layer_ = 28;
    // tokenizer specials
    int cls_id_ = -1, sep_id_ = -1, mask_id_ = -1, pad_id_ = -1;

    // weights (f32, owned)
    std::vector<float> tok_emb_;  // [50368,1024]
    std::vector<float> emb_norm_;  // [1024]
    std::vector<float> final_norm_;  // [1024]
    struct LayerW {
        std::vector<float> Wo, Wqkv, attn_norm, Wi, Wo_mlp, mlp_norm;
    };
    std::vector<LayerW> layers_;
    // head
    std::vector<float> type_emb_;  // [3,1024]
    struct HeadLayerW {
        std::vector<float> n1w, n1b, n2w, n2b, in_w, in_b, out_w, out_b, l1w, l1b, l2w, l2b;
    };
    std::vector<HeadLayerW> head_layers_;
    std::vector<float> sc0w, sc0b, sc1w, sc1b, sc3w, sc3b;  // scorer
    std::vector<float> act0w, act0b, act2w, act2b;          // act_head
    std::vector<float> temperature_;                        // [3]
    std::vector<std::pair<std::string, float>> temperature_by_options_;

    std::string tok_path_;
    std::string err_;
};

}  // namespace onebit::laya
