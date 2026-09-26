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

// laya/scorer.cpp — non-autoregressive System 1 decision scorer (Laya router).
//
// Port of NandhaKishorM/laya (ModernBERT-large encoder + RLCD decision head).
// Zero Python at runtime: weights via Safetensors (laya/safetensors.h), tokens
// via the engine's tokenizer.json reader (npu::Tokenizer). One forward pass
// emits typed decisions (choice / score / noul). Source: 1bit-MONSTER
// src/laya_scorer.cpp on backup/laya-and-results-2026-09-22.
#include "scorer.h"

#include "safetensors.h"
#include "tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace onebit::laya {

namespace {

constexpr int NHEAD = 16, HD = 64, D = 1024, NLAYER = 28;
constexpr float kInvSqrt2 = 0.70710678118654752440f;

float gelu(float x) { return 0.5f * x * (1.0f + erff(x * kInvSqrt2)); }

// LayerNorm over the last dim; bias may be null.
void layernorm(const float* in, const float* w, const float* b, int rows, int cols, float* out) {
    const float eps = 1e-5f;
    for (int r = 0; r < rows; ++r) {
        const float* x = in + (size_t)r * cols;
        float mean = 0, var = 0;
        for (int c = 0; c < cols; ++c) mean += x[c];
        mean /= cols;
        for (int c = 0; c < cols; ++c) { float d = x[c] - mean; var += d * d; }
        var /= cols;
        float inv = 1.0f / sqrtf(var + eps);
        float* o = out + (size_t)r * cols;
        for (int c = 0; c < cols; ++c) o[c] = (x[c] - mean) * inv * w[c] + (b ? b[c] : 0.0f);
    }
}

int worker_threads() {
    static const int n = int(std::clamp(std::thread::hardware_concurrency() / 2, 1u, 16u));
    return n;
}

// Runs fn(i0, i1) over [0, n) split into t contiguous ranges, one thread each.
template <class F>
void parallel_ranges(int n, int t, F fn) {
    t = std::max(1, std::min(t, n));
    if (t == 1) return fn(0, n);
    std::vector<std::thread> pool;
    for (int p = 0; p < t; ++p) pool.emplace_back(fn, int((size_t)n * p / t), int((size_t)n * (p + 1) / t));
    for (auto& th : pool) th.join();
}

// out[m,k] = in[m,n] @ W^T[k,n] + bias[k]. The output columns are split across threads; every
// sum keeps its order, so the result is the same as one thread's, bit for bit.
void linear(const float* in, const float* W, const float* bias, int m, int n, int k, float* out) {
    // Each weight row is read once per four input rows; the four sums are independent, so the
    // loads are shared without reordering any sum.
    auto cols = [&](int j0, int j1) {
        for (int j = j0; j < j1; ++j) {
            const float* wr = W + (size_t)j * n;
            const float b = bias ? bias[j] : 0.0f;
            int i = 0;
            for (; i + 4 <= m; i += 4) {
                const float *x0 = in + (size_t)i * n, *x1 = x0 + n, *x2 = x1 + n, *x3 = x2 + n;
                float a0 = b, a1 = b, a2 = b, a3 = b;
                for (int c = 0; c < n; ++c) {
                    const float w = wr[c];
                    a0 += x0[c] * w; a1 += x1[c] * w; a2 += x2[c] * w; a3 += x3[c] * w;
                }
                out[(size_t)i * k + j] = a0; out[(size_t)(i + 1) * k + j] = a1;
                out[(size_t)(i + 2) * k + j] = a2; out[(size_t)(i + 3) * k + j] = a3;
            }
            for (; i < m; ++i) {
                const float* x = in + (size_t)i * n;
                float acc = b;
                for (int c = 0; c < n; ++c) acc += x[c] * wr[c];
                out[(size_t)i * k + j] = acc;
            }
        }
    };
    const int t = std::min(worker_threads(), std::max(1, int((size_t)m * n * k / (1u << 20))));   // small products stay on one thread
    parallel_ranges(k, t, cols);
}

// Multi-head self attention (batch_first, bidirectional). key_pad[b*L+j] = ignore key j.
void mha(const float* x, const float* inW, const float* inB, const float* outW, const float* outB,
         int N, int L, const int8_t* key_pad, float* out) {
    std::vector<float> qkv((size_t)N * L * 3 * D);
    linear(x, inW, inB, N * L, D, 3 * D, qkv.data());
    std::vector<float> attn_out((size_t)N * L * D);
    const float scale = 1.0f / sqrtf((float)HD);
    std::vector<float> scores((size_t)L * L);
    for (int b = 0; b < N; ++b) {
        const float* qkv_b = qkv.data() + (size_t)b * L * 3 * D;
        for (int h = 0; h < NHEAD; ++h) {
            for (int i = 0; i < L; ++i) {
                const float* qrow = qkv_b + (size_t)i * 3 * D + (size_t)h * HD;
                float* sr = scores.data() + (size_t)i * L;
                for (int j = 0; j < L; ++j) {
                    if (key_pad[(size_t)b * L + j]) { sr[j] = -INFINITY; continue; }
                    const float* krow = qkv_b + (size_t)j * 3 * D + D + (size_t)h * HD;
                    float acc = 0;
                    for (int d = 0; d < HD; ++d) acc += qrow[d] * krow[d];
                    sr[j] = acc * scale;
                }
            }
            for (int i = 0; i < L; ++i) {
                float* sr = scores.data() + (size_t)i * L;
                float mx = -INFINITY;
                for (int j = 0; j < L; ++j) mx = fmaxf(mx, sr[j]);
                float sum = 0;
                for (int j = 0; j < L; ++j) { sr[j] = expf(sr[j] - mx); sum += sr[j]; }
                for (int j = 0; j < L; ++j) sr[j] /= sum;
            }
            for (int i = 0; i < L; ++i) {
                float* orow = attn_out.data() + (size_t)b * L * D + (size_t)i * D + (size_t)h * HD;
                for (int d = 0; d < HD; ++d) orow[d] = 0;
                for (int j = 0; j < L; ++j) {
                    float s = scores[(size_t)i * L + j];
                    const float* vrow = qkv_b + (size_t)j * 3 * D + 2 * D + (size_t)h * HD;
                    for (int d = 0; d < HD; ++d) orow[d] += s * vrow[d];
                }
            }
        }
    }
    linear(attn_out.data(), outW, outB, N * L, D, D, out);
}

// GeGLU MLP for the ModernBERT encoder: Wi [5248,1024] -> gelu(up)*gate -> Wo [1024,2624].
void geglu(const float* h, const float* Wi, const float* Wo, int rows, float* out) {
    const int UP = 5248, MID = 2624;
    std::vector<float> up((size_t)rows * UP), act((size_t)rows * MID);
    linear(h, Wi, nullptr, rows, D, UP, up.data());
    for (int r = 0; r < rows; ++r) {
        const float* u = up.data() + (size_t)r * UP;
        float* a = act.data() + (size_t)r * MID;
        for (int j = 0; j < MID; ++j) a[j] = gelu(u[j]) * u[MID + j];
    }
    linear(act.data(), Wo, nullptr, rows, MID, D, out);
}

}  // namespace

bool Scorer::load(const std::string& model_dir) {
    err_.clear();
    Safetensors r;
    const std::string st_path = model_dir + "/model.safetensors";
    if (!r.open(st_path)) { err_ = "open safetensors: " + r.error(); return false; }
    auto load_t = [&](const std::string& name, std::vector<float>& dst) -> bool {
        if (!r.get_f32(name, dst)) { err_ = "missing tensor " + name; return false; }
        return true;
    };
    // encoder embeddings + norm
    if (!load_t("encoder.embeddings.tok_embeddings.weight", tok_emb_)) return false;
    if (!load_t("encoder.embeddings.norm.weight", emb_norm_)) return false;
    if (!load_t("encoder.final_norm.weight", final_norm_)) return false;
    layers_.resize(NLAYER);
    for (int i = 0; i < NLAYER; ++i) {
        char buf[128];
        LayerW& l = layers_[i];
        snprintf(buf, sizeof buf, "encoder.layers.%d.attn.Wo.weight", i);
        if (!load_t(buf, l.Wo)) return false;
        snprintf(buf, sizeof buf, "encoder.layers.%d.attn.Wqkv.weight", i);
        if (!load_t(buf, l.Wqkv)) return false;
        if (i > 0) {  // layer 0 has no attn_norm
            snprintf(buf, sizeof buf, "encoder.layers.%d.attn_norm.weight", i);
            if (!load_t(buf, l.attn_norm)) return false;
        }
        snprintf(buf, sizeof buf, "encoder.layers.%d.mlp.Wi.weight", i);
        if (!load_t(buf, l.Wi)) return false;
        snprintf(buf, sizeof buf, "encoder.layers.%d.mlp.Wo.weight", i);
        if (!load_t(buf, l.Wo_mlp)) return false;
        snprintf(buf, sizeof buf, "encoder.layers.%d.mlp_norm.weight", i);
        if (!load_t(buf, l.mlp_norm)) return false;
    }
    // head
    if (!load_t("type_emb.weight", type_emb_)) return false;
    head_layers_.resize(2);
    for (int i = 0; i < 2; ++i) {
        char buf[256]; HeadLayerW& h = head_layers_[i];
        auto L = [&](const char* fmt, std::vector<float>& dst) -> bool {
            char b[256]; snprintf(b, sizeof b, fmt, i);
            return load_t(b, dst);
        };
        if (!L("head.layers.%d.norm1.weight", h.n1w)) return false;
        if (!L("head.layers.%d.norm1.bias", h.n1b)) return false;
        if (!L("head.layers.%d.norm2.weight", h.n2w)) return false;
        if (!L("head.layers.%d.norm2.bias", h.n2b)) return false;
        if (!L("head.layers.%d.self_attn.in_proj_weight", h.in_w)) return false;
        if (!L("head.layers.%d.self_attn.in_proj_bias", h.in_b)) return false;
        if (!L("head.layers.%d.self_attn.out_proj.weight", h.out_w)) return false;
        if (!L("head.layers.%d.self_attn.out_proj.bias", h.out_b)) return false;
        if (!L("head.layers.%d.linear1.weight", h.l1w)) return false;
        if (!L("head.layers.%d.linear1.bias", h.l1b)) return false;
        if (!L("head.layers.%d.linear2.weight", h.l2w)) return false;
        if (!L("head.layers.%d.linear2.bias", h.l2b)) return false;
    }
    // scorer
    if (!load_t("scorer.0.weight", sc0w)) return false;
    if (!load_t("scorer.0.bias", sc0b)) return false;
    if (!load_t("scorer.1.weight", sc1w)) return false;
    if (!load_t("scorer.1.bias", sc1b)) return false;
    if (!load_t("scorer.3.weight", sc3w)) return false;
    if (!load_t("scorer.3.bias", sc3b)) return false;
    // act_head
    if (!load_t("act_head.0.weight", act0w)) return false;
    if (!load_t("act_head.0.bias", act0b)) return false;
    if (!load_t("act_head.2.weight", act2w)) return false;
    if (!load_t("act_head.2.bias", act2b)) return false;
    if (!load_t("temperature", temperature_)) return false;

    // tokenizer (special tokens by name)
    tok_path_ = model_dir + "/tokenizer/tokenizer.json";
    try {
        const onebit::npu::Tokenizer tok(tok_path_);
        cls_id_ = tok.token_id("[CLS]");
        sep_id_ = tok.token_id("[SEP]");
        mask_id_ = tok.token_id("[MASK]");
        pad_id_ = tok.token_id("[PAD]");
    } catch (const std::exception& e) {
        err_ = std::string("load tokenizer: ") + e.what();
        return false;
    }
    if (cls_id_ < 0 || sep_id_ < 0 || mask_id_ < 0 || pad_id_ < 0) {
        err_ = "special tokens missing from tokenizer.json";
        return false;
    }

    // config: max_len, head_max_len, temperature from rl_agent_config.json
    std::ifstream cf(model_dir + "/rl_agent_config.json");
    if (cf) {
        const auto c = nlohmann::json::parse(cf);
        max_len_ = c.value("max_len", max_len_);
        head_max_len_ = c.value("head_max_len", head_max_len_);
        if (c.contains("temperature") && c["temperature"].is_array()) {
            temperature_.clear();
            for (const auto& t : c["temperature"]) temperature_.push_back(std::clamp(t.get<float>(), 0.5f, 5.0f));
        }
        if (c.contains("temperature_by_options") && c["temperature_by_options"].is_object()) {
            temperature_by_options_.clear();
            for (auto it = c["temperature_by_options"].begin(); it != c["temperature_by_options"].end(); ++it)
                temperature_by_options_.emplace_back(it.key(), std::clamp(it.value().get<float>(), 0.5f, 5.0f));
        }
    }
    if (temperature_.empty()) temperature_ = {1.0f, 1.0f, 1.0f};
    return true;
}

namespace {

// Tokenize `text` with the engine tokenizer (no special tokens added).
std::vector<int> encode(const onebit::npu::Tokenizer& tok, const std::string& text) {
    return tok.encode(text);
}

}  // namespace

bool Scorer::score(const std::string& state, const std::vector<Question>& questions,
                   std::vector<Answer>& answers, RawOutput* raw) {
    err_.clear();
    std::unique_ptr<onebit::npu::Tokenizer> tok;
    try {
        tok = std::make_unique<onebit::npu::Tokenizer>(tok_path_);
    } catch (const std::exception& e) {
        err_ = std::string("load tokenizer: ") + e.what();
        return false;
    }
    const int N = (int)questions.size();
    if (N == 0) { answers.clear(); return true; }

    // ---- build token sequences + markers (laya/common.py:build_sequence) ----
    std::vector<std::vector<int>> seqs(N), markers(N);
    std::vector<int> qtype(N);
    for (int i = 0; i < N; ++i) {
        const Question& q = questions[i];
        std::vector<std::string> opts;
        int qt;
        if (q.type == "choice") {
            qt = 0;
            for (auto& kv : q.criteria)
                opts.push_back(kv.second.empty() ? kv.first : kv.first + ": " + kv.second);
        } else if (q.type == "score") {
            qt = 1;
            for (size_t j = 0; j < q.criteria.size(); ++j)
                opts.push_back("level " + std::to_string(j) + ": " + q.criteria[j].second);
        } else {
            qt = 2;
            const std::string *fc = nullptr, *tc = nullptr;
            for (auto& kv : q.criteria) { if (kv.first == "false") fc = &kv.second; else if (kv.first == "true") tc = &kv.second; }
            opts.push_back("false: " + (fc && !fc->empty() ? *fc : "no, the statement does not hold"));
            opts.push_back("true: " + (tc && !tc->empty() ? *tc : "yes, the statement holds"));
        }
        qtype[i] = qt;

        std::vector<int> head_ids = encode(*tok, q.type + " question: " + q.instructions);
        std::vector<std::vector<int>> opt_ids(opts.size());
        for (size_t oi = 0; oi < opts.size(); ++oi) {
            std::vector<int> o = encode(*tok, " " + opts[oi]);
            if ((int)o.size() > 48) o.resize(48);
            opt_ids[oi].push_back(mask_id_);
            opt_ids[oi].insert(opt_ids[oi].end(), o.begin(), o.end());
        }
        int opt_budget = head_max_len_;
        for (auto& o : opt_ids) opt_budget -= (int)o.size();
        if (opt_budget < 16) {
            int per = std::max(4, (head_max_len_ - 16) / std::max(1, (int)opt_ids.size()));
            for (auto& o : opt_ids) o.resize(std::min((size_t)per, o.size()));
            opt_budget = head_max_len_;
            for (auto& o : opt_ids) opt_budget -= (int)o.size();
        }
        int hb = std::max(8, opt_budget);
        if ((int)head_ids.size() > hb) head_ids.resize(hb);

        std::vector<int> ids;
        ids.push_back(cls_id_);
        ids.insert(ids.end(), head_ids.begin(), head_ids.end());
        ids.push_back(sep_id_);
        for (auto& o : opt_ids) {
            markers[i].push_back((int)ids.size());
            ids.insert(ids.end(), o.begin(), o.end());
        }
        ids.push_back(sep_id_);
        int room = std::max(0, max_len_ - (int)ids.size() - 1);
        std::vector<int> st = encode(*tok, state);
        if ((int)st.size() > room) st.resize(room);  // truncate_left=false
        ids.insert(ids.end(), st.begin(), st.end());
        ids.push_back(sep_id_);
        if ((int)ids.size() > max_len_) ids.resize(max_len_);
        seqs[i] = std::move(ids);
        std::vector<int> m2;
        for (int mm : markers[i]) if (mm < max_len_) m2.push_back(mm);
        markers[i] = std::move(m2);
    }

    // ---- collate (pad to L, K) ----
    int L = 0, K = 0;
    for (int i = 0; i < N; ++i) { L = std::max(L, (int)seqs[i].size()); K = std::max(K, (int)markers[i].size()); }
    std::vector<int64_t> input_ids((size_t)N * L, pad_id_);
    std::vector<int64_t> attn((size_t)N * L, 0);
    std::vector<int64_t> mpos((size_t)N * K, 0);
    std::vector<int8_t> mmask((size_t)N * K, 0);
    for (int i = 0; i < N; ++i) {
        for (size_t j = 0; j < seqs[i].size(); ++j) { input_ids[(size_t)i * L + j] = seqs[i][j]; attn[(size_t)i * L + j] = 1; }
        for (size_t j = 0; j < markers[i].size(); ++j) { mpos[(size_t)i * K + j] = markers[i][j]; mmask[(size_t)i * K + j] = 1; }
    }

    // ---- ModernBERT encoder forward ----
    std::vector<float> h((size_t)N * L * D);
    for (int b = 0; b < N; ++b)
        for (int t = 0; t < L; ++t)
            memcpy(h.data() + ((size_t)b * L + t) * D, tok_emb_.data() + (size_t)input_ids[(size_t)b * L + t] * D, D * 4);
    {
        std::vector<float> tmp((size_t)N * L * D);
        layernorm(h.data(), emb_norm_.data(), nullptr, N * L, D, tmp.data());
        h.swap(tmp);
    }
    // RoPE tables
    auto build_cs = [](float theta, int L) {
        std::vector<float> cos(L * HD), sin(L * HD);
        for (int p = 0; p < L; ++p) for (int i = 0; i < HD / 2; ++i) {
            float inv = 1.0f / powf(theta, (float)(2 * i) / HD);
            float f = (float)p * inv;
            cos[p * HD + i] = cosf(f); cos[p * HD + HD/2 + i] = cosf(f);
            sin[p * HD + i] = sinf(f); sin[p * HD + HD/2 + i] = sinf(f);
        }
        return std::make_pair(cos, sin);
    };
    auto full_cs = build_cs(160000.0f, L);
    auto slid_cs = build_cs(10000.0f, L);
    std::vector<int8_t> key_pad((size_t)N * L);
    for (int b = 0; b < N; ++b) for (int t = 0; t < L; ++t) key_pad[(size_t)b * L + t] = (attn[(size_t)b * L + t] == 0);

    std::vector<float> hn((size_t)N * L * D), qkv((size_t)N * L * 3072), attn_out((size_t)N * L * D);
    std::vector<float> q((size_t)N * NHEAD * L * HD), k((size_t)N * NHEAD * L * HD), v((size_t)N * NHEAD * L * HD);
    for (int li = 0; li < NLAYER; ++li) {
        LayerW& Lw = layers_[li];
        bool is_full = (li % 3 == 0);
        const auto& cs = is_full ? full_cs : slid_cs;
        if (li > 0) layernorm(h.data(), Lw.attn_norm.data(), nullptr, N * L, D, hn.data());
        else memcpy(hn.data(), h.data(), h.size() * 4);
        linear(hn.data(), Lw.Wqkv.data(), nullptr, N * L, D, 3072, qkv.data());
        for (int b = 0; b < N; ++b) for (int t = 0; t < L; ++t) {
            const float* qkv_t = qkv.data() + ((size_t)b * L + t) * 3072;
            for (int hd = 0; hd < NHEAD; ++hd) {
                float* qo = q.data() + (((size_t)b * NHEAD + hd) * L + t) * HD;
                float* ko = k.data() + (((size_t)b * NHEAD + hd) * L + t) * HD;
                float* vo = v.data() + (((size_t)b * NHEAD + hd) * L + t) * HD;
                for (int d = 0; d < HD; ++d) { qo[d] = qkv_t[hd*HD+d]; ko[d] = qkv_t[1024+hd*HD+d]; vo[d] = qkv_t[2048+hd*HD+d]; }
            }
        }
        // RoPE
        for (int b = 0; b < N; ++b) for (int hd = 0; hd < NHEAD; ++hd) for (int t = 0; t < L; ++t) {
            float* qo = q.data() + (((size_t)b * NHEAD + hd) * L + t) * HD;
            float* ko = k.data() + (((size_t)b * NHEAD + hd) * L + t) * HD;
            const float* c = cs.first.data() + (size_t)t * HD;
            const float* s = cs.second.data() + (size_t)t * HD;
            for (int d = 0; d < HD/2; ++d) {
                float qa = qo[d], qb = qo[d+HD/2];
                qo[d] = qa*c[d] - qb*s[d]; qo[d+HD/2] = qb*c[d+HD/2] + qa*s[d+HD/2];
                float ka = ko[d], kb = ko[d+HD/2];
                ko[d] = ka*c[d] - kb*s[d]; ko[d+HD/2] = kb*c[d+HD/2] + ka*s[d+HD/2];
            }
        }
        // attention
        const float scale = 1.0f / sqrtf((float)HD);
        auto heads = [&](int bh0, int bh1) {
          std::vector<float> scores((size_t)L * L);
          for (int bh = bh0; bh < bh1; ++bh) {
            const int b = bh / NHEAD, hd = bh % NHEAD;
            const float* qb = q.data() + (((size_t)b * NHEAD + hd) * L) * HD;
            const float* kb = k.data() + (((size_t)b * NHEAD + hd) * L) * HD;
            for (int qi = 0; qi < L; ++qi) {
                float* sr = scores.data() + (size_t)qi * L;
                for (int kj = 0; kj < L; ++kj) {
                    if (attn[b * L + kj] == 0) { sr[kj] = -INFINITY; continue; }
                    if (!is_full && abs(qi - kj) > 64) { sr[kj] = -INFINITY; continue; }
                    const float* qr = qb + (size_t)qi * HD;
                    const float* kr = kb + (size_t)kj * HD;
                    float acc = 0; for (int d = 0; d < HD; ++d) acc += qr[d]*kr[d];
                    sr[kj] = acc * scale;
                }
                float mx = -INFINITY; for (int kj = 0; kj < L; ++kj) mx = fmaxf(mx, sr[kj]);
                float sum = 0; for (int kj = 0; kj < L; ++kj) { sr[kj] = expf(sr[kj]-mx); sum += sr[kj]; }
                for (int kj = 0; kj < L; ++kj) sr[kj] /= sum;
                float* orow = attn_out.data() + ((size_t)b * L + qi) * D + (size_t)hd * HD;
                for (int d = 0; d < HD; ++d) orow[d] = 0;
                for (int kj = 0; kj < L; ++kj) {
                    const float* vr = v.data() + (((size_t)b * NHEAD + hd) * L + kj) * HD;
                    float s = sr[kj];
                    for (int d = 0; d < HD; ++d) orow[d] += s * vr[d];
                }
            }
          }
        };
        parallel_ranges(N * NHEAD, worker_threads(), heads);
        // Wo projection + residual
        linear(attn_out.data(), Lw.Wo.data(), nullptr, N * L, D, D, hn.data());
        for (size_t z = 0; z < h.size(); ++z) h[z] += hn[z];
        // mlp_norm + geglu + residual
        layernorm(h.data(), Lw.mlp_norm.data(), nullptr, N * L, D, hn.data());
        geglu(hn.data(), Lw.Wi.data(), Lw.Wo_mlp.data(), N * L, attn_out.data());
        for (size_t z = 0; z < h.size(); ++z) h[z] += attn_out[z];
    }
    {
        std::vector<float> tmp((size_t)N * L * D);
        layernorm(h.data(), final_norm_.data(), nullptr, N * L, D, tmp.data());
        h.swap(tmp);
    }

    // ---- decision head ----
    for (int b = 0; b < N; ++b) {
        const float* te = type_emb_.data() + (size_t)qtype[b] * D;
        for (int t = 0; t < L; ++t) {
            float* o = h.data() + ((size_t)b * L + t) * D;
            for (int d = 0; d < D; ++d) o[d] += te[d];
        }
    }
    std::vector<float> tmp((size_t)N * L * D), tmp2((size_t)N * L * D), ffn((size_t)N * L * 4096);
    for (int layer = 0; layer < (int)head_layers_.size(); ++layer) {
        HeadLayerW& H = head_layers_[layer];
        layernorm(h.data(), H.n1w.data(), H.n1b.data(), N * L, D, tmp.data());
        mha(tmp.data(), H.in_w.data(), H.in_b.data(), H.out_w.data(), H.out_b.data(), N, L, key_pad.data(), tmp2.data());
        for (size_t z = 0; z < h.size(); ++z) h[z] += tmp2[z];
        layernorm(h.data(), H.n2w.data(), H.n2b.data(), N * L, D, tmp.data());
        linear(tmp.data(), H.l1w.data(), H.l1b.data(), N * L, D, 4096, ffn.data());
        for (size_t z = 0; z < (size_t)N * L * 4096; ++z) ffn[z] = fmaxf(0.0f, ffn[z]);  // relu
        linear(ffn.data(), H.l2w.data(), H.l2b.data(), N * L, 4096, D, tmp.data());
        for (size_t z = 0; z < h.size(); ++z) h[z] += tmp[z];
    }
    // gather markers
    std::vector<float> m((size_t)N * K * D);
    for (int b = 0; b < N; ++b) for (int kk = 0; kk < K; ++kk) {
        long long p = mpos[b * K + kk]; if (p < 0) p = 0;
        memcpy(m.data() + ((size_t)b * K + kk) * D, h.data() + ((size_t)b * L + p) * D, D * 4);
    }
    // scorer: LayerNorm -> Linear(D->D) -> GELU -> Linear(D->1)
    std::vector<float> mn((size_t)N * K * D), m1((size_t)N * K * D);
    layernorm(m.data(), sc0w.data(), sc0b.data(), N * K, D, mn.data());
    linear(mn.data(), sc1w.data(), sc1b.data(), N * K, D, D, m1.data());
    for (auto& x : m1) x = gelu(x);
    std::vector<float> logits((size_t)N * K);
    linear(m1.data(), sc3w.data(), sc3b.data(), N * K, D, 1, logits.data());
    for (int b = 0; b < N; ++b) for (int kk = 0; kk < K; ++kk)
        if (!mmask[b * K + kk]) logits[b * K + kk] = -1e4f;
    // act_head
    std::vector<float> act_logits((size_t)N * 2);
    for (int b = 0; b < N; ++b) {
        const float* lrow = logits.data() + (size_t)b * K;
        int kvalid = 0; for (int kk = 0; kk < K; ++kk) if (mmask[b * K + kk]) kvalid++;
        float keff = kvalid < 2 ? 2.0f : (float)kvalid;
        float p[64], mx = -INFINITY, sum = 0;
        for (int kk = 0; kk < K; ++kk) mx = fmaxf(mx, lrow[kk]);
        for (int kk = 0; kk < K; ++kk) { p[kk] = expf(lrow[kk] - mx); sum += p[kk]; }
        for (int kk = 0; kk < K; ++kk) p[kk] /= sum;
        float t1 = -INFINITY, t2 = -INFINITY;
        for (int kk = 0; kk < K; ++kk) { if (p[kk] > t1) { t2 = t1; t1 = p[kk]; } else if (p[kk] > t2) t2 = p[kk]; }
        if (t2 == -INFINITY) t2 = t1;
        float ent = 0; for (int kk = 0; kk < K; ++kk) ent += -p[kk] * logf(fmaxf(p[kk], 1e-9f));
        ent /= logf(keff);
        float feats[4] = { t1, t1 - t2, ent, keff / 255.0f };
        const float* pooled = h.data() + (size_t)b * L * D;
        float cat[1028];
        memcpy(cat, pooled, D * 4); memcpy(cat + D, feats, 4 * 4);
        float h256[256];
        linear(cat, act0w.data(), act0b.data(), 1, 1028, 256, h256);
        for (auto& x : h256) x = gelu(x);
        float a2[2];
        linear(h256, act2w.data(), act2b.data(), 1, 256, 2, a2);
        act_logits[b * 2] = a2[0]; act_logits[b * 2 + 1] = a2[1];
    }

    // raw pre-softmax outputs, for gating
    if (raw) {
        raw->logits.clear();
        raw->act_logits.clear();
        raw->logits.reserve(N);
        raw->act_logits.reserve(N);
        for (int b = 0; b < N; ++b) {
            raw->logits.emplace_back(logits.begin() + (size_t)b * K, logits.begin() + (size_t)(b + 1) * K);
            raw->act_logits.emplace_back(act_logits.begin() + (size_t)b * 2, act_logits.begin() + (size_t)(b + 1) * 2);
        }
    }

    // ---- temperature + softmax + confidence -> answers ----
    answers.clear();
    answers.reserve(N);
    for (int b = 0; b < N; ++b) {
        const Question& q = questions[b];
        int kvalid = 0; for (int kk = 0; kk < K; ++kk) if (mmask[b * K + kk]) kvalid++;
        // temperature bucket
        float tscale = temperature_[qtype[b]];
        std::string bucket;
        int sz = kvalid <= 2 ? 2 : kvalid <= 5 ? 5 : kvalid <= 10 ? 10 : 11;
        bucket = std::string(qtype[b] == 0 ? "choice:" : qtype[b] == 1 ? "score:" : "noul:") +
                 (sz == 2 ? "2" : sz == 5 ? "3-5" : sz == 10 ? "6-10" : "11+");
        for (auto& kv : temperature_by_options_) if (kv.first == bucket) { tscale = kv.second; break; }
        float z[64], mx = -INFINITY, sum = 0;
        for (int kk = 0; kk < kvalid; ++kk) { z[kk] = logits[b * K + kk] / tscale; mx = fmaxf(mx, z[kk]); }
        for (int kk = 0; kk < kvalid; ++kk) { z[kk] = expf(z[kk] - mx); sum += z[kk]; }
        for (int kk = 0; kk < kvalid; ++kk) z[kk] /= sum;
        float ent = 0; for (int kk = 0; kk < kvalid; ++kk) ent += -z[kk] * logf(fmaxf(z[kk], 1e-9f));
        float conf = 1.0f - ent / logf((float)kvalid);
        float actp = 0.0f;
        {
            // stable softmax: the act logits are large (~1e3-1e4), so expf alone overflows.
            float mx = fmaxf(act_logits[b*2], act_logits[b*2+1]);
            float ea0 = expf(act_logits[b*2] - mx), ea1 = expf(act_logits[b*2+1] - mx);
            actp = ea0 / (ea0 + ea1);
        }

        Answer a;
        a.type = q.type;
        a.confidence = std::clamp(conf, 0.0f, 1.0f);
        a.act_probability = actp;
        if (q.type == "choice") {
            int bi = 0; for (int kk = 1; kk < kvalid; ++kk) if (z[kk] > z[bi]) bi = kk;
            a.choice = q.criteria[bi].first;
            for (int kk = 0; kk < kvalid; ++kk) a.probabilities.push_back({q.criteria[kk].first, z[kk]});
        } else if (q.type == "score") {
            float exp_score = 0; for (int kk = 0; kk < kvalid; ++kk) exp_score += (float)kk * z[kk];
            a.score = exp_score;
            for (int kk = 0; kk < kvalid; ++kk) a.probabilities.push_back({std::to_string(kk), z[kk]});
        } else {
            a.noul = z[1];
            a.confidence = fmaxf(z[1], 1.0f - z[1]);
        }
        answers.push_back(std::move(a));
    }
    return true;
}

}  // namespace onebit::laya
