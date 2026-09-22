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
#include "cpu_model.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <memory>
#include <set>

#include "dequant.h"

namespace onebit {

namespace {

// KV cache length. Kept well under the training context so the reference
// stays small; raise it when a test needs longer sequences.
constexpr uint32_t kMaxCtx = 4096;

void rms_norm(const float* x, const float* w, float* out, uint32_t n, float eps) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; ++i) ss += double(x[i]) * x[i];
    const float scale = 1.0f / std::sqrt(float(ss / n) + eps);
    for (uint32_t i = 0; i < n; ++i) out[i] = w[i] * (x[i] * scale);
}

void rope(float* x, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta, RopeStyle style) {
    const uint32_t half = head_dim / 2;
    for (uint32_t i = 0; i < half; ++i) {
        const float inv_freq = 1.0f / std::pow(theta, float(2 * i) / float(head_dim));
        const float angle = float(pos) * inv_freq;
        const float c = std::cos(angle), s = std::sin(angle);
        for (uint32_t h = 0; h < n_heads; ++h) {
            float* v = x + h * head_dim;
            const uint32_t a = style == RopeStyle::Neox ? i : 2 * i;
            const uint32_t b = style == RopeStyle::Neox ? i + half : 2 * i + 1;
            const float x0 = v[a], x1 = v[b];
            v[a] = x0 * c - x1 * s;
            v[b] = x0 * s + x1 * c;
        }
    }
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

class Loader {
public:
    explicit Loader(const GgufFile& f) : f_(f) {}

    // Dequantizes tensor `name`, checking its shape is exactly `shape`.
    std::expected<std::vector<float>, std::string> take(const std::string& name, std::vector<uint64_t> shape) {
        const GgufTensor* t = f_.tensor(name);
        if (!t) return std::unexpected(std::format("missing tensor {}", name));
        if (t->ne != shape) return std::unexpected(std::format("tensor {} has an unexpected shape", name));
        std::vector<float> out(t->n_elements());
        if (auto r = dequantize(t->type, t->data, out.data(), out.size()); !r)
            return std::unexpected(std::format("tensor {}: {}", name, r.error()));
        used_.insert(name);
        return out;
    }

    bool has(const std::string& name) const { return f_.tensor(name) != nullptr; }

    std::expected<void, std::string> check_all_used() const {
        for (const GgufTensor& t : f_.tensors())
            if (!used_.contains(t.name)) return std::unexpected(std::format("tensor {} is not used by the reference", t.name));
        return {};
    }

private:
    const GgufFile& f_;
    std::set<std::string> used_;
};

}  // namespace

std::expected<CpuModel, std::string> CpuModel::load(const std::string& gguf_path, size_t n_threads) {
    auto file = GgufFile::open(gguf_path);
    if (!file) return std::unexpected(file.error());
    auto cfg = read_model_config(*file);
    if (!cfg) return std::unexpected(cfg.error());

    CpuModel m;
    m.cfg_ = *cfg;
    const ModelConfig& c = m.cfg_;
    const uint64_t E = c.n_embd, D = c.head_dim, Q = uint64_t{c.n_head} * D, KV = uint64_t{c.n_head_kv} * D;
    const uint64_t F = c.n_ff, V = c.n_vocab;

    Loader ld(*file);
#define TAKE(dst, name, ...)                                   \
    do {                                                       \
        auto r_ = ld.take(name, {__VA_ARGS__});                \
        if (!r_) return std::unexpected(std::move(r_.error())); \
        dst = std::move(*r_);                                  \
    } while (0)

    TAKE(m.tok_embd_, "token_embd.weight", E, V);
    TAKE(m.output_norm_, "output_norm.weight", E);
    if (ld.has("output.weight")) TAKE(m.output_, "output.weight", E, V);

    m.layers_.resize(c.n_layer);
    for (uint32_t i = 0; i < c.n_layer; ++i) {
        Layer& L = m.layers_[i];
        const std::string p = std::format("blk.{}.", i);
        TAKE(L.attn_norm, p + "attn_norm.weight", E);
        TAKE(L.wq, p + "attn_q.weight", E, Q);
        TAKE(L.wk, p + "attn_k.weight", E, KV);
        TAKE(L.wv, p + "attn_v.weight", E, KV);
        TAKE(L.wo, p + "attn_output.weight", Q, E);
        TAKE(L.q_norm, p + "attn_q_norm.weight", D);
        TAKE(L.k_norm, p + "attn_k_norm.weight", D);
        TAKE(L.ffn_norm, p + "ffn_norm.weight", E);
        TAKE(L.w_gate, p + "ffn_gate.weight", E, F);
        TAKE(L.w_up, p + "ffn_up.weight", E, F);
        TAKE(L.w_down, p + "ffn_down.weight", F, E);
    }
#undef TAKE
    if (auto r = ld.check_all_used(); !r) return std::unexpected(r.error());

    m.n_ctx_ = std::min(c.n_ctx_train, kMaxCtx);
    const size_t cache = size_t{c.n_layer} * m.n_ctx_ * KV;
    m.k_cache_.assign(cache, 0.0f);
    m.v_cache_.assign(cache, 0.0f);
    m.x_.resize(E);
    m.xn_.resize(E);
    m.q_.resize(Q);
    m.k_.resize(KV);
    m.v_.resize(KV);
    m.attn_.resize(Q);
    m.scores_.resize(m.n_ctx_);
    m.gate_.resize(F);
    m.up_.resize(F);
    m.logits_.resize(V);
    m.pool_ = std::make_unique<ThreadPool>(n_threads);
    return m;
}

void CpuModel::reset() { n_past_ = 0; }

void CpuModel::matvec(const std::vector<float>& w, const float* x, float* y, uint32_t rows, uint32_t cols) {
    pool_->parallel_for(rows, [&](size_t begin, size_t end) {
        for (size_t r = begin; r < end; ++r) {
            const float* row = w.data() + r * cols;
            float acc = 0.0f;
            for (uint32_t i = 0; i < cols; ++i) acc += row[i] * x[i];
            y[r] = acc;
        }
    });
}

std::expected<const std::vector<float>*, std::string> CpuModel::forward(int32_t token) {
    const ModelConfig& c = cfg_;
    if (token < 0 || uint32_t(token) >= c.n_vocab) return std::unexpected(std::format("token {} out of range", token));
    if (n_past_ >= n_ctx_) return std::unexpected(std::format("context full ({} tokens)", n_ctx_));

    const uint32_t E = c.n_embd, D = c.head_dim, H = c.n_head, HKV = c.n_head_kv;
    const uint32_t KV = HKV * D, pos = n_past_, group = H / HKV;
    const float att_scale = 1.0f / std::sqrt(float(D));

    std::copy_n(tok_embd_.data() + size_t(token) * E, E, x_.data());

    for (uint32_t l = 0; l < c.n_layer; ++l) {
        const Layer& L = layers_[l];

        rms_norm(x_.data(), L.attn_norm.data(), xn_.data(), E, c.rms_eps);
        matvec(L.wq, xn_.data(), q_.data(), H * D, E);
        matvec(L.wk, xn_.data(), k_.data(), KV, E);
        matvec(L.wv, xn_.data(), v_.data(), KV, E);
        for (uint32_t h = 0; h < H; ++h) rms_norm(q_.data() + h * D, L.q_norm.data(), q_.data() + h * D, D, c.rms_eps);
        for (uint32_t h = 0; h < HKV; ++h) rms_norm(k_.data() + h * D, L.k_norm.data(), k_.data() + h * D, D, c.rms_eps);
        rope(q_.data(), H, D, pos, c.rope_theta, c.rope_style);
        rope(k_.data(), HKV, D, pos, c.rope_theta, c.rope_style);

        float* kc = k_cache_.data() + (size_t(l) * n_ctx_) * KV;
        float* vc = v_cache_.data() + (size_t(l) * n_ctx_) * KV;
        std::copy_n(k_.data(), KV, kc + size_t(pos) * KV);
        std::copy_n(v_.data(), KV, vc + size_t(pos) * KV);

        for (uint32_t h = 0; h < H; ++h) {
            const float* q = q_.data() + h * D;
            const uint32_t kvh = h / group;
            float mx = -INFINITY;
            for (uint32_t t = 0; t <= pos; ++t) {
                const float* k = kc + size_t(t) * KV + kvh * D;
                float s = 0.0f;
                for (uint32_t i = 0; i < D; ++i) s += q[i] * k[i];
                scores_[t] = s * att_scale;
                mx = std::max(mx, scores_[t]);
            }
            double sum = 0.0;
            for (uint32_t t = 0; t <= pos; ++t) {
                scores_[t] = std::exp(scores_[t] - mx);
                sum += scores_[t];
            }
            float* out = attn_.data() + h * D;
            std::fill_n(out, D, 0.0f);
            for (uint32_t t = 0; t <= pos; ++t) {
                const float p = float(scores_[t] / sum);
                const float* v = vc + size_t(t) * KV + kvh * D;
                for (uint32_t i = 0; i < D; ++i) out[i] += p * v[i];
            }
        }
        matvec(L.wo, attn_.data(), xn_.data(), E, H * D);
        for (uint32_t i = 0; i < E; ++i) x_[i] += xn_[i];

        rms_norm(x_.data(), L.ffn_norm.data(), xn_.data(), E, c.rms_eps);
        matvec(L.w_gate, xn_.data(), gate_.data(), c.n_ff, E);
        matvec(L.w_up, xn_.data(), up_.data(), c.n_ff, E);
        for (uint32_t i = 0; i < c.n_ff; ++i) gate_[i] = silu(gate_[i]) * up_[i];
        matvec(L.w_down, gate_.data(), xn_.data(), E, c.n_ff);
        for (uint32_t i = 0; i < E; ++i) x_[i] += xn_[i];
    }

    rms_norm(x_.data(), output_norm_.data(), xn_.data(), E, c.rms_eps);
    matvec(output_.empty() ? tok_embd_ : output_, xn_.data(), logits_.data(), c.n_vocab, E);
    ++n_past_;
    return &logits_;
}

}  // namespace onebit
