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
// Proprietary and confidential. Distributed only as compiled binaries under the
// 1bit Binary License (LICENSE-BINARY).

#include "q4nx_vision.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace q4nx_vision {
namespace {

constexpr float kLnEps = 1e-6f;   // the tower's LayerNorm epsilon, measured (corr 1.000000 at 1e-6, 0.0001 off at 1e-5)

void layernorm(const float* in, const float* w, const float* b, int rows, int cols, float* out) {
    for (int r = 0; r < rows; ++r) {
        const float* p = in + size_t(r) * cols;
        float mu = 0.0f;
        for (int c = 0; c < cols; ++c) mu += p[c];
        mu /= float(cols);
        float var = 0.0f;
        for (int c = 0; c < cols; ++c) { const float d = p[c] - mu; var += d * d; }
        var /= float(cols);
        const float inv = 1.0f / std::sqrt(var + kLnEps);
        float* o = out + size_t(r) * cols;
        for (int c = 0; c < cols; ++c) o[c] = (p[c] - mu) * inv * (w ? w[c] : 1.0f) + (b ? b[c] : 0.0f);
    }
}

// y = x @ W^T + bias, x (rows, k), W (n, k), y (rows, n)
void matmul_t(const float* x, const float* W, const float* b, int rows, int k, int n, float* y) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + size_t(r) * k;
        for (int j = 0; j < n; ++j) {
            const float* wj = W + size_t(j) * k;
            float acc = b ? b[j] : 0.0f;
            for (int c = 0; c < k; ++c) acc += xr[c] * wj[c];
            y[size_t(r) * n + j] = acc;
        }
    }
}

inline float gelu_tanh(float v) {
    // 0.5*v*(1 + tanh(sqrt(2/pi)*(v + 0.044715 v^3))) -- config `gelu_pytorch_tanh`
    return 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
}

}  // namespace

void block_forward(float* x, int tokens, int hidden, int heads, int head_dim, int intermediate,
                   const VisionBlockWeights& w, const float* inv_freq, const int* pos_h, const int* pos_w) {
    if (!x || !w.qkv_w || tokens <= 0 || hidden <= 0 || heads <= 0 || head_dim <= 0 || heads * head_dim != hidden ||
        intermediate <= 0 || (head_dim % 4)) {
        std::fprintf(stderr, "vision block_forward: bad geometry tokens=%d h=%d heads=%d hd=%d im=%d\n", tokens, hidden,
                     heads, head_dim, intermediate);
        return;
    }
    const size_t th = size_t(tokens) * hidden;
    std::vector<float> n1(th), qkv(size_t(tokens) * 3 * hidden), attn(th), prj(th), n2(th),
        f1(size_t(tokens) * intermediate), f2(th);

    layernorm(x, w.ln1_w, w.ln1_b, tokens, hidden, n1.data());
    matmul_t(n1.data(), w.qkv_w, w.qkv_b, tokens, hidden, 3 * hidden, qkv.data());

    // three contiguous hidden-wide chunks: q | k | v (the head-major reading scores 0.41 against this one's 0.99)
    for (int t = 0; t < tokens; ++t) {
        float* q = qkv.data() + size_t(t) * 3 * hidden;
        float* k = q + hidden;
        float* v = k + hidden;
        rope_axial(q, 1, heads, head_dim, inv_freq, pos_h + t, pos_w + t);
        rope_axial(k, 1, heads, head_dim, inv_freq, pos_h + t, pos_w + t);
        (void)v;
    }
    const float scale = 1.0f / std::sqrt(float(head_dim));
    std::vector<float> scores(static_cast<size_t>(tokens));
    for (int t = 0; t < tokens; ++t) {
        const float* qt = qkv.data() + size_t(t) * 3 * hidden;
        for (int h = 0; h < heads; ++h) {
            const float* qh = qt + size_t(h) * head_dim;
            float mx = -INFINITY;
            for (int s = 0; s < tokens; ++s) {
                const float* kh = qkv.data() + size_t(s) * 3 * hidden + hidden + size_t(h) * head_dim;
                float acc = 0.0f;
                for (int d = 0; d < head_dim; ++d) acc += qh[d] * kh[d];
                acc *= scale;
                scores[size_t(s)] = acc;
                if (acc > mx) mx = acc;
            }
            float sum = 0.0f;
            for (int s = 0; s < tokens; ++s) { scores[size_t(s)] = std::exp(scores[size_t(s)] - mx); sum += scores[size_t(s)]; }
            float* o = attn.data() + size_t(t) * hidden + size_t(h) * head_dim;
            for (int d = 0; d < head_dim; ++d) o[d] = 0.0f;
            for (int s = 0; s < tokens; ++s) {
                const float p = scores[size_t(s)] / sum;
                const float* vh = qkv.data() + size_t(s) * 3 * hidden + 2 * hidden + size_t(h) * head_dim;
                for (int d = 0; d < head_dim; ++d) o[d] += p * vh[d];
            }
        }
    }
    matmul_t(attn.data(), w.proj_w, w.proj_b, tokens, hidden, hidden, prj.data());
    for (size_t i = 0; i < th; ++i) x[i] += prj[i];

    layernorm(x, w.ln2_w, w.ln2_b, tokens, hidden, n2.data());
    matmul_t(n2.data(), w.fc1_w, w.fc1_b, tokens, hidden, intermediate, f1.data());
    for (auto& v : f1) v = gelu_tanh(v);
    matmul_t(f1.data(), w.fc2_w, w.fc2_b, tokens, intermediate, hidden, f2.data());
    for (size_t i = 0; i < th; ++i) x[i] += f2[i];
}

void merger_forward(const float* x, int tokens, int hidden, int out_hidden, const float* ln_w, const float* ln_b,
                    const float* fc1_w, const float* fc1_b, const float* fc2_w, const float* fc2_b, float* out) {
    if (!x || !out || tokens <= 0 || hidden <= 0 || out_hidden <= 0 || (tokens % 4)) {
        std::fprintf(stderr, "vision merger_forward: bad geometry tokens=%d h=%d out=%d\n", tokens, hidden, out_hidden);
        return;
    }
    std::vector<float> n(size_t(tokens) * hidden);
    layernorm(x, ln_w, ln_b, tokens, hidden, n.data());
    const int rows = tokens / 4;
    const int wide = 4 * hidden;                       // the 2x2 merge: four tokens concatenated
    std::vector<float> f1(size_t(rows) * wide);
    matmul_t(n.data(), fc1_w, fc1_b, rows, wide, wide, f1.data());
    for (auto& v : f1) v = gelu_tanh(v);
    matmul_t(f1.data(), fc2_w, fc2_b, rows, wide, out_hidden, out);
}

}  // namespace q4nx_vision

namespace q4nx_vision {

void tower_forward(const float* patches, int tokens, int patch_dim, int gh, int gw, int hidden, int heads, int head_dim,
                   int intermediate, int out_hidden, const VisionTowerWeights& w, const float* inv_freq, float* out) {
    if (!patches || !out || !w.conv_w || !w.pos_table || !w.blocks || tokens <= 0 || hidden <= 0 ||
        gh * gw != tokens || heads * head_dim != hidden) {
        std::fprintf(stderr, "vision tower_forward: bad geometry tokens=%d gh=%d gw=%d h=%d heads=%d hd=%d\n", tokens, gh, gw,
                     hidden, heads, head_dim);
        return;
    }
    std::vector<float> x(size_t(tokens) * hidden);
    matmul_t(patches, w.conv_w, w.conv_b, tokens, patch_dim, hidden, x.data());
    std::vector<float> pe(size_t(tokens) * hidden);
    pos_embed(w.pos_table, gh, gw, hidden, pe.data());
    for (size_t i = 0; i < x.size(); ++i) x[i] += pe[i];

    std::vector<int> ph(static_cast<size_t>(tokens)), pw(static_cast<size_t>(tokens));
    for (int t = 0; t < tokens; ++t) block_position(t, gh, gw, &ph[size_t(t)], &pw[size_t(t)]);

    for (int b = 0; b < w.n_blocks; ++b)
        block_forward(x.data(), tokens, hidden, heads, head_dim, intermediate, w.blocks[b], inv_freq, ph.data(), pw.data());

    merger_forward(x.data(), tokens, hidden, out_hidden, w.m_ln_w, w.m_ln_b, w.m_fc1_w, w.m_fc1_b, w.m_fc2_w, w.m_fc2_b, out);
}

}  // namespace q4nx_vision
