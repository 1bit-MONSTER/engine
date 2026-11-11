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

// dflash_forward.cpp -- see dflash_forward.h for what is computed and why it is written this plainly.

#include "dflash_forward.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace onebit::npu::dflash {

namespace {

void need(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("dflash forward: " + what);
}

// y[rows][n_out] = x[rows][n_in] * W^T, W row-major [n_out][n_in].
std::vector<float> linear(const std::vector<float>& W, int n_out, int n_in, const float* x, int rows) {
    std::vector<float> y(size_t(rows) * size_t(n_out));
    for (int r = 0; r < rows; r++) {
        const float* xr = x + size_t(r) * size_t(n_in);
        for (int n = 0; n < n_out; n++) {
            const float* wr = W.data() + size_t(n) * size_t(n_in);
            double acc = 0.0;
            for (int k = 0; k < n_in; k++) acc += double(wr[k]) * double(xr[k]);
            y[size_t(r) * size_t(n_out) + size_t(n)] = float(acc);
        }
    }
    return y;
}

// RMS norm over groups of `width` values: x * gain / sqrt(mean(x^2) + eps). `gain` has `width` entries.
void rms_norm(float* x, size_t groups, int width, const std::vector<float>& gain, float eps) {
    for (size_t g = 0; g < groups; g++) {
        float* v = x + g * size_t(width);
        double ss = 0.0;
        for (int i = 0; i < width; i++) ss += double(v[i]) * double(v[i]);
        const double inv = 1.0 / std::sqrt(ss / double(width) + double(eps));
        for (int i = 0; i < width; i++) v[i] = float(double(v[i]) * inv * double(gain[size_t(i)]));
    }
}

// Rotate one head vector of width d at position `pos`: pairs (i, i + d/2) turn by pos * theta^(-2i/d).
void rope(float* v, int d, long pos, float theta) {
    const int half = d / 2;
    for (int i = 0; i < half; i++) {
        const double ang = double(pos) * std::pow(double(theta), -2.0 * double(i) / double(d));
        const double c = std::cos(ang), s = std::sin(ang);
        const double a = v[i], b = v[i + half];
        v[i] = float(a * c - b * s);
        v[i + half] = float(b * c + a * s);
    }
}

}  // namespace

void check_weights(const DraftWeights& w) {
    need(w.hidden > 0 && w.intermediate > 0 && w.heads > 0 && w.kv_heads > 0 && w.head_dim > 0 && w.taps > 0,
         "a geometry field is zero or negative");
    need(w.head_dim % 2 == 0, "head_dim " + std::to_string(w.head_dim) + " is odd, so it cannot be rotated in pairs");
    need(w.heads % w.kv_heads == 0, std::to_string(w.heads) + " heads are not a multiple of " +
                                        std::to_string(w.kv_heads) + " key/value heads");
    need(w.rope_theta > 0.0f, "rope_theta is not positive");
    need(!w.layers.empty(), "no layers");
    const size_t H = size_t(w.hidden), I = size_t(w.intermediate);
    const size_t Q = size_t(w.heads) * size_t(w.head_dim), KV = size_t(w.kv_heads) * size_t(w.head_dim);
    need(w.fc.size() == H * H * size_t(w.taps), "fc is not hidden x (taps * hidden)");
    need(w.hidden_norm.size() == H && w.norm.size() == H, "hidden_norm or norm is not hidden wide");
    for (size_t l = 0; l < w.layers.size(); l++) {
        const DraftLayerWeights& L = w.layers[l];
        const std::string at = "layer " + std::to_string(l) + ": ";
        need(L.q.size() == Q * H, at + "q_proj is not (heads * head_dim) x hidden");
        need(L.k.size() == KV * H && L.v.size() == KV * H, at + "k_proj or v_proj is not (kv_heads * head_dim) x hidden");
        need(L.o.size() == H * Q, at + "o_proj is not hidden x (heads * head_dim)");
        need(L.gate.size() == I * H && L.up.size() == I * H, at + "gate or up is not intermediate x hidden");
        need(L.down.size() == H * I, at + "down is not hidden x intermediate");
        need(L.input_norm.size() == H && L.post_norm.size() == H, at + "a layer norm is not hidden wide");
        need(L.q_norm.size() == size_t(w.head_dim) && L.k_norm.size() == size_t(w.head_dim),
             at + "q_norm or k_norm is not head_dim wide");
    }
}

DraftWeights decode_draft(const DraftModel& m) {
    DraftWeights w;
    w.hidden = m.cfg.hidden;
    w.intermediate = m.cfg.intermediate;
    w.heads = m.cfg.heads;
    w.kv_heads = m.cfg.kv_heads;
    w.head_dim = m.cfg.head_dim;
    w.taps = m.cfg.taps();
    w.rms_eps = m.cfg.rms_eps;
    w.rope_theta = m.cfg.rope_theta;
    w.fc = decode_fusion(m.fc);
    w.hidden_norm = decode_gain(m.hidden_norm);
    w.norm = decode_gain(m.norm);
    w.layers.reserve(m.layers.size());
    for (const DraftLayer& L : m.layers) {
        DraftLayerWeights d;
        d.q = decode_projection(L.q);
        d.k = decode_projection(L.k);
        d.v = decode_projection(L.v);
        d.o = decode_projection(L.o);
        d.gate = decode_projection(L.gate);
        d.up = decode_projection(L.up);
        d.down = decode_projection(L.down);
        d.input_norm = decode_gain(L.input_norm);
        d.post_norm = decode_gain(L.post_norm);
        d.q_norm = decode_gain(L.q_norm);
        d.k_norm = decode_gain(L.k_norm);
        w.layers.push_back(std::move(d));
    }
    check_weights(w);
    return w;
}

std::vector<float> draft_forward(const DraftWeights& w, const float* taps, int ctx, const float* noise, int block,
                                 long pos0, int window) {
    check_weights(w);
    need(block > 0 && noise != nullptr, "an empty block");
    need(ctx >= 0 && (ctx == 0 || taps != nullptr), "a context length without context rows");
    need(window >= 0, "a negative window");
    const int H = w.hidden, I = w.intermediate, D = w.head_dim;
    const int Q = w.heads * D, KV = w.kv_heads * D;
    const int keys = ctx + block;
    const int group = w.heads / w.kv_heads;   // query heads per key/value head, consecutive

    // The context: fused, normalised once, and the same rows for every layer.
    std::vector<float> c;
    if (ctx > 0) {
        c = linear(w.fc, H, H * w.taps, taps, ctx);
        rms_norm(c.data(), size_t(ctx), H, w.hidden_norm, w.rms_eps);
    }

    std::vector<float> x(noise, noise + size_t(block) * size_t(H));
    std::vector<float> kv_in(size_t(keys) * size_t(H));
    std::copy(c.begin(), c.end(), kv_in.begin());
    std::vector<double> score(static_cast<size_t>(keys));
    std::vector<float> attn(size_t(block) * size_t(Q));

    for (const DraftLayerWeights& L : w.layers) {
        // attention
        std::vector<float> h = x;
        rms_norm(h.data(), size_t(block), H, L.input_norm, w.rms_eps);
        std::copy(h.begin(), h.end(), kv_in.begin() + ptrdiff_t(size_t(ctx) * size_t(H)));

        std::vector<float> q = linear(L.q, Q, H, h.data(), block);
        std::vector<float> k = linear(L.k, KV, H, kv_in.data(), keys);
        std::vector<float> v = linear(L.v, KV, H, kv_in.data(), keys);
        rms_norm(q.data(), size_t(block) * size_t(w.heads), D, L.q_norm, w.rms_eps);
        rms_norm(k.data(), size_t(keys) * size_t(w.kv_heads), D, L.k_norm, w.rms_eps);
        for (int i = 0; i < block; i++)
            for (int hd = 0; hd < w.heads; hd++)
                rope(q.data() + size_t(i) * size_t(Q) + size_t(hd) * size_t(D), D, pos0 + ctx + i, w.rope_theta);
        for (int t = 0; t < keys; t++)
            for (int hd = 0; hd < w.kv_heads; hd++)
                rope(k.data() + size_t(t) * size_t(KV) + size_t(hd) * size_t(D), D, pos0 + t, w.rope_theta);

        const double scale = 1.0 / std::sqrt(double(D));
        for (int i = 0; i < block; i++) {
            for (int hd = 0; hd < w.heads; hd++) {
                const int kh = hd / group;
                const float* qv = q.data() + size_t(i) * size_t(Q) + size_t(hd) * size_t(D);
                double top = -INFINITY;
                for (int t = 0; t < keys; t++) {
                    const long dist = long(ctx + i) - long(t);
                    if (window > 0 && (dist >= window || -dist >= window)) {
                        score[size_t(t)] = -INFINITY;
                        continue;
                    }
                    const float* kvp = k.data() + size_t(t) * size_t(KV) + size_t(kh) * size_t(D);
                    double s = 0.0;
                    for (int d = 0; d < D; d++) s += double(qv[d]) * double(kvp[d]);
                    score[size_t(t)] = s * scale;
                    if (score[size_t(t)] > top) top = score[size_t(t)];
                }
                double sum = 0.0;
                for (int t = 0; t < keys; t++) {
                    score[size_t(t)] = std::isinf(score[size_t(t)]) ? 0.0 : std::exp(score[size_t(t)] - top);
                    sum += score[size_t(t)];
                }
                need(sum > 0.0 && std::isfinite(sum), "an attention row has no visible key or a non-finite score");
                float* out = attn.data() + size_t(i) * size_t(Q) + size_t(hd) * size_t(D);
                for (int d = 0; d < D; d++) {
                    double acc = 0.0;
                    for (int t = 0; t < keys; t++)
                        acc += score[size_t(t)] * double(v[size_t(t) * size_t(KV) + size_t(kh) * size_t(D) + size_t(d)]);
                    out[d] = float(acc / sum);
                }
            }
        }
        std::vector<float> o = linear(L.o, H, Q, attn.data(), block);
        for (size_t j = 0; j < x.size(); j++) x[j] += o[j];

        // MLP
        std::vector<float> p = x;
        rms_norm(p.data(), size_t(block), H, L.post_norm, w.rms_eps);
        std::vector<float> g = linear(L.gate, I, H, p.data(), block);
        std::vector<float> u = linear(L.up, I, H, p.data(), block);
        for (size_t j = 0; j < g.size(); j++) {
            const double a = g[j];
            g[j] = float(a / (1.0 + std::exp(-a)) * double(u[j]));   // silu(gate) * up
        }
        std::vector<float> dn = linear(L.down, H, I, g.data(), block);
        for (size_t j = 0; j < x.size(); j++) x[j] += dn[j];
    }

    rms_norm(x.data(), size_t(block), H, w.norm, w.rms_eps);
    for (float f : x) need(std::isfinite(f), "a non-finite output");
    return x;
}

}  // namespace onebit::npu::dflash
