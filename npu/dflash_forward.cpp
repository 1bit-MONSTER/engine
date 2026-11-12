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

GemmHook g_hook = nullptr;
void* g_hook_user = nullptr;

// y[rows][n_out] = x[rows][n_in] * W^T, W row-major [n_out][n_in].
std::vector<float> linear(const std::vector<float>& W, int n_out, int n_in, const float* x, int rows) {
    std::vector<float> y(size_t(rows) * size_t(n_out));
    if (g_hook && rows > 0 && g_hook(W.data(), n_out, n_in, x, rows, y.data(), g_hook_user)) return y;
    // Output rows are independent, so they are shared out across threads when the build has OpenMP; the sum itself
    // stays in double and may be taken in vector lanes, so two builds can differ in the last bits of a result.
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < n_out; n++) {
        const float* wr = W.data() + size_t(n) * size_t(n_in);
        for (int r = 0; r < rows; r++) {
            const float* xr = x + size_t(r) * size_t(n_in);
            double acc = 0.0;
            #pragma omp simd reduction(+ : acc)
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

// One half of a DFlash 2 convolution over `rows` rows of H. `dyn` is [rows][kernel][groups] for this half.
std::vector<float> convolve(const float* x, int rows, int H, int kernel, int group, const float* base, const float* dyn) {
    const int groups = H / group;
    std::vector<float> out(size_t(rows) * size_t(H), 0.0f);
    for (int t = 0; t < rows; t++)
        for (int off = 0; off < kernel && off <= t; off++) {
            const float* src = x + size_t(t - off) * size_t(H);
            const float* b = base + size_t(off) * size_t(H);
            const float* d = dyn + (size_t(t) * size_t(kernel) + size_t(off)) * size_t(groups);
            float* o = out.data() + size_t(t) * size_t(H);
            for (int g = 0; g < groups; g++)
                for (int j = 0; j < group; j++) {
                    const int i = g * group + j;
                    o[i] += (b[i] + d[g]) * src[i];
                }
        }
    return out;
}

// prepare(): the row's kernels from the row, the first half applied now, the second half kept for finish().
std::vector<float> conv_prepare(const DraftWeights& w, const DraftConv& c, const std::vector<float>& x, int rows,
                                std::vector<float>& keep) {
    const int H = w.hidden, K = w.conv_kernel, G = H / w.conv_group;
    const std::vector<float> dyn = linear(c.proj, 2 * K * G, H, x.data(), rows);   // [rows][2][K][G]
    std::vector<float> first(size_t(rows) * size_t(K) * size_t(G));
    keep.assign(first.size(), 0.0f);
    for (int t = 0; t < rows; t++) {
        const float* d = dyn.data() + size_t(t) * size_t(2 * K * G);
        std::copy(d, d + K * G, first.begin() + ptrdiff_t(size_t(t) * size_t(K * G)));
        std::copy(d + K * G, d + 2 * K * G, keep.begin() + ptrdiff_t(size_t(t) * size_t(K * G)));
    }
    return convolve(x.data(), rows, H, K, w.conv_group, c.base.data(), first.data());
}

std::vector<float> conv_finish(const DraftWeights& w, const DraftConv& c, const std::vector<float>& x, int rows,
                               const std::vector<float>& keep) {
    return convolve(x.data(), rows, w.hidden, w.conv_kernel, w.conv_group,
                    c.base.data() + size_t(w.conv_kernel) * size_t(w.hidden), keep.data());
}

}  // namespace

std::vector<float> draft_conv(const float* x, int rows, int hidden, int kernel, int group, const float* base,
                              const float* dyn) {
    need(x && base && dyn && rows > 0 && hidden > 0 && kernel > 0 && group > 0 && hidden % group == 0,
         "a convolution with an empty input, or a group size that does not divide the hidden width");
    return convolve(x, rows, hidden, kernel, group, base, dyn);
}

int selector_pick(const float* unary, int k, const float* prev, const float* h, const float* succ, int rank) {
    need(unary && prev && h && succ && k > 0 && rank > 0, "a selector row with no candidates or no codebook rows");
    int best = 0;
    double top = 0.0;
    for (int c = 0; c < k; c++) {
        double score = unary[c];
        const float* s = succ + size_t(c) * size_t(rank);
        for (int r = 0; r < rank; r++) score += double(prev[r]) * double(h[r]) * double(s[r]);
        if (c == 0 || score > top) {
            top = score;
            best = c;
        }
    }
    return best;
}

void set_gemm_hook(GemmHook hook, void* user) {
    g_hook = hook;
    g_hook_user = user;
}

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
        for (const DraftConv* c : {&L.attn_conv, &L.mlp_conv}) {
            if (w.conv_kernel == 0) {
                need(!c->present(), at + "a convolution on a draft that declares none");
                continue;
            }
            need(w.conv_kernel > 0 && w.conv_group > 0 && w.hidden % w.conv_group == 0,
                 "the convolution's kernel or group size does not fit the hidden width");
            const size_t G = size_t(w.hidden / w.conv_group);
            need(c->base.size() == 2 * size_t(w.conv_kernel) * H, at + "a convolution's base kernel is not 2 x kernel x hidden");
            need(c->proj.size() == 2 * size_t(w.conv_kernel) * G * H,
                 at + "a convolution's kernel projection is not (2 * kernel * groups) x hidden");
        }
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
    w.fc = m.cfg.generation == 2 ? decode_projection(m.fc) : decode_fusion(m.fc);
    w.conv_kernel = m.cfg.conv_kernel;
    w.conv_group = m.cfg.conv_group;
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
        if (m.cfg.generation == 2) {
            d.attn_conv.base = decode_plain(L.attn_conv_base);
            d.attn_conv.proj = decode_plain(L.attn_conv_proj);
            d.mlp_conv.base = decode_plain(L.mlp_conv_base);
            d.mlp_conv.proj = decode_plain(L.mlp_conv_proj);
        }
        w.layers.push_back(std::move(d));
    }
    check_weights(w);
    return w;
}

std::vector<float> fuse_context(const DraftWeights& w, const float* taps, int ctx) {
    check_weights(w);
    need(ctx >= 0 && (ctx == 0 || taps != nullptr), "a context length without context rows");
    std::vector<float> c;
    if (ctx > 0) {
        c = linear(w.fc, w.hidden, w.hidden * w.taps, taps, ctx);
        rms_norm(c.data(), size_t(ctx), w.hidden, w.hidden_norm, w.rms_eps);
    }
    return c;
}

std::vector<float> draft_forward(const DraftWeights& w, const float* taps, int ctx, const float* noise, int block,
                                 long pos0, int window) {
    const std::vector<float> c = fuse_context(w, taps, ctx);
    return draft_forward_fused(w, c.data(), ctx, noise, block, pos0, window);
}

void cache_extend(const DraftWeights& w, DraftKvCache& c, const float* fused_rows, int n) {
    check_weights(w);
    need(n >= 0 && (n == 0 || fused_rows != nullptr), "context rows to add without the rows");
    const int H = w.hidden, D = w.head_dim, KV = w.kv_heads * D;
    if (c.k.size() != w.layers.size()) {
        need(c.rows == 0, "a cache whose layer count does not match the draft");
        c.k.assign(w.layers.size(), {});
        c.v.assign(w.layers.size(), {});
    }
    if (n == 0) return;
    for (size_t l = 0; l < w.layers.size(); l++) {
        const DraftLayerWeights& L = w.layers[l];
        std::vector<float> k = linear(L.k, KV, H, fused_rows, n);
        const std::vector<float> v = linear(L.v, KV, H, fused_rows, n);
        rms_norm(k.data(), size_t(n) * size_t(w.kv_heads), D, L.k_norm, w.rms_eps);
        for (int t = 0; t < n; t++)
            for (int hd = 0; hd < w.kv_heads; hd++)
                rope(k.data() + size_t(t) * size_t(KV) + size_t(hd) * size_t(D), D, c.pos0 + c.rows + t, w.rope_theta);
        c.k[l].insert(c.k[l].end(), k.begin(), k.end());
        c.v[l].insert(c.v[l].end(), v.begin(), v.end());
    }
    c.rows += n;
}

std::vector<float> draft_forward_fused(const DraftWeights& w, const float* fused, int ctx, const float* noise, int block,
                                       long pos0, int window) {
    need(ctx >= 0 && (ctx == 0 || fused != nullptr), "a context length without context rows");
    DraftKvCache c;
    c.pos0 = pos0;
    cache_extend(w, c, fused, ctx);
    return draft_forward_cached(w, c, noise, block, window);
}

std::vector<float> draft_forward_cached(const DraftWeights& w, const DraftKvCache& c, const float* noise, int block,
                                        int window) {
    check_weights(w);
    need(block > 0 && noise != nullptr, "an empty block");
    need(window >= 0, "a negative window");
    need(c.rows == 0 || c.k.size() == w.layers.size(), "a cache that was not built for this draft");
    const int H = w.hidden, I = w.intermediate, D = w.head_dim;
    const int Q = w.heads * D, KV = w.kv_heads * D;
    const int ctx = c.rows;
    const long pos0 = c.pos0;
    const int keys = ctx + block;
    const int group = w.heads / w.kv_heads;   // query heads per key/value head, consecutive

    // The context's keys and values come from the cache: they depend only on the fused context rows and the layer,
    // never on the block, so a caller that drafts many blocks over one growing context computes each row once.
    std::vector<float> x(noise, noise + size_t(block) * size_t(H));
    std::vector<double> score(static_cast<size_t>(keys));
    std::vector<float> attn(size_t(block) * size_t(Q));

    for (size_t l = 0; l < w.layers.size(); l++) {
        const DraftLayerWeights& L = w.layers[l];
        // attention
        std::vector<float> h = x;
        rms_norm(h.data(), size_t(block), H, L.input_norm, w.rms_eps);
        std::vector<float> attn_keep;
        if (L.attn_conv.present()) h = conv_prepare(w, L.attn_conv, h, block, attn_keep);

        std::vector<float> q = linear(L.q, Q, H, h.data(), block);
        std::vector<float> kb = linear(L.k, KV, H, h.data(), block);
        const std::vector<float> vb = linear(L.v, KV, H, h.data(), block);
        rms_norm(q.data(), size_t(block) * size_t(w.heads), D, L.q_norm, w.rms_eps);
        rms_norm(kb.data(), size_t(block) * size_t(w.kv_heads), D, L.k_norm, w.rms_eps);
        for (int i = 0; i < block; i++) {
            for (int hd = 0; hd < w.heads; hd++)
                rope(q.data() + size_t(i) * size_t(Q) + size_t(hd) * size_t(D), D, pos0 + ctx + i, w.rope_theta);
            for (int hd = 0; hd < w.kv_heads; hd++)
                rope(kb.data() + size_t(i) * size_t(KV) + size_t(hd) * size_t(D), D, pos0 + ctx + i, w.rope_theta);
        }
        // keys and values: the cached context rows, then the block's own
        const auto key_at = [&](int t) {
            return t < ctx ? c.k[l].data() + size_t(t) * size_t(KV) : kb.data() + size_t(t - ctx) * size_t(KV);
        };
        const auto val_at = [&](int t) {
            return t < ctx ? c.v[l].data() + size_t(t) * size_t(KV) : vb.data() + size_t(t - ctx) * size_t(KV);
        };

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
                    const float* kvp = key_at(t) + size_t(kh) * size_t(D);
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
                    for (int t = 0; t < keys; t++) acc += score[size_t(t)] * double(val_at(t)[size_t(kh) * size_t(D) + size_t(d)]);
                    out[d] = float(acc / sum);
                }
            }
        }
        std::vector<float> o = linear(L.o, H, Q, attn.data(), block);
        if (L.attn_conv.present()) o = conv_finish(w, L.attn_conv, o, block, attn_keep);
        for (size_t j = 0; j < x.size(); j++) x[j] += o[j];

        // MLP
        std::vector<float> p = x;
        rms_norm(p.data(), size_t(block), H, L.post_norm, w.rms_eps);
        std::vector<float> mlp_keep;
        if (L.mlp_conv.present()) p = conv_prepare(w, L.mlp_conv, p, block, mlp_keep);
        std::vector<float> g = linear(L.gate, I, H, p.data(), block);
        std::vector<float> u = linear(L.up, I, H, p.data(), block);
        for (size_t j = 0; j < g.size(); j++) {
            const double a = g[j];
            g[j] = float(a / (1.0 + std::exp(-a)) * double(u[j]));   // silu(gate) * up
        }
        std::vector<float> dn = linear(L.down, H, I, g.data(), block);
        if (L.mlp_conv.present()) dn = conv_finish(w, L.mlp_conv, dn, block, mlp_keep);
        for (size_t j = 0; j < x.size(); j++) x[j] += dn[j];
    }

    rms_norm(x.data(), size_t(block), H, w.norm, w.rms_eps);
    for (float f : x) need(std::isfinite(f), "a non-finite output");
    return x;
}

}  // namespace onebit::npu::dflash
