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

#include "embed_model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>

namespace onebit::embed {
namespace {

constexpr int64_t kHeaderPad = 16384;
constexpr const char* kFormat = "1bit-embed-1";

// ------------------------------------------------------------------ container

struct Reader {
    std::ifstream f;
    nlohmann::json hdr;
    int64_t size = 0;
};

bool open_container(const std::string& path, Reader& r, std::string& why) {
    r.f.open(path, std::ios::binary);
    if (!r.f) {
        why = "cannot open " + path;
        return false;
    }
    uint64_t n = 0;
    r.f.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (n == 0 || n > uint64_t(kHeaderPad)) {
        why = path + ": header length " + std::to_string(n) + " is not inside the 16 KiB header";
        return false;
    }
    std::string h(n, '\0');
    r.f.read(h.data(), static_cast<std::streamsize>(n));
    try {
        r.hdr = nlohmann::json::parse(h);
    } catch (const std::exception& e) {
        why = path + ": header is not JSON: " + e.what();
        return false;
    }
    if (r.hdr.value("format", std::string()) != kFormat) {
        why = path + ": format " + r.hdr.value("format", std::string("?")) + ", this build reads " + kFormat;
        return false;
    }
    r.f.seekg(0, std::ios::end);
    r.size = static_cast<int64_t>(r.f.tellg());
    return true;
}

// Read one tensor. `want` is "i8" or "f32"; the shape must match `count` elements.
bool read_tensor(Reader& r, const std::string& name, const char* want, int64_t count, void* dst,
                 std::string& why) {
    const auto it = r.hdr["tensors"].find(name);
    if (it == r.hdr["tensors"].end()) {
        why = "container is missing tensor '" + name + "'";
        return false;
    }
    const auto& e = *it;
    if (e.value("dtype", std::string()) != want) {
        why = "tensor '" + name + "' is " + e.value("dtype", std::string("?")) + ", expected " + want;
        return false;
    }
    int64_t elems = 1;
    for (const auto& d : e["shape"]) elems *= d.get<int64_t>();
    if (elems != count) {
        why = "tensor '" + name + "' has " + std::to_string(elems) + " elements, expected " +
              std::to_string(count);
        return false;
    }
    const int64_t lo = e["offsets"][0].get<int64_t>(), hi = e["offsets"][1].get<int64_t>();
    if (lo < kHeaderPad || hi > r.size || hi - lo != count * int64_t(want[0] == 'i' ? 1 : 4)) {
        why = "tensor '" + name + "' has offsets outside its payload";
        return false;
    }
    r.f.seekg(lo);
    r.f.read(static_cast<char*>(dst), static_cast<std::streamsize>(hi - lo));
    return static_cast<bool>(r.f);
}

// ------------------------------------------------------------------ math (the reference's formulas)

// `quant_act`: int8 with a power-of-two scale per token, or per 256-channel block when the width is a multiple
// of 256 and wider than it; round half to even; clamp +-127; return the dequantised value, as the reference does.
void quant_act(std::vector<float>& x, int width) {
    const int rows = static_cast<int>(x.size()) / width;
    const bool grouped = width > 256 && width % 256 == 0;
    const int group = grouped ? 256 : width;
    for (int r = 0; r < rows; ++r) {
        for (int g = 0; g < width; g += group) {
            float m = 1e-5f;
            for (int k = g; k < g + group; ++k) m = std::fmax(m, std::fabs(x[size_t(r) * width + k]));
            int e = 0;
            const float f = std::frexp(m, &e);
            const int carry = (2.0f * f > 127.0f / 64.0f) ? 1 : 0;
            const float s = std::ldexp(1.0f, e - 7 + carry);
            for (int k = g; k < g + group; ++k) {
                float& v = x[size_t(r) * width + k];
                const float q = std::nearbyint(v / s);  // FE_TONEAREST: half to even, like torch.round
                v = std::fmin(127.0f, std::fmax(-128.0f, q)) * s;
            }
        }
    }
}

void layer_norm(std::vector<float>& x, int width, const std::vector<float>& w, const std::vector<float>& b,
                float eps) {
    const int rows = static_cast<int>(x.size()) / width;
    for (int r = 0; r < rows; ++r) {
        float* p = x.data() + size_t(r) * width;
        double mean = 0;
        for (int k = 0; k < width; ++k) mean += p[k];
        mean /= width;
        double var = 0;
        for (int k = 0; k < width; ++k) {
            const double d = p[k] - mean;
            var += d * d;
        }
        var /= width;
        const float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
        for (int k = 0; k < width; ++k) p[k] = static_cast<float>((p[k] - mean) * inv) * w[k] + b[k];
    }
}

// The reference's activation: quantise (int8, power-of-two per token), then the ternary matmul, then the bias.
// code is int8 {-1,0,+1}, so the dot is a signed sum; the per-output-row scale comes out of the sum.
void tern_linear(const Tern& t, std::vector<float>& x, int width) {
    const int rows = static_cast<int>(x.size()) / width;
    quant_act(x, width);
    std::vector<float> out(size_t(rows) * t.n);
    for (int r = 0; r < rows; ++r) {
        const float* a = x.data() + size_t(r) * width;
        for (int n = 0; n < t.n; ++n) {
            const int8_t* c = t.code.data() + size_t(n) * t.k;
            float acc = 0.0f;
            for (int k = 0; k < t.k; ++k) {
                const int8_t v = c[k];
                if (v == 1) acc += a[k];
                else if (v == -1) acc -= a[k];
            }
            out[size_t(r) * t.n + n] = acc * t.scale[size_t(n)] + t.bias[size_t(n)];
        }
    }
    x.swap(out);
}

inline float gelu_tanh(float v) {
    return 0.5f * v * (1.0f + std::tanh(std::sqrt(2.0f / 3.14159265358979323846f) *
                                       (v + 0.044715f * v * v * v)));
}

}  // namespace

bool Model::load(const std::string& dir, std::string& why) {
    Reader r;
    if (!open_container(dir + "/model.embed", r, why)) return false;
    const auto& d = r.hdr["dims"];
    cfg_.vocab = d.value("vocab", 0);
    cfg_.d = d.value("emb", d.value("d", 0));
    cfg_.heads = d.value("heads", 0);
    cfg_.ffn = d.value("ffn", 0);
    cfg_.layers = d.value("layers", 0);
    cfg_.max_pos = d.value("max_pos", 0);
    cfg_.out_dim = d.value("out_dim", cfg_.d);
    cfg_.eps = r.hdr.value("eps", 1e-12f);
    cfg_.pooling = r.hdr.value("pooling", std::string("mean"));
    cfg_.normalized = r.hdr.value("normalized", true);
    cfg_.max_tokens = r.hdr.value("max_tokens", cfg_.max_pos);
    if (cfg_.vocab <= 0 || cfg_.d <= 0 || cfg_.heads <= 0 || cfg_.ffn <= 0 || cfg_.layers <= 0 ||
        cfg_.max_pos <= 0 || cfg_.d % cfg_.heads != 0) {
        why = "model.embed: dims are not usable (d " + std::to_string(cfg_.d) + ", heads " +
              std::to_string(cfg_.heads) + ")";
        return false;
    }
    if (cfg_.pooling != "mean" || !cfg_.normalized) {
        why = "model.embed: pooling '" + cfg_.pooling + "' / normalized " +
              (cfg_.normalized ? "true" : "false") + " are not the contract this encoder implements";
        return false;
    }
    const int64_t V = cfg_.vocab, D = cfg_.d, F = cfg_.ffn;
    word_q_.resize(V * D);
    pos_q_.resize(int64_t(cfg_.max_pos) * D);
    word_s_.resize(V);
    pos_s_.resize(cfg_.max_pos);
    tok_type_.resize(2 * D);
    ln_e_w_.resize(D);
    ln_e_b_.resize(D);
    out_w_.resize(int64_t(cfg_.out_dim) * D);
    if (!read_tensor(r, "word.qweight", "i8", V * D, word_q_.data(), why) ||
        !read_tensor(r, "pos.qweight", "i8", int64_t(cfg_.max_pos) * D, pos_q_.data(), why) ||
        !read_tensor(r, "word.scale", "f32", V, word_s_.data(), why) ||
        !read_tensor(r, "pos.scale", "f32", cfg_.max_pos, pos_s_.data(), why) ||
        !read_tensor(r, "tok_type.weight", "f32", 2 * D, tok_type_.data(), why) ||
        !read_tensor(r, "ln_e.weight", "f32", D, ln_e_w_.data(), why) ||
        !read_tensor(r, "ln_e.bias", "f32", D, ln_e_b_.data(), why) ||
        !read_tensor(r, "out.weight", "f32", int64_t(cfg_.out_dim) * D, out_w_.data(), why))
        return false;
    layers_.resize(cfg_.layers);
    for (int i = 0; i < cfg_.layers; ++i) {
        Layer& L = layers_[i];
        const std::string p = "layers." + std::to_string(i) + ".";
        auto tern = [&](const char* name, Tern& t, int64_t n, int64_t k) {
            t.n = int(n);
            t.k = int(k);
            t.code.resize(n * k);
            t.scale.resize(n);
            t.bias.resize(n);
            return read_tensor(r, p + name + ".weight", "i8", n * k, t.code.data(), why) &&
                   read_tensor(r, p + name + ".scale", "f32", n, t.scale.data(), why) &&
                   read_tensor(r, p + name + ".bias", "f32", n, t.bias.data(), why);
        };
        auto vec = [&](const char* name, std::vector<float>& v) {
            v.resize(D);
            return read_tensor(r, p + name, "f32", D, v.data(), why);
        };
        if (!tern("q", L.q, D, D) || !tern("k", L.k, D, D) || !tern("v", L.v, D, D) ||
            !tern("o", L.o, D, D) || !tern("fc1", L.fc1, F, D) || !tern("fc2", L.fc2, D, F) ||
            !vec("ln1.weight", L.ln1_w) || !vec("ln1.bias", L.ln1_b) || !vec("ln2.weight", L.ln2_w) ||
            !vec("ln2.bias", L.ln2_b))
            return false;
    }
    if (!tok_.load_vocab(dir + "/vocab.txt")) {
        why = dir + "/vocab.txt: cannot load the WordPiece vocabulary";
        return false;
    }
    tok_vocab_ = tok_.vocab_size();
    if (tok_vocab_ != cfg_.vocab) {
        why = "vocab.txt has " + std::to_string(tok_vocab_) + " pieces, model.embed declares " +
              std::to_string(cfg_.vocab);
        return false;
    }
    return true;
}

std::vector<int> Model::tokenize(const std::string& text) const {
    return tok_.encode(text, cfg_.max_tokens, true);
}

std::vector<float> Model::encode_ids(const std::vector<int>& ids, std::string& why) const {
    const int D = cfg_.d, nh = cfg_.heads, hd = D / nh, T = static_cast<int>(ids.size());
    if (T <= 0) {
        why = "empty token sequence";
        return {};
    }
    if (T > cfg_.max_pos) {
        why = "sequence of " + std::to_string(T) + " tokens exceeds the model's " +
              std::to_string(cfg_.max_pos) + " positions";
        return {};
    }
    for (int id : ids) {
        if (id < 0 || id >= cfg_.vocab) {
            why = "token id " + std::to_string(id) + " outside the vocabulary";
            return {};
        }
    }
    // embeddings: int8 rows * per-row scale, plus tok_type row 0, then LN_e
    std::vector<float> x(size_t(T) * D);
    for (int t = 0; t < T; ++t) {
        const int8_t* wq = word_q_.data() + size_t(ids[t]) * D;
        const int8_t* pq = pos_q_.data() + size_t(t) * D;
        const float ws = word_s_[size_t(ids[t])], ps = pos_s_[size_t(t)];
        for (int k = 0; k < D; ++k)
            x[size_t(t) * D + k] = float(wq[k]) * ws + float(pq[k]) * ps + tok_type_[size_t(k)];
    }
    layer_norm(x, D, ln_e_w_, ln_e_b_, cfg_.eps);

    std::vector<float> q(size_t(T) * D), k_(size_t(T) * D), v(size_t(T) * D), att(size_t(T) * D),
        o(size_t(T) * D), tmp, prob;
    const float scale = 1.0f / std::sqrt(float(hd));
    for (const Layer& L : layers_) {
        q = x;
        tern_linear(L.q, q, D);
        k_ = x;
        tern_linear(L.k, k_, D);
        v = x;
        tern_linear(L.v, v, D);
        // attention: no mask needed -- one sentence, no padding, so every token attends to every token
        for (int h = 0; h < nh; ++h) {
            for (int t = 0; t < T; ++t) {
                prob.assign(T, 0.0f);
                float best = -INFINITY;
                for (int s = 0; s < T; ++s) {
                    float acc = 0.0f;
                    for (int c = 0; c < hd; ++c)
                        acc += q[size_t(t) * D + h * hd + c] * k_[size_t(s) * D + h * hd + c];
                    prob[s] = acc * scale;
                    if (prob[s] > best) best = prob[s];
                }
                float sum = 0.0f;
                for (int s = 0; s < T; ++s) {
                    prob[s] = std::exp(prob[s] - best);
                    sum += prob[s];
                }
                for (int c = 0; c < hd; ++c) {
                    float acc = 0.0f;
                    for (int s = 0; s < T; ++s) acc += prob[s] * v[size_t(s) * D + h * hd + c];
                    o[size_t(t) * D + h * hd + c] = acc / sum;
                }
            }
        }
        tmp = o;
        tern_linear(L.o, tmp, D);
        for (size_t i = 0; i < x.size(); ++i) x[i] += tmp[i];
        layer_norm(x, D, L.ln1_w, L.ln1_b, cfg_.eps);
        // MLP: fc1 -> GELU(tanh) -> fc2, each ternary linear quantising its own input
        tmp = x;
        tern_linear(L.fc1, tmp, D);
        for (float& val : tmp) val = gelu_tanh(val);
        tern_linear(L.fc2, tmp, cfg_.ffn);
        for (size_t i = 0; i < x.size(); ++i) x[i] += tmp[i];
        layer_norm(x, D, L.ln2_w, L.ln2_b, cfg_.eps);
    }
    // mean pool over the tokens (no padding in this path), output projection, L2
    std::vector<float> pooled(size_t(cfg_.out_dim), 0.0f);
    for (int t = 0; t < T; ++t)
        for (int n = 0; n < cfg_.out_dim; ++n) {
            float acc = 0.0f;
            const float* row = out_w_.data() + size_t(n) * D;
            for (int c = 0; c < D; ++c) acc += row[c] * x[size_t(t) * D + c];
            pooled[size_t(n)] += acc;
        }
    float norm = 0.0f;
    for (float& val : pooled) {
        val /= float(T);
        norm += val * val;
    }
    norm = std::sqrt(norm);
    if (!(norm > 0.0f)) {
        why = "the pooled vector has zero norm";
        return {};
    }
    for (float& val : pooled) val /= norm;
    return pooled;
}

}  // namespace onebit::embed
