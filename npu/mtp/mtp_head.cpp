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

// mtp_head.cpp -- loader + the head's own maths (see mtp_head.h for what it is and why it is in this form).
#include "mtp_head.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace onebit::npu::mtp {

bool g_fc_hidden_first = false;

namespace {

uint16_t rd16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }

float bf16(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

bool slurp(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return !out.empty();
}

// One matrix from <stem>.pb16i + .add + .eref, in the sealed tile order ([c][ki][nt][r], 4,608 B per tile).
bool load_matrix(const std::string& dir, const std::string& stem, int K, int N, Matrix& m, std::string& err) {
    std::vector<uint8_t> img;
    if (!slurp(dir + "/" + stem + ".pb16i", img)) { err = "cannot read " + stem + ".pb16i"; return false; }
    std::vector<uint8_t> addraw;
    if (!slurp(dir + "/" + stem + ".add", addraw)) { err = "cannot read " + stem + ".add"; return false; }
    std::vector<uint8_t> erefraw;
    if (!slurp(dir + "/" + stem + ".eref", erefraw)) { err = "cannot read " + stem + ".eref"; return false; }
    const int KT = K / 256, NT = N / (8 * 4 * 32);
    const size_t want = (size_t)8 * KT * NT * 4 * 4608;
    if (img.size() != want) {
        err = stem + ": image is " + std::to_string(img.size()) + " B, want " + std::to_string(want);
        return false;
    }
    if (addraw.size() != (size_t)(K / kG) * N * sizeof(float)) { err = stem + ": .add size mismatch"; return false; }
    if (erefraw.size() != (size_t)N * sizeof(int16_t)) { err = stem + ": .eref size mismatch"; return false; }
    m.K = K;
    m.N = N;
    m.codes.assign((size_t)K * N, 0);
    m.ms.assign((size_t)(K / kG) * N, 0);
    m.eref.resize((size_t)N);
    m.add.resize((size_t)(K / kG) * N);
    std::memcpy(m.eref.data(), erefraw.data(), erefraw.size());
    std::memcpy(m.add.data(), addraw.data(), addraw.size());

    // Index formulas taken line for line from the tested Python decoder (tools/mtp_head_ref.py
    // unpack_tiles): tile = ((c*KT + ki)*NT + nt)*4 + r, and inside a tile
    //   code byte  = ((kb*2 + nb)*16 + rr)*8 + cc/2   (low nibble = even cc, high = odd cc)
    //   k          = ki*256 + kb*16 + rr
    //   n          = ((c*4 + r)*NT + nt)*32 + nb*16 + cc        (shard-major columns)
    //   magnitude  = mags[(nb*8 + g)*16 + cc] for group ki*8 + g
    const uint8_t* img_p = img.data();
    for (int c = 0; c < 8; c++)
        for (int ki = 0; ki < KT; ki++)
            for (int nt = 0; nt < NT; nt++)
                for (int r = 0; r < 4; r++) {
                    const size_t tile = (((size_t)c * KT + ki) * NT + nt) * 4 + r;
                    const uint8_t* t = img_p + tile * 4608;
                    const int16_t* mags = (const int16_t*)(t + 4096);
                    for (int kb = 0; kb < 16; kb++)
                        for (int nb = 0; nb < 2; nb++)
                            for (int rr = 0; rr < 16; rr++) {
                                const int k = ki * 256 + kb * 16 + rr;
                                for (int cc = 0; cc < 16; cc++) {
                                    const int n = ((c * 4 + r) * NT + nt) * 32 + nb * 16 + cc;
                                    const uint8_t byte = t[((kb * 2 + nb) * 16 + rr) * 8 + (cc / 2)];
                                    const uint8_t v = (cc & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0xF);
                                    m.codes[(size_t)k * N + n] = (int8_t)(v >= 8 ? (int)v - 16 : (int)v);
                                }
                            }
                    for (int nb = 0; nb < 2; nb++)
                        for (int g = 0; g < 8; g++)
                            for (int cc = 0; cc < 16; cc++) {
                                const int gg = ki * 8 + g;
                                const int n = ((c * 4 + r) * NT + nt) * 32 + nb * 16 + cc;
                                m.ms[(size_t)gg * N + n] = mags[(nb * 8 + g) * 16 + cc];
                            }
                }
    return true;
}

inline float rms_norm_inplace(float* x, int n, const float* w) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float r = 1.0f / std::sqrt((float)(ss / n) + 1e-6f);
    for (int i = 0; i < n; i++) x[i] = x[i] * r * w[i];
    return r;
}

void rope_partial(float* x, int pos, int rot) {
    const int half = rot / 2;
    for (int i = 0; i < half; i++) {
        const double th = std::pow(1e7, -2.0 * i / (double)rot);
        const double ang = pos * th;
        const float c = (float)std::cos(ang), s = (float)std::sin(ang);
        const float a = x[i], b = x[i + half];
        x[i] = a * c - b * s;
        x[i + half] = b * c + a * s;
    }
}

}  // namespace

void Matrix::matvec(const float* a, float* y) const {
    const int G = K / kG;
    std::vector<double> acc((size_t)N, 0.0);
#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        double acc_n = 0.0;
        for (int g = 0; g < G; g++) {
            const int8_t* cg = &codes[(size_t)(g * kG) * N + n];
            double P = 0.0;
            for (int k = 0; k < kG; k++) P += (double)a[g * kG + k] * (double)cg[(size_t)k * N];
            acc_n += P * ((double)ms[(size_t)g * N + n] * std::ldexp(1.0, (int)eref[(size_t)n]));
        }
        double ag_sum = 0.0;
        for (int g = 0; g < G; g++) {
            double s = 0.0;
            for (int k = 0; k < kG; k++) s += a[g * kG + k];
            acc_n += s * (double)add[(size_t)g * N + n];
            ag_sum += s;
        }
        (void)ag_sum;
        acc[(size_t)n] = acc_n;
    }
    for (int n = 0; n < N; n++) y[n] = (float)acc[(size_t)n];
}

bool load_weights(const std::string& exact_dir, const std::string& q4k_dir, Weights& w, std::string& err) {
    struct Ent { const char* stem; int K, N; Matrix* m; };
    const Ent ents[] = {
        {"fc", 10240, 5120, &w.fc},  {"q", 5120, 12288, &w.q},   {"k", 5120, 1024, &w.k},
        {"v", 5120, 1024, &w.v},     {"o", 6144, 5120, &w.o},    {"gate", 5120, 17408, &w.gate},
        {"up", 5120, 17408, &w.up},  {"down", 17408, 5120, &w.down},
    };
    for (const auto& e : ents)
        if (!load_matrix(exact_dir, e.stem, e.K, e.N, *e.m, err)) return false;

    // Norms come from the q4_k artifact (BF16, plain tensors in the container config order).
    std::vector<uint8_t> blob;
    if (!slurp(q4k_dir + "/model.q4nx", blob)) { err = "cannot read q4_k model.q4nx"; return false; }
    // [u64 header_len][json index][cells]; read the index to find each norm's offset.
    uint64_t hlen = 0;
    std::memcpy(&hlen, blob.data(), 8);
    const std::string index((const char*)blob.data() + 8, (size_t)hlen);
    const size_t base = 8 + (size_t)hlen;
    auto load_norm = [&](const char* tensor, std::vector<float>& out, int n) {
        const std::string key = std::string("\"") + tensor + "\"";
        const size_t at = index.find(key);
        if (at == std::string::npos) return false;
        const size_t off_at = index.find("data_offsets", at);
        if (off_at == std::string::npos) return false;
        size_t p = index.find('[', off_at);
        const long long o0 = std::strtoll(index.c_str() + p + 1, nullptr, 10);
        const uint8_t* src = blob.data() + base + (size_t)o0;
        out.resize((size_t)n);
        for (int i = 0; i < n; i++) out[(size_t)i] = bf16(rd16(src + 2 * i));
        return true;
    };
    if (!load_norm("model.mtp.pre_fc_norm_embedding.weight", w.pre_fc_emb, kH) ||
        !load_norm("model.mtp.pre_fc_norm_hidden.weight", w.pre_fc_hid, kH) ||
        !load_norm("model.mtp.layers.0.input_layernorm.weight", w.ln_in, kH) ||
        !load_norm("model.mtp.layers.0.post_attention_layernorm.weight", w.ln_post, kH) ||
        !load_norm("model.mtp.norm.weight", w.norm, kH) ||
        !load_norm("model.mtp.layers.0.self_attn.q_norm.weight", w.q_norm, kHD) ||
        !load_norm("model.mtp.layers.0.self_attn.k_norm.weight", w.k_norm, kHD)) {
        err = "q4_k: a norm tensor is missing";
        return false;
    }
    return true;
}

bool head_step(const Weights& w, const float* hidden, const float* tok_emb, int pos, KvCache& cache,
               std::vector<float>& out_hidden, const std::function<bool(const std::vector<float>&,
                                                                      std::vector<float>&)>& lm_head,
               std::vector<float>& logits_out) {
    // input fusion: [embedding block | hidden block], each RMS-normed by its own pre_fc_norm
    std::vector<float> u((size_t)2 * kH);
    const float* first = g_fc_hidden_first ? hidden : tok_emb;
    const float* second = g_fc_hidden_first ? tok_emb : hidden;
    const float* w_first = g_fc_hidden_first ? w.pre_fc_hid.data() : w.pre_fc_emb.data();
    const float* w_second = g_fc_hidden_first ? w.pre_fc_emb.data() : w.pre_fc_hid.data();
    std::memcpy(u.data(), first, sizeof(float) * kH);
    std::memcpy(u.data() + kH, second, sizeof(float) * kH);
    rms_norm_inplace(u.data(), kH, w_first);
    rms_norm_inplace(u.data() + kH, kH, w_second);
    std::vector<float> y((size_t)kH);
    w.fc.matvec(u.data(), y.data());

    std::vector<float> hn(y);
    rms_norm_inplace(hn.data(), kH, w.ln_in.data());
    std::vector<float> qg((size_t)2 * kDQ), kk((size_t)kKV), vv((size_t)kKV);
    w.q.matvec(hn.data(), qg.data());
    w.k.matvec(hn.data(), kk.data());
    w.v.matvec(hn.data(), vv.data());

    // QK-norm per head, then partial rope on the first 64 of 256
    for (int h = 0; h < kNH; h++) rms_norm_inplace(&qg[(size_t)h * kHD], kHD, w.q_norm.data());
    for (int h = 0; h < kNKV; h++) rms_norm_inplace(&kk[(size_t)h * kHD], kHD, w.k_norm.data());
    for (int h = 0; h < kNH; h++) rope_partial(&qg[(size_t)h * kHD], pos, kRot);
    for (int h = 0; h < kNKV; h++) rope_partial(&kk[(size_t)h * kHD], pos, kRot);

    // append to the cache
    const int t_prev = cache.t;
    cache.k.resize((size_t)(t_prev + 1) * kKV);
    cache.v.resize((size_t)(t_prev + 1) * kKV);
    std::memcpy(&cache.k[(size_t)t_prev * kKV], kk.data(), sizeof(float) * kKV);
    std::memcpy(&cache.v[(size_t)t_prev * kKV], vv.data(), sizeof(float) * kKV);
    cache.t = t_prev + 1;

    const float scale = 1.0f / std::sqrt((float)kHD);
    std::vector<float> attn((size_t)kDQ);
    const int rep = kNH / kNKV;
    for (int h = 0; h < kNH; h++) {
        const int kh = h / rep;
        float mx = -1e30f;
        std::vector<float> sc((size_t)cache.t);
        for (int t = 0; t < cache.t; t++) {
            const float* kt = &cache.k[((size_t)t * kNKV + kh) * kHD];
            double s = 0;
            for (int d = 0; d < kHD; d++) s += (double)kt[d] * qg[(size_t)h * kHD + d];
            sc[(size_t)t] = (float)s * scale;
            if (sc[(size_t)t] > mx) mx = sc[(size_t)t];
        }
        double sum = 0;
        for (int t = 0; t < cache.t; t++) { sc[(size_t)t] = std::exp(sc[(size_t)t] - mx); sum += sc[(size_t)t]; }
        for (int d = 0; d < kHD; d++) {
            double acc = 0;
            for (int t = 0; t < cache.t; t++)
                acc += (double)sc[(size_t)t] * cache.v[((size_t)t * kNKV + kh) * kHD + d];
            attn[(size_t)h * kHD + d] = (float)(acc / sum);
        }
    }
    // the output gate is the second half of the q projection
    for (int i = 0; i < kDQ; i++)
        attn[(size_t)i] *= 1.0f / (1.0f + std::exp(-qg[(size_t)kDQ + i]));
    std::vector<float> o((size_t)kH);
    w.o.matvec(attn.data(), o.data());
    for (int i = 0; i < kH; i++) y[(size_t)i] += o[(size_t)i];

    std::vector<float> hp(y);
    rms_norm_inplace(hp.data(), kH, w.ln_post.data());
    std::vector<float> g((size_t)kIM), up((size_t)kIM);
    w.gate.matvec(hp.data(), g.data());
    w.up.matvec(hp.data(), up.data());
    for (int i = 0; i < kIM; i++) g[(size_t)i] = (g[(size_t)i] / (1.0f + std::exp(-g[(size_t)i]))) * up[(size_t)i];
    std::vector<float> d((size_t)kH);
    w.down.matvec(g.data(), d.data());
    for (int i = 0; i < kH; i++) y[(size_t)i] += d[(size_t)i];

    out_hidden = y;
    std::vector<float> yf(y);
    rms_norm_inplace(yf.data(), kH, w.norm.data());
    if (lm_head) return lm_head(yf, logits_out);
    logits_out.clear();
    return true;
}

}  // namespace onebit::npu::mtp
