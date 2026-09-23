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

#include "pack.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace onebit::npu {

uint16_t f32_to_bf16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    return uint16_t((bits + (((bits >> 16) & 1) + 0x7FFF)) >> 16);
}

float bf16_to_f32(uint16_t v) {
    const uint32_t bits = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

void reorder_tiles(uint8_t* dst, const uint8_t* src, int n_tiles, int G) {
    // S = ceil(G/2), and the in-group position is o % G, so the map is a
    // permutation for odd G too; for even G both reduce to G/2 and raw o.
    const int S = (G + 1) / 2;
    for (int o = 0; o < n_tiles; ++o) {
        const int og = o % G;
        const int i = G * (o / G) + (og / 2) % S + S * (og % 2);
        std::memcpy(dst + size_t(o) * kTileBytes, src + size_t(i) * kTileBytes, kTileBytes);
    }
}

namespace {

int groups(int k) { return (k + 127) / 128; }  // 128-wide contraction groups, ceil

const char* const kProj[] = {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                             "self_attn.o_proj.weight", "mlp.up_proj.weight",      "mlp.gate_proj.weight",
                             "mlp.down_proj.weight"};

}  // namespace

size_t layer_weight_bytes(const Model& m) {
    int most = 0;
    for (int L = 0; L < m.dims().layers; ++L) {
        int t = 0;
        for (const char* p : kProj) t += m.layer(L, p).tiles();
        most = std::max(most, t);
    }
    return size_t(most) * kTileBytes;
}

int pack_layer_weights(const Model& m, int L, uint8_t* dst) {
    const Dims& d = m.dims();
    const Tensor &q = m.layer(L, kProj[0]), &k = m.layer(L, kProj[1]), &v = m.layer(L, kProj[2]),
                 &o = m.layer(L, kProj[3]), &up = m.layer(L, kProj[4]), &gate = m.layer(L, kProj[5]),
                 &down = m.layer(L, kProj[6]);
    const int G_h = groups(d.hidden), G_o = groups(d.heads * d.head_dim), G_d = groups(d.intermediate);
    const int CH = d.hidden / 16;  // gate/up chunk, 8 * G_h

    const int off_q = 0, off_k = off_q + q.tiles(), off_v = off_k + k.tiles(), off_o = off_v + v.tiles(),
              off_gu = off_o + o.tiles(), off_d = off_gu + up.tiles() + gate.tiles(), total = off_d + down.tiles();
    std::memset(dst, 0, size_t(total) * kTileBytes);
    auto proj = [&](const Tensor& t, int off, int G) { reorder_tiles(dst + size_t(off) * kTileBytes, t.data, t.tiles(), G); };
    proj(q, off_q, G_h);
    proj(k, off_k, G_h);
    proj(v, off_v, G_h);
    proj(o, off_o, G_o);
    // gate/up: alternating CH-tile chunks, up first.
    const int up_t = up.tiles(), gate_t = gate.tiles();
    for (int c = 0; c < (up_t + CH - 1) / CH; ++c) {
        const int up_n = std::min(CH, up_t - c * CH), gate_n = std::min(CH, gate_t - c * CH);
        const size_t base = size_t(off_gu + c * 2 * CH);
        if (up_n > 0) reorder_tiles(dst + base * kTileBytes, up.data + size_t(c) * CH * kTileBytes, up_n, G_h);
        if (gate_n > 0)
            reorder_tiles(dst + (base + CH) * kTileBytes, gate.data + size_t(c) * CH * kTileBytes, gate_n, G_h);
    }
    proj(down, off_d, G_d);
    return total;
}

size_t lmhead_weight_bytes(const Model& m) { return size_t(m.tensor("lm_head.weight").tiles()) * kTileBytes; }

void pack_lmhead_weights(const Model& m, uint8_t* dst) {
    const Tensor& t = m.tensor("lm_head.weight");
    reorder_tiles(dst, t.data, t.tiles(), m.dims().hidden / 128);
}

void fill_i5(const Model& m, int L, uint8_t* dst) {
    const size_t H2 = size_t(m.dims().hidden) * 2;
    std::memcpy(dst, m.layer(L, "input_layernorm.weight").data, H2);
    std::memcpy(dst + H2, m.layer(L, "post_attention_layernorm.weight").data, H2);
}

void fill_i6(const Model& m, int L, uint8_t* dst) {
    std::memset(dst, 0, kNormBytes);
    fill_rope(reinterpret_cast<uint16_t*>(dst), 0, m.dims().rope_theta);
    // Models without q/k norms leave those slots zero, as the reference runtime does.
    if (const Tensor* qn = m.find_layer(L, "self_attn.q_norm.weight")) std::memcpy(dst + 256, qn->data, 256);
    if (const Tensor* kn = m.find_layer(L, "self_attn.k_norm.weight")) std::memcpy(dst + 512, kn->data, 256);
}

// inv_freq for theta = 1e6, as float32 literals. The reference runtime keeps this
// table rather than computing it, and the last bits matter: computing it in
// double flips i6 values at positions >= 3.
static const float kInvFreq1e6[64] = {
    1.000000000e+00f, 8.058400154e-01f, 6.493800282e-01f, 5.232999921e-01f, 4.217000008e-01f, 3.398199975e-01f,
    2.738400102e-01f, 2.206699997e-01f, 1.778299958e-01f, 1.432999969e-01f, 1.154799983e-01f, 9.305699915e-02f,
    7.498899847e-02f, 6.043000147e-02f, 4.869699851e-02f, 3.924199939e-02f, 3.162299842e-02f, 2.548299916e-02f,
    2.053499967e-02f, 1.654800028e-02f, 1.333499979e-02f, 1.074600033e-02f, 8.659600280e-03f, 6.978299934e-03f,
    5.623400211e-03f, 4.531600047e-03f, 3.651699983e-03f, 2.942699939e-03f, 2.371399896e-03f, 1.910999999e-03f,
    1.539899968e-03f, 1.240900019e-03f, 1.000000047e-03f, 8.058400126e-04f, 6.493799738e-04f, 5.233000265e-04f,
    4.217000096e-04f, 3.398199915e-04f, 2.738400071e-04f, 2.206700010e-04f, 1.778300066e-04f, 1.432999998e-04f,
    1.154799975e-04f, 9.305700223e-05f, 7.498900231e-05f, 6.043000030e-05f, 4.869699842e-05f, 3.924200064e-05f,
    3.162299981e-05f, 2.548299926e-05f, 2.053500066e-05f, 1.654799962e-05f, 1.333500040e-05f, 1.074600004e-05f,
    8.659600098e-06f, 6.978300007e-06f, 5.623399829e-06f, 4.531600098e-06f, 3.651699899e-06f, 2.942699894e-06f,
    2.371399887e-06f, 1.911000027e-06f, 1.539900040e-06f, 1.240900019e-06f,
};

void fill_rope(uint16_t* dst, int pos, float theta) {
    const float fpos = float(pos);
    for (int j = 0; j < 64; ++j) {
        const float inv = theta == 1000000.0f ? kInvFreq1e6[j] : ::powf(theta, -2.0f * float(j) / 128.0f);
        const float phi = inv * fpos;  // float32 multiply
        float s, c;
        ::sincosf(phi, &s, &c);
        dst[j] = f32_to_bf16(c);
        dst[64 + j] = f32_to_bf16(s);
    }
}

}  // namespace onebit::npu
