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

// Built without floating-point contraction (CMakeLists.txt): requant_q8 must round every
// float32 step on its own, as the reference packer's NumPy does.
#include "lax_pack.h"

#include "q4nx.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace onebit::npu::lax {

namespace {

float f32_of_bf16(uint16_t v) {
    const uint32_t u = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

uint32_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// f32 -> bf16 toward -inf, toward +inf, and to nearest even.
uint16_t bf16_floor(float x) {
    const uint32_t u = bits(x);
    return uint16_t((u >> 31) ? (u + 0xFFFFu) >> 16 : u >> 16);
}
uint16_t bf16_ceil(float x) {
    const uint32_t u = bits(x);
    return uint16_t((u >> 31) ? u >> 16 : (u + 0xFFFFu) >> 16);
}
uint16_t bf16_rne(float x) {
    const uint32_t u = bits(x);
    return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

uint16_t rd16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
void wr16(uint8_t* p, uint16_t v) {
    p[0] = uint8_t(v);
    p[1] = uint8_t(v >> 8);
}

// Nibble / int8 code index of (row r, 32-column group g, lane i) inside a chunk.
constexpr size_t code_index(int r, int g, int i) { return size_t((r / 16) * 4096 + g * 512 + i * 16 + r % 16); }

std::string lname(int layer, const char* suffix) { return "model.layer." + std::to_string(layer) + "." + suffix; }

size_t chunk_bytes(const Tensor& t) {
    if (t.dtype != "I8" || t.shape.empty()) throw std::runtime_error("not a quantized tensor");
    return size_t(t.shape.back());
}

// Chunks [c0, c0 + n) of a quantized tensor as q4_1: native q4_1 in place, anything else
// converted into `store`. Mirrors the reference packer's source handling: q8 is
// re-quantized, Q4_K transcoded, and a 5120-byte tensor whose first selected chunk has
// every min zero is the signed-nibble form.
class Q41 {
public:
    Q41(const Tensor& t, const std::string& name, size_t c0, size_t n) {
        const size_t cb = chunk_bytes(t);
        if (cb != kQ41 && cb != kQ8 && cb != kQ4K)
            throw std::runtime_error(name + ": " + std::to_string(cb) + "-byte chunks; the packer reads 5120, 8704 and 4736");
        if ((c0 + n) * cb > t.bytes) throw std::runtime_error(name + ": too few chunks");
        const uint8_t* src = t.data + c0 * cb;
        if (cb == kQ41 && !is_signed_q4(src)) {
            base_ = src;
            return;
        }
        store_.resize(n * kQ41);
        for (size_t i = 0; i < n; ++i) {
            uint8_t* d = store_.data() + i * kQ41;
            if (cb == kQ8) requant_q8(src + i * cb, d);
            else if (cb == kQ4K) q4nx::q4k_to_q4_1(src + i * cb, d);
            else q4_0_to_q4_1(src + i * cb, d);
        }
        base_ = store_.data();
    }
    const uint8_t* operator[](size_t i) const { return base_ + i * kQ41; }

private:
    const uint8_t* base_ = nullptr;
    std::vector<uint8_t> store_;
};

// std band law: nch pool chunks of an [out, in_dim] tensor starting at source chunk c0.
void std_band(const Model& m, const std::string& name, size_t c0, size_t nch, size_t in_dim, uint8_t* dst) {
    if (in_dim % 256) throw std::runtime_error(name + ": in_dim is not whole 256-column tiles");
    const Tensor& t = m.tensor(name);
    const Q41 src(t, name, c0, nch);
    for (size_t c = 0; c < nch; ++c) std::memcpy(dst + c * kQ41, src[std_src(c, in_dim)], kQ41);
}

void put(const Model& m, const std::string& name, size_t cap, uint8_t* dst) {
    const Tensor& t = m.tensor(name);
    if (t.bytes > cap) throw std::runtime_error(name + ": does not fit its " + std::to_string(cap) + " B slot");
    std::memcpy(dst, t.data, t.bytes);
}

int need(const nlohmann::json& j, const char* key) {
    if (!j.contains(key)) throw std::runtime_error(std::string("config.json has no ") + key);
    return j.at(key).get<int>();
}

}  // namespace

// ---- the chunk transforms --------------------------------------------------------------

void requant_q8(const uint8_t* q8, uint8_t* q41) {
    std::memset(q41, 0, kQ41);
    for (int r = 0; r < 32; ++r) {
        for (int g = 0; g < 8; ++g) {
            const int meta = g * 32 + r;
            const float sc = f32_of_bf16(rd16(q8 + 2 * meta));
            float v[32];
            float mn = 0, mx = 0;
            for (int i = 0; i < 32; ++i) {
                v[i] = float(int8_t(q8[512 + code_index(r, g, i)])) * sc;  // exact: 8 x 8 significant bits
                // Ties go to the later element (x86 minps / maxps, as NumPy reduces): in a
                // block whose scale is zero every value is +0 or -0, and the min's sign is
                // stored in m.
                mn = i && mn < v[i] ? mn : v[i];
                mx = i && mx > v[i] ? mx : v[i];
            }
            const uint16_t mu = bf16_floor(mn);
            const float mf = f32_of_bf16(mu);
            const float range = mx - mf;
            const uint16_t du = bf16_ceil(range / 15.0f);
            const float d = f32_of_bf16(du);
            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
            wr16(q41 + 2 * meta, du);
            wr16(q41 + 512 + 2 * meta, mu);
            for (int i = 0; i < 32; ++i) {
                const float diff = v[i] - mf;
                const float scaled = diff * inv;
                const float t = scaled + 0.5f;
                const int q = std::clamp(int(t), 0, 15);  // truncation toward zero, then clip
                const size_t n = code_index(r, g, i);
                q41[1024 + n / 2] |= uint8_t((n & 1) ? q << 4 : q);
            }
        }
    }
}

bool is_signed_q4(const uint8_t* c) {
    for (size_t i = 512; i < 1024; ++i)
        if (c[i]) return false;
    return true;
}

void q4_0_to_q4_1(const uint8_t* s, uint8_t* d) {
    std::memcpy(d, s, 512);
    for (int k = 0; k < 256; ++k) wr16(d + 512 + 2 * k, bf16_rne(-8.0f * f32_of_bf16(rd16(s + 2 * k))));
    for (size_t i = 1024; i < kQ41; ++i) d[i] = uint8_t(s[i] ^ 0x88);
}

size_t std_src(size_t c, size_t in_dim) {
    // A band is 64 rows x in_dim: in_dim/128 chunks, chunk i of a band covers row half i%2
    // and k-tile i/2. The source is a plain raster of 32-row blocks x 256-column tiles.
    const size_t per_band = in_dim / 128;
    const size_t rb = 2 * (c / per_band) + c % 2, kt = (c % per_band) / 2;
    return rb * (in_dim / 256) + kt;
}

size_t down_src(size_t c) { return 2 * (4 * (c / 8) + c % 4) + (c / 4) % 2; }

size_t stripe_src(size_t c) { return 8 * (c % 4) + c / 4; }  // in_dim 2048: 8 k-tiles per row block

size_t lmhead_src(size_t k, size_t hidden) {
    // 128-row supertiles: pool chunk c of a band is (row quarter c%4, k-tile c/4).
    const size_t nk = hidden / 256, per_band = 4 * nk;
    const size_t s = k / per_band, r = k % per_band;
    return (4 * s + r % 4) * nk + r / 4;
}

// ---- configuration ---------------------------------------------------------------------

Config Config::from_model(const Model& m, const std::string& dir) {
    std::ifstream f(dir + "/config.json");
    if (!f) throw std::runtime_error("cannot read " + dir + "/config.json");
    const auto j = nlohmann::json::parse(f);
    const auto& t = j.contains("text_config") ? j.at("text_config") : j;
    Config c;
    c.hidden = need(t, "hidden_size");
    c.layers = need(t, "num_hidden_layers");
    c.heads = need(t, "num_attention_heads");
    c.kv_heads = need(t, "num_key_value_heads");
    c.head_dim = need(t, "head_dim");
    c.experts = need(t, "num_experts");
    c.topk = need(t, "num_experts_per_tok");
    c.moe_inter = need(t, "moe_intermediate_size");
    c.shared_inter = need(t, "shared_expert_intermediate_size");
    c.lin_key_heads = need(t, "linear_num_key_heads");
    c.lin_value_heads = need(t, "linear_num_value_heads");
    c.lin_key_dim = need(t, "linear_key_head_dim");
    c.lin_value_dim = need(t, "linear_value_head_dim");
    c.conv_kernel = need(t, "linear_conv_kernel_dim");
    const auto& rp = t.contains("rope_parameters") ? t.at("rope_parameters") : t;
    c.rope_theta = rp.value("rope_theta", t.value("rope_theta", 0.0));
    const double prf = rp.value("partial_rotary_factor", t.value("partial_rotary_factor", 1.0));
    c.rotary_dim = int(c.head_dim * prf);
    for (const auto& lt : t.at("layer_types")) {
        const auto s = lt.get<std::string>();
        if (s == "linear_attention") c.kinds.push_back(Kind::Linear);
        else if (s == "full_attention") c.kinds.push_back(Kind::Full);
        else throw std::runtime_error("layer type " + s + " has no lax kernel");
    }
    c.vocab = int(m.tensor("model.embed_tokens.weight").shape.at(0));

    // The kernels are compiled for exactly this geometry (recipes/specs/qwen36-35b-a3b.json).
    const bool ok = c.hidden == 2048 && c.layers == 40 && int(c.kinds.size()) == 40 && c.heads == 16 &&
                    c.kv_heads == 2 && c.head_dim == 256 && c.rotary_dim == 64 && c.experts == 256 && c.topk == 8 &&
                    c.moe_inter == 512 && c.shared_inter == 512 && c.lin_key_heads == 16 && c.lin_value_heads == 32 &&
                    c.lin_key_dim == 128 && c.lin_value_dim == 128 && c.conv_kernel == 4 && c.vocab == 248320 &&
                    c.rope_theta > 0;
    if (!ok) throw std::runtime_error(dir + ": not the Qwen3.6-35B-A3B geometry the lax kernels are built for");
    return c;
}

size_t consts_bytes(Kind k) { return k == Kind::Linear ? kConstsBytes : kConstsFull; }

// ---- the buffers ----------------------------------------------------------------------

void pack_pool(const Model& m, const Config& c, int L, uint8_t* dst) {
    std::memset(dst, 0, kPoolBytes);
    const MoeLayout g;
    const size_t E = size_t(c.experts), stripes = size_t(c.moe_inter) / 128, per_stripe = g.stripe / kQ41;

    // Routed experts' up and gate: per expert, `stripes` pairs [up stripe k | gate stripe k],
    // each stripe's 4 row blocks x 8 k-tiles transposed to k-tile-major.
    {
        const std::string un = lname(L, "mlp.up_exps_proj.weight"), gn = lname(L, "mlp.gate_exps_proj.weight");
        const Tensor &ut = m.tensor(un), &gt = m.tensor(gn);
        const size_t n = E * stripes * per_stripe;
        const Q41 up(ut, un, 0, n), gate(gt, gn, 0, n);
        for (size_t e = 0; e < E; ++e)
            for (size_t k = 0; k < stripes; ++k) {
                const size_t src = (stripes * e + k) * per_stripe;
                uint8_t* d = dst + (2 * stripes * e + 2 * k) * g.stripe;
                for (size_t j = 0; j < per_stripe; ++j) {
                    std::memcpy(d + j * kQ41, up[src + stripe_src(j)], kQ41);
                    std::memcpy(d + g.stripe + j * kQ41, gate[src + stripe_src(j)], kQ41);
                }
            }
    }
    // Routed experts' down slices.
    {
        const std::string dn = lname(L, "mlp.down_exps_proj.weight");
        const size_t per = g.up_bytes / kQ41;
        const Q41 down(m.tensor(dn), dn, 0, E * per);
        for (size_t e = 0; e < E; ++e)
            for (size_t j = 0; j < per; ++j)
                std::memcpy(dst + g.pool_down + e * g.up_bytes + j * kQ41, down[e * per + down_src(j)], kQ41);
    }
    // The shared expert.
    const size_t H = size_t(c.hidden), FF = size_t(c.shared_inter), band = FF * H / 8192;
    std_band(m, lname(L, "mlp.share_up_exps_proj.weight"), 0, band, H, dst + g.share_up);
    std_band(m, lname(L, "mlp.share_gate_exps_proj.weight"), 0, band, H, dst + g.share_gate);
    std_band(m, lname(L, "mlp.share_down_exps_proj.weight"), 0, band, FF, dst + g.share_down);

    uint8_t* a = dst + g.attn;
    if (c.kinds[size_t(L)] == Kind::Linear) {
        // DeltaNet: qkv (2*16*128 + 32*128 = 8192 rows), then the z gate (4096 rows).
        const size_t qkv = size_t(2 * c.lin_key_heads * c.lin_key_dim + c.lin_value_heads * c.lin_value_dim) * H / 8192;
        const size_t z = size_t(c.lin_value_heads * c.lin_value_dim) * H / 8192;
        std_band(m, lname(L, "linear_attn.qkv_proj.weight"), 0, qkv, H, a);
        std_band(m, lname(L, "self_attn.gate_proj.weight"), 0, z, H, a + qkv * kQ41);
    } else {
        // Full attention: q_proj's first half, k, v, q_proj's second half, o.
        const size_t qh = size_t(c.heads * c.head_dim) * H / 8192, kv = size_t(c.kv_heads * c.head_dim) * H / 8192;
        const size_t o = H * size_t(c.heads * c.head_dim) / 8192;
        const std::string q = lname(L, "self_attn.q_proj.weight");
        std_band(m, q, 0, qh, H, a);
        a += qh * kQ41;
        std_band(m, lname(L, "self_attn.k_proj.weight"), 0, kv, H, a);
        a += kv * kQ41;
        std_band(m, lname(L, "self_attn.v_proj.weight"), 0, kv, H, a);
        a += kv * kQ41;
        std_band(m, q, qh, qh, H, a);
        a += qh * kQ41;
        std_band(m, lname(L, "self_attn.o_proj.weight"), 0, o, size_t(c.heads * c.head_dim), a);
    }
}

void pack_consts(const Model& m, const Config& c, int L, uint8_t* dst) {
    const Kind k = c.kinds[size_t(L)];
    std::memset(dst, 0, consts_bytes(k));
    const size_t H = size_t(c.hidden);
    if (k == Kind::Full) {
        put(m, lname(L, "input_layernorm.weight"), 4096, dst + 0);
        put(m, lname(L, "post_attention_layernorm.weight"), 4096, dst + 4096);
        put(m, lname(L, "self_attn.q_norm.weight"), 512, dst + 8192);
        put(m, lname(L, "self_attn.k_norm.weight"), 512, dst + 8704);
        put(m, lname(L, "moe_router.weight"), 1u << 20, dst + 10240);
        put(m, lname(L, "shared_expert_gate.weight"), 4096, dst + 1058816);
        return;
    }
    put(m, lname(L, "input_layernorm.weight"), 4096, dst + 0);
    put(m, lname(L, "linear_attn.ssm_alpha_proj.weight"), 131072, dst + 4096);
    put(m, lname(L, "linear_attn.ssm_beta_proj.weight"), 131072, dst + 135168);
    put(m, lname(L, "linear_attn.ssm_a"), 128, dst + 266240);
    put(m, lname(L, "linear_attn.ssm_dt.bias"), 128, dst + 266368);
    {
        // conv1d bf16 [taps, 8 x 1024] -> [8 groups][taps][1024]
        const std::string n = lname(L, "linear_attn.ssm_conv1d.weight");
        const Tensor& t = m.tensor(n);
        const size_t taps = size_t(c.conv_kernel), groups = 8, width = 1024;
        if (t.bytes != taps * groups * width * 2) throw std::runtime_error(n + ": unexpected size");
        uint8_t* d = dst + 270336;
        for (size_t gi = 0; gi < groups; ++gi)
            for (size_t ti = 0; ti < taps; ++ti)
                std::memcpy(d + (gi * taps + ti) * width * 2, t.data + (ti * groups + gi) * width * 2, width * 2);
    }
    put(m, lname(L, "linear_attn.ssm_norm.weight"), 256, dst + 335872);
    put(m, lname(L, "post_attention_layernorm.weight"), 4096, dst + 339968);
    put(m, lname(L, "moe_router.weight"), 1u << 20, dst + 344064);
    put(m, lname(L, "shared_expert_gate.weight"), 4096, dst + 1392640);
    const size_t vd = size_t(c.lin_value_heads * c.lin_value_dim);
    std_band(m, lname(L, "linear_attn.ssm_out_proj.weight"), 0, H * vd / 8192, vd, dst + 1396736);
}

void pack_lmhead(const Model& m, uint8_t* dst) {
    const Tensor& t = m.tensor("lm_head.weight");
    if (chunk_bytes(t) != kQ8) throw std::runtime_error("lm_head.weight: the lax lm head is q8");
    const size_t n = t.bytes / kQ8;
    if (n * kQ8 > kLmPoolBytes) throw std::runtime_error("lm_head larger than its pool");
    for (size_t k = 0; k < n; ++k) std::memcpy(dst + k * kQ8, t.data + lmhead_src(k, kHidden) * kQ8, kQ8);
    std::memset(dst + n * kQ8, 0, kLmPoolBytes - n * kQ8);
}

void pack_norm(const Model& m, uint8_t* dst) { put(m, "model.norm.weight", kHidden * 2, dst); }

void pack_ptab(const Config& c, int rows, uint8_t* dst) {
    // Row p: i32 valid = p (the cached rows before it), i32 nf = max(p, 1) (rows streamed:
    // position 0 streams one masked dummy), then the RoPE pairs (i, i + rot/2) as f32 cos at
    // 512 and f32 sin right after. inv_freq = theta^(-i/half) and the angle are double, the
    // stored values float32, as the reference computes them.
    const int half = c.rotary_dim / 2;
    std::vector<double> inv(static_cast<size_t>(half));
    for (int i = 0; i < half; ++i) inv[size_t(i)] = std::pow(c.rope_theta, -double(i) / double(half));
    std::memset(dst, 0, size_t(rows) * kPtabRow);
    for (int p = 0; p < rows; ++p) {
        uint8_t* row = dst + size_t(p) * kPtabRow;
        const int32_t valid = p, nf = std::max(p, 1);
        std::memcpy(row, &valid, 4);
        std::memcpy(row + 4, &nf, 4);
        for (int i = 0; i < half; ++i) {
            const double ang = double(p) * inv[size_t(i)];
            const float cs = float(std::cos(ang)), sn = float(std::sin(ang));
            std::memcpy(row + 512 + 4 * i, &cs, 4);
            std::memcpy(row + 512 + 4 * half + 4 * i, &sn, 4);
        }
    }
}

// ---- SHA-256 -------------------------------------------------------------------------

std::array<uint8_t, 32> sha256(std::span<const uint8_t> data) {
    static constexpr uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    auto block = [&](const uint8_t* p) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 | uint32_t(p[4 * i + 2]) << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
    };
    const size_t n = data.size(), full = n / 64;
    for (size_t i = 0; i < full; ++i) block(data.data() + 64 * i);
    uint8_t tail[128] = {};
    const size_t rem = n - full * 64;
    std::memcpy(tail, data.data() + full * 64, rem);
    tail[rem] = 0x80;
    const size_t tl = rem < 56 ? 64 : 128;
    const uint64_t bitlen = uint64_t(n) * 8;
    for (int i = 0; i < 8; ++i) tail[tl - 1 - i] = uint8_t(bitlen >> (8 * i));
    for (size_t o = 0; o < tl; o += 64) block(tail + o);
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) out[size_t(4 * i + j)] = uint8_t(h[i] >> (24 - 8 * j));
    return out;
}

std::string hex(std::span<const uint8_t> d) {
    static const char* x = "0123456789abcdef";
    std::string s;
    for (uint8_t b : d) {
        s += x[b >> 4];
        s += x[b & 15];
    }
    return s;
}

}  // namespace onebit::npu::lax
