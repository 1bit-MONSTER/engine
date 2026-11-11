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
//
// dflash_draft_test.cpp -- the DFlash draft loader (npu/dflash_draft.h, engine#1780).
//
// Two halves:
//   1. a synthetic package written here, small enough to build in a moment, with known answers: every projection is
//      encoded tile by tile from a formula and must decode to that formula's values; the fusion matrix must come back
//      in row-major order; and each way a package can be wrong is refused with a message naming the thing;
//   2. the real Bolt draft, when a path is given (argv[1] or ONEBIT_TEST_DFLASH_DRAFT): every tensor loads and decodes
//      finite, one at a time, and the fusion matrix's element order is checked against a property of the weights
//      themselves (see fusion_order_holds). Without a path this half prints SKIP and the test still passes.
//
// No device, no kernels, no forward.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

#include "dflash_draft.h"

namespace {

namespace fs = std::filesystem;
using namespace onebit::npu::dflash;

int g_failures = 0;
int g_checks = 0;

void check(bool cond, const std::string& what) {
    ++g_checks;
    if (!cond) {
        std::printf("FAIL %s\n", what.c_str());
        ++g_failures;
    }
}

uint16_t to_bf16(float f) {   // exact for the values used here (few mantissa bits)
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return uint16_t(bits >> 16);
}
void put16(std::string& s, uint16_t v) { s.append(reinterpret_cast<const char*>(&v), 2); }
void put32(std::string& s, const void* p) { s.append(static_cast<const char*>(p), 4); }

// The synthetic weights, as formulas a decoder has to reproduce.
int code_of(int salt, int n, int k) { return (n * 7 + k * 3 + salt) & 15; }
float scale_of(int salt, int n, int group) { return 0.0078125f * float(1 + ((n + group + salt) % 4)); }    // 1/128 .. 4/128
float min_of(int salt, int n, int group) { return -0.0625f * float(1 + ((n * 3 + group + salt) % 5)); }
float weight_of(int salt, int n, int k) {
    return float(code_of(salt, n, k)) * scale_of(salt, n, k / 32) + min_of(salt, n, k / 32);
}
float fusion_of(int n, int k) { return 0.03125f * float((n * 5 + k * 11) % 64 - 32); }
float gain_of(int salt, int i) { return 1.0f + 0.125f * float((i + salt) % 8); }

// One N x K projection as a q4_1 tile grid: tiles row-major, each 5120 B = 256 bf16 scales (index group*32 + row),
// 256 bf16 minimums (same index), then 4096 B of 4-bit codes (two 16-row halves x 8 groups, 256 B each; inside one,
// byte 8*j + row/2 for column j of the group, even row in the low nibble).
std::string encode_projection(int salt, int N, int K) {
    std::string out;
    for (int tr = 0; tr < N / 32; ++tr)
        for (int tc = 0; tc < K / 256; ++tc) {
            std::string tile(5120, '\0');
            for (int g = 0; g < 8; ++g)
                for (int r = 0; r < 32; ++r) {
                    const int n = tr * 32 + r, group = tc * 8 + g;
                    const uint16_t s = to_bf16(scale_of(salt, n, group)), z = to_bf16(min_of(salt, n, group));
                    std::memcpy(&tile[size_t(2 * (g * 32 + r))], &s, 2);
                    std::memcpy(&tile[size_t(512 + 2 * (g * 32 + r))], &z, 2);
                }
            for (int r = 0; r < 32; ++r)
                for (int c = 0; c < 256; ++c) {
                    const int half = r / 16, rr = r % 16, g = c / 32, j = c % 32;
                    const int code = code_of(salt, tr * 32 + r, tc * 256 + c);
                    unsigned char& b = reinterpret_cast<unsigned char&>(tile[size_t(1024 + (half * 8 + g) * 256 + 8 * j + rr / 2)]);
                    b = (rr % 2 == 0) ? (unsigned char)((b & 0xF0) | code) : (unsigned char)((b & 0x0F) | (code << 4));
                }
            out += tile;
        }
    return out;
}

struct Geometry {
    int hidden = 256, intermediate = 512, layers = 2, heads = 4, kv_heads = 2, head_dim = 64, block_size = 8;
    std::vector<int> taps = {1, 3};
};

struct Entry {
    std::string name, dtype, shape, bytes;
};

// The package's tensors in container order. `edit` may rewrite the list (drop one, add one, damage one).
std::vector<Entry> tensors_of(const Geometry& g) {
    std::vector<Entry> t;
    const auto scalar_f = [&](const std::string& n, float v) { std::string b; put32(b, &v); t.push_back({n, "F32", "[1]", b}); };
    const auto scalar_i = [&](const std::string& n, int32_t v) { std::string b; put32(b, &v); t.push_back({n, "I32", "[1]", b}); };
    scalar_f("dflash.embedding_scale", 16.0f);
    scalar_f("dflash.rms_eps", 1e-6f);
    scalar_f("dflash.rope_theta", 1000000.0f);
    scalar_i("dflash.block_size", g.block_size);
    scalar_i("dflash.sliding_window", 1024);
    {
        std::string b;
        for (int id : g.taps) { const int32_t v = id; put32(b, &v); }
        t.push_back({"dflash.target_layer_ids", "I32", "[" + std::to_string(g.taps.size()) + "]", b});
    }
    const int fused = int(g.taps.size()) * g.hidden;
    {
        std::string b;   // [hidden/32][fused][32]
        for (int a = 0; a < g.hidden / 32; ++a)
            for (int k = 0; k < fused; ++k)
                for (int r = 0; r < 32; ++r) put16(b, to_bf16(fusion_of(a * 32 + r, k)));
        t.push_back({"model.fc.weight", "BF16",
                     "[" + std::to_string(g.hidden / 32) + "," + std::to_string(fused) + ",32]", b});
    }
    const auto gain = [&](const std::string& n, int salt, int width) {
        std::string b;
        for (int i = 0; i < width; ++i) put16(b, to_bf16(gain_of(salt, i)));
        t.push_back({n, "BF16", "[" + std::to_string(width) + "]", b});
    };
    const auto proj = [&](const std::string& n, int salt, int N, int K) {
        const std::string b = encode_projection(salt, N, K);
        t.push_back({n, "I8", "[" + std::to_string(b.size() / 5120) + ",5120]", b});
    };
    gain("model.hidden_norm.weight", 1, g.hidden);
    const int q = g.heads * g.head_dim, kv = g.kv_heads * g.head_dim;
    for (int l = 0; l < g.layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        const int s = 10 * (l + 1);
        gain(p + "input_layernorm.weight", s + 1, g.hidden);
        gain(p + "post_attention_layernorm.weight", s + 2, g.hidden);
        gain(p + "self_attn.k_norm.weight", s + 3, g.head_dim);
        gain(p + "self_attn.q_norm.weight", s + 4, g.head_dim);
        proj(p + "self_attn.q_proj.weight", s + 1, q, g.hidden);
        proj(p + "self_attn.k_proj.weight", s + 2, kv, g.hidden);
        proj(p + "self_attn.v_proj.weight", s + 3, kv, g.hidden);
        proj(p + "self_attn.o_proj.weight", s + 4, g.hidden, q);
        proj(p + "mlp.gate_proj.weight", s + 5, g.intermediate, g.hidden);
        proj(p + "mlp.up_proj.weight", s + 6, g.intermediate, g.hidden);
        proj(p + "mlp.down_proj.weight", s + 7, g.hidden, g.intermediate);
    }
    gain("model.norm.weight", 2, g.hidden);
    return t;
}

std::string config_of(const Geometry& g) {
    std::string taps;
    for (size_t i = 0; i < g.taps.size(); ++i) taps += (i ? "," : "") + std::to_string(g.taps[i]);
    return "{\"model_type\":\"dflash\",\"architectures\":[\"DFlashDraftModel\"],\"base_model\":\"synthetic\","
           "\"draft_weight\":\"dflash_weight.q4nx\",\"block_size\":" + std::to_string(g.block_size) +
           ",\"mask_token_id\":4,\"target_layer_ids\":[" + taps + "],\"num_hidden_layers\":" + std::to_string(g.layers) +
           ",\"hidden_size\":" + std::to_string(g.hidden) + ",\"intermediate_size\":" + std::to_string(g.intermediate) +
           ",\"num_attention_heads\":" + std::to_string(g.heads) + ",\"num_key_value_heads\":" + std::to_string(g.kv_heads) +
           ",\"head_dim\":" + std::to_string(g.head_dim) +
           ",\"rms_norm_eps\":1e-06,\"rope_theta\":1000000.0,\"sliding_window\":1024,\"embedding_scale\":16.0,\"causal\":false}";
}

void write_package(const fs::path& dir, const std::string& config, const std::vector<Entry>& tensors, size_t drop_tail = 0) {
    fs::create_directories(dir);
    std::ofstream(dir / "config.json") << config;
    std::string header = "{\"__metadata__\":{}", payload;
    for (const Entry& e : tensors) {
        header += ",\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":" + e.shape + ",\"data_offsets\":[" +
                  std::to_string(payload.size()) + "," + std::to_string(payload.size() + e.bytes.size()) + "]}";
        payload += e.bytes;
    }
    header += "}";
    const uint64_t n = header.size();
    std::ofstream f(dir / "dflash_weight.q4nx", std::ios::binary);
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(header.data(), std::streamsize(header.size()));
    f.write(payload.data(), std::streamsize(payload.size() - drop_tail));
}

std::string replaced(std::string s, const std::string& from, const std::string& to) {
    const size_t at = s.find(from);
    if (at == std::string::npos) {
        std::printf("FAIL test bug: '%s' is not in the config\n", from.c_str());
        ++g_failures;
        return s;
    }
    return s.replace(at, from.size(), to);
}

// load_draft() must throw, and the message must name `needle`.
void refused(const fs::path& dir, const std::string& needle, const std::string& what) {
    try {
        (void)load_draft(dir.string());
        check(false, what + ": accepted");
    } catch (const std::runtime_error& e) {
        const bool named = std::string(e.what()).find(needle) != std::string::npos;
        check(named, what + ": refused, but the message does not say '" + needle + "': " + e.what());
    }
}

void test_synthetic(const fs::path& root) {
    const Geometry g;
    const std::vector<Entry> good = tensors_of(g);
    const std::string cfg = config_of(g);

    {   // the package as it should be
        write_package(root / "good", cfg, good);
        DraftModel m;
        try {
            m = load_draft((root / "good").string());
        } catch (const std::exception& e) {
            check(false, std::string("the synthetic package loads: ") + e.what());
            return;
        }
        check(m.cfg.hidden == 256 && m.cfg.intermediate == 512 && m.cfg.layers == 2 && m.cfg.heads == 4 && m.cfg.kv_heads == 2 &&
                  m.cfg.head_dim == 64 && m.cfg.block_size == 8 && m.cfg.mask_token_id == 4 && !m.cfg.causal,
              "the config's geometry is read");
        check(m.cfg.target_layer_ids == std::vector<int>({1, 3}) && m.cfg.taps() == 2 && m.cfg.base_model == "synthetic",
              "and its target layers, in order");
        check(m.layers.size() == 2 && m.tensors == int(good.size()) && m.tensors == 6 + 3 + 2 * 11, "every tensor is accounted for");
        check(m.fc.rows == 256 && m.fc.cols == 512, "the fusion matrix is hidden x (taps * hidden)");

        const auto same = [&](const TensorView& v, int salt, int N, int K, const std::string& name) {
            check(v.rows == N && v.cols == K, name + " is " + std::to_string(N) + " x " + std::to_string(K));
            const std::vector<float> w = decode_projection(v);
            size_t bad = 0;
            for (int n = 0; n < N; ++n)
                for (int k = 0; k < K; ++k)
                    if (w[size_t(n) * size_t(K) + size_t(k)] != weight_of(salt, n, k)) ++bad;
            check(w.size() == size_t(N) * size_t(K) && bad == 0,
                  name + " decodes to the encoded values at every element (" + std::to_string(bad) + " differ)");
        };
        for (int l = 0; l < 2; ++l) {
            const DraftLayer& L = m.layers[size_t(l)];
            const int s = 10 * (l + 1);
            const std::string p = "layer " + std::to_string(l) + " ";
            same(L.q, s + 1, 256, 256, p + "q_proj");
            same(L.k, s + 2, 128, 256, p + "k_proj");
            same(L.v, s + 3, 128, 256, p + "v_proj");
            same(L.o, s + 4, 256, 256, p + "o_proj");
            same(L.gate, s + 5, 512, 256, p + "gate_proj");
            same(L.up, s + 6, 512, 256, p + "up_proj");
            same(L.down, s + 7, 256, 512, p + "down_proj");
            const std::vector<float> in = decode_gain(L.input_norm), qn = decode_gain(L.q_norm), kn = decode_gain(L.k_norm),
                                     post = decode_gain(L.post_norm);
            bool ok = in.size() == 256 && post.size() == 256 && qn.size() == 64 && kn.size() == 64;
            for (size_t i = 0; ok && i < in.size(); ++i) ok = in[i] == gain_of(s + 1, int(i)) && post[i] == gain_of(s + 2, int(i));
            for (size_t i = 0; ok && i < qn.size(); ++i) ok = kn[i] == gain_of(s + 3, int(i)) && qn[i] == gain_of(s + 4, int(i));
            check(ok, p + "norm gains decode to the encoded values, each under its own name");
        }
        const std::vector<float> fc = decode_fusion(m.fc);
        size_t bad = 0;
        for (int n = 0; n < 256; ++n)
            for (int k = 0; k < 512; ++k)
                if (fc[size_t(n) * 512 + size_t(k)] != fusion_of(n, k)) ++bad;
        check(fc.size() == 256u * 512u && bad == 0, "the fusion matrix comes back row-major (" + std::to_string(bad) + " elements differ)");
        // and the stored order really is a different one: read plainly, element (1, 0) is not what is stored second
        check(fusion_of(1, 0) != fusion_of(0, 1), "the fixture distinguishes the two element orders");
        const std::vector<float> hn = decode_gain(m.hidden_norm), fn = decode_gain(m.norm);
        check(hn[3] == gain_of(1, 3) && fn[3] == gain_of(2, 3) && hn[3] != fn[3], "hidden_norm and the final norm are not swapped");
    }

    // What must be refused, each by name.
    const auto with = [&](const std::string& name, const std::function<void(std::vector<Entry>&)>& edit, const std::string& config) {
        std::vector<Entry> t = good;
        edit(t);
        write_package(root / name, config, t);
        return root / name;
    };
    const auto none = [](std::vector<Entry>&) {};
    const auto find = [](std::vector<Entry>& t, const std::string& n) -> Entry& {
        for (Entry& e : t)
            if (e.name == n) return e;
        std::printf("FAIL test bug: no tensor %s\n", n.c_str());
        ++g_failures;
        return t[0];
    };
    refused(with("missing", [&](std::vector<Entry>& t) {
                for (size_t i = 0; i < t.size(); ++i)
                    if (t[i].name == "model.layers.1.mlp.up_proj.weight") { t.erase(t.begin() + long(i)); break; }
            }, cfg),
            "no tensor 'model.layers.1.mlp.up_proj.weight'", "a missing projection");
    refused(with("short", [&](std::vector<Entry>& t) { find(t, "model.layers.0.self_attn.k_proj.weight").bytes.resize(5120); }, cfg),
            "'model.layers.0.self_attn.k_proj.weight' holds 5120 B", "a projection with too few tiles");
    refused(with("extra", [&](std::vector<Entry>& t) { t.push_back({"model.selector.weight", "BF16", "[4]", std::string(8, '\0')}); }, cfg),
            "'model.selector.weight' is not part of a DFlashDraftModel", "a tensor of another architecture");
    refused(with("dtype", [&](std::vector<Entry>& t) { find(t, "model.norm.weight").dtype = "F16"; }, cfg),
            "'model.norm.weight' is declared F16", "a gain under another dtype label");
    refused(with("plainfc", [&](std::vector<Entry>& t) { find(t, "model.fc.weight").shape = "[256,512]"; }, cfg),
            "'model.fc.weight' is not declared [hidden/32, taps*hidden, 32]", "a fusion matrix declared in another element order");
    refused(with("block", [&](std::vector<Entry>& t) { const int32_t v = 16; std::memcpy(find(t, "dflash.block_size").bytes.data(), &v, 4); }, cfg),
            "'dflash.block_size' is 16 and config.json says 8", "a container that disagrees with its config on the block size");
    refused(with("taps", [&](std::vector<Entry>& t) { const int32_t v = 2; std::memcpy(find(t, "dflash.target_layer_ids").bytes.data() + 4, &v, 4); }, cfg),
            "'dflash.target_layer_ids[1]' is 2 and config.json says 3", "a container that disagrees on a target layer");
    refused(with("dflash2", none, replaced(cfg, "DFlashDraftModel", "DFlash2DraftModel")),
            "architecture 'DFlash2DraftModel' is not read by this loader", "a DFlash2 package");
    refused(with("type", none, replaced(cfg, "\"model_type\":\"dflash\"", "\"model_type\":\"qwen3\"")),
            "model_type is 'qwen3'", "a config that is not a draft's");
    refused(with("nokey", none, replaced(cfg, "\"head_dim\":64,", "")), "no 'head_dim'", "a config without head_dim");
    refused(with("order", none, replaced(cfg, "[1,3]", "[3,1]")), "'target_layer_ids' is not strictly increasing", "target layers out of order");
    refused(with("path", none, replaced(cfg, "\"dflash_weight.q4nx\"", "\"../good/dflash_weight.q4nx\"")),
            "'draft_weight' must be a file name inside the package", "a weight path that leaves the package");
    refused(with("grid", none, replaced(cfg, "\"intermediate_size\":512", "\"intermediate_size\":500")),
            "is not a whole 32 x 256 tile grid", "a geometry that does not fill whole tiles");
    refused(with("heads", none, replaced(cfg, "\"num_key_value_heads\":2", "\"num_key_value_heads\":3")),
            "4 heads cannot share 3 key/value heads", "key/value heads that do not divide the heads");
    {   // the last tensor's bytes are not all there
        write_package(root / "cut", cfg, good, 100);
        refused(root / "cut", "", "a container cut short of its last tensor");
    }
    {
        fs::create_directories(root / "empty");
        refused(root / "empty", "cannot open", "a directory with no config.json");
    }
}

// The fusion matrix's stored element order, checked on the weights themselves. Magnitudes that share an input column
// are correlated far more strongly than magnitudes that share an output row (each column multiplies one feature of a
// target layer, and features differ in scale). In the right order that holds whichever neighbours are compared: the next
// row and the row 32 away agree with each other, and so do the next column and the column 32 away. Read in the plain
// row-major order instead, "the next column" is really the same column and "32 columns away" is not, so the two
// column figures disagree by several times. The assertion is that consistency, not a threshold on one number.
bool fusion_order_holds(const std::vector<float>& w, int rows, int cols, double out[4]) {
    std::vector<float> lw(w.size());
    double mean = 0.0;
    for (size_t i = 0; i < w.size(); ++i) {
        lw[i] = std::log(std::fabs(w[i]) + 1e-6f);
        mean += lw[i];
    }
    mean /= double(w.size());
    double var = 0.0;
    for (float v : lw) var += (double(v) - mean) * (double(v) - mean);
    var /= double(w.size());
    const auto corr = [&](int dn, int dk) {
        double acc = 0.0;
        size_t count = 0;
        for (int n = 0; n + dn < rows; n += 3)
            for (int k = 0; k + dk < cols; k += 7) {
                acc += (double(lw[size_t(n) * size_t(cols) + size_t(k)]) - mean) *
                       (double(lw[size_t(n + dn) * size_t(cols) + size_t(k + dk)]) - mean);
                ++count;
            }
        return acc / double(count) / var;
    };
    out[0] = corr(1, 0);    // next row, same column
    out[1] = corr(32, 0);   // 32 rows on, same column
    out[2] = corr(0, 1);    // same row, next column
    out[3] = corr(0, 32);   // same row, 32 columns on
    const auto near = [](double a, double b) { return std::fabs(a - b) <= 0.5 * std::max(std::fabs(a), std::fabs(b)); };
    return near(out[0], out[1]) && near(out[2], out[3]) && out[0] > 2.0 * std::max(out[2], out[3]);
}

void test_real(const std::string& dir) {
    DraftModel m;
    try {
        m = load_draft(dir);
    } catch (const std::exception& e) {
        check(false, std::string("the real package loads: ") + e.what());
        return;
    }
    const DraftConfig& c = m.cfg;
    std::printf("real package: %d tensors, %zu bytes; %d layers, hidden %d, intermediate %d, %d heads / %d kv x %d, block %d, taps",
                m.tensors, m.container_bytes, c.layers, c.hidden, c.intermediate, c.heads, c.kv_heads, c.head_dim, c.block_size);
    for (int id : c.target_layer_ids) std::printf(" %d", id);
    std::printf("\n");
    check(int(m.layers.size()) == c.layers && m.tensors == 6 + 3 + 11 * c.layers, "every tensor of the real package is accounted for");
    size_t elements = 0;
    for (const DraftLayer& L : m.layers) {
        for (const TensorView* v : {&L.q, &L.k, &L.v, &L.o, &L.gate, &L.up, &L.down}) {
            const std::vector<float> w = decode_projection(*v);   // one at a time: the set is 1.3 GB as f32
            size_t nonfinite = 0;
            double sum2 = 0.0;
            for (float x : w) {
                if (!std::isfinite(x)) ++nonfinite;
                else sum2 += double(x) * double(x);
            }
            const double rms = std::sqrt(sum2 / double(w.size()));
            check(nonfinite == 0 && rms > 1e-4 && rms < 1.0,
                  v->name + " decodes finite with a weight-like rms (" + std::to_string(rms) + ", " + std::to_string(nonfinite) + " non-finite)");
            elements += w.size();
        }
        for (const TensorView* v : {&L.input_norm, &L.post_norm, &L.q_norm, &L.k_norm}) {
            // Finite and bounded, not "positive": three of the Bolt draft's post-attention norms hold one gain just
            // below zero (-0.0009 .. -0.0024), which is what a learned gain crossing zero looks like.
            bool ok = true;
            for (float x : decode_gain(*v)) ok = ok && std::isfinite(x) && std::fabs(x) < 100.0f;
            check(ok, v->name + " is finite and bounded");
        }
    }
    std::printf("real package: %zu projection weights decoded\n", elements);
    const std::vector<float> fc = decode_fusion(m.fc);
    double r[4];
    const bool right = fusion_order_holds(fc, m.fc.rows, m.fc.cols, r);
    std::printf("fusion matrix, as decoded:   next row %.4f, 32 rows on %.4f | next column %.4f, 32 columns on %.4f\n", r[0], r[1], r[2], r[3]);
    check(right, "the fusion matrix's decoded order is self-consistent: column-sharing neighbours agree, row-sharing neighbours agree");
    // the control: the same bytes read as plain row-major must NOT pass, or the check above proves nothing
    std::vector<float> plain(fc.size());
    for (size_t i = 0; i < plain.size(); ++i) {
        uint16_t h;
        std::memcpy(&h, m.fc.data + i * 2, 2);
        const uint32_t bits = uint32_t(h) << 16;
        std::memcpy(&plain[i], &bits, 4);
    }
    const bool wrong = fusion_order_holds(plain, m.fc.rows, m.fc.cols, r);
    std::printf("fusion matrix, read plainly: next row %.4f, 32 rows on %.4f | next column %.4f, 32 columns on %.4f\n", r[0], r[1], r[2], r[3]);
    check(!wrong, "and the plain row-major reading of the same bytes is not");
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path root = fs::temp_directory_path() / ("dflash-draft-test-" + std::to_string(::getpid()));
    fs::remove_all(root);
    test_synthetic(root);
    fs::remove_all(root);
    const char* real = (argc > 1 && argv[1][0] != '\0') ? argv[1] : std::getenv("ONEBIT_TEST_DFLASH_DRAFT");
    if (real != nullptr && *real != '\0') test_real(real);
    else std::printf("SKIP the real package: pass its directory, or set ONEBIT_TEST_DFLASH_DRAFT\n");
    if (g_failures == 0) std::printf("dflash_draft: ok (%d checks)\n", g_checks);
    return g_failures == 0 ? 0 : 1;
}
