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
// tests/npu_pack_import_test [golden dir] [pack dir]
//
// 1. MatMulNBits -> (codes, scale, zero), encoded here and decoded by the importer.
// 2. q4_1 tiling against tests/golden/oga_pack_tiles.bin, which
//    tools/oga_pack_tiles_golden.py writes with the container's established layout
//    (tools/gguf_to_q4nx.py); the same synthetic tensor is rebuilt here.
// 3. The tiles as the engine reads them: q4nx::dequant_tile reproduces codes*scale+zero
//    to the bf16 parameter bound, and the codes come back out unchanged, padding included.
// 4. With a pack dir: `1bit npu-pack`'s plan refuses every projection by name, and an
//    --allow-incomplete import produces a container the engine's own Model can open.

#include "model.h"
#include "pack.h"
#include "pack_import.h"
#include "q4nx.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace pi = onebit::npu::pack_import;
namespace fs = std::filesystem;

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what.c_str());
        ++failures;
    }
}

// ── 0. external-data bounds are overflow-proof ──────────────────────────────
// A crafted model.onnx can declare offset = UINT64_MAX - 4 with length = 8, which sums to 3 and used to pass
// an `offset + length > size` check and then read 8 bytes from a wild pointer.
void test_bounds() {
    const uint64_t big = UINT64_MAX;
    const uint64_t size = 1024;
    check(!pi::ext_range_ok(size, big - 4, 8), "bounds: offset+length wrapping is refused");
    check(!pi::ext_range_ok(size, big, 0), "bounds: an offset past the file is refused, even with length 0");
    check(!pi::ext_range_ok(size, big - 1, 4), "bounds: a length that would wrap is refused");
    check(pi::ext_range_ok(size, 0, size), "bounds: the whole file is in range");
    check(pi::ext_range_ok(size, size, 0), "bounds: the empty range at EOF is in range");
    check(!pi::ext_range_ok(size, size - 1, 2), "bounds: one byte past the end is refused");
    check(!pi::ext_range_ok(size, size + 1, 0), "bounds: an offset one past the end is refused");

    // a length is only honoured when the product that produced it cannot wrap (the readers' second gate)
    uint64_t out = 0;
    check(!pi::mul_ok(UINT64_MAX, 2, out), "lengths: UINT64_MAX * 2 is refused, not wrapped");
    check(!pi::mul_ok(UINT64_MAX, UINT64_MAX, out), "lengths: a 64x64 square that overflows is refused");
    check(pi::mul_ok(0, UINT64_MAX, out) && out == 0, "lengths: 0 * anything is 0");
    check(pi::mul_ok(1ULL << 32, 4, out) && out == (1ULL << 34), "lengths: a product that fits is exact");
    check(pi::mul_ok(uint64_t(151936), 7, out) && out == 1063552, "lengths: the head's own N*groups is exact");
}


// ── crafted model.onnx files the importer must refuse ───────────────────────
// The lengths this reader takes come from the file's own metadata, so the only way to prove the wiring is a
// file that overstates what it carries. Same pattern as moe_gguf_index_test: build the bytes at run time.
namespace pb {
void varint(std::string& o, uint64_t v) {
    while (v >= 0x80) {
        o.push_back(char((v & 0x7F) | 0x80));
        v >>= 7;
    }
    o.push_back(char(v));
}
void tag(std::string& o, int field, int wire) { varint(o, (uint64_t(field) << 3) | uint64_t(wire)); }
void ints(std::string& o, int field, int64_t v) {
    tag(o, field, 0);
    varint(o, uint64_t(v));
}
void str(std::string& o, int field, const std::string& v) {
    tag(o, field, 2);
    varint(o, v.size());
    o += v;
}
std::string kv(const std::string& k, const std::string& v) {
    std::string o;
    str(o, 1, k);
    str(o, 2, v);
    return o;
}
// a tensor whose values live in an external file, optionally with dims that overstate it
std::string tensor(const std::string& name, const std::vector<int64_t>& dims, const std::string& location,
                   uint64_t offset, uint64_t length) {
    std::string t;
    for (auto d : dims) ints(t, 1, d);                 // dims
    ints(t, 2, 1);                                     // data_type (FLOAT; the importer only uses it for width)
    str(t, 8, name);
    for (const auto& [k, v] : {std::pair<std::string, std::string>{"location", location},
                               {"offset", std::to_string(offset)}, {"length", std::to_string(length)}}) {
        tag(t, 13, 2);
        const auto e = kv(k, v);
        varint(t, e.size());
        t += e;
    }
    ints(t, 14, 1);                                    // data_location = EXTERNAL
    return t;
}
std::string matmulnbits(const std::string& name, const std::vector<std::string>& ins, int64_t K, int64_t N,
                        int64_t bits, int64_t block) {
    std::string n;
    for (const auto& i : ins) str(n, 1, i);
    str(n, 4, "MatMulNBits");
    for (const auto& [k, v] : {std::pair<std::string, int64_t>{"K", K}, {"N", N}, {"bits", bits},
                               {"block_size", block}}) {
        std::string a;
        str(a, 1, k);
        ints(a, 3, v);
        tag(n, 5, 2);
        varint(n, a.size());
        n += a;
    }
    return n;
}
std::string model(const std::vector<std::string>& tensors, const std::vector<std::string>& nodes) {
    std::string g;
    for (const auto& n : nodes) {
        tag(g, 1, 2);
        varint(g, n.size());
        g += n;
    }
    str(g, 2, "crafted");
    for (const auto& t : tensors) {
        tag(g, 5, 2);
        varint(g, t.size());
        g += t;
    }
    std::string m;
    ints(m, 1, 8);                                     // ir_version
    tag(m, 7, 2);                                      // graph
    varint(m, g.size());
    m += g;
    return m;
}
}  // namespace pb

void put_file(const std::string& path, const std::string& bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), std::streamsize(bytes.size()));
}

// The two overstatements the readers must catch: declared dims larger than the external file, and a
// MatMulNBits node whose N/K/block_size demand more qweight than the tensor declares.
void test_crafted_refusals(const std::string& root) {
    const fs::path dir = fs::path(root) / "crafted";
    fs::remove_all(dir);
    fs::create_directories(dir);
    put_file((dir / "model.onnx.data").string(), std::string(4096, '\x11'));   // 4 KiB of "weights"
    // the plan() step reads config.json before anything else; the crafted packs are only about the readers,
    // so the geometry here is deliberately tiny
    fs::create_directories(dir / "cache");   // plan() reads the pack's meta twins when they are there
    put_file((dir / "config.json").string(),
             R"({"model_type":"qwen2","hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,)"
             R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"vocab_size":64,)"
             R"("rope_theta":10000.0,"tie_word_embeddings":true})");

    // 1. embed dims [1024, 1024] x 4 B = 4 MiB declared, 1 KiB present
    put_file((dir / "model.onnx").string(),
             pb::model({pb::tensor("model.embed_tokens.weight", {1024, 1024}, "model.onnx.data", 0, 1024)}, {}));
    pi::Options allow;
    allow.allow_incomplete = true;   // so the plan's own refusals do not short-circuit before the readers
    try {
        const auto r = pi::import_pack(dir.string(), (dir / "out1").string(), allow);
        check(false, "crafted: overstated embed dims were refused");
    } catch (const std::exception& e) {
        const std::string what = e.what();
        std::printf("  crafted embed refusal: %s\n", what.c_str());
        check(what.find("model.embed_tokens.weight") != std::string::npos,
              "crafted: the refusal names the tensor (" + what + ")");
    }
    check(!fs::exists(dir / "out1" / "model.q4nx"), "crafted: nothing written for the overstated embed");

    // 2. a MatMulNBits whose qweight tensor carries 256 B while N=64, K=512, block=128 demand 16 KiB
    put_file((dir / "model.onnx").string(),
             pb::model({pb::tensor("model.embed_tokens.weight", {8, 8}, "model.onnx.data", 0, 256),
                        // scales and zeros sized exactly as the node's N/K/block demand (N*(K/block)*4 and
                        // N*(K/block+1)/2), so the refusal that fires is the qweight one
                        pb::tensor("lm_head_scales", {}, "model.onnx.data", 256, 1024),
                        pb::tensor("lm_head_zeros", {}, "model.onnx.data", 1280, 160),
                        pb::tensor("lm_head_qweight", {}, "model.onnx.data", 1440, 256)},
                       {pb::matmulnbits("lm_head", {"x", "lm_head_qweight", "lm_head_scales", "lm_head_zeros"},
                                        512, 64, 4, 128)}));
    try {
        const auto r = pi::import_pack(dir.string(), (dir / "out2").string(), allow);
        check(false, "crafted: overstated MatMulNBits geometry was refused");
    } catch (const std::exception& e) {
        const std::string what = e.what();
        std::printf("  crafted matmulnbits refusal: %s\n", what.c_str());
        check(what.find("lm_head_qweight") != std::string::npos,
              "crafted: the refusal names the qweight tensor (" + what + ")");
    }
    check(!fs::exists(dir / "out2" / "model.q4nx"), "crafted: nothing written for the overstated MatMulNBits");
    fs::remove_all(dir);
}

// ── 1. MatMulNBits: encode here, decode with the importer ────────────────────
void test_matmulnbits() {
    const int N = 64, K = 256, B = 128;
    std::vector<uint8_t> qw(size_t(N) * (K / B) * (B / 2));
    std::vector<float> sc(size_t(N) * (K / B));
    std::vector<uint8_t> zp(size_t(N) * ((K / B + 1) / 2));
    auto want = [&](int n, int k) { return uint8_t((n * 5 + k * 3) % 16); };
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < K / B; ++b) {
            sc[size_t(n) * (K / B) + b] = 0.01f * float(b + 1);
            const uint8_t z = uint8_t((n + b) % 16);
            // zero points: two per byte, block 0 in the low nibble
            zp[size_t(n) * ((K / B + 1) / 2) + b / 2] |= uint8_t((b & 1) ? (z << 4) : z);
            for (int i = 0; i < B / 2; ++i) {
                const int k = b * B + 2 * i;
                const uint8_t lo = want(n, k), hi = want(n, k + 1);
                qw[(size_t(n) * (K / B) + b) * (B / 2) + i] = uint8_t(lo | (hi << 4));
            }
        }
    }
    const pi::Q4 q = pi::decode_matmulnbits(qw.data(), sc.data(), zp.data(), N, K, B);
    bool codes_ok = true, params_ok = true;
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
            if (q.codes[size_t(n) * K + k] != want(n, k)) codes_ok = false;
            const int b = k / B;
            const uint8_t z = uint8_t((n + b) % 16);
            const float want_scale = sc[size_t(n) * (K / B) + b];
            const size_t g = size_t(n) * (K / 32) + size_t(k) / 32;
            if (q.scale[g] != want_scale) params_ok = false;
            if (std::fabs(q.zero[g] - (-float(z) * want_scale)) > 1e-7f * std::fabs(want_scale) * 16)
                params_ok = false;
        }
    check(codes_ok, "MatMulNBits: every 4-bit code decodes to the encoded one");
    check(params_ok, "MatMulNBits: scale and zero (-zp*scale) per 32-column group");

    // the two decoders the self-check uses as controls must be wrong
    const pi::Q4 swap = pi::decode_matmulnbits(qw.data(), sc.data(), zp.data(), N, K, B, true, false);
    const pi::Q4 nozp = pi::decode_matmulnbits(qw.data(), sc.data(), nullptr, N, K, B, false, true);
    int same_swap = 0;
    for (size_t i = 0; i < q.codes.size(); ++i) same_swap += (swap.codes[i] == q.codes[i]);
    check(same_swap < int(q.codes.size()) / 4, "control: nibble swap really changes the codes");
    const size_t nz = size_t(1) * (K / 32);  // row 1 has zero-point 1, so its zero is not 0
    check(nozp.zero[nz] == 0.0f && q.zero[nz] != 0.0f,
          "control: dropping the zero point really changes it");
}

// ── 2. tiling against the committed golden ──────────────────────────────────
pi::Q4 synthetic() {
    const int N = 64, K = 384, KP = 512;
    pi::Q4 q;
    q.N = N;
    q.K = K;
    q.block = 128;
    q.codes.assign(size_t(N) * K, 0);
    q.scale.assign(size_t(N) * (K / 32), 0.0f);
    q.zero.assign(size_t(N) * (K / 32), 0.0f);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) q.codes[size_t(n) * K + k] = uint8_t((n * 7 + k * 13) % 16);
    for (int n = 0; n < N; ++n)
        for (int g = 0; g < KP / 32; ++g) {
            if (g * 32 >= K) continue;  // padding groups stay zero
            q.scale[size_t(n) * (K / 32) + g] = 0.001f * float(g % 7 + 1);
            q.zero[size_t(n) * (K / 32) + g] = -0.05f * float(g % 3) - 0.01f * float(n % 5);
        }
    return q;
}

// The same shape, but with parameters that are exactly representable in bf16 (powers of
// two), so tile -> dequant must reproduce codes*scale+zero with no error at all.
pi::Q4 synthetic_exact() {
    const int N = 64, K = 384, KP = 512;
    pi::Q4 q;
    q.N = N;
    q.K = K;
    q.block = 128;
    q.codes.assign(size_t(N) * K, 0);
    q.scale.assign(size_t(N) * (K / 32), 0.0f);
    q.zero.assign(size_t(N) * (K / 32), 0.0f);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) q.codes[size_t(n) * K + k] = uint8_t((n * 7 + k * 13) % 16);
    for (int n = 0; n < N; ++n)
        for (int g = 0; g < KP / 32; ++g) {
            if (g * 32 >= K) continue;
            q.scale[size_t(n) * (K / 32) + g] = 0.0625f * float(g % 7 + 1);          // 2^-4 steps
            q.zero[size_t(n) * (K / 32) + g] = -0.25f * float(g % 3) - 0.125f * float(n % 5);
        }
    return q;
}

void test_tiles(const std::string& golden_dir) {
    std::vector<uint8_t> tiled;
    pi::tile_q4_1(synthetic(), tiled);
    check(tiled.size() == 4 * size_t(onebit::npu::kTileBytes), "tiles: 64x384 pads to 2 row tiles x 2 col tiles");

    // 3. the codes must survive the tile layout, padding included
    const pi::Q4 qs = synthetic();
    bool codes_ok = true, pad_zero = true;
    for (int rt = 0; rt < 2; ++rt)
        for (int kt = 0; kt < 2; ++kt) {
            const uint8_t* tile = tiled.data() + (size_t(rt) * 2 + kt) * onebit::npu::kTileBytes;
            for (int h = 0; h < 2; ++h)
                for (int g = 0; g < 8; ++g)
                    for (int j = 0; j < 32; ++j) {
                        const int k = kt * 256 + g * 32 + j;
                        for (int r2 = 0; r2 < 8; ++r2) {
                            const uint8_t byte = tile[1024 + h * 2048 + g * 256 + j * 8 + r2];
                            const int r_even = rt * 32 + h * 16 + r2 * 2;
                            if (k >= qs.K) {
                                if (byte != 0) pad_zero = false;
                                continue;
                            }
                            if ((byte & 0x0F) != qs.codes[size_t(r_even) * qs.K + k]) codes_ok = false;
                            if ((byte >> 4) != qs.codes[size_t(r_even + 1) * qs.K + k]) codes_ok = false;
                        }
                    }
        }
    check(codes_ok, "tiles: every code is at region h*8+g, byte 8*j + r2, even row in the low nibble");
    check(pad_zero, "tiles: the padded columns are zero, not stray weights");

    // and the engine's own decoder must reproduce codes*scale+zero: exactly, because
    // this variant's parameters are bf16 values to begin with
    std::vector<uint8_t> exact_tiles;
    pi::tile_q4_1(synthetic_exact(), exact_tiles);
    const pi::Q4 q = synthetic_exact();
    double worst = 0;
    std::vector<float> w(32 * 256);
    for (int i = 0; i < 4; ++i) {
        onebit::npu::q4nx::dequant_tile(onebit::npu::q4nx::Chunk::Q4_1,
                                        exact_tiles.data() + size_t(i) * onebit::npu::kTileBytes, w.data(),
                                        256);
        const int rt = i / 2, kt = i % 2;
        for (int r = 0; r < 32; ++r)
            for (int c = 0; c < 256; ++c) {
                const int k = kt * 256 + c;
                if (k >= q.K) continue;
                const int row = rt * 32 + r;
                const size_t g = size_t(row) * (q.K / 32) + size_t(k) / 32;
                const double ref = double(q.codes[size_t(row) * q.K + k]) * double(q.scale[g]) +
                                   double(q.zero[g]);
                worst = std::max(worst, std::fabs(double(w[size_t(r) * 256 + c]) - ref));
            }
    }
    check(worst == 0.0, "tiles: dequant_tile reproduces codes*scale+zero exactly for bf16 parameters"
                        " (worst " + std::to_string(worst) + ")");

    if (golden_dir.empty()) return;
    const std::string path = golden_dir + "/oga_pack_tiles.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::printf("skip golden: %s absent\n", path.c_str());
        return;
    }
    std::vector<uint8_t> golden((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    check(golden.size() == tiled.size(), "golden: same tile count as tools/gguf_to_q4nx.py");
    check(golden == tiled, "golden: tiles byte-identical to the container's established layout");
    if (golden != tiled)
        for (size_t i = 0; i < std::min(golden.size(), tiled.size()); ++i)
            if (golden[i] != tiled[i]) {
                std::fprintf(stderr, "  first difference at byte %zu: golden %02x, ours %02x\n", i,
                             golden[i], tiled[i]);
                break;
            }
}

// ── 4. a real pack ──────────────────────────────────────────────────────────
void test_pack(const std::string& pack, const std::string& golden_dir) {
    const pi::Plan p = pi::plan(pack);
    const int layers = p.dims.layers;
    const bool qwen05b = p.dims.hidden == 896 && layers == 24 && p.dims.vocab == 151936;
    check(p.dims.hidden > 0 && layers > 0 && p.dims.vocab > 0,
          "pack: geometry from config.json and the embedding");
    if (qwen05b) check(true, "pack: this is the Qwen2.5-0.5B geometry the lane started from");
    const auto refused = p.refused();
    check(int(refused.size()) == layers * 7 + 1,  // the projections, plus the final norm
          "pack: all " + std::to_string(layers * 7) + " projections and the final norm are refused by "
          "name, got " + std::to_string(refused.size()));
    bool named = true;
    for (const auto& s : refused)
        if (s.find(".fconst") == std::string::npos) named = false;
    check(named, "pack: every refusal names the .fconst and the missing layout fact");

    int norms = 0, tiles_planned = 0;
    for (const auto& t : p.tensors) {
        if (t.name.find("layernorm") != std::string::npos || t.name == "model.norm.weight") ++norms;
        if (t.dtype == "q4_1" && t.refusal.empty())
            tiles_planned = int((t.shape[0] / 32) * ((t.shape[1] + 255) / 256));
    }
    check(norms == layers * 2 + 1,
          "pack: 2 norms per layer plus the final norm are planned (the final one refused)");
    // K is padded to the 256-column grid, so this holds for any hidden size
    const int head_rows = int(p.tensors.at(1).shape.at(0));
    check(tiles_planned == (head_rows / 32) * ((p.tensors.at(1).shape.at(1) + 255) / 256),
          "pack: lm_head tile count follows from its shape and the 256-column grid");
    if (qwen05b) check(tiles_planned == (151936 / 32) * 4, "pack: 0.5B lm_head plans 4748 x 4 tiles");

    // refusing must write nothing at all
    const std::string out = (fs::temp_directory_path() / "oga_pack_import_test").string();
    fs::remove_all(out);
    const pi::Report r0 = pi::import_pack(pack, out, pi::Options{});
    check(!r0.refused.empty() && !fs::exists(out + "/model.q4nx"),
          "pack: the refusal writes no container");
    fs::remove_all(out);
    if (golden_dir.empty()) { /* unreachable in practice */ }

    const pi::Report r = pi::import_pack(pack, out, pi::Options{true});
    check(r.lmhead_corr > 0.999, "pack: the decode reproduces the fp16 embedding (corr " +
                                     std::to_string(r.lmhead_corr) + ")");
    check(r.ctrl_corr_nibble < 0.5 && r.ctrl_corr_nozp < 0.95,
          "pack: both wrong decoders are rejected as controls");
    // The container's q4_1 chunk holds a bf16 zero, and this pack's f32 zero is -zp*scale,
    // which is bf16-exact for only ~43 % of groups; the residual is that rounding, measured
    // at 3.3e-3 relative and <5e-4 absolute. Scales are exactly bf16 here, so no code moves.
    check(r.q4nx_rel < 5e-3, "pack: container read back within the bf16 parameter bound (" +
                                 std::to_string(r.q4nx_rel) + ")");
    check(r.zero_exact_pct > 40.0 && r.zero_exact_pct < 50.0,
          "pack: the bf16 zero point is exact for ~43 % of groups as measured (" +
              std::to_string(r.zero_exact_pct) + " %)");
    // engine#734: the geometry is MEASURED from the anchors, so it holds for any hidden size
    check(r.stride > 0 && r.second_header > 0 && r.second_header < r.stride,
          "pack: the .fconst layer stride and second header were located (" +
              std::to_string(r.stride) + ", " + std::to_string(r.second_header) + ")");
    check(r.stride % 64 == 0, "pack: the measured stride is 64-byte aligned");
    check(r.norm_anchors_ok == r.norm_anchors_total && r.norm_anchors_total == layers * 2,
          "pack: every layer RMSNorm sits at its epsilon anchor, whatever the hidden size (" +
              std::to_string(r.norm_anchors_ok) + "/" + std::to_string(r.norm_anchors_total) + ")");
    if (qwen05b)
        check(r.stride == 8708288 && r.second_header == 1115968,
              "pack: 0.5B reproduces the offsets that were measured by hand");

    // the container must open with the engine's own loader
    onebit::npu::Model m(out);
    const auto& head = m.tensor("lm_head.weight");
    check(head.shape.at(0) == int64_t((head_rows / 32) * ((p.tensors.at(1).shape.at(1) + 255) / 256)),
          "pack: lm_head tile count matches the container");
    check(m.tensor("model.layers.0.input_layernorm.weight").shape.at(0) == p.dims.hidden,
          "pack: a layer norm is `hidden` bf16 values, whatever the hidden size");
    check(m.dims().hidden == p.dims.hidden && m.dims().layers == layers,
          "pack: config.json drives the loader");
    fs::remove_all(out);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string golden_dir = argc > 1 ? argv[1] : "";
    const std::string pack = argc > 2 ? argv[2] : "";
    try {
        test_bounds();
        test_crafted_refusals(golden_dir.empty() ? "." : golden_dir);
        test_matmulnbits();
        test_tiles(golden_dir);
        if (!pack.empty() && fs::exists(pack + "/model.onnx")) test_pack(pack, golden_dir);
        else std::printf("skip pack: no pack directory given\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all npu-pack checks passed\n");
    return 0;
}
