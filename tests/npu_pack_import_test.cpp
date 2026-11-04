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
// 4. One tile out of a real pack's `.fconst` (tests/golden/oga_attn_tile.bin, copied verbatim
//    from Qwen2.5-0.5B-Instruct_rai_1.8.0_npu_4K by tools/oga_attn_tile_golden.py): the
//    importer's decoder must re-encode it to the same bytes, and its container output must match
//    the recorded q4_1 tiles. That checks the tile layout against the vendor writer's own
//    output with no vendor stack present -- which is what a CI run has.
// 4b. The same, for the MLP section's 68,608-byte gate|up unit
//    (tests/golden/oga_mlp_gu_group.bin, tools/oga_mlp_gu_golden.py): one decode, then the fused
//    tensor split into gate and up, both against their recorded container tiles.
// 5. With a pack dir: `1bit npu-pack`'s plan refuses only the final norm, imports the attention
//    projections and their biases and the MLP projections, and produces a container the engine's
//    own Model can open.

#include "model.h"
#include "pack.h"
#include "pack_import.h"
#include "q4nx.h"

#include <nlohmann/json.hpp>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

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
// ── 3b. the .fconst attention tile against shipped pack bytes ───────────────
void test_attn_tile(const std::string& golden_dir) {
    if (golden_dir.empty()) return;
    const std::string tile_path = golden_dir + "/oga_attn_tile.bin";
    std::ifstream f(tile_path, std::ios::binary);
    if (!f) {
        std::printf("skip attention golden: %s absent\n", tile_path.c_str());
        return;
    }
    std::vector<uint8_t> tile((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const int k = 896;
    const int64_t want = pi::attn_tile_bytes(k);
    check(int64_t(tile.size()) == want,
          "attention golden: the fixture is one " + std::to_string(want) + "-byte tile of K=" +
              std::to_string(k) + " (got " + std::to_string(tile.size()) + ")");
    if (int64_t(tile.size()) != want) return;

    const pi::TileW t = pi::decode_tile(tile.data(), k);
    check(t.bias.size() == 64 && t.code.size() == 64u * size_t(k),
          "attention golden: the tile decodes to 64 rows");

    // the self-check the importer runs on every tile: same bytes back, or the fields are not understood
    const std::vector<uint8_t> back = pi::encode_tile(t);
    check(back == tile, "attention golden: encode(decode(tile)) reproduces the pack's bytes exactly");
    if (back != tile)
        for (size_t i = 0; i < std::min(back.size(), tile.size()); ++i)
            if (back[i] != tile[i]) {
                std::fprintf(stderr, "  first difference at byte %zu: ours %02x, pack %02x\n", i, back[i],
                             tile[i]);
                break;
            }

    // ... and the check has to be able to fail: one nibble moved and one scale perturbed
    {
        pi::TileW bad = t;
        bad.code[0] ^= 0x01;
        check(pi::encode_tile(bad) != tile, "attention golden: a single changed code nibble is caught");
        // a scale one bf16 ulp away (2^-8 relative) has to be caught: the encoder rounds to bf16, so a
        // smaller perturbation would legitimately come back unchanged and prove nothing
        pi::TileW bad2 = t;
        bad2.scale[0] *= 1.004f;
        check(pi::encode_tile(bad2) != tile, "attention golden: a one-ulp scale change is caught");
        pi::TileW bad3 = t;
        bad3.bias.at(3) ^= 0x0001;
        check(pi::encode_tile(bad3) != tile, "attention golden: the bias slot is part of the check");
    }

    // engine#849: the zero-point nibbles pair with rows even/odd, exactly like the codes. The round trip
    // above cannot see this class -- it is invariant under any permutation applied on both sides -- so it
    // gets two checks of its own.
    {
        // (a) structural, off bytes whose nibbles are all distinct: byte i must give rows 2i and 2i+1.
        const int kbs = k / 128;
        const int64_t subv = pi::attn_tile_bytes(k) / (2 * kbs + 2);
        std::vector<uint8_t> raw(size_t(pi::attn_tile_bytes(k)), 0);
        for (int h = 0; h < 2; ++h)
            for (int kb = 0; kb < kbs; ++kb)
                for (int i = 0; i < 16; ++i)
                    raw[size_t((2 + 2 * kb + h) * subv + 2048 + i)] = uint8_t(i | ((15 - i) << 4));
        const pi::TileW z = pi::decode_tile(raw.data(), k);
        bool pairs_even_odd = true;
        for (int h = 0; h < 2 && pairs_even_odd; ++h)
            for (int kb = 0; kb < kbs && pairs_even_odd; ++kb)
                for (int i = 0; i < 16; ++i)
                    // accumulate: assigning here would let an early wrong pair be hidden by a later
                    // right one, so only the last pair of each block was ever checked
                    pairs_even_odd = pairs_even_odd &&
                                     z.zp[size_t(h * 32 + 2 * i) * size_t(kbs) + size_t(kb)] == uint8_t(i) &&
                                     z.zp[size_t(h * 32 + 2 * i + 1) * size_t(kbs) + size_t(kb)] ==
                                         uint8_t(15 - i);
        check(pairs_even_odd, "attention tile: zero-point nibbles pair with rows 2i / 2i+1 (even/odd)");

        // (b) on the pack's own tile: a correctly paired zero point sits where its 128 codes cluster, so
        // corr(zp, mean code) is near 1 (measured +0.9977); pairing in halves scores 0 (-0.0227).
        double n = 0, sz = 0, sm = 0, szz = 0, smm = 0, szm = 0;
        for (int r = 0; r < 64; ++r)
            for (int b = 0; b < kbs; ++b) {
                double sum = 0;
                for (int c = 0; c < 128; ++c) sum += t.code[size_t(r) * size_t(k) + size_t(b * 128 + c)];
                const double m = sum / 128.0, zz = double(t.zp[size_t(r) * size_t(kbs) + size_t(b)]);
                n += 1; sz += zz; sm += m; szz += zz * zz; smm += m * m; szm += zz * m;
            }
        const double cov = szm / n - (sz / n) * (sm / n);
        const double sd = std::sqrt(std::max(0.0, szz / n - (sz / n) * (sz / n)));
        const double sm2 = std::sqrt(std::max(0.0, smm / n - (sm / n) * (sm / n)));
        const double corr = (sd > 0 && sm2 > 0) ? cov / (sd * sm2) : 0.0;
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "attention tile: the zero points sit under their own codes (corr %+0.4f, want > 0.9)",
                      corr);
        check(corr > 0.9, buf);
    }

    // the container tiles the importer derives from that tile, against the recorded ones
    std::ifstream qf(golden_dir + "/oga_attn_tile_q4_1.bin", std::ios::binary);
    std::vector<uint8_t> golden((std::istreambuf_iterator<char>(qf)), std::istreambuf_iterator<char>());
    const pi::Q4 q = pi::q_from_tiles({t});
    std::vector<uint8_t> tiled;
    pi::tile_q4_1(q, tiled);
    check(golden.size() == tiled.size(),
          "attention golden: q4_1 tile count follows from 64 rows x the 256-column grid (" +
              std::to_string(tiled.size() / onebit::npu::kTileBytes) + " tiles)");
    check(golden == tiled, "attention golden: container tiles byte-identical to the recorded ones");
    if (golden != tiled)
        for (size_t i = 0; i < std::min(golden.size(), tiled.size()); ++i)
            if (golden[i] != tiled[i]) {
                std::fprintf(stderr, "  first tile difference at byte %zu: golden %02x, ours %02x\n", i,
                             golden[i], tiled[i]);
                break;
            }

    // what the tile says, recorded in the json beside it: the bias subvolumes are per-row data
    std::ifstream jf(golden_dir + "/oga_attn_tile.json");
    const std::string text((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());
    if (!text.empty()) {
        bool seen = text.find("bias_bf16") != std::string::npos;
        for (int i = 0; i < 4; ++i) {
            char buf[16];
            std::snprintf(buf, sizeof buf, "%u", unsigned(t.bias[size_t(i)]));
            seen = seen && text.find(buf) != std::string::npos;
        }
        check(seen, "attention golden: the first bias values match the json beside the fixture");
    }
    std::printf("attention golden: 1 tile, encode/decode exact, %zu container tiles identical\n",
                tiled.size() / onebit::npu::kTileBytes);
}

// ── 3c. the .fconst gate|up unit against shipped pack bytes ─────────────────
void test_mlp_gu(const std::string& golden_dir) {
    if (golden_dir.empty()) return;
    const std::string path = golden_dir + "/oga_mlp_gu_group.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::printf("skip MLP golden: %s absent\n", path.c_str());
        return;
    }
    std::vector<uint8_t> group((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const int k = 896;
    const int64_t want = pi::group_bytes(k, 128);
    check(int64_t(group.size()) == want,
          "MLP golden: the fixture is one " + std::to_string(want) + "-byte gate|up unit of K=" +
              std::to_string(k) + " (got " + std::to_string(group.size()) + ")");
    if (int64_t(group.size()) != want) return;

    const pi::TileW g = pi::decode_tile(group.data(), k, 128);
    check(g.rows == 128 && g.code.size() == 128u * size_t(k),
          "MLP golden: the unit decodes to 128 columns");
    const std::vector<uint8_t> back = pi::encode_tile(g);
    check(back == group, "MLP golden: encode(decode(unit)) reproduces the pack's bytes exactly");
    if (back != group)
        for (size_t i = 0; i < std::min(back.size(), group.size()); ++i)
            if (back[i] != group[i]) {
                std::fprintf(stderr, "  first difference at byte %zu: ours %02x, pack %02x\n", i, back[i],
                             group[i]);
                break;
            }
    {
        pi::TileW bad = g;
        bad.code[70] ^= 0x01;
        check(pi::encode_tile(bad) != group, "MLP golden: a single changed code nibble is caught");
        pi::TileW bad2 = g;
        bad2.scale[100] *= 1.004f;
        check(pi::encode_tile(bad2) != group, "MLP golden: a one-ulp scale change is caught");
    }

    // engine#849's class at the MLP width: the round trip above cannot see the zero-point pairing (it is
    // invariant under any permutation applied on both sides), and the fused unit's 4,288-byte subvolume
    // carries 64 rows per half instead of 32, so the same two checks are made here at `rows = 128`.
    {
        const int kbs = k / 128;
        const int64_t subv = pi::group_bytes(k, 128) / (2 * kbs + 2);
        const int subv_rows = 64;                       // = rows / 2 for the fused width
        std::vector<uint8_t> raw(size_t(pi::group_bytes(k, 128)), 0);
        for (int h = 0; h < 2; ++h)
            for (int kb = 0; kb < kbs; ++kb)
                for (int i = 0; i < subv_rows / 2; ++i) {
                    // a nibble holds 0..15, and the fused width uses 32 of these bytes per half, so the
                    // pattern repeats: what matters is which row each nibble lands on, not that the 32
                    // bytes are distinct.
                    const int v = i & 0x0F;
                    raw[size_t((2 + 2 * kb + h) * subv + 64 * subv_rows + i)] = uint8_t(v | ((15 - v) << 4));
                }
        const pi::TileW z = pi::decode_tile(raw.data(), k, 128);
        bool pairs_even_odd = true;
        for (int h = 0; h < 2 && pairs_even_odd; ++h)
            for (int kb = 0; kb < kbs && pairs_even_odd; ++kb)
                for (int i = 0; i < subv_rows / 2; ++i) {
                    const uint8_t v = uint8_t(i & 0x0F);
                    pairs_even_odd = pairs_even_odd &&
                                     z.zp[size_t(h * subv_rows + 2 * i) * size_t(kbs) + size_t(kb)] == v &&
                                     z.zp[size_t(h * subv_rows + 2 * i + 1) * size_t(kbs) + size_t(kb)] ==
                                         uint8_t(15 - v);
                }
        check(pairs_even_odd, "MLP golden: zero-point nibbles pair with rows 2i / 2i+1 (even/odd) at 128 rows");

        // and on the pack's own unit: a correctly paired zero point sits where its 128 codes cluster
        double n = 0, sz = 0, sm = 0, szz = 0, smm = 0, szm = 0;
        for (int r = 0; r < 128; ++r)
            for (int b = 0; b < kbs; ++b) {
                double sum = 0;
                for (int c = 0; c < 128; ++c) sum += g.code[size_t(r) * size_t(k) + size_t(b * 128 + c)];
                const double m = sum / 128.0, zz = double(g.zp[size_t(r) * size_t(kbs) + size_t(b)]);
                n += 1; sz += zz; sm += m; szz += zz * zz; smm += m * m; szm += zz * m;
            }
        const double cov = szm / n - (sz / n) * (sm / n);
        const double sd = std::sqrt(std::max(0.0, szz / n - (sz / n) * (sz / n)));
        const double sm2 = std::sqrt(std::max(0.0, smm / n - (sm / n) * (sm / n)));
        const double corr = (sd > 0 && sm2 > 0) ? cov / (sd * sm2) : 0.0;
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "MLP golden: the zero points sit under their own codes (corr %+0.4f, want > 0.9)", corr);
        check(corr > 0.9, buf);
    }

    // the fused unit holds gate and up interleaved in 32-column chunks: both halves have to reproduce the
    // container tiles recorded from an independent decode of the same bytes
    std::ifstream qf(golden_dir + "/oga_mlp_gu_q4_1.bin", std::ios::binary);
    std::vector<uint8_t> golden((std::istreambuf_iterator<char>(qf)), std::istreambuf_iterator<char>());
    std::vector<uint8_t> tiled;
    for (const bool up : {false, true}) {
        const pi::Q4 q = pi::q_from_tiles({pi::gu_half(g, up)});
        std::vector<uint8_t> part;
        pi::tile_q4_1(q, part);
        tiled.insert(tiled.end(), part.begin(), part.end());
    }
    check(golden.size() == tiled.size(),
          "MLP golden: gate and up give " + std::to_string(tiled.size() / onebit::npu::kTileBytes) +
              " container tiles of 64 rows each");
    check(golden == tiled, "MLP golden: container tiles byte-identical to the recorded ones");
    if (golden != tiled)
        for (size_t i = 0; i < std::min(golden.size(), tiled.size()); ++i)
            if (golden[i] != tiled[i]) {
                std::fprintf(stderr, "  first tile difference at byte %zu: golden %02x, ours %02x\n", i,
                             golden[i], tiled[i]);
                break;
            }

    // the split has to be able to fail: swapping the two halves of one column-block must change the tiles
    {
        pi::TileW swapped = g;
        const size_t kb = size_t(k);
        for (int r = 0; r < 32; ++r) {
            std::swap_ranges(&swapped.code[size_t(r) * kb], &swapped.code[size_t(r) * kb] + kb,
                             &swapped.code[size_t(32 + r) * kb]);
        }
        std::vector<uint8_t> other;  // gate AND up, as `tiled` is: a gate-only stream differs in length
        for (const bool up : {false, true}) {
            std::vector<uint8_t> part;
            pi::tile_q4_1(pi::q_from_tiles({pi::gu_half(swapped, up)}), part);
            other.insert(other.end(), part.begin(), part.end());
        }
        check(other.size() == tiled.size() && other != tiled,
              "MLP golden: a gate/up swap inside a column block is caught");
    }
    std::printf("MLP golden: 1 unit, encode/decode exact, %zu container tiles identical\n",
                tiled.size() / onebit::npu::kTileBytes);
}

// A pack directory is only an OGA NPU pack if it ships the compiler's programs: `dd_metastate_*.fconst`.
// The *hybrid* flavour of these packs (Qwen2.5-0.5B-Instruct_rai_1.8.0_hybrid) ships model.bin instead and
// has no cache/ directory either -- it is a different on-disk layout, and the importer has nothing to read
// from it. That is a skip, named, not a failure: before this, the leg died with
// "FAIL exception: filesystem error: directory iterator cannot open directory: .../cache", which says
// nothing about what the pack is.
bool is_oga_npu_pack(const std::string& pack) {
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(pack, ec)) {
        if (e.path().filename().string().rfind("dd_metastate_", 0) == 0) return true;
    }
    return false;
}

void test_pack(const std::string& pack, const std::string& golden_dir) {
    const pi::Plan p = pi::plan(pack);
    const int layers = p.dims.layers;
    const bool qwen05b = p.dims.hidden == 896 && layers == 24 && p.dims.vocab == 151936;
    check(p.dims.hidden > 0 && layers > 0 && p.dims.vocab > 0,
          "pack: geometry from config.json and the embedding");
    if (qwen05b) check(true, "pack: this is the Qwen2.5-0.5B geometry the lane started from");
    const auto refused = p.refused();
    // Refusals are counted by what they are, not by a total that only holds for one family: the MLP
    // payload is open whenever its three parts fill the block stride, the final norm is always refused,
    // and the attention projections are refused only when the measured payload is not a whole number of
    // tiles -- which is the Llama and Qwen3 packs' case and not the Qwen2.5-0.5B family's. engine#734's
    // acceptance is the FIRST row: no layer norm may be refused.
    int mlp = 0, tail = 0, attn = 0, norm = 0, other = 0;
    for (const auto& t : p.tensors) {
        if (t.refusal.empty()) continue;
        const std::string& n = t.name;
        const bool is_norm = n.find("layernorm") != std::string::npos || n == "model.norm.weight";
        if (is_norm && n != "model.norm.weight") ++norm;
        else if (n == "model.norm.weight") ++tail;
        else if (n.find("mlp.") != std::string::npos) ++mlp;
        else if (n.find("self_attn") != std::string::npos) ++attn;
        else ++other;
    }
    check(norm == 0, "pack: no layer norm is refused at hidden " + std::to_string(p.dims.hidden) +
                         " (engine#734; 2 per layer planned, " + std::to_string(norm) + " refused)");
    check(mlp == (p.mlp_note.empty() ? 0 : layers * 3),
          "pack: the " + std::to_string(layers * 3) + " MLP projections are planned when their geometry "
          "adds up and refused by name when it does not (" + std::to_string(mlp) + ")");
    check(tail == 1, "pack: the final norm is the one refused norm (got " + std::to_string(tail) + ")");
    check(attn == (p.attn_note.empty() ? 0 : layers * 7),
          "pack: attention projections are refused only when the payload does not tile (" +
              std::to_string(attn) + " of " + std::to_string(layers * 7) + ")");
    check(other == 0, "pack: no refusal falls outside the named classes (" + std::to_string(other) + ")");
    bool named = true;
    for (const auto& s : refused)
        if (s.find("fconst") == std::string::npos && s.find("layer block") == std::string::npos &&
            s.find("payload") == std::string::npos)
            named = false;
    check(named, "pack: every refusal names where the bytes are and the fact that is missing");
    check(p.attn_note.empty() == (p.attn_tiles > 0),
          "pack: the attention payload's tile count was measured, or the note says why not (" +
              std::to_string(p.attn_tiles) + " tiles)");
    check(p.mlp_note.empty() == (p.mlp_gu_units > 0),
          "pack: the MLP section's geometry was measured, or the note says why not (" +
              std::to_string(p.mlp_gu_units) + " gate|up units)");
    if (p.mlp_note.empty())
        check(p.mlp_gu_units == p.dims.intermediate / 64 && p.mlp_down_groups == p.dims.hidden / 64,
              "pack: 64 columns per gate|up unit and per down_proj group");

    int norms = 0, attn_planned = 0, mlp_planned = 0, lm_tiles_planned = 0;
    for (const auto& t : p.tensors) {
        if (t.name.find("layernorm") != std::string::npos || t.name == "model.norm.weight") ++norms;
        if (t.dtype == "q4_1" && t.refusal.empty() && t.name.compare(0, 9, "lm_head.w") == 0)
            lm_tiles_planned = int((t.shape[0] / 32) * ((t.shape[1] + 255) / 256));
        if (t.dtype == "q4_1" && t.refusal.empty() && t.name.find("self_attn") != std::string::npos) ++attn_planned;
        if (t.dtype == "q4_1" && t.refusal.empty() && t.name.find(".mlp.") != std::string::npos) ++mlp_planned;
    }
    check(norms == layers * 2 + 1,
          "pack: 2 norms per layer plus the final norm are planned (the final one refused)");
    check(p.attn_note.empty() ? attn_planned == layers * 4 : true,
          "pack: the 4 attention projections per layer are planned when the payload tiles (" +
              std::to_string(attn_planned) + ")");
    check(mlp_planned == (p.mlp_note.empty() ? layers * 3 : 0),
          "pack: gate/up/down are planned when the MLP geometry adds up, refused otherwise (" +
              std::to_string(mlp_planned) + ")");
    // K is padded to the 256-column grid, so this holds for any hidden size
    const int head_rows = int(p.tensors.at(1).shape.at(0));
    check(lm_tiles_planned == (head_rows / 32) * ((p.tensors.at(1).shape.at(1) + 255) / 256),
          "pack: lm_head tile count follows from its shape and the 256-column grid");
    if (qwen05b) check(lm_tiles_planned == (151936 / 32) * 4, "pack: 0.5B lm_head plans 4748 x 4 tiles");

    // refusing must write nothing at all
    // Per-process scratch dir: two concurrent runs of this binary (or two ctest shards) otherwise write
    // into the same directory and one dies with "cannot write .../model.q4nx" (measured 2026-11-03).
    const std::string out = (fs::temp_directory_path() /
                             ("oga_pack_import_test." + std::to_string(::getpid()))).string();
    fs::remove_all(out);
    const pi::Report r0 = pi::import_pack(pack, out, pi::Options{});
    check(!r0.refused.empty() && !fs::exists(out + "/model.q4nx"),
          "pack: the refusal writes no container");
    fs::remove_all(out);
    if (golden_dir.empty()) { /* unreachable in practice */ }

    const pi::Report r = pi::import_pack(pack, out, pi::Options{true});
    // A container written with --allow-incomplete has to SAY what it is missing. The plan's manifest cannot
    // know what only the import discovers (a group that does not survive decode->encode, a nonempty bias
    // subvolume, a missing tail envelope), so the runtime refusals must be merged into it: attributed to the
    // tensor they name, and listed in `incomplete`. Checks the file itself, not the in-memory report.
    if (!r.refused.empty()) {
        std::ifstream cf(out + "/model.q4nx", std::ios::binary);
        uint64_t hlen = 0;
        cf.read(reinterpret_cast<char*>(&hlen), 8);
        std::string header(size_t(hlen), '\0');
        cf.read(header.data(), std::streamsize(hlen));
        while (!header.empty() && (header.back() == ' ' || header.back() == '\0')) header.pop_back();
        const auto doc = json::parse(header);
        const auto& inc = doc.at("__metadata__").at("incomplete");
        bool every = true;
        for (const auto& s : r.refused) {
            bool listed = false;
            for (const auto& e : inc)
                if (e.is_string() && e.get<std::string>() == s) listed = true;
            every = every && listed;
        }
        check(every, "pack: --allow-incomplete lists every runtime refusal in the manifest's incomplete "
                     "list (" + std::to_string(inc.size()) + " listed, " + std::to_string(r.refused.size()) +
                         " refused)");
        bool attributed = true;
        for (const auto& s : r.refused) {
            const size_t c = s.find(": ");
            if (c == std::string::npos) continue;
            const std::string name = s.substr(0, c);
            bool found = false;
            for (const auto& t : doc.at("__metadata__").at("tensors"))
                if (t.value("name", std::string()) == name)
                    found = t.value("imported", true) == false;
            attributed = attributed && found;
        }
        check(attributed, "pack: each runtime-refused tensor is marked imported=false in the manifest");
    }
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
    // the attention payload: every tile of every layer survives its own round trip, and what came out of
    // them is in the container -- 4 projections and, where the pack has them, the q/k/v biases
    // every tile that WAS imported survived its own round trip, whatever the geometry
    check(r.attn_tiles_ok == r.attn_tiles_total,
          "pack: every imported .fconst tile decoded and re-encoded byte-identically (" +
              std::to_string(r.attn_tiles_ok) + "/" + std::to_string(r.attn_tiles_total) + ")");
    if (r.attn_tiles_total) check(r.attn_projections == layers * 4,
                                  "pack: 4 attention projections per layer were imported (" +
                                      std::to_string(r.attn_projections) + ")");
    if (qwen05b)  // the geometry the tile layout is measured on
        check(r.attn_tiles_total == int64_t(layers) * 32 && r.attn_bias_tensors == layers * 3,
              "pack: 0.5B imports 32 tiles and 3 biases per layer (" +
                  std::to_string(r.attn_tiles_total) + " tiles, " +
                  std::to_string(r.attn_bias_tensors) + " biases)");

    // the MLP section, same contract: every group decoded and re-encoded byte-identically, and what came
    // out of them is in the container
    if (p.mlp_note.empty()) {
        check(r.mlp_groups_total > 0 && r.mlp_groups_ok == r.mlp_groups_total,
              "pack: every MLP group decoded and re-encoded byte-identically (" +
                  std::to_string(r.mlp_groups_ok) + "/" + std::to_string(r.mlp_groups_total) + ")");
        check(r.mlp_projections == layers * 3,
              "pack: gate/up/down were imported for every layer (" +
                  std::to_string(r.mlp_projections) + ")");
        if (qwen05b)
            check(r.mlp_groups_total == int64_t(layers) * (p.dims.intermediate / 64 + p.dims.hidden / 64),
                  "pack: 0.5B imports 76 gate|up units and 14 down groups per layer (" +
                      std::to_string(r.mlp_groups_total) + " groups)");
    }

    // the container must open with the engine's own loader
    onebit::npu::Model m(out);
    const auto& head = m.tensor("lm_head.weight");
    check(head.shape.at(0) == int64_t((head_rows / 32) * ((p.tensors.at(1).shape.at(1) + 255) / 256)),
          "pack: lm_head tile count matches the container");
    check(m.tensor("model.layers.0.input_layernorm.weight").shape.at(0) == p.dims.hidden,
          "pack: a layer norm is `hidden` bf16 values, whatever the hidden size");
    if (r.attn_projections) {
        const auto& qw = m.tensor("model.layers.0.self_attn.q_proj.weight");
        check(qw.shape.at(1) == int64_t(onebit::npu::kTileBytes),
              "pack: an imported projection is q4_1 chunks of the container's tile size");
        // Not every family carries attention biases: Llama-3.2 has none, and its attention section now
        // tiles (engine#979), so the check has to ask the pack rather than the geometry.
        if (r.attn_bias_tensors)
            check(m.tensor("model.layers.0.self_attn.q_proj.bias").shape.at(0) ==
                      int64_t(p.dims.heads * p.dims.head_dim),
                  "pack: the q_proj bias is one bf16 value per output row");
    }
    if (r.mlp_projections) {
        const auto& gw = m.tensor("model.layers.0.mlp.gate_proj.weight");
        const auto& dw = m.tensor("model.layers.0.mlp.down_proj.weight");
        check(gw.shape.at(1) == int64_t(onebit::npu::kTileBytes) && dw.shape.at(1) == int64_t(onebit::npu::kTileBytes),
              "pack: the MLP projections are q4_1 chunks of the container's tile size");
        check(gw.shape.at(0) == int64_t((p.dims.intermediate / 32) * ((p.dims.hidden + 255) / 256)),
              "pack: gate_proj's tile count follows from its shape and the 256-column grid");
        check(dw.shape.at(0) == int64_t((p.dims.hidden / 32) * ((p.dims.intermediate + 255) / 256)),
              "pack: down_proj's tile count follows from its shape and the 256-column grid");
    }
    check(m.dims().hidden == p.dims.hidden && m.dims().layers == layers,
          "pack: config.json drives the loader");
    fs::remove_all(out);
}

}  // namespace


// ── the epsilon anchor is the pack's own, not a constant (engine#734) ───────
// The layer-block locator searched for bf16 1e-6 (bytes 86 35), which is what every Qwen pack declares;
// amd/Llama-3.2-1B declares rms_norm_eps 1e-5 (bf16 0x3728, bytes 28 37) and so had no anchors at all and
// refused all 33 of its layer norms.  Two crafted .fconst files here, one per epsilon: each must locate the
// same geometry with its own epsilon, and each must find NOTHING with the other one.
// ── the attention payload's offset is derived from hidden, not a constant (engine#979) ──────────────
// The importer located the payload at the constant 1,856, which is correct only for hidden 896 (the 0.5B
// pack: 64 + 2 x 896). Every other pack was then measured from 2 x (hidden - 896) bytes early, so its
// payload came out as "whole tiles plus a stray run" and the attention section was refused. The two
// numbers below are the packs' OWN measured second-header offsets, so this is a known-answer test: with
// the derived offset each section is exactly the tiles q/k/v/o need, and with the old constant it is not.
void test_attn_payload_off() {
    check(pi::attn_payload_off(896) == 1856,
          "attn off: hidden 896 is the 1,856 the importer shipped with (64 + 2 x 896)");
    check(pi::attn_payload_off(2048) == 4160, "attn off: hidden 2,048 gives 4,160, not 1,856");
    check(pi::attn_payload_off(2560) == 5184, "attn off: hidden 2,560 gives 5,184");

    // Llama-3.2-1B (hidden 2,048, q/k/v/o = 2048/512/512/2048 rows): the second header is at 5,922,880 B.
    const int64_t h = 2048, h1 = 5922880, tile = pi::attn_tile_bytes(h);
    const int64_t derived = h1 - pi::attn_payload_off(h);
    check(tile == 73984, "attn off: a hidden-2,048 tile is 73,984 B");
    check(derived == 80 * tile,
          "attn off: Llama's payload is exactly 80 tiles from the derived offset (" + std::to_string(derived) +
              " B), and " + std::to_string(h1 - 1856) + " B from the old constant = 80 tiles + " +
              std::to_string(h1 - 1856 - 80 * tile) + " strip");
    const int want = (512 / 64) * 2 + (2048 / 64) * 2;
    check(derived / tile == want, "attn off: the derived count is the 80 tiles q/k/v/o need (" +
                                      std::to_string(want) + ")");
    // The 0.5B is unchanged by the fix, which is why it never showed the bug.
    const int64_t h1_05 = 1115968;
    check(h1_05 - pi::attn_payload_off(896) == 32 * pi::attn_tile_bytes(896),
          "attn off: the 0.5B payload is still 32 tiles at the derived offset");
    std::printf("attn off: 896 -> 1,856 B, 2,048 -> 4,160 B, 2,560 -> 5,184 B; Llama's 5,922,880 B header is "
                "exactly %d tiles (the old constant gave %lld B = %d tiles + %lld strip)\n",
                int(derived / tile), (long long)(h1 - 1856), int((h1 - 1856) / tile),
                (long long)(h1 - 1856 - 80 * tile));
}

// engine#979: the .fconst attention section is not always one uniform run of q/k/v/o tiles. The Qwen3
// packs carry FOUR epsilon headers per layer block (the Qwen2.5/Llama packs carry two), and the three
// sections between them decompose to the byte as [k,q,v tiles][header + per-head k_norm][header +
// per-head q_norm + o_proj tiles]. The offsets below are the packs' OWN measured header positions; the
// pack's own metadata names the two norms (attn/k_norm/MLADFRMSNORM before attn/q_norm/MLADFRMSNORM in
// the Token program's op_list) and declares `o_proj N=2560 K=4096` on the 4B, i.e. o_proj's K is
// q_proj's output width, not hidden. The two 256-byte gains match Qwen/Qwen3-1.7B's and -4B's public
// q_norm/k_norm weights bit for bit at those offsets (128 of 128 values, and 127-128 of 128 on the swap).
void test_qwen3_attn_sections() {
    struct Pack { const char* name; int64_t hidden, q_out, kv_out, hd, h0, h1, h2; };
    const Pack packs[] = {{"Qwen3-1.7B", 2048, 2048, 1024, 128, 4739136, 4739456, 7107264},
                          {"Qwen3-4B", 2560, 4096, 1024, 128, 8778816, 8779136, 14524096}};
    for (const Pack& p : packs) {
        const int64_t tile = pi::attn_tile_bytes(p.hidden);
        const int64_t tile_o = pi::attn_tile_bytes(p.q_out);
        const int64_t norm = 64 + 2 * p.hd;  // a 64-byte header plus one bf16 [head_dim] gain
        const int64_t s0 = p.h0 - pi::attn_payload_off(p.hidden);
        const int64_t s1 = p.h1 - p.h0;
        const int64_t s2 = p.h2 - p.h1;
        const int64_t kv_tiles = (p.kv_out / 64) * 2 + p.q_out / 64;  // k, q, v in that order
        const std::string n = p.name;
        check(s0 == kv_tiles * tile,
              n + ": the payload before the first per-head norm is " + std::to_string(s0) +
                  " B, and k/q/v are " + std::to_string(kv_tiles) + " tiles = " +
                  std::to_string(kv_tiles * tile));
        check(s1 == norm, n + ": the section between the two per-head norms is " + std::to_string(s1) +
                              " B, a header plus a bf16 [128] gain is " + std::to_string(norm));
        check(s2 == norm + (p.hidden / 64) * tile_o,
              n + ": the section holding o_proj is " + std::to_string(s2) + " B, a header, a gain and " +
                  std::to_string(p.hidden / 64) + " o_proj tiles at K = q_proj's " +
                  std::to_string(p.q_out) + " are " + std::to_string(norm + (p.hidden / 64) * tile_o));
        // The negative control: with o_proj's K taken as hidden the same rows are a different size, and
        // with all four projections in one run the first section cannot hold them. Both are what the
        // importer refused before this fix, so the check is that the measured numbers are NOT those.
        if (p.q_out != p.hidden)
            check(s2 != norm + (p.hidden / 64) * tile,
                  n + ": o_proj's section is not 40 tiles at K = hidden (" +
                      std::to_string(norm + (p.hidden / 64) * tile) + " B), which is how the pack states "
                      "o_proj's K is q_proj's width");
        check(s0 != (kv_tiles + p.hidden / 64) * tile,
              n + ": q/k/v/o do not share one uniform run (that would be " +
                  std::to_string((kv_tiles + p.hidden / 64) * tile) + " B against a measured " +
                  std::to_string(s0) + ")");
        std::printf("qwen3 sections: %s -> k/q/v %d tiles, norm %lld B, norm+o_proj %lld B (%d tiles at "
                    "K=%lld)\n",
                    p.name, int(kv_tiles), (long long)s1, (long long)s2, int(p.hidden / 64),
                    (long long)p.q_out);
    }
}

// engine#1003: the post_attention_layernorm weight is the bytes after the block's post-attention header.
// On a 2-header block that is the second header; on a Qwen3 4-header block it is the FOURTH, because the
// second and third open the per-head QK norms. Reading the second one on every pack imported the k_norm
// section's padding as post_attention_layernorm -- measured on Qwen3-1.7B: 1,920 of 2,048 values zero,
// max 6.97, against the pack's real post-attention norm (all nonzero, 0.9919 correlated with the upstream
// checkpoint). The crafted pack below puts a different recognisable value behind each header, so the check
// is a known answer rather than a shape: the imported weight must be the last header's ramp and must NOT
// be the second header's marker.
void test_post_attn_norm_header(const std::string& root) {
    const fs::path dir = fs::path(root) / "post-attn-hdr";
    fs::remove_all(dir);
    fs::create_directories(dir / "cache");
    const int H = 128, L = 8;                 // locate_fconst wants >= 6 headers in the first GiB
    const int64_t stride = 1 << 20;           // the locator ignores periods below 1e6
    const int64_t h2 = 262144, h3 = 524288, h4 = 786432;   // the three in-block headers after the first
    std::string fc(size_t(int64_t(L) * stride + 64 + H * 2), '\0');
    auto put_bf16 = [&](int64_t off, float v) {
        const uint16_t bits = pi::bf16_bits_of(v);
        fc[size_t(off)] = char(bits & 0xFF);
        fc[size_t(off) + 1] = char(bits >> 8);
    };
    for (int l = 0; l < L; ++l) {
        const int64_t base = int64_t(l) * stride;
        for (int64_t h : {int64_t(0), h2, h3, h4}) put_bf16(base + h, 1e-6f);   // bf16 0x3586, the Qwen eps
        for (int i = 0; i < H; ++i) {
            put_bf16(base + h2 + 64 + 2 * i, -3.0f);          // the k_norm section: NOT the post-attention norm
            put_bf16(base + h4 + 64 + 2 * i, float(i + 1));   // the post-attention norm
        }
    }
    put_bf16(int64_t(L) * stride, 1e-6f);     // the tail envelope's header
    put_file((dir / "config.json").string(),
             R"({"model_type":"qwen3","hidden_size":128,"intermediate_size":256,"num_hidden_layers":8,)"
             R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,"vocab_size":128,)"
             R"("rope_theta":1000000.0,"tie_word_embeddings":true,"rms_norm_eps":1e-06})");
    put_file((dir / "model.onnx.data").string(), std::string(size_t(H) * size_t(H) * 4, '\x11'));
    put_file((dir / "model.onnx").string(),
             pb::model({pb::tensor("model.embed_tokens.weight", {H, H}, "model.onnx.data", 0,
                                   int64_t(H) * H * 4)}, {}));
    const std::string tok = (dir / "dd_metastate_Llm_Token_Token_crafted.fconst").string();
    put_file(tok, fc);

    const pi::FconstGeom g = pi::locate_fconst(tok, 0x3586);
    check(g.stride == stride && g.period == 4,
          "post-attn header: the crafted 4-header .fconst is locatable (stride " +
              std::to_string(g.stride) + ", period " + std::to_string(g.period) +
              (g.why.empty() ? "" : ", why: " + g.why) + ")");
    check(g.h1 == h2, "post-attn header: h1 is the block's second header (" + std::to_string(g.h1) + ")");
    check(g.post_attn_hdr() == h4,
          "post-attn header: the post-attention header on a 4-header block is the LAST one (" +
              std::to_string(g.post_attn_hdr()) + " vs " + std::to_string(h4) + ")");
    check(g.post_attn_hdr() != g.h1,
          "post-attn header: it is not h1, which is where the k_norm section opens");
    {   // the 2-header case must be unchanged by the rule
        std::string two(size_t(int64_t(L) * stride + 64 + H * 2), '\0');
        for (int l = 0; l < L; ++l) {
            const int64_t base = int64_t(l) * stride;
            two[size_t(base)] = char(0x86);
            two[size_t(base) + 1] = char(0x35);
            two[size_t(base + h2)] = char(0x86);
            two[size_t(base + h2) + 1] = char(0x35);
        }
        two[size_t(int64_t(L) * stride)] = char(0x86);
        const std::string p2 = (dir / "dd_metastate_Llm_Token_Token_two.fconst").string();
        put_file(p2, two);
        const pi::FconstGeom g2 = pi::locate_fconst(p2, 0x3586);
        check(g2.period == 2 && g2.post_attn_hdr() == g2.h1,
              "post-attn header: on a 2-header block it is still h1 (" + std::to_string(g2.h1) + ")");
        fs::remove(p2);
    }

    pi::Options allow;
    allow.allow_incomplete = true;   // the crafted block has no tiles, so the payload legs are refused
    const std::string out = (fs::temp_directory_path() /
                             ("oga_pack_import_pa." + std::to_string(::getpid()))).string();
    fs::remove_all(out);
    try {
        const pi::Report r = pi::import_pack(dir.string(), out, allow);
        check(r.stride == stride, "post-attn header: the import read the crafted stride");
        check(fs::exists(fs::path(out) / "model.q4nx"),
              "post-attn header: the import wrote a container to inspect");
        if (fs::exists(fs::path(out) / "model.q4nx")) {
            onebit::npu::Model m(out);
            for (int l : {0, L - 1}) {
                const std::string n = "model.layers." + std::to_string(l) +
                                      ".post_attention_layernorm.weight";
                const auto& t = m.tensor(n);
                bool ramp = t.bytes >= size_t(H) * 2, not_marker = true;
                for (int i = 0; ramp && i < H; ++i) {
                    const uint16_t want = pi::bf16_bits_of(float(i + 1));
                    ramp = t.data[2 * i] == uint8_t(want & 0xFF) && t.data[2 * i + 1] == uint8_t(want >> 8);
                    const uint16_t marker = pi::bf16_bits_of(-3.0f);
                    not_marker = not_marker && !(t.data[2 * i] == uint8_t(marker & 0xFF) &&
                                                 t.data[2 * i + 1] == uint8_t(marker >> 8));
                }
                check(ramp, n + " is the bytes after the block's LAST header");
                check(not_marker, n + " is not the k_norm section's marker (the engine#1003 bug)");
            }
        }
    } catch (const std::exception& e) {
        check(false, std::string("post-attn header: the crafted pack imported: ") + e.what());
    }
    fs::remove_all(out);
    fs::remove_all(dir);
}

void test_fconst_eps(const std::string& root) {    const fs::path dir = fs::path(root) / "fconst-eps";
    fs::remove_all(dir);
    fs::create_directories(dir);

    check(pi::bf16_bits_of(1e-6f) == 0x3586, "fconst eps: 1e-6 is bf16 0x3586 (bytes 86 35)");
    check(pi::bf16_bits_of(1e-5f) == 0x3728, "fconst eps: 1e-5 is bf16 0x3728 (bytes 28 37)");

    const int blocks = 16;                    // Llama-3.2-1B's layer count
    const int64_t stride = 1048576;           // the locator ignores periods below 1e6
    const int64_t second = 262144;            // the post-attention header, inside the block
    auto write_pack = [&](const std::string& name, unsigned short bits) {
        std::string bytes(size_t(int64_t(blocks + 1) * stride), '\0');
        auto header = [&](int64_t off) {
            bytes[size_t(off)] = char(bits & 0xFF);
            bytes[size_t(off) + 1] = char(bits >> 8);
        };
        for (int L = 0; L < blocks; ++L) {
            header(int64_t(L) * stride);
            header(int64_t(L) * stride + second);
        }
        header(int64_t(blocks) * stride);     // the tail header, exactly at the end of the last block
        const std::string path = (dir / name).string();
        put_file(path, bytes);
        return path;
    };

    const std::string qwen = write_pack("dd_metastate_Llm_Token_Token_qwen.fconst", 0x3586);
    const std::string llama = write_pack("dd_metastate_Llm_Token_Token_llama.fconst", 0x3728);

    for (const auto& [path, bits, label] : {std::tuple<std::string, unsigned short, std::string>{qwen, 0x3586, "qwen 1e-6"},
                                            {llama, 0x3728, "llama 1e-5"}}) {
        const pi::FconstGeom g = pi::locate_fconst(path, bits);
        check(g.stride == stride && g.h1 == second && g.period == 2,
              "fconst eps: " + label + " anchors give stride " + std::to_string(g.stride) + ", second header " +
                  std::to_string(g.h1) + ", period " + std::to_string(g.period));
        check(g.blocks == blocks * 2 + 1,
              "fconst eps: " + label + " found every header including the tail (" + std::to_string(g.blocks) + ")");
        const unsigned short other = bits == 0x3586 ? 0x3728 : 0x3586;
        const pi::FconstGeom w = pi::locate_fconst(path, other);
        check(w.stride == 0 && !w.why.empty(),
              "fconst eps: " + label + " finds no anchor with the other epsilon, and says why");
        std::printf("fconst eps: %-9s stride %lld, second header %lld, period %d, %d headers, 0 anchors with the "
                    "other epsilon\n",
                    label.c_str(), (long long)g.stride, (long long)g.h1, g.period, g.blocks);
    }
    fs::remove_all(dir);
}

int main(int argc, char** argv) {
    const std::string golden_dir = argc > 1 ? argv[1] : "";
    const std::string pack = argc > 2 ? argv[2] : "";
    // The pack leg is the only one that can regress a real pack, and ctest runs this binary with the
    // golden dir only (CMakeLists add_test npu_pack_import), so the leg self-skips. That skip must not
    // be reported as a pass: three of the four packs on this box were red behind a green
    // "all npu-pack checks passed" line (engine#842 review). The final line now says which legs ran.
    const bool pack_given = !pack.empty() && fs::exists(pack + "/model.onnx");
    const bool pack_ran = pack_given && is_oga_npu_pack(pack);
    if (pack_given && !pack_ran)
        std::printf("skip pack: %s ships no dd_metastate_*.fconst, so it is the hybrid/model.bin flavour of "
                    "these packs and not an OGA NPU pack this importer reads\n", pack.c_str());
    try {
        test_bounds();
        test_crafted_refusals(golden_dir.empty() ? "." : golden_dir);
        test_matmulnbits();
        test_tiles(golden_dir);
        test_attn_tile(golden_dir);
        test_mlp_gu(golden_dir);
        test_fconst_eps(golden_dir.empty() ? "." : golden_dir);
        test_attn_payload_off();
        test_qwen3_attn_sections();
        test_post_attn_norm_header(golden_dir.empty() ? "." : golden_dir);
        if (pack_ran) test_pack(pack, golden_dir);
        else std::printf("skip pack: no pack directory given (the crafted and golden legs only)\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all npu-pack checks passed%s\n",
                pack_ran ? " (including the pack leg)"
                         : (pack_given ? " -- pack leg SKIPPED, not an OGA NPU pack"
                                       : " -- pack leg SKIPPED, no pack directory given"));
    return 0;
}
