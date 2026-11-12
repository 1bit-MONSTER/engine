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
// 5. With a pack dir: `1bit npu-pack` imports every tensor of the pack -- the attention projections and
//    their biases, the MLP projections and the final norm -- and produces a container the engine's own
//    Model can open.

#include "../npu/forward/include/file_create.h"   // written files are 0600, never through a symlink
#include "model.h"
#include "pack.h"
#include "pack_import.h"
#include "q4nx.h"
#include "q4nx_device.h"

#include <set>
#include <nlohmann/json.hpp>

#include <unistd.h>

#include <algorithm>
#include <thread>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "tmp_root.h"   // engine#2029: scratch goes on disk, not in tmpfs

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
                   uint64_t offset, uint64_t length, int64_t type = 1) {
    std::string t;
    for (auto d : dims) ints(t, 1, d);                 // dims
    ints(t, 2, type);   // data_type: FLOAT unless the caller says otherwise (a quantised table is UINT8)
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
// An initializer with a name and nothing else: the `*.qweight.empty` placeholders a 1.9 external-data pack puts in
// the node's input slot while the bytes live in a `*.qweight.packed` sibling (engine#1503).
std::string placeholder(const std::string& name) {
    std::string t;
    ints(t, 2, 1);      // data_type FLOAT
    str(t, 8, name);
    return t;
}
std::string matmulnbits(const std::string& name, const std::vector<std::string>& ins, int64_t K, int64_t N,
                        int64_t bits, int64_t block, const std::string& op = "MatMulNBits") {
    std::string n;
    for (const auto& i : ins) str(n, 1, i);
    str(n, 4, op);
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
std::string if_node(const std::string& name, const std::string& in, const std::string& out,
                     const std::string& branch, const std::string& graph) {
    std::string n;
    str(n, 1, in);
    str(n, 2, out);
    str(n, 4, "If");
    std::string a;
    str(a, 1, branch);
    tag(a, 6, 2);                                      // AttributeProto.graph = the branch GraphProto
    varint(a, graph.size());
    a += graph;
    tag(n, 5, 2);
    varint(n, a.size());
    n += a;
    return n;
}
// A bare GraphProto -- what an `If` branch attribute carries (ModelProto wraps one, a branch is one).
std::string graph(const std::vector<std::string>& tensors, const std::vector<std::string>& nodes) {
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
    return g;
}
std::string model(const std::vector<std::string>& tensors, const std::vector<std::string>& nodes) {
    const std::string g = graph(tensors, nodes);
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

// A model.q4nx tensor table, as npu/model.cpp reads it (8-byte header length, JSON, then data).
json container_table(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    uint64_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 8);
    std::string header(size_t(hlen), '\0');
    f.read(header.data(), std::streamsize(hlen));
    while (!header.empty() && (header.back() == ' ' || header.back() == '\0')) header.pop_back();
    return json::parse(header);
}

// The container's payload, i.e. what `data_offsets` are measured from: after the 8-byte header length and the JSON.
std::vector<uint8_t> container_payload(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    uint64_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 8);
    f.seekg(std::streamoff(8 + hlen));
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// How many I8 tensors of a container are the imported 6,144-byte q4_1f form -- exactly the
// trigger of the forward loader's refusal (npu/forward/src/model.c, engine#1185).
int q4_1f_tensors(const json& table) {
    int n = 0;
    for (const auto& [name, t] : table.items()) {
        if (name == "__metadata__" || !t.is_object() || t.value("dtype", std::string()) != "I8") continue;
        const auto shape = t.value("shape", std::vector<int64_t>{});
        if (!shape.empty() && shape.back() == int64_t(onebit::npu::kTileBytesF32)) ++n;
    }
    return n;
}

// The 1.9 external-data genre (engine#1503, measured on phi-4): every node's qweight slot holds a
// `*.qweight.empty` placeholder with no dims, and the bytes are in a `*.qweight.packed` sibling the node never
// references. A reader that follows node inputs only reports a weightless graph for a 9.7 GB pack; one that sees the
// sibling must also say whether the bytes fit the stride the .fconst tile already carries --
// length == N x (K/128 + 1) x 68 -- as a count, not an assumption.
// The external-data tile's own decode (engine#1503): a synthetic blob laid out the way the pack lays it out must come
// back exactly, under BOTH code orderings -- the decoder's constant picks one, and the test holds each to the spec it
// claims, so flipping the constant cannot silently change what the other branch means.
void test_external_tile_decode() {
    const int N = 128, K = 256;                       // 2 tiles of 64 rows, 2 K-blocks -> 3 subvolume pairs
    const int64_t tiles = N / 64, per_tile = 2 * (K / 128 + 1);
    const int64_t bytes = tiles * per_tile * 2176;
    // A code matrix and per-block scale/zero-points to encode, then to find again.
    std::vector<uint8_t> codes(size_t(N) * K);
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = uint8_t((i * 7 + i / 13) % 16);
    const auto bf16_bits = [](float v) {
        unsigned int u;
        std::memcpy(&u, &v, 4);
        return uint16_t((u + 0x8000u + ((u >> 16) & 1u)) >> 16);   // round to nearest even, as the importer does
    };
    for (const bool row_major : {false, true}) {
    // DISTINCT scale and zero points per (tile, half, kk, r): a fixture that writes one value everywhere lets a decoder
    // that reads row 0's scale for all 32 rows pass, which is exactly the addressing this test exists to pin.
    const auto scale_for = [&](int64_t tile, int half, int64_t kk, int r) {
        return 0.001953125f * float(1 + int(tile) * 7 + half * 5 + int(kk) * 3 + (r % 4));   // bf16-exact multiples of 2^-9
    };
    const auto zero_for = [&](int64_t tile, int half, int64_t kk, int r, int g) {
        return uint8_t((int(tile) * 3 + half * 5 + int(kk) * 7 + r + g) % 16);
    };
    std::vector<uint8_t> blob(size_t(bytes), 0);
    for (int64_t tile = 0; tile < tiles; ++tile)
        for (int half = 0; half < 2; ++half)
            for (int64_t kk = 1; kk <= K / 128; ++kk) {
                uint8_t* subv = blob.data() + (tile * per_tile + kk * 2 + half) * 2176;
                for (int r = 0; r < 32; ++r) {
                    // The decoder reads the block's FIRST 32 rows from the pair's SECOND subvolume, so the fixture encodes
                    // its rows in that order; without this the test pins a mapping the decoder does not use (engine#1503).
                    const int64_t row = tile * 64 + (1 - half) * 32 + r;
                    const uint16_t sc = bf16_bits(scale_for(tile, half, kk, r));
                    subv[2112 + r * 2] = uint8_t(sc & 0xFF);
                    subv[2113 + r * 2] = uint8_t(sc >> 8);
                    subv[2048 + r * 2] = uint8_t(zero_for(tile, half, kk, r, 0) | (zero_for(tile, half, kk, r, 1) << 4));
                    subv[2049 + r * 2] = uint8_t(zero_for(tile, half, kk, r, 2) | (zero_for(tile, half, kk, r, 3) << 4));
                    for (int c = 0; c < 128; ++c) {
                        const uint8_t nib = codes[size_t(row) * K + (kk - 1) * 128 + c];
                        if (row_major) {
                            if (c % 2 == 0) subv[r * 64 + c / 2] = uint8_t((subv[r * 64 + c / 2] & 0xF0) | nib);
                            else subv[r * 64 + c / 2] = uint8_t((subv[r * 64 + c / 2] & 0x0F) | (nib << 4));
                        } else {
                            const int cp = r / 2;      // one byte per column, holding the row pair 2cp / 2cp+1
                            uint8_t& b = subv[(cp % 4) + c * 4 + (cp / 4) * 512];
                            if (r % 2 == 0) b = uint8_t((b & 0xF0) | nib);
                            else b = uint8_t((b & 0x0F) | (nib << 4));
                        }
                    }
                }
            }
    std::string err;
    const pi::Q4 q = pi::decode_external_tile(blob.data(), blob.size(), N, K, err, row_major);
    check(err.empty() && q.N == N && q.K == K && q.block == 32,
          "external tile: a blob of the measured size decodes (" + err + ")");
    bool codes_ok = q.codes.size() == codes.size(), scale_ok = true, zero_ok = true;
    // The pack's codes are int4 TWO'S COMPLEMENT; the container's form is the offset from the group minimum, i.e.
    // c_signed + 8. A test that expected the raw nibble passed on the unsigned bug (engine#1503).
    for (size_t i = 0; codes_ok && i < codes.size(); ++i) {
        const int raw = codes[i];
        const int cs = (raw >= 8) ? raw - 16 : raw;
        if (q.codes[i] != uint8_t(cs + 8)) codes_ok = false;
    }
    // Every row's own scale and all four of its zero points, so the addressing (row r*2 in the param regions, the
    // group index (kk-1)*4+g, the half and tile choice) is pinned rather than the value of one row repeated.
    for (int64_t tile = 0; tile < tiles; ++tile)
        for (int half = 0; half < 2; ++half)
            for (int64_t kk = 1; kk <= K / 128; ++kk)
                for (int r = 0; r < 32; ++r) {
                    // The decoder reads the block's FIRST 32 rows from the pair's SECOND subvolume, so the fixture encodes
                    // its rows in that order; without this the test pins a mapping the decoder does not use (engine#1503).
                    const int64_t row = tile * 64 + (1 - half) * 32 + r;
                    for (int g = 0; g < 4; ++g) {
                        const size_t at = size_t(row) * (K / 32) + (kk - 1) * 4 + g;
                        if (q.scale[at] != scale_for(tile, half, kk, r)) scale_ok = false;
                        // and the container's zero is the offset, m = (-8 - z) * scale, for a signed int4 z
                        const int zraw = zero_for(tile, half, kk, r, g);
                        const int zs = (zraw >= 8) ? zraw - 16 : zraw;
                        if (q.zero[at] != float(-8 - zs) * scale_for(tile, half, kk, r)) zero_ok = false;
                    }
                }
    check(codes_ok, std::string("external tile: every 4-bit code comes back exactly, ") + (row_major ? "row-major" : "w8 interleave"));
    check(scale_ok, "external tile: every row keeps its own bf16 scale, repeated over its four groups");
    check(zero_ok, "external tile: every (row, group) keeps its own int4 zero point");
    std::string err2;
    (void)pi::decode_external_tile(blob.data(), blob.size() - 2176, N, K, err2);
    check(!err2.empty() && err2.find("is not") != std::string::npos,
          "external tile: a blob one subvolume short is refused with the arithmetic (" + err2 + ")");
    }
}

// The conversion half of the writer (engine#1503): the pack's `(code - zp)*scale` and the container's
// `code*d + m` are different conventions, and the mapping has to reproduce the pack's own numbers. The obvious
// alternative -- dequantise and re-quantise with the container's quantiser -- was measured on a real phi-4 tile and
// reproduced the pack's codes on only 2,826 of 8,192 entries, so the identity below is the thing to pin.
void test_external_to_container() {
    pi::Q4 pack;
    pack.N = 4; pack.K = 128; pack.block = 32;
    pack.codes.assign(size_t(pack.N) * pack.K, 0);
    for (size_t i = 0; i < pack.codes.size(); ++i) pack.codes[i] = uint8_t((i * 5 + 1) % 16);
    pack.scale.assign(size_t(pack.N) * (pack.K / 32), 0.0068359375f);   // exact in bf16
    pack.zero.assign(pack.scale.size(), 0.0f);
    for (int r = 0; r < pack.N; ++r)
        for (int g = 0; g < pack.K / 32; ++g) pack.zero[size_t(r) * (pack.K / 32) + g] = float(3 + (g % 4));

    long long m_inexact = -1;
    const pi::Q4 q = pi::container_from_external_q4(pack, &m_inexact);
    // The conversion is a PASS-THROUGH: decode_external_tile already returns the container's convention, so the
    // identity to pin is that nothing is converted a SECOND time. It was: `zero` became -((-8-z)*scale)*scale ~
    // +0.0002 instead of -0.04, every weight sat at code*scale -- a near-constant POSITIVE matrix -- and a down
    // projection summed to K*mean(act)*0.04 ~ 110,000 where a healthy layer gives ~1 (engine#1503).
    check(q.codes == pack.codes, "external->container: codes pass through unchanged");
    check(q.scale == pack.scale, "external->container: scales pass through unchanged");
    check(q.zero == pack.zero, "external->container: the offset passes through unchanged (the mapping is NOT re-applied)");
    check(m_inexact >= 0, "external->container: the bf16-exactness of the offset is reported, not assumed");
}

// The writer's role mapping (engine#1503): the graph's own names, mapped onto the roles the tile writer already
// speaks, with an unknown name reported rather than skipped -- a projection nothing maps is a weight that would
// silently not be written.
void test_external_roles() {
    struct Case { const char* name; const char* role; };
    const Case cases[] = {
        {"model.layers.0.attn.qkv_proj.MatMulNBits.qweight.packed", "QKV"},
        {"model.layers.39.attn.o_proj.MatMulNBits.qweight.packed", "O"},
        {"model.layers.7.mlp.gate_proj.MatMulNBits.qweight.packed", "G"},
        {"model.layers.7.mlp.up_proj.MatMulNBits.qweight.packed", "U"},
        {"model.layers.7.mlp.down_proj.MatMulNBits.qweight.packed", "D"},
        {"lm_head.MatMulNBits.qweight.packed", "HEAD"},
    };
    for (const auto& c : cases) {
        const char* got = pi::external_role_of(c.name);
        check(got && std::string(got) == c.role,
              std::string("external role: ") + c.name + " -> " + c.role + " (got " + (got ? got : "null") + ")");
    }
    // The three that must NOT be mistaken for a projection: a norm, the embedding, and a name from another genre.
    check(pi::external_role_of("model.layers.0.input_layernorm.weight") == nullptr,
          "external role: a norm is not a projection");
    check(pi::external_role_of("model.embed_tokens.weight") == nullptr,
          "external role: the embedding is not a projection");
    check(pi::external_role_of("model.layers.0.attn.qkv_proj_matmul") == nullptr,
          "external role: an unknown suffix is not guessed");
}

// The assembly's naming half (engine#1503): the container stores q/k/v separately while the pack stores them as one
// matmul, so the split is geometry, not a guess. phi-4's qkv is N 7680 = 40 heads x 128 (q) + 2 x 10 kv heads x 128.
void test_external_container_names() {
    std::vector<pi::ContainerTensor> t;
    std::string err;
    check(pi::external_container_tensors("QKV", 40, 10, 128, t, err) && t.size() == 3,
          "container names: QKV splits into three tensors");
    check(t.size() == 3 && t[0].suffix == "self_attn.q_proj.weight" && t[0].row_lo == 0 && t[0].row_hi == 5120,
          "container names: q takes rows [0, nh*hd)");
    check(t.size() == 3 && t[1].suffix == "self_attn.k_proj.weight" && t[1].row_lo == 5120 && t[1].row_hi == 6400,
          "container names: k follows q, nkv*hd rows");
    check(t.size() == 3 && t[2].suffix == "self_attn.v_proj.weight" && t[2].row_lo == 6400 && t[2].row_hi == 7680,
          "container names: v closes the range, which must equal the pack's N exactly");
    for (const char* role : {"O", "G", "U", "D", "HEAD"}) {
        std::vector<pi::ContainerTensor> one;
        std::string e2;
        check(pi::external_container_tensors(role, 40, 10, 128, one, e2) && one.size() == 1,
              std::string("container names: ") + role + " is one tensor");
    }
    std::vector<pi::ContainerTensor> bad;
    std::string e3;
    check(!pi::external_container_tensors("MOE_EXPERT", 40, 10, 128, bad, e3) && !e3.empty(),
          "container names: a role with no container tensor is refused, naming it (" + e3 + ")");
    std::string e4;
    check(!pi::external_container_tensors("QKV", 0, 0, 0, bad, e4) && e4.find("geometry") != std::string::npos,
          "container names: a QKV split with no geometry is refused by name");
}

void test_external_packed_genre(const std::string& root) {
    const fs::path dir = fs::path(root) / "external-packed";
    fs::remove_all(dir);
    fs::create_directories(dir / "cache");
    put_file((dir / "config.json").string(),
             R"({"model_type":"phi3","hidden_size":5120,"intermediate_size":17920,"num_hidden_layers":1,)"
             R"("num_attention_heads":40,"num_key_value_heads":10,"head_dim":128,"vocab_size":64,)"
             R"("rope_theta":10000.0,"tie_word_embeddings":false})");
    // The graph's name comes from genai_config.json, as in the real pack (phi-4's is eager-mm.onnx).
    put_file((dir / "genai_config.json").string(),
             R"({"model":{"decoder":{"filename":"eager-mm.onnx","hidden_size":5120,"num_hidden_layers":1,)"
             R"("num_attention_heads":40,"num_key_value_heads":10,"head_size":128,"intermediate_size":17920}}})");
    // K 5120 -> 41 blocks of 68 B = 2,788 B per row; N 768, so a fitting tensor is 2,141,184 B.
    const int64_t K = 5120, N = 768, stride = (K / 128 + 1) * 68;
    const int64_t embed_bytes = 64 * 5120 * 4;          // vocab 64 x hidden 5120, and pb::tensor declares FLOAT
    const auto write_data = [&](int64_t packed_bytes) {
        put_file((dir / "eager-mm.onnx.data").string(), std::string(size_t(embed_bytes + packed_bytes), '\x22'));
    };
    // `with_sibling = false` writes the placeholder with NO `*.qweight.packed` initializer at all -- the truncated-pack
    // shape engine#1690's review is about, which is a different case from the byte-less sibling.
    const auto write_graph = [&](const std::string& packed, int64_t packed_bytes, bool with_sibling = true) {
        std::vector<std::string> inits{pb::tensor("model.embed_tokens.weight", {64, 5120}, "eager-mm.onnx.data", 0,
                                                  uint64_t(embed_bytes)),
                                       pb::placeholder("model.layers.0.attn.qkv_proj.MatMulNBits.qweight.empty"),
                                       pb::placeholder("model.layers.0.attn.qkv_proj.MatMulNBits.scales.empty")};
        if (with_sibling)
            inits.push_back(pb::tensor(packed, {packed_bytes}, "eager-mm.onnx.data", uint64_t(embed_bytes),
                                       uint64_t(packed_bytes)));
        put_file((dir / "eager-mm.onnx").string(),
                 pb::model(inits, {pb::matmulnbits("qkv", {"x", "/model/layers.0/attn/qkv_proj/MatMulNBits.qweight.empty",
                                                           "/model/layers.0/attn/qkv_proj/MatMulNBits.scales.empty", "y"},
                                                   K, N, 4, 128, "MatMulNBitsBf")}));
    };

    write_data(stride * N);
    write_graph("model.layers.0.attn.qkv_proj.MatMulNBits.qweight.packed", stride * N);
    const json m = json::parse(pi::plan(dir.string()).manifest());
    const json& g = m["graph_weights"];
    check(g["ryzenai_matmuls_ops"] == 1, "external-packed: MatMulNBitsBf is counted as a quantised matmul");
    check(g["matmulnbits_ops"] == 1, "external-packed: and it enters the matmul list (the reader follows it)");
    check(g["packed_weight_siblings"] == 1,
          "external-packed: the *.empty placeholder resolves to its *.packed sibling (the weights are here, "
          "behind a placeholder)");
    check(g["external_row_stride"]["matched"] == 1 && g["external_row_stride"]["of"] == 1,
          "external-packed: the bytes fit length == N x (K/128 + 1) x 68, reported as a count");

    // engine#1671: when the `*.packed` sibling is ALSO a placeholder (dims [0], no data), this graph carries no
    // weight for that matmul at all -- and the node must be DROPPED rather than left naming the `.empty`
    // placeholder. Left naming it, the reader calls bytes_of() on a byte-less tensor and throws
    // "model.layers.0.attn.qkv_proj.MatMulNBits.qweight.empty: no data", which broke `npu-pack import` for every
    // Ryzen AI 1.9 fconst pack: those declare all of their graph initializers this way (363 `.empty` and 363
    // `.packed`, every one dims [0]) because the weights live in the Token .fconst.
    write_data(0);
    write_graph("model.layers.0.attn.qkv_proj.MatMulNBits.qweight.packed", 0);
    {
        // Both empty AND no .fconst: the weights are genuinely absent from this pack, so it is refused by name
        // (#1687's rule) rather than reporting a capacity or tile-layout fault. The SAME graph imports fine from a
        // Token .fconst -- that is the 1.9 shape, where every initializer is spelled this way and the weights live in
        // the .fconst -- which `npu_pack_import_test <golden> <1.9 pack>` covers end to end with a real pack.
        bool refused = false;
        std::string what;
        try {
            (void)pi::plan(dir.string());
        } catch (const std::exception& e) {
            refused = true;
            what = e.what();
        }
        check(refused, "placeholder pair: with no .fconst the pack is refused rather than read from a byte-less tensor");
        check(what.find("qweight.empty") != std::string::npos && what.find("qweight.packed") != std::string::npos &&
                  what.find("genuinely absent") != std::string::npos,
              "placeholder pair: the refusal names the placeholder, its sibling and the reason (" + what + ")");
    }

    // The SAME graph, but with a Token .fconst beside it: now the weights have a source, so the drop is what makes the
    // pack readable -- and this is the arm that pins the drop. Reverting it (the weightless node pushed back into the
    // matmul list) leaves matmulnbits_ops at 1 here, because the node names a tensor with no bytes.
    put_file((dir / "dd_metastate_Llm_Token_Token_crafted.fconst").string(), std::string(4096, '\0'));
    {
        const json p3 = json::parse(pi::plan(dir.string()).manifest());
        const json& g3 = p3["graph_weights"];
        check(g3["weightless_matmuls"] == 1,
              "placeholder pair: with an .fconst the weightless matmul is COUNTED (" +
                  std::to_string(g3["weightless_matmuls"].get<int64_t>()) + ")");
        check(g3["matmulnbits_ops"] == 0 && g3["packed_weight_siblings"] == 0,
              "placeholder pair: and dropped from the matmul list, so nothing downstream reads a byte-less tensor "
              "(engine#1671: this is the path a 1.9 pack takes, where the weights are in the .fconst)");
    }
    fs::remove(dir / "dd_metastate_Llm_Token_Token_crafted.fconst");

    // engine#1690 review: a `*.empty` placeholder with NO `*.packed` sibling at all must KEEP its matmul. Dropping it
    // empties onnx.matmuls, and an empty matmul list is what tells plan() "this is a 1.9 fconst pack, take the head
    // from the tail" (:1685) -- so a truncated pack would silently get a tail head instead of #1687's refusal. The
    // point of the arm is the count, not the refusal: with the drop the node is gone and `matmulnbits_ops` reads 0.
    write_data(0);
    write_graph("model.layers.0.attn.qkv_proj.MatMulNBits.qweight.packed", 0, /*with_sibling=*/false);
    {
        bool refused = false;
        std::string what;
        json g4;
        try {
            g4 = json::parse(pi::plan(dir.string()).manifest())["graph_weights"];
        } catch (const std::exception& e) {
            refused = true;
            what = e.what();
        }
        if (refused)
            check(what.find("fconst") == std::string::npos,
                  "placeholder with no sibling: any refusal is not the \"read the head from the tail\" path (" + what + ")");
        else
            check(g4["matmulnbits_ops"] == 1 && g4["weightless_matmuls"] == 0,
                  "placeholder with no sibling: the matmul is KEPT, so the pack is not mistaken for a tail-head pack");
    }

    // Which `If` branch is decode (engine#1740). An OGA pack splits prefill from decode with
    // `If (Equal(Shape(input_ids), Constant(value_int 1)))`, and the two branches do NOT hold the same tensors: measured
    // on amd/Qwen3-8B_rai_1.9.0_npu, `then_branch` holds exactly one MatMulNBits -- lm_head [151936,32,64] with real
    // bytes -- while `else_branch` holds 109 layer projections and an `lm_head.MatMulNBits.qweight.empty` placeholder
    // with none. The Token .fconst is the decode path, so a walk that merges the branches counts the prefill
    // placeholder as "this pack has no head", falls back to the tail, and refuses (these packs' tails are the 8,256-byte
    // envelope alone). The arm pins the reading of the condition, both ways round.
    {
        const auto graph_with_cond = [&](bool one_token_when_true, const std::string& head_branch) {
            std::string named;
            pb::str(named, 4, "Constant");
            std::string a;
            pb::str(a, 1, "value_int");
            pb::ints(a, 3, one_token_when_true ? 1 : 2);
            pb::tag(named, 5, 2);
            pb::varint(named, a.size());
            named += a;
            // A pack this fixture can actually plan: it needs its own embedding tensor in the graph, exactly as the
            // write_graph lambda above gives it.
            const std::string emb = pb::tensor("model.embed_tokens.weight", {64, 5120}, "eager-mm.onnx.data", 0,
                                               uint64_t(embed_bytes));
            const std::string head = pb::matmulnbits("lm_head", {"x", "w", "w.scales", "w.zeros"}, K, N, 4, 128);
            const std::string prefill = pb::graph({emb}, {head});
            const std::string decode = pb::graph({emb}, {head});
            return pb::model({}, {named, pb::if_node("c", "input_ids", "logits", head_branch, decode),
                                  pb::if_node("c2", "input_ids", "logits",
                                              head_branch == "then_branch" ? "else_branch" : "then_branch", prefill)});
        };
        // The graph this pack imports is the one genai_config.json names (eager-mm.onnx); a crafted model.onnx beside
        // it is not read at all, which is why the first version of this arm measured the fixture's own graph.
        put_file((dir / "eager-mm.onnx").string(), graph_with_cond(/*one_token_when_true=*/true, "then_branch"));
        const json d1 = json::parse(pi::plan(dir.string()).manifest())["graph_weights"];
        check(d1.contains("decode_branch") && d1["decode_branch"].get<std::string>() == "then_branch",
              "decode branch: a condition comparing the shape with 1 makes then_branch the decode path");
        put_file((dir / "eager-mm.onnx").string(), graph_with_cond(/*one_token_when_true=*/false, "else_branch"));
        const json d2 = json::parse(pi::plan(dir.string()).manifest())["graph_weights"];
        check(d2["decode_branch"].get<std::string>() == "else_branch",
              "decode branch: an inverted condition moves the decode path to else_branch");
        write_graph("model.layers.0.attn.qkv_proj.MatMulNBits.qweight.packed", stride * N);
    }

    // A packed length that does not fit the stride must be reported as a mismatch, not crash and not claim a match.
    write_data(stride * N - 68);
    write_graph("model.layers.0.attn.qkv_proj.MatMulNBits.qweight.packed", stride * N - 68);
    const json m2 = json::parse(pi::plan(dir.string()).manifest());
    std::printf("external-packed: %s, %s\n",
                m["graph_weights"]["packed_weight_siblings"] == 1 ? "placeholder resolved to .packed" : "NOT resolved",
                "stride rule measured");
    check(m2["graph_weights"]["external_row_stride"]["matched"] == 0 &&
              m2["graph_weights"]["external_row_stride"]["of"] == 1,
          "external-packed: a length that does not fit the stride is reported as 0 of 1, with the rule named");

    // …and widening the CENSUS must not widen the IMPORT. This genre has no writer yet, so the import answer has to
    // stay the refusal it was before this change: the check is on the answer, not on a sentence in the PR.
    {
        pi::Options allow;
        allow.allow_incomplete = true;   // so the plan's own refusals do not short-circuit before the readers
        bool threw = false;
        std::string why;
        write_data(stride * N);
        write_graph("model.layers.0.attn.qkv_proj.MatMulNBits.qweight.packed", stride * N);
        try {
            (void)pi::import_pack(dir.string(), (dir / "out").string(), allow);
        } catch (const std::exception& e) {
            threw = true;
            why = e.what();
        }
        std::printf("  external-packed import refusal: %s\n", why.c_str());
        check(threw, "external-packed: import refuses a pack whose geometry no split can satisfy");
        // The genre HAS a writer now (engine#1503), so the property is no longer "refused because nothing can
        // write it": it is that a geometry the container split cannot satisfy is REFUSED with the numbers, not
        // written wrong. This fixture's qkv is N 768 against a 40-head x 128 geometry, which is exactly that case.
        check(why.find("container range") != std::string::npos || why.find("does not fit") != std::string::npos,
              "external-packed: a geometry the split cannot satisfy is refused with the numbers (" + why + ")");
        check(!fs::exists(dir / "out" / "model.q4nx"),
              "external-packed: nothing is written for a genre this importer cannot yet walk");
    }
    fs::remove_all(dir);

    // The FFN width in the external-data genre is the NODE's declaration, because the weight is a 1-D byte blob with
    // no shape: a pack whose config states no intermediate_size must still be readable from gate/up (N) or down (K).
    const fs::path dir2 = fs::path(root) / "external-packed-im";
    fs::remove_all(dir2);
    fs::create_directories(dir2 / "cache");
    put_file((dir2 / "config.json").string(),
             R"({"model_type":"phi3","hidden_size":5120,"num_hidden_layers":1,"num_attention_heads":40,)"
             R"("num_key_value_heads":10,"head_dim":128,"vocab_size":64,"rope_theta":10000.0})");
    put_file((dir2 / "genai_config.json").string(),
             R"({"model":{"decoder":{"filename":"model.onnx","hidden_size":5120,"num_hidden_layers":1,)"
             R"("num_attention_heads":40,"num_key_value_heads":10,"head_size":128}}})");
    const int64_t im = 17920;
    put_file((dir2 / "model.onnx").string(),
             pb::model({pb::tensor("model.embed_tokens.weight", {64, 5120}, "model.onnx.data", 0, uint64_t(embed_bytes))},
                       {pb::matmulnbits("gate", {"x", "/model/layers.0/mlp/gate_proj/MatMulNBits.qweight.empty",
                                                 "/s", "y"}, K, im, 4, 128, "MatMulNBitsBf"),
                        pb::matmulnbits("down", {"x", "/model/layers.0/mlp/down_proj/MatMulNBits.qweight.empty",
                                                 "/s", "y"}, im, 5120, 4, 128, "MatMulNBitsBf")}));
    put_file((dir2 / "model.onnx.data").string(), std::string(size_t(embed_bytes + 4096), '\x33'));
    const json mi = json::parse(pi::plan(dir2.string()).manifest());
    check(mi["dims"]["intermediate_size"] == im,
          "external-packed: intermediate_size comes from the node's N/K when the config states none");
    fs::remove_all(dir2);
}

// engine#1849 (the CHANGES on #1712 that the merge went past): the external-data WRITER had no success-path arm. Every
// import_pack() on the genre above expects a refusal, so the decode -> split -> tile chain, the BF16 copies and the header
// keys #1751/#1793 added were exercised only on real packs. This builds a pack whose geometry the split CAN satisfy,
// imports it to completion and reads the container back through q4nx's own decoders -- independent of the writer -- against
// values the fixture computes itself. Every row has its own scale, and every (row, block, group) its own zero point, so a
// row that lands in the wrong tensor or a q/k/v range that is off by one tile changes a weight and fails here.
void test_external_import_success(const std::string& root) {
    const fs::path dir = fs::path(root) / "external-success";
    fs::remove_all(dir);
    fs::create_directories(dir / "cache");
    // 2 heads + 1 kv head of 128: q 256 rows, k 128, v 128 -> one qkv matmul of N 512 at K = hidden 256.
    const int H = 256, NH = 2, NKV = 1, HD = 128, V = 64;
    const int N = (NH + 2 * NKV) * HD, K = H;
    put_file((dir / "config.json").string(),
             R"({"model_type":"phi3","hidden_size":256,"intermediate_size":512,"num_hidden_layers":1,)"
             R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":128,"vocab_size":64,)"
             R"("rope_theta":10000.0,"tie_word_embeddings":false})");
    put_file((dir / "genai_config.json").string(),
             R"({"model":{"decoder":{"filename":"eager-mm.onnx","hidden_size":256,"num_hidden_layers":1,)"
             R"("num_attention_heads":2,"num_key_value_heads":1,"head_size":128,"intermediate_size":512}}})");
    put_file((dir / "tokenizer.json").string(), R"({"model":{"type":"BPE"}})");

    // The fixture's own weights: w(row, col) = (code - zp) * scale, chosen so every product is exact in f32 (scales are
    // bf16-exact with 7 mantissa bits, |code - zp| <= 15), so the container's `code*scale + m` form must reproduce them
    // bit for bit. Both nibbles are int4 TWO'S COMPLEMENT (the pack's writer asserts -8 <= x <= 7), so the reference
    // sign-extends them: the generators walk all 16 nibble values, and for a code and zero of opposite sign an
    // unsigned reading is off by 16 * scale (engine#1503, #1874).
    auto scale_of = [](int row) { return (1.0f + float(row % 128) / 128.0f) * std::ldexp(1.0f, -(row / 128 + 2)); };
    auto zp_of = [](int row, int kb, int g) { return uint8_t((row + 3 * kb + 5 * g) % 16); };
    auto code_of = [](int row, int col) { return uint8_t((row * 7 + col * 3) % 16); };
    auto w_of = [&](int row, int col) {
        auto sx = [](uint8_t v) { return v >= 8 ? int(v) - 16 : int(v); };
        return float(sx(code_of(row, col)) - sx(zp_of(row, col / 128, (col % 128) / 32))) * scale_of(row);
    };
    // Encode the blob as the pack lays it out -- the inverse of decode_external_tile() for the shipped ordering
    // (pi::kExtQuantsRowMajor == false: byte (cp%4) + c*4 + (cp/4)*512 holds rows 2cp/2cp+1 of column c).
    const int blocks = K / 128, tiles = N / 64, per_tile = 2 * (blocks + 1);
    std::string blob(size_t(tiles) * per_tile * 2176, '\0');
    for (int tile = 0; tile < tiles; ++tile)
        for (int half = 0; half < 2; ++half)
            for (int kk = 1; kk <= blocks; ++kk) {
                char* subv = &blob[size_t(tile * per_tile + kk * 2 + half) * 2176];
                for (int r = 0; r < 32; ++r) {
                    // A 64-row block's FIRST 32 rows live in the pair's SECOND subvolume (measured on the pack, #1874).
                    const int row = tile * 64 + (1 - half) * 32 + r;
                    const uint16_t s = pi::bf16_bits_of(scale_of(row));
                    subv[2112 + r * 2] = char(s & 0xFF);
                    subv[2112 + r * 2 + 1] = char(s >> 8);
                    subv[2048 + r * 2] = char(zp_of(row, kk - 1, 0) | (zp_of(row, kk - 1, 1) << 4));
                    subv[2048 + r * 2 + 1] = char(zp_of(row, kk - 1, 2) | (zp_of(row, kk - 1, 3) << 4));
                    for (int c = 0; c < 128; ++c) {
                        const int cp = r / 2;
                        uint8_t& b = reinterpret_cast<uint8_t&>(subv[(cp % 4) + c * 4 + (cp / 4) * 512]);
                        const uint8_t nib = code_of(row, (kk - 1) * 128 + c);
                        b = (r % 2 == 0) ? uint8_t((b & 0xF0) | nib) : uint8_t((b & 0x0F) | (nib << 4));
                    }
                }
            }
    check(int64_t(blob.size()) == int64_t(N) * (K / 128 + 1) * 68,
          "external success: the crafted blob has the genre's length N x (K/128 + 1) x 68");
    // The embedding, F32 in the pack, values exact in bf16.
    std::string embed(size_t(V) * H * 4, '\0');
    for (int i = 0; i < V * H; ++i) {
        const float v = float((i % 61) - 30) * 0.5f;
        std::memcpy(&embed[size_t(i) * 4], &v, 4);
    }
    put_file((dir / "eager-mm.onnx.data").string(), embed + blob);
    put_file((dir / "eager-mm.onnx").string(),
             pb::model({pb::tensor("model.embed_tokens.weight", {V, H}, "eager-mm.onnx.data", 0, uint64_t(embed.size())),
                        pb::placeholder("model.layers.0.attn.qkv_proj.MatMulNBits.qweight.empty"),
                        pb::placeholder("model.layers.0.attn.qkv_proj.MatMulNBits.scales.empty"),
                        pb::tensor("model.layers.0.attn.qkv_proj.MatMulNBits.qweight.packed", {int64_t(blob.size())},
                                   "eager-mm.onnx.data", uint64_t(embed.size()), uint64_t(blob.size()))},
                       {pb::matmulnbits("qkv", {"x", "/model/layers.0/attn/qkv_proj/MatMulNBits.qweight.empty",
                                                "/model/layers.0/attn/qkv_proj/MatMulNBits.scales.empty", "y"},
                                        K, N, 4, 128, "MatMulNBitsBf")}));

    struct Container {
        json table;
        std::string data;
    };
    auto read_container = [](const fs::path& out) {
        Container c;
        std::ifstream f(out / "model.q4nx", std::ios::binary);
        uint64_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 8);
        std::string hdr(size_t(hlen), '\0');
        f.read(hdr.data(), std::streamsize(hlen));
        while (!hdr.empty() && (hdr.back() == ' ' || hdr.back() == '\0')) hdr.pop_back();
        c.table = json::parse(hdr);
        c.data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        return c;
    };
    auto json_of = [](const fs::path& p) {
        std::ifstream f(p);
        return json::parse(f);
    };

    for (const std::string layout : {"", "b8"}) {
        const fs::path out = dir / (layout.empty() ? "out" : "out-b8");
        fs::remove_all(out);
        pi::Options opt;
        opt.allow_incomplete = true;   // the plan's own refusals (no .fconst, no programs) must not stop the writer
        opt.weight_layout = layout;
        pi::Report r;
        std::string why;
        try {
            r = pi::import_pack(dir.string(), out.string(), opt);
        } catch (const std::exception& e) {
            why = e.what();
        }
        const std::string tag = "external success" + std::string(layout.empty() ? "" : " (b8)") + ": ";
        check(why.empty() && r.refused.empty() && fs::exists(out / "model.q4nx"),
              tag + "a pack whose geometry the split satisfies imports to completion (" + why +
                  (r.refused.empty() ? "" : r.refused.front()) + ")");
        if (!fs::exists(out / "model.q4nx")) continue;

        // config.json: the geometry, and q4nx_weight_layout ONLY for a non-default layout (engine#1840 / #1793).
        const json cfg = json_of(out / "config.json");
        check(cfg.value("num_attention_heads", 0) == NH && cfg.value("num_key_value_heads", 0) == NKV &&
                  cfg.value("head_dim", 0) == HD && cfg.value("hidden_size", 0) == H,
              tag + "config.json carries the pack's geometry");
        if (layout.empty())
            check(!cfg.contains("q4nx_weight_layout"),
                  tag + "a default-layout container carries no q4nx_weight_layout key (every earlier container depends "
                        "on that)");
        else
            check(cfg.value("q4nx_weight_layout", std::string()) == "b8",
                  tag + "--weight-layout b8 writes q4nx_weight_layout \"b8\" into config.json");
        // The text artifacts (engine#1751): without tokenizer.json the directory is not a model directory.
        check(fs::exists(out / "tokenizer.json"), tag + "tokenizer.json is copied beside the container");
        if (!layout.empty()) continue;   // the weights do not depend on the layout key; check them once

        const Container c = read_container(out);
        // q/k/v land as three tensors with the geometry's row ranges, and every weight is the fixture's.
        struct Want {
            const char* name;
            int lo, rows;
        };
        for (const Want& want : {Want{"model.layers.0.self_attn.q_proj.weight", 0, NH * HD},
                                 Want{"model.layers.0.self_attn.k_proj.weight", NH * HD, NKV * HD},
                                 Want{"model.layers.0.self_attn.v_proj.weight", NH * HD + NKV * HD, NKV * HD}}) {
            const std::string nm = want.name;
            if (!c.table.contains(nm)) {
                check(false, tag + nm + " is in the container");
                continue;
            }
            const json& e = c.table.at(nm);
            const int64_t lo = e.at("data_offsets").at(0), hi = e.at("data_offsets").at(1);
            const int RT = want.rows / 32, KT = K / 256;
            const size_t tb = size_t(onebit::npu::q4nx::Chunk::Q4_1F);
            check(e.at("shape").at(1) == int64_t(tb) && (hi - lo) == int64_t(RT) * KT * int64_t(tb),
                  tag + nm + " is " + std::to_string(RT * KT) + " q4_1f tiles");
            if ((hi - lo) != int64_t(RT) * KT * int64_t(tb)) continue;
            int bad = 0, first_r = -1, first_c = -1;
            std::vector<float> w(32 * 256);
            for (int rt = 0; rt < RT; ++rt)
                for (int kt = 0; kt < KT; ++kt) {
                    const auto* tile =
                        reinterpret_cast<const uint8_t*>(c.data.data() + lo + (size_t(rt) * KT + kt) * tb);
                    onebit::npu::q4nx::dequant_tile(onebit::npu::q4nx::Chunk::Q4_1F, tile, w.data(), 256);
                    for (int r = 0; r < 32; ++r)
                        for (int col = 0; col < 256; ++col) {
                            const int row = rt * 32 + r, gcol = kt * 256 + col;
                            if (w[size_t(r) * 256 + col] != w_of(want.lo + row, gcol)) {
                                if (bad++ == 0) first_r = row, first_c = gcol;
                            }
                        }
                }
            check(bad == 0, tag + nm + ": every weight is the pack's row " + std::to_string(want.lo) + "+r, exactly (" +
                                std::to_string(bad) + " differ, first at " + std::to_string(first_r) + "," +
                                std::to_string(first_c) + ")");
        }
        // The embedding is copied as BF16 with its shape (exact here by construction).
        bool embed_ok = c.table.contains("model.embed_tokens.weight");
        if (embed_ok) {
            const json& e = c.table.at("model.embed_tokens.weight");
            const int64_t lo = e.at("data_offsets").at(0), hi = e.at("data_offsets").at(1);
            embed_ok = e.at("dtype") == "BF16" && e.at("shape") == json::array({V, H}) && hi - lo == int64_t(V) * H * 2;
            for (int i = 0; embed_ok && i < V * H; ++i) {
                uint16_t b = 0;
                std::memcpy(&b, c.data.data() + lo + size_t(i) * 2, 2);
                embed_ok = onebit::npu::bf16_to_f32(b) == float((i % 61) - 30) * 0.5f;
            }
        }
        check(embed_ok, tag + "the embedding is copied as BF16 [64, 256] with every value intact");
    }
    fs::remove_all(dir);
    std::printf("external success: crafted external-data pack imported; q/k/v split, weights, embedding and keys hold\n");
}


// The two overstatements the readers must catch: declared dims larger than the external file, and a
// MatMulNBits node whose N/K/block_size demand more qweight than the tensor declares.
// engine#1671: a pack whose graph is all `[0]`-dims placeholders and whose bytes are in model.bin (the 1.8/hybrid
// genre). The importer has no model.bin reader, so the honest outcome is a refusal that NAMES the file — not
// "<tensor>.qweight.empty: no data", which reports a placeholder as though the bytes were absent rather than elsewhere.
// Crafted, so this needs no 1.9 pack: a graph with one resolvable tensor, one MatMulNBits whose qweight input is a
// `.empty` placeholder with no sibling, and a model.bin next to it.
// engine#1985: the canonical copy lives at a path every import of the same pack shares, and it used to be written in place.
// A second import that started mid-write saw a short file, truncated it and wrote again, and the first then read its own
// copy while it was being refilled: Qwen3-1.7B came back with 280 refused tensors and exit 0 in 4 of 24 runs of three
// imports 0.15 s apart. The arms: the bytes arrive and no temporary file stays; a failed write leaves nothing; the
// temporary name is out of reach of another import's prefix sweep and inside the age sweep; and the race itself, with
// threads doing what an import does (reuse the copy if it is complete, else write it, then read it).
void test_write_canonical_copy() {
    const fs::path dir = fs::temp_directory_path() / ("1bit-atomic-" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    const std::string prefix = "1bit-fconst19v1-7-2048-16-28-";
    const std::string name = prefix + "957724736-1234.fconst";
    const std::string work = (dir / name).string();
    const size_t n = size_t(4) << 20;
    std::vector<uint8_t> body(n);
    for (size_t i = 0; i < n; ++i) body[i] = uint8_t((i * 2654435761u) >> 24) | 1;   // never 0: a refilled hole reads as 0
    auto same = [&](const std::string& f) {
        std::ifstream is(f, std::ios::binary);
        if (!is) return false;
        std::vector<uint8_t> got(n + 1);
        is.read(reinterpret_cast<char*>(got.data()), std::streamsize(got.size()));
        return size_t(is.gcount()) == n && std::memcmp(got.data(), body.data(), n) == 0;
    };
    auto entries = [&] {
        int k = 0;
        for (const auto& e : fs::directory_iterator(dir)) { (void)e; ++k; }
        return k;
    };

    pi::write_canonical_copy(work, body.data(), n);
    check(same(work), "atomic copy: the bytes written are the bytes at the path");
    check(entries() == 1, "atomic copy: no temporary file is left beside the copy");
    std::vector<uint8_t> other(body);
    other[n / 2] ^= 0xff;
    pi::write_canonical_copy(work, other.data(), n);
    check(!same(work) && entries() == 1, "atomic copy: a second write replaces the copy, again leaving one file");
    pi::write_canonical_copy(work, body.data(), n);

    bool threw = false;
    try {
        pi::write_canonical_copy((dir / "no-such-dir" / name).string(), body.data(), n);
    } catch (const std::runtime_error&) { threw = true; }
    check(threw && entries() == 1, "atomic copy: a copy that cannot be written throws and leaves nothing behind");

    // the temporary name: beside the copy, not the copy, unique per call, invisible to the prefix sweep, visible to the age sweep
    const std::string t1 = pi::canonical_copy_temp_name(work), t2 = pi::canonical_copy_temp_name(work);
    const std::string b1 = fs::path(t1).filename().string();
    check(fs::path(t1).parent_path() == fs::path(work).parent_path(), "atomic copy: the temporary file is in the copy's directory");
    check(t1 != work && t1 != t2, "atomic copy: the temporary name is not the copy's and is new on every call");
    check(b1.compare(0, prefix.size(), prefix) != 0 && b1.compare(0, 13, "1bit-fconst19") == 0,
          "atomic copy: the temporary name starts with the scratch stem and not with the pack's prefix");
    { std::ofstream o(t1, std::ios::binary); o << "being written"; }
    pi::evict_stale_canonical_copies(dir.string(), name, prefix, std::string(), 3600);
    check(fs::exists(t1) && fs::exists(work), "atomic copy: another import's sweep does not remove a copy that is being written");
    fs::last_write_time(t1, fs::file_time_type::clock::now() - std::chrono::hours(2));
    pi::evict_stale_canonical_copies(dir.string(), name, prefix, std::string(), 3600);
    check(!fs::exists(t1) && fs::exists(work), "atomic copy: a temporary file a killed run left is removed by the age rule");

    // the race: every thread is an import. It reuses a complete-length copy or writes one, then reads the path, as
    // canonical_token_fconst()'s caller does. One thread also removes the copy now and then, so the cycle restarts.
    fs::remove(work, ec);
    std::atomic<int> bad{0}, read_ok{0}, wrote{0};
    auto importer = [&](int id) {
        for (int r = 0; r < 24; ++r) {
            if (!pi::canonical_copy_reusable(work, int64_t(n))) {
                try { pi::write_canonical_copy(work, body.data(), n); ++wrote; } catch (const std::runtime_error&) { ++bad; }
            }
            std::error_code e2;
            // An open that fails is the remover having been there, not a bad copy; a copy that OPENS must be whole.
            if (std::ifstream(work, std::ios::binary).good()) {
                std::ifstream is(work, std::ios::binary);
                std::vector<uint8_t> got(n + 1);
                if (is) {
                    is.read(reinterpret_cast<char*>(got.data()), std::streamsize(got.size()));
                    if (size_t(is.gcount()) == n && std::memcmp(got.data(), body.data(), n) == 0) ++read_ok;
                    else ++bad;
                }
            }
            if (id == 0 && r % 4 == 3) fs::remove(work, e2);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < 6; ++i) pool.emplace_back(importer, i);
    for (auto& t : pool) t.join();
    check(bad.load() == 0, "atomic copy: no import ever reads a copy another import is still writing (engine#1985)");
    check(read_ok.load() > 0 && wrote.load() > 1, "atomic copy: the race arm really wrote more than once and read copies back");
    int leftover = 0;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().filename().string() != name) ++leftover;
    check(leftover == 0, "atomic copy: the race leaves no temporary files");
    fs::remove_all(dir, ec);
    std::printf("  atomic canonical copy: 11 arms (bytes, replace, failed write, temporary name x4, %d racing writes / %d reads, no leftovers)\n",
                wrote.load(), read_ok.load());
}

// engine#1662: the ~291 MB canonical Token .fconst rewrite is skipped when the copy on the scratch path already holds
// exactly the bytes this run would write. The decision is a length test against a length that is a function of the source
// alone, so what can go wrong is the test being too willing -- a truncated copy from an interrupted run, a leftover
// directory, a copy from a bigger pack. Those are the arms.
void test_canonical_copy_reusable() {
    const fs::path dir = ::onebit::test::tmp_root() / ("1bit-reuse-" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    auto write = [&](const fs::path& f, size_t n) {
        std::ofstream o(f, std::ios::binary | std::ios::trunc);
        const std::string blob(n, 'x');
        o.write(blob.data(), std::streamsize(blob.size()));
    };
    const int64_t want = 4096;

    write(dir / "complete.fconst", size_t(want));
    check(pi::canonical_copy_reusable((dir / "complete.fconst").string(), want),
          "reuse: a copy of exactly the expected length is reused");
    write(dir / "short.fconst", size_t(want) - 1);
    check(!pi::canonical_copy_reusable((dir / "short.fconst").string(), want),
          "reuse: a truncated copy is rewritten, not trusted (interrupted run)");
    write(dir / "long.fconst", size_t(want) + 1);
    check(!pi::canonical_copy_reusable((dir / "long.fconst").string(), want),
          "reuse: a longer copy (another pack) is not reused");
    check(!pi::canonical_copy_reusable((dir / "absent.fconst").string(), want), "reuse: a missing copy is not reused");
    fs::create_directories(dir / "a-directory");
    check(!pi::canonical_copy_reusable((dir / "a-directory").string(), want),
          "reuse: a directory on the scratch path is not reused");
    check(!pi::canonical_copy_reusable((dir / "complete.fconst").string(), 0),
          "reuse: a non-positive expected length never reuses");
    write(dir / "empty.fconst", 0);
    check(!pi::canonical_copy_reusable((dir / "empty.fconst").string(), 1),
          "reuse: an empty copy is not the canonical form of a non-empty body");
    fs::remove_all(dir, ec);
    std::printf("  canonical copy reuse: 7 arms (complete/truncated/long/absent/directory/zero-length/empty)\n");
}

// engine#1770: the canonical copy is scratch that used to only ever accumulate -- reuse writes a new file when the source
// changes, so 699 files / 8.7 GiB of tmpfs (RAM) had built up on one box. The eviction rule is "same pack and geometry,
// different source identity", and what can go wrong is deleting one of the neighbours that must survive, so each is an arm.
// engine#1841 (second point): the implicit zero point's bit width must follow the SPELLING that matched, not a literal at
// one call site and the node's attribute at the other. The arms below are the drift the issue described: `weight_Q4` must
// give 4 bits (zero point 8) and a hypothetical `weight_Q8` must give 8 (zero point 128) -- the number that would have
// stayed at 8, silently, if the literal were left in place.
void test_implicit_zp_bits_follow_the_spelling() {
    check(pi::zp_bits_for_spelling("model.layers.0.attn.q_proj.MatMul.weight_Q4", 4) == 4,
          "zp bits: weight_Q4 declares 4 bits");
    check(pi::zp_bits_for_spelling("model.layers.0.attn.q_proj.MatMul.weight_Q8", 4) == 8,
          "zp bits: weight_Q8 overrides the path's 4 -- this is the arm a hardcoded literal fails");
    check(pi::zp_bits_for_spelling("model.layers.0.mlp.down_proj.MatMulNBits.qweight", 3) == 3,
          "zp bits: a spelling with no _Q<d> suffix falls back to the node's own value");
    check(pi::zp_bits_for_spelling("weight_Q4", 8) == 4,
          "zp bits: the suffix wins over a node value for a spelled tensor");
    // and the effect on the tensor itself: the zero term moves by 2^(bits-1) * scale.
    pi::Q4 q;
    q.N = 1; q.K = 32; q.block = 128;
    q.codes.assign(32, 0); q.scale.assign(1, 0.5f); q.zero.assign(1, 0.0f);
    pi::implicit_zp_zero(q, 4);
    check(std::fabs(q.zero[0] + 8.0f * 0.5f) < 1e-6f, "zp: 4 bits subtracts 8 * scale = 4.0");
    q.zero.assign(1, 0.0f);
    pi::implicit_zp_zero(q, 8);
    check(std::fabs(q.zero[0] + 128.0f * 0.5f) < 1e-6f, "zp: 8 bits subtracts 128 * scale = 64.0");
    std::printf("  implicit zero point: 6 arms (spelling -> bits x4, shift x2)\n");
}

void test_evict_stale_canonical_copies() {
    const fs::path dir = ::onebit::test::tmp_root() / ("1bit-evict-" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    auto touch = [&](const std::string& base) {
        std::ofstream o(dir / base, std::ios::binary | std::ios::trunc);
        o << 'x';
    };
    const std::string keep = "1bit-fconst19v1-12345-896-14-24-291653952-76263210.fconst";
    const std::string prefix = "1bit-fconst19v1-12345-896-14-24-";
    const std::string stale = "1bit-fconst19v1-12345-896-14-24-291653952-99999.fconst";     // same pack+geometry, new mtime
    const std::string stale_size = "1bit-fconst19v1-12345-896-14-24-300000000-76263210.fconst";  // repacked source
    const std::string other_geometry = "1bit-fconst19v1-12345-1536-12-28-995712640-73345656.fconst";
    const std::string other_pack = "1bit-fconst19v1-99999-896-14-24-291653952-76263210.fconst";
    const std::string other_form = "1bit-fconst19v2-12345-896-14-24-291653952-76263210.fconst";
    const std::string unrelated = "notes.txt";
    // The split must not change the NAME, or every run rewrites a copy it already had (a ~1 GB rewrite, silently). The
    // legacy formula concatenated these fields directly; the fix only cuts the string at the size field.
    check(keep == prefix + "291653952-76263210.fconst",
          "evict: prefix + source identity is the name the reuse path looks for, unchanged");
    for (const auto& f : {keep, stale, stale_size, other_geometry, other_pack, other_form, unrelated}) touch(f);

    pi::evict_stale_canonical_copies(dir.string(), keep, prefix);

    auto exists = [&](const std::string& base) { return fs::exists(dir / base, ec); };
    check(exists(keep), "evict: the copy being written is kept");
    check(!exists(stale), "evict: a copy of the same pack and geometry at another mtime goes (the accumulation)");
    check(!exists(stale_size), "evict: a repacked source's copy goes (different size, same pack and geometry)");
    check(exists(other_geometry), "evict: another geometry of the same pack is kept -- it is not this run's to remove");
    check(exists(other_pack), "evict: another pack with the same geometry is kept");
    check(exists(other_form), "evict: another canonical form version is kept");
    check(exists(unrelated), "evict: unrelated files in the directory are kept");

    // engine#1770 (second generation): the pre-version naming has no source identity, so a prefix match on the current form
    // never sees it -- three such files sat in /tmp for a day at 3.6 GB each, i.e. 10.67 GiB of RAM.
    const std::string legacy = "1bit-fconst19-12345-896-14-24.fconst";
    const std::string legacy_other_geometry = "1bit-fconst19-12345-1536-12-28.fconst";
    touch(legacy);
    touch(legacy_other_geometry);
    pi::evict_stale_canonical_copies(dir.string(), keep, prefix, legacy);
    check(!exists(legacy), "evict: the pre-version name for THIS geometry goes (no identity to reuse)");
    check(exists(legacy_other_geometry), "evict: ... but another geometry's pre-version name is kept");

    // and the abandoned tail: a pack nobody imports again is only reachable by age, because the sweep runs when a pack is
    // written. 0 must disable it, so the callers that pass nothing keep the old behaviour.
    const std::string old_other = "1bit-fconst19v1-77777-1536-12-28-995712640-73345656.fconst";
    const std::string fresh_other = "1bit-fconst19v1-88888-1536-12-28-995712640-73345656.fconst";
    touch(old_other);
    touch(fresh_other);
    const auto seven_hours_ago = fs::file_time_type::clock::now() - std::chrono::hours(7);
    fs::last_write_time(dir / old_other, seven_hours_ago, ec);
    pi::evict_stale_canonical_copies(dir.string(), keep, prefix, legacy, 6 * 3600);
    check(!exists(old_other), "evict: an abandoned geometry older than the age goes (the tail a write cannot reach)");
    check(exists(fresh_other), "evict: a fresh copy of another geometry stays -- another run may be writing it");

    // engine#1797 review: the age rule reads `last_write_time`, and reuse used to READ the copy without writing it -- so
    // a file in continuous use aged past `max_age_seconds` and was swept, and the next run rewrote it. The repair is
    // that reuse now touches the copy, which makes the timestamp mean "last use". This arm pins the rule that repair
    // relies on: a copy with a FRESH mtime survives an age sweep, and one with an old mtime does not.
    {
        // DIFFERENT geometry (hidden 1536), so the name rule cannot reach them and only the age rule decides -- which is
        // the rule this arm is about. Same geometry would test the name rule instead, and my first version of this arm
        // made exactly that mistake and failed on it.
        const std::string stem = "1bit-fconst19v1-";
        const std::string aged = stem + "12345-1536-12-28-995712640-73345656.fconst";
        const std::string fresh_name = stem + "12345-1536-12-28-500000000-11111111.fconst";
        touch(aged);
        touch(fresh_name);
        std::error_code tec;
        fs::last_write_time(dir / aged, fs::file_time_type::clock::now() - std::chrono::hours(10), tec);
        check(!tec, "evict age: the fixture can age a copy");
        // 6 h window: the 10 h copy goes, the one just written stays
        pi::evict_stale_canonical_copies(dir.string(), keep, prefix, "", 6 * 3600);
        check(!exists(aged), "evict age: a copy nobody has touched for 10 h is swept");
        check(exists(fresh_name), "evict age: a copy with a fresh mtime survives -- what reuse's touch produces");
        check(exists(keep), "evict age: and the copy being written is still kept");
    }
    // The degenerate arguments must not turn into "delete the directory".
    for (const auto& f : {keep, stale}) touch(f);
    pi::evict_stale_canonical_copies(dir.string(), "", prefix);
    check(exists(keep) && exists(stale), "evict: an empty keep-name removes nothing");
    pi::evict_stale_canonical_copies(dir.string(), keep, "");
    check(exists(keep) && exists(stale), "evict: an empty prefix removes nothing");
    pi::evict_stale_canonical_copies((dir / "absent").string(), keep, prefix);
    check(exists(keep), "evict: an absent directory is not an error");
    fs::remove_all(dir, ec);
    std::printf("  canonical copy eviction: 16 arms (name composition + keep/stale/geometry/pack/form/unrelated + legacy name +"
                " geometry/age rules + 3 degenerate)\n");
}

// engine#1907: the sweep ran only in the directory being written. Since #1801 that is the on-disk cache, so /tmp was never swept
// again, and 17.0 GiB of copies left there by older binaries stayed in tmpfs (RAM). Every candidate is swept now; one that does
// not exist is not created.
void test_sweep_fconst_scratch_dirs() {
    const fs::path root = ::onebit::test::tmp_root() / ("1bit-sweep-" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(root, ec);
    const fs::path written = root / "cache", other = root / "tmp", absent = root / "never-made";
    fs::create_directories(written);
    fs::create_directories(other);
    auto touch = [&](const fs::path& d, const std::string& base) {
        std::ofstream o(d / base, std::ios::binary | std::ios::trunc);
        o << 'x';
    };
    const std::string prefix = "1bit-fconst19v1-12345-896-14-24-";
    const std::string keep = prefix + "291653952-76263210.fconst";
    const std::string stale = prefix + "291653952-99999.fconst";
    const std::string legacy = "1bit-fconst19-12345-896-14-24.fconst";
    const std::string old_other = "1bit-fconst19v1-77777-1536-12-28-995712640-73345656.fconst";
    const std::string fresh_other = "1bit-fconst19v1-88888-1536-12-28-995712640-73345656.fconst";
    for (const auto& f : {keep, stale, legacy, old_other, fresh_other}) touch(other, f);
    touch(written, keep);
    const auto seven_hours_ago = fs::file_time_type::clock::now() - std::chrono::hours(7);
    fs::last_write_time(other / old_other, seven_hours_ago, ec);
    pi::sweep_fconst_scratch_dirs({written.string(), other.string(), absent.string()}, written.string(), keep, prefix, legacy,
                                  6 * 3600);
    const auto in_other = [&](const std::string& b) { return fs::exists(other / b); };
    check(!in_other(old_other), "sweep: an abandoned copy in a directory that was NOT written goes (the /tmp left behind by #1801)");
    check(!in_other(stale) && !in_other(legacy),
          "sweep: a stale identity and the legacy name of this pack go from the other directory too");
    check(in_other(keep), "sweep: the exact name is kept in every directory (another environment may reuse it there)");
    check(in_other(fresh_other), "sweep: a fresh copy of another pack stays -- another run may be writing it");
    check(fs::exists(written / keep), "sweep: the copy being written is kept");
    check(!fs::exists(absent), "sweep: a candidate that does not exist is not created");
    // #1915 review, boundary arms. (a) A file that is not fconst19 scratch is kept however old it is. (b) The current
    // directory is never swept unless it is the directory being written -- under every spelling of it.
    {
        const fs::path lane = root / "lane-cwd";
        fs::create_directories(lane);
        const std::string notes = "notes-about-1bit-fconst19.txt";
        for (const auto& d : {other, lane}) {
            touch(d, notes);
            touch(d, old_other);
            fs::last_write_time(d / notes, seven_hours_ago, ec);
            fs::last_write_time(d / old_other, seven_hours_ago, ec);
        }
        const fs::path was = fs::current_path();
        fs::current_path(lane);
        pi::sweep_fconst_scratch_dirs({written.string(), other.string(), ".", lane.string(), (lane / ".." / "lane-cwd").string()},
                                      written.string(), keep, prefix, legacy, 6 * 3600);
        const bool cwd_kept = fs::exists(lane / old_other);
        // and when the cwd IS the directory being written, the evict-on-write rule still applies to it
        pi::sweep_fconst_scratch_dirs({lane.string(), other.string()}, lane.string(), keep, prefix, legacy, 6 * 3600);
        const bool cwd_written_swept = !fs::exists(lane / old_other);
        fs::current_path(was);
        check(!in_other(old_other), "sweep: (control) the same old file in a directory that is not the cwd goes");
        check(in_other(notes) && fs::exists(lane / notes), "sweep: a file that is not fconst19 scratch is kept however old");
        check(cwd_kept, "sweep: the current directory is never swept, as '.', as its absolute path or through '..'");
        check(cwd_written_swept, "sweep: unless it is the directory being written (evict-on-write, engine#1770)");
    }
    fs::remove_all(root, ec);
    std::printf("  scratch sweep: every candidate directory, 10 arms\n");
}

// #1915 review: the arms above call the sweep; nothing ran the importer and looked. "Delete the call site and the suite
// stays green" was true. This runs the production path once -- import_pack() on a crafted pack with a Token .fconst --
// with a scratch directory that already holds an abandoned copy of ANOTHER pack, and asserts that copy is gone. With an
// override set only that directory is swept, so the second directory's copy must still be there.
void test_sweep_is_wired(const std::string& root) {
    const fs::path dir = fs::path(root) / "sweep-wired";
    const fs::path scratch = ::onebit::test::tmp_root() / ("1bit-sweep-wired-" + std::to_string(::getpid()));
    const fs::path cache_home = ::onebit::test::tmp_root() / ("1bit-sweep-wired-xdg-" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::remove_all(scratch, ec);
    fs::remove_all(cache_home, ec);
    fs::create_directories(dir / "cache");
    fs::create_directories(scratch);
    fs::create_directories(cache_home / "1bit" / "fconst19");
    const int H = 128, L = 8, V = 64;
    {
        const int64_t stride = 1 << 20, second = 262144;
        std::string fc(size_t(int64_t(L) * stride + 64 + size_t(H) * 2), '\0');
        for (int l = 0; l <= L; ++l) {
            fc[size_t(int64_t(l) * stride)] = char(0x86);
            fc[size_t(int64_t(l) * stride) + 1] = char(0x35);
            if (l < L) {
                fc[size_t(int64_t(l) * stride + second)] = char(0x86);
                fc[size_t(int64_t(l) * stride + second) + 1] = char(0x35);
            }
        }
        put_file((dir / "dd_metastate_Llm_Token_Token_crafted.fconst").string(), fc);
    }
    put_file((dir / "config.json").string(),
             R"({"model_type":"qwen2","hidden_size":128,"intermediate_size":256,"num_hidden_layers":8,)"
             R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,"vocab_size":64,)"
             R"("rope_theta":10000.0,"tie_word_embeddings":true,"rms_norm_eps":1e-06})");
    put_file((dir / "model.onnx.data").string(), std::string(size_t(V) * size_t(H) * 2, '\x11'));
    put_file((dir / "model.onnx").string(),
             pb::model({pb::tensor("model.embed_tokens.weight", {V, H}, "model.onnx.data", 0, uint64_t(V) * uint64_t(H) * 2, 10)},
                       {}));
    const std::string foreign_old = "1bit-fconst19v1-77777-1536-12-28-995712640-73345656.fconst";
    const std::string foreign_fresh = "1bit-fconst19v1-88888-1536-12-28-995712640-73345656.fconst";
    const std::string unrelated = "keep-me.txt";
    const auto seven_hours_ago = fs::file_time_type::clock::now() - std::chrono::hours(7);
    for (const fs::path& d : {scratch, cache_home / "1bit" / "fconst19"})
        for (const auto& f : {foreign_old, foreign_fresh, unrelated}) {
            put_file((d / f).string(), "x");
            if (f != foreign_fresh) fs::last_write_time(d / f, seven_hours_ago, ec);
        }
    struct Env {
        std::string name, value;
        bool had;
    };
    std::vector<Env> saved;
    for (const char* n : {"ONEBIT_FCONST_SCRATCH", "XDG_CACHE_HOME"}) {
        const char* v = std::getenv(n);
        saved.push_back({n, v ? v : "", v != nullptr});
    }
    setenv("ONEBIT_FCONST_SCRATCH", scratch.c_str(), 1);
    setenv("XDG_CACHE_HOME", cache_home.c_str(), 1);
    const std::string out = (::onebit::test::tmp_root() / ("oga_pack_import_sweepwired." + std::to_string(::getpid()))).string();
    fs::remove_all(out, ec);
    pi::Options allow;
    allow.allow_incomplete = true;
    try {
        (void)pi::import_pack(dir.string(), out, allow);
    } catch (const std::exception&) {
        // the import's own outcome is not what is under test: the scratch copy is made, and swept around, before any
        // refusal about this fixture's empty payload
    }
    for (const auto& e : saved) {
        if (e.had) setenv(e.name.c_str(), e.value.c_str(), 1);
        else unsetenv(e.name.c_str());
    }
    bool wrote_here = false;
    for (const auto& e : fs::directory_iterator(scratch, ec))
        if (e.path().filename().string().rfind("1bit-fconst19v", 0) == 0 && e.path().filename().string() != foreign_fresh)
            wrote_here = true;
    check(wrote_here, "sweep wiring: (premise) the import put its scratch copy in the override directory");
    check(!fs::exists(scratch / foreign_old), "sweep wiring: an abandoned copy of another pack is GONE after a real import");
    check(fs::exists(scratch / foreign_fresh), "sweep wiring: a fresh copy of another pack is still there");
    check(fs::exists(scratch / unrelated), "sweep wiring: a file that is not fconst19 scratch is still there");
    check(fs::exists(cache_home / "1bit" / "fconst19" / foreign_old),
          "sweep wiring: with an override set, the cache directory is NOT swept (another lane's files stay)");
    fs::remove_all(out, ec);
    fs::remove_all(dir, ec);
    fs::remove_all(scratch, ec);
    fs::remove_all(cache_home, ec);
}

void test_modelbin_genre_refusal(const std::string& root) {
    const fs::path dir = fs::path(root) / "crafted-modelbin";
    fs::remove_all(dir);
    fs::create_directories(dir);
    fs::create_directories(dir / "cache");
    put_file((dir / "model.onnx.data").string(), std::string(4096, '\x11'));
    put_file((dir / "config.json").string(),
             R"({"model_type":"qwen2","hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,)"
             R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"vocab_size":64,)"
             R"("rope_theta":10000.0,"tie_word_embeddings":true})");
    put_file((dir / "model.bin").string(), std::string(8192, '\x22'));   // the file the refusal must name
    put_file((dir / "model.onnx").string(),
             pb::model({pb::tensor("model.embed_tokens.weight", {8, 8}, "model.onnx.data", 0, 256)},
                       {pb::matmulnbits("lm_head", {"x", "lm_head.MatMulNBits.qweight.empty",
                                                    "lm_head.MatMulNBits.scales.empty",
                                                    "lm_head.MatMulNBits.qzeros.empty"}, 512, 64, 4, 128)}));
    pi::Options allow;
    allow.allow_incomplete = true;
    const fs::path out = dir / "out";
    try {
        const auto r = pi::import_pack(dir.string(), out.string(), allow);
        check(false, "modelbin genre: a placeholder-only graph next to model.bin was refused");
    } catch (const std::exception& e) {
        const std::string what = e.what();
        std::printf("  modelbin genre refusal: %s\n", what.c_str());
        check(what.find("model.bin") != std::string::npos,
              "modelbin genre: the refusal names model.bin, not a placeholder tensor (" + what + ")");
        check(what.find("no data") == std::string::npos,
              "modelbin genre: it does not say \"no data\" (the bytes are elsewhere, not absent)");
        check(what.find("engine#1671") != std::string::npos, "modelbin genre: the refusal cites the issue");
    }
    check(!fs::exists(out / "model.q4nx"), "modelbin genre: nothing written");

    // the control: the SAME graph with the placeholder's bytes present must NOT hit this refusal, or the guard would
    // be masking a pack that can be read.
    fs::remove_all(dir / "model.bin");
    put_file((dir / "model.onnx").string(),
             pb::model({pb::tensor("model.embed_tokens.weight", {8, 8}, "model.onnx.data", 0, 256),
                        pb::tensor("lm_head.MatMulNBits.qweight.empty", {}, "model.onnx.data", 256, 4096),
                        pb::tensor("lm_head.MatMulNBits.scales.empty", {}, "model.onnx.data", 4352, 1024),
                        pb::tensor("lm_head.MatMulNBits.qzeros.empty", {}, "model.onnx.data", 5376, 160)},
                       {pb::matmulnbits("lm_head", {"x", "lm_head.MatMulNBits.qweight.empty",
                                                    "lm_head.MatMulNBits.scales.empty",
                                                    "lm_head.MatMulNBits.qzeros.empty"}, 512, 64, 4, 128)}));
    std::string control;
    try {
        const auto r = pi::import_pack(dir.string(), (dir / "out2").string(), allow);
        control = "ok";
    } catch (const std::exception& e) {
        control = e.what();
    }
    check(control.find("model.bin reader") == std::string::npos,
          "modelbin genre: control — the same graph WITH the bytes present is not refused by this guard (" +
              control.substr(0, 90) + ")");
    fs::remove_all(dir);   // leave the tree as we found it, like the crafted leg above
}

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

    // 3. engine#1671: a node names a `.qweight.empty` placeholder and its `.qweight.packed` sibling exists but is
    // ITSELF empty (dims [0], no external-data location). The weights are genuinely absent, so the refusal must say
    // so and name the tensor -- not defer to a `no data` lookup much later, which named the placeholder and read as
    // a tile-layout fault. Measured on Qwen2.5-0.5B-Instruct rai_1.9.0, whose 241 MatMulNBits are all this shape.
    put_file((dir / "model.onnx").string(),
             pb::model({pb::tensor("model.embed_tokens.weight", {8, 8}, "model.onnx.data", 0, 256),
                        pb::tensor("gguf_qweight.empty", {0}, "", 0, 0),
                        pb::tensor("gguf_qweight.packed", {0}, "", 0, 0),
                        pb::tensor("gguf_scales", {}, "model.onnx.data", 256, 1024),
                        pb::tensor("gguf_zeros", {}, "model.onnx.data", 1280, 160)},
                       {pb::matmulnbits("mm", {"x", "gguf_qweight.empty", "gguf_scales", "gguf_zeros"},
                                        64, 512, 4, 128)}));
    try {
        const auto r = pi::import_pack(dir.string(), (dir / "out3").string(), allow);
        check(false, "crafted: an empty .packed sibling was refused");
    } catch (const std::exception& e) {
        const std::string what = e.what();
        std::printf("  crafted packed-empty refusal: %s\n", what.c_str());
        check(what.find("gguf_qweight.empty") != std::string::npos && what.find("gguf_qweight.packed") != std::string::npos,
              "crafted: the refusal names the placeholder and its sibling (" + what + ")");
        check(what.find("genuinely absent") != std::string::npos,
              "crafted: and says the bytes are absent rather than reporting a tile/lookup fault (" + what + ")");
    }
    check(!fs::exists(dir / "out3" / "model.qnx") && !fs::exists(dir / "out3" / "model.q4nx"),
          "crafted: nothing written for the empty .packed sibling");

    // 4. engine#1503: the 1.8 hybrid spelling of the same thing. The node names `<w>` ITSELF as the [0] placeholder (no
    // `.empty` suffix) beside an equally empty `<w>.packed`, and the weights are in a model.bin next to the graph
    // (Llama-3.2-1B-Instruct_rai_1.8.0_hybrid: all 162 `.MatMulNBits.*` initializers are this shape). Before the fix the
    // walker only knew the `.empty` spelling, kept the node as a real weight, `info` called every projection importable
    // and `import` died much later on `...MatMulNBits.bias: no data`.
    put_file((dir / "model.onnx").string(),
             pb::model({pb::tensor("model.embed_tokens.weight", {8, 8}, "model.onnx.data", 0, 256),
                        pb::tensor("rai18_qweight", {0}, "", 0, 0),
                        pb::tensor("rai18_qweight.packed", {0}, "", 0, 0),
                        pb::tensor("rai18_scales", {0}, "", 0, 0),
                        pb::tensor("rai18_zeros", {0}, "", 0, 0)},
                       {pb::matmulnbits("mm", {"x", "rai18_qweight", "rai18_scales", "rai18_zeros"}, 64, 512, 4, 128)}));
    put_file((dir / "model.bin").string(), std::string(64, '\0'));
    try {
        (void)pi::plan(dir.string());
        check(false, "crafted 1.8: a [0] qweight with an empty .packed sibling was refused");
    } catch (const std::exception& e) {
        const std::string what = e.what();
        std::printf("  crafted 1.8 placeholder refusal: %s\n", what.c_str());
        check(what.find("rai18_qweight (its rai18_qweight.packed sibling carries no data either)") != std::string::npos &&
                  what.find("genuinely absent") != std::string::npos,
              "crafted 1.8: plan() refuses by name, naming the placeholder and its sibling (" + what + ")");
        check(what.find("weights are in its model.bin") != std::string::npos,
              "crafted 1.8: and says where the weights are when a model.bin is there (" + what + ")");
    }
    fs::remove(dir / "model.bin");
    try {
        (void)pi::plan(dir.string());
        check(false, "crafted 1.8 without model.bin: refused");
    } catch (const std::exception& e) {
        check(std::string(e.what()).find("model.bin") == std::string::npos,
              "crafted 1.8: no model.bin, so the refusal does not claim one (" + std::string(e.what()) + ")");
    }
    fs::remove_all(dir);
}

// engine#1671, device-free arm for the plan() head gate: a graph whose FIRST MatMulNBits is a projection and whose head comes
// later must report `lm_head.weight` as the head's shape, not the projection's. Revert the gate and this fails.
void test_1671_plan_head_arm(const std::string& root) {
    const fs::path base = fs::path(root) / "plan-head-1671";
    const auto pack = [&](const std::string& tag) {
        const fs::path dir = base / tag;
        fs::remove_all(dir);
        fs::create_directories(dir / "cache");
        put_file((dir / "config.json").string(),
                 R"({"model_type":"qwen2","hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,)"
                 R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"vocab_size":64,)"
                 R"("rope_theta":10000.0,"tie_word_embeddings":false})");
        put_file((dir / "model.onnx.data").string(), std::string(4096, '\x11'));
        return dir;
    };
    // 2. The FIRST MatMulNBits is a projection and the head comes later. plan() took the first node as the head, so it
    // reported the projection's geometry as lm_head.weight; the head is the node whose N == vocab and K == hidden.
    {
        const fs::path dir = pack("head-not-first");
        put_file((dir / "model.onnx").string(),
                 pb::model({pb::tensor("model.embed_tokens.weight", {64, 16}, "model.onnx.data", 0, 4096),
                            pb::tensor("proj_qweight", {}, "model.onnx.data", 3840, 16),
                            pb::tensor("proj_scales", {}, "model.onnx.data", 3856, 16),
                            pb::tensor("proj_zeros", {}, "model.onnx.data", 3872, 8),
                            pb::tensor("head_qweight", {}, "model.onnx.data", 3880, 32),
                            pb::tensor("head_scales", {}, "model.onnx.data", 3912, 16),
                            pb::tensor("head_zeros", {}, "model.onnx.data", 3928, 8)},
                           {pb::matmulnbits("proj", {"x", "proj_qweight", "proj_scales", "proj_zeros"}, 16, 32, 4, 8),
                            pb::matmulnbits("head", {"x", "head_qweight", "head_scales", "head_zeros"}, 16, 64, 4, 8)}));
        const pi::Plan p = pi::plan(dir.string());
        std::vector<int64_t> head;
        int heads = 0;
        for (const auto& t : p.tensors)
            if (t.name == "lm_head.weight") {
                head = t.shape;
                ++heads;
            }
        // exactly one, not merely the right shape: a future bug that pushes two heads (the shape of a bad merge between the two
        // code paths this change unifies) would be tolerated by "take the last match", and the property worth pinning is ONE head
        // chosen by geometry (review of #1732).
        check(heads == 1, "gate: plan() pushes exactly one lm_head.weight (" + std::to_string(heads) + ")");
        check(head.size() == 2 && head.at(0) == 64 && head.at(1) == 16,
              "gate: plan() takes the node whose N x K IS the head, not the first MatMulNBits (lm_head " +
                  (head.size() == 2 ? std::to_string(head.at(0)) + " x " + std::to_string(head.at(1))
                                    : std::string("absent")) +
                  ", and the projection is 32 x 16)");
    }
    fs::remove_all(base);
    fs::remove_all(base);
}

// ── 1. MatMulNBits: encode here, decode with the importer ────────────────────
// engine#1990: a head whose row count is not a multiple of the 32-row tile quantum (CodeLlama's vocabulary is 32,016) is
// padded with rows that decode to zero, and the container's vocab_size is NOT changed -- the padding is invisible to the
// vocabulary. Both halves are pinned here because widening N without staying invisible is the failure that would show up
// as two plausible extra tokens at the end of every distribution.
void test_head_row_padding() {
    const auto make = [](int N) {
        pi::Q4 q;
        q.N = N;
        q.K = 128;      // two 64-column tiles
        q.block = 32;   // four groups per row
        q.codes.assign(size_t(N) * size_t(q.K), 1);          // non-zero, so padding is distinguishable
        q.scale.assign(size_t(N) * size_t(q.K / q.block), 0.5f);   // 1.0 in the padded rows; see below
        q.zero.assign(size_t(N) * size_t(q.K / q.block), 0.25f);
        return q;
    };
    {
        pi::Q4 q = make(32016);
        pi::pad_rows_to_tile_quantum(q, 32016);
        check(q.N == 32032 && q.N % 32 == 0, "head padding: 32,016 rows become 32,032 (the 32-row quantum)");
        check(q.codes.size() == size_t(32032) * 128 && q.scale.size() == size_t(32032) * 4,
              "head padding: codes and scales grow with the rows, one group's worth each");
        bool zeros = true;
        for (int r = 32016; r < q.N; r++) {
            for (int k = 0; k < q.K; k++) zeros = zeros && q.codes[size_t(r) * 128 + size_t(k)] == 0;
            for (int g = 0; g < 4; g++)
                zeros = zeros && q.scale[size_t(r) * 4 + size_t(g)] == 1.0f && q.zero[size_t(r) * 4 + size_t(g)] == 0.0f;
        }
        check(zeros, "head padding: every padded row is code 0, scale 1, zero 0 -- which decodes to 0.0, not to a weight");
        bool kept = true;
        for (int r = 0; r < 32016; r++)
            for (int k = 0; k < 128; k++) kept = kept && q.codes[size_t(r) * 128 + size_t(k)] == 1;
        check(kept, "head padding: the vocabulary's own rows are untouched");
    }
    {   // an aligned vocabulary is left exactly as it was: no note, no growth
        pi::Q4 q = make(32000);
        const size_t before = q.codes.size();
        pi::pad_rows_to_tile_quantum(q, 32000);
        check(q.N == 32000 && q.codes.size() == before, "head padding: a 32-aligned vocabulary (32,000) is not touched");
    }
    {   // 16 rows of slack and 1 row of slack both round up to one tile boundary, not two
        pi::Q4 a = make(32017), b = make(32031);
        pi::pad_rows_to_tile_quantum(a, 32017);
        pi::pad_rows_to_tile_quantum(b, 32031);
        check(a.N == 32032 && b.N == 32032, "head padding: 32,017 and 32,031 both round to 32,032");
    }
}

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

    // Block 32: the decoder is parameterised by the block, and the container's Q4 chunks are per-32-column groups, so
    // a block-32 table is one scale per group rather than two. Pinned because the block-size GATE upstream refuses 32
    // for fconst packs, and the reason it gives ("every subvolume offset in the .fconst moves") cannot apply to a pack
    // whose weights are the graph's own initializers -- amd/Qwen3-VL-4B-Instruct_rai_1.9.0_npu declares 253
    // MatMulNBits ops at block 32 and carries no .fconst at all (engine#1503).
    {
        const int B32 = 32;
        std::vector<uint8_t> qw32(size_t(N) * (K / B32) * (B32 / 2));
        std::vector<float> sc32(size_t(N) * (K / B32));
        for (int n = 0; n < N; ++n)
            for (int b = 0; b < K / B32; ++b) {
                sc32[size_t(n) * (K / B32) + b] = 0.02f * float(b + 1);
                for (int i = 0; i < B32 / 2; ++i) {
                    const int k = b * B32 + 2 * i;
                    qw32[(size_t(n) * (K / B32) + b) * (B32 / 2) + i] =
                        uint8_t(want(n, k) | (want(n, k + 1) << 4));
                }
            }
        const pi::Q4 q32 = pi::decode_matmulnbits(qw32.data(), sc32.data(), nullptr, N, K, B32, false, true);
        bool ok32 = true;
        for (int n = 0; n < N && ok32; ++n)
            for (int k = 0; k < K; ++k) {
                if (q32.codes[size_t(n) * K + k] != want(n, k)) { ok32 = false; break; }
                const size_t g = size_t(n) * (K / 32) + size_t(k) / 32;
                if (q32.scale[g] != sc32[size_t(n) * (K / B32) + size_t(k) / B32]) { ok32 = false; break; }
            }
        check(ok32, "MatMulNBits: block-32 decodes too, one scale per 32-column group");
    }
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

// ── engine#1850: the tied block-32 head synthesis, and the untied refusal, on a crafted pack ──
// #1633 makes `import_pack()` synthesise `lm_head` from the embedding when a 1.9 pack quantises its HEAD at a K block other than
// 128 and declares tied embeddings, refuses an untied pack by name, records `q4nx_head_note`, and self-checks the synthesis
// against the embedding. None of that had an arm: it was exercised only by hand on the real Coder-1.5B pack, and the Python
// tail-head test's "block-32 arm" covers the tool's exit code, not this code. One fixture, two arms, no pack and no device.
void test_tied_block32_head(const std::string& root) {
    const auto build = [&](bool tied, const char* tag) {
        const fs::path dir = fs::path(root) / tag;
        fs::remove_all(dir);
        fs::create_directories(dir / "cache");
        const int H = 128, L = 8, V = 64;
        const int64_t stride = 1 << 20, second = 262144;
        std::string fc(size_t(int64_t(L) * stride + 64 + size_t(H) * 2), '\0');
        auto put_eps = [&](int64_t off) {
            fc[size_t(off)] = char(0x86);
            fc[size_t(off) + 1] = char(0x35);
        };
        for (int l = 0; l < L; ++l) {
            put_eps(int64_t(l) * stride);
            put_eps(int64_t(l) * stride + second);
        }
        put_eps(int64_t(L) * stride);
        for (int i = 0; i < H; ++i) {
            const uint16_t bits = pi::bf16_bits_of(float(i + 1));
            fc[size_t(int64_t(L) * stride + 64 + 2 * i)] = char(bits & 0xFF);
            fc[size_t(int64_t(L) * stride + 64 + 2 * i) + 1] = char(bits >> 8);
        }
        put_file((dir / "dd_metastate_Llm_Token_Token_crafted.fconst").string(), fc);
        const std::string cfg =
            std::string(R"({"model_type":"qwen2","hidden_size":128,"intermediate_size":256,"num_hidden_layers":8,)"
                        R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,"vocab_size":64,)"
                        R"("rope_theta":10000.0,"tie_word_embeddings":)") +
            (tied ? "true" : "false") + R"(,"rms_norm_eps":1e-06})";
        put_file((dir / "config.json").string(), cfg);
        // The head's declared geometry (N=V=64, K=H=128, block=32) has to be carried by the bytes: the gate checks
        // qweight >= N x (K/block) x (block/2) = 4,096 B, f32 scales >= N x (K/block) x 4 = 1,024 B, zeros >= 128 B.
        const int64_t kblocks = H / 32;
        const uint64_t embed_bytes = uint64_t(V) * uint64_t(H) * 2;
        const uint64_t qw_off = embed_bytes, qw_len = uint64_t(V) * uint64_t(kblocks) * 16;
        const uint64_t sc_off = qw_off + qw_len, sc_len = uint64_t(V) * uint64_t(kblocks) * 4;
        const uint64_t z_off = sc_off + sc_len, z_len = uint64_t(V) * ((kblocks + 1) / 2);
        // A VARIED embedding on purpose: a constant fill quantises to one code with a near-zero scale, so the self-check
        // would compute 0/0 and the arm would fail for the fixture's sake rather than the code's.
        std::string data(size_t(z_off + z_len), '\x11');
        for (int r = 0; r < V; ++r)
            for (int c = 0; c < H; ++c) {
                const uint16_t bits = pi::bf16_bits_of(0.05f * float((r * 7 + c * 13) % 97) - 2.4f);
                const size_t at = (size_t(r) * size_t(H) + size_t(c)) * 2;
                data[at] = char(bits & 0xFF);
                data[at + 1] = char(bits >> 8);
            }
        put_file((dir / "model.onnx.data").string(), data);
        // engine#1850: `p.head_block` comes from the pack's `cache/*_meta.json` (read_declared), NOT from the graph -- which
        // is what makes the synthesis reachable at all: the branch needs `onnx.matmuls.empty()`, so the head cannot be a node
        // the walker keeps. Writing the declaration into cache/ is the shape the real 1.9 packs have.
        put_file((dir / "cache" / "lm_head_meta.json").string(),
                 R"({"op_list":[{"type":"MladfMatMul","name":"lm_head.MatMulNBits",)"
                 R"("attrs":{"N":{"value":[64]},"K":{"value":[128]},"block_size":{"value":[32]},"bits":{"value":[4]}}}],)"
                 R"("tensor_map":{}})");
        put_file((dir / "model.onnx").string(),
                 pb::model({pb::tensor("model.embed_tokens.weight", {V, H}, "model.onnx.data", 0,
                                       uint64_t(V) * uint64_t(H) * 2, 10),
                            // PLACEHOLDERS, not weights (engine#1850): the synthesis path runs only when the walk ends
                            // with NO matmuls, which is the shape a real pack has when its head is a placeholder with no
                            // `*.packed` sibling -- the node is dropped from `matmuls` while its `block_size` is still
                            // counted by the census that sets `p.head_block`. A tensor that carries bytes here would be
                            // read by the graph-head loop instead and refused for its declared length.
                            pb::placeholder("lm_head.weight.MatMulNBits.qweight"),
                            pb::placeholder("lm_head.weight.MatMulNBits.scales"),
                            pb::placeholder("lm_head.weight.MatMulNBits.zeros")},
                           {}));
        return dir;
    };

    pi::Options allow;
    allow.allow_incomplete = true;
    {   // arm 1: declared tied -> synthesised from the embedding, and the note says so
        const fs::path dir = build(true, "tied-block32");
        const std::string out = (::onebit::test::tmp_root() /
                                 ("oga_pack_import_tied32." + std::to_string(::getpid()))).string();
        fs::remove_all(out);
        bool threw = false;
        std::string why;
        pi::Report r;
        try {
            r = pi::import_pack(dir.string(), out, allow);
        } catch (const std::exception& e) {
            threw = true;
            why = e.what();
        }
        check(!threw, "tied block-32: a tied pack imports (" + (why.empty() ? std::string("ok") : why.substr(0, 140)) + ")");
        check(!r.head_synthesized_note.empty() && r.head_synthesized_note.find("block 32") != std::string::npos,
              "tied block-32: the synthesis is recorded and names the block (" + r.head_synthesized_note + ")");
        check(r.selfcheck_tied && r.lmhead_corr > 0.999,
              "tied block-32: the self-check measures the SYNTHESIS against the embedding (corr " +
                  std::to_string(r.lmhead_corr) + ")");
        if (fs::exists(out + "/model.q4nx")) {
            // `q4nx_head_note` is written into the container's config.json (that is what the forward reads), not into the
            // q4nx tensor table's metadata.
            std::ifstream cf(out + "/config.json");
            check(cf.good(), "tied block-32: the container carries a config.json");
            if (cf.good()) {
                const json cfg = json::parse(cf);
                check(cfg.contains("q4nx_head_note") && cfg["q4nx_head_note"].get<std::string>().find("block 32") !=
                                                              std::string::npos,
                      "tied block-32: config.json carries q4nx_head_note and names the block");
            }
            check(container_table(out + "/model.q4nx").contains("lm_head.weight"),
                  "tied block-32: the container carries the synthesised head");
            // engine#1882 review: presence is not the assertion. The self-check above measures `q` BEFORE `tile_q4_1_f32`
            // turns it into the bytes the container carries, so an all-zero written head passed every arm -- and that is
            // exactly the state a serve reads. Decode the container's own bytes with the decoder the forward uses.
            const json tbl = container_table(out + "/model.q4nx");
            if (tbl.contains("lm_head.weight")) {
                const std::vector<uint8_t> blob = container_payload(out + "/model.q4nx");
                const auto offs = tbl["lm_head.weight"]["data_offsets"].get<std::vector<uint64_t>>();
                std::vector<float> w(size_t(onebit::npu::q4nx::kRows) * onebit::npu::q4nx::kCols, 0.f);
                onebit::npu::q4nx::dequant_tile(onebit::npu::q4nx::Chunk::Q4_1F, blob.data() + offs[0], w.data(),
                                                onebit::npu::q4nx::kCols);
                // Compare against the embedding THE CONTAINER CARRIES, not against this test's recollection of what it
                // wrote: the synthesis's claim is that the head IS the embedding, so both sides must come from the file.
                std::ifstream cfin(out + "/config.json");
                const json cf = json::parse(cfin);
                const int V = cf.at("vocab_size").get<int>(), H = cf.at("hidden_size").get<int>();
                const auto eoffs = tbl["model.embed_tokens.weight"]["data_offsets"].get<std::vector<uint64_t>>();
                std::vector<float> emb(size_t(V) * size_t(H), 0.f);
                for (size_t i = 0; i < emb.size(); ++i) {
                    const uint16_t b = uint16_t(blob[size_t(eoffs[0]) + 2 * i]) |
                                       (uint16_t(blob[size_t(eoffs[0]) + 2 * i + 1]) << 8);
                    const uint32_t u = uint32_t(b) << 16;
                    float v;
                    std::memcpy(&v, &u, 4);
                    emb[i] = v;
                }
                double num = 0, da = 0, db = 0;
                int n = 0;
                for (int r = 0; r < onebit::npu::q4nx::kRows && r < V; ++r)
                    for (int c = 0; c < H; ++c) {
                        const double want = double(emb[size_t(r) * size_t(H) + size_t(c)]);
                        const double got = double(w[size_t(r) * onebit::npu::q4nx::kCols + size_t(c)]);
                        num += want * got, da += want * want, db += got * got, ++n;
                    }
                const double corr = (da > 0 && db > 0) ? num / std::sqrt(da * db) : 0.0;
                check(n > 0 && corr > 0.999,
                      "tied block-32: the container's head, decoded from its own bytes, is the embedding (corr " +
                          std::to_string(corr) + ", n " + std::to_string(n) + ")");
            }
        }
        fs::remove_all(out);
        fs::remove_all(dir);
    }
    {   // arm 2: the same pack without the declaration -> refused by name, nothing written
        const fs::path dir = build(false, "untied-block32");
        const std::string out = (::onebit::test::tmp_root() /
                                 ("oga_pack_import_untied32." + std::to_string(::getpid()))).string();
        fs::remove_all(out);
        bool threw = false;
        std::string why;
        pi::Report r;
        try {
            // allow-incomplete ON purpose: a strict import bails out at the FIRST refusal (this fixture's layer payload is
            // empty, so it has 80 of them) and never reaches the head gate -- asserting on it there would test the
            // bail-out, not the head. With it on, the run reaches the gate and the refusal lands in `refused`.
            allow.allow_incomplete = true;
            r = pi::import_pack(dir.string(), out, allow);
        } catch (const std::exception& e) {
            threw = true;
            why = e.what();
        }
        std::string said = why;
        for (const auto& w : r.refused)
            if (w.rfind("lm_head.weight", 0) == 0) said += " | " + w;
        check(threw || !r.refused.empty(),
              "untied block-32: an untied pack is refused rather than synthesised (" + said.substr(0, 140) + ")");
        check(said.find("block 32") != std::string::npos && said.find("does not declare tied") != std::string::npos,
              "untied block-32: the refusal names the block and the missing declaration (" + said.substr(0, 160) + ")");
        if (fs::exists(out + "/model.q4nx"))
            check(!container_table(out + "/model.q4nx").contains("lm_head.weight"),
                  "untied block-32: the container carries no head (nothing stands in for an untied one)");
        fs::remove_all(out);
        fs::remove_all(dir);
    }
}

// ── engine#1740: the block-32 head READ from the .fconst tail ──────────────────────────────────────────────────────
// The layout was measured on Qwen-2.5_1.5B_Instruct 1.9 (pack_import.h, above read_block32_head). This writes it from
// that description -- NOT through the reader -- so a reader that disagrees with the description fails here: 64-row
// units of 2,560 zero bytes + 2*(K/32) strips of 640 B; strip s is K block s/2, rows 32*(s%2)..+31; code position
// `(r/8)*256 + c*8 + r%8`, low nibble first; 16 B of zero-point nibbles (8); 48 zero bytes; 32 signed bf16 scales,
// `-m/8` with m the block's largest-magnitude weight.
static std::string encode_block32_units(const std::vector<float>& W, int rows, int K) {
    const int kb = K / 32, units = rows / 64;
    const size_t unit = size_t(2560 + 2 * kb * 640);
    std::string out(size_t(units) * unit, '\0');
    for (int u = 0; u < units; ++u)
        for (int s = 0; s < 2 * kb; ++s) {
            char* sp = out.data() + size_t(u) * unit + 2560 + size_t(s) * 640;
            for (int r = 0; r < 32; ++r) {
                const float* w = W.data() + size_t(u * 64 + 32 * (s % 2) + r) * size_t(K) + size_t((s / 2) * 32);
                float m = 0;
                for (int c = 0; c < 32; ++c)
                    if (std::fabs(w[c]) > std::fabs(m)) m = w[c];
                const uint16_t bits = pi::bf16_bits_of(-m / 8.0f);
                const uint32_t wide = uint32_t(bits) << 16;
                float sc;
                std::memcpy(&sc, &wide, 4);
                sp[576 + 2 * r] = char(bits & 0xFF);
                sp[577 + 2 * r] = char(bits >> 8);
                sp[512 + r / 2] = char(uint8_t(sp[512 + r / 2]) | uint8_t(8 << (4 * (r & 1))));
                for (int c = 0; c < 32; ++c) {
                    const int code = sc == 0 ? 8 : std::min(15, std::max(0, int(std::lround(w[c] / sc)) + 8));
                    const int pos = (r / 8) * 256 + c * 8 + r % 8;
                    sp[pos / 2] = char(uint8_t(sp[pos / 2]) | uint8_t(code << (4 * (pos & 1))));
                }
            }
        }
    return out;
}

void test_block32_tail_head(const std::string& root) {
    const int H = 128, L = 8, V = 64;
    const int R = onebit::npu::q4nx::kRows;   // the rows of container tile 0, which is what the arms decode
    std::vector<float> emb(size_t(V) * H), other(size_t(V) * H);
    for (int r = 0; r < V; ++r)
        for (int c = 0; c < H; ++c) {
            emb[size_t(r) * H + c] = 0.05f * float((r * 7 + c * 13) % 97) - 2.4f;
            other[size_t(r) * H + c] = 0.04f * float((r * 31 + c * 5 + (r * c) % 11) % 89) - 1.7f;   // an UNRELATED head
        }
    // Pearson over the first `a.size()` values of both
    const auto ncorr = [](const std::vector<float>& a, const std::vector<float>& b) {
        double num = 0, da = 0, db = 0, sa = 0, sb = 0;
        const double n = double(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            num += double(a[i]) * b[i], da += double(a[i]) * a[i], db += double(b[i]) * b[i];
            sa += a[i], sb += b[i];
        }
        const double den = (n * da - sa * sa) * (n * db - sb * sb);
        return den > 0 ? (n * num - sa * sb) / std::sqrt(den) : 0.0;
    };
    // The fixture's own premise, asserted: the two matrices must not resemble each other, or "reads the tail" and
    // "falls back to the embedding" could not be told apart by a correlation.
    check(std::fabs(ncorr(emb, other)) < 0.3,
          "block-32 tail: the fixture's two heads are unrelated (corr " + std::to_string(ncorr(emb, other)) + ")");

    // A Ryzen AI fused MLP node as the 1.9 graphs spell it: op SSMLP, the gate/up/down shapes as int attributes, and
    // NO weight tensor (the weights are in the .fconst). Every 1.9 pack measured carries one per layer.
    const auto ssmlp_node = [&]() {
        std::string n;
        pb::str(n, 1, "/model/layers.0/x");
        pb::str(n, 4, "SSMLP");
        const std::pair<const char*, int64_t> attrs[] = {{"gate_K", 128}, {"gate_N", 256}, {"up_K", 128},
                                                         {"up_N", 256},   {"down_K", 256}, {"down_N", 128}};
        for (const auto& [name, value] : attrs) {
            std::string a;
            pb::str(a, 1, name);
            pb::ints(a, 3, value);
            pb::tag(n, 5, 2);
            pb::varint(n, a.size());
            n += a;
        }
        return n;
    };
    const auto build = [&](bool tied, const char* tag, const std::string& tail_after_env, bool fused_mlp = false) {
        const fs::path dir = fs::path(root) / tag;
        fs::remove_all(dir);
        fs::create_directories(dir / "cache");
        const int64_t stride = 1 << 20, second = 262144;
        std::string fc(size_t(int64_t(L) * stride + 64 + size_t(H) * 2), '\0');
        auto put_eps = [&](int64_t off) {
            fc[size_t(off)] = char(0x86);
            fc[size_t(off) + 1] = char(0x35);
        };
        for (int l = 0; l < L; ++l) {
            put_eps(int64_t(l) * stride);
            put_eps(int64_t(l) * stride + second);
        }
        put_eps(int64_t(L) * stride);
        for (int i = 0; i < H; ++i) {
            const uint16_t bits = pi::bf16_bits_of(float(i + 1));
            fc[size_t(int64_t(L) * stride + 64 + 2 * i)] = char(bits & 0xFF);
            fc[size_t(int64_t(L) * stride + 64 + 2 * i) + 1] = char(bits >> 8);
        }
        fc += tail_after_env;
        put_file((dir / "dd_metastate_Llm_Token_Token_crafted.fconst").string(), fc);
        put_file((dir / "config.json").string(),
                 std::string(R"({"model_type":"qwen2","hidden_size":128,"intermediate_size":256,"num_hidden_layers":8,)"
                             R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,"vocab_size":64,)"
                             R"("rope_theta":10000.0,"tie_word_embeddings":)") +
                     (tied ? "true" : "false") + R"(,"rms_norm_eps":1e-06})");
        std::string data(size_t(V) * size_t(H) * 2, '\0');
        for (size_t i = 0; i < emb.size(); ++i) {
            const uint16_t bits = pi::bf16_bits_of(emb[i]);
            data[2 * i] = char(bits & 0xFF);
            data[2 * i + 1] = char(bits >> 8);
        }
        put_file((dir / "model.onnx.data").string(), data);
        // the declaration that sets `p.head_block` (see test_tied_block32_head for why it lives in cache/)
        put_file((dir / "cache" / "lm_head_meta.json").string(),
                 R"({"op_list":[{"type":"MladfMatMul","name":"lm_head.MatMulNBits",)"
                 R"("attrs":{"N":{"value":[64]},"K":{"value":[128]},"block_size":{"value":[32]},"bits":{"value":[4]}}}],)"
                 R"("tensor_map":{}})");
        put_file((dir / "model.onnx").string(),
                 pb::model({pb::tensor("model.embed_tokens.weight", {V, H}, "model.onnx.data", 0,
                                       uint64_t(V) * uint64_t(H) * 2, 16),   // 16 = BFLOAT16: the bytes ARE bf16
                            pb::placeholder("lm_head.weight.MatMulNBits.qweight"),
                            pb::placeholder("lm_head.weight.MatMulNBits.scales"),
                            pb::placeholder("lm_head.weight.MatMulNBits.zeros")},
                           fused_mlp ? std::vector<std::string>{ssmlp_node()} : std::vector<std::string>{}));
        return dir;
    };
    struct Run {
        pi::Report r;
        std::string said, note;
        bool has_head = false;
        std::vector<float> head;   // the container's lm_head tile 0, decoded from its own bytes: [kRows][H]
    };
    const auto run = [&](const fs::path& dir, const char* tag) {
        Run x;
        const std::string out =
            (::onebit::test::tmp_root() /
             (std::string("oga_pack_import_b32tail_") + tag + "." + std::to_string(::getpid()))).string();
        fs::remove_all(out);
        pi::Options allow;
        allow.allow_incomplete = true;   // the fixture's layer payload is empty; the head gate is what is under test
        try {
            x.r = pi::import_pack(dir.string(), out, allow);
        } catch (const std::exception& e) {
            x.said = e.what();
        }
        for (const auto& w : x.r.refused)
            if (w.rfind("lm_head.weight", 0) == 0) x.said += " | " + w;
        if (fs::exists(out + "/model.q4nx")) {
            const json tbl = container_table(out + "/model.q4nx");
            x.has_head = tbl.contains("lm_head.weight");
            if (x.has_head) {
                const std::vector<uint8_t> blob = container_payload(out + "/model.q4nx");
                const auto offs = tbl["lm_head.weight"]["data_offsets"].get<std::vector<uint64_t>>();
                std::vector<float> w(size_t(R) * onebit::npu::q4nx::kCols, 0.f);
                onebit::npu::q4nx::dequant_tile(onebit::npu::q4nx::Chunk::Q4_1F, blob.data() + offs[0], w.data(),
                                                onebit::npu::q4nx::kCols);
                x.head.assign(size_t(R) * H, 0.f);
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c < H; ++c) x.head[size_t(r) * H + c] = w[size_t(r) * onebit::npu::q4nx::kCols + c];
            }
            std::ifstream cf(out + "/config.json");
            if (cf.good()) {
                const json cfg = json::parse(cf);
                if (cfg.contains("q4nx_head_note")) x.note = cfg["q4nx_head_note"].get<std::string>();
            }
        }
        fs::remove_all(out);
        fs::remove_all(dir);
        return x;
    };

    {   // arm 1: UNTIED, the tail holds a head that is NOT the embedding, then padding that is not a whole unit
        const Run x = run(build(false, "b32-untied-read", encode_block32_units(other, V, H) + std::string(10000, '\0')),
                          "untied");
        check(x.said.empty(),
              "block-32 tail: an untied pack whose tail holds the head imports it (" + x.said.substr(0, 160) + ")");
        check(x.has_head && !x.head.empty(), "block-32 tail: the container carries the head");
        if (!x.head.empty()) {
            const double c_tail = ncorr(x.head, other), c_emb = ncorr(x.head, emb);
            check(c_tail > 0.99,
                  "block-32 tail: the container's head is the matrix the TAIL holds (corr " + std::to_string(c_tail) + ")");
            check(std::fabs(c_emb) < 0.3, "block-32 tail: and it is not the embedding (corr " + std::to_string(c_emb) + ")");
        }
        check(x.r.head_synthesized_note.empty() && x.r.head_read_note.find("block 32") != std::string::npos &&
                  x.r.head_read_note.find("untied") != std::string::npos,
              "block-32 tail: the report says the head was read, and that an untied one is not verified (" +
                  x.r.head_read_note + ")");
        check(x.note == x.r.head_read_note && !x.note.empty(), "block-32 tail: config.json carries the same note");
    }
    {   // arm 2: TIED, the tail holds the embedding's own quantisation -> read (not synthesised), verified
        const Run x = run(build(true, "b32-tied-read", encode_block32_units(emb, V, H)), "tied");
        check(x.said.empty(), "block-32 tail, tied: imports (" + x.said.substr(0, 160) + ")");
        check(x.r.head_synthesized_note.empty() && x.r.head_read_note.find("reproduces") != std::string::npos,
              "block-32 tail, tied: the head is READ and the note says it reproduces the embedding (" +
                  x.r.head_read_note + ")");
        check(x.r.selfcheck_tied && x.r.lmhead_corr > 0.99,
              "block-32 tail, tied: the self-check measures the read head (corr " + std::to_string(x.r.lmhead_corr) + ")");
        check(!x.head.empty() && ncorr(x.head, emb) > 0.99,
              "block-32 tail, tied: the container's head is the embedding (corr " +
                  std::to_string(x.head.empty() ? 0.0 : ncorr(x.head, emb)) + ")");
    }
    {   // arm 2b: the same pack with a fused-MLP node in its graph, as every real 1.9 pack has. The node names no weight,
        // so it must not make the graph look like it carries quantised matmuls: that is the switch for "the head is in
        // the .fconst tail", and with it off the import wrote a container with no lm_head at all and exit 0
        // (measured on Qwen-2.5 1.5B 1.9 with the SSMLP attributes counted as graph matmuls: 338 tensors, not 339).
        const Run x = run(build(true, "b32-tied-ssmlp", encode_block32_units(emb, V, H), true), "ssmlp");
        check(x.said.empty(), "block-32 tail, fused MLP in the graph: imports (" + x.said.substr(0, 160) + ")");
        check(x.has_head && !x.head.empty() && ncorr(x.head, emb) > 0.99,
              "block-32 tail, fused MLP in the graph: the container still carries the head read from the tail");
        check(x.r.selfcheck_tied && x.r.head_read_note.find("reproduces") != std::string::npos,
              "block-32 tail, fused MLP in the graph: the tied self-check still runs (" + x.r.head_read_note + ")");
        const fs::path plan_dir = build(true, "b32-tied-ssmlp-plan", encode_block32_units(emb, V, H), true);
        const json g = json::parse(pi::plan(plan_dir.string()).manifest());
        fs::remove_all(plan_dir);
        check(g.value("graph_weights", json::object()).value("matmulnbits_ops", int64_t(-1)) == 0,
              "block-32 tail, fused MLP in the graph: SSMLP attributes are not counted as graph matmuls");
    }
    {   // arm 3: TIED, but the tail's head is some other matrix -> the read is NOT taken, the synthesis stands, says why
        const Run x = run(build(true, "b32-tied-mismatch", encode_block32_units(other, V, H)), "mismatch");
        check(x.r.head_read_note.empty() && x.r.head_synthesized_note.find("correlates") != std::string::npos,
              "block-32 tail, tied mismatch: a tail head that is not the embedding is not taken (" +
                  x.r.head_synthesized_note.substr(0, 260) + ")");
        const double c_emb = x.head.empty() ? 0.0 : ncorr(x.head, emb), c_oth = x.head.empty() ? 1.0 : ncorr(x.head, other);
        check(c_emb > 0.99 && std::fabs(c_oth) < 0.3,
              "block-32 tail, tied mismatch: the container's head is the embedding's synthesis, not the tail's matrix "
              "(corr " + std::to_string(c_emb) + " / " + std::to_string(c_oth) + ")");
    }
    {   // arm 4: UNTIED, a non-zero byte after the units -> refused by name with the reason, nothing written
        std::string tail = encode_block32_units(other, V, H) + std::string(4096, '\0');
        tail[tail.size() - 100] = 1;
        const Run x = run(build(false, "b32-untied-junk", tail), "junk");
        check(!x.has_head, "block-32 tail, junk after the head: no head is written");
        check(x.said.find("block 32") != std::string::npos && x.said.find("does not declare tied") != std::string::npos &&
                  x.said.find("not zero after") != std::string::npos,
              "block-32 tail, junk after the head: the refusal names the block, the missing tie and the reason (" +
                  x.said.substr(0, 300) + ")");
    }
    {   // arm 5: known answers on the reader itself, typed byte by byte (K 64: one unit is 2,560 + 4 x 640 = 5,120 B)
        std::vector<uint8_t> u(5120, 0);
        uint8_t* sp = u.data() + 2560 + 3 * 640;          // strip 3: K block 1, rows 32..63
        const auto put_scale = [&](int r, float v) {
            const uint16_t bits = pi::bf16_bits_of(v);
            sp[576 + 2 * r] = uint8_t(bits & 0xFF);
            sp[577 + 2 * r] = uint8_t(bits >> 8);
        };
        put_scale(9, 0.5f);
        put_scale(10, -0.25f);
        sp[512 + 4] = 0x80;                               // row 9's zero point (high nibble of byte 4) = 8
        sp[512 + 5] = 0x06;                               // row 10's zero point (low nibble of byte 5) = 6
        sp[(256 + 5 * 8 + 1) / 2] |= 0xB0;                // row 9 (band 1, r%8 1), column 5: position 297, high nibble
        sp[(256 + 31 * 8 + 2) / 2] |= 0x03;               // row 10 (band 1, r%8 2), column 31: position 506, low nibble
        pi::Block32Head h = pi::read_block32_head(u.data(), int64_t(u.size()), 64, 64);
        check(h.ok && h.units == 1 && h.q.N == 64 && h.q.K == 64 && h.q.block == 32,
              "block-32 reader: one unit of K 64 reads (" + h.why + ")");
        if (h.ok) {
            const auto val = [&](int row, int col) {
                const size_t g = size_t(row) * 2 + size_t(col / 32);
                return double(h.q.codes[size_t(row) * 64 + col]) * h.q.scale[g] + h.q.zero[g];
            };
            check(val(41, 37) == 1.5,
                  "block-32 reader: (row 41, col 37) = (11 - 8) x 0.5 (" + std::to_string(val(41, 37)) + ")");
            check(val(41, 36) == -4.0,
                  "block-32 reader: its neighbour is code 0, (0 - 8) x 0.5 (" + std::to_string(val(41, 36)) + ")");
            check(val(42, 63) == 0.75,
                  "block-32 reader: a NEGATIVE scale, (3 - 6) x -0.25 (" + std::to_string(val(42, 63)) + ")");
            check(val(42, 32) == 1.5,
                  "block-32 reader: and its code 0, (0 - 6) x -0.25 (" + std::to_string(val(42, 32)) + ")");
            check(h.q.scale[size_t(42) * 2 + 1] == 0.25f, "block-32 reader: the stored scale is positive");
            check(val(41, 5) == 0.0 && val(0, 0) == 0.0, "block-32 reader: a block with a zero scale decodes to zero");
        }
        check(!pi::read_block32_head(u.data(), 5119, 64, 64).ok, "block-32 reader: a tail one byte short is refused");
        std::vector<uint8_t> two = u;
        two.resize(5120 + 300, 0);
        check(pi::read_block32_head(two.data(), int64_t(two.size()), 64, 64).padding == 300,
              "block-32 reader: zeros after the last unit are padding, and counted");
        two.back() = 7;
        check(!pi::read_block32_head(two.data(), int64_t(two.size()), 64, 64).ok,
              "block-32 reader: a non-zero byte after the last unit is refused");
        std::vector<uint8_t> hdr = u;
        hdr[100] = 1;
        check(!pi::read_block32_head(hdr.data(), int64_t(hdr.size()), 64, 64).ok,
              "block-32 reader: a non-zero unit header is refused");
        std::vector<uint8_t> gap = u;
        gap[2560 + 640 + 540] = 1;
        check(!pi::read_block32_head(gap.data(), int64_t(gap.size()), 64, 64).ok,
              "block-32 reader: a non-zero byte between the zero points and the scales is refused");
        check(!pi::read_block32_head(u.data(), int64_t(u.size()), 64, 48).ok, "block-32 reader: K 48 is refused");
    }
}

// ── engine#1503: a pack whose EVERY MatMulNBits is block 32 ────────────────────────────────────────────────────────
// The gate/up unit, written from its description (pack_import.h, above read_block32_gate_up) and NOT through the
// reader: 64-row units of 4,864 zero bytes + 2*(K/32) strips of 1,216 B; strip s is K block s/2 and holds 64 slots,
// slots 0..31 = gate rows 32*(s%2)..+31, slots 32..63 = the same rows of up; code position
// `(slot/8)*256 + c*8 + slot%8`, low nibble first; 32 B of zero-point nibbles (8) by slot; 32 zero bytes; 64 signed
// bf16 scales by slot, `-m/8` as in encode_block32_units.
static std::string encode_block32_gate_up(const std::vector<float>& G, const std::vector<float>& U, int rows, int K) {
    const int kb = K / 32, units = rows / 64;
    const size_t unit = size_t(4864 + 2 * kb * 1216);
    std::string out(size_t(units) * unit, '\0');
    for (int u = 0; u < units; ++u)
        for (int s = 0; s < 2 * kb; ++s) {
            char* sp = out.data() + size_t(u) * unit + 4864 + size_t(s) * 1216;
            for (int slot = 0; slot < 64; ++slot) {
                const std::vector<float>& M = slot < 32 ? G : U;
                const float* w = M.data() + size_t(u * 64 + 32 * (s % 2) + slot % 32) * size_t(K) + size_t((s / 2) * 32);
                float m = 0;
                for (int c = 0; c < 32; ++c)
                    if (std::fabs(w[c]) > std::fabs(m)) m = w[c];
                const uint16_t bits = pi::bf16_bits_of(-m / 8.0f);
                const uint32_t wide = uint32_t(bits) << 16;
                float sc;
                std::memcpy(&sc, &wide, 4);
                sp[1088 + 2 * slot] = char(bits & 0xFF);
                sp[1089 + 2 * slot] = char(bits >> 8);
                sp[1024 + slot / 2] = char(uint8_t(sp[1024 + slot / 2]) | uint8_t(8 << (4 * (slot & 1))));
                for (int c = 0; c < 32; ++c) {
                    const int code = sc == 0 ? 8 : std::min(15, std::max(0, int(std::lround(w[c] / sc)) + 8));
                    const int pos = (slot / 8) * 256 + c * 8 + slot % 8;
                    sp[pos / 2] = char(uint8_t(sp[pos / 2]) | uint8_t(code << (4 * (pos & 1))));
                }
            }
        }
    return out;
}

void test_block32_layers(const std::string& root) {
    const auto ncorr = [](const std::vector<float>& a, const std::vector<float>& b) {
        double num = 0, da = 0, db = 0, sa = 0, sb = 0;
        const double n = double(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            num += double(a[i]) * b[i], da += double(a[i]) * a[i], db += double(b[i]) * b[i];
            sa += a[i], sb += b[i];
        }
        const double den = (n * da - sa * sa) * (n * db - sb * sb);
        return den > 0 ? (n * num - sa * sb) / std::sqrt(den) : 0.0;
    };
    const auto value = [](const pi::Q4& q, int row, int col) {
        const size_t g = size_t(row) * size_t(q.K / 32) + size_t(col / 32);
        return double(q.codes[size_t(row) * size_t(q.K) + col]) * q.scale[g] + q.zero[g];
    };
    // a deterministic matrix per (seed): values in about [-2, 2], unrelated between seeds
    const auto matrix = [](int rows, int cols, uint32_t seed) {
        std::vector<float> m(size_t(rows) * size_t(cols));
        uint32_t x = seed * 2654435761u + 12345u;
        for (auto& v : m) {
            x = x * 1664525u + 1013904223u;
            v = float(int((x >> 8) & 0xFFFF) - 32768) / 16384.0f;
        }
        return m;
    };

    {   // arm 1: the unit reader with ROWS as a parameter, typed bytes (K 64: one unit is 5,120 B)
        std::vector<uint8_t> u(2 * 5120, 0);
        uint8_t* sp = u.data() + 5120 + 2560 + 2 * 640;   // unit 1, strip 2: K block 1, rows 0..31 of the unit
        const uint16_t bits = pi::bf16_bits_of(-0.5f);
        sp[576 + 2 * 20] = uint8_t(bits & 0xFF);          // row 20's scale
        sp[577 + 2 * 20] = uint8_t(bits >> 8);
        sp[512 + 10] = 0x05;                              // row 20's zero point: low nibble of byte 10
        sp[(2 * 256 + 3 * 8 + 4) / 2] |= 0x09;            // row 20 (band 2, r%8 4), column 3: position 540, low nibble
        pi::Block32Units r = pi::read_block32_units(u.data(), int64_t(u.size()), 128, 64, "q_proj", 7000);
        check(r.ok && r.units == 2 && r.bytes == 10240 && r.q.N == 128 && r.q.K == 64 && r.q.block == 32,
              "block-32 units: 128 rows of K 64 are two units (" + r.why + ")");
        if (r.ok) {
            check(value(r.q, 64 + 20, 32 + 3) == -2.0,
                  "block-32 units: unit 1 row 20 col 35 = (9 - 5) x -0.5 (" + std::to_string(value(r.q, 84, 35)) + ")");
            check(value(r.q, 84, 34) == 2.5, "block-32 units: its neighbour is code 0, (0 - 5) x -0.5");
            check(r.q.scale[size_t(84) * 2 + 1] == 0.5f, "block-32 units: the stored scale is positive");
            check(value(r.q, 20, 35) == 0.0 && value(r.q, 84, 3) == 0.0,
                  "block-32 units: the same position in unit 0 and in K block 0 is untouched");
        }
        check(pi::read_block32_units(u.data(), int64_t(u.size()), 65, 64).units == 2,
              "block-32 units: 65 rows round up to two units");
        const pi::Block32Units one = pi::read_block32_units(u.data(), int64_t(u.size()), 64, 64);
        check(one.ok && one.units == 1 && one.q.N == 64,
              "block-32 units: 64 rows read one unit and nothing is required of the bytes after it");
        const pi::Block32Units sh = pi::read_block32_units(u.data(), 10239, 128, 64, "q_proj", 7000);
        check(!sh.ok && sh.why.find("q_proj") != std::string::npos && sh.why.find("7000") != std::string::npos,
              "block-32 units: a buffer one byte short is refused, naming the tensor and the offset (" + sh.why + ")");
        std::vector<uint8_t> hdr = u;
        hdr[5120 + 9] = 1;
        const pi::Block32Units hr = pi::read_block32_units(hdr.data(), int64_t(hdr.size()), 128, 64, "q_proj", 7000);
        check(!hr.ok && hr.why.find("12120 B") != std::string::npos,
              "block-32 units: a non-zero unit header is refused with the unit's offset, 7000 + 5120 (" + hr.why + ")");
        std::vector<uint8_t> nan = u;
        nan[2560 + 576] = 0xC0, nan[2560 + 577] = 0x7F;   // bf16 quiet NaN in unit 0, strip 0, row 0
        check(!pi::read_block32_units(nan.data(), int64_t(nan.size()), 128, 64).ok,
              "block-32 units: a scale that is not finite is refused");
    }

    {   // arm 2: the gate/up reader, typed bytes (K 32: one unit is 4,864 + 2 x 1,216 = 7,296 B)
        check(pi::block32_mlp_unit_bytes(32) == 7296 && pi::block32_mlp_unit_bytes(576) == 48640,
              "block-32 gate/up: the unit is 4,864 + 2*(K/32)*1,216 B (48,640 at K 576)");
        std::vector<uint8_t> u(7296, 0);
        uint8_t* sp = u.data() + 4864 + 1216;             // strip 1: K block 0, rows 32..63
        const auto put_scale = [&](int slot, float v) {
            const uint16_t bits = pi::bf16_bits_of(v);
            sp[1088 + 2 * slot] = uint8_t(bits & 0xFF);
            sp[1089 + 2 * slot] = uint8_t(bits >> 8);
        };
        put_scale(5, 0.5f);                               // slot 5: GATE row 32 + 5
        put_scale(41, -0.25f);                            // slot 41: UP row 32 + 9
        sp[1024 + 2] = 0x80;                              // slot 5's zero point (high nibble of byte 2) = 8
        sp[1024 + 20] = 0x60;                             // slot 41's zero point (high nibble of byte 20) = 6
        sp[(0 * 256 + 7 * 8 + 5) / 2] |= 0xB0;            // slot 5 (band 0, 5), column 7: position 61, high nibble
        sp[(5 * 256 + 31 * 8 + 1) / 2] |= 0x30;           // slot 41 (band 5, 1), column 31: position 1529, high nibble
        pi::Block32GateUp h = pi::read_block32_gate_up(u.data(), int64_t(u.size()), 64, 32, 900);
        check(h.ok && h.units == 1 && h.bytes == 7296 && h.gate.N == 64 && h.up.N == 64 && h.gate.K == 32 &&
                  h.up.block == 32,
              "block-32 gate/up: one unit of K 32 reads (" + h.why + ")");
        if (h.ok) {
            check(value(h.gate, 37, 7) == 1.5,
                  "block-32 gate/up: slot 5 is GATE row 37, (11 - 8) x 0.5 (" + std::to_string(value(h.gate, 37, 7)) + ")");
            check(value(h.gate, 37, 6) == -4.0, "block-32 gate/up: its neighbour is code 0, (0 - 8) x 0.5");
            check(value(h.up, 37, 7) == 0.0, "block-32 gate/up: and UP row 37 holds nothing");
            check(value(h.up, 41, 31) == 0.75,
                  "block-32 gate/up: slot 41 is UP row 41, a NEGATIVE scale, (3 - 6) x -0.25 (" +
                      std::to_string(value(h.up, 41, 31)) + ")");
            check(value(h.up, 41, 0) == 1.5, "block-32 gate/up: its code 0, (0 - 6) x -0.25");
            check(h.up.scale[41] == 0.25f, "block-32 gate/up: the stored scale is positive");
            check(value(h.gate, 41, 31) == 0.0 && value(h.gate, 5, 7) == 0.0,
                  "block-32 gate/up: GATE row 41 and the other strip's row 5 hold nothing");
        }
        const pi::Block32GateUp sh = pi::read_block32_gate_up(u.data(), 7295, 64, 32, 900);
        check(!sh.ok && sh.why.find("900") != std::string::npos,
              "block-32 gate/up: a buffer one byte short is refused with the offset (" + sh.why + ")");
        std::vector<uint8_t> hdr = u;
        hdr[4863] = 1;
        check(!pi::read_block32_gate_up(hdr.data(), int64_t(hdr.size()), 64, 32).ok,
              "block-32 gate/up: a non-zero unit header is refused");
        std::vector<uint8_t> gap = u;
        gap[4864 + 1216 + 1056 + 31] = 1;
        check(!pi::read_block32_gate_up(gap.data(), int64_t(gap.size()), 64, 32).ok,
              "block-32 gate/up: a non-zero byte between the zero points and the scales is refused");
        std::vector<uint8_t> inf = u;
        inf[4864 + 1088 + 2 * 63] = 0x80, inf[4864 + 1089 + 2 * 63] = 0x7F;   // bf16 +inf in strip 0, slot 63
        check(!pi::read_block32_gate_up(inf.data(), int64_t(inf.size()), 64, 32).ok,
              "block-32 gate/up: a scale that is not finite is refused");
        check(!pi::read_block32_gate_up(u.data(), int64_t(u.size()), 64, 48).ok, "block-32 gate/up: K 48 is refused");
    }

    {   // arm 3: the description's encoder against the reader, two unrelated matrices over two units and three K blocks
        const int rows = 128, K = 96;
        const std::vector<float> G = matrix(rows, K, 1), U = matrix(rows, K, 2);
        const std::string enc = encode_block32_gate_up(G, U, rows, K);
        const pi::Block32GateUp h =
            pi::read_block32_gate_up(reinterpret_cast<const uint8_t*>(enc.data()), int64_t(enc.size()), rows, K);
        check(h.ok, "block-32 gate/up: the encoded stream reads (" + h.why + ")");
        if (h.ok) {
            std::vector<float> g(G.size()), up(U.size());
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < K; ++c) {
                    g[size_t(r) * K + c] = float(value(h.gate, r, c));
                    up[size_t(r) * K + c] = float(value(h.up, r, c));
                }
            check(ncorr(g, G) > 0.99 && ncorr(up, U) > 0.99,
                  "block-32 gate/up: gate and up come back (corr " + std::to_string(ncorr(g, G)) + " / " +
                      std::to_string(ncorr(up, U)) + ")");
            check(std::fabs(ncorr(g, U)) < 0.3 && std::fabs(ncorr(up, G)) < 0.3,
                  "block-32 gate/up: and not swapped (corr " + std::to_string(ncorr(g, U)) + " / " +
                      std::to_string(ncorr(up, G)) + ")");
        }
    }

    {   // arm 4: the layer block's arithmetic at the one measured geometry (SmolLM2-135M-Instruct 1.9)
        pi::Dims d;
        d.hidden = 576, d.intermediate = 1536, d.layers = 30, d.heads = 9, d.kv_heads = 3, d.head_dim = 64;
        d.vocab = 49152;
        const pi::Block32Layout l = pi::block32_layout(d);
        check(l.why.empty() && l.q_off == 1216 && l.k_off == 231616 && l.v_off == 308416 && l.sink_off == 385216 &&
                  l.o_off == 385252 && l.post_off == 615652 && l.gu_off == 616868 && l.hdr_off == 1784228 &&
                  l.down_off == 1784292 && l.stride == 2360292,
              "block-32 layout: SmolLM2-135M's layer block, stride 2,360,292 (" + l.why + " " +
                  std::to_string(l.stride) + ")");
        check(l.head_off == 70809976 && l.head_units == 768 && l.min_size == 90470776,
              "block-32 layout: its head at 70,809,976 and the file's 90,470,776 B (" + std::to_string(l.min_size) + ")");
        d.intermediate = 1500;
        check(pi::block32_layout(d).why.find("intermediate 1500") != std::string::npos,
              "block-32 layout: a width that is not whole 64-row units is refused by name");
    }

    // arm 5: a crafted all-block-32 pack through import_pack(). hidden 128, intermediate 256, 2 heads x 64, 1 KV head,
    // 2 layers, vocab 64, tied. Every matrix is its own pseudo-random one, so a tensor read from another's offset
    // cannot correlate with what the container is compared against.
    const int H = 128, IM = 256, LAYERS = 2, V = 64, NQ = 128, NKV = 64;
    const int R = onebit::npu::q4nx::kRows;
    enum { Q, K_, V_, O, G, U, D, NPROJ };
    const char* const pname[NPROJ] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                      "mlp.gate_proj",    "mlp.up_proj",      "mlp.down_proj"};
    const int prow[NPROJ] = {NQ, NKV, NKV, H, IM, IM, H}, pcol[NPROJ] = {H, H, H, NQ, H, H, IM};
    std::vector<std::vector<float>> mats;
    for (int l = 0; l < LAYERS; ++l)
        for (int i = 0; i < NPROJ; ++i) mats.push_back(matrix(prow[i], pcol[i], uint32_t(100 + l * NPROJ + i)));
    const std::vector<float> emb = matrix(V, H, 7);
    const auto norm_bytes = [&](int which) {   // bf16 [H], a different ramp per norm
        std::string b(size_t(H) * 2, '\0');
        for (int i = 0; i < H; ++i) {
            const uint16_t bits = pi::bf16_bits_of(float(which) + float(i) / 256.0f);
            b[2 * i] = char(bits & 0xFF), b[2 * i + 1] = char(bits >> 8);
        }
        return b;
    };
    const auto envelope = [&](int which) {
        std::string e(64, '\0');
        e[0] = char(0x86), e[1] = char(0x35);   // bf16 1e-06
        return e + norm_bytes(which);
    };
    const auto fconst = [&] {
        std::string fc;
        for (int l = 0; l < LAYERS; ++l) {
            const auto& m = [&](int i) -> const std::vector<float>& { return mats[size_t(l * NPROJ + i)]; };
            fc += envelope(10 * l + 1);
            fc += encode_block32_units(m(Q), NQ, H) + encode_block32_units(m(K_), NKV, H) +
                  encode_block32_units(m(V_), NKV, H);
            fc += std::string(4 * 2, '\x7f');   // 4 x heads bytes that are not a weight (their content is not read)
            fc += encode_block32_units(m(O), H, NQ);
            fc += envelope(10 * l + 2);
            fc += encode_block32_gate_up(m(G), m(U), IM, H);
            fc += std::string(1, '\x01') + std::string(63, '\0');
            fc += encode_block32_units(m(D), H, IM);
        }
        fc += envelope(99);
        fc += encode_block32_units(emb, V, H);
        return fc;
    };
    const std::string good = fconst();
    {
        pi::Dims d;
        d.hidden = H, d.intermediate = IM, d.layers = LAYERS, d.heads = 2, d.kv_heads = 1, d.head_dim = 64, d.vocab = V;
        check(pi::block32_layout(d).min_size == int64_t(good.size()),
              "all-block-32 fixture: the stream built from the description is the layout's size (" +
                  std::to_string(good.size()) + ")");
    }
    // `layer_block` / `head_block` are what cache/ops_meta.json DECLARES; the .fconst bytes are the caller's.
    const auto build = [&](const char* tag, const std::string& fc, bool layers_declared, int layer_block = 32,
                           int head_block = 32) {
        const fs::path dir = fs::path(root) / tag;
        fs::remove_all(dir);
        fs::create_directories(dir / "cache");
        // the 1.9 spelling of the Token file: this path reads the raw file, and it must not be mistaken for a 1.8 one
        put_file((dir / "dd_metastate_Llm_Token_aie2_token_crafted.fconst").string(), fc);
        put_file((dir / "config.json").string(),
                 R"({"model_type":"llama","hidden_size":128,"intermediate_size":256,"num_hidden_layers":2,)"
                 R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,"vocab_size":64,)"
                 R"("rope_theta":10000.0,"tie_word_embeddings":true,"rms_norm_eps":1e-06})");
        std::string data(size_t(V) * size_t(H) * 2, '\0');
        for (size_t i = 0; i < emb.size(); ++i) {
            const uint16_t bits = pi::bf16_bits_of(emb[i]);
            data[2 * i] = char(bits & 0xFF);
            data[2 * i + 1] = char(bits >> 8);
        }
        put_file((dir / "model.onnx.data").string(), data);
        const auto op_at = [](const std::string& name, int n, int k, int block) {
            return R"({"type":"MladfMatMul","name":")" + name + R"(.MatMulNBits","attrs":{"N":{"value":[)" +
                   std::to_string(n) + R"(]},"K":{"value":[)" + std::to_string(k) + R"(]},"block_size":{"value":[)" +
                   std::to_string(block) + R"(]},"bits":{"value":[4]}}})";
        };
        const auto op = [&](const std::string& name, int n, int k) { return op_at(name, n, k, layer_block); };
        std::string ops = op_at("lm_head", V, H, head_block);
        if (layers_declared)
            for (int l = 0; l < LAYERS; ++l) {
                const std::string a = "model.layers." + std::to_string(l) + ".attn.",
                                  m = "model.layers." + std::to_string(l) + ".mlp.";
                ops += "," + op(a + "q_proj", NQ, H) + "," + op(a + "k_proj", NKV, H) + "," + op(a + "v_proj", NKV, H) +
                       "," + op(a + "o_proj", H, NQ) + "," + op(m + "down_proj", H, IM);
            }
        put_file((dir / "cache" / "ops_meta.json").string(), R"({"op_list":[)" + ops + R"(],"tensor_map":{}})");
        put_file((dir / "model.onnx").string(),
                 pb::model({pb::tensor("model.embed_tokens.weight", {V, H}, "model.onnx.data", 0,
                                       uint64_t(V) * uint64_t(H) * 2, 16),
                            pb::placeholder("lm_head.weight.MatMulNBits.qweight"),
                            pb::placeholder("lm_head.weight.MatMulNBits.scales"),
                            pb::placeholder("lm_head.weight.MatMulNBits.zeros")},
                           {}));
        return dir;
    };
    const auto out_of = [](const char* tag) {
        return (fs::temp_directory_path() /
                (std::string("oga_pack_import_b32layers_") + tag + "." + std::to_string(::getpid()))).string();
    };
    // the first `cols` columns of rows 0..R-1 of a container tensor, from its own bytes (tile 0)
    const auto tile0 = [&](const json& tbl, const std::vector<uint8_t>& blob, const std::string& name, int cols) {
        std::vector<float> w(size_t(R) * onebit::npu::q4nx::kCols, 0.f), out(size_t(R) * size_t(cols), 0.f);
        if (!tbl.contains(name)) return std::vector<float>();
        const auto offs = tbl[name]["data_offsets"].get<std::vector<uint64_t>>();
        onebit::npu::q4nx::dequant_tile(onebit::npu::q4nx::Chunk::Q4_1F, blob.data() + offs[0], w.data(),
                                        onebit::npu::q4nx::kCols);
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < cols; ++c) out[size_t(r) * cols + c] = w[size_t(r) * onebit::npu::q4nx::kCols + c];
        return out;
    };
    const auto head_of = [&](const std::vector<float>& m, int cols_total, int cols) {   // the same window of a source
        std::vector<float> out(size_t(R) * size_t(cols));
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < cols; ++c) out[size_t(r) * cols + c] = m[size_t(r) * cols_total + c];
        return out;
    };

    {   // 5a: the pack imports, complete, and every container tensor is the matrix the description put at its offset
        const fs::path dir = build("b32-layers", good, true);
        const std::string out = out_of("good");
        fs::remove_all(out);
        pi::Report r;
        std::string said;
        try {
            const pi::Plan p = pi::plan(dir.string());
            check(p.block32_layers && p.block32_note.empty() && p.refused().empty(),
                  "all-block-32: the plan takes the block-32 layout and refuses nothing (" + p.block32_note + ")");
            r = pi::import_pack(dir.string(), out, pi::Options{});
        } catch (const std::exception& e) {
            said = e.what();
        }
        check(said.empty() && r.refused.empty(),
              "all-block-32: a crafted pack imports with nothing refused (" + said +
                  (r.refused.empty() ? "" : r.refused.front()) + ")");
        check(r.attn_projections == 4 * LAYERS && r.mlp_projections == 3 * LAYERS && r.attn_bias_tensors == 0,
              "all-block-32: 4 attention + 3 MLP projections per layer, no bias tensor");
        check(r.stride == (int64_t(good.size()) - (64 + 2 * H) - pi::block32_unit_bytes(H)) / LAYERS,
              "all-block-32: the report carries the stride (" + std::to_string(r.stride) + ")");
        check(r.selfcheck_tied && r.lmhead_corr > 0.99 && r.head_read_note.find("block 32") != std::string::npos,
              "all-block-32: the head is read and verified against the embedding (corr " +
                  std::to_string(r.lmhead_corr) + ")");
        if (fs::exists(out + "/model.q4nx")) {
            const json tbl = container_table(out + "/model.q4nx");
            const std::vector<uint8_t> blob = container_payload(out + "/model.q4nx");
            for (int l = 0; l < LAYERS; ++l)
                for (int i = 0; i < NPROJ; ++i) {
                    const std::string name = "model.layers." + std::to_string(l) + "." + pname[i] + ".weight";
                    const int cols = std::min(pcol[i], onebit::npu::q4nx::kCols);
                    const std::vector<float> got = tile0(tbl, blob, name, cols);
                    const double c = got.empty() ? 0.0 : ncorr(got, head_of(mats[size_t(l * NPROJ + i)], pcol[i], cols));
                    check(c > 0.99, "all-block-32: container " + name + " is its own matrix (corr " +
                                        std::to_string(c) + ")");
                    // the control: the matrix one slot along (gate <-> up among them) must not match
                    const int j = (i == G) ? U : (i == U) ? G : -1;
                    if (j >= 0 && !got.empty()) {
                        const double x = ncorr(got, head_of(mats[size_t(l * NPROJ + j)], pcol[j], cols));
                        check(std::fabs(x) < 0.3, "all-block-32: container " + name + " is not the other MLP half "
                                                  "(corr " + std::to_string(x) + ")");
                    }
                }
            const std::vector<float> hd = tile0(tbl, blob, "lm_head.weight", H);
            check(!hd.empty() && ncorr(hd, head_of(emb, H, H)) > 0.99, "all-block-32: container lm_head is the tail's head");
            const auto norm_is = [&](const std::string& name, int which) {
                if (!tbl.contains(name)) return false;
                const auto offs = tbl[name]["data_offsets"].get<std::vector<uint64_t>>();
                const std::string want = norm_bytes(which);
                return offs[1] - offs[0] == want.size() && std::memcmp(blob.data() + offs[0], want.data(), want.size()) == 0;
            };
            check(norm_is("model.layers.0.input_layernorm.weight", 1) &&
                      norm_is("model.layers.1.input_layernorm.weight", 11) &&
                      norm_is("model.layers.0.post_attention_layernorm.weight", 2) &&
                      norm_is("model.layers.1.post_attention_layernorm.weight", 12) && norm_is("model.norm.weight", 99),
                  "all-block-32: the five norms are the envelopes' bytes");
            check(!tbl.contains("model.layers.0.self_attn.q_proj.bias"), "all-block-32: no bias tensor is invented");
        } else {
            check(false, "all-block-32: the container was written");
        }
        fs::remove_all(out);
        fs::remove_all(dir);
    }
    // what a refused import says, and that it wrote no container
    const auto refused = [&](const char* tag, const std::string& fc) {
        const fs::path dir = build(tag, fc, true);
        const std::string out = out_of(tag);
        fs::remove_all(out);
        std::string said;
        try {
            const pi::Report r = pi::import_pack(dir.string(), out, pi::Options{});
            for (const auto& w : r.refused) said += w + " | ";
        } catch (const std::exception& e) {
            said = e.what();
        }
        if (fs::exists(out + "/model.q4nx")) said = "WROTE A CONTAINER " + said;
        fs::remove_all(out);
        fs::remove_all(dir);
        return said;
    };
    {   // 5b: a file one byte short of the layout is refused with the arithmetic
        const std::string said = refused("b32-layers-short", good.substr(0, good.size() - 1));
        check(said.find("WROTE") == std::string::npos && said.find("all-block-32 layout") != std::string::npos &&
                  said.find(std::to_string(good.size())) != std::string::npos,
              "all-block-32: a short file is refused with the size the layout needs (" + said.substr(0, 300) + ")");
    }
    {   // 5c: layer 1's post-attention envelope without its epsilon
        pi::Dims d;
        d.hidden = H, d.intermediate = IM, d.layers = LAYERS, d.heads = 2, d.kv_heads = 1, d.head_dim = 64, d.vocab = V;
        const pi::Block32Layout l = pi::block32_layout(d);
        std::string bad = good;
        bad[size_t(l.stride + l.post_off)] = 0;
        const std::string said = refused("b32-layers-eps", bad);
        check(said.find("WROTE") == std::string::npos && said.find("no RMSNorm epsilon at " +
                                                                   std::to_string(l.stride + l.post_off)) !=
                                                             std::string::npos,
              "all-block-32: a missing epsilon is refused with its offset (" + said.substr(0, 300) + ")");
        // 5d: a non-zero byte in a gate/up unit header of layer 1 -> both halves refused by name
        bad = good;
        bad[size_t(l.stride + l.gu_off + pi::block32_mlp_unit_bytes(H) + 17)] = 1;
        const std::string s2 = refused("b32-layers-gu", bad);
        check(s2.find("WROTE") == std::string::npos &&
                  s2.find("model.layers.1.mlp.gate_proj.weight: block-32 gate/up unit 1") != std::string::npos &&
                  s2.find("model.layers.1.mlp.up_proj.weight") != std::string::npos &&
                  s2.find("model.layers.0") == std::string::npos,
              "all-block-32: a non-zero gate/up unit header refuses that layer's gate and up by name (" +
                  s2.substr(0, 300) + ")");
        // 5e: trailing junk after the head is the head's refusal
        const std::string s3 = refused("b32-layers-junk", good + std::string(99, '\0') + "x");
        check(s3.find("WROTE") == std::string::npos && s3.find("lm_head.weight") != std::string::npos &&
                  s3.find("not zero after") != std::string::npos,
              "all-block-32: a non-zero byte after the head refuses the head (" + s3.substr(0, 300) + ")");
    }
    {   // 5f: zero padding after the head is accepted (the measured mixed pack has 110 MB of it)
        const fs::path dir = build("b32-layers-pad", good + std::string(5000, '\0'), true);
        const std::string out = out_of("pad");
        fs::remove_all(out);
        std::string said;
        try {
            const pi::Report r = pi::import_pack(dir.string(), out, pi::Options{});
            for (const auto& w : r.refused) said += w + " | ";
        } catch (const std::exception& e) {
            said = e.what();
        }
        check(said.empty() && fs::exists(out + "/model.q4nx"), "all-block-32: zero padding after the head imports (" +
                                                                   said.substr(0, 200) + ")");
        fs::remove_all(out);
        fs::remove_all(dir);
    }
    {   // 5g: the same pack declaring ONLY its head at block 32 is not this class (it is the mixed pack's path)
        const fs::path dir = build("b32-layers-headonly", good, false);
        bool block32 = true;
        try {
            block32 = pi::plan(dir.string()).block32_layers;
        } catch (const std::exception&) {
            block32 = false;
        }
        check(!block32, "all-block-32: a pack that declares only lm_head at block 32 does not take the all-block-32 path");
        fs::remove_all(dir);
    }
    // 5h..5k: the path is chosen by what the pack DECLARES, one arm per declaration. The bytes are the same good
    // all-block-32 file every time, so the only thing that changes is cache/ops_meta.json -- and a declaration this
    // importer has no layout for must be refused with its number, never read at 32 or at 128 by default.
    const auto declared_as = [&](const char* tag, int layer_block, int head_block) {
        const fs::path dir = build(tag, good, true, layer_block, head_block);
        const std::string out = out_of(tag);
        fs::remove_all(out);
        std::string said;
        bool block32 = false;
        try {
            block32 = pi::plan(dir.string()).block32_layers;
            const pi::Report r = pi::import_pack(dir.string(), out, pi::Options{});
            for (const auto& w : r.refused) said += w + " | ";
        } catch (const std::exception& e) {
            said = e.what();
        }
        if (block32) said = "TOOK THE ALL-BLOCK-32 PATH " + said;
        if (fs::exists(out + "/model.q4nx")) said = "WROTE A CONTAINER " + said;
        fs::remove_all(out);
        fs::remove_all(dir);
        return said;
    };
    for (const int b : {3, 64, 256}) {   // 5h: every op at a block nobody has measured
        const std::string said = declared_as(("b32-layers-all" + std::to_string(b)).c_str(), b, b);
        check(said.find("WROTE") == std::string::npos && said.find("TOOK") == std::string::npos &&
                  said.find("block size along K is " + std::to_string(b) + " (11 ops)") != std::string::npos &&
                  said.find("lm_head @ " + std::to_string(b)) != std::string::npos,
              "block declaration: every op at " + std::to_string(b) + " is refused with that number (" +
                  said.substr(0, 200) + ")");
    }
    {   // 5i: layers at 32 under a block-128 head is not the all-block-32 class, and is refused by tensor
        const std::string said = declared_as("b32-layers-head128", 32, 128);
        check(said.find("WROTE") == std::string::npos && said.find("TOOK") == std::string::npos &&
                  said.find("128 (1 ops)") != std::string::npos && said.find("32 (10 ops)") != std::string::npos &&
                  said.find("lm_head @") == std::string::npos,
              "block declaration: layers at 32 with the head at 128 is refused, naming the layers (" +
                  said.substr(0, 260) + ")");
    }
    {   // 5j: layers at 32 with the head at 64 -- one unmeasured op is enough to leave the class
        const std::string said = declared_as("b32-layers-head64", 32, 64);
        check(said.find("WROTE") == std::string::npos && said.find("TOOK") == std::string::npos &&
                  said.find("64 (1 ops)") != std::string::npos,
              "block declaration: layers at 32 with the head at 64 is refused (" + said.substr(0, 260) + ")");
    }
    {   // 5k: the same ops all declared at 128 stay on the block-128 walk (which then refuses THESE bytes, because
        // they are a block-32 file: the point is that the declaration, not a default, picked the path)
        const std::string said = declared_as("b32-layers-all128", 128, 128);
        check(said.find("WROTE") == std::string::npos && said.find("TOOK") == std::string::npos &&
                  said.find("block size along K") == std::string::npos && !said.empty(),
              "block declaration: every op at 128 does not take the all-block-32 path or the block gate (" +
                  said.substr(0, 260) + ")");
    }
}

// ── the final norm out of the tail envelope, on a crafted pack (engine#844) ──────────────────
// The real-pack leg only covers this where a pack is supplied, and three of the four packs on the
// box ship no prefill program, so the import's tail-envelope read had no coverage in the default
// CTest path. This builds the smallest pack that reaches it and checks the bytes that land in the
// container -- plus the control: with the tail epsilon corrupted the norm must be refused, nothing
// may be written without --allow-incomplete, and with it the omission must be recorded.
void test_final_norm_envelope(const std::string& root) {
    const fs::path dir = fs::path(root) / "final-norm";
    fs::remove_all(dir);
    fs::create_directories(dir / "cache");
    const int H = 128, L = 8;   // locate_fconst refuses fewer than 6 epsilon headers in the first GiB
    const int64_t stride = 1 << 20;      // locate_fconst ignores periods below 1e6
    const int64_t second = 262144;
    const size_t unit = size_t(L * stride) + size_t(64) + size_t(H * 2);
    std::string fc(unit, '\0');
    auto put_eps = [&](int64_t off, unsigned short bits) {
        fc[size_t(off)] = char(bits & 0xFF);
        fc[size_t(off) + 1] = char(bits >> 8);
    };
    for (int l = 0; l < L; ++l) {
        put_eps(int64_t(l) * stride, 0x3586);                 // the block header
        put_eps(int64_t(l) * stride + second, 0x3586);        // the post-attention header
    }
    put_eps(int64_t(L) * stride, 0x3586);                     // the tail envelope
    for (int i = 0; i < H; ++i) {                             // a recognisable norm: 1.0, 2.0, 3.0 ...
        const uint16_t bits = pi::bf16_bits_of(float(i + 1));
        fc[size_t(int64_t(L) * stride + 64 + 2 * i)] = char(bits & 0xFF);
        fc[size_t(int64_t(L) * stride + 64 + 2 * i) + 1] = char(bits >> 8);
    }
    put_file((dir / "config.json").string(),
             R"({"model_type":"qwen2","hidden_size":128,"intermediate_size":256,"num_hidden_layers":8,)"
             R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,"vocab_size":128,)"
             R"("rope_theta":10000.0,"tie_word_embeddings":true,"rms_norm_eps":1e-06})");
    // pb::tensor's default element type is f32, and the importer's over-read guard takes 4 B per element
    // for anything that is not declared f32 (2 B), so the data is one f32 per element here.
    put_file((dir / "model.onnx.data").string(), std::string(size_t(H) * size_t(H) * 4, '\x11'));
    put_file((dir / "model.onnx").string(),
             pb::model({pb::tensor("model.embed_tokens.weight", {H, H}, "model.onnx.data", 0,
                                   int64_t(H) * H * 4)}, {}));
    const std::string tok = (dir / "dd_metastate_Llm_Token_Token_crafted.fconst").string();
    put_file(tok, fc);

    const pi::FconstGeom geom = pi::locate_fconst(tok, 0x3586);
    check(geom.stride == stride && geom.h1 == second,
          "final norm: the crafted .fconst is locatable (stride " + std::to_string(geom.stride) +
              (geom.why.empty() ? "" : ", why: " + geom.why) + ")");
    pi::Options allow;
    allow.allow_incomplete = true;
    const std::string out = (::onebit::test::tmp_root() /
                             ("oga_pack_import_norm." + std::to_string(::getpid()))).string();
    fs::remove_all(out);
    const pi::Report r = pi::import_pack(dir.string(), out, allow);
    check(r.final_norm_imported, "final norm: the crafted tail envelope yields model.norm.weight");
    check(!r.prefill_tail_seen && !r.final_norm_crosscheck,
          "final norm: with no prefill program the cross-check is unavailable, not failed");
    if (r.final_norm_imported) {
        onebit::npu::Model m(out);
        const auto& t = m.tensor("model.norm.weight");
        bool same = t.bytes >= size_t(H) * 2;
        for (int i = 0; same && i < H; ++i) {
            const uint16_t want = pi::bf16_bits_of(float(i + 1));
            same = t.data[2 * i] == uint8_t(want & 0xFF) && t.data[2 * i + 1] == uint8_t(want >> 8);
        }
        check(same, "final norm: the container carries exactly the bf16 bytes at tail+64");
    }
    fs::remove_all(out);

    // control: corrupt the tail epsilon -- the norm must be refused, and the refusal must be recorded
    std::string bad = fc;
    bad[size_t(int64_t(L) * stride)] = '\x28';
    bad[size_t(int64_t(L) * stride) + 1] = '\x37';   // 1e-5, not this pack's 1e-6
    put_file(tok, bad);
    std::string refused_by_name = "0123456789";
    bool wrote_anyway = false;
    try {
        const pi::Report r0 = pi::import_pack(dir.string(), out, pi::Options{});
        wrote_anyway = fs::exists(out + "/model.q4nx");
        (void)r0;
    } catch (const std::exception&) {
        wrote_anyway = fs::exists(out + "/model.q4nx");
    }
    check(!wrote_anyway, "final norm: a pack whose tail epsilon is missing writes nothing by default");
    fs::remove_all(out);
    const pi::Report rb = pi::import_pack(dir.string(), out, allow);
    bool named = false;
    for (const auto& s : rb.refused)
        if (s.rfind("model.norm.weight", 0) == 0) named = true;
    check(!rb.final_norm_imported && named,
          "final norm: a missing tail epsilon refuses model.norm.weight by name");
    // ... and the container it still writes must say so (the manifest merge, engine#842 review)
    if (named && fs::exists(out + "/model.q4nx")) {
        std::ifstream cf(out + "/model.q4nx", std::ios::binary);
        uint64_t hlen = 0;
        cf.read(reinterpret_cast<char*>(&hlen), 8);
        std::string hdr(size_t(hlen), '\0');
        cf.read(hdr.data(), std::streamsize(hlen));
        while (!hdr.empty() && (hdr.back() == ' ' || hdr.back() == '\0')) hdr.pop_back();
        const auto doc = json::parse(hdr);
        bool listed = false;
        for (const auto& e : doc.at("__metadata__").at("incomplete"))
            if (e.is_string() && e.get<std::string>().rfind("model.norm.weight", 0) == 0) listed = true;
        check(listed, "final norm: the incomplete container lists the refused final norm");
    }
    fs::remove_all(out);
    fs::remove_all(dir);
    std::printf("final norm: crafted tail envelope imported from tail+64, missing-epsilon control refused\n");
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
    // #844 imports the final norm, so it is no longer a refusal: the count is 0, and the tail envelope
    // assertion below is what proves it was found.
    check(tail == 0, "pack: the final norm is imported, not refused (got " + std::to_string(tail) + ")");
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
    bool head_in_tail = false;
    for (const auto& t : p.tensors) {
        if (t.name.find("layernorm") != std::string::npos || t.name == "model.norm.weight") ++norms;
        if (t.dtype == "q4_1" && t.refusal.empty() && t.name.compare(0, 9, "lm_head.w") == 0) {
            lm_tiles_planned = int((t.shape[0] / 32) * ((t.shape[1] + 255) / 256));
            head_in_tail = t.source == "fconst:tail";
        }
        if (t.dtype == "q4_1" && t.refusal.empty() && t.name.find("self_attn") != std::string::npos) ++attn_planned;
        if (t.dtype == "q4_1" && t.refusal.empty() && t.name.find(".mlp.") != std::string::npos) ++mlp_planned;
    }
    check(norms == layers * 2 + 1,
          "pack: 2 norms per layer plus the final norm are planned");
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

    // a fully representable pack imports without --allow-incomplete; a partial one writes nothing.
    // Per-process scratch dir: two concurrent runs of this binary (or two ctest shards) otherwise write
    // into the same directory and one dies with "cannot write .../model.q4nx" (measured 2026-11-03).
    const std::string out = (::onebit::test::tmp_root() /
                             ("oga_pack_import_test." + std::to_string(::getpid()))).string();
    fs::remove_all(out);
    const pi::Report r0 = pi::import_pack(pack, out, pi::Options{});
    check(refused.empty() ? (r0.refused.empty() && fs::exists(out + "/model.q4nx"))
                          : (!r0.refused.empty() && !fs::exists(out + "/model.q4nx")),
          "pack: without --allow-incomplete a container is written only when nothing is refused");
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
    // Only a TIED pack can be checked this way: there lm_head IS the embedding, so correlating them measures the decoder.
    // An untied pack's lm_head is a separate matrix and the number is meaningless by construction (Qwen3-8B: 0.0043 on a
    // head that decodes correctly -- std 0.026, plausible per-row scale, weakly correlated rows). Asserting 0.999
    // unconditionally demanded every pack be tied, which no untied pack can satisfy.
    if (r.selfcheck_tied)
        check(r.lmhead_corr > 0.999, "pack: the decode reproduces the fp16 embedding, tied (corr " +
                                         std::to_string(r.lmhead_corr) + ")");
    else
        check(true, "pack: untied, so the embedding correlation is not a decoder check (corr " +
                        std::to_string(r.lmhead_corr) + ")");
    check(r.ctrl_corr_nibble < 0.5 && r.ctrl_corr_nozp < 0.95,
          "pack: both wrong decoders are rejected as controls");
    // engine#1185: an imported container carries fp32 scale/zero (layout q4nx-2, npu/q4nx.h
    // q4_1f), so the read-back is exact -- the bf16 bound this used to assert (3.3e-3 relative,
    // ~43 % of groups) was the container's old format limit, not the import's.
    check(r.q4nx_rel == 0.0 && r.q4nx_max_abs == 0.0,
          "pack: container read back exactly, no bf16 parameter round trip (rel " +
              std::to_string(r.q4nx_rel) + ", max " + std::to_string(r.q4nx_max_abs) + ")");
    check(r.zero_exact_pct == 100.0 && r.attn_zero_inexact_pct == 0.0 && r.mlp_zero_inexact_pct == 0.0,
          "pack: every per-32-group scale/zero is the pack's own f32 value (" +
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
    // A head read from the .fconst tail keeps the pack's padding to whole 64-row tiles (Hunyuan: 120,818 rows are stored
    // as 120,832), so the container's count follows the stored rows; the plan's shape stays the vocabulary.
    const int64_t stored_rows = pi::head_stored_rows(head_rows, head_in_tail);
    check(head.shape.at(0) == (stored_rows / 32) * ((p.tensors.at(1).shape.at(1) + 255) / 256),
          "pack: lm_head tile count matches the container (" + std::to_string(head.shape.at(0)) + " tiles for " +
              std::to_string(stored_rows) + " stored rows)");
    check(m.tensor("model.layers.0.input_layernorm.weight").shape.at(0) == p.dims.hidden,
          "pack: a layer norm is `hidden` bf16 values, whatever the hidden size");
    check(m.tensor("model.norm.weight").shape.at(0) == int64_t(p.dims.hidden),
          "pack: the final norm is in the container");
    // The final norm comes from the Token_Token tail -- the program this importer reads and the device
    // runs. The Llm_Prefill duplicate is evidence about the PACK, not a gate on the import, so it is
    // reported rather than asserted: on this box the 0.5B pack's prefill .fconst (rewritten
    // 2026-11-03 15:10) disagrees with its Token copy in exactly one bf16 of this tensor, and three of
    // the four packs ship no prefill program at all. Refusing the import over a duplicated copy made a
    // complete pack unimportable, which is why this is a warning in the importer and a note here.
    if (r.stride && !pack.empty()) {
        bool has_prefill = false;
        if (fs::is_directory(pack))
            for (const auto& e : fs::directory_iterator(pack))
                if (e.path().filename().string().rfind("dd_metastate_Llm_Prefill_", 0) == 0) has_prefill = true;
        if (!has_prefill)
            std::printf("note: no Llm_Prefill .fconst in this pack, so the norm cross-check is skipped\n");
        else
            std::printf("note: the pack's two norm copies %s\n",
                        r.final_norm_crosscheck ? "agree" : "DISAGREE (reported; the Token copy is used)");
    }
    if (r.attn_projections) {
        const auto& qw = m.tensor("model.layers.0.self_attn.q_proj.weight");
        check(qw.shape.at(1) == int64_t(onebit::npu::kTileBytesF32),
              "pack: an imported projection is q4_1f chunks (fp32 scale/zero, engine#1185)");
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
        check(gw.shape.at(1) == int64_t(onebit::npu::kTileBytesF32) &&
                  dw.shape.at(1) == int64_t(onebit::npu::kTileBytesF32),
              "pack: the MLP projections are q4_1f chunks of the container's tile size");
        check(gw.shape.at(0) == int64_t((p.dims.intermediate / 32) * ((p.dims.hidden + 255) / 256)),
              "pack: gate_proj's tile count follows from its shape and the 256-column grid");
        check(dw.shape.at(0) == int64_t((p.dims.hidden / 32) * ((p.dims.intermediate + 255) / 256)),
              "pack: down_proj's tile count follows from its shape and the 256-column grid");
    }
    check(m.dims().hidden == p.dims.hidden && m.dims().layers == layers,
          "pack: config.json drives the loader");
    fs::remove_all(out);
}

// ── engine#1185: the imported container's parameters are the pack's, bit for bit ────────────────
// The two q4_1 forms side by side: the 5,120-byte one every GGUF-derived container holds (and the
// device streams), and the 6,144-byte fp32 one an OGA pack import writes. The bf16 form's loss is
// the issue's measured floor; the fp32 form is what removes it, and the chunk's own shape is what
// the decoder keys on, so a container with no layout metadata (all of them, before engine#1185)
// still reads as the bf16 form.
void test_q4_1f() {
    const pi::Q4 q = synthetic();
    const int RT = 2, KT = 2;  // 64x384 -> the 32x256 grid pads K to 512

    std::vector<uint8_t> f32_tiles;
    long long inexact = -1;
    pi::tile_q4_1_f32(q, f32_tiles, &inexact);
    check(f32_tiles.size() == size_t(RT) * KT * onebit::npu::kTileBytesF32,
          "q4_1f: 64x384 is 2 x 2 tiles of 6,144 B");
    check(inexact == 0, "q4_1f: every stored parameter is the source float, bit for bit (" +
                            std::to_string(inexact) + " differ)");

    // the loader keys the decoder off the tensor's own shape: 5,120 B is bf16, 6,144 B is fp32
    onebit::npu::Tensor f32t;
    f32t.shape = {int64_t(RT) * KT, int64_t(onebit::npu::kTileBytesF32)};
    f32t.data = f32_tiles.data();
    f32t.bytes = f32_tiles.size();
    f32t.avail = f32_tiles.size();
    check(onebit::npu::q4nx::chunk_of(f32t) == onebit::npu::q4nx::Chunk::Q4_1F,
          "q4_1f: chunk_of([.., 6144]) is Q4_1F");
    check(onebit::npu::q4nx::chunk_of(onebit::npu::q4nx::Params::F32) ==
              onebit::npu::q4nx::Chunk::Q4_1F &&
              onebit::npu::q4nx::chunk_of(onebit::npu::q4nx::Params::Bf16) ==
                  onebit::npu::q4nx::Chunk::Q4_1,
          "q4_1f: the parameter form names the chunk kind");

    std::vector<uint8_t> bf16_tiles;
    long long bf16_inexact = 0;
    pi::tile_q4_1(q, bf16_tiles, &bf16_inexact);

    // every group's scale and zero, read back out of both written tiles
    std::vector<float> ws(256), wz(256), bs(256), bz(256);
    bool params_ok = true, codes_ok = true;
    long long groups = 0, bf16_lost = 0;
    for (int i = 0; i < RT * KT; ++i) {
        const uint8_t* ft = f32_tiles.data() + size_t(i) * onebit::npu::kTileBytesF32;
        const uint8_t* bt = bf16_tiles.data() + size_t(i) * onebit::npu::kTileBytes;
        onebit::npu::q4nx::q4_1_params(onebit::npu::q4nx::Chunk::Q4_1F, ft, ws.data(), wz.data());
        onebit::npu::q4nx::q4_1_params(onebit::npu::q4nx::Chunk::Q4_1, bt, bs.data(), bz.data());
        if (std::memcmp(ft + 2048, bt + 1024, 4096) != 0) codes_ok = false;
        const int rt = i / KT, kt = i % KT;
        for (int r = 0; r < 32; ++r)
            for (int g = 0; g < 8; ++g) {
                if ((kt * 8 + g) * 32 >= q.K) continue;
                const size_t idx = size_t(rt * 32 + r) * (q.K / 32) + size_t(kt) * 8 + g;
                const float want_s = q.scale[idx], want_z = q.zero[idx];
                ++groups;
                if (std::memcmp(&ws[g * 32 + r], &want_s, 4) != 0 ||
                    std::memcmp(&wz[g * 32 + r], &want_z, 4) != 0)
                    params_ok = false;
                if (std::memcmp(&bs[g * 32 + r], &want_s, 4) != 0 ||
                    std::memcmp(&bz[g * 32 + r], &want_z, 4) != 0)
                    ++bf16_lost;
            }
    }
    check(params_ok, "q4_1f: the stored scale/zero are the source's, bit for bit (" +
                         std::to_string(groups) + " groups)");
    check(codes_ok, "q4_1f: the code region is byte-identical to the bf16 form's");
    check(bf16_lost > 0, "q4_1f control: the bf16 form loses " + std::to_string(bf16_lost) + " of " +
                             std::to_string(groups) + " groups on this data, so the check is live");

    // the device's 5,120-byte form still comes out of either chunk kind, one bf16 rounding away
    const std::vector<uint8_t> back = onebit::npu::q4nx::as_q4_1(f32t);
    check(back.size() == size_t(RT) * KT * onebit::npu::kTileBytes,
          "q4_1f: as_q4_1 repacks it to the device's 5,120-byte q4_1 tiles");
    std::vector<float> s32(256), z32(256), sb(256), zb(256);
    bool repack_ok = true;
    for (int i = 0; i < RT * KT; ++i) {
        const uint8_t* ft = f32_tiles.data() + size_t(i) * 6144;
        const uint8_t* bt = back.data() + size_t(i) * 5120;
        if (std::memcmp(ft + 2048, bt + 1024, 4096) != 0) repack_ok = false;  // codes unchanged
        onebit::npu::q4nx::q4_1_params(onebit::npu::q4nx::Chunk::Q4_1F, ft, s32.data(), z32.data());
        onebit::npu::q4nx::q4_1_params(onebit::npu::q4nx::Chunk::Q4_1, bt, sb.data(), zb.data());
        for (int j = 0; j < 256; ++j) {
            const float want_s = onebit::npu::bf16_to_f32(onebit::npu::f32_to_bf16(s32[j]));
            const float want_z = onebit::npu::bf16_to_f32(onebit::npu::f32_to_bf16(z32[j]));
            if (std::memcmp(&sb[j], &want_s, 4) != 0 || std::memcmp(&zb[j], &want_z, 4) != 0)
                repack_ok = false;
        }
    }
    check(repack_ok, "q4_1f: the repack keeps the codes and rounds the parameters to bf16, nothing else");

    // the container-level form: absent and q4nx-1 are the bf16 form every container on disk has
    using onebit::npu::q4nx::Params;
    check(onebit::npu::q4nx::params_of_layout("") == Params::Bf16,
          "layout: an absent layout is the bf16 form");
    check(onebit::npu::q4nx::params_of_layout("q4nx-1") == Params::Bf16, "layout: q4nx-1 is bf16");
    check(onebit::npu::q4nx::params_of_layout("q4nx-2") == Params::F32, "layout: q4nx-2 is fp32");
    check(std::string(onebit::npu::q4nx::layout_of(Params::F32)) == "q4nx-2",
          "layout: the fp32 form writes q4nx-2");
    bool threw = false;
    try {
        onebit::npu::q4nx::params_of_layout("q4nx-99");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "layout: an unknown version is refused rather than read as the bf16 form");
    std::printf("q4_1f: %lld groups bit-equal; the bf16 form loses %lld of them\n", groups, bf16_lost);
}

// ── engine#1185: an imported pack's scale/zero survive the container bit for bit ────────────────
// The acceptance is bit equality, not a percentage. A real OGA pack is a ~6 GB download and the
// shipped legs must stay device-free, so this builds the smallest pack whose .fconst has the 0.5B
// pack's MEASURED second-header offset (1,115,968 B) and real quantized tiles in every section --
// attention, the fused MLP and down_proj -- plus a MatMulNBits head with f32 scales. It imports the
// pack and reads the container back through the engine's own loader: every per-group scale and zero
// must equal the pack's, and every code must stay at its q4_1 index. The control is the issue's own
// number: the 5,120-byte bf16 form loses these groups, which is why the fp32 form exists.
void test_import_fp32_params(const std::string& root) {
    const fs::path dir = fs::path(root) / "fp32-params";
    fs::remove_all(dir);
    fs::create_directories(dir / "cache");
    const int H = 896, L = 3, heads = 14, kv_heads = 2, hd = 64, inter = 256, V = 64;
    const int n_kv = (kv_heads * hd) / 64, n_q = (heads * hd) / 64;
    const int64_t tile = pi::attn_tile_bytes(H);              // 34,816 B, the 64-row tile
    const int64_t attn_off = pi::attn_payload_off(H);         // 1,856 B
    const int64_t second = attn_off + int64_t(2 * n_kv + 2 * n_q) * tile;
    const int64_t gu_unit = pi::group_bytes(H, 128), down_group = pi::group_bytes(inter, 64);
    const int64_t gu_units = inter / 64, down_units = H / 64;
    const int64_t mlp_off = second + 64 + 2 * H;
    const int64_t stride = mlp_off + gu_units * gu_unit + 64 + down_units * down_group;
    check(second == 1115968, "fp32 import: the crafted .fconst has the 0.5B pack's second header (" +
                                 std::to_string(second) + ")");

    // Scales are bf16-storable f32 values and codes/zero points are in 0..7, so no tile byte pair
    // can look like the 86 35 + ten-zero epsilon header: the locator must see exactly the headers
    // this test writes, nothing else.
    auto scale_of = [](int i) {
        static const float v[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
        return v[i & 7];
    };
    auto make_tile = [&](int rows, int k, int seed) {
        pi::TileW t;
        t.k = k;
        t.rows = rows;
        t.code.assign(size_t(rows) * size_t(k), 0);
        t.scale.assign(size_t(rows) * size_t(k / 128), 0.0f);
        t.zp.assign(size_t(rows) * size_t(k / 128), 0);
        t.bias.assign(rows, 0);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < k; ++c) t.code[size_t(r) * k + c] = uint8_t((r * 3 + c * 5 + seed) % 8);
            for (int b = 0; b < k / 128; ++b) {
                t.scale[size_t(r) * (k / 128) + b] = scale_of(r + b + seed);
                t.zp[size_t(r) * (k / 128) + b] = uint8_t((r + b + seed) % 8);
            }
        }
        return pi::encode_tile(t);
    };
    std::string fc(size_t(int64_t(L) * stride + 64 + H * 2), '\0');
    auto put16 = [&](int64_t off, uint16_t v) {
        fc[size_t(off)] = char(v & 0xFF);
        fc[size_t(off) + 1] = char(v >> 8);
    };
    auto put_bytes = [&](int64_t off, const std::vector<uint8_t>& b) {
        std::memcpy(&fc[size_t(off)], b.data(), b.size());
    };
    std::vector<pi::TileW> q_proj0, down0;
    for (int l = 0; l < L; ++l) {
        const int64_t base = int64_t(l) * stride;
        put16(base, 0x3586);
        put16(base + second, 0x3586);
        for (int i = 0; i < H; ++i) {
            put16(base + 64 + 2 * i, pi::bf16_bits_of(float(i + 1)));
            put16(base + second + 64 + 2 * i, pi::bf16_bits_of(float(i + 1)));
        }
        // k, q, v, o share the one payload at K = hidden on a 2-header block
        for (int i = 0; i < 2 * n_kv + 2 * n_q; ++i) {
            const std::vector<uint8_t> b = make_tile(64, H, 40 + i);
            put_bytes(base + attn_off + int64_t(i) * tile, b);
            if (l == 0 && i >= n_kv && i < n_kv + n_q) q_proj0.push_back(pi::decode_tile(b.data(), H));
        }
        for (int u = 0; u < int(gu_units); ++u)
            put_bytes(base + mlp_off + int64_t(u) * gu_unit, make_tile(128, H, 200 + u));
        for (int u = 0; u < int(down_units); ++u) {
            const std::vector<uint8_t> b = make_tile(64, inter, 300 + u);
            put_bytes(base + mlp_off + gu_units * gu_unit + 64 + int64_t(u) * down_group, b);
            if (l == 0) down0.push_back(pi::decode_tile(b.data(), inter));
        }
    }
    const int64_t tail = int64_t(L) * stride;
    put16(tail, 0x3586);
    for (int i = 0; i < H; ++i) put16(tail + 64 + 2 * i, pi::bf16_bits_of(float(i + 1)));
    put_file((dir / "dd_metastate_Llm_Token_Token_crafted.fconst").string(), fc);

    // The head the importer reads from model.onnx: MatMulNBits, block 128, f32 scales (the other
    // half of what the container has to keep; these are deliberately not bf16-representable).
    const int block = 128, kblocks = H / block;
    std::vector<uint8_t> qw(size_t(V) * size_t(kblocks) * size_t(block / 2), 0);
    std::vector<float> scales(size_t(V) * size_t(kblocks), 0.0f);
    std::vector<uint8_t> qz(size_t(V) * size_t((kblocks + 1) / 2), 0);
    for (int n = 0; n < V; ++n)
        for (int b = 0; b < kblocks; ++b) {
            scales[size_t(n) * kblocks + b] = 0.017f * float(b + 1) + 0.0003f * float(n);
            const uint8_t z = uint8_t((n + 2 * b) % 8);
            qz[size_t(n) * ((kblocks + 1) / 2) + b / 2] |= uint8_t((b & 1) ? (z << 4) : (z & 0x0F));
            for (int i = 0; i < block / 2; ++i) {
                const int k = b * block + 2 * i;
                const uint8_t lo = uint8_t((n * 3 + k * 5 + 1) % 8);
                const uint8_t hi = uint8_t((n * 5 + k * 3 + 2) % 8);
                qw[(size_t(n) * kblocks + b) * (block / 2) + i] = uint8_t(lo | (hi << 4));
            }
        }
    std::string data;
    auto append = [&](const void* p, size_t n) {
        data.append(static_cast<const char*>(p), n);
        return int64_t(data.size() - n);
    };
    std::vector<float> embed(size_t(V) * size_t(H));
    for (size_t i = 0; i < embed.size(); ++i) embed[i] = 0.01f * float(i % 97) - 0.5f;
    const int64_t embed_off = append(embed.data(), embed.size() * sizeof(float));
    const int64_t qw_off = append(qw.data(), qw.size());
    const int64_t sc_off = append(scales.data(), scales.size() * sizeof(float));
    const int64_t qz_off = append(qz.data(), qz.size());
    put_file((dir / "model.onnx.data").string(), data);
    put_file((dir / "config.json").string(),
             R"({"model_type":"qwen2","hidden_size":896,"intermediate_size":256,"num_hidden_layers":3,)"
             R"("num_attention_heads":14,"num_key_value_heads":2,"head_dim":64,"vocab_size":64,)"
             R"("rope_theta":10000.0,"tie_word_embeddings":false,"rms_norm_eps":1e-06})");
    put_file((dir / "model.onnx").string(),
             pb::model({pb::tensor("model.embed_tokens.weight", {V, H}, "model.onnx.data",
                                   uint64_t(embed_off), uint64_t(V) * H * 4),
                        pb::tensor("lm_head_qweight", {}, "model.onnx.data", uint64_t(qw_off),
                                   uint64_t(qw.size())),
                        pb::tensor("lm_head_scales", {}, "model.onnx.data", uint64_t(sc_off),
                                   uint64_t(scales.size()) * 4),
                        pb::tensor("lm_head_zeros", {}, "model.onnx.data", uint64_t(qz_off),
                                   uint64_t(qz.size()))},
                       {pb::matmulnbits("lm_head", {"x", "lm_head_qweight", "lm_head_scales", "lm_head_zeros"},
                                        H, V, 4, block)}));

    const std::string out = (::onebit::test::tmp_root() /
                             ("oga_pack_import_fp32." + std::to_string(::getpid()))).string();
    fs::remove_all(out);
    const pi::Report r = pi::import_pack(dir.string(), out, pi::Options{});
    for (const auto& s : r.refused) std::fprintf(stderr, "  fp32 import refusal: %s\n", s.c_str());
    check(r.refused.empty(), "fp32 import: the crafted pack is fully representable");
    check(r.q4nx_rel == 0.0 && r.q4nx_max_abs == 0.0,
          "fp32 import: the container's weights are the pack's exactly (rel " + std::to_string(r.q4nx_rel) +
              ", max " + std::to_string(r.q4nx_max_abs) + ")");
    check(r.zero_exact_pct == 100.0 && r.attn_zero_inexact_pct == 0.0 && r.mlp_zero_inexact_pct == 0.0,
          "fp32 import: no stored parameter is a bf16 rounding of the pack's (" +
              std::to_string(r.zero_exact_pct) + " %)");
    check(r.attn_tiles_total == int64_t(L) * (2 * n_kv + 2 * n_q) && r.attn_tiles_ok == r.attn_tiles_total,
          "fp32 import: every .fconst attention tile survives decode->encode (" +
              std::to_string(r.attn_tiles_ok) + "/" + std::to_string(r.attn_tiles_total) + ")");
    check(r.mlp_groups_total == int64_t(L) * (gu_units + down_units) && r.mlp_groups_ok == r.mlp_groups_total,
          "fp32 import: every MLP group survives decode->encode (" + std::to_string(r.mlp_groups_ok) + "/" +
              std::to_string(r.mlp_groups_total) + ")");
    if (!r.refused.empty()) {
        fs::remove_all(dir);
        return;
    }

    {   // the container states its parameter form in the manifest and in config.json
        std::ifstream f(out + "/model.q4nx", std::ios::binary);
        uint64_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 8);
        std::string header(size_t(hlen), '\0');
        f.read(header.data(), std::streamsize(hlen));
        while (!header.empty() && (header.back() == ' ' || header.back() == '\0')) header.pop_back();
        const auto doc = json::parse(header);
        const std::string layout = doc.at("__metadata__").at("layout").get<std::string>();
        check(layout == "q4nx-2", "fp32 import: the manifest says layout q4nx-2 (got \"" + layout + "\")");
        check(onebit::npu::q4nx::params_of_layout(layout) == onebit::npu::q4nx::Params::F32,
              "fp32 import: that layout names the fp32 parameter form");
    }
    {
        std::ifstream cf(out + "/config.json");
        const auto c = json::parse(cf);
        check(c.at("q4nx_source").at("layout") == "q4nx-2", "fp32 import: config.json agrees");
    }

    const onebit::npu::Model m(out);
    const auto& head = m.tensor("lm_head.weight");
    check(head.shape.size() == 2 && head.shape.at(1) == int64_t(onebit::npu::kTileBytesF32),
          "fp32 import: I8 tensors are the 6,144-byte q4_1f chunks");
    check(head.tiles() == int(head.shape.at(0)),
          "fp32 import: the loader counts one tile per 6,144-byte row");

    long long groups = 0, bf16_lost = 0;
    auto verify = [&](const std::string& name, const pi::Q4& q) {
        const auto& t = m.tensor(name);
        const int K = q.K, KT = (K + 255) / 256;
        check(t.shape.size() == 2 && t.shape.at(1) == int64_t(onebit::npu::kTileBytesF32),
              name + ": is a q4_1f tensor (" + std::to_string(t.shape.at(1)) + " B rows)");
        check(t.shape.at(0) == int64_t((q.N / 32) * KT), name + ": tile count follows from the 32x256 grid");
        std::vector<float> sc(256), zr(256);
        bool params_ok = true, codes_ok = true;
        for (int i = 0; i < int(t.shape.at(0)); ++i) {
            const int rt = i / KT, kt = i % KT;
            const uint8_t* tl = t.data + size_t(i) * onebit::npu::kTileBytesF32;
            onebit::npu::q4nx::q4_1_params(onebit::npu::q4nx::Chunk::Q4_1F, tl, sc.data(), zr.data());
            for (int r = 0; r < 32; ++r)
                for (int g = 0; g < 8; ++g) {
                    const int k = kt * 256 + g * 32;
                    if (k >= K) continue;
                    const size_t idx = size_t(rt * 32 + r) * (K / 32) + size_t(kt * 8 + g);
                    const float want_s = q.scale[idx], want_z = q.zero[idx];
                    ++groups;
                    if (std::memcmp(&sc[g * 32 + r], &want_s, 4) != 0 ||
                        std::memcmp(&zr[g * 32 + r], &want_z, 4) != 0)
                        params_ok = false;
                    const float bs = onebit::npu::bf16_to_f32(onebit::npu::f32_to_bf16(want_s));
                    const float bz = onebit::npu::bf16_to_f32(onebit::npu::f32_to_bf16(want_z));
                    if (std::memcmp(&bs, &want_s, 4) != 0 || std::memcmp(&bz, &want_z, 4) != 0) ++bf16_lost;
                }
            for (int h = 0; h < 2; ++h)
                for (int g = 0; g < 8; ++g)
                    for (int j = 0; j < 32; ++j) {
                        const int k = kt * 256 + g * 32 + j;
                        if (k >= K) continue;
                        for (int r2 = 0; r2 < 8; ++r2) {
                            const uint8_t b = tl[2048 + size_t(h) * 2048 + size_t(g) * 256 + size_t(j) * 8 +
                                                size_t(r2)];
                            const size_t row = size_t(rt * 32 + h * 16 + r2 * 2);
                            if ((b & 0x0F) != (q.codes[row * K + k] & 0x0F) ||
                                (b >> 4) != (q.codes[(row + 1) * K + k] & 0x0F))
                                codes_ok = false;
                        }
                    }
        }
        check(params_ok, name + ": every per-group scale and zero is the pack's, bit for bit");
        check(codes_ok, name + ": every code is at its q4_1 index, unchanged");
    };
    verify("model.layers.0.self_attn.q_proj.weight", pi::q_from_tiles(q_proj0));
    verify("model.layers.0.mlp.down_proj.weight", pi::q_from_tiles(down0));
    verify("lm_head.weight", pi::decode_matmulnbits(qw.data(), scales.data(), qz.data(), V, H, block));
    check(groups > 0 && bf16_lost > 0,
          "fp32 import control: the bf16 form would have lost " + std::to_string(bf16_lost) + " of " +
              std::to_string(groups) + " groups on this data, so the check is live");

    // ── the serve boundary (engine#1185): the same container in the device's form ────────────────
    // `1bit serve -m <imported dir> --device npu` loads an imported container through the per-op
    // forward, whose loader streams 5,120-byte q4_1 chunks and refuses a q4nx-2 container rather
    // than misread it (npu/forward/src/model.c). Q4nxNpuForward::init_impl converts first through
    // npu/q4nx_device.cpp; this is that conversion, and the proof it changes only the parameter
    // width: the device tiles are byte-identical to the old bf16 container's for the same weights.
    const std::string cache = (::onebit::test::tmp_root() /
                               ("oga_q4nx_device_form." + std::to_string(::getpid()))).string();
    fs::remove_all(cache);
    {
        const auto form = onebit::npu::q4nx::device_form(out + "/model.q4nx", cache);
        check(form.converted && !form.reused && form.tensors > 0 && form.tiles > 0,
              "device form: the imported container converts (" + std::to_string(form.tensors) +
                  " tensors, " + std::to_string(form.tiles) + " tiles)");
        check(form.model != out + "/model.q4nx" && fs::exists(form.model),
              "device form: a cache directory, not the imported container in place");
        check(fs::exists(out + "/model.q4nx") &&
                  q4_1f_tensors(container_table(out + "/model.q4nx")) > 0,
              "device form: the imported container on disk is unchanged and still q4nx-2");
        // (c) the forward loader's refusal trigger is absent from what it is handed ...
        const json converted_table = container_table(form.model);
        check(q4_1f_tensors(converted_table) == 0,
              "device form: the converted container holds no q4_1f chunk, so the loader's refusal "
              "cannot fire on it");
        check(converted_table.at("__metadata__").at("layout") == "q4nx-1",
              "device form: the converted container says it holds the bf16 q4_1 form");
        // ... and the safety net still sees the trigger on the file it would otherwise load.
        check(q4_1f_tensors(container_table(out + "/model.q4nx")) > 0,
              "device form control: the unconverted container still trips the loader's refusal");

        const onebit::npu::Model conv(form.dir);
        check(conv.tensor("lm_head.weight").shape.at(1) == int64_t(onebit::npu::kTileBytes),
              "device form: the converted I8 tensors are 5,120-byte q4_1 chunks");
        auto same_tiles = [&](const std::string& name, const pi::Q4& q) {
            std::vector<uint8_t> old_form;
            pi::tile_q4_1(q, old_form);
            const auto& t = conv.tensor(name);
            check(t.bytes == old_form.size() && std::memcmp(t.data, old_form.data(), old_form.size()) == 0,
                  name + ": the converted device tiles are byte-identical to the old bf16 container's (" +
                      std::to_string(t.bytes) + " B)");
        };
        same_tiles("model.layers.0.self_attn.q_proj.weight", pi::q_from_tiles(q_proj0));
        same_tiles("model.layers.0.mlp.down_proj.weight", pi::q_from_tiles(down0));
        same_tiles("lm_head.weight", pi::decode_matmulnbits(qw.data(), scales.data(), qz.data(), V, H, block));

        // the fast lane's boundary (npu/pack.cpp) does the same repack in memory
        std::vector<uint8_t> lane;
        const uint8_t* lane_tiles = onebit::npu::device_tiles(m.tensor("model.layers.0.self_attn.q_proj.weight"),
                                                              lane);
        std::vector<uint8_t> want_lane;
        pi::tile_q4_1(pi::q_from_tiles(q_proj0), want_lane);
        check(lane_tiles == lane.data() && lane == want_lane,
              "lane: device_tiles repacks a q4_1f projection to the old bf16 form's tiles, bit for bit");

        const auto again = onebit::npu::q4nx::device_form(out + "/model.q4nx", cache);
        check(again.converted && again.reused && again.model == form.model,
              "device form: a second load reuses the cached conversion");
        const auto already = onebit::npu::q4nx::device_form(form.model, cache);
        check(!already.converted && already.dir == form.dir,
              "device form: a container already in the device form is returned unchanged");

        // engine#2263: the conversion is a WRITER of the cache, so it sweeps it like serve start does. An older entry
        // (2 GiB, sparse: no real disk) sits in the cache beyond the age window; a new conversion under a 1 GiB bound
        // removes it, never the entry it just wrote, and with the bound at 0 (no sweep) leaves it alone.
        {
            const char* had = std::getenv("ONEBIT_Q4NX_CACHE_MAX_GIB");
            const std::string saved = had ? had : "";
            const fs::path older = fs::path(cache) / "older-q4nx1-0000000000000000";
            fs::create_directories(older);
            { std::ofstream(older / "model.q4nx", std::ios::binary); }
            fs::resize_file(older / "model.q4nx", uintmax_t(2) << 30);
            fs::last_write_time(older, fs::file_time_type::clock::now() - std::chrono::hours(2));

            fs::remove_all(form.dir);
            ::setenv("ONEBIT_Q4NX_CACHE_MAX_GIB", "0", 1);
            const auto off = onebit::npu::q4nx::device_form(out + "/model.q4nx", cache);
            check(off.converted && !off.reused && fs::exists(older / "model.q4nx"),
                  "cache sweep control: with ONEBIT_Q4NX_CACHE_MAX_GIB=0 a new conversion removes nothing");

            fs::remove_all(off.dir);
            ::setenv("ONEBIT_Q4NX_CACHE_MAX_GIB", "1", 1);
            const auto on = onebit::npu::q4nx::device_form(out + "/model.q4nx", cache);
            check(on.converted && !on.reused && fs::exists(on.model),
                  "cache sweep: the new conversion is written and kept");
            check(!fs::exists(older),
                  "cache sweep: a new conversion over a 1 GiB bound removes the older 2 GiB entry (engine#2263)");

            if (had) ::setenv("ONEBIT_Q4NX_CACHE_MAX_GIB", saved.c_str(), 1);
            else ::unsetenv("ONEBIT_Q4NX_CACHE_MAX_GIB");
            fs::remove_all(older);
        }
        std::printf("device form: %lld I8 tensor(s), %lld tile(s) repacked; %d q4_1f trigger(s) on "
                    "the imported container, 0 on the converted one, and the device tiles are the old "
                    "bf16 form's byte for byte\n",
                    form.tensors, form.tiles, q4_1f_tensors(container_table(out + "/model.q4nx")));
    }
    fs::remove_all(cache);
    fs::remove_all(out);
    fs::remove_all(dir);
    std::printf("fp32 import: %lld groups bit-equal to the pack; the bf16 form loses %lld\n", groups, bf16_lost);
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

// engine#1648: the import's PACK leg never ran anywhere, so the shape a hybrid pack actually has -- geometry under
// `text_config`, none at the top level -- was exercised by no test end to end. `test_text_config_geometry` above checks
// the reader; this runs import_pack() on a crafted pack of that shape and checks what lands in the container, which is
// what a pack actually goes through. The crafted .fconst is the one the final-norm test uses (8 blocks, eps headers at
// both boundaries), so the only thing that varies here is where the geometry is declared.
void test_hybrid_pack_import(const std::string& root) {
    const fs::path dir = fs::path(root) / "hybrid-import";
    fs::remove_all(dir);
    fs::create_directories(dir / "cache");
    const int H = 128, L = 8;
    const int64_t stride = 1 << 20, second = 262144;
    std::string fc(size_t(L * stride) + size_t(64) + size_t(H * 2), '\0');
    auto put_eps = [&](int64_t off) {
        fc[size_t(off)] = char(0x86);
        fc[size_t(off) + 1] = char(0x35);
    };
    for (int l = 0; l < L; ++l) {
        put_eps(int64_t(l) * stride);
        put_eps(int64_t(l) * stride + second);
    }
    put_eps(int64_t(L) * stride);
    for (int i = 0; i < H; ++i) {
        const uint16_t bits = pi::bf16_bits_of(float(i + 1));
        fc[size_t(int64_t(L) * stride + 64 + 2 * i)] = char(bits & 0xFF);
        fc[size_t(int64_t(L) * stride + 64 + 2 * i) + 1] = char(bits >> 8);
    }
    put_file((dir / "config.json").string(),
             R"({"architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl",)"
             R"("tie_word_embeddings":true,"text_config":{"model_type":"qwen3_vl_text",)"
             R"("hidden_size":128,"intermediate_size":256,"num_hidden_layers":8,"num_attention_heads":2,)"
             R"("num_key_value_heads":1,"head_dim":64,"vocab_size":128,"rope_theta":10000.0,)"
             R"("rms_norm_eps":1e-06},"vision_config":{"hidden_size":1024,"depth":24}})");
    put_file((dir / "model.onnx.data").string(), std::string(size_t(H) * size_t(H) * 4, '\x11'));
    put_file((dir / "model.onnx").string(),
             pb::model({pb::tensor("model.embed_tokens.weight", {H, H}, "model.onnx.data", 0,
                                   int64_t(H) * H * 4)}, {}));
    put_file((dir / "dd_metastate_Llm_Token_Token_crafted.fconst").string(), fc);

    const std::string out = (::onebit::test::tmp_root() /
                             ("oga_pack_import_hybrid." + std::to_string(::getpid()))).string();
    fs::remove_all(out);
    pi::Options allow;
    allow.allow_incomplete = true;
    const pi::Report r = pi::import_pack(dir.string(), out, allow);
    check(r.final_norm_imported,
          "hybrid import: a pack whose geometry is only in text_config imports, tail envelope included");
    bool hidden_ok = false;
    std::ifstream cf(out + "/config.json");
    if (cf) {
        nlohmann::json c;
        cf >> c;
        hidden_ok = c.value("hidden_size", 0) == H;
    }
    check(hidden_ok, "hybrid import: out/config.json states the text tower's hidden_size, so a serve reads the same "
                     "geometry the import used");
    std::printf("hybrid import: geometry only in text_config -> imported%s, container hidden_size %s\n",
                r.final_norm_imported ? " (tail envelope included)" : "", hidden_ok ? "from text_config" : "WRONG");
    fs::remove_all(out);
    fs::remove_all(dir);
}

// The embedding component (engine #1503): a hybrid pack keeps its table in `embedding.onnx` (+ `.data`), gathered by
// GatherBlockQuantized. Shape measured on amd/Qwen3-VL-4B-Instruct_rai_1.9.0_npu: weight_Q4 [V, H/2] UINT8 and
// weight_scales [V, groups] FLOAT with group 128. Driving the reader against known codes and scales pins the
// dequantisation without a 3 GB pack.

// A hybrid pack keeps its text tower under `text_config` INSIDE config.json, with model_config.json carrying
// model_attributes: null (measured on amd/Qwen3-VL-4B-Instruct_rai_1.9.0_npu), so the geometry has to come from the
// inline block. The text tower is the decoder this importer builds, so its keys fill gaps and never override.
void test_text_config_geometry(const std::string& root) {
    const fs::path dir = fs::path(root) / "textcfg";
    fs::remove_all(dir);
    fs::create_directories(dir);
    put_file((dir / "config.json").string(),
             R"({"architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl",)"
             R"("image_token_id":151655,"tie_word_embeddings":true,)"
             R"("text_config":{"model_type":"qwen3_vl_text","hidden_size":2560,"num_hidden_layers":36,)"
             R"("num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"intermediate_size":9728,)"
             R"("vocab_size":151936,"rope_theta":5000000.0,"rms_norm_eps":1e-06},)"
             R"("vision_config":{"hidden_size":1024,"depth":24,"in_channels":3}})");
    std::string src;
    const nlohmann::json g = pi::load_geometry_exported(dir.string(), &src);
    const pi::Dims d = pi::read_dims_exported(g, src);
    check(d.hidden == 2560 && d.layers == 36 && d.heads == 32 && d.intermediate == 9728 && d.vocab == 151936,
          "text_config: the text tower's geometry is read from one level down (hidden " + std::to_string(d.hidden) + ")");
    check(d.head_dim == 128 && d.kv_heads == 8, "text_config: head_dim and num_key_value_heads come with it");
    check(src.find("text_config") != std::string::npos, "text_config: the source names where the geometry came from (" + src + ")");
    check(!g.contains("vision_config") || g["vision_config"]["hidden_size"] == 1024, "text_config: vision_config untouched");
    put_file((dir / "config.json").string(),
             R"({"model_type":"qwen3_vl","hidden_size":4096,"text_config":{"hidden_size":2560,)"
             R"("num_hidden_layers":36,"num_attention_heads":32,"intermediate_size":9728,"vocab_size":151936}})");
    const nlohmann::json g2 = pi::load_geometry_exported(dir.string(), &src);
    check(pi::read_dims_exported(g2, src).hidden == 4096, "text_config: a top-level hidden_size is NOT overridden");
    std::printf("  text_config: hidden %d, %dL, %d heads, intermediate %d, vocab %d\n", d.hidden, d.layers, d.heads,
                d.intermediate, d.vocab);
    fs::remove_all(dir);
}


// engine#1503: a THREE-input MatMulNBits with no qzeros, weights spelled `<proj>.MatMul.weight_Q4` +
// `.weight_scales` -- Qwen3-VL-4B's form. `walk_graph` used to require four inputs, so it recorded NO matmuls for that
// pack: `info` printed matmulnbits_ops 0 and the import wrote the embedding only ("imported 1 tensors, 0 q4_1 tiles").
// On the real pack the fix yields 147 tensors / 47,480 tiles and the pack's own self-check reads corr 0.993981 against its
// embedding, with the no-zero-point control at 0.3333 -- this arm pins the mechanism without the 3 GB pack.
// engine#1503: `rope_theta` nested one level deeper still — `text_config.rope_parameters.rope_theta` — which is how
// Qwen3-VL-4B declares 5,000,000 with no `rope_theta` anywhere else. The reader's 10000 default made the importer warn that
// the pack's cos/sin caches "no plain theta explains"; measured against the pack's own cos_cache (262144 x 64), theta 5e6
// reproduces row 1 to max|d| 0.000000 while 10000 is off by 0.088. This pins the lift and the never-override rule.
void test_rope_parameters_theta(const std::string& root) {
    const fs::path dir = fs::path(root) / "ropeparams";
    fs::remove_all(dir);
    fs::create_directories(dir);
    put_file((dir / "config.json").string(),
             R"({"model_type":"qwen3_vl","text_config":{"hidden_size":64,"num_hidden_layers":1,)"
             R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"intermediate_size":128,)"
             R"("vocab_size":32,"rope_parameters":{"mrope_interleaved":true,"mrope_section":[24,20,20],)"
             R"("rope_theta":5000000,"rope_type":"default"}}})");
    std::string src;
    const nlohmann::json g = pi::load_geometry_exported(dir.string(), &src);
    const pi::Dims d = pi::read_dims_exported(g, src);
    check(d.rope_theta == 5000000.0f,
          "rope_parameters: rope_theta comes from the nested block (" + std::to_string(d.rope_theta) + ")");
    check(src.find("rope_parameters") != std::string::npos, "rope_parameters: and the source says so (" + src + ")");
    // a top-level rope_theta still wins
    put_file((dir / "config.json").string(),
             R"({"model_type":"qwen3_vl","rope_theta":10000.0,"text_config":{"hidden_size":64,)"
             R"("num_hidden_layers":1,"num_attention_heads":2,"head_dim":32,"intermediate_size":128,"vocab_size":32,)"
             R"("rope_parameters":{"rope_theta":5000000}}})");
    const nlohmann::json g2 = pi::load_geometry_exported(dir.string(), &src);
    check(pi::read_dims_exported(g2, src).rope_theta == 10000.0f,
          "rope_parameters: a top-level rope_theta is NOT overridden by the nested one");
    std::printf("  rope_parameters: theta %.0f nested, top-level still wins\n", (double)d.rope_theta);
    fs::remove_all(dir);
}

void test_three_input_matmulnbits(const std::string& root) {
    const fs::path dir = fs::path(root) / "mm3";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const int64_t H = 64, N = 32, block = 32, groups = H / block;
    std::vector<uint8_t> qw(size_t(N) * size_t(groups) * size_t(block / 2), 0x21);   // codes 1/2 alternating
    std::vector<float> sc(size_t(N) * size_t(groups), 0.01f);
    std::string data(reinterpret_cast<const char*>(qw.data()), qw.size());
    data.append(reinterpret_cast<const char*>(sc.data()), sc.size() * sizeof(float));
    put_file((dir / "model.onnx.data").string(), data);
    const auto q4 = pb::tensor("model.layers.0.attn.q_proj.MatMul.weight_Q4", {N, groups, block / 2},
                               "model.onnx.data", 0, qw.size(), 2);
    const auto scl = pb::tensor("model.layers.0.attn.q_proj.MatMul.weight_scales", {N, groups},
                                "model.onnx.data", qw.size(), sc.size() * sizeof(float), 1);
    const auto node = pb::matmulnbits("q", {"x", "model.layers.0.attn.q_proj.MatMul.weight_Q4",
                                            "model.layers.0.attn.q_proj.MatMul.weight_scales"}, H, N, 4, block);
    put_file((dir / "model.onnx").string(), pb::model({q4, scl}, {node}));
    int mm = -1, noz = -1, K = 0, Nn = 0, blk = 0;
    pi::parse_onnx_matmul_census((dir / "model.onnx").string(), &mm, &noz, &K, &Nn, &blk);
    check(mm == 1, "3-input MatMulNBits: the walker records the node (" + std::to_string(mm) + " of 1)");
    check(noz == 1, "3-input MatMulNBits: it is counted as carrying no qzeros, so the implicit zero point applies");
    check(K == H && Nn == N && blk == block,
          "3-input MatMulNBits: K/N/block come from the node's own attributes (" + std::to_string(K) + "/" +
              std::to_string(Nn) + "/" + std::to_string(blk) + ")");
    std::printf("  3-input MatMulNBits: %d matmul(s), %d without qzeros, K %d N %d block %d\n", mm, noz, K, Nn, blk);
    fs::remove_all(dir);
}

// engine#1841: genre A (projections are graph initializers, no .fconst) decodes each layer projection by name, and a
// lookup that missed used to return quietly -- the container then reported itself complete with no layer projections
// (the #1790 CHANGES: `...o_projMatMul.weight_Q4`, a missing dot). The dot is normalised on main; this pins the other
// half: a projection the decoder cannot read is a NAMED refusal, not a skip. plan() refuses a projection whose codes are
// absent, but it counts one present on `<proj>.MatMul.weight_Q4` (or any `.MatMulNBits.` tensor) alone, while the
// decoder needs the scales beside it -- so a pack missing only the scales passed plan() and lost the projection without a
// word. One crafted 1-layer pack in Qwen3-VL-4B's three-input spelling, imported whole (every projection lands, nothing
// refused), then with down_proj's scales misnamed (refused by name; nothing written without --allow-incomplete).
void test_genre_a_projection_gap(const std::string& root) {
    const fs::path dir = fs::path(root) / "genre-a-gap";
    const int64_t H = 64, I = 64, V = 64, hd = 32, heads = 2, kv = 1, block = 32, groups = H / block;
    auto build = [&](bool spell_down_wrong) {
        fs::remove_all(dir);
        fs::create_directories(dir);
        std::string data;
        std::vector<std::string> inits, nodes;
        auto append = [&](const void* q, size_t n) {
            data.append(static_cast<const char*>(q), n);
            return uint64_t(data.size() - n);
        };
        std::vector<float> embed(size_t(V * H));
        for (size_t i = 0; i < embed.size(); ++i) embed[i] = 0.01f * float(i % 89) - 0.4f;
        const uint64_t eo = append(embed.data(), embed.size() * 4);
        inits.push_back(pb::tensor("model.embed_tokens.weight", {V, H}, "model.onnx.data", eo, uint64_t(V * H) * 4));
        std::vector<uint16_t> ones(size_t(H), 0x3C00);   // f16 1.0
        for (const char* nm : {"model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight",
                               "model.norm.weight"}) {
            const uint64_t o = append(ones.data(), ones.size() * 2);
            inits.push_back(pb::tensor(nm, {H}, "model.onnx.data", o, uint64_t(H) * 2, 10));
        }
        auto proj = [&](const std::string& pre, int64_t N, int64_t K, bool wrong) {
            const int64_t g = K / block;
            std::vector<uint8_t> qw(size_t(N * g * (block / 2)));
            for (size_t i = 0; i < qw.size(); ++i) qw[i] = uint8_t((i * 7 + 3) & 0xFF);
            std::vector<float> sc(size_t(N * g));
            for (size_t i = 0; i < sc.size(); ++i) sc[i] = 0.01f * float(1 + i % 5);
            // `wrong`: the codes keep their name and the scales sibling does not -- plan() counts a projection present
            // on its weight_Q4 alone, so only the decoder can see that the pair is incomplete.
            const std::string wq = pre + ".MatMul.weight_Q4";
            const std::string ws = pre + (wrong ? ".MatMul.weight_scale" : ".MatMul.weight_scales");
            const uint64_t qo = append(qw.data(), qw.size()), so = append(sc.data(), sc.size() * 4);
            inits.push_back(pb::tensor(wq, {N, g, block / 2}, "model.onnx.data", qo, qw.size(), 2));
            inits.push_back(pb::tensor(ws, {N, g}, "model.onnx.data", so, sc.size() * 4, 1));
            nodes.push_back(pb::matmulnbits(pre, {"x", wq, ws}, K, N, 4, block));
        };
        const std::string a = "model.layers.0.attn.", m = "model.layers.0.mlp.";
        proj(a + "q_proj", heads * hd, H, false);
        proj(a + "k_proj", kv * hd, H, false);
        proj(a + "v_proj", kv * hd, H, false);
        proj(a + "o_proj", H, heads * hd, false);
        proj(m + "gate_proj", I, H, false);
        proj(m + "up_proj", I, H, false);
        proj(m + "down_proj", H, I, spell_down_wrong);
        proj("lm_head", V, H, false);
        put_file((dir / "model.onnx.data").string(), data);
        put_file((dir / "model.onnx").string(), pb::model(inits, nodes));
        put_file((dir / "config.json").string(),
                 R"({"model_type":"qwen2","hidden_size":64,"intermediate_size":64,"num_hidden_layers":1,)"
                 R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"vocab_size":64,)"
                 R"("rope_theta":10000.0,"tie_word_embeddings":false,"rms_norm_eps":1e-06})");
        (void)groups;
    };
    const std::string out = (::onebit::test::tmp_root() / ("oga_pack_genre_a." + std::to_string(::getpid()))).string();
    auto names_down = [](const pi::Report& r) {
        for (const auto& s : r.refused)
            if (s.find("mlp.down_proj.weight") != std::string::npos && s.find("no quantised projection the decoder can pair") != std::string::npos)
                return true;
        return false;
    };

    build(false);
    fs::remove_all(out);
    pi::Options allow;
    allow.allow_incomplete = true;
    pi::Report whole;
    try {
        whole = pi::import_pack(dir.string(), out, allow);
    } catch (const std::exception& e) {
        check(false, std::string("genre-A gap: the whole crafted pack imports (threw: ") + e.what() + ")");
        fs::remove_all(dir);
        return;
    }
    for (const auto& s : whole.refused) std::fprintf(stderr, "  genre-A whole: refused %s\n", s.c_str());
    bool proj_refused = false;
    for (const auto& s : whole.refused) proj_refused |= s.find("no quantised projection the decoder can pair") != std::string::npos;
    check(!proj_refused, "genre-A gap: with every projection present, none is refused as missing");
    check(whole.tensors >= 7, "genre-A gap: the seven layer projections land in the container (" +
                                  std::to_string(whole.tensors) + " tensors)");

    build(true);
    fs::remove_all(out);
    const pi::Report gap = pi::import_pack(dir.string(), out, pi::Options{});
    for (const auto& s : gap.refused) std::fprintf(stderr, "  genre-A gap: refused %s\n", s.c_str());
    check(names_down(gap), "genre-A gap: a projection plan() accepted but the decoder cannot pair is refused BY NAME (down_proj), not skipped");
    check(!fs::exists(out + "/model.q4nx"), "genre-A gap: and without --allow-incomplete no container is written");
    const pi::Report gap2 = pi::import_pack(dir.string(), out, allow);
    check(names_down(gap2) && fs::exists(out + "/model.q4nx"),
          "genre-A gap: --allow-incomplete writes the rest and still lists down_proj");
    std::printf("genre-A gap: whole pack %zu refused / %d tensors; down_proj scales misnamed -> %zu refused, named %s\n",
                whole.refused.size(), whole.tensors, gap.refused.size(), names_down(gap) ? "yes" : "NO");
    fs::remove_all(out);
    fs::remove_all(dir);
}

void test_embedding_component(const std::string& root) {
    const fs::path dir = fs::path(root) / "embedcomp";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const int64_t V = 8, H = 256, groups = H / 128;
    auto want = [&](int64_t v, int64_t k) { return uint8_t((v * 7 + k * 5) % 16); };
    std::vector<uint8_t> table(size_t(V) * size_t(H / 2), 0);
    std::vector<float> scales(size_t(V) * size_t(groups), 0.0f);
    for (int64_t v = 0; v < V; ++v) {
        for (int64_t g = 0; g < groups; ++g) scales[size_t(v) * groups + size_t(g)] = 0.5f * float(g + 1);
        for (int64_t k = 0; k < H; k += 2)
            table[size_t(v) * size_t(H / 2) + size_t(k / 2)] = uint8_t(want(v, k) | (want(v, k + 1) << 4));
    }
    std::string data;
    data.append(reinterpret_cast<const char*>(table.data()), table.size());
    const size_t table_bytes = table.size();
    data.append(reinterpret_cast<const char*>(scales.data()), scales.size() * sizeof(float));
    put_file((dir / "embedding.onnx.data").string(), data);
    const auto t = pb::tensor("language_model.embed_tokens.weight_Q4", {V, H / 2}, "embedding.onnx.data", 0,
                              table_bytes, 2);
    const auto sc = pb::tensor("language_model.embed_tokens.weight_scales", {V, groups}, "embedding.onnx.data",
                               table_bytes, scales.size() * sizeof(float), 1);
    put_file((dir / "embedding.onnx").string(), pb::model({t, sc}, {}));

    std::string why;
    const std::vector<float> got = pi::read_embedding_component(dir.string(), &why);
    check(got.size() == size_t(V) * size_t(H),
          "embedding component: reads V*H values (" + std::to_string(got.size()) + ", why: " + why + ")");
    bool ok = got.size() == size_t(V) * size_t(H);
    for (int64_t v = 0; v < V && ok; ++v)
        for (int64_t k = 0; k < H; ++k) {
            // asymmetric int4: the codes are centred on 8, which is what ONNX Runtime's GatherBlockQuantized does
            const float exp = (float(want(v, k)) - 8.0f) * 0.5f * float(k / 128 + 1);
            if (std::fabs(got[size_t(v) * size_t(H) + size_t(k)] - exp) > 1e-6f) { ok = false; break; }
        }
    check(ok, "embedding component: (code - 8) times its group's scale, group 128, low nibble first");
    std::printf("embedding component: %zu values, V=%lld H=%lld group 128\n", got.size(), (long long)V, (long long)H);

    const fs::path none = dir / "none";
    fs::create_directories(none);
    std::string why2;
    check(pi::read_embedding_component(none.string(), &why2).empty() && !why2.empty(),
          "embedding component: absent -> empty with a reason (" + why2 + ")");
    if (!getenv("KEEP_EMBEDCOMP")) fs::remove_all(dir);
}


// engine#1690: the head's self-check must decode the embedding at the width the importer holds. `embed_bytes` is the
// container's bf16 copy on every path (#1661 made that true and left four readers decoding F16), so a tied
// Qwen2.5-0.5B 1.9 pack reported corr 0.885220 -- and "untied" -- for a container whose own numbers are 0.999999.
// This pins the statistic against a known matrix at both widths: the bf16 buffer must correlate ~1.0, and the same
// values encoded as F16 must not, which is the arm that would have caught #1661 when it landed.
void test_selfcheck_reads_bf16() {
    const int V = 64, H = 64;
    auto code_of = [&](int n, int k) { return (n * 7 + k) % 8; };
    pi::Q4 q;
    q.N = V;
    q.K = H;
    q.block = 32;
    q.codes.resize(size_t(V) * H);
    q.scale.assign(size_t(V) * (H / 32), 0.5f);
    q.zero.assign(size_t(V) * (H / 32), 0.0f);
    for (int n = 0; n < V; ++n)
        for (int k = 0; k < H; ++k) q.codes[size_t(n) * H + k] = uint8_t(code_of(n, k));
    // the same values, once as bf16 (what the importer holds) and once as f16 (what it used to assume)
    std::vector<uint8_t> bf(size_t(V) * H * 2), f16(size_t(V) * H * 2);
    for (int n = 0; n < V; ++n)
        for (int k = 0; k < H; ++k) {
            const float v = float(code_of(n, k)) * 0.5f;   // 0.0 .. 3.5, exact in both formats
            const uint16_t b = pi::bf16_bits_of(v);
            const uint16_t h = uint16_t(v == 0.0f ? 0x0000 : (0x3800 | (uint16_t(std::lround(v * 2.0f) - 2) << 10)));
            const size_t i = (size_t(n) * H + k) * 2;
            bf[i] = uint8_t(b & 0xFF); bf[i + 1] = uint8_t(b >> 8);
            f16[i] = uint8_t(h & 0xFF); f16[i + 1] = uint8_t(h >> 8);
        }
    const double good = pi::corr_bf16(q, bf.data(), H, V);
    const double wrong = pi::corr_bf16(q, f16.data(), H, V);
    std::printf("selfcheck-bf16: corr(bf16 buffer) %.6f, corr(f16 buffer) %.6f\n", good, wrong);
    check(good > 0.999, "selfcheck-bf16: the statistic reads the bf16 buffer the importer holds (corr " +
                            std::to_string(good) + ")");
    check(wrong < 0.99, "selfcheck-bf16: and reading the same values at the wrong width does not (corr " +
                            std::to_string(wrong) + ") -- the arm that would have caught #1661");

    // engine#1721: the width now travels with the pointer, so the accident #1661 made -- repointing the bytes under
    // four readers that all decoded F16 -- cannot recur silently: a reader that wants F16 has to say Fmt::F16, and the
    // default is the format the importer actually holds.
    {
        uint32_t f32bits = 0;
        const float v = 1.5f;
        std::memcpy(&f32bits, &v, 4);
        const uint16_t bits = uint16_t(f32bits >> 16);   // bf16 1.5 = 0x3FC0; the same bits read as f16 are 1.75
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(&bits);
        const pi::EmbedTable as_bf16{raw, pi::EmbedTable::Fmt::BF16, 1};
        const pi::EmbedTable as_f16{raw, pi::EmbedTable::Fmt::F16, 1};
        std::printf("embed-table: same bytes, bf16 %.3f, f16 %.3f, default %s\n", pi::embed_at(as_bf16, 0, 0),
                    pi::embed_at(as_f16, 0, 0),
                    pi::EmbedTable{}.fmt == pi::EmbedTable::Fmt::BF16 ? "BF16" : "F16");
        check(std::fabs(pi::embed_at(as_bf16, 0, 0) - 1.5) < 1e-6,
              "embed-table: Fmt::BF16 decodes the bytes as bf16");
        // 0x3FC0 read as f16 is 1.9375 (exp 15 -> 2^0, mantissa 0x3C0/1024), not the 1.75 I first wrote here: the
        // point of the arm is that the two formats DISAGREE, and getting the disagreement's value wrong is the same
        // mistake in the test rather than the code.
        check(std::fabs(pi::embed_at(as_f16, 0, 0) - 1.9375) < 1e-6,
              "embed-table: Fmt::F16 on the SAME bytes gives a different number, so the width cannot be forgotten");
        check(pi::EmbedTable{}.fmt == pi::EmbedTable::Fmt::BF16,
              "embed-table: the default format is what the importer holds, so an unset width is the right one");
    }
}

// engine#1633: the two behaviours that decide correctness in the 1.9 rule table, plus the one shape the table must
// never contain. The row is the product of this PR -- every field measured against the 1.8 twin by
// tools/fconst19_measure.py -- so a wrong row rotates a pack's weights into the wrong projections while leaving the
// prefix and the epsilon anchors untouched, i.e. it reads as a correct parse and answers quietly wrong text.
void test_fconst19_rule_table() {
    // 1. a measured geometry returns its row, with the numbers that were measured
    std::string why;
    const pi::Fconst19Rule* r = pi::select_fconst19_rule(896, 14, 24, &why);
    check(r && r->prefix == 1856 && r->n_tiles == 16 && r->rot == 2 && r->tile == 34816,
          "fconst19 rule: Qwen2.5-0.5B's geometry returns its measured row (prefix " +
              std::to_string(r ? r->prefix : -1) + ", " + std::to_string(r ? r->n_tiles : -1) + " tiles of " +
              std::to_string(r ? r->tile : -1) + " B, rot " + std::to_string(r ? r->rot : -1) + ")");

    // 2. a geometry that has never been measured is refused, and the refusal names the geometry
    why.clear();
    check(!pi::select_fconst19_rule(1234, 7, 11, &why) && why.find("1234") != std::string::npos &&
              why.find("has not been measured") != std::string::npos,
          "fconst19 rule: an unmeasured geometry is refused, naming it (" + why + ")");

    // 3. the shape the real table must never contain: one geometry measured TWICE with different block shapes. A pack's
    //    geometry cannot choose between them, and picking either is the silent mis-rotation this table exists to avoid.
    const pi::Fconst19Rule two[] = {{2048, 16, 28, 8512, 3, 1, 1183744},     // Qwen3-1.7B's
                                    {2048, 16, 28, 8512, 5, 1, 591872}};     // Llama-3.2-1B's bundle, same geometry
    why.clear();
    check(!pi::select_fconst19_rule_from(two, 2, 2048, 16, 28, &why) && why.find("DIFFERENT") != std::string::npos &&
              why.find("1183744") != std::string::npos && why.find("591872") != std::string::npos,
          "fconst19 rule: two rows for one geometry refuse rather than pick (" + why.substr(0, 120) + "...)");

    // control for 3: the same geometry listed twice with the SAME shape is not ambiguous
    const pi::Fconst19Rule dup[] = {{1536, 12, 28, 3136, 7, 1, 226304}, {1536, 12, 28, 3136, 7, 1, 226304}};
    why.clear();
    check(pi::select_fconst19_rule_from(dup, 2, 1536, 12, 28, &why) != nullptr,
          "fconst19 rule: a geometry measured twice with the same shape is fine -- that is the h1536 pair, which is "
          "byte-identical over all 28 blocks");
    // 4. engine#1774 / TNPU-ALL W2's third gate clause, made able to fail (engine#1854). The expected rows live HERE, as
    //    literals copied from the measured table, not read back from it: an arm that selects each table row by its own key
    //    and compares it with itself passes a wrong digit (h4096 prefix 12608 -> 12609) and a dropped row (10 >= 8), the
    //    two failures this arm exists for. Adding or re-measuring a row is therefore a deliberate edit in BOTH places.
    {
        const pi::Fconst19Rule want[] = {
            {896, 14, 24, 1856, 16, 2, 34816},       // Qwen2.5-0.5B-Instruct
            {2048, 16, 28, 8512, 3, 1, 1183744},     // Qwen3-1.7B
            {576, 9, 30, 3776, 4, 1, 76800},         // SmolLM2-135M-Instruct
            {1536, 12, 28, 3136, 7, 1, 226304},      // Qwen2.5-Coder-1.5B-Instruct (and Qwen-2.5_1.5B_Instruct)
            {2560, 32, 36, 9536, 5, 1, 1462272},     // Qwen3-4B
            {2048, 32, 16, 8512, 5, 1, 591872},      // Llama-3.2-1B-Instruct
            {3072, 32, 32, 10560, 2, 1, 5222400},    // Phi-3.5-mini-instruct (and Phi-3-mini-128k)
            {3584, 28, 28, 7232, 8, 1, 1009664},     // the Qwen2.5-7B family, measured on two models
            {2048, 16, 36, 4160, 9, 1, 295936},      // Qwen2.5_3B_Instruct
            {4096, 32, 32, 12608, 5, 1, 2297856},    // Mistral-7B-Instruct-v0.2
            {4096, 32, 36, 12608, 5, 1, 2297856},    // Qwen3-8B (measured against ..._rai_1.8.0_npu_16K)
            {3072, 24, 28, 10560, 4, 1, 1740800},    // Llama-3.2-3B (and -Instruct)
        };
        const int want_n = (int)(sizeof(want) / sizeof(want[0]));
        int rules_n = 0;
        const pi::Fconst19Rule* rules = pi::fconst19_rules(&rules_n);
        check(rules && rules_n == want_n,
              "fconst19 rule: the table has exactly the " + std::to_string(want_n) + " measured rows (" +
                  std::to_string(rules_n) + "); a dropped row is a silent un-measure of a whole class, an added one "
                  "must be added here too");
        for (const pi::Fconst19Rule& row : want) {
            std::string w;
            const pi::Fconst19Rule* got = pi::select_fconst19_rule(row.hidden, row.heads, row.layers, &w);
            check(got && got->prefix == row.prefix && got->n_tiles == row.n_tiles && got->rot == row.rot &&
                      got->tile == row.tile,
                  "fconst19 rule: " + std::to_string(row.hidden) + "/" + std::to_string(row.heads) + "h/" +
                      std::to_string(row.layers) + "L selects its measured numbers (" +
                      std::to_string(got ? got->prefix : -1) + ", " + std::to_string(got ? got->n_tiles : -1) + ", " +
                      std::to_string(got ? got->rot : -1) + ", " + std::to_string(got ? got->tile : -1) + ")");
        }
        // what the old self-select loop really tested: no two rows of the real table share a key with different
        // shapes (select_rule_in refuses that as ambiguous, so every row must still select)
        for (int i = 0; rules && i < rules_n; i++) {
            std::string w;
            check(pi::select_fconst19_rule(rules[i].hidden, rules[i].heads, rules[i].layers, &w) != nullptr,
                  "fconst19 rule: no two rows share the key " + std::to_string(rules[i].hidden) + "/" +
                      std::to_string(rules[i].heads) + "h/" + std::to_string(rules[i].layers) + "L with different shapes");
        }
    }

    // 5. the case this row is: the SECOND geometry at hidden 2048 / 16 heads, told apart only by its layer count. If a
    //    later edit relaxed the key to (hidden, heads) or merged rows, these two must still separate.
    why.clear();
    const pi::Fconst19Rule* q17 = pi::select_fconst19_rule(2048, 16, 28, &why);
    const pi::Fconst19Rule* q3 = pi::select_fconst19_rule(2048, 16, 36, &why);
    check(q17 && q17->prefix == 8512 && q17->n_tiles == 3 && q17->tile == 1183744,
          "fconst19 rule: 2048/16h/28L is still Qwen3-1.7B's row (8,512 / 3 tiles of 1,183,744)");
    check(q3 && q3->prefix == 4160 && q3->n_tiles == 9 && q3->rot == 1 && q3->tile == 295936,
          "fconst19 rule: 2048/16h/36L is Qwen2.5_3B's row (4,160 / 9 tiles of 295,936) -- the prefix is NOT the 8,512 "
          "the Qwen3 formula predicts, which is why the row is measured");
    why.clear();
    check(!pi::select_fconst19_rule(2048, 16, 32, &why) && why.find("has not been measured") != std::string::npos,
          "fconst19 rule: 2048/16h/32L is still unmeasured and refuses by name (" + why + ")");

    std::printf("fconst19 rule: measured row, unmeasured refusal, and the ambiguity guard all behave\n");
}

// engine#1831: the header-derived layout keeps EVERY in-block header the anchors support. #1819's version kept only the most
// common residue and always wrote a 2-header block, so a Qwen3 4-header block (three equally frequent inner headers) was read
// as 2-header with k_norm's header standing in for the post-attention one -- the engine#1003 misread, reintroduced.
void test_fconst_block_headers() {
    const int64_t stride = 1000;
    auto blocks = [&](std::initializer_list<int64_t> inner, int nblocks, std::initializer_list<int64_t> noise) {
        std::vector<int64_t> pos;
        for (int b = 0; b < nblocks; ++b) {
            pos.push_back(int64_t(b) * stride);                          // the block's own (first) header
            for (const int64_t r : inner) pos.push_back(int64_t(b) * stride + r);
        }
        for (const int64_t q : noise) pos.push_back(q);                  // eps-SHAPED payload bytes: once or twice
        std::sort(pos.begin(), pos.end());
        return pos;
    };
    // 2-header (Qwen2.5/Llama): one inner header, 8 blocks, two coincidences elsewhere in the payload
    {
        const auto inner = pi::fconst_block_headers(blocks({400}, 8, {1234, 5678}), stride, 3);
        check(inner.size() == 1 && inner[0] == 400, "block headers: a 2-header block yields its one inner header");
        pi::FconstGeom g;
        g.stride = stride;
        check(pi::fconst_set_block_headers(g, inner) && g.period == 2 && g.h1 == 400 && g.hdr.size() == 1 &&
                  g.post_attn_hdr() == 400,
              "block headers: and is set as a 2-header block whose post-attention header is h1");
    }
    // 4-header (Qwen3): three inner headers repeating equally -- ALL must be kept, in order, so post_attn is the LAST one
    {
        const auto inner = pi::fconst_block_headers(blocks({300, 320, 340}, 8, {777}), stride, 3);
        check(inner.size() == 3 && inner[0] == 300 && inner[1] == 320 && inner[2] == 340,
              "block headers: a 4-header block yields all three inner headers, sorted (not just the most common one)");
        pi::FconstGeom g;
        g.stride = stride;
        check(pi::fconst_set_block_headers(g, inner) && g.period == 4 && g.h1 == 300 && g.hdr.size() == 3 &&
                  g.post_attn_hdr() == 340,
              "block headers: and is set as a 4-header block whose post-attention header is the last one, not k_norm's");
    }
    // coincidences alone are not a header, and a count that is neither 1 nor 3 sets nothing
    {
        check(pi::fconst_block_headers(blocks({}, 8, {111, 222, 222}), stride, 3).empty(),
              "block headers: payload coincidences that recur fewer than min_repeat times are not headers");
        pi::FconstGeom g;
        g.stride = stride;
        g.h1 = 7;
        check(!pi::fconst_set_block_headers(g, {100, 200}) && g.hdr.empty() && g.h1 == 7 && g.period == 0,
              "block headers: two inner headers is not a known layout, and nothing is changed");
        check(pi::fconst_block_headers({0, 1000, 2000}, 0, 1).empty(), "block headers: a zero stride yields nothing");
    }
    std::printf("  fconst block headers: 2-header and 4-header layouts kept whole; unknown counts refused\n");
}

// engine#1801's reviewer asked for the test that would catch the case the fix itself found once: the canonical .fconst copy is a
// cache whose size is the pack's own .fconst, and on this box /tmp is tmpfs, so where it lands is a RAM decision. The bug was a
// cache directory that was never created -- every pack then fell through to /tmp and nothing noticed.
void test_fconst_scratch_order() {
    // engine#1907: this arm rewrites four variables, and it used to leave HOME=/home/x with TMPDIR and the override unset, so
    // every crafted import after it fell through to the literal /tmp (179 of the 188 files found there were this suite's).
    // Save and restore them, so the rest of the run keeps the per-run scratch directory main() set.
    struct Saved { const char* name; bool had; std::string value; };
    std::vector<Saved> saved;
    for (const char* n : {"ONEBIT_FCONST_SCRATCH", "XDG_CACHE_HOME", "HOME", "TMPDIR"}) {
        const char* v = std::getenv(n);
        saved.push_back({n, v != nullptr, v ? v : ""});
    }
    struct Restore {
        std::vector<Saved>& s;
        ~Restore() { for (const auto& x : s) { if (x.had) setenv(x.name, x.value.c_str(), 1); else unsetenv(x.name); } }
    } restore{saved};
    const auto first_with = [](const char* scratch, const char* xdg, const char* home) {
        if (scratch) setenv("ONEBIT_FCONST_SCRATCH", scratch, 1); else unsetenv("ONEBIT_FCONST_SCRATCH");
        if (xdg) setenv("XDG_CACHE_HOME", xdg, 1); else unsetenv("XDG_CACHE_HOME");
        if (home) setenv("HOME", home, 1); else unsetenv("HOME");
        const auto dirs = pi::fconst_scratch_dirs();
        return dirs.empty() ? std::string() : dirs.front();
    };
    // 1. an explicit override wins, because a lane's own scratch dir is the only place it controls
    check(first_with("/tmp/scratch-override", nullptr, "/home/x") == "/tmp/scratch-override",
          "fconst scratch: ONEBIT_FCONST_SCRATCH is chosen first");
    // 2. otherwise the on-disk cache directory, never the temp directory -- the whole point of engine#1801
    check(first_with(nullptr, "/xdg-cache", "/home/x") == "/xdg-cache/1bit/fconst19",
          "fconst scratch: XDG_CACHE_HOME/1bit/fconst19 is chosen when no override is set");
    check(first_with(nullptr, nullptr, "/home/x") == "/home/x/.cache/1bit/fconst19",
          "fconst scratch: HOME/.cache/1bit/fconst19 is the default -- disk, not tmpfs");
    // 3. the temp directory must never come first: on this box that is RAM equal to the pack's .fconst
    const auto dirs = pi::fconst_scratch_dirs();
    check(!dirs.empty() && dirs.front().find("/tmp") != 0,
          "fconst scratch: the first choice is not the temp directory (it was, before #1801)");
    // 4. the bug the fix shipped once: a cache directory that does not exist must be CREATED, not skipped. Without this the loop
    //    walked past the new first choice and every pack's 3.6 GB scratch copy landed in tmpfs anyway.
    const std::string fresh = "/tmp/1bit-scratch-test-" + std::to_string(::getpid()) + "/nested";
    std::system(("rm -rf /tmp/1bit-scratch-test-" + std::to_string(::getpid())).c_str());
    check(!std::filesystem::exists(fresh) && pi::fconst_scratch_ready(fresh) && std::filesystem::is_directory(fresh),
          "fconst scratch: a missing cache directory is created, not skipped (the #1801 first-version bug)");
    std::system(("rm -rf /tmp/1bit-scratch-test-" + std::to_string(::getpid())).c_str());
    // 5. and there is always somewhere to fall back to
    check(dirs.size() >= 2, "fconst scratch: more than one candidate, so a box with no HOME still has a fallback");
    // 6. engine#1818 review: `$TMPDIR` sits BETWEEN the cache and the literal /tmp. It was dropped by the extraction, and a
    //    lane that points TMPDIR at disk -- which is what lanes here do, because /tmp is RAM -- then landed its scratch copy
    //    in tmpfs whenever the cache directory was unusable. That is the outcome #1801 exists to prevent, so it is pinned.
    {
        const std::string tdir = "/tmp/1bit-tmpdir-test-" + std::to_string(::getpid());
        std::system(("mkdir -p " + tdir).c_str());
        unsetenv("ONEBIT_FCONST_SCRATCH");
        unsetenv("XDG_CACHE_HOME");
        unsetenv("HOME");                       // no usable cache: the case the reviewer named
        setenv("TMPDIR", tdir.c_str(), 1);
        const auto with_tmpdir = pi::fconst_scratch_dirs();
        size_t at_tmpdir = with_tmpdir.size(), at_slash_tmp = with_tmpdir.size();
        for (size_t i = 0; i < with_tmpdir.size(); ++i) {
            if (with_tmpdir[i] == tdir) at_tmpdir = i;
            if (with_tmpdir[i] == "/tmp") at_slash_tmp = i;
        }
        check(at_tmpdir < with_tmpdir.size(),
              "fconst scratch: a TMPDIR with no override and no cache is a candidate at all (it was dropped)");
        check(at_tmpdir < at_slash_tmp,
              "fconst scratch: and it comes BEFORE the literal /tmp, which is RAM on this box");
        // control: with TMPDIR unset the resolver returns /tmp, so the literal must still be there and unused twice
        unsetenv("TMPDIR");
        const auto no_tmpdir = pi::fconst_scratch_dirs();
        check(std::count(no_tmpdir.begin(), no_tmpdir.end(), std::string("/tmp")) == 1,
              "fconst scratch: with TMPDIR unset, /tmp appears exactly once (no duplicate from the resolver)");
        std::system(("rm -rf " + tdir).c_str());
    }
    std::printf("fconst scratch: order is override -> disk cache -> temp/cwd, pinned\n");
}

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
    const std::string out = (::onebit::test::tmp_root() /
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

// The text artifacts are not universal: AMD's Qwen3 packs ship none (engine#1024), so `info` has to
// say so before an import and `import` has to say so after it. A pack with them reports the split.
// The kernel set a container is linked to has to be the family the container declares: the forward
// looks up `full_i8_<role>_K<k>_N<n>[_b8][_m16].elf` by name, so a b8 container pointed at a row-major
// set loads, finds no matching name and fails at serve time with "no ELF for K=.. N=..". Refusing at
// import time is the difference between a named error and a container that looks broken.
void test_kernel_set_refusal() {
    const std::vector<std::string> b8 = {"full_i8_QKV_K3072_N9216_b8.elf",
                                        "full_i8_QKV_K3072_N9216_b8_m16.elf",
                                        "full_i8_GU_K3072_N16384_b8.elf"};
    const std::vector<std::string> plain = {"full_i8_QKV_K3072_N9216.elf",
                                           "full_i8_QKV_K3072_N9216_m16.elf"};
    const std::vector<std::string> mixed = {"full_i8_QKV_K3072_N9216_b8.elf",
                                           "full_i8_QKV_K3072_N9216_b8_m16.elf",
                                           "full_i8_GU_K3072_N16384.elf"};
    check(pi::kernel_set_refusal(b8, "b8").empty(), "kernels: a b8 set for a b8 container is accepted");
    check(pi::kernel_set_refusal(plain, "").empty(),
          "kernels: a row-major set for a container with no layout key is accepted");
    check(pi::kernel_set_refusal(mixed, "").empty(),
          "kernels: a set with row-major ELFs is accepted for a row-major container even if it also "
          "carries b8 twins");
    check(!pi::kernel_set_refusal(b8, "").empty(),
          "kernels: a b8 set is refused when the container declares no layout (it needs --weight-layout "
          "b8)");
    check(!pi::kernel_set_refusal(plain, "b8").empty(),
          "kernels: a row-major set is refused for a b8 container");
    check(!pi::kernel_set_refusal({}, "").empty(), "kernels: an empty directory is refused");
    check(!pi::kernel_set_refusal({"README.md", "notes.txt"}, "b8").empty(),
          "kernels: a directory with no full_i8 ELF is refused");
    check(!pi::kernel_set_refusal({"full_i8_QKV_K3072_N9216_b8.elf.bak"}, "b8").empty(),
          "kernels: a file that is not an .elf is not a kernel");
}

// engine#1751 follow-up: the one copier both import paths use. #1751 added a second copier that counted successes and
// swallowed failures, so an import could "succeed" without tokenizer.json and serve then refused the directory for a
// missing tokenizer -- the misleading report #1751 was written to remove. A file the pack has and cannot copy is refused
// BY NAME; a file the pack does not ship is not a failure.
void test_copy_text_artifacts() {
    const fs::path base = ::onebit::test::tmp_root() / ("1bit-textcopy-" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(base, ec);
    const fs::path pack = base / "pack", out = base / "out";
    fs::create_directories(pack);
    fs::create_directories(out);
    put_file((pack / "tokenizer.json").string(), "{}");
    put_file((pack / "tokenizer_config.json").string(), "{}");
    // (a) the files the pack has are copied, the rest skipped without complaint
    int copied = -1;
    std::string why;
    try { copied = pi::copy_text_artifacts_exported(pack.string(), out.string()); } catch (const std::exception& e) { why = e.what(); }
    check(why.empty() && copied == 2 && fs::exists(out / "tokenizer.json") && fs::exists(out / "tokenizer_config.json"),
          "text copy: the two files the pack ships are copied, the six it does not are skipped (" + why + ")");
    // (b) a copy that FAILS is refused, naming the file -- not counted down to a smaller number nobody reads
    why.clear();
    try { (void)pi::copy_text_artifacts_exported(pack.string(), (base / "no-such-dir").string()); } catch (const std::exception& e) { why = e.what(); }
    check(why.find("tokenizer.json") != std::string::npos,
          "text copy: a file the pack has that cannot be copied is refused by name (" + why + ")");
    fs::remove_all(base, ec);
    std::printf("  text copy: present files copied, absent ones skipped, a failed copy refused by name\n");
}

void test_text_artifacts(const std::string& root) {
    const fs::path dir = fs::path(root) / "text-artifacts";
    fs::remove_all(dir);
    fs::create_directories(dir);
    put_file((dir / "model.onnx.data").string(), std::string(4096, '\x11'));
    put_file((dir / "config.json").string(),
             R"({"model_type":"qwen2","hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,)"
             R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"vocab_size":64,)"
             R"("rope_theta":10000.0,"tie_word_embeddings":true})");
    put_file((dir / "model.onnx").string(),
             pb::model({pb::tensor("model.embed_tokens.weight", {8, 8}, "model.onnx.data", 0, 256)}, {}));

    pi::TextArtifacts t = pi::scan_text_artifacts(dir.string());
    check(t.present.empty() && t.missing.size() == 8 && !t.tokenizer,
          "text artifacts: a pack with none reports all 8 missing and not servable");
    check(json::parse(pi::plan(dir.string()).manifest())["text_artifacts"]["servable"] == false,
          "text artifacts: info says the pack is not servable");

    put_file((dir / "tokenizer.json").string(), "{}");
    put_file((dir / "vocab.json").string(), "{}");
    t = pi::scan_text_artifacts(dir.string());
    check(t.present == std::vector<std::string>({"tokenizer.json", "vocab.json"}) && t.missing.size() == 6 &&
              t.tokenizer,
          "text artifacts: a pack with two reports those two present, the other six missing");
    const json m = json::parse(pi::plan(dir.string()).manifest());
    check(m["text_artifacts"]["servable"] == true && m["text_artifacts"]["present"].size() == 2 &&
              m["text_artifacts"]["missing"].size() == 6,
          "text artifacts: info lists present, missing and servable");
    std::printf("text artifacts: %zu present, %zu missing, servable %s\n", t.present.size(),
                t.missing.size(), t.tokenizer ? "yes" : "no");
    fs::remove_all(dir);
}

// --weight-layout: the flag may only declare a layout this importer can actually produce a working
// container for. b8 needs nothing else; the default (no key at all) is the row-major family; t2s and
// pb16i need a version and a sidecar image directory, so accepting them here would write a container
// that refuses to load (engine#1189 follow-up: the Phi-3.5 pack's kernel set is the _b8 family).
void test_weight_layout() {
    check(pi::weight_layout_refusal("").empty(), "weight layout: no flag is allowed (the default)");
    check(pi::weight_layout_refusal("rm").empty(), "weight layout: rm (no key) is allowed");
    check(pi::weight_layout_refusal("b8").empty(), "weight layout: b8 is allowed");
    check(!pi::weight_layout_refusal("pb16i").empty(),
          "weight layout: pb16i is refused (it also needs a version and sidecar images)");
    check(!pi::weight_layout_refusal("t2s").empty(),
          "weight layout: t2s is refused (it also needs a version and sidecar images)");
    const std::string why = pi::weight_layout_refusal("b7");
    check(!why.empty() && why.find("b7") != std::string::npos,
          "weight layout: an unknown name is refused by name");
    std::printf("weight layout: b8 allowed, pb16i/t2s/unknown refused\n");
}

// engine#1650: a config.json that declares an OPTIONAL field as JSON null means "unset", not
// "broken". AMD's 1.9 Mistral-7B-Instruct-v0.2 pack ships `"head_dim": null` and
// `"sliding_window": null`, and nlohmann's value() throws on a present-but-null key, so the reader
// reported `type must be number, but is null` and no geometry at all: an 8 GB pack named "info
// failed" with nothing said about what it contains. head_dim's default is the one every
// Llama-family config implies (hidden / heads); a WRONG TYPE must still be refused.
void test_null_optional_dims() {
    using nlohmann::json;
    const json base = {{"hidden_size", 4096}, {"num_hidden_layers", 32},
                       {"num_attention_heads", 32}, {"intermediate_size", 14336}};
    json c = base;
    for (const char* k : {"head_dim", "sliding_window", "num_key_value_heads", "rope_theta",
                          "vocab_size", "rms_norm_eps", "model_type"})
        c[k] = nullptr;                       // the measured pack, field for field
    const pi::Dims d = pi::read_dims_exported(c, "config.json");
    check(d.head_dim == 128, "null head_dim defaults to hidden/heads (4096/32 = 128)");
    check(d.kv_heads == d.heads, "null num_key_value_heads defaults to the head count");
    check(d.rope_theta == 10000.0f, "null rope_theta defaults to 10000");
    check(d.vocab == 0, "null vocab_size reads as 0 (unset)");
    check(std::abs(d.rms_norm_eps - 1e-6f) < 1e-12f, "null rms_norm_eps defaults to 1e-6");
    check(d.model_type.empty(), "null model_type reads as empty, not as a crash");

    json good = base;
    good["head_dim"] = 128;                   // an explicit value still wins over the default
    check(pi::read_dims_exported(good, "config.json").head_dim == 128, "an explicit head_dim wins");

    json bad = base;
    bad["head_dim"] = "128";                  // null is absence; a string is a disagreement
    bool threw = false;
    try { (void)pi::read_dims_exported(bad, "config.json"); } catch (const std::exception&) { threw = true; }
    check(threw, "a non-numeric head_dim is still refused");
    std::printf("null optional dims: head_dim falls back to hidden/heads, a wrong type still throws\n");
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

// Opt-in: read a REAL pack's embedding component (NPU_EMBED_COMPONENT=<dir>) and dump one row for an oracle comparison.
// The 4B pack's component is 206 MB, not 3 GB, so this is the cheap way to settle the two facts the reader's comment
// calls pack-settled -- the nibble order and the implicit zero point -- against ONNX Runtime running the pack's own
// embedding.onnx.
static void dump_real_component(const std::string& dir) {
    std::string why;
    const std::vector<float> v = pi::read_embedding_component(dir, &why);
    if (v.empty()) {
        std::printf("real component: EMPTY (%s)\n", why.c_str());
        return;
    }
    const std::string cfg = dir + "/config.json";
    size_t H = 0;
    if (FILE* f = std::fopen(cfg.c_str(), "rb")) { fclose(f); }
    // H comes from the file the caller names next to the dump; keep this independent of the JSON parser.
    H = v.size() / 151936;                     // the 4B shape; asserted below against the advertised V
    std::printf("real component: %zu values, H=%zu\n", v.size(), H);
    const int row = 1;                          // token id 1: a plain text token ORT embeds
    if ((row + 1) * H > v.size()) return;
    const std::string out = dir + "/eng_row.f32";
    if (FILE* f = onebit_fopen_private_wb(out.c_str())) {
        std::fwrite(&v[size_t(row) * H], sizeof(float), H, f);
        std::fclose(f);
        double s = 0;
        for (size_t k = 0; k < H; ++k) s += std::fabs(v[size_t(row) * H + k]);
        std::printf("real component: row %d -> %s (mean|x| %.6f)\n", row, out.c_str(), s / double(H));
    }
}

// A multi-component pack (gemma-4 1.9, `type: compositemodel`) states its geometry in model_config.json's
// config.model_attributes.text_config -- not in config.json (which is `{}`) nor in genai_config.json's decoder block
// (which stops at head_size). Measured on amd/gemma-4-e2b-it-mobius-int4_rai_1.9.0_npu: hidden 1536, 35L, 8 heads,
// 1 kv, head_dim 256, intermediate_size 6144, vocab 262144. Without this source the pack refuses with "the pack
// states no intermediate_size", which is exactly what it did on the real pack.
void test_model_config_geometry(const std::string& root) {
    const fs::path dir = fs::path(root) / "modelcfg";
    fs::remove_all(dir);
    fs::create_directories(dir);
    // what the genai_config decoder block resolves on such a pack: everything but intermediate_size
    nlohmann::json cfg = {{"hidden_size", 1536},        {"num_hidden_layers", 35}, {"num_attention_heads", 8},
                          {"num_key_value_heads", 1},   {"head_dim", 256},         {"model_type", "gemma4"}};
    put_file((dir / "model_config.json").string(),
             R"({"type":"compositemodel","config":{"model_attributes":{"text_config":{)"
             R"("hidden_size":1536,"num_hidden_layers":48,"num_attention_heads":16,"num_key_value_heads":8,)"
             R"("head_dim":256,"intermediate_size":6144,"vocab_size":262144,)"
             R"("max_position_embeddings":131072,"model_type":"gemma4_text"}}}})");
    const int added = pi::fill_from_model_config(dir.string(), cfg);
    check(added > 0, "model_config: the text tower's geometry is read from model_config.json (" + std::to_string(added) +
                         " key(s) added)");
    check(cfg.value("intermediate_size", 0) == 6144,
          "model_config: intermediate_size comes from the text tower (6144)");
    check(cfg.value("vocab_size", 0) == 262144 && cfg.value("max_position_embeddings", 0) == 131072,
          "model_config: vocab and context come with it");
    std::printf("model_config: added %d key(s), intermediate %d, vocab %d, layers %d (genai's value kept)\n", added,
                cfg.value("intermediate_size", 0), cfg.value("vocab_size", 0), cfg.value("num_hidden_layers", 0));
    check(cfg.value("hidden_size", 0) == 1536 && cfg.value("num_hidden_layers", 0) == 35 &&
              cfg.value("num_attention_heads", 0) == 8,
          "model_config: a value already resolved (genai's 1536/35/8) is NOT overridden by the tower's 1536/48/16");
    // A pack without the file contributes nothing and reports nothing added.
    fs::remove(dir / "model_config.json");
    nlohmann::json cfg2 = {{"hidden_size", 1536}};
    check(pi::fill_from_model_config(dir.string(), cfg2) == 0 && !cfg2.contains("intermediate_size"),
          "model_config: no model_config.json adds nothing (the 1.6/gemma-3 shape is unchanged)");
    fs::remove_all(dir);
}

// The 1.9 external-data packs state the FFN width in no file and in no initializer dim (dims are `[0]`, the bytes
// live in a *.qweight.packed sibling), so plan()'s initializer heuristic sees nothing. The NODE carries K and N,
// and the decoder sits in an `If` branch. Measured on amd/gemma-4-e4b-it-mobius-int4_rai_1.9.0_npu: hidden 2560,
// 42 activations each feeding two (N=10240,K=2560) gate/up projections and 42 down projections (N=2560,K=10240).
void test_im_from_node_attrs(const std::string& root) {
    const fs::path dir = fs::path(root) / "imattrs";
    fs::remove_all(dir);
    fs::create_directories(dir);
    auto mm_named = [](const std::string& weight, const std::string& in, int64_t K, int64_t N) {
        return pb::matmulnbits(weight, {in, weight, weight + ".scales", weight + ".zeros"}, K, N, 4, 32);
    };
    auto mm = [&](const std::string& n, int64_t K, int64_t N) { return mm_named(n, "x", K, N); };
    const std::string mlp = "model.layers.0.mlp.";
    // the branch: gate and up (K = hidden, N = IM) reading ONE activation, down (K = IM, N = hidden), a "mobius" width,
    // and lm_head -- the e4b shape, with the role names the genre carries
    const std::string branch = pb::graph(
        {}, {mm_named(mlp + "gate_proj.MatMulNBits.qweight", "h", 2560, 10240),
             mm_named(mlp + "up_proj.MatMulNBits.qweight", "h", 2560, 10240),
             mm_named(mlp + "down_proj.MatMulNBits.qweight", "a", 10240, 2560),
             mm_named("model.layers.0.mobius.MatMulNBits.qweight", "m", 2560, 2048),
             mm_named("lm_head.MatMulNBits.qweight", "l", 2560, 262144)});
    put_file((dir / "model.onnx").string(), pb::model({}, {pb::if_node("cond", "inputs_embeds", "logits", "else_branch", branch)}));
    const int im = pi::intermediate_size_from_graph((dir / "model.onnx").string(), 2560);
    std::printf("im-from-node-attrs: derived %d from the If branch's gate/up and down pair (hidden 2560)\n", im);
    check(im == 10240, "im-from-node-attrs: the FFN pair inside the If branch gives 10240 (" + std::to_string(im) + ")");

    // The review's case (#1657): attention forms the same (N == other's K, K == hidden) shape. With q_dim 4096 > hidden
    // 2560, a bare "pair that agrees" returns 4096 as the FFN width while the true value is 9728 (fused gate_up N=19456 =
    // 2w, down K=9728 = w: down_proj reads the gated product, engine#1756). Attention must not be able to answer, and the
    // fused pair must.
    const std::string attn = "model.layers.0.attn.";
    const std::string wider = pb::graph(
        {}, {mm_named(attn + "q_proj.MatMulNBits.qweight", "h", 2560, 4096),
             mm_named(attn + "o_proj.MatMulNBits.qweight", "q", 4096, 2560),
             mm_named(attn + "k_proj.MatMulNBits.qweight", "h", 2560, 2048),
             mm_named(attn + "v_proj.MatMulNBits.qweight", "h", 2560, 2048),
             mm_named(mlp + "gate_up_proj.MatMulNBits.qweight", "h", 2560, 19456),
             mm_named(mlp + "down_proj.MatMulNBits.qweight", "a", 9728, 2560)});
    put_file((dir / "model2.onnx").string(), pb::model({}, {pb::if_node("cond", "inputs_embeds", "logits", "else_branch", wider)}));
    const int im2 = pi::intermediate_size_from_graph((dir / "model2.onnx").string(), 2560);
    std::printf("im-from-node-attrs: q_dim 4096 > hidden 2560, fused gate_up -> %d (must not be 4096)\n", im2);
    check(im2 == 9728, "im-from-node-attrs: wider attention does not become the FFN width; the fused pair gives 9728 (" +
                           std::to_string(im2) + ")");

    // engine#1756: the same shapes with OPAQUE tensor names, so the role filter cannot help. o_proj's K and q_proj's N are both
    // q_dim (4096); the old fused rule (gate_n.count(dk), halved) matched that pair and returned 2048 -- a wrong width, not a
    // refusal. The rule that pairs a down K with a gate N of twice it cannot be met by attention, so this must be 9728.
    const std::string opaque = pb::graph(
        {}, {mm_named("n0", "h", 2560, 4096), mm_named("n1", "q", 4096, 2560), mm_named("n2", "h", 2560, 2048),
             mm_named("n3", "h", 2560, 2048), mm_named("n4", "h", 2560, 19456), mm_named("n5", "a", 9728, 2560)});
    put_file((dir / "model2o.onnx").string(), pb::model({}, {pb::if_node("cond", "inputs_embeds", "logits", "else_branch", opaque)}));
    const int im2o = pi::intermediate_size_from_graph((dir / "model2o.onnx").string(), 2560);
    check(im2o == 9728, "im-from-node-attrs: with opaque names the fused pair still gives 9728, never q_dim/2 (" +
                            std::to_string(im2o) + ")");

    // engine#1756 (review of #1757): a FUSED qkv at a 2:1 GQA ratio has N = q_dim + 2*kv_dim = 2*q_dim, so with opaque
    // names it satisfies the fused rule against o_proj (K = q_dim) and offers q_dim as a candidate. Beside a fused gate_up
    // that is two candidates, and the size() != 1 guard must refuse rather than answer q_dim. Never q_dim.
    const std::string qkv21 = pb::graph(
        {}, {mm_named("m0", "h", 2560, 8192), mm_named("m1", "q", 4096, 2560),      // qkv: N = 4096 + 2*2048; o: K 4096
             mm_named("m2", "h", 2560, 19456), mm_named("m3", "a", 9728, 2560)});   // fused gate_up / down
    put_file((dir / "model2q.onnx").string(), pb::model({}, {pb::if_node("cond", "inputs_embeds", "logits", "else_branch", qkv21)}));
    const int im2q = pi::intermediate_size_from_graph((dir / "model2q.onnx").string(), 2560);
    check(im2q != 4096, "im-from-node-attrs: a 2:1 fused qkv beside a fused gate_up never yields q_dim (" +
                            std::to_string(im2q) + ")");
    check(im2q == 0, "im-from-node-attrs: ... it yields a refusal (two candidates), not a guess (" + std::to_string(im2q) + ")");

    // Attention only, wider than hidden, no FFN at all: 0, never q_dim.
    const std::string attnonly = pb::graph(
        {}, {mm_named(attn + "q_proj.MatMulNBits.qweight", "h", 2560, 4096),
             mm_named(attn + "o_proj.MatMulNBits.qweight", "q", 4096, 2560)});
    put_file((dir / "model3.onnx").string(), pb::model({}, {pb::if_node("cond", "inputs_embeds", "logits", "else_branch", attnonly)}));
    check(pi::intermediate_size_from_graph((dir / "model3.onnx").string(), 2560) == 0,
          "im-from-node-attrs: an attention-only graph wider than hidden yields 0, not the attention width");

    // Two widths that both pair up is a derivation that can be wrong in two ways: refused, not picked from.
    const std::string ambiguous = pb::graph(
        {}, {mm_named(mlp + "gate_proj.MatMulNBits.qweight", "h", 2560, 10240),
             mm_named(mlp + "up_proj.MatMulNBits.qweight", "h", 2560, 10240),
             mm_named(mlp + "down_proj.MatMulNBits.qweight", "a", 10240, 2560),
             mm_named("model.layers.1.mlp.gate_proj.MatMulNBits.qweight", "h2", 2560, 4096),
             mm_named("model.layers.1.mlp.up_proj.MatMulNBits.qweight", "h2", 2560, 4096),
             mm_named("model.layers.1.mlp.down_proj.MatMulNBits.qweight", "a2", 4096, 2560)});
    put_file((dir / "model4.onnx").string(), pb::model({}, {pb::if_node("cond", "inputs_embeds", "logits", "else_branch", ambiguous)}));
    check(pi::intermediate_size_from_graph((dir / "model4.onnx").string(), 2560) == 0,
          "im-from-node-attrs: two agreeing widths are refused rather than picked from");

    // An FFN with no down projection at all (attention-only graphs, the hybrid line's main graph) is 0, not a guess.
    const std::string noffn = pb::graph({}, {mm_named(attn + "q_proj.MatMulNBits.qweight", "h", 2560, 256),
                                             mm_named(attn + "o_proj.MatMulNBits.qweight", "q", 256, 2560)});
    put_file((dir / "model5.onnx").string(), pb::model({}, {pb::if_node("cond", "inputs_embeds", "logits", "else_branch", noffn)}));
    check(pi::intermediate_size_from_graph((dir / "model5.onnx").string(), 2560) == 0,
          "im-from-node-attrs: a graph with no FFN pair yields 0 (the caller keeps refusing)");
    fs::remove_all(dir);
}


// A multi-component pack (`type: compositemodel`) keeps its decoder graph AND its Token .fconst in the component
// `genai_config.json` names, while the files that state the geometry stay at the pack root. Pointing npu-pack at the
// root used to report "the pack carries no Token .fconst either" for a pack that has one two directories down (#1657).
// Measured on amd/gemma-4-e2b-it-mobius-int4_rai_1.9.0_npu: root has config/model_config/genai_config.json,
// decoder/ has model.onnx, model.bin, model.onnx.data and dd_metastate_Llm_Token_….fconst (1.34 GiB).
void test_fconst_in_component(const std::string& root) {
    const fs::path dir = fs::path(root) / "compfconst";
    fs::remove_all(dir);
    fs::create_directories(dir / "decoder");
    put_file((dir / "genai_config.json").string(),
             R"({"model":{"type":"gemma4","decoder":{"filename":"decoder/model.onnx","hidden_size":1536}}})");
    put_file((dir / "decoder" / "model.onnx").string(), pb::model({}, {}));
    put_file((dir / "decoder" / "dd_metastate_Llm_Token_aie2_token_decoder.fconst").string(), std::string(64, '\x11'));
    const std::string found = pi::find_token_fconst_exported(dir.string());
    std::printf("component fconst: found %s\n", found.empty() ? "(nothing)" : found.c_str());
    check(!found.empty() && found.find("decoder") != std::string::npos,
          "component fconst: the Token .fconst is found where genai_config.json says the decoder is (" + found + ")");
    // a flat pack is unchanged: the same filename at the root still wins
    const fs::path flat = fs::path(root) / "flatfconst";
    fs::remove_all(flat);
    fs::create_directories(flat);
    put_file((flat / "dd_metastate_Llm_Token_Token_layer0.fconst").string(), std::string(64, '\x22'));
    const std::string flat_found = pi::find_token_fconst_exported(flat.string());
    check(!flat_found.empty() && flat_found.find("Token_Token_") != std::string::npos,
          "component fconst: a flat pack's root fconst is still found first (" + flat_found + ")");
    fs::remove_all(dir);
    fs::remove_all(flat);
}


// A multi-component pack names its embedding component in genai_config.json (`model.embedding.filename =
// "embedding/model.onnx"`) and keeps the external data beside it, while a single-graph pack uses the flat
// `embedding.onnx` this reader was written for. Measured on amd/gemma-4-e2b-it-mobius-int4_rai_1.9.0_npu: the component
// is `embedding/model.onnx` + `embedding/model.onnx.data` (10.26 GiB), and the reader reported "no embedding.onnx".
void test_embedding_in_component(const std::string& root) {
    const fs::path dir = fs::path(root) / "embedcomp2";
    fs::remove_all(dir);
    fs::create_directories(dir / "embedding");
    const int64_t V = 8, H = 256, groups = H / 128;
    std::vector<uint8_t> table(size_t(V) * size_t(H / 2), 0);
    std::vector<float> scales(size_t(V) * size_t(groups), 0.5f);
    for (size_t i = 0; i < table.size(); ++i) table[i] = uint8_t(i * 13 % 256);
    std::string data;
    data.append(reinterpret_cast<const char*>(table.data()), table.size());
    const size_t table_bytes = table.size();
    data.append(reinterpret_cast<const char*>(scales.data()), scales.size() * sizeof(float));
    put_file((dir / "embedding" / "model.onnx.data").string(), data);
    // the location is BARE -- relative to the graph, not to the pack root
    const auto t = pb::tensor("language_model.embed_tokens.weight_Q4", {V, H / 2}, "model.onnx.data", 0, table_bytes, 2);
    const auto sc = pb::tensor("language_model.embed_tokens.weight_scales", {V, groups}, "model.onnx.data",
                               table_bytes, scales.size() * sizeof(float), 1);
    put_file((dir / "embedding" / "model.onnx").string(), pb::model({t, sc}, {}));
    put_file((dir / "genai_config.json").string(),
             R"({"model":{"type":"gemma4","embedding":{"filename":"embedding/model.onnx"},)"
             R"("decoder":{"filename":"decoder/model.onnx"}}})");
    std::string why;
    const std::vector<float> got = pi::read_embedding_component(dir.string(), &why);
    std::printf("embedding component (declared): %zu values (why: %s)\n", got.size(), why.c_str());
    check(got.size() == size_t(V) * size_t(H),
          "embedding component: the graph genai_config.json names is read, with its data beside it ("
              + std::to_string(got.size()) + ")");
    fs::remove_all(dir);
}


// engine#1725: the gemma-4 component is f32 and unquantised -- an `embed_tokens` table with no scales beside it -- and its
// decoder also needs a PER-LAYER table. Two arms: the f32 token table alone is read, and a per-layer table is refused by
// name rather than reported as a missing table. Measured sizes on amd/gemma-4-e2b-it-mobius-int4_rai_1.9.0_npu:
// embed_tokens.weight (262144,1536) f32 and embed_tokens_per_layer.weight (262144,8960) f32.
void test_embedding_f32_component(const std::string& root) {
    auto build = [&](const fs::path& dir, bool with_per_layer) {
        fs::remove_all(dir);
        fs::create_directories(dir / "embedding");
        const int64_t V = 8, H = 256;
        std::vector<float> tok(size_t(V) * size_t(H));
        for (size_t i = 0; i < tok.size(); ++i) tok[i] = float(int(i % 97) - 48) * 0.25f;
        std::string data(reinterpret_cast<const char*>(tok.data()), tok.size() * sizeof(float));
        const size_t tok_bytes = data.size();
        std::vector<std::string> inits{pb::tensor("embedding.embed_tokens.weight", {V, H}, "model.onnx.data", 0,
                                                  tok_bytes, 1)};
        if (with_per_layer) {
            std::vector<float> per(size_t(V) * size_t(320));
            for (size_t i = 0; i < per.size(); ++i) per[i] = 0.5f;
            data.append(reinterpret_cast<const char*>(per.data()), per.size() * sizeof(float));
            inits.push_back(pb::tensor("embedding.embed_tokens_per_layer.weight", {V, 320}, "model.onnx.data", tok_bytes,
                                       per.size() * sizeof(float), 1));
        }
        put_file((dir / "embedding" / "model.onnx.data").string(), data);
        put_file((dir / "embedding" / "model.onnx").string(), pb::model(inits, {}));
        put_file((dir / "genai_config.json").string(),
                 R"({"model":{"type":"gemma4","embedding":{"filename":"embedding/model.onnx"}}})");
    };
    const fs::path a = fs::path(root) / "f32emb";
    build(a, false);
    std::string why;
    const std::vector<float> got = pi::read_embedding_component(a.string(), &why);
    std::printf("f32 embedding: %zu values (why: %s)\n", got.size(), why.c_str());
    check(got.size() == 8u * 256u, "f32 embedding: an unquantised f32 table is read (" + std::to_string(got.size()) + ")");

    const fs::path b = fs::path(root) / "f32emb_perlayer";
    build(b, true);
    std::string why2;
    const std::vector<float> none = pi::read_embedding_component(b.string(), &why2);
    std::printf("f32 embedding (per-layer): %zu values, why: %s\n", none.size(), why2.c_str());
    check(!none.empty(),
          "f32 embedding: a per-layer table no longer blocks the TOKEN table (slice 2 handles it; " + why2 + ")");
    fs::remove_all(a);
    fs::remove_all(b);
}

// engine#1786 (item F1): the container genre's geometry, pinned as a pure function over the three header shapes the
// census measured on this box -- no pack, no device, no file, which is what makes the two genres with no config.json
// (`d3-edge`, `d3-flash`) pinnable at all.
void test_container_geometry() {
    using nlohmann::json;
    // 1. self-describing: `dims` answers everything, and config.json must NOT be preferred over it.
    {
        json meta = {{"dims", {{"hidden_size", 576}, {"intermediate_size", 1536}, {"num_hidden_layers", 30},
                               {"num_attention_heads", 9}, {"num_key_value_heads", 3}, {"vocab_size", 49152},
                               {"head_dim", 64}, {"rope_theta", 10000.0}, {"model_type", "llama"}}}};
        json cfg = {{"hidden_size", 1024}};   // a deliberate disagreement: the container's own statement wins
        const auto g = pi::container_geometry(meta, cfg);
        check(g.source == "dims", "self-describing: source is dims");
        check(g.dims.hidden == 576, "self-describing: hidden comes from dims, not config.json");
        check(g.dims.layers == 30 && g.dims.heads == 9 && g.dims.kv_heads == 3 && g.dims.vocab == 49152,
              "self-describing: every field dims states is taken");
        check(g.dims.head_dim == 64, "self-describing: a stated head_dim is used as stated");
        check(g.missing.empty(), "self-describing: nothing is missing");
        check(g.inferred.empty(), "self-describing: nothing is inferred");
    }
    // 2. gguf-converted: no dims, so config.json answers and the source says so.
    {
        json meta = {{"converter", "tools/gguf_to_q4nx.py"}, {"format", "q4nx"}, {"source", "x.gguf"}};
        json cfg = {{"hidden_size", 1024}, {"num_hidden_layers", 28}, {"num_attention_heads", 16},
                    {"num_key_value_heads", 8}, {"vocab_size", 151936}, {"intermediate_size", 3072},
                    {"head_dim", 64}, {"rope_theta", 1000000.0}};
        const auto g = pi::container_geometry(meta, cfg);
        check(g.source == "config.json", "gguf-converted: source is config.json");
        check(g.dims.hidden == 1024 && g.dims.vocab == 151936, "gguf-converted: config.json supplies the geometry");
        check(g.missing.empty(), "gguf-converted: nothing is missing when config.json is complete");
    }
    // 3. silent: neither answers, so every field is named missing rather than left as a zero that reads like a value.
    {
        const auto g = pi::container_geometry(json::object(), json());
        check(g.source.empty(), "silent: no source is claimed");
        // eight, and named: the first run of this arm asserted seven and failed in CI -- rope_theta is a required
        // field of the arithmetic (it is the RoPE base the forward needs), so the count was wrong and the function
        // was right. Asserted by name now, so the next field added to Dims fails this arm instead of shrinking it.
        const std::vector<std::string> want = {"hidden_size",         "num_hidden_layers",   "num_attention_heads",
                                               "num_key_value_heads", "vocab_size",         "intermediate_size",
                                               "head_dim",            "rope_theta"};
        check(g.missing.size() == want.size(), "silent: every required field is named missing");
        bool all_named = true;
        for (const auto& w : want)
            all_named = all_named && std::find(g.missing.begin(), g.missing.end(), w) != g.missing.end();
        check(all_named, "silent: the missing list names each required field");
        check(g.dims.hidden == 0 && g.dims.layers == 0, "silent: no field is invented");
    }
    // 4. head_dim absent but derivable: reported as INFERRED, not as declared.
    {
        json meta = {{"dims", {{"hidden_size", 2048}, {"intermediate_size", 6144}, {"num_hidden_layers", 28},
                               {"num_attention_heads", 16}, {"num_key_value_heads", 8}, {"vocab_size", 151936},
                               {"rope_theta", 1000000.0}}}};
        const auto g = pi::container_geometry(meta, json());
        check(g.dims.head_dim == 128, "inferred: head_dim is hidden/heads");
        check(g.inferred.size() == 1 && g.inferred[0] == "head_dim", "inferred: the inference is named");
        check(g.missing.size() == 0, "inferred: an inferred field is not also missing");
    }
}

// engine#1786 (item F1), slice 2: `plan()` reports WHERE a container's geometry came from. The three genres the census
// measured are the three cases: `dims` answers (and beats a disagreeing config.json), config.json answers when there is
// no `dims`, and neither answering is reported as unknown rather than as a zero geometry.
void test_container_plan(const std::string& root) {
    namespace fs = std::filesystem;
    // Under ctest `root` is the source tree's tests/golden, so these three directories are removed on every way out:
    // the arm is new, and leaving fixtures behind is the defect #2024 fixes for its neighbour -- not repeating it here.
    struct Cleanup {
        fs::path base;
        ~Cleanup() {
            for (const char* n : {"cg-dims", "cg-cfg", "cg-silent"}) {
                std::error_code e;
                fs::remove_all(base / n, e);
            }
        }
    } cleanup{fs::path(root)};
    const auto write_pack = [&](const std::string& name, const std::string& meta_json, const std::string& config) {
        const fs::path dir = fs::path(root) / name;
        std::error_code ec;
        fs::remove_all(dir, ec);
        fs::create_directories(dir, ec);
        std::string hdr = "{";
        if (!meta_json.empty()) hdr += "\"__metadata__\":" + meta_json + ",";
        hdr += "\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[0,16]}}";
        std::string out(8, '\0');
        const int64_t n = int64_t(hdr.size());
        for (int i = 0; i < 8; ++i) out[size_t(i)] = char((n >> (8 * i)) & 0xFF);
        out += hdr;
        out.append(16, '\x22');
        std::ofstream(dir / "model.q4nx", std::ios::binary).write(out.data(), std::streamsize(out.size()));
        if (!config.empty()) std::ofstream(dir / "config.json") << config;
        return dir.string();
    };
    const std::string dims = "{\"dims\":{\"hidden_size\":576,\"intermediate_size\":1536,\"num_hidden_layers\":30,"
                             "\"num_attention_heads\":9,\"num_key_value_heads\":3,\"vocab_size\":49152,"
                             "\"head_dim\":64,\"rope_theta\":10000.0,\"model_type\":\"llama\"}}";

    // 1. self-describing, and a config.json that disagrees: the container's own statement wins and is named.
    {
        const auto pl = pi::plan(write_pack("cg-dims", dims, "{\"hidden_size\": 1024}"));
        check(pl.container_geometry_source == "dims", "plan: a self-describing container reports source=dims");
        check(pl.dims.hidden == 576, "plan: dims.hidden comes from the container, not the disagreeing config.json");
        check(pl.dims.layers == 30 && pl.dims.vocab == 49152, "plan: the container's geometry reaches the plan");
        check(pl.container_geometry_missing.empty(), "plan: nothing is reported missing for a complete container");
        check(pl.container_note.find("geometry from dims") != std::string::npos,
              "plan: the note says which source answered");
    }
    // 2. gguf-converted: no dims, so config.json answers -- and the source says so.
    {
        const auto pl = pi::plan(write_pack("cg-cfg", "{\"converter\":\"tools/gguf_to_q4nx.py\"}",
                                           "{\"hidden_size\":1024,\"num_hidden_layers\":28,"
                                           "\"num_attention_heads\":16,\"num_key_value_heads\":8,"
                                           "\"vocab_size\":151936,\"intermediate_size\":3072,"
                                           "\"head_dim\":64,\"rope_theta\":1000000.0}"));
        check(pl.container_geometry_source == "config.json", "plan: a converted container reports source=config.json");
        check(pl.dims.hidden == 1024 && pl.dims.layers == 28, "plan: config.json supplies the geometry");
    }
    // 3. silent: neither answers, so the plan says the geometry is unknown instead of reporting zeros as facts.
    {
        const auto pl = pi::plan(write_pack("cg-silent", "", ""));
        check(pl.container_geometry_source.empty(), "plan: the silent genre claims no source");
        check(pl.container_geometry_missing.size() == 8, "plan: every required field is reported missing, by name");
        check(pl.container_note.find("geometry is not known here") != std::string::npos,
              "plan: the note says the geometry is unknown rather than printing a zero geometry");
    }
}

// engine#1786 (item F1), slice 3a: roles from names alone. The cases are the three the census measured -- a transformer
// pack that classifies completely, the alternate spellings that must classify the same way, and the two residues (SSM
// and MTP) which must be RECOGNISED rather than reported as unknown, since `unassigned` is only useful if it means
// "nothing knows this".
// engine#1788, slice 3c step 1: (role, layer type), the key the writer needs. Measured on gemma4-e2b (q_proj N=384 on all
// 28 sliding layers, N=768 on all 7 full) and bolt-e4b ((640,5120) on 35 sliding, (1280,5120) on 7 full).
void test_container_role_layer_types() {
    const auto t = [](const std::string& n, std::vector<int64_t> shape, const char* dt = "I8") {
        pi::Q4nxTensor x; x.name = n; x.dtype = dt; x.shape = std::move(shape); return x;
    };
    // 2 sliding layers + 1 full layer, q_proj with the sliding/full widths that the real packs show
    const std::vector<std::string> lt = {"sliding_attention", "sliding_attention", "full_attention"};
    const std::vector<pi::Q4nxTensor> v = {
        t("model.layers.0.self_attn.q_proj.weight", {384, 8, 4736}),
        t("model.layers.1.self_attn.q_proj.weight", {384, 8, 4736}),
        t("model.layers.2.self_attn.q_proj.weight", {768, 8, 4736}),
        t("model.embed_tokens.weight", {248144, 2048}, "BF16")};
    const pi::RoleLayerReport r = pi::container_role_layer_types(v, lt);
    check(r.spans.empty(), "role+layer: the two widths split by layer type, they do not span");
    bool sliding = false, full = false;
    for (const auto& row : r.rows)
        if (row.stem == "model.layers.N.self_attn.q_proj.weight" && row.layer_type == "sliding_attention") {
            sliding = row.count == 2 && row.shape == "[384,8,4736] I8";
        } else if (row.stem == "model.layers.N.self_attn.q_proj.weight" && row.layer_type == "full_attention") {
            full = row.count == 1 && row.shape == "[768,8,4736] I8";
        }
    check(sliding && full, "role+layer: one row per (role, layer type) with its shape and count");
    check(!r.no_layer.empty(), "role+layer: a role outside any layer is reported, not dropped");

    // a role carried in two shapes under ONE layer type is the case a third key would be needed for: reported, not resolved
    const std::vector<pi::Q4nxTensor> two = {t("model.layers.0.self_attn.q_proj.weight", {384, 8, 4736}),
                                             t("model.layers.1.self_attn.q_proj.weight", {512, 8, 4736})};
    const pi::RoleLayerReport r2 = pi::container_role_layer_types(two, lt);
    check(r2.spans.size() == 1 && r2.rows.empty(), "role+layer: two shapes in one layer type is a span, not a row");

    // a pack whose layer count disagrees with its declared layer_types is counted, not silently clipped
    const std::vector<pi::Q4nxTensor> oor = {t("model.layers.9.self_attn.q_proj.weight", {384, 8, 4736})};
    check(pi::container_role_layer_types(oor, lt).out_of_range == 1,
          "role+layer: a layer index past layer_types is counted as out of range");
}

// engine#1788, 3c: the writer iterates TENSORS, not roles, so the placement has to answer per tensor and use that
// tensor's own layer type. The arm that matters is the pair: the same role lands in two different places on two layers.
// engine#1788, 3c: the same decision driven by a REAL container. The pack leg gates on model.onnx, so a container
// (model.q4nx) could never reach the placement above; this reads the container with the shipped reader and reports the
// three counts, so the numbers in a PR body are measured on the five containers rather than on the synthetic fixture.
static void test_container_placement_real(const std::string& dir, const std::vector<std::string>& layer_types) {
    using namespace onebit::npu::pack_import;
    const Q4nxContainer c = read_q4nx_container(dir + "/model.q4nx");
    const PlacementReport r = container_placement(c.tensors, layer_types);
    std::printf("placement: %s -> put %lld, held %lld, unnamed %lld (of %zu tensors)\n", dir.c_str(),
                (long long)r.placed, (long long)r.unplaced, (long long)r.no_role, c.tensors.size());
    std::map<std::string, int64_t> held_by_reason;
    for (const auto& row : r.rows) if (row.slot.empty()) held_by_reason[row.why]++;
    for (const auto& kv : held_by_reason) std::printf("    held: %lld x %s\n", (long long)kv.second, kv.first.c_str());
    std::map<std::string, int64_t> roles_seen;
    for (const auto& row : r.rows) if (!row.role.empty()) roles_seen[row.role]++;
    // REVIEW of #2291 direction: the engine documents a rank and a dtype per slot (model.h), and both are
    // architecture-independent -- a router that is 3-D or an expert that is BF16 is wrong for every model.
    {
        // Two source tensors reaching one destination is silent data loss: the pack carries one, the forward reads the
        // other. This check is what caught the placement being non-writable (733 tensors / 26 names) and then all three
        // mapping classes: a `.scale` companion filed as its weight, per_layer_proj_norm filed as the final norm, and the
        // 1B's `.bf16.` copies reaching the untagged slot.
        std::map<std::string, int> per_slot;
        for (const auto& row : r.rows) if (!row.slot.empty()) per_slot[row.slot]++;
        std::vector<std::string> dup;
        for (const auto& kv : per_slot) if (kv.second > 1) dup.push_back(kv.first);
        std::string dl;
        for (const auto& d : dup) dl += d + " ";
        check(per_slot.size() == size_t(r.placed),
              "placement: every placed tensor has its own destination in " + dir + " (" + dl + ")");
    }
    {
        std::map<std::string, std::string> seen;
        for (const auto& row : r.rows) {
            if (row.slot.empty() || seen.count(row.slot)) continue;
            std::string sh = "[";
            for (size_t i = 0; i < row.shape.size(); ++i) sh += (i ? "," : "") + std::to_string(row.shape[i]);
            sh += "]";
            seen[row.slot] = sh + " " + row.dtype;
            std::printf("    slot %-42s %s %s\n", row.slot.c_str(), sh.c_str(), row.dtype.c_str());
        }
    }
    // The rank and dtype are architecture-independent even though the extents are not: model.h documents the router as a
    // 2-D BF16 tensor and the expert stacks as 3-D I8. A placement that reached the right NAME with the wrong shape would
    // load a tensor the forward reads as something else, so these are asserted per placed row on the real container.
    for (const auto& row : r.rows) {
        if (row.slot.empty()) continue;
        if (row.slot == "moe_router.weight") {
            check(row.shape.size() == 2, "placement: the router is 2-D in " + dir);
            check(row.dtype == "BF16", "placement: and BF16 in " + dir);
        }
        if (row.slot.find("exps_proj") != std::string::npos) {
            check(row.shape.size() == 3, "placement: an expert stack is 3-D (" + row.slot + ") in " + dir);
            check(row.dtype == "I8", "placement: and I8 (" + row.slot + ") in " + dir);
        }
    }
    std::printf("    roles:"); for (const auto& kv : roles_seen) std::printf(" %s=%lld", kv.first.c_str(), (long long)kv.second);
    std::printf("\n");
    std::printf("    UNKNOWN:");
    for (const auto& kv : held_by_reason) if (kv.first.find("does not know") != std::string::npos) {
        for (const auto& row : r.rows) if (row.slot.empty() && row.why == kv.first && !row.role.empty()) {
            std::printf(" %s", row.role.c_str()); break; }
    }
    std::printf("\n");
    check(r.placed + r.unplaced + r.no_role == int64_t(c.tensors.size()), std::string("placement: accounts for every tensor in ") + dir);
    // REVIEW of #2291: the result this PR exists for -- an empty unknown bucket on a real container -- was printed and
    // never checked, so a removed refusal entry still exited 0. Now it fails where it happens, with the role named.
    std::set<std::string> unknown;
    for (const auto& row : r.rows)
        if (!row.slot.empty() || row.role.empty()) continue;
        else if (row.why.find("does not know") != std::string::npos) unknown.insert(row.role);
    std::string names;
    for (const auto& u : unknown) names += (names.empty() ? "" : ", ") + u;
    check(unknown.empty(), std::string("placement: no role is unknown to the table in ") + dir + " (" + names + ")");
}

// REVIEW of #2291 named a hazard that review caught and code should catch: during one edit this file briefly held two
// role->destination tables with the SAME keys and DIFFERENT values, and container_role_slot() would have returned
// whichever came first. The observable invariant is that no role is both placed and refused -- asserted here once, over
// the whole report, so a duplicated or contradictory entry fails without needing a real container.
static void test_role_tables_disagree_nowhere() {
    using namespace onebit::npu::pack_import;
    ContainerRoles r;
    for (const char* role : {"embed", "final_norm", "input_norm", "post_norm", "lm_head", "q_proj", "k_proj", "v_proj",
                             "o_proj", "q_norm", "k_norm", "self_attn_gate_proj", "gate_proj", "up_proj", "down_proj",
                             "gate_up_proj", "qkv_proj", "moe_router", "gate_exps", "up_exps", "down_exps",
                             "share_gate_exps", "share_up_exps", "share_down_exps", "shared_expert_gate",
                             "shortconv_in_proj", "ssm", "gemma_local", "container_extra"})
        r.counts[role] = 1;
    const RoleSlotReport s = container_role_slots(r);
    for (const auto& kv : s.slot_of) {
        check(!s.why.count(kv.first), "tables: " + kv.first + " is not both placed and refused");
    }
    check(s.placed.size() + s.no_slot.size() == r.counts.size(), "tables: every role is in exactly one bucket");
    // the MoE destinations must be the CONTAINER spelling; the EngineConverter one is a different pack format
    check(s.slot_of.count("gate_exps") && s.slot_of.at("gate_exps") == "mlp.gate_exps_proj.weight",
          "tables: the routed gate is the FLM spelling, not mlp.experts.*");
    check(s.slot_of.count("moe_router") && s.slot_of.at("moe_router") == "moe_router.weight",
          "tables: the router is moe_router.weight, not mlp.gate.weight");
    // two refusals are correct and both are about the SECOND KEY, not about a missing slot: qkv_proj needs the layer type,
    // gate_up_proj needs a measured split. Anything else appearing here is a regression, so the set is pinned exactly.
    // Five refusals, and the mix is the point: two are about the SECOND KEY (qkv_proj needs the layer type,
    // gate_up_proj needs a measured split) and three are GROUP roles (ssm, gemma_local, container_extra) which have no
    // single destination at role granularity even though their members are placed per tensor by suffix. Role-level
    // refusal plus per-tensor placement are two different answers, and both are correct.
    { std::string list; for (const auto& r : s.no_slot) list += r + " ";
      check(s.no_slot.size() == 5, "tables: five roles are refused at role granularity (got: " + list + ")"); }
    // REVIEW of #2291, applied to myself: the real-container leg arms the table only where a pack is given, and CI runs
    // with NO pack -- so before this, a wrong destination for any single role (o_proj, up_exps, ssm_conv1d …) was
    // invisible in CI. Every layer-type-independent destination is pinned here, exactly.
    {
        const std::map<std::string, std::string> want = {
            {"embed", "model.embed_tokens.weight"},        {"final_norm", "model.norm.weight"},
            {"input_norm", "input_layernorm.weight"},      {"post_norm", "post_attention_layernorm.weight"},
            {"lm_head", "lm_head.weight"},                 {"q_proj", "self_attn.q_proj.weight"},
            {"k_proj", "self_attn.k_proj.weight"},         {"v_proj", "self_attn.v_proj.weight"},
            {"o_proj", "self_attn.o_proj.weight"},         {"q_norm", "self_attn.q_norm.weight"},
            {"k_norm", "self_attn.k_norm.weight"},         {"self_attn_gate_proj", "self_attn.gate_proj.weight"},
            {"gate_proj", "mlp.gate_proj.weight"},         {"up_proj", "mlp.up_proj.weight"},
            {"down_proj", "mlp.down_proj.weight"},         {"gate_exps", "mlp.gate_exps_proj.weight"},
            {"up_exps", "mlp.up_exps_proj.weight"},        {"down_exps", "mlp.down_exps_proj.weight"},
            {"shortconv_in_proj", "shortconv.in_proj.weight"}, {"per_layer_proj_norm", "model.per_layer_proj_norm.weight"},
            {"moe_router", "moe_router.weight"},           {"share_gate_exps", "mlp.share_gate_exps_proj.weight"},
            {"share_up_exps", "mlp.share_up_exps_proj.weight"}, {"share_down_exps", "mlp.share_down_exps_proj.weight"},
            {"shared_expert_gate", "mlp.share_router.weight"},
        };
        for (const auto& kv : want) {
            check(s.slot_of.count(kv.first) && s.slot_of.at(kv.first) == kv.second,
                  "tables: " + kv.first + " -> " + kv.second + " (got: " +
                      (s.slot_of.count(kv.first) ? s.slot_of.at(kv.first) : std::string("<absent>")) + ")");
        }
    }
    check(s.no_slot[0] == "container_extra" && s.no_slot[1] == "gate_up_proj" && s.no_slot[2] == "gemma_local" &&
          s.no_slot[3] == "qkv_proj" && s.no_slot[4] == "ssm",
          "tables: and they are exactly the pinned five, so a new refusal cannot slip in unnoticed");
}

// REVIEW of #2291: the `.scale` branch and its ORDERING were both unarmed -- deleting the branch left every arm green,
// even though its own comment says the cost is a pack that writes scales over the embedding with no error. These arms
// fail if the branch is deleted AND if it is moved inside the needle loop, which is the refactor that would restore the
// defect silently.
static void test_scale_and_proj_norm_roles() {
    using pi::Q4nxTensor;
    const auto t = [](const std::string& n) { Q4nxTensor x; x.name = n; x.dtype = "BF16"; return x; };

    // 1. the classifier: a .scale companion is its own role, not the weight it accompanies
    {
        const std::vector<Q4nxTensor> v = {t("model.embed_tokens.weight"), t("model.embed_tokens.weight.scale")};
        const pi::ContainerRoles r = pi::container_roles(v);
        check(r.role_of.at("model.embed_tokens.weight") == "embed", "scale: the embedding is still the embedding");
        check(r.role_of.at("model.embed_tokens.weight.scale") == "weight_scale",
              "scale: the .scale companion is NOT filed as embed");
    }
    // 2. the destination: identity, and it is a literal the forward reads
    {
        const std::vector<Q4nxTensor> v = {t("model.embed_tokens.weight"), t("model.embed_tokens.weight.scale")};
        const pi::PlacementReport p = pi::container_placement(v, {});
        check(p.rows[1].slot == "model.embed_tokens.weight.scale",
              "scale: its destination is its OWN name, which q4nx_forward reads");
        check(p.rows[0].slot != p.rows[1].slot, "scale: and it does not collide with the embedding");
    }
    // 3. the ordering, as a negative: nothing ending .scale may be placed under a WEIGHT role's slot
    {
        const std::vector<Q4nxTensor> v = {t("model.layers.0.mlp.down_proj.weight"),
                                           t("model.layers.0.mlp.down_proj.weight.scale")};
        const pi::PlacementReport p = pi::container_placement(v, {"full_attention"});
        check(p.rows[1].role == "weight_scale", "scale: a per-layer companion is weight_scale too");
        check(p.rows[1].slot == "model.layers.0.mlp.down_proj.weight.scale",
              "scale: identity, so it cannot land in the down_proj slot");
        check(p.rows[0].slot == "model.layers.0.mlp.down_proj.weight", "scale: and the weight keeps its own slot");
    }
    // 4. per_layer_proj_norm is not the model's final norm (same class, one level simpler)
    {
        const std::vector<Q4nxTensor> v = {t("model.norm.weight"), t("model.per_layer_proj_norm.weight")};
        const pi::ContainerRoles r = pi::container_roles(v);
        check(r.role_of.at("model.norm.weight") == "final_norm", "proj_norm: model.norm is the final norm");
        check(r.role_of.at("model.per_layer_proj_norm.weight") == "per_layer_proj_norm",
              "proj_norm: per_layer_proj_norm is NOT the final norm");
        const pi::PlacementReport p = pi::container_placement(v, {});
        check(p.rows[1].slot == "model.per_layer_proj_norm.weight", "proj_norm: and it has its own top-level slot");
    }
}

void test_container_placement() {
    using pi::Q4nxTensor;
    const auto t = [](const std::string& n) { Q4nxTensor x; x.name = n; x.dtype = "I8"; return x; };
    const std::vector<Q4nxTensor> v = {
        t("model.embed_tokens.weight"),
        t("model.layers.0.self_attn.qkv_proj.weight"),   // layer 0 is linear attention
        t("model.layers.1.self_attn.qkv_proj.weight"),   // layer 1 is full attention
        t("model.layers.1.self_attn.q_proj.weight"),
        t("model.layers.0.mlp.gate_up_proj.weight"),     // a role this slice still refuses
        t("model.layers.1.unknown_thing.weight"),        // no role at all
    };
    const std::vector<std::string> lt = {"linear_attention", "full_attention"};
    const pi::PlacementReport r = pi::container_placement(v, lt);

    check(r.rows.size() == v.size(), "placement: a row per tensor, so nothing is silently skipped");
    check(r.placed + r.unplaced + r.no_role == int64_t(v.size()),
          "placement: placed + unplaced + no-role accounts for every tensor");
    check(r.rows[0].slot == "model.embed_tokens.weight", "placement: a tensor with no layer still gets a slot");
    check(r.rows[0].layer.empty() && r.rows[0].layer_type.empty(), "placement: and reports that it has no layer");
    check(r.rows[1].slot == "model.layers.0.linear_attn.qkv_proj.weight", "placement: the fused qkv on the LINEAR layer, layer-qualified");
    check(r.rows[2].slot.empty(), "placement: the same role on the FULL-attention layer has no destination");
    check(r.rows[2].why.size() > 10, "placement: and says why");
    check(r.rows[3].slot == "model.layers.1.self_attn.q_proj.weight", "placement: the query projection is layer-type-independent, layer-qualified");
    check(r.rows[4].slot.empty() && r.rows[4].role == "gate_up_proj", "placement: gate_up_proj is still refused");
    check(r.rows[5].slot.empty() && r.rows[5].role.empty(), "placement: a tensor with no role is counted apart");
    check(r.no_role == 1, "placement: no_role counts the vocabulary gap, not the deliberate refusal");
    check(r.unplaced == 2, "placement: unplaced counts the two decisions, not the vocabulary gap");
    {
        // REVIEW of #2291: the containers' whole role vocabulary, no pack needed, because CI runs this binary with no
        // pack. A role that reaches the "this table does not know" default, or a reason that is a bare placeholder, is a
        // hole -- and this arm is what makes removing a refusal entry fail in CI rather than print UNKNOWN on the box.
        pi::ContainerRoles roles_c;
        for (const char* role : {"ssm", "gemma_local", "container_extra", "gate_up_proj", "dtype_variant"})
            roles_c.counts[role] = 1;
        const pi::RoleSlotReport slots_c = pi::container_role_slots(roles_c);
        for (const auto& kv : slots_c.why) {
            check(kv.second.find("does not know") == std::string::npos,
                  "slots: the table explains " + kv.first + " rather than defaulting it");
            check(kv.second.rfind("same", 0) != 0 && kv.second != "same",
                  "slots: and the reason for " + kv.first + " is a reason, not a placeholder");
            check(kv.second.size() > 20, "slots: and it is long enough to be one (" + kv.first + ")");
        }
    }
    {
        // the group roles: the destination is the tensor's own suffix, and only the members the engine has a slot for move
        const std::vector<Q4nxTensor> g = {
            t("model.layers.0.pre_feedforward_layernorm.weight"),
            t("model.layers.0.post_feedforward_layernorm.weight"),
            t("model.layers.0.inp_gate.weight"),
        };
        const pi::PlacementReport gr = pi::container_placement(g, lt);
        check(gr.rows[0].role == "gemma_local", "placement: the gemma group is one role");
        check(gr.rows[0].slot == "model.layers.0.pre_feedforward_layernorm.weight", "placement: the suffix IS the destination (pre), layer-qualified");
        check(gr.rows[1].slot == "model.layers.0.post_feedforward_layernorm.weight", "placement: and for post, layer-qualified");
        check(gr.rows[2].slot.empty(), "placement: a member with no engine slot stays unplaced");
        check(gr.placed == 2 && gr.unplaced == 1, "placement: two placed, one held, for the three members");
    }
}

void test_container_roles_moe_and_singular_layer() {
    const auto t = [](const std::string& n) { pi::Q4nxTensor x; x.name = n; x.dtype = "I8"; return x; };
    const auto roles = [&](std::initializer_list<const char*> names) {
        std::vector<pi::Q4nxTensor> v;
        for (const char* n : names) v.push_back(t(n));
        return pi::container_roles(v);
    };

    // 1a. engine#2277 review: an ATTENTION output gate is not the FFN gate, and this is the arm that says so --
    // without the `self_attn_gate_proj` rule (main's behaviour) the first of these is filed as `gate_proj` and the
    // suite stayed green, which is what the review found.
    {
        const auto r = roles({"model.layer.0.self_attn.gate_proj.weight", "model.layer.0.mlp.gate_proj.weight"});
        check(r.role_of.at("model.layer.0.self_attn.gate_proj.weight") == "self_attn_gate_proj",
              "35b: an attention output gate is not the FFN gate");
        check(r.role_of.at("model.layer.0.mlp.gate_proj.weight") == "gate_proj", "35b: the FFN gate is still gate_proj");
        check(r.counts.at("gate_proj") == 1, "35b: the attention gate does not also count as gate_proj");
    }
    // 1. the eight MoE families, spelled as the container spells them
    {
        const auto r = roles({"model.layer.0.moe_router.weight",
                              "model.layer.0.shared_expert_gate.weight",
                              "model.layer.0.mlp.gate_exps_proj.weight",
                              "model.layer.0.mlp.up_exps_proj.weight",
                              "model.layer.0.mlp.down_exps_proj.weight",
                              "model.layer.0.mlp.share_gate_exps_proj.weight",
                              "model.layer.0.mlp.share_up_exps_proj.weight",
                              "model.layer.0.mlp.share_down_exps_proj.weight"});
        check(r.unassigned.empty(), "35b: the MoE families are named, not unassigned");
        check(r.role_of.at("model.layer.0.moe_router.weight") == "moe_router", "35b: moe_router");
        check(r.role_of.at("model.layer.0.shared_expert_gate.weight") == "shared_expert_gate",
              "35b: shared_expert_gate is its own role, not a projection");
        check(r.role_of.at("model.layer.0.mlp.gate_exps_proj.weight") == "gate_exps", "35b: gate_exps_proj");
        check(r.role_of.at("model.layer.0.mlp.share_gate_exps_proj.weight") == "share_gate_exps",
              "35b: share_gate_exps_proj is the shared expert, not the routed gate");
        check(r.counts.at("gate_exps") == 1 && r.counts.at("share_gate_exps") == 1,
              "35b: the shared spelling does not also count as the routed one");
    }

    // 2. a per-layer norm under the singular block spelling stays a layer norm: it must NOT become final_norm
    {
        const auto r = roles({"model.norm.weight", "model.layer.0.linear_attn.ssm_norm.weight",
                              "model.layer.0.input_layernorm.weight"});
        check(r.role_of.at("model.norm.weight") == "final_norm", "35b: the top-level norm is final_norm");
        check(r.counts.at("final_norm") == 1, "35b: exactly one final_norm, even with `model.layer.N.` blocks");
        check(r.role_of.at("model.layer.0.linear_attn.ssm_norm.weight") == "ssm",
              "35b: a per-layer ssm_norm is ssm, which is what kSsm says it is");
    }
}

void test_container_roles_gemma_local() {
    const auto t = [](const std::string& n) { pi::Q4nxTensor x; x.name = n; x.dtype = "BF16"; return x; };
    const auto roles = [&](std::initializer_list<const char*> names) {
        std::vector<pi::Q4nxTensor> v;
        for (const char* n : names) v.push_back(t(n));
        return pi::container_roles(v);
    };
    {
        const auto r = roles({"model.layers.0.pre_feedforward_layernorm.weight",
                              "model.layers.0.inp_gate.weight", "model.layers.0.inp_gate.weight_prefill",
                              "model.layers.0.per_layer_projection.weight", "model.layers.0.layer_output_scale.weight"});
        check(r.unassigned.empty(), "gemma local: the families this forward does not read are recognised");
        check(r.counts.at("gemma_local") == 5, "gemma local: all five land on one role");
    }
    {
        // the forward's own norms must still win: this is the shadowing the ordering could cause
        const auto r = roles({"model.layers.0.post_attention_layernorm.weight",
                              "model.layers.0.input_layernorm.weight"});
        check(r.role_of.at("model.layers.0.post_attention_layernorm.weight") == "post_norm" &&
                  r.role_of.at("model.layers.0.input_layernorm.weight") == "input_norm",
              "gemma local: a norm the forward reads is not swallowed by the recogniser");
    }
}

void test_container_roles() {
    const auto t = [](const std::string& n) { pi::Q4nxTensor x; x.name = n; x.dtype = "I8"; return x; };
    const auto roles = [&](std::initializer_list<const char*> names) {
        std::vector<pi::Q4nxTensor> v;
        for (const char* n : names) v.push_back(t(n));
        return pi::container_roles(v);
    };

    // 1. a transformer pack, HF spelling
    {
        const auto r = roles({"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
                              "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.v_proj.weight",
                              "model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.gate_proj.weight",
                              "model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.down_proj.weight",
                              "model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight",
                              "model.norm.weight", "lm_head.weight"});
        check(r.unassigned.empty(), "roles: a transformer pack classifies completely");
        check(r.counts.at("q_proj") == 1 && r.counts.at("down_proj") == 1, "roles: projections are named by role");
        check(r.role_of.at("model.norm.weight") == "final_norm", "roles: the top-level norm is not a layer norm");
        check(r.role_of.at("model.layers.0.input_layernorm.weight") == "input_norm",
              "roles: a layer norm is the input norm, not the final one");
    }
    // 1b. engine#2097 review: the two cases the table got wrong, pinned so the order cannot regress.
    // `attn_q` / `attn_k` are prefixes of `attn_q_norm` / `attn_k_norm`, so with the projection rules first both
    // norm needles were dead and a per-head q/k norm was filed as a projection.
    {
        const auto r = roles({"blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight",
                              "blk.0.attn_q.weight", "blk.0.attn_k.weight"});
        check(r.role_of.at("blk.0.attn_q_norm.weight") == "q_norm", "roles: attn_q_norm is a norm, not q_proj");
        check(r.role_of.at("blk.0.attn_k_norm.weight") == "k_norm", "roles: attn_k_norm is a norm, not k_proj");
        check(r.role_of.at("blk.0.attn_q.weight") == "q_proj" && r.role_of.at("blk.0.attn_k.weight") == "k_proj",
              "roles: the un-suffixed spellings are still projections");
    }
    // 1c. `in_proj` is not unique to attention: LFM2's short-convolution input projection was counted as qkv_proj.
    {
        const auto r = roles({"model.layers.0.shortconv.in_proj.weight", "model.layers.0.attn.in_proj.weight"});
        check(r.role_of.at("model.layers.0.shortconv.in_proj.weight") == "shortconv_in_proj",
              "roles: shortconv.in_proj is the short-conv projection, not the attention qkv");
        check(r.role_of.at("model.layers.0.attn.in_proj.weight") == "qkv_proj",
              "roles: an attention in_proj is still the fused qkv");
    }
    // 2. the alternate spellings must land on the same roles
    {
        const auto r = roles({"model.layers.0.self_attn.attn_q.weight", "model.layers.0.self_attn.wq.weight",
                              "model.layers.0.mlp.ffn_gate.weight", "model.layers.0.mlp.ffn_down.weight"});
        check(r.unassigned.empty(), "roles: alternate spellings classify");
        check(r.role_of.at("model.layers.0.self_attn.attn_q.weight") == "q_proj", "roles: attn_q is q_proj");
        check(r.role_of.at("model.layers.0.self_attn.wq.weight") == "q_proj", "roles: wq is q_proj");
        check(r.role_of.at("model.layers.0.mlp.ffn_gate.weight") == "gate_proj", "roles: ffn_gate is gate_proj");
    }
    // 3. a bias belongs to its projection's role, and a qkv projection is not a q projection
    {
        const auto r = roles({"model.layers.0.self_attn.q_proj.bias", "model.layers.0.self_attn.qkv_proj.weight"});
        check(r.role_of.at("model.layers.0.self_attn.q_proj.bias") == "q_proj", "roles: a bias keeps its projection's role");
        check(r.role_of.at("model.layers.0.self_attn.qkv_proj.weight") == "qkv_proj",
              "roles: qkv_proj is not classified as q_proj");
    }
    // 4. the two measured residues are recognised, not unknown
    {
        const auto r = roles({"model.layers.0.linear_attn.ssm_alpha_proj.weight",
                              "model.layers.0.linear_attn.ssm_conv1d.weight", "model.mtp.fc.weight",
                              "model.mtp.pre_fc_norm_hidden.weight"});
        check(r.unassigned.empty(), "roles: the SSM and MTP families are recognised, not unassigned");
        check(r.counts.at("ssm") == 2 && r.counts.at("mtp") == 2, "roles: each residue is counted under its own family");
    }
    // 5. something nothing knows is named, not guessed
    {
        const auto r = roles({"model.layers.0.mystery_block.weight"});
        check(r.unassigned.size() == 1, "roles: an unknown tensor is listed");
        check(r.unassigned[0] == "model.layers.0.mystery_block.weight", "roles: it is named, not summarised");
        check(r.counts.empty(), "roles: and no role is invented for it");
    }
}

// engine#1786 (item F1): an NPU2 model source is a safetensors container, and this is the reader that decides whether
// a pack is that genre. The failure that matters is the silent one -- a wrong header length yields an empty inventory,
// which reads as "a pack with no tensors" rather than "a malformed header" -- so each arm asserts the reason, not just
// the emptiness. No weights are read and no device is opened: the header is 26,496 B on the smallest real pack.
void test_q4nx_container(const std::string& root) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::path(root) / "q4nx-container";
    std::error_code ec;
    fs::remove_all(dir, ec);   // a directory left by a run that died is not this run's fixture
    fs::create_directories(dir, ec);
    // `root` is the golden directory when CTest runs this, i.e. the source tree: without the removal every run left
    // tests/golden/q4nx-container/ behind as untracked files. Removed on every way out of the function.
    struct Cleanup {
        fs::path p;
        ~Cleanup() { std::error_code e; fs::remove_all(p, e); }
    } cleanup{dir};

    const auto write_container = [&](const std::string& name, const std::string& header_json, int64_t payload,
                                     bool truncate_header) {
        const fs::path p = dir / name;
        std::string out(8, '\0');
        int64_t n = int64_t(header_json.size());
        for (int i = 0; i < 8; ++i) out[size_t(i)] = char((n >> (8 * i)) & 0xFF);
        out += truncate_header ? header_json.substr(0, header_json.size() / 2) : header_json;
        out.append(size_t(payload), '\x22');
        std::ofstream f(p, std::ios::binary);
        f.write(out.data(), std::streamsize(out.size()));
        return p.string();
    };

    // (a) the real shape: one BF16 vector and one packed I8 matrix, with the payload the offsets imply.
    {
        const std::string hdr =
            R"({"model.layers.0.input_layernorm.weight":{"dtype":"BF16","shape":[640],"data_offsets":[0,1280]},)"
            R"("model.layers.0.mlp.down_proj.weight":{"dtype":"I8","shape":[640,1280],"data_offsets":[1280,820480]}})";
        const auto c = pi::read_q4nx_container(write_container("two.q4nx", hdr, 820480, false));
        check(!c.tensors.empty(), "q4nx container: a two-tensor header reads (" + c.why + ")");
        check(c.tensors.size() == 2, "q4nx container: both tensors are reported");
        check(c.header_bytes == int64_t(hdr.size()), "q4nx container: the 8-byte length is the header's own");
        check(c.payload_bytes == 820480, "q4nx container: the payload is the sum of the data_offsets spans (got " +
                                             std::to_string(c.payload_bytes) + ")");
        bool saw_bf16 = false, saw_i8 = false;
        for (const auto& t : c.tensors) {
            if (t.dtype == "BF16" && t.shape.size() == 1 && t.shape[0] == 640) saw_bf16 = true;
            // The packed matrix: I8 with TWICE the logical element count of a 640-wide FFN is the NPU2 shape.
            if (t.dtype == "I8" && t.shape.size() == 2 && t.shape[0] == 640 && t.shape[1] == 1280) saw_i8 = true;
        }
        check(saw_bf16 && saw_i8, "q4nx container: dtype and shape survive for a vector and a packed matrix");
    }

    // (b) `__metadata__` is a string map, not a tensor, and the vendor's converter does not always write one.
    {
        const std::string hdr =
            R"({"__metadata__":{"group_size":"32"},"t":{"dtype":"I8","shape":[32,256],"data_offsets":[0,5120]}})";
        const auto c = pi::read_q4nx_container(write_container("meta.q4nx", hdr, 5120, false));
        check(c.tensors.size() == 1 && c.tensors[0].name == "t",
              "q4nx container: __metadata__ is skipped, not counted as a tensor");
    }

    // (c) a truncated header is named, and reports no tensors rather than a partial inventory.
    {
        const std::string hdr = R"({"a":{"dtype":"I8","shape":[32,256],"data_offsets":[0,5120]}})";
        const auto c = pi::read_q4nx_container(write_container("short.q4nx", hdr, 5120, true));
        check(c.tensors.empty() && !c.why.empty(),
              "q4nx container: a truncated header names the problem instead of reporting no tensors (" + c.why + ")");
    }

    // (e) engine#1786: a pack that IS a container and has NO graph is recognised and reported, not refused at the first
    // byte. The alternative -- `npu-pack info` answering a 0.5 GB model source with "cannot open <pack>/model.onnx" --
    // describes a missing file on a pack that has no graph by design, and says nothing about what it does contain.
    {
        const fs::path pdir = dir / "container-genre";
        fs::create_directories(pdir, ec);
        const std::string hdr =
            R"({"model.embed_tokens.weight":{"dtype":"BF16","shape":[64,8],"data_offsets":[0,1024]},)"
            R"("model.layers.0.mlp.down_proj.weight":{"dtype":"I8","shape":[8,16],"data_offsets":[1024,1152]}})";
        write_container("container-genre/model.q4nx", hdr, 1152, false);
        const pi::Plan p = pi::plan(pdir.string());
        check(p.container_tensors == 2, "container genre: plan() reports the container's tensors (" +
                                            std::to_string(p.container_tensors) + ")");
        check(p.container_payload_bytes == 1152, "container genre: and the payload the offsets imply");
        check(p.container_dtypes.count("BF16") == 1 && p.container_dtypes.count("I8") == 1,
              "container genre: and both dtypes it found");
        // The note must say it is NOT importable: a plan that listed these as importable would be worse than the
        // refusal it replaces, because it would claim a compute mapping this importer does not have.
        check(p.container_note.find("not importable") != std::string::npos,
              "container genre: the note says the pack is recognised and not importable (" + p.container_note + ")");
        const json m = json::parse(p.manifest());
        check(m.contains("container") && m["container"]["tensors"] == 2,
              "container genre: and manifest() reports it, so npu-pack info prints it");
    }

    // (d) an implausible length is refused by the number, so a mis-typed length cannot make the caller allocate it.
    {
        const fs::path p = dir / "huge.q4nx";
        std::string out(8, '\0');
        const int64_t n = int64_t(1) << 40;
        for (int i = 0; i < 8; ++i) out[size_t(i)] = char((n >> (8 * i)) & 0xFF);
        {
            std::ofstream f(p, std::ios::binary);
            f.write(out.data(), std::streamsize(out.size()));
        }   // flushed and closed: the reader must see the bytes, not an empty file
        const auto c = pi::read_q4nx_container(p.string());
        check(c.tensors.empty() && c.why.find("implausible") != std::string::npos,
              "q4nx container: an implausible header length is refused by the number (" + c.why + ")");
    }
}


// engine#1725: a multi-component embedding carries a PER-LAYER table and its projection norm beside the token table, and
// the importer writes both into the container so the forward path can gather them. Measured shapes on
// amd/gemma-4-e2b-it-mobius-int4_rai_1.9.0_npu: embed_tokens.weight (262144,1536) f32,
// embed_tokens_per_layer.weight (262144,8960) f32, per_layer_projection_norm.weight (256,) -- all f32, no scales.
// This drives the writer on a crafted pack of that shape (V=4, width=6) and reads the container header back.
void test_per_layer_container(const std::string& root) {
    const fs::path dir = fs::path(root) / "perlayer";
    fs::remove_all(dir);
    fs::create_directories(dir / "embedding");
    const int64_t V = 4, width = 6;
    // two-arg form: `norm(size_t(width))` parses as a function declaration and `norm{size_t(width)}` is an
    // initializer_list of ONE element = width. Both mistakes produce a plausible-looking fixture.
    std::vector<float> tok(size_t(V) * 3, 0.0f), per(size_t(V) * size_t(width), 0.0f), norm(size_t(width), 0.0f);
    for (size_t i = 0; i < tok.size(); ++i) tok[i] = float(i) * 0.5f;
    for (size_t i = 0; i < per.size(); ++i) per[i] = float(int(i) - 8) * 0.25f;
    for (size_t i = 0; i < norm.size(); ++i) norm[i] = 1.0f + float(i) * 0.125f;
    std::string data;
    const size_t tok_bytes = tok.size() * 4, per_bytes = per.size() * 4;
    data.append(reinterpret_cast<const char*>(tok.data()), tok_bytes);
    data.append(reinterpret_cast<const char*>(per.data()), per_bytes);
    data.append(reinterpret_cast<const char*>(norm.data()), norm.size() * 4);
    put_file((dir / "embedding" / "model.onnx.data").string(), data);
    put_file((dir / "embedding" / "model.onnx").string(),
             pb::model({pb::tensor("embedding.embed_tokens.weight", {V, 3}, "model.onnx.data", 0, tok_bytes, 1),
                        pb::tensor("embedding.embed_tokens_per_layer.weight", {V, width}, "model.onnx.data", tok_bytes,
                                   per_bytes, 1),
                        pb::tensor("embedding.per_layer_projection_norm.weight", {width}, "model.onnx.data",
                                   tok_bytes + per_bytes, norm.size() * 4, 1)},
                       {}));
    put_file((dir / "genai_config.json").string(),
             R"({"model":{"type":"gemma4","embedding":{"filename":"embedding/model.onnx"},)"
             R"("decoder":{"filename":"decoder/model.onnx"}}})");
    put_file((dir / "config.json").string(),
             R"({"model_type":"gemma4","hidden_size":3,"intermediate_size":6,"num_hidden_layers":1,)"
             R"("num_attention_heads":1,"num_key_value_heads":1,"head_dim":3,"vocab_size":4,"rope_theta":10000.0})");
    std::string why;
    const pi::EmbedPerLayer pl = pi::read_embedding_per_layer(dir.string(), 3, &why);
    std::printf("per-layer: V %lld width %lld, why: %s (tok %zu per %zu norm %zu bytes)\n", (long long) pl.V,
                (long long) pl.width, why.c_str(), tok_bytes, per_bytes, norm.size() * 4);
    check(pl.ok(), "per-layer: the table and its projection norm are read from the component genai_config.json names");
    bool same = pl.ok() && pl.V == V && pl.width == width;
    for (size_t i = 0; same && i < per.size(); ++i) same = pl.at(int64_t(i) / width, int64_t(i) % width) == per[i];
    for (size_t i = 0; same && i < norm.size(); ++i) same = pl.norm[i] == norm[i];
    check(same, "per-layer: the values come back exactly as the component states them");
    // the token table is still this reader's other half: it must NOT be refused because a per-layer table is present
    std::string why2;
    const std::vector<float> t = pi::read_embedding_component(dir.string(), &why2);
    check(t.size() == size_t(V) * 3, "per-layer: the token table still reads beside it (" + std::to_string(t.size()) + ")");

    // #1825 review: the band converter had no arm, and it rounded "up iff the kept lsb is 1" -- half of all values off by
    // one ulp, including values bf16 holds EXACTLY. per[] here is multiples of 0.25 (exact); the fixture below adds the
    // three cases that separate the rules: an exact value with an odd kept lsb (1.0078125 = 0x3F81'0000 must stay
    // 0x3F81), a tie on an even lsb (stays), a tie on an odd lsb (goes up to even), and one just above a tie.
    {
        const auto f32_of = [](uint32_t bits) {
            float f;
            std::memcpy(&f, &bits, 4);
            return f;
        };
        const fs::path d2 = fs::path(root) / "perlayer-band";
        fs::remove_all(d2);
        fs::create_directories(d2 / "embedding");
        const int64_t V2 = 70, W2 = 5;   // 70 rows: two whole 32-row bands and a 6-row one
        std::vector<float> per2(size_t(V2) * size_t(W2));
        for (size_t i = 0; i < per2.size(); ++i) per2[i] = 0.001f * float(int(i * 37 % 211) - 100);   // inexact in bf16
        per2[0] = f32_of(0x3F810000u);   // exact, odd lsb          -> 0x3F81
        per2[1] = f32_of(0x3F808000u);   // tie, even lsb           -> 0x3F80
        per2[2] = f32_of(0x3F818000u);   // tie, odd lsb            -> 0x3F82
        per2[3] = f32_of(0x3F808001u);   // just above a tie        -> 0x3F81
        per2[size_t(69) * size_t(W2) + 4] = f32_of(0xBF810000u);   // the last value of the short band, exact and negative
        std::vector<float> norm2(size_t(W2), 1.0f), tok2(size_t(V2) * 3, 0.5f);
        std::string data2;
        data2.append(reinterpret_cast<const char*>(tok2.data()), tok2.size() * 4);
        data2.append(reinterpret_cast<const char*>(per2.data()), per2.size() * 4);
        data2.append(reinterpret_cast<const char*>(norm2.data()), norm2.size() * 4);
        put_file((d2 / "embedding" / "model.onnx.data").string(), data2);
        put_file((d2 / "embedding" / "model.onnx").string(),
                 pb::model({pb::tensor("embedding.embed_tokens.weight", {V2, 3}, "model.onnx.data", 0, tok2.size() * 4, 1),
                            pb::tensor("embedding.embed_tokens_per_layer.weight", {V2, W2}, "model.onnx.data",
                                       tok2.size() * 4, per2.size() * 4, 1),
                            pb::tensor("embedding.per_layer_projection_norm.weight", {W2}, "model.onnx.data",
                                       tok2.size() * 4 + per2.size() * 4, norm2.size() * 4, 1)},
                           {}));
        put_file((d2 / "genai_config.json").string(),
                 R"({"model":{"type":"gemma4","embedding":{"filename":"embedding/model.onnx"},)"
                 R"("decoder":{"filename":"decoder/model.onnx"}}})");
        std::string why3;
        const pi::EmbedPerLayer p2 = pi::read_embedding_per_layer(d2.string(), 3, &why3);
        check(p2.ok() && p2.V == V2 && p2.width == W2, "per-layer band: the table is located (" + why3 + ")");
        if (p2.ok()) {
            // the reference is as_bf16-equivalent: pi::bf16_bits_of is the file's exported round-to-nearest-even
            std::vector<uint16_t> want(per2.size());
            for (size_t i = 0; i < per2.size(); ++i) want[i] = pi::bf16_bits_of(per2[i]);
            check(want[0] == 0x3F81 && want[1] == 0x3F80 && want[2] == 0x3F82 && want[3] == 0x3F81,
                  "per-layer band: the reference rounds to nearest even on the four separating values");
            std::vector<uint16_t> got(per2.size(), 0xFFFF);
            bool bands = true;
            for (int64_t row = 0; row < V2; row += 32) {
                const int64_t n = std::min<int64_t>(32, V2 - row);
                std::string bw;
                bands = bands && pi::read_per_layer_table_band(
                                     p2, row, n, reinterpret_cast<uint8_t*>(got.data() + size_t(row) * size_t(W2)), &bw) == n;
            }
            check(bands, "per-layer band: 32 + 32 + 6 rows read");
            size_t differ = 0;
            for (size_t i = 0; i < got.size(); ++i) differ += got[i] != want[i];
            check(differ == 0, "per-layer band: every banded value is round-to-nearest-even bf16 (" +
                                   std::to_string(differ) + " of " + std::to_string(got.size()) + " differ)");
            check(got[0] == 0x3F81, "per-layer band: an exactly representable value with an odd lsb is not moved");
            std::string bw;
            std::vector<uint8_t> scratch(size_t(W2) * 2 * 2);
            check(pi::read_per_layer_table_band(p2, V2 - 1, 2, scratch.data(), &bw) == -1 &&
                      pi::read_per_layer_table_band(p2, -1, 1, scratch.data(), &bw) == -1,
                  "per-layer band: a band past the table, or before it, is refused");
        }
        // a table that declares more rows than its data carries is refused when it is LOCATED, not at some later band
        put_file((d2 / "embedding" / "model.onnx").string(),
                 pb::model({pb::tensor("embedding.embed_tokens_per_layer.weight", {V2 + 1000, W2}, "model.onnx.data",
                                       tok2.size() * 4, per2.size() * 4, 1)},
                           {}));
        std::string why4;
        const pi::EmbedPerLayer p3 = pi::read_embedding_per_layer(d2.string(), 3, &why4);
        check(!p3.ok() && why4.find("more bytes than its data carries") != std::string::npos,
              "per-layer band: a table longer than its data is refused by the reader (" + why4 + ")");
        fs::remove_all(d2);
    }
    fs::remove_all(dir);
}



// engine#1725, #1825 review: the per-layer block THROUGH import_pack(). The reader arms above call the helpers; nothing
// ran the writer, so the streamed table, the offsets of every tensor after it, the per-layer split of the context
// projection and the two config keys that switch the block on were all unpinned. The fixture is the tied block-32 one
// (it reaches the embedding writer under allow-incomplete) plus an embedding component shaped like gemma-4's: a table
// NL x P wide, a norm P wide (shared by the layers -- not `width` wide), and the context projection quantised at
// block 32, stored at a NON-ZERO offset of the external data.
void test_per_layer_import(const std::string& root) {
    const fs::path dir = fs::path(root) / "perlayer-import";
    fs::remove_all(dir);
    fs::create_directories(dir / "cache");
    fs::create_directories(dir / "embedding");
    const int H = 128, L = 8, V = 64;
    const int64_t P = 4, NL = 2, W = P * NL;
    {
        const int64_t stride = 1 << 20, second = 262144;
        std::string fc(size_t(int64_t(L) * stride + 64 + size_t(H) * 2), '\0');
        auto put_eps = [&](int64_t off) {
            fc[size_t(off)] = char(0x86);
            fc[size_t(off) + 1] = char(0x35);
        };
        for (int l = 0; l < L; ++l) {
            put_eps(int64_t(l) * stride);
            put_eps(int64_t(l) * stride + second);
        }
        put_eps(int64_t(L) * stride);
        for (int i = 0; i < H; ++i) {
            const uint16_t bits = pi::bf16_bits_of(float(i + 1));
            fc[size_t(int64_t(L) * stride + 64 + 2 * i)] = char(bits & 0xFF);
            fc[size_t(int64_t(L) * stride + 64 + 2 * i) + 1] = char(bits >> 8);
        }
        put_file((dir / "dd_metastate_Llm_Token_Token_crafted.fconst").string(), fc);
    }
    put_file((dir / "config.json").string(),
             R"({"model_type":"qwen2","hidden_size":128,"intermediate_size":256,"num_hidden_layers":8,)"
             R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,"vocab_size":64,)"
             R"("rope_theta":10000.0,"tie_word_embeddings":true,"rms_norm_eps":1e-06})");
    {
        std::string data(size_t(V) * size_t(H) * 2, '\0');
        for (int r = 0; r < V; ++r)
            for (int c = 0; c < H; ++c) {
                const uint16_t bits = pi::bf16_bits_of(0.05f * float((r * 7 + c * 13) % 97) - 2.4f);
                data[(size_t(r) * H + c) * 2] = char(bits & 0xFF);
                data[(size_t(r) * H + c) * 2 + 1] = char(bits >> 8);
            }
        put_file((dir / "model.onnx.data").string(), data);
    }
    put_file((dir / "cache" / "lm_head_meta.json").string(),
             R"({"op_list":[{"type":"MladfMatMul","name":"lm_head.MatMulNBits",)"
             R"("attrs":{"N":{"value":[64]},"K":{"value":[128]},"block_size":{"value":[32]},"bits":{"value":[4]}}}],)"
             R"("tensor_map":{}})");
    put_file((dir / "model.onnx").string(),
             pb::model({pb::tensor("model.embed_tokens.weight", {V, H}, "model.onnx.data", 0,
                                   uint64_t(V) * uint64_t(H) * 2, 10),
                        pb::placeholder("lm_head.weight.MatMulNBits.qweight"),
                        pb::placeholder("lm_head.weight.MatMulNBits.scales"),
                        pb::placeholder("lm_head.weight.MatMulNBits.zeros")},
                       {}));
    // the embedding component: table [V][W] f32, norm [P] f32, projection [W][H] int4 at block 32 + f32 scales
    std::vector<float> per(size_t(V) * size_t(W), 0.0f), norm(size_t(P), 0.0f), scales(size_t(W) * size_t(H / 32), 0.5f);
    for (size_t i = 0; i < per.size(); ++i) per[i] = 0.003f * float(int(i * 53 % 409) - 200);
    {
        uint32_t odd = 0x3F810000u;   // exact in bf16 with an odd kept lsb: must come back as 0x3F81
        std::memcpy(&per[0], &odd, 4);
    }
    for (size_t i = 0; i < norm.size(); ++i) norm[i] = 1.0f + 0.25f * float(i);
    const std::string qw(size_t(W) * size_t(H) / 2, char(0xBB));   // every code 11 -> (11 - 8) x 0.5 = 1.5
    std::string data;
    data.append(reinterpret_cast<const char*>(per.data()), per.size() * 4);
    const uint64_t norm_off = data.size();
    data.append(reinterpret_cast<const char*>(norm.data()), norm.size() * 4);
    const uint64_t qw_off = data.size();
    data += qw;
    const uint64_t sc_off = data.size();
    data.append(reinterpret_cast<const char*>(scales.data()), scales.size() * 4);
    put_file((dir / "embedding" / "model.onnx.data").string(), data);
    put_file((dir / "embedding" / "model.onnx").string(),
             pb::model({pb::tensor("embedding.embed_tokens_per_layer.weight", {V, W}, "model.onnx.data", 0, per.size() * 4, 1),
                        pb::tensor("embedding.per_layer_projection_norm.weight", {P}, "model.onnx.data", norm_off,
                                   norm.size() * 4, 1),
                        pb::tensor("embedding.per_layer_model_projection.weight_t_Q4", {W, int64_t(H / 32), 16},
                                   "model.onnx.data", qw_off, qw.size(), 2),
                        pb::tensor("embedding.per_layer_model_projection.weight_t_scales", {W * int64_t(H / 32)},
                                   "model.onnx.data", sc_off, scales.size() * 4, 1)},
                       {}));
    put_file((dir / "genai_config.json").string(), R"({"model":{"embedding":{"filename":"embedding/model.onnx"}}})");

    const std::string out =
        (::onebit::test::tmp_root() / ("oga_pack_import_perlayer." + std::to_string(::getpid()))).string();
    fs::remove_all(out);
    pi::Options allow;
    allow.allow_incomplete = true;
    std::string why;
    pi::Report r;
    try {
        r = pi::import_pack(dir.string(), out, allow);
    } catch (const std::exception& e) {
        why = e.what();
    }
    check(why.empty(), "per-layer import: the pack imports (" + why.substr(0, 200) + ")");
    check(r.per_layer_P == P && r.per_layer_V == V, "per-layer import: the report carries P " + std::to_string(r.per_layer_P) +
                                                        " and V " + std::to_string(r.per_layer_V));
    if (fs::exists(out + "/model.q4nx")) {
        const json tbl = container_table(out + "/model.q4nx");
        const std::vector<uint8_t> blob = container_payload(out + "/model.q4nx");
        const auto u16 = [&](const std::string& name, size_t i) {
            const auto offs = tbl[name]["data_offsets"].get<std::vector<uint64_t>>();
            return uint16_t(uint16_t(blob[size_t(offs[0]) + 2 * i]) | uint16_t(uint16_t(blob[size_t(offs[0]) + 2 * i + 1]) << 8));
        };
        const bool have = tbl.contains("model.per_layer_token_embd.weight") && tbl.contains("model.per_layer_proj_norm.weight") &&
                          tbl.contains("model.per_layer_model_proj.weight_layer0") &&
                          tbl.contains("model.per_layer_model_proj.weight_layer1") &&
                          !tbl.contains("model.per_layer_model_proj.weight_layer2");
        check(have, "per-layer import: the container carries the table, the norm and one projection per layer (2)");
        if (have) {
            const auto shape = tbl["model.per_layer_token_embd.weight"]["shape"].get<std::vector<int64_t>>();
            const auto offs = tbl["model.per_layer_token_embd.weight"]["data_offsets"].get<std::vector<uint64_t>>();
            check(shape == std::vector<int64_t>({V, W}) && offs[1] - offs[0] == uint64_t(V) * uint64_t(W) * 2 &&
                      tbl["model.per_layer_token_embd.weight"]["dtype"] == "BF16",
                  "per-layer import: the table is declared BF16 [V, NL*P] with exactly that many bytes");
            size_t differ = 0;
            for (size_t i = 0; i < per.size(); ++i) differ += u16("model.per_layer_token_embd.weight", i) != pi::bf16_bits_of(per[i]);
            check(differ == 0, "per-layer import: every value of the STREAMED table is round-to-nearest-even bf16 (" +
                                   std::to_string(differ) + " of " + std::to_string(per.size()) + " differ)");
            check(u16("model.per_layer_token_embd.weight", 0) == 0x3F81,
                  "per-layer import: an exactly representable value with an odd lsb is not moved");
            // the tensors AFTER the streamed one: their offsets are only right if the stream wrote what the header says
            bool norm_ok = tbl["model.per_layer_proj_norm.weight"]["shape"].get<std::vector<int64_t>>() == std::vector<int64_t>({P});
            for (size_t i = 0; norm_ok && i < norm.size(); ++i)
                norm_ok = u16("model.per_layer_proj_norm.weight", i) == pi::bf16_bits_of(norm[i]);
            check(norm_ok, "per-layer import: the P-wide norm follows the streamed table intact (shared by the layers)");
            bool proj_ok = tbl["model.per_layer_model_proj.weight_layer1"]["shape"].get<std::vector<int64_t>>() ==
                           std::vector<int64_t>({P, int64_t(H)});
            for (size_t i = 0; proj_ok && i < size_t(P) * size_t(H); ++i)
                proj_ok = u16("model.per_layer_model_proj.weight_layer0", i) == pi::bf16_bits_of(1.5f) &&
                          u16("model.per_layer_model_proj.weight_layer1", i) == pi::bf16_bits_of(1.5f);
            check(proj_ok, "per-layer import: the context projection is read at ITS offset of the external data and split "
                           "per layer ((11 - 8) x 0.5 = 1.5 everywhere)");
        }
        std::ifstream cf(out + "/config.json");
        check(cf.good(), "per-layer import: the container carries a config.json");
        if (cf.good()) {
            const json cfg = json::parse(cf);
            check(cfg.value("hidden_size_per_layer_input", int64_t(0)) == P && cfg.value("vocab_size_per_layer_input", int64_t(0)) == V,
                  "per-layer import: config.json switches the block on (hidden_size_per_layer_input 4, vocab 64)");
        }
    } else {
        check(false, "per-layer import: no container was written");
    }
    fs::remove_all(out);
    fs::remove_all(dir);
}

// engine#1503: a 1.9 fconst whose blocks are NOT uniform (the mobius line). `fconst19_detect` requires one stride for the
// whole file, so it fails -- and the failure used to be indistinguishable from "not a 1.9 fconst at all". The refusal now
// names the structure and says the rule table cannot describe it. Measured on gemma-4-e2b: 35 blocks of 8 anchors, sizes
// 21.87-288.06 MiB.
void test_fconst_variable_stride(const std::string& root) {
    const fs::path dir = fs::path(root) / "fconst-varstride";
    fs::remove_all(dir);
    fs::create_directories(dir / "cache");
    const int H = 128, L = 4, heads = 2;
    const int64_t hdr = heads * 4;
    // four blocks, each header then a body of a DIFFERENT size: 64 KiB, 96 KiB, 128 KiB, 160 KiB
    const int64_t bodies[4] = {64 << 10, 96 << 10, 128 << 10, 160 << 10};
    std::string fc;
    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < heads; ++h) {                  // 00 00 7f ff per head
            fc.push_back('\0'); fc.push_back('\0'); fc.push_back(char(0x7f)); fc.push_back(char(0xff));
        }
        fc.append(size_t(bodies[l]), '\0');
    }
    check(fc.size() > size_t(L * hdr), "variable stride: the crafted fconst is larger than its headers");
    put_file((dir / "dd_metastate_Llm_Token_Token_crafted.fconst").string(), fc);
    put_file((dir / "config.json").string(),
             R"({"model_type":"gemma4","hidden_size":128,"intermediate_size":256,"num_hidden_layers":4,)"
             R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,"vocab_size":64,"rope_theta":10000.0})");
    put_file((dir / "model.onnx").string(), pb::model({}, {}));
    put_file((dir / "model.onnx.data").string(), std::string(4096, '\x11'));
    pi::Options allow;
    allow.allow_incomplete = true;
    const std::string out = (::onebit::test::tmp_root() / ("varstride_out." + std::to_string(::getpid()))).string();
    fs::remove_all(out);
    bool refused = false;
    std::string what;
    try {
        (void) pi::import_pack(dir.string(), out, allow);
    } catch (const std::exception& e) {
        what = e.what();
        refused = what.find("variable-stride") != std::string::npos || what.find("NOT uniform") != std::string::npos;
    }
    std::printf("variable stride: %s\n", refused ? "refused by name (variable-stride container)" : what.substr(0, 140).c_str());
    check(refused, "variable stride: a 1.9 fconst whose blocks differ in size is refused BY NAME, not as 'not a 1.9 fconst'");
    fs::remove_all(out);
    fs::remove_all(dir);
}

// engine#1740 (W1a): the tail-head placement arithmetic, pinned without a pack, a device or a fixture. The rows, the
// tile size and the envelope are the hidden-896 pack's, so the numbers here are the real ones: 1,856 B envelope,
// 2,374 tiles of 122,880 B, tail 1,856 + 2,374 x 122,880 = 291,653,952 B.
void test_tail_head_placement() {
    // The real hidden-896 geometry: group_bytes(896, 64) = 8 x 68 x 64 = 34,816 B per head tile, V/64 = 2,374 tiles,
    // 1,856 B envelope, so the tail is 1,856 + 2,374 x 34,816 = 82,655,040 B -- which is the 0.5B pack's measured tail
    // span (its tile 0 sits at 209,000,768 B in a 291,653,952 B file).
    const int64_t env = 1856, rows = 2374, tile = 34816;
    const int64_t head = rows * tile;
    const int64_t exact_bytes = env + head;              // the file ends with the head
    // (a) the head ends the file: exactly one candidate, no slack
    {
        const pi::TailHeadPlacement p = pi::plan_tail_head(exact_bytes, 0, env, rows, tile, 0);
        check(p.why.empty() && p.exact, "tail placement: a tail that ends with the head is exact");
        check(p.first == env && p.last == env, "tail placement: and the only candidate is the envelope's end");
    }
    // (b2) engine#1773's review: the zero run is counted FROM EOF, so a head whose own last tiles are zero contributes
    //      to it exactly like padding does, and `last` then computes as base - n*tile with only the `last < base` clamp
    //      keeping it at the envelope's end. Measured by lifting `plan_tail_head` verbatim and running the hidden-896
    //      geometry through it: all four cases below give first == last == 1,856. They are the cases where a clamp
    //      mistake would hide, so they are pinned here rather than left to the pad-whole-tiles arm.
    {
        const pi::TailHeadPlacement p = pi::plan_tail_head(exact_bytes, 0, env, rows, tile, 2 * tile);
        check(p.why.empty() && p.last == env && p.first == env,
              "tail placement: a head whose own last two tiles are zero still lands on the envelope's end");
    }
    {
        const pi::TailHeadPlacement p = pi::plan_tail_head(exact_bytes + 5 * tile, 0, env, rows, tile, 7 * tile);
        check(p.why.empty() && p.last == env && p.first == env,
              "tail placement: a zero run spanning the head's own tail tiles AND its padding is not counted twice");
    }
    {
        const pi::TailHeadPlacement p = pi::plan_tail_head(exact_bytes + 512, 0, env, rows, tile, 512);
        check(p.why.empty() && p.last == env && p.first == env,
              "tail placement: padding smaller than one tile still lands on the envelope's end");
    }
    {
        const pi::TailHeadPlacement p = pi::plan_tail_head(exact_bytes + rows * tile, 0, env, rows, tile, rows * tile);
        check(p.why.empty() && p.last == env && p.first == env,
              "tail placement: a zero run longer than the head is clamped to its start");
    }
    // (b) n whole zero tiles after the head: the candidate walks back by exactly n tiles, never past the envelope
    for (int64_t n : {1, 3, 9}) {
        const int64_t bytes = exact_bytes + n * tile;
        const pi::TailHeadPlacement p = pi::plan_tail_head(bytes, 0, env, rows, tile, n * tile);
        check(p.why.empty() && !p.exact, "tail placement: trailing padding is not the exact case");
        check(p.last == env,
              "tail placement: " + std::to_string(n) + " zero tile(s) at EOF still land the head at the envelope's end");
        check(p.first == env, "tail placement: and the search never walks to before the envelope");
    }
    // (c) a clean zero run of ANY length puts the head at the envelope's end -- the padding's size is not a displacement
    {
        const int64_t bytes = exact_bytes + 20 * tile;
        const pi::TailHeadPlacement p = pi::plan_tail_head(bytes, 0, env, rows, tile, 20 * tile);
        check(p.why.empty() && p.first == env && p.last == env,
              "tail placement: trailing padding of 20 tiles still ends the head at the envelope's end");
    }
    // (c2) but when the file ends in NON-zero bytes after the head, the zero run is short and the candidates span the
    // slack -- this is the shape the caller's correlation has to decide (the hidden-1536 case)
    {
        const int64_t bytes = exact_bytes + 20 * tile;
        const pi::TailHeadPlacement p = pi::plan_tail_head(bytes, 0, env, rows, tile, 3 * tile);
        check(p.why.empty() && p.first > env && p.last > p.first,
              "tail placement: a short zero run before non-zero junk leaves a span of candidates, not one guess");
    }
    // (d) refinements: a tail shorter than envelope + head is refused by name, with both numbers in it
    {
        const pi::TailHeadPlacement p = pi::plan_tail_head(env + head - 1, 0, env, rows, tile, 0);
        check(!p.why.empty(), "tail placement: a short tail is refused, not read");
        check(p.why.find(std::to_string(env + head)) != std::string::npos,
              "tail placement: and the refusal carries the length it wanted (" + p.why + ")");
    }
    // (e) one byte short is refused for the length, and one byte more is a candidate rather than a refusal: the
    // boundary is exact, which is what keeps a part-tile payload from being rounded up into a head
    {
        const pi::TailHeadPlacement short_by_one = pi::plan_tail_head(env + head - 1, 0, env, rows, tile, 0);
        const pi::TailHeadPlacement one_more = pi::plan_tail_head(env + head + 1, 0, env, rows, tile, 0);
        check(!short_by_one.why.empty() && one_more.why.empty(),
              "tail placement: the length boundary is exact (short refused, one byte more is a candidate)");
        check(one_more.last == env, "tail placement: and the extra byte does not move the head");
    }
    // (f) a tail_off that moves the window is honoured: the same arithmetic one layer block later
    {
        const int64_t tail_off = 5 * 8708288;   // the 0.5B pack's measured layer stride
        const pi::TailHeadPlacement p = pi::plan_tail_head(tail_off + exact_bytes, tail_off, env, rows, tile, 0);
        check(p.why.empty() && p.last == tail_off + env, "tail placement: the window is relative to tail_off");
    }
    // and the SEARCH itself (engine#1740 review): the candidate loop, best/runner-up tracking, the kTieCorr cut, the
    // ambiguity case and the untied refusal. Synthetic head through encode_tile, so no pack and no device.
    {
        std::string why;
        // called first, then checked: building the message inside the check() call read `why` before the call filled it,
        // because argument evaluation order is unspecified -- the first run of this arm failed with an empty reason.
        const bool ok = pi::tail_head_selftest(&why);
        check(ok, "tail search: a synthetic tied head locates, survives a one-tile displacement, refuses "
                  "untied-with-padding and takes an untied exact tail unverified"
                  + (why.empty() ? std::string() : " (" + why + ")"));
    }
}


/* engine#1889: a pack that declares per-head q/k gains and whose gains the importer did not locate must be REFUSED,
 * not imported without them. Hunyuan-7B is the measured case: `use_qk_norm: true`, 32x128 + 8x128 gains in its block,
 * a 96-tile attention payload against the 160 the same geometry gets elsewhere -- and the import reported
 * `attention 5120/5120 tiles` byte-exact while the served model answered `Trease/ft//////////` three of three.
 * The three cases are separate on purpose: declared-and-not-found is the refusal; found-and-not-declared is the Qwen3
 * shape (the sections are there, the key is not) and must NOT be refused. */
void test_qk_norm_refusal() {
    const std::string a = pi::qk_norm_refusal(true, false);
    check(!a.empty(), "qk norm: declared but not found is a refusal (importing without the gains answers gibberish)");
    check(a.find("use_qk_norm") != std::string::npos, "qk norm: the refusal names the key the pack declared");
    check(a.find("not where the block layout says") != std::string::npos,
          "qk norm: and says what was not found, rather than failing generically");
    check(a.find("engine#1889") != std::string::npos, "qk norm: the refusal carries its issue, like the others here");
    check(pi::qk_norm_refusal(true, true).empty(),
          "qk norm: declared and found is allowed (the arithmetic or the graph located the gains)");
    check(pi::qk_norm_refusal(false, true).empty(),
          "qk norm: found and not declared is allowed (Qwen3 ships the sections without the key)");
    check(pi::qk_norm_refusal(false, false).empty(), "qk norm: neither is the ordinary pack");
    std::printf("  qk norm: 7 arms (declared+not-found refuses; found-only and neither are allowed)\n");
}

/* engine#1921: the nibble packer applies `& 0x0F`, so a code outside 0..15 WRAPS a full code range instead of
 * saturating -- measured as `container == expected mod 4.0` over 32/32 columns of a group, i.e. a wrong weight that still
 * looks plausible. The writer now refuses, and these arms pin the two ends of the legal range plus the refusal, because an
 * arm that only checked the throw would pass on a writer that refused everything. */
void test_encode_tile_code_range() {
    const auto tile_with = [](int v) {
        pi::TileW t;
        t.k = 128;      // one 128-wide group, the minimum the layout allows
        t.rows = 64;
        t.code.assign(size_t(t.rows) * size_t(t.k), uint8_t(v));
        t.scale.assign(size_t(t.rows), 1.0f);
        t.zp.assign(size_t(t.rows), 0);
        t.bias.assign(size_t(t.rows), 0);
        return t;
    };
    bool ok0 = true, ok15 = true;
    try { (void)pi::encode_tile(tile_with(0)); } catch (const std::exception&) { ok0 = false; }
    try { (void)pi::encode_tile(tile_with(15)); } catch (const std::exception&) { ok15 = false; }
    check(ok0, "encode tile: code 0 (the bottom of the legal range) encodes");
    check(ok15, "encode tile: code 15 (the top of the legal range) encodes");
    std::string why;
    try { (void)pi::encode_tile(tile_with(16)); } catch (const std::exception& e) { why = e.what(); }
    check(!why.empty(), "encode tile: code 16 is REFUSED, not wrapped to 0");
    check(why.find("engine#1921") != std::string::npos, "encode tile: the refusal carries its issue");
    check(why.find("0..15") != std::string::npos, "encode tile: and states the range it wants");
    check(why.find("wrap") != std::string::npos, "encode tile: and says what would have happened instead");
    std::printf("  encode tile: 6 arms (0 and 15 encode, 16 refuses by name)\n");
}

// engine#1921 follow-up: the container writer (tile_q4_1 / tile_q4_1_f32) packs with the same `& 0x0F` as
// encode_tile(), and it is the writer engine#1918's wrap was measured on. Both tile forms refuse a code above 15 by
// position, and the legal extremes still write: a guard that refused everything would pass a throw-only arm.
void test_q4_writer_code_range() {
    auto make = [](uint8_t code_at_last) {
        pi::Q4 q;
        q.N = 32; q.K = 256; q.block = 32;
        q.codes.assign(size_t(q.N) * q.K, 0);
        q.scale.assign(size_t(q.N) * (q.K / 32), 1.0f);
        q.zero.assign(size_t(q.N) * (q.K / 32), 0.0f);
        q.codes[0] = 15;                              // the top of the range, legal
        q.codes[size_t(31) * q.K + 255] = code_at_last;
        return q;
    };
    for (int f32 = 0; f32 < 2; ++f32) {
        const char* form = f32 ? "q4_1f" : "q4_1";
        auto run = [&](const pi::Q4& q, std::vector<uint8_t>& out) {
            if (f32) pi::tile_q4_1_f32(q, out); else pi::tile_q4_1(q, out);
        };
        std::vector<uint8_t> out;
        bool ok = false;
        try { run(make(15), out); ok = !out.empty(); } catch (const std::exception&) {}
        check(ok, std::string("q4 writer (") + form + "): codes 0 and 15 write a tile");
        std::string why;
        std::vector<uint8_t> out2;
        try { run(make(16), out2); } catch (const std::exception& e) { why = e.what(); }
        check(!why.empty(), std::string("q4 writer (") + form + "): code 16 is REFUSED, not wrapped to 0");
        check(why.find("row 31, column 255") != std::string::npos,
              std::string("q4 writer (") + form + "): the refusal names the row and column");
        check(why.find("0..15") != std::string::npos && why.find("wrap") != std::string::npos,
              std::string("q4 writer (") + form + "): and states the range and what would have happened");
        check(out2.empty(), std::string("q4 writer (") + form + "): nothing is written before the refusal");
    }
    std::printf("  q4 writer: 10 arms (both tile forms: 15 writes, 16 refuses by row and column)\n");
}

// engine#1889: the arithmetic that claims the per-head QK gains out of the 640 B Hunyuan-7B carries between its last qkv
// tile and o_proj. A wrong offset there reads tile bytes as bf16 and mis-rotates every pack of the family, so both the
// right shape (2 * (64 + head_dim*2) = 640 at head_dim 128) and the neighbours that must NOT be taken for it are driven.
// The rows a tail head occupies in the container: the pack's own padding to whole 64-row tiles stays. The pack leg's
// tile-count check used the vocabulary, so it failed on every pack whose vocabulary is not a multiple of 64 although
// the import was right (Hunyuan-1.8B: 15,104 tiles written, 15,100 expected).
void test_head_stored_rows() {
    check(pi::head_stored_rows(120818, true) == 120832, "head rows: Hunyuan's 120,818 are stored as 120,832");
    check(pi::head_stored_rows(151936, true) == 151936, "head rows: a multiple of 64 is stored as it is");
    check(pi::head_stored_rows(49152 + 32, true) == 49152 + 64, "head rows: a multiple of 32 that is not one of 64 is padded");
    check(pi::head_stored_rows(120818, false) == 120818, "head rows: a graph head is not padded");
    check(pi::head_stored_rows(0, true) == 0, "head rows: no vocabulary, no rows");
}

void test_qk_gain_offsets() {
    const pi::QkGainOffsets o = pi::qk_gain_offsets(1000, 128, 640);
    // engine#1934: block 0 is Q. Hunyuan-7B's own model.onnx carries both gains as FP16 initializers, and q_norm is the
    // first block and k_norm the second, bit for bit, in 32 of 32 layers (layer 0: q_norm bytes at 13,795,456 = gap + 64,
    // k_norm bytes 320 B later). The order this arm replaced had them the other way round, and nothing else could see
    // it: the two gains differ by about one bf16 ulp, so both orders "look like a gain".
    check(o.q_norm == 1064, "qk gains: block 0, one kNormOff (64 B) after the gap, is q_norm (engine#1934)");
    check(o.k_norm == 1384, "qk gains: block 1, after block 0 (64 B header + a 256 B bf16 [128] gain), is k_norm");
    check(o.k_norm - o.q_norm == 320, "qk gains: the two blocks are 320 B apart, q first");
    // the two windows are disjoint and inside the gap: neither a copy of the other nor a read past o_proj's start
    check(o.q_norm + 256 <= o.k_norm - 64, "qk gains: q's window ends before k's 64 B header starts");
    check(o.q_norm >= 1000 + 64 && o.k_norm + 256 == 1000 + 640, "qk gains: both windows lie inside the 640 B gap");
    // Layer 0's real numbers, so the arm is tied to the measurement rather than to a round example.
    const pi::QkGainOffsets hy = pi::qk_gain_offsets(13795392, 128, 640);
    check(hy.q_norm == 13795456 && hy.k_norm == 13795776,
          "qk gains: Hunyuan-7B layer 0 -- q_norm at 13,795,456 and k_norm at 13,795,776, where its graph's bytes are");
    const pi::QkGainOffsets n1 = pi::qk_gain_offsets(1000, 128, 639);
    const pi::QkGainOffsets n2 = pi::qk_gain_offsets(1000, 128, 641);
    const pi::QkGainOffsets n3 = pi::qk_gain_offsets(1000, 128, 512);   // just the two gains, no headers
    const pi::QkGainOffsets n4 = pi::qk_gain_offsets(1000, 256, 640);   // the right byte count at the wrong head_dim
    check(n1.k_norm == 0 && n1.q_norm == 0, "qk gains: a gap one byte short is not the gains' shape");
    check(n2.k_norm == 0 && n2.q_norm == 0, "qk gains: a gap one byte long is not either");
    check(n3.k_norm == 0 && n3.q_norm == 0, "qk gains: two bare gains with no headers are not this shape");
    check(n4.k_norm == 0 && n4.q_norm == 0, "qk gains: the byte count alone is not enough, head_dim has to agree");
}

// engine#1889, the pack whose q_proj is wider than hidden (Hunyuan-0.5B-Instruct: hidden 1024, q_out 2048). Its
// one-header attention section is [64 q/k/v tiles at K=1024][640 B][16 o_proj tiles at K=2048]; the reader placed
// o_proj and left the gains unnamed, so the pack was refused as "gains are not where the block layout says". The
// numbers are the pack's own: payload 3,691,136 B from attn_off 2,112, tiles of 39,168 and 73,984 B, and the gains'
// bytes found in the .fconst from the graph's initializers at 2,508,928 (q) and 2,509,248 (k) in every block.
void test_wide_q_attn_section() {
    const pi::WideQAttnSection s = pi::wide_q_attn_section(2112, 3691136, 64, 39168, 16, 73984, 128);
    check(s.ok && s.gap == 640, "wide-q section: Hunyuan-0.5B's remainder is the 640 B gap");
    check(s.o_off == 2112 + 64 * 39168 + 640, "wide-q section: o_proj starts after the gap");
    check(s.q_norm == 2508928, "wide-q section: q_norm is named, where the graph's bytes are (engine#1889)");
    check(s.k_norm == 2509248, "wide-q section: k_norm is named, 320 B after q_norm");
    check(pi::qk_norm_refusal(true, s.k_norm != 0).empty(),
          "wide-q section: a pack declaring use_qk_norm is therefore not refused for unlocated gains");
    // a pack of the same shape with no gap (no per-head norms): read, and no gain is invented
    const pi::WideQAttnSection z = pi::wide_q_attn_section(2112, 3691136 - 640, 64, 39168, 16, 73984, 128);
    check(z.ok && z.gap == 0 && z.q_norm == 0 && z.k_norm == 0 && z.o_off == 2112 + 64 * 39168,
          "wide-q section: no gap is read as no gains, o_proj directly after q/k/v");
    // a norms-sized gap that is not the gains' shape: the section is read, the gains stay unnamed (so a pack that
    // declares them is still refused rather than handed tile bytes)
    const pi::WideQAttnSection odd = pi::wide_q_attn_section(2112, 3691136 - 128, 64, 39168, 16, 73984, 128);
    check(odd.ok && odd.gap == 512 && odd.q_norm == 0 && odd.k_norm == 0,
          "wide-q section: a 512 B gap is not the [header][gain] pair and names nothing");
    check(!pi::qk_norm_refusal(true, odd.k_norm != 0).empty(), "wide-q section: and such a pack is still refused");
    // not this layout at all: a tile too many, or a payload shorter than the tiles
    check(!pi::wide_q_attn_section(2112, 3691136 + 39168, 64, 39168, 16, 73984, 128).ok,
          "wide-q section: a remainder of a whole tile is not a norms gap");
    check(!pi::wide_q_attn_section(2112, 3691136 - 641, 64, 39168, 16, 73984, 128).ok,
          "wide-q section: a payload shorter than the tiles is refused");
    check(!pi::wide_q_attn_section(2112, 0, 64, 39168, 16, 73984, 128).ok, "wide-q section: an empty payload is refused");
    // the same layout with one more [header][gain] section (the gains displaced by 320 B): not a norms gap, so the
    // caller falls through to the refusals it had before
    check(!pi::wide_q_attn_section(2112, 3691136 + 320, 64, 39168, 16, 73984, 128).ok,
          "wide-q section: a 960 B remainder is not this layout");
    // the offsets are arithmetic on the arguments, not this pack's constants: another head_dim and tile count
    const pi::WideQAttnSection h64 = pi::wide_q_attn_section(1000, 10 * 500 + 384 + 4 * 900, 10, 500, 4, 900, 64);
    check(h64.ok && h64.gap == 384 && h64.q_norm == 1000 + 5000 + 64 && h64.k_norm == h64.q_norm + 128 + 64 &&
              h64.o_off == 1000 + 5000 + 384,
          "wide-q section: at head_dim 64 the gap is 384 B and every offset follows the arguments");
}

// Review of engine#2063: the gap's offsets are one number for every layer, so the import checks each layer's bytes to be
// the [64 B zeros][gain][64 B zeros][gain] pair before writing a gain, and refuses the layer otherwise.
void test_qk_gap_block_why() {
    std::vector<uint8_t> gap(640, 0);
    auto put = [&](size_t off, uint16_t v) { gap[off] = uint8_t(v & 0xFF); gap[off + 1] = uint8_t(v >> 8); };
    for (int i = 0; i < 128; ++i) {
        put(64 + 2 * size_t(i), 0x3F80);        // q_norm = 1.0
        put(384 + 2 * size_t(i), 0x3FC0);       // k_norm = 1.5
    }
    check(pi::qk_gap_block_why(gap.data(), 128).empty(), "qk gap: two zero headers and two finite gains are the pair");
    // the windows displaced by one section (320 B): q's window is then k's header + gain start, as a reader that
    // placed the gains one section late would see it
    std::vector<uint8_t> shifted(640, 0);
    std::copy(gap.begin() + 320, gap.end(), shifted.begin());   // [hdr][k] then zeros where k should be
    const std::string a = pi::qk_gap_block_why(shifted.data(), 128);
    check(!a.empty() && a.find("k_norm is all zero") != std::string::npos,
          "qk gap: a gain displaced by 320 B is refused, naming the gain that is not there");
    std::vector<uint8_t> hdr = gap;
    hdr[320 + 7] = 1;
    const std::string b = pi::qk_gap_block_why(hdr.data(), 128);
    check(b.find("before k_norm are not zero") != std::string::npos && b.find("byte 7") != std::string::npos,
          "qk gap: a non-zero header byte is refused with its index");
    std::vector<uint8_t> inf = gap;
    inf[64 + 2 * 5] = 0x80; inf[64 + 2 * 5 + 1] = 0x7F;       // +inf in element 5 of q_norm
    const std::string c = pi::qk_gap_block_why(inf.data(), 128);
    check(c.find("q_norm element 5 is not finite") != std::string::npos, "qk gap: a non-finite gain is refused by element");
    std::vector<uint8_t> tile(640);
    for (size_t i = 0; i < tile.size(); ++i) tile[i] = uint8_t(17 * i + 3);   // tile-like bytes where the gap should be
    check(!pi::qk_gap_block_why(tile.data(), 128).empty(), "qk gap: tile bytes are not the pair");
}

/* engine#1985: a REGRESSION, not a new bug -- `Qwen3-1.7B` imported with 0 gaps on yesterday's main and refused with 280
 * on today's, because #1838's early return trusted an epsilon-anchor vote that had recorded ONE in-block header on a block
 * that has three. The tell in the report is `qk_norm: true -> false`: a pack that declares per-head q/k gains keeps them in
 * the two sections a 2-header block does not have, so the two witnesses disagree and neither may be preferred silently.
 *
 * The arms below pin the refusal and, just as importantly, the three cases it must NOT fire on: a vote that found three
 * headers (a correct 4-header pack), a genuine 2-header pack that declares nothing, and no vote at all (0), which is the
 * residue step's case and not a disagreement. */
void test_header_witness_refusal() {
    const std::string a = pi::header_witness_refusal(1, true);
    check(!a.empty(), "header witness: a one-header vote on a pack declaring use_qk_norm is refused");
    check(a.find("engine#1985") != std::string::npos, "header witness: the refusal carries its issue");
    check(a.find("use_qk_norm") != std::string::npos, "header witness: it names the declaration it contradicts");
    check(a.find("ONE in-block header") != std::string::npos, "header witness: and what the vote measured");
    check(a.find("neither is being preferred") != std::string::npos,
          "header witness: and that no witness is being chosen, which is the point");
    check(pi::header_witness_refusal(3, true).empty(), "header witness: a three-header vote is consistent with the declaration");
    check(pi::header_witness_refusal(1, false).empty(), "header witness: a 2-header pack that declares nothing is fine");
    check(pi::header_witness_refusal(0, true).empty(),
          "header witness: NO vote is not a disagreement (the residue step decides it)");
    std::printf("  header witness: 8 arms (one-header vote vs declaration refuses; 3/1-without-declaration/0 allowed)\n");
}

// engine#1889: which tile each projection starts at inside the attention payload. Every family before Hunyuan is
// k, q, v; the Hunyuan block is q, k, v, and reading it k, q, v handed the forward q_proj's first rows as k_proj
// (byte-exact round trip, filler-word answers). Hunyuan-7B's own counts: 16 k/v tiles, 64 q tiles.
void test_attn_first_tiles() {
    const pi::AttnFirstTiles kqv = pi::attn_first_tiles(16, 64, false);
    check(kqv.k == 0 && kqv.q == 16 && kqv.v == 80, "attn order: k,q,v puts k at tile 0, q at 16, v at 80");
    const pi::AttnFirstTiles qkv = pi::attn_first_tiles(16, 64, true);
    check(qkv.q == 0 && qkv.k == 64 && qkv.v == 80, "attn order: q,k,v puts q at tile 0, k at 64, v at 80 (engine#1889)");
    check(qkv.k + 16 == qkv.v && kqv.q + 64 == kqv.v, "attn order: v follows the second run in both orders");
    const pi::AttnFirstTiles small = pi::attn_first_tiles(2, 14, false);   // Qwen2.5-0.5B: k 2, q 14, v 2
    check(small.k == 0 && small.q == 2 && small.v == 16, "attn order: the Qwen2.5-0.5B run (2/14/2) is unchanged");
    pi::Plan p;
    check(!p.attn_q_first, "attn order: a plan is k,q,v unless the block measured otherwise");

    // Review of engine#2063: the line that makes the wide-q container right is the tile ORDER, and nothing guarded it.
    // Hunyuan-0.5B's own section (hidden 1024, q_out 2048: 32 q, 16 k and 16 v tiles of 64 rows), with its gains in the gap.
    const pi::WideQAttnSection wq = pi::wide_q_attn_section(2112, 3691136, 64, 39168, 16, 73984, 128);
    pi::Plan w;
    check(pi::apply_wide_q_section(w, wq, 16, 32, 2048, 1024), "wide-q plan: Hunyuan-0.5B's section is taken");
    check(w.attn_q_first, "wide-q plan: a wide-q block with its gains in the gap is q,k,v (engine#1889, review of #2063)");
    const pi::AttnFirstTiles wt = pi::attn_first_tiles(int(16), int(32), w.attn_q_first);
    check(wt.q == 0 && wt.k == 32 && wt.v == 48,
          "wide-q plan: q at tile 0, k at 32, v at 48 -- the rows the checkpoint matches (q 0-31, k 32-47, v 48-63)");
    check(w.attn_tiles == 64 && w.o_k == 2048 && w.o_off == wq.o_off && w.qnorm_off == 2508928 && w.knorm_off == 2509248,
          "wide-q plan: 64 q/k/v tiles, o_proj at K=q_out after the gap, both gains named");
    // the same shape without gains in the gap: the section is still read, the order is not claimed
    const pi::WideQAttnSection wz = pi::wide_q_attn_section(2112, 3691136 - 640, 64, 39168, 16, 73984, 128);
    pi::Plan z;
    check(pi::apply_wide_q_section(z, wz, 16, 32, 2048, 1024) && !z.attn_q_first && z.knorm_off == 0,
          "wide-q plan: no gains in the gap, so no q,k,v claim (k,q,v stays the default)");
    // q no wider than hidden is not this branch: the plan is left as it was
    pi::Plan n;
    check(!pi::apply_wide_q_section(n, wq, 16, 32, 1024, 1024) && !n.attn_q_first && n.attn_tiles == 0,
          "wide-q plan: q_out == hidden is not the wide-q branch, and the plan is untouched");
    pi::Plan bad;
    check(!pi::apply_wide_q_section(bad, pi::WideQAttnSection{}, 16, 32, 2048, 1024) && !bad.attn_q_first,
          "wide-q plan: a section that was not read is not applied");
}

// engine#1503: the tied-head rule as ONE predicate -- correlated AND not mis-scaled. Two packs on this box correlate
// (0.9915 / 0.9961) while being scaled copies of their own embedding (rel_err 0.1361 / 0.0949): they import cleanly and
// serve garbage, where the pack that serves correctly reports rel_err 0.000000. Correlation alone cannot tell them apart,
// which is why the correlation-only check passed both and cost two device tickets.
static void selfcheck_head_rule_arms() {
    check(pi::selfcheck_head_ok(0.996901, 0.000000), "head rule: the pack that serves correctly passes (corr .9969, rel 0)");
    check(!pi::selfcheck_head_ok(0.991539, 0.136140), "head rule: a correlated but scaled head is refused (corr .9915, rel .1361)");
    check(!pi::selfcheck_head_ok(0.996053, 0.094874), "head rule: the second measured pack is refused too (corr .9961, rel .0949)");
    check(pi::selfcheck_head_ok(0.996053, 0.000000),
          "head rule: correlated and unscaled passes -- the rule is not 'refuse everything'");
    check(!pi::selfcheck_head_ok(0.50, 0.000000), "head rule: an uncorrelated head is still refused on correlation");
    check(pi::selfcheck_head_ok(0.996053, pi::kSelfcheckRelMax), "head rule: exactly at the threshold passes (inclusive bound)");
    std::printf("  head rule: 6 arms (two measured bad packs, the serving one, the unscaled control, no-corr, boundary)\n");

    // engine#1503: the rule on what it claims (scale, rms ratio, rel against the embedding's own 4-bit noise floor).
    // Measured fits, from the import's own numbers:
    const pi::HeadFit qwen2_15b{0.094316, 0.994173, 1.001808, 0.083865};   // Qwen2-1.5B 1.9, block-32 tail head
    const pi::HeadFit gemma_awq{0.094874, 0.992235, 1.003849, 0.088985};   // gemma-2-2b AWQ hybrid, graph head
    check(!pi::selfcheck_head_ok(0.995969, qwen2_15b.rel),
          "fit rule: the rel-only rule refuses Qwen2-1.5B's head (the reason this rule exists)");
    check(pi::selfcheck_head_ok(0.995969, qwen2_15b),
          "fit rule: Qwen2-1.5B's head passes: scale .994, rms 1.002, rel 1.12x its 4-bit floor");
    check(pi::selfcheck_head_ok(0.996053, gemma_awq),
          "fit rule: gemma-2-2b AWQ's head passes too -- by these numbers its head is not what served garbage");
    check(pi::selfcheck_head_ok(0.996901, pi::HeadFit{0.0, 1.0, 1.0, 0.08}),
          "fit rule: a head that IS its embedding (rel 0) passes");
    check(!pi::selfcheck_head_ok(0.996, pi::HeadFit{0.09, 0.5, 2.0, 0.08}), "fit rule: a head scaled by 2 is refused");
    check(!pi::selfcheck_head_ok(0.996, pi::HeadFit{0.09, 0.94, 1.0, 0.08}), "fit rule: scale 0.94 (outside 5 %) is refused");
    check(!pi::selfcheck_head_ok(0.996, pi::HeadFit{0.09, 1.0, 1.06, 0.08}), "fit rule: rms ratio 1.06 is refused");
    check(!pi::selfcheck_head_ok(0.996, pi::HeadFit{0.13, 1.0, 1.0, 0.08}),
          "fit rule: rel 1.6x the floor is refused (a partial decode is noisier than quantisation)");
    check(pi::selfcheck_head_ok(0.996, pi::HeadFit{0.12, 1.0, 1.0, 0.08}), "fit rule: exactly 1.5x the floor passes");
    check(!pi::selfcheck_head_ok(0.50, qwen2_15b), "fit rule: an uncorrelated head is still refused on correlation");

    // fit_head itself, on a synthetic table: the head is the embedding quantised asymmetric 4-bit at block 32, the way
    // a correct pack's head is, so the fit must read scale ~1 and rel == floor; the same head scaled by 0.5 must not.
    const int N = 64, K = 128;
    std::vector<uint8_t> emb(size_t(N) * K * 2);
    std::vector<float> ef(size_t(N) * K);
    uint32_t st = 12345;
    for (size_t i = 0; i < ef.size(); ++i) {
        st = st * 1664525u + 1013904223u;
        float v = (float(st >> 8) / float(1u << 24) - 0.5f) * 0.2f;
        uint32_t b;
        std::memcpy(&b, &v, 4);
        const uint16_t h = uint16_t(b >> 16);   // bf16: the top half, so ef[] holds exactly what the table says
        b = uint32_t(h) << 16;
        std::memcpy(&v, &b, 4);
        ef[i] = v;
        emb[2 * i] = uint8_t(h & 0xff);
        emb[2 * i + 1] = uint8_t(h >> 8);
    }
    pi::EmbedTable et;
    et.p = emb.data();
    et.H = K;
    pi::Q4 q;
    q.N = N; q.K = K; q.block = 32;
    q.codes.assign(size_t(N) * K, 0);
    q.scale.assign(size_t(N) * (K / 32), 0.f);
    q.zero.assign(size_t(N) * (K / 32), 0.f);
    for (int n = 0; n < N; ++n)
        for (int g = 0; g < K / 32; ++g) {
            float lo = 1e30f, hi = -1e30f;
            for (int k = 0; k < 32; ++k) { lo = std::min(lo, ef[size_t(n) * K + g * 32 + k]); hi = std::max(hi, ef[size_t(n) * K + g * 32 + k]); }
            const float sc = (hi - lo) / 15.f;
            q.scale[size_t(n) * (K / 32) + g] = sc;
            q.zero[size_t(n) * (K / 32) + g] = lo;
            for (int k = 0; k < 32; ++k)
                q.codes[size_t(n) * K + g * 32 + k] =
                    uint8_t(std::min(15.0, std::max(0.0, std::round((double(ef[size_t(n) * K + g * 32 + k]) - lo) / sc))));
        }
    const pi::HeadFit good = pi::fit_head(q, et, N, 32);
    check(std::fabs(good.scale - 1) < 0.01 && std::fabs(good.rms_ratio - 1) < 0.01 && good.rel > 0 &&
              std::fabs(good.rel - good.floor) < 1e-6 && pi::selfcheck_head_ok(0.999, good),
          "fit_head: a correct 4-bit head reads scale ~1, rel == its floor, and passes (scale " +
              std::to_string(good.scale) + " rel " + std::to_string(good.rel) + " floor " + std::to_string(good.floor) + ")");
    // The floor follows the head's block (the block-128 tail head passes 128): the same table quantised at block 128
    // must read rel == its block-128 floor, which is LARGER than the block-32 floor of the same table.
    pi::Q4 q128 = q;
    for (int n = 0; n < N; ++n)
        for (int g = 0; g < K / 128; ++g) {
            float lo = 1e30f, hi = -1e30f;
            for (int k = 0; k < 128; ++k) { lo = std::min(lo, ef[size_t(n) * K + g * 128 + k]); hi = std::max(hi, ef[size_t(n) * K + g * 128 + k]); }
            const float sc = (hi - lo) / 15.f;
            for (int j = 0; j < 4; ++j) {   // a 128-block's scale and zero apply to its four 32-column groups
                q128.scale[size_t(n) * (K / 32) + g * 4 + j] = sc;
                q128.zero[size_t(n) * (K / 32) + g * 4 + j] = lo;
            }
            for (int k = 0; k < 128; ++k)
                q128.codes[size_t(n) * K + g * 128 + k] =
                    uint8_t(std::min(15.0, std::max(0.0, std::round((double(ef[size_t(n) * K + g * 128 + k]) - lo) / sc))));
        }
    const pi::HeadFit f128 = pi::fit_head(q128, et, N, 128);
    check(std::fabs(f128.rel - f128.floor) < 1e-6 && f128.floor > good.floor && pi::selfcheck_head_ok(0.999, f128),
          "fit_head: at block 128 the floor is the block-128 one (rel " + std::to_string(f128.rel) + " == floor " +
              std::to_string(f128.floor) + " > block-32 floor " + std::to_string(good.floor) + ")");
    pi::Q4 half = q;
    for (auto& v : half.scale) v *= 0.5f;
    for (auto& v : half.zero) v *= 0.5f;
    const pi::HeadFit bad = pi::fit_head(half, et, N, 32);
    check(std::fabs(bad.scale - 2) < 0.05 && !pi::selfcheck_head_ok(0.999, bad),
          "fit_head: the same head at half scale reads scale ~2 and is refused (scale " + std::to_string(bad.scale) + ")");
    check(pi::known_bad_architecture("gemma2") != nullptr &&
              std::string(pi::known_bad_architecture("gemma2")).find("not yet found") != std::string::npos,
          "known-bad: gemma2 stays refused, by name and with the reason (the head fit no longer refuses it)");
    check(pi::known_bad_architecture("qwen2") == nullptr && pi::known_bad_architecture("llama") == nullptr &&
              pi::known_bad_architecture("") == nullptr,
          "known-bad: nothing else is on the list");
    std::printf("  head fit rule: 15 arms (two measured heads, scaled/rms/noisy/no-corr controls, fit_head at blocks 32 and 128, gemma2 by name)\n");
}

// engine#1788, slice 3c: the writer's destination side. The arm with teeth is the last one: a role the classifier can
// emit must be EITHER placed or explained, so adding a recogniser without deciding its destination fails here rather
// than in a pack that loads a tensor into the wrong slot.
static void test_container_role_slots() {
    using namespace onebit::npu::pack_import;
    {
        ContainerRoles r;
        r.counts["q_proj"] = 1; r.counts["gate_exps"] = 1; r.counts["input_norm"] = 1;
        const RoleSlotReport s = container_role_slots(r);
        check(s.slot_of.at("q_proj") == "self_attn.q_proj.weight", "slots: the attention query projection");
        check(s.slot_of.at("gate_exps") == "mlp.gate_exps_proj.weight", "slots: the routed gate, in the CONTAINER (FLM) spelling");
        check(s.slot_of.at("input_norm") == "input_layernorm.weight", "slots: the pre-attention norm");
        check(s.no_slot.empty(), "slots: three placed roles leave nothing unexplained");
    }
    {
        // the layer type decides exactly one destination, and it decides it the way the ENGINE does: the only fused-qkv
        // slot npu/forward/src reads is the linear-attention one, so a full-attention qkv has no destination at all.
        check(container_role_slot("qkv_proj", "linear_attention") == "linear_attn.qkv_proj.weight",
              "slots: the fused qkv on a linear layer");
        check(container_role_slot("qkv_proj", "full_attention").empty(),
              "slots: a full-attention qkv has no engine slot, so it is not placed");
        check(container_role_slot("q_proj", "full_attention") == "self_attn.q_proj.weight",
              "slots: the layer type does not move a role it does not decide");
        check(container_role_slot("qkv_proj", "sliding_attention").empty(),
              "slots: sliding attention is not linear, so the fused qkv stays unplaced");
        // the arm for the substring defect: a value that CONTAINS "linear" and is not linear attention must not match
        check(container_role_slot("qkv_proj", "nonlinear_attention").empty(),
              "slots: a layer type that merely contains 'linear' is not linear attention");
        check(container_role_slot("qkv_proj", "linear_sliding_attention").empty(),
              "slots: neither is a compound name containing it");
    }
    {
        // no two roles may claim the same slot: a collision silently drops one of them
        std::map<std::string, std::string> seen;
        for (const auto& kv : container_role_slots(ContainerRoles{}).slot_of) {
            check(!seen.count(kv.second), "slots: no two roles share a destination (" + kv.second + ")");
            seen[kv.second] = kv.first;
        }
    }
    {
        // qkv_proj is the one role everybody gets wrong: its PREFIX is the layer type's, so this slice must refuse it
        ContainerRoles r; r.counts["qkv_proj"] = 1;
        const RoleSlotReport s = container_role_slots(r);
        check(!s.slot_of.count("qkv_proj"), "slots: the fused qkv is not placed without the layer type");
        check(s.why.count("qkv_proj") && s.why.at("qkv_proj").size() > 20, "slots: and the refusal says why");
    }
    {
        // the classifier's whole vocabulary: every role either placed or explained, none unknown
        static const char* kRoles[] = {"embed", "final_norm", "input_norm", "post_norm", "lm_head", "q_proj", "k_proj",
            "v_proj", "o_proj", "q_norm", "k_norm", "self_attn_gate_proj", "gate_proj", "up_proj", "down_proj",
            "gate_up_proj", "qkv_proj", "moe_router", "gate_exps", "up_exps", "down_exps", "share_gate_exps",
            "share_up_exps", "share_down_exps", "shared_expert_gate", "shortconv_in_proj"};
        ContainerRoles r;
        for (const char* k : kRoles) r.counts[k] = 1;
        const RoleSlotReport s = container_role_slots(r);
        check(int64_t(s.placed.size() + s.no_slot.size()) == int64_t(sizeof(kRoles) / sizeof(kRoles[0])),
              "slots: every role the classifier can emit is accounted for");
        for (const auto& kv : s.why) {
            check(kv.second.find("does not know") == std::string::npos,
                  "slots: no role falls through the table (" + kv.first + ")");
        }
        check(s.placed.size() >= 17, "slots: at least the layer-type-independent ones are placed");
    }
}

int main(int argc, char** argv) {
    selfcheck_head_rule_arms();
    // engine#1907: every crafted 1.9 import writes a canonical copy into the scratch cache. Without this the suite left 0-byte
    // copies in the user's real ~/.cache/1bit/fconst19 and in /tmp on every run. Give the run its own directory, removed at exit.
    const fs::path run_scratch = ::onebit::test::tmp_root() / ("1bit-pack-test-scratch-" + std::to_string(::getpid()));
    if (!std::getenv("ONEBIT_FCONST_SCRATCH")) setenv("ONEBIT_FCONST_SCRATCH", run_scratch.c_str(), 1);
    struct ScratchCleanup {
        fs::path p;
        ~ScratchCleanup() { std::error_code ec; fs::remove_all(p, ec); }
    } scratch_cleanup{run_scratch};
    test_header_witness_refusal();
    test_encode_tile_code_range();
    test_q4_writer_code_range();
    test_attn_first_tiles();
    test_qk_norm_refusal();
    test_head_stored_rows();
    test_qk_gain_offsets();
    test_wide_q_attn_section();
    test_qk_gap_block_why();
    test_tail_head_placement();
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
        test_external_tile_decode();
    test_external_to_container();
    test_external_roles();
    test_external_container_names();
    test_external_packed_genre(golden_dir.empty() ? "." : golden_dir);
    test_external_import_success(golden_dir.empty() ? "." : golden_dir);
        test_write_canonical_copy();
        test_canonical_copy_reusable();        test_evict_stale_canonical_copies();
        test_implicit_zp_bits_follow_the_spelling();
        test_sweep_fconst_scratch_dirs();
        test_sweep_is_wired(golden_dir.empty() ? "." : golden_dir);
        test_crafted_refusals(golden_dir.empty() ? "." : golden_dir);
        test_modelbin_genre_refusal(golden_dir.empty() ? "." : golden_dir);
        test_1671_plan_head_arm(golden_dir.empty() ? "." : golden_dir);
        test_head_row_padding();
    test_matmulnbits();
        test_tiles(golden_dir);
        test_q4_1f();
        test_attn_tile(golden_dir);
        test_mlp_gu(golden_dir);
        test_fconst_eps(golden_dir.empty() ? "." : golden_dir);
        test_fconst_variable_stride(golden_dir.empty() ? "." : golden_dir);
        test_attn_payload_off();
        test_text_config_geometry(golden_dir.empty() ? "." : golden_dir);
        test_q4nx_container(golden_dir.empty() ? "." : golden_dir);
        test_container_geometry();
        test_container_roles();
        test_container_roles_gemma_local();
        test_container_roles_moe_and_singular_layer();
        test_container_placement();
        test_scale_and_proj_norm_roles();
        test_role_tables_disagree_nowhere();
        if (argc > 4) {
            std::vector<std::string> lt;
            std::string all = argv[4];
            std::string cur;
            for (char ch : all) { if (ch == ',') { lt.push_back(cur); cur.clear(); } else cur += ch; }
            if (!cur.empty()) lt.push_back(cur);
            test_container_placement_real(argv[3], lt);
        }
        test_container_role_slots();
        test_container_role_layer_types();
        test_container_plan(golden_dir.empty() ? "." : golden_dir);
        test_hybrid_pack_import(golden_dir.empty() ? "." : golden_dir);
        test_rope_parameters_theta(golden_dir.empty() ? "." : golden_dir);
        test_three_input_matmulnbits(golden_dir.empty() ? "." : golden_dir);
        test_genre_a_projection_gap(golden_dir.empty() ? "." : golden_dir);
        test_embedding_component(golden_dir.empty() ? "." : golden_dir);
        test_fconst19_rule_table();
        test_fconst_scratch_order();
        test_fconst_block_headers();
        test_selfcheck_reads_bf16();
        if (const char* rc = std::getenv("NPU_EMBED_COMPONENT")) dump_real_component(rc);
        test_model_config_geometry(golden_dir.empty() ? "." : golden_dir);
        test_im_from_node_attrs(golden_dir.empty() ? "." : golden_dir);
        test_fconst_in_component(golden_dir.empty() ? "." : golden_dir);
        test_embedding_in_component(golden_dir.empty() ? "." : golden_dir);
        test_per_layer_container(golden_dir.empty() ? "." : golden_dir);
        test_per_layer_import(golden_dir.empty() ? "." : golden_dir);
        test_embedding_f32_component(golden_dir.empty() ? "." : golden_dir);
        test_qwen3_attn_sections();
        test_post_attn_norm_header(golden_dir.empty() ? "." : golden_dir);
        test_tied_block32_head(golden_dir.empty() ? "." : golden_dir);
        test_block32_tail_head(golden_dir.empty() ? "." : golden_dir);
        test_block32_layers(golden_dir.empty() ? "." : golden_dir);
        test_final_norm_envelope(golden_dir.empty() ? "." : golden_dir);
        test_import_fp32_params(golden_dir.empty() ? "." : golden_dir);
        test_text_artifacts(golden_dir.empty() ? "." : golden_dir);
        test_copy_text_artifacts();
        test_kernel_set_refusal();
        test_weight_layout();
        test_null_optional_dims();
        // The pack leg gets its OWN catch, so one unreadable pack is a counted failed check rather than the end of the
        // run. Before this, a throw here reached the handler below and printed "FAIL exception:", skipping the summary
        // that says which legs ran -- which is how pack-leg breakage stayed invisible behind a green line (engine#842,
        // engine#1671: the 0.5B-Instruct 1.9 pack's placeholders threw "no data" and nothing after it ran).
        if (pack_ran) {
            try {
                test_pack(pack, golden_dir);
            } catch (const std::exception& e) {
                check(false, std::string("pack leg: ") + e.what());
            }
        } else {
            std::printf("skip pack: no pack directory given (the crafted and golden legs only)\n");
        }
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
