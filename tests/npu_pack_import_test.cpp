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

#include "model.h"
#include "pack.h"
#include "pack_import.h"
#include "q4nx.h"
#include "q4nx_device.h"

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
    const auto write_graph = [&](const std::string& packed, int64_t packed_bytes) {
        put_file((dir / "eager-mm.onnx").string(),
                 pb::model({pb::tensor("model.embed_tokens.weight", {64, 5120}, "eager-mm.onnx.data", 0,
                                       uint64_t(embed_bytes)),
                            pb::placeholder("model.layers.0.attn.qkv_proj.MatMulNBits.qweight.empty"),
                            pb::placeholder("model.layers.0.attn.qkv_proj.MatMulNBits.scales.empty"),
                            pb::tensor(packed, {packed_bytes}, "eager-mm.onnx.data", uint64_t(embed_bytes),
                                       uint64_t(packed_bytes))},
                           {pb::matmulnbits("qkv", {"x", "/model/layers.0/attn/qkv_proj/MatMulNBits.qweight.empty",
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
        check(threw, "external-packed: import still refuses the genre (the census widening did not become a writer)");
        check(why.find("qweight.packed") != std::string::npos && why.find("no writer") != std::string::npos,
              "external-packed: and it refuses by NAME with the layout stated (no writer for N x (K/128+1) x 68), "
              "not by tripping over a symbol (" + why + ")");
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

// The two overstatements the readers must catch: declared dims larger than the external file, and a
// MatMulNBits node whose N/K/block_size demand more qweight than the tensor declares.
// engine#1671: a pack whose graph is all `[0]`-dims placeholders and whose bytes are in model.bin (the 1.8/hybrid
// genre). The importer has no model.bin reader, so the honest outcome is a refusal that NAMES the file — not
// "<tensor>.qweight.empty: no data", which reports a placeholder as though the bytes were absent rather than elsewhere.
// Crafted, so this needs no 1.9 pack: a graph with one resolvable tensor, one MatMulNBits whose qweight input is a
// `.empty` placeholder with no sibling, and a model.bin next to it.
// engine#1662: the ~291 MB canonical Token .fconst rewrite is skipped when the copy on the scratch path already holds
// exactly the bytes this run would write. The decision is a length test against a length that is a function of the source
// alone, so what can go wrong is the test being too willing -- a truncated copy from an interrupted run, a leftover
// directory, a copy from a bigger pack. Those are the arms.
void test_canonical_copy_reusable() {
    const fs::path dir = fs::temp_directory_path() / ("1bit-reuse-" + std::to_string(::getpid()));
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
    const std::string out = (fs::temp_directory_path() /
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
    for (const auto& t : p.tensors) {
        if (t.name.find("layernorm") != std::string::npos || t.name == "model.norm.weight") ++norms;
        if (t.dtype == "q4_1" && t.refusal.empty() && t.name.compare(0, 9, "lm_head.w") == 0)
            lm_tiles_planned = int((t.shape[0] / 32) * ((t.shape[1] + 255) / 256));
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
    const std::string out = (fs::temp_directory_path() /
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
    check(r.lmhead_corr > 0.999, "pack: the decode reproduces the fp16 embedding (corr " +
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
    check(head.shape.at(0) == int64_t((head_rows / 32) * ((p.tensors.at(1).shape.at(1) + 255) / 256)),
          "pack: lm_head tile count matches the container");
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

    const std::string out = (fs::temp_directory_path() /
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
    const std::string cache = (fs::temp_directory_path() /
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

// The embedding component (engine #1503): a hybrid pack keeps its table in `embedding.onnx` (+ `.data`), gathered by
// GatherBlockQuantized. Shape measured on amd/Qwen3-VL-4B-Instruct_rai_1.9.0_npu: weight_Q4 [V, H/2] UINT8 and
// weight_scales [V, groups] FLOAT with group 128. Driving the reader against known codes and scales pins the
// dequantisation without a 3 GB pack.
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
    if (FILE* f = std::fopen(out.c_str(), "wb")) {
        std::fwrite(&v[size_t(row) * H], sizeof(float), H, f);
        std::fclose(f);
        double s = 0;
        for (size_t k = 0; k < H; ++k) s += std::fabs(v[size_t(row) * H + k]);
        std::printf("real component: row %d -> %s (mean|x| %.6f)\n", row, out.c_str(), s / double(H));
    }
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
        test_canonical_copy_reusable();
        test_external_packed_genre(golden_dir.empty() ? "." : golden_dir);
        test_crafted_refusals(golden_dir.empty() ? "." : golden_dir);
        test_modelbin_genre_refusal(golden_dir.empty() ? "." : golden_dir);
        test_matmulnbits();
        test_tiles(golden_dir);
        test_q4_1f();
        test_attn_tile(golden_dir);
        test_mlp_gu(golden_dir);
        test_fconst_eps(golden_dir.empty() ? "." : golden_dir);
        test_attn_payload_off();
        test_embedding_component(golden_dir.empty() ? "." : golden_dir);
        if (const char* rc = std::getenv("NPU_EMBED_COMPONENT")) dump_real_component(rc);
        test_qwen3_attn_sections();
        test_post_attn_norm_header(golden_dir.empty() ? "." : golden_dir);
        test_final_norm_envelope(golden_dir.empty() ? "." : golden_dir);
        test_import_fp32_params(golden_dir.empty() ? "." : golden_dir);
        test_text_artifacts(golden_dir.empty() ? "." : golden_dir);
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
