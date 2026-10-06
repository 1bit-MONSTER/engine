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

// npu_model_bounds_test
//
// The Q4NX loader (npu/model.cpp) and the fast-lane weight packer (npu/pack.cpp)
// against small crafted model directories, written to a temporary directory:
// a well-formed one-layer model loads and packs; a header length past the end of
// the file, data_offsets whose sum with the data start wraps around, and a
// projection whose shape asks for more tiles than the file holds are refused
// with an error instead of reading outside the mapping.
#include "model.h"
#include "pack.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include <unistd.h>

using namespace onebit::npu;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// Runs fn and returns the exception text, or "" when it did not throw.
std::string thrown(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

struct Entry {
    std::string name;
    std::vector<int64_t> shape;
    uint64_t bytes;
};

// Writes dir/model.q4nx (tensors packed in order, zero-filled) and dir/config.json.
// `header_len` / `offsets` override the honest values when set.
void write_model(const fs::path& dir, const std::vector<Entry>& entries, uint64_t header_len = 0,
                 const json& bad_offsets = nullptr, const std::string& bad_name = "") {
    fs::create_directories(dir);
    json table = json::object();
    uint64_t off = 0;
    for (const auto& e : entries) {
        table[e.name] = {{"dtype", "U8"}, {"shape", e.shape}, {"data_offsets", {off, off + e.bytes}}};
        off += e.bytes;
    }
    if (!bad_name.empty()) table[bad_name]["data_offsets"] = bad_offsets;
    const std::string h = table.dump();
    const uint64_t len = header_len ? header_len : h.size();
    std::ofstream f(dir / "model.q4nx", std::ios::binary);
    f.write(reinterpret_cast<const char*>(&len), 8);
    f << h;
    const std::vector<char> zeros(off, 0);
    f.write(zeros.data(), long(zeros.size()));
    std::ofstream(dir / "config.json") << json{{"hidden_size", 128},
                                               {"num_hidden_layers", 1},
                                               {"num_attention_heads", 1},
                                               {"num_key_value_heads", 1},
                                               {"head_dim", 128},
                                               {"intermediate_size", 128}}
                                              .dump();
}

// A one-layer model: every projection is one 5120-byte tile, except gate/up, which fill one
// 8-tile chunk each (hidden 128 / 16); q_proj is last in the file.
std::vector<Entry> one_layer(std::vector<int64_t> q_shape = {1, 5120}, int64_t gu_tiles = 8) {
    const std::string L = "model.layers.0.";
    std::vector<Entry> e = {
        {"model.embed_tokens.weight", {2, 128}, 512},
        {"model.norm.weight", {128}, 256},
        {"lm_head.weight", {1, 5120}, 5120},
        {L + "input_layernorm.weight", {128}, 256},
        {L + "post_attention_layernorm.weight", {128}, 256},
    };
    for (const char* p : {"self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                          "mlp.down_proj.weight"})
        e.push_back({L + p, {1, 5120}, 5120});
    for (const char* p : {"mlp.up_proj.weight", "mlp.gate_proj.weight"})
        e.push_back({L + p, {gu_tiles, 5120}, uint64_t(gu_tiles) * 5120});
    e.push_back({L + "self_attn.q_proj.weight", q_shape, 5120});
    return e;
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / ("npu_model_bounds_test." + std::to_string(::getpid()));
    fs::remove_all(root);

    {  // well-formed: loads and packs
        write_model(root / "good", one_layer());
        std::string why = thrown([&] {
            Model m((root / "good").string());
            std::vector<uint8_t> w(layer_weight_bytes(m)), i5(1024), lm(lmhead_weight_bytes(m));
            pack_layer_weights(m, 0, w.data());
            fill_i5(m, 0, i5.data());
            pack_lmhead_weights(m, lm.data());
        });
        check(why.empty(), "a well-formed model loads and packs" + (why.empty() ? "" : " (" + why + ")"));
    }
    {  // header length past the end of the file, close enough to 2^64 that 8 + len wraps
        write_model(root / "header", one_layer(), ~uint64_t(0) - 4);
        const std::string why = thrown([&] { Model m((root / "header").string()); });
        check(why.find("header past the end") != std::string::npos, "an oversized header length is refused");
    }
    {  // data_offsets whose sum with the data start wraps below the file size
        write_model(root / "wrap", one_layer(), 0, json::array({~uint64_t(0) - 5000, ~uint64_t(0) - 100}),
                    "model.layers.0.self_attn.q_proj.weight");
        const std::string why = thrown([&] { Model m((root / "wrap").string()); });
        check(why.find("bad data_offsets") != std::string::npos, "wrapping data_offsets are refused");
    }
    {  // end offset past the end of the file
        write_model(root / "past", one_layer(), 0, json::array({0, 1u << 30}), "model.norm.weight");
        const std::string why = thrown([&] { Model m((root / "past").string()); });
        check(why.find("bad data_offsets") != std::string::npos, "data_offsets past the end are refused");
    }
    {  // q_proj claims four tiles, the file holds one: the packer must not read past the file
        write_model(root / "tiles", one_layer({4, 5120}));
        const std::string why = thrown([&] {
            Model m((root / "tiles").string());
            std::vector<uint8_t> w(layer_weight_bytes(m));
            pack_layer_weights(m, 0, w.data());
        });
        check(why.find("runs past the end") != std::string::npos, "a projection larger than the file is refused");
    }
    {  // gate/up of one tile each: the interleave would write a whole chunk past the layer's buffer
        write_model(root / "chunks", one_layer({1, 5120}, 1));
        const std::string why = thrown([&] {
            Model m((root / "chunks").string());
            std::vector<uint8_t> w(layer_weight_bytes(m));
            pack_layer_weights(m, 0, w.data());
        });
        check(why.find("whole chunks") != std::string::npos, "gate/up that do not fill whole chunks are refused");
    }
    fs::remove_all(root);
    std::printf("%s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 1 : 0;
}
