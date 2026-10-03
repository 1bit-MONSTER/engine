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

// laya_bounds_test
//
// The Laya safetensors reader and scorer loader against small crafted
// checkpoints, written to a temporary directory: a header length or
// data_offsets that wrap past the file, a shape that disagrees with its
// data_offsets, a negative dimension, and tensors whose element counts differ
// from the ModernBERT dims the scorer indexes with are all refused at load,
// instead of being read out of bounds when a request is scored.
#include "safetensors.h"
#include "scorer.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// Writes a safetensors file of F32 tensors {name: (shape, element count)}, zero-filled,
// with optional overrides for the header length and one tensor's data_offsets.
void write_st(const fs::path& path, const std::vector<std::pair<std::string, std::pair<std::vector<int64_t>, size_t>>>& t,
              uint64_t header_len = 0, const std::string& bad = "", const json& bad_offsets = nullptr) {
    json h = json::object();
    uint64_t off = 0;
    for (const auto& [name, v] : t) {
        h[name] = {{"dtype", "F32"}, {"shape", v.first}, {"data_offsets", {off, off + 4 * v.second}}};
        off += 4 * v.second;
    }
    if (!bad.empty()) h[bad]["data_offsets"] = bad_offsets;
    const std::string s = h.dump();
    const uint64_t len = header_len ? header_len : s.size();
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&len), 8);
    f << s;
    const std::vector<char> zeros(off, 0);
    f.write(zeros.data(), long(zeros.size()));
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / ("laya_bounds_test." + std::to_string(::getpid()));
    fs::remove_all(root);
    fs::create_directories(root);
    std::vector<float> out;

    {
        write_st(root / "good.safetensors", {{"w", {{2, 3}, 6}}});
        onebit::laya::Safetensors st;
        check(st.open((root / "good.safetensors").string()) && st.get_f32("w", out) && out.size() == 6,
              "a well-formed tensor reads");
    }
    {
        write_st(root / "hdr.safetensors", {{"w", {{2}, 2}}}, ~uint64_t(0) - 4);
        onebit::laya::Safetensors st;
        check(!st.open((root / "hdr.safetensors").string()), "a header length that wraps is refused");
    }
    {
        write_st(root / "wrap.safetensors", {{"w", {{2}, 2}}}, 0, "w", json::array({~uint64_t(0) - 64, ~uint64_t(0) - 8}));
        onebit::laya::Safetensors st;
        check(!st.open((root / "wrap.safetensors").string()), "data_offsets that wrap are refused");
    }
    {
        // the shape claims 1M elements, data_offsets hold 2
        write_st(root / "shape.safetensors", {{"w", {{1024, 1024}, 2}}});
        onebit::laya::Safetensors st;
        check(st.open((root / "shape.safetensors").string()) && !st.get_f32("w", out),
              "a shape larger than its data_offsets is refused");
    }
    {
        write_st(root / "neg.safetensors", {{"w", {{-1, 2}, 2}}});
        onebit::laya::Safetensors st;
        check(st.open((root / "neg.safetensors").string()) && !st.get_f32("w", out), "a negative dimension is refused");
    }
    {
        // a self-consistent checkpoint whose embedding rows are not 1024 wide
        fs::create_directories(root / "narrow");
        write_st(root / "narrow" / "model.safetensors", {{"encoder.embeddings.tok_embeddings.weight", {{4, 100}, 400}}});
        onebit::laya::Scorer sc;
        const bool loaded = sc.load((root / "narrow").string());
        check(!loaded && sc.error().find("1024") != std::string::npos,
              "an embedding narrower than the scorer's width is refused (" + sc.error() + ")");
    }
    {
        // the right embedding width, but a norm smaller than the scorer reads
        fs::create_directories(root / "small");
        write_st(root / "small" / "model.safetensors", {{"encoder.embeddings.tok_embeddings.weight", {{4, 1024}, 4096}},
                                                        {"encoder.embeddings.norm.weight", {{10}, 10}}});
        onebit::laya::Scorer sc;
        const bool loaded = sc.load((root / "small").string());
        check(!loaded && sc.error().find("has 10 elements") != std::string::npos,
              "a tensor smaller than the scorer's dims is refused (" + sc.error() + ")");
    }
    fs::remove_all(root);
    std::printf("%s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 1 : 0;
}
