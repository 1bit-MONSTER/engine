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

// onnx_inline_lifetime_test <scratch dir>
//
// engine#2668: OnnxTensor::raw points into the graph file's mapping, and parse_onnx() used to unmap that mapping before it
// returned, so an INLINE initializer (TensorProto.raw_data, no external data) was read from freed memory by every reader
// that runs after the parse. read_embedding_component()'s f32 token table is one: it parses embedding.onnx, then copies
// `table->raw`. Every measured pack keeps its tables in external data, so no real pack had hit it.
//
// The arm writes an embedding.onnx whose f32 table is inline, large enough (256 KiB) that its pages are the graph file's
// own, reads it through the public reader, and compares every float bit for bit. Built with -fsanitize=address (CMake), so
// before the fix the read of the unmapped pages is reported by ASan (SEGV on the unmapped address) or, if the address was
// reused, the values come back wrong; after it, the values are exact. A second arm, a table in external data, is the
// control: it never touched `raw`, so it passes either way.
#include "pack_import.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// The arm is only an arm under AddressSanitizer: without it, a read of reused pages can return the right floats by luck and
// pass. So a build without the sanitizer fails here, loudly, instead of going green (PR #2670 review).
#if defined(__SANITIZE_ADDRESS__)
#define ONNX_INLINE_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define ONNX_INLINE_ASAN 1
#endif
#endif
#ifndef ONNX_INLINE_ASAN
#define ONNX_INLINE_ASAN 0
#endif

namespace fs = std::filesystem;
namespace pi = onebit::npu::pack_import;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// The protobuf writers this needs: a ModelProto holding one GraphProto holding initializers.
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
void bytes(std::string& o, int field, const std::string& v) {
    tag(o, field, 2);
    varint(o, v.size());
    o += v;
}
// TensorProto: dims (1), data_type (2), name (8), raw_data (9) -- the INLINE form this issue is about
std::string inline_f32(const std::string& name, const std::vector<int64_t>& dims, const std::vector<float>& v) {
    std::string t;
    for (auto d : dims) ints(t, 1, d);
    ints(t, 2, 1);   // FLOAT
    bytes(t, 8, name);
    bytes(t, 9, std::string(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float)));
    return t;
}
// the same table in external data: location / offset / length (13), data_location = EXTERNAL (14)
std::string external_f32(const std::string& name, const std::vector<int64_t>& dims, const std::string& location,
                         uint64_t length) {
    std::string t;
    for (auto d : dims) ints(t, 1, d);
    ints(t, 2, 1);
    bytes(t, 8, name);
    for (const auto& [k, val] : {std::pair<std::string, std::string>{"location", location}, {"offset", "0"},
                                 {"length", std::to_string(length)}}) {
        std::string e;
        bytes(e, 1, k);
        bytes(e, 2, val);
        bytes(t, 13, e);
    }
    ints(t, 14, 1);
    return t;
}
std::string model(const std::string& tensor) {
    std::string g;
    bytes(g, 2, "crafted");
    bytes(g, 5, tensor);   // GraphProto.initializer
    std::string m;
    ints(m, 1, 8);         // ir_version
    bytes(m, 7, g);        // ModelProto.graph
    return m;
}
void put(const fs::path& p, const std::string& s) {
    std::ofstream f(p, std::ios::binary);
    f.write(s.data(), std::streamsize(s.size()));
}
bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (!ONNX_INLINE_ASAN) {
        std::printf("FAIL this test was built without -fsanitize=address, so it cannot see a dangling read (CMake target "
                    "onnx_inline_lifetime_test must carry the sanitizer)\n");
        return 1;
    }
    const fs::path root = argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path();
    const int64_t V = 64, H = 1024;   // 64 x 1024 f32 = 256 KiB inline: 64 pages of the graph file's own mapping
    std::vector<float> tok(size_t(V) * size_t(H));
    for (size_t i = 0; i < tok.size(); ++i) tok[i] = float(int(i % 251) - 125) * 0.125f + float(i) * 1e-6f;

    {   // the arm: an inline f32 token table, read after parse_onnx() has returned
        const fs::path dir = root / "onnx_inline_lifetime";
        fs::remove_all(dir);
        fs::create_directories(dir);
        put(dir / "embedding.onnx", model(inline_f32("model.embed_tokens.weight", {V, H}, tok)));
        std::string why;
        const std::vector<float> got = pi::read_embedding_component(dir.string(), &why);
        std::printf("inline f32 table: %zu values (why: %s)\n", got.size(), why.c_str());
        check(same_bits(got, tok), "an INLINE f32 embed_tokens table read after the parse is bit-exact (" +
                                       std::to_string(got.size()) + " of " + std::to_string(tok.size()) + " floats)");
        fs::remove_all(dir);
    }
    {   // the control: the same table in external data never reads `raw`
        const fs::path dir = root / "onnx_inline_lifetime_ext";
        fs::remove_all(dir);
        fs::create_directories(dir);
        put(dir / "embedding.onnx.data",
            std::string(reinterpret_cast<const char*>(tok.data()), tok.size() * sizeof(float)));
        put(dir / "embedding.onnx",
            model(external_f32("model.embed_tokens.weight", {V, H}, "embedding.onnx.data", tok.size() * sizeof(float))));
        std::string why;
        const std::vector<float> got = pi::read_embedding_component(dir.string(), &why);
        check(same_bits(got, tok), "control: the same table in external data is bit-exact (" + std::to_string(got.size()) +
                                       " floats)");
        fs::remove_all(dir);
    }
    std::printf("%s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
