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

// moe_gguf_index_test
//
// The --moe-slots GGUF index parser (moe/gguf_index.cpp) against small crafted
// GGUF files, written to a temporary directory: a good one with one routed-expert
// tensor is indexed, while a zero alignment, a zero expert count, an array whose
// byte size wraps (it used to seek backwards and loop forever), a string length
// past the end of the file (it used to allocate it first) and too many entries
// are refused with an error.
#include "gguf_index.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
using onebit::moe::GgufIndex;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

struct Gguf {
    std::string b;
    template <class T> Gguf& put(T v) { b.append(reinterpret_cast<const char*>(&v), sizeof v); return *this; }
    Gguf& str(const std::string& s) { put<uint64_t>(s.size()); b += s; return *this; }
};

// Header: magic, version 3, tensor count, KV count.
Gguf header(uint64_t n_tensors, uint64_t n_kv) {
    Gguf g;
    g.put<uint32_t>(0x46554747u).put<uint32_t>(3).put<uint64_t>(n_tensors).put<uint64_t>(n_kv);
    return g;
}

// One 3-D F32 routed-expert tensor [32, 2, n_expert] at offset 0.
Gguf& expert_tensor(Gguf& g, uint64_t n_expert) {
    g.str("blk.0.ffn_up_exps.weight").put<uint32_t>(3).put<uint64_t>(32).put<uint64_t>(2).put<uint64_t>(n_expert);
    return g.put<uint32_t>(0).put<uint64_t>(0);
}

// Writes g (padded with data bytes) and returns the error GgufIndex::open throws, or "".
std::string open_error(const fs::path& path, Gguf g, size_t data = 64) {
    g.b.append(data, '\0');
    std::ofstream(path, std::ios::binary) << g.b;
    try {
        GgufIndex::open(path.string());
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / ("moe_gguf_index_test." + std::to_string(::getpid()));
    fs::remove_all(root);
    fs::create_directories(root);

    {
        Gguf g = header(1, 1);
        g.str("general.alignment").put<uint32_t>(4).put<uint32_t>(32);
        expert_tensor(g, 4);
        g.b.append(32 * 2 * 4 * 4 + 64, '\0');
        std::ofstream(root / "good.gguf", std::ios::binary) << g.b;
        std::string why;
        GgufIndex idx;
        try {
            idx = GgufIndex::open((root / "good.gguf").string());
        } catch (const std::exception& e) {
            why = e.what();
        }
        check(why.empty() && idx.n_expert == 4 && idx.experts.count(0) && idx.expert_bytes(0) == 32 * 2 * 4,
              "a good file is indexed" + (why.empty() ? "" : " (" + why + ")"));
    }
    {
        Gguf g = header(1, 1);
        g.str("general.alignment").put<uint32_t>(4).put<uint32_t>(0);
        expert_tensor(g, 4);
        check(open_error(root / "align.gguf", g).find("alignment") != std::string::npos, "a zero alignment is refused");
    }
    {
        Gguf g = header(1, 0);
        expert_tensor(g, 0);
        check(open_error(root / "experts.gguf", g).find("expert count") != std::string::npos,
              "a zero expert count is refused");
    }
    {
        // an int32 array whose byte size wraps to -64 as a seek offset
        Gguf g = header(0, 2);
        g.str("k").put<uint32_t>(9).put<uint32_t>(5).put<uint64_t>(0x3FFFFFFFFFFFFFF0ull);
        check(!open_error(root / "seek.gguf", g).empty(), "an array that would seek backwards is refused");
    }
    {
        // a 1.5 GiB key in a 65-byte file
        Gguf g = header(0, 1);
        g.put<uint64_t>(1536ull << 20).b += "x";
        check(open_error(root / "str.gguf", g, 32).find("string length") != std::string::npos,
              "a string longer than the file is refused before allocating it");
    }
    {
        Gguf g = header(0, ~uint64_t(0));
        check(open_error(root / "count.gguf", g).find("counts") != std::string::npos, "an impossible KV count is refused");
    }
    fs::remove_all(root);
    std::printf("%s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 1 : 0;
}
