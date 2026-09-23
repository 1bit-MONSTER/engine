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

// npu_full_elf_test [<captured-dir> <pdi> <golden-dir>]
//
// Without arguments: MD5 against the RFC 1321 vectors (runs in CI).
// With arguments, against a model's kernel artifacts:
//   captured-dir  layer_ctx<N>.elf instruction ELFs; every one present must be
//                 reproduced byte for byte from layer_ctx1/2/17.elf, and
//                 elf_0002_lmhead.bin
//   golden-dir    full ELFs that ran the fast lane with logits bit-identical to
//                 the xclbin lane: init.elf, fl_ctx<N>.elf, lmhead.elf (and
//                 optionally load_ctx1.elf, a stand-alone kernel with load_pdi)
#include "full_elf.h"

#include <cstdio>
#include <filesystem>
#include <regex>
#include <string>

using namespace onebit::npu;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

static std::string hex(const std::array<uint8_t, 16>& d) {
    std::string s;
    char b[3];
    for (uint8_t x : d) { std::snprintf(b, sizeof b, "%02x", x); s += b; }
    return s;
}

int main(int argc, char** argv) {
    const std::string abc = "abc", fox = "The quick brown fox jumps over the lazy dog";
    check(hex(md5({})) == "d41d8cd98f00b204e9800998ecf8427e", "md5(\"\")");
    check(hex(md5({reinterpret_cast<const uint8_t*>(abc.data()), abc.size()})) == "900150983cd24fb0d6963f7d28e17f72", "md5(abc)");
    check(hex(md5({reinterpret_cast<const uint8_t*>(fox.data()), fox.size()})) == "9e107d9d372bb6826bd81d3542a419d6", "md5(fox)");
    if (argc == 1) {
        std::printf("md5: %s\n", failures ? "FAILED" : "ok (pass <captured-dir> <pdi> <golden-dir> for the model checks)");
        return failures ? 1 : 0;
    }
    if (argc != 4) {
        std::fprintf(stderr, "usage: npu_full_elf_test [<captured-dir> <pdi> <golden-dir>]\n");
        return 2;
    }
    namespace fs = std::filesystem;
    const std::string cap = argv[1], golden = argv[3];
    const Bytes pdi = read_file(argv[2]);
    const auto ctx = [&](int n) { return read_file(cap + "/layer_ctx" + std::to_string(n) + ".elf"); };
    const Bytes ctx1 = ctx(1);
    const ContextMap map = ContextMap::derive(ctx1, ctx(2), ctx(17));
    std::printf("context map: %zu words\n", map.words.size());

    // Every captured context, from context 1 and the map.
    int n_ctx = 0, max_n = 0;
    const std::regex re(R"(layer_ctx(\d+)\.elf)");
    for (const auto& ent : fs::directory_iterator(cap)) {
        std::smatch m;
        const std::string fn = ent.path().filename().string();
        if (!std::regex_match(fn, m, re)) continue;
        const int n = std::stoi(m[1]);
        check(derive_context(ctx1, map, n) == read_file(ent.path().string()), "derived context " + std::to_string(n));
        ++n_ctx;
        max_n = std::max(max_n, n);
    }
    std::printf("captured contexts reproduced: %d/%d (N up to %d)\n", n_ctx - failures, n_ctx, max_n);

    // Full ELFs, against the ones the lane ran.
    int n_full = 0;
    const auto full = [&](const Bytes& got, const std::string& file) {
        const std::string p = golden + "/" + file;
        if (!fs::exists(p)) return;
        check(got == read_file(p), "full ELF " + file);
        ++n_full;
    };
    full(assemble_full_elf(ctx1, pdi, "flinit", PdiMode::kInitOnly), "init.elf");
    full(assemble_full_elf(read_file(cap + "/elf_0002_lmhead.bin"), pdi, "flhead", PdiMode::kNone), "lmhead.elf");
    full(assemble_full_elf(ctx1, pdi, "fl00001", PdiMode::kLoad), "load_ctx1.elf");
    for (int n = 1; fs::exists(golden + "/fl_ctx" + std::to_string(n) + ".elf"); ++n) {
        char name[16];
        std::snprintf(name, sizeof name, "fl%05d", n);
        full(assemble_full_elf(derive_context(ctx1, map, n), pdi, name, PdiMode::kNone),
             "fl_ctx" + std::to_string(n) + ".elf");
    }
    std::printf("full ELFs identical to the lane's: %d checked\n", n_full);
    std::printf("%s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
