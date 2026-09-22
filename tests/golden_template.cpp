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
// golden_template: render chat-template cases and compare with HF transformers.
//
// usage: golden_template <model.gguf> <cases.jsonl>
// cases.jsonl comes from tools/golden/make_template_golden.py. A case passes
// when the rendered prompt equals HF's byte for byte.
#include <cstdio>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "chat_template.h"
#include "gguf.h"

using namespace onebit;
using json = nlohmann::ordered_json;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <cases.jsonl>\n", argv[0]);
        return 2;
    }
    auto file = GgufFile::open(argv[1]);
    if (!file) {
        std::fprintf(stderr, "%s\n", file.error().c_str());
        return 1;
    }
    auto tmpl = ChatTemplate::from_gguf(*file);
    if (!tmpl) {
        std::fprintf(stderr, "%s\n", tmpl.error().c_str());
        return 1;
    }
    std::ifstream in(argv[2]);
    std::string line;
    size_t n = 0, fail = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        json c = json::parse(line);
        ++n;
        auto got = tmpl->render(c["messages"], c["tools"], c["add_generation_prompt"].get<bool>(), c["kwargs"]);
        const std::string want = c["expected"].get<std::string>();
        if (!got || *got != want) {
            ++fail;
            std::printf("MISMATCH case %zu\n--- want\n%s\n--- got\n%s\n---\n", n, want.c_str(),
                        got ? got->c_str() : got.error().c_str());
        }
    }
    if (n == 0) {
        std::fprintf(stderr, "no cases\n");
        return 2;
    }
    std::printf("%zu cases, mismatches %zu\n%s\n", n, fail, fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
