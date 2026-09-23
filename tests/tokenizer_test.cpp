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

// tokenizer_test <tokenizer.json> <golden.json>
//
// Every case in golden.json (tests/data/qwen3_tokenizer_golden.json, made with
// Hugging Face `tokenizers` from the same tokenizer.json) must encode to the
// same ids, and those ids must decode back to the text.
#include "tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: tokenizer_test <tokenizer.json> <golden.json>\n");
        return 2;
    }
    const onebit::npu::Tokenizer tok(argv[1]);
    std::ifstream f(argv[2]);
    const auto golden = nlohmann::json::parse(f);
    int n = 0, failures = 0;
    for (const auto& c : golden.at("cases")) {
        const std::string text = c.at("text");
        const auto want = c.at("ids").get<std::vector<int>>();
        const auto got = tok.encode(text);
        ++n;
        if (got != want) {
            ++failures;
            std::printf("FAIL encode %s\n  want %s\n  got  %s\n", nlohmann::json(text).dump().c_str(),
                        nlohmann::json(want).dump().c_str(), nlohmann::json(got).dump().c_str());
        } else if (tok.decode(got) != text) {
            ++failures;
            std::printf("FAIL decode %s -> %s\n", nlohmann::json(text).dump().c_str(),
                        nlohmann::json(tok.decode(got)).dump().c_str());
        }
    }
    std::printf("%d/%d cases match %s\n", n - failures, n, golden.value("source", "the golden file").c_str());
    return failures ? 1 : 0;
}
