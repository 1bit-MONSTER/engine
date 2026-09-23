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
// hf_tokenizer_test <tokenizer.json> <golden.json> [<golden.json> ...]
//
// Every case in each golden file (made by tools/make_tokenizer_golden.py with
// the Python `tokenizers` of the same release as third_party/tokenizers) must
// encode to the same ids through the C ABI, and decode back to the same text.
#include "hf_tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: hf_tokenizer_test <tokenizer.json> <golden.json>...\n");
        return 2;
    }
    const onebit::HfTokenizer tok(argv[1]);
    int n = 0, failures = 0;
    for (int a = 2; a < argc; a++) {
        std::ifstream f(argv[a]);
        const auto golden = nlohmann::json::parse(f);
        const bool add_special = golden.value("add_special", false);
        for (const auto& c : golden.at("cases")) {
            const std::string text = c.at("text");
            const auto want = c.at("ids").get<std::vector<uint32_t>>();
            const auto got = tok.encode(text, add_special);
            ++n;
            if (got != want) {
                ++failures;
                std::printf("FAIL encode %s\n  want %s\n  got  %s\n", nlohmann::json(text).dump().c_str(),
                            nlohmann::json(want).dump().c_str(), nlohmann::json(got).dump().c_str());
            } else if (c.contains("decoded") && tok.decode(got) != c.at("decoded").get<std::string>()) {
                ++failures;
                std::printf("FAIL decode %s -> %s\n", nlohmann::json(text).dump().c_str(),
                            nlohmann::json(tok.decode(got)).dump().c_str());
            }
        }
    }
    std::printf("%d/%d cases match (vocab %zu)\n", n - failures, n, tok.vocab_size());
    return failures ? 1 : 0;
}
