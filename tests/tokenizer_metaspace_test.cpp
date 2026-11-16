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

// tokenizer_metaspace_test <tokenizer_metaspace.json>
//
// A BPE vocabulary with merges whose PRE-TOKENIZER is the metaspace step ({Metaspace, prepend_scheme "first",
// split false}) and which has no normalizer at all, with the decoder that turns the symbol back into a space.
// Mistral-7B-Instruct v0.2 and v0.3 (the Ryzen AI 1.9 packs, engine#1503) spell their SentencePiece model this way.
// The engine read only the normalizer, so this shape fell to the byte-level path: "The capital of France is" came out
// as 7 ids where the reference gives 5, and every reply came back with the symbols still in it.
//
// tests/data/tokenizer_metaspace.json is an eleven-token vocabulary of that shape. The expected ids and decoded strings
// below are what the Hugging Face `tokenizers` library returns for the same file (0.23.3), not what this code returns.

#include <cstdio>
#include <string>
#include <vector>

#include "tokenizer.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: tokenizer_metaspace_test <tokenizer_metaspace.json>\n");
        return 2;
    }
    const onebit::npu::Tokenizer tok(argv[1]);
    struct Case {
        const char* what;
        std::string text;
        std::vector<int> want;
        std::string decoded;   // the library's decode: its Strip drops one leading space
    };
    const std::vector<Case> cases = {
        {"the first word gets the prefix, and a space joins the word after it", "a b", {7, 8}, "a b"},
        {"merges go by rank across the whole run", "ab ab", {10, 10}, "ab ab"},
        {"a leading space IS the prefix: no second symbol is added (prepend_scheme first)", " a", {7}, "a"},
        {"a newline is a byte-fallback piece and decodes back to a newline", "a\nb", {7, 3, 6}, "a\nb"},
        {"two spaces are a lone symbol and a joined one", "a  b", {7, 4, 8}, "a  b"},
        {"one letter", "b", {8}, "b"},
        {"a leading space, two spaces and a newline at once", " a  b\na", {7, 4, 8, 3, 5}, "a  b\na"},
    };
    int fail = 0;
    for (const Case& c : cases) {
        const std::vector<int> got = tok.encode(c.text);
        const std::string back = tok.decode(c.want);
        const bool ok = got == c.want && back == c.decoded;
        std::printf("%s %s\n", ok ? "ok  " : "FAIL", c.what);
        if (!ok) {
            std::printf("     got:");
            for (int id : got) std::printf(" %d", id);
            std::printf("   want:");
            for (int id : c.want) std::printf(" %d", id);
            std::printf("   decode: '%s' want '%s'\n", back.c_str(), c.decoded.c_str());
            ++fail;
        }
    }
    if (fail) {
        std::printf("tokenizer_metaspace: %d of %zu cases FAILED\n", fail, cases.size());
        return 1;
    }
    std::printf("tokenizer_metaspace: ok (%zu cases)\n", cases.size());
    return 0;
}
