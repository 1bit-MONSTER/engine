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

// tokenizer_replace_meta_test <tokenizer_replace_meta.json>
//
// A BPE vocabulary with merges whose normalizer rewrites every space to the metaspace symbol and prepends none, and
// whose decoder turns the symbol back into a space. Gemma4's tokenizer.json is this shape. The engine used to send
// it down the byte-level path: a Gemma4 prompt came out as 31 ids where the reference gives 21, and a reply came
// back with the symbols still in it.
//
// tests/data/tokenizer_replace_meta.json is a nine-token vocabulary of that shape. The expected ids below are what
// the Hugging Face `tokenizers` library returns for the same file (0.23.3), not what this code returns.

#include <cstdio>
#include <string>
#include <vector>

#include "tokenizer.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: tokenizer_replace_meta_test <tokenizer_replace_meta.json>\n");
        return 2;
    }
    const onebit::npu::Tokenizer tok(argv[1]);
    struct Case {
        const char* what;
        std::string text;
        std::vector<int> want;
    };
    const std::vector<Case> cases = {
        {"a space joins the word after it, and nothing is prepended", "a b", {1, 4}},
        {"merges go by rank across the whole run, not word by word", "ab ab", {5, 3, 2}},
        {"a leading space is a real space", " a", {3}},
        {"a newline is its own piece", "a\nb", {1, 6, 2}},
        {"an added token splits the run", "<bos>a b", {8, 1, 4}},
        {"two spaces are two symbols", "a  b", {1, 0, 4}},
        {"one letter", "b", {2}},
    };
    int fail = 0;
    for (const Case& c : cases) {
        const std::vector<int> got = tok.encode(c.text);
        const std::string back = tok.decode(got);
        const bool ok = got == c.want && back == c.text;
        std::printf("%s %s\n", ok ? "ok  " : "FAIL", c.what);
        if (!ok) {
            fail++;
            std::printf("     ids:");
            for (int i : got) std::printf(" %d", i);
            std::printf("   decoded: [%s]\n", back.c_str());
        }
    }
    // One token of a stream: the symbol becomes a space there too.
    if (tok.decode(4) != " b") { std::printf("FAIL a single streamed token keeps its space: [%s]\n", tok.decode(4).c_str()); fail++; }
    else std::printf("ok   a single streamed token keeps its space\n");
    if (fail) { std::printf("tokenizer_replace_meta: %d FAILED\n", fail); return 1; }
    std::printf("tokenizer_replace_meta: ok\n");
    return 0;
}
