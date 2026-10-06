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

// tokenizer_linear_test <tokenizer_linear.json>
//
// Large inputs must tokenize in time linear in their length: a 1 MiB pre-token
// (BPE merges), 1 MiB of short pre-tokens (the regex split), and 1 MiB with an
// added token every few bytes and another that never occurs (the added-token
// search). Each case checks its exact ids and a time bound that the quadratic
// versions missed by orders of magnitude. tests/data/tokenizer_linear.json is a
// byte-level BPE vocabulary of a, b, aa, aaaa, aaaaaaaa, ab plus <think> and
// </think> as added tokens.
#include "tokenizer.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(const onebit::npu::Tokenizer& tok, const char* what, const std::string& text, const std::vector<int>& want) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto got = tok.encode(text);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const bool ok_ids = got == want, ok_time = s < 30.0;  // sanitizer builds included
    std::printf("%s %s: %zu bytes -> %zu ids in %.3f s\n", ok_ids && ok_time ? "ok  " : "FAIL", what, text.size(),
                got.size(), s);
    if (!ok_ids) {
        ++failures;
        size_t i = 0;
        while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
        std::printf("     ids differ at %zu (got %zu ids, want %zu)\n", i, got.size(), want.size());
    }
    if (!ok_time) {
        ++failures;
        std::printf("     took %.1f s; the bound is 30 s\n", s);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: tokenizer_linear_test <tokenizer_linear.json>\n");
        return 2;
    }
    const onebit::npu::Tokenizer tok(argv[1]);
    const size_t MiB = size_t(1) << 20;

    // small cases first: merge order and added tokens
    check(tok, "merge order", "aaaaaaaaaaab", {5, 3, 6});
    check(tok, "added tokens", "a<think>ab</think>b", {0, 7, 6, 8, 1});
    check(tok, "spaces", "a a", {0, 2, 0});

    // one 1 MiB pre-token: 2^17 merges of eight a's each
    check(tok, "long pre-token", std::string(MiB, 'a'), std::vector<int>(MiB / 8, 5));
    // "ab" repeated is one pre-token too
    {
        std::string s;
        while (s.size() < MiB) s += "ab";
        check(tok, "long ab pre-token", s, std::vector<int>(s.size() / 2, 6));
    }
    // 1 MiB of "a " pieces: one regex match per two bytes
    {
        std::string s = "a";
        std::vector<int> want = {0};
        while (s.size() < MiB) {
            s += " a";
            want.push_back(2);
            want.push_back(0);
        }
        check(tok, "many pre-tokens", s, want);
    }
    // an added token every 15 bytes, and </think> never present
    {
        std::string s;
        std::vector<int> want;
        while (s.size() < MiB) {
            s += "<think>aaaaaaaa";
            want.push_back(7);
            want.push_back(5);
        }
        check(tok, "many added tokens", s, want);
    }
    std::printf("%s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 1 : 0;
}
