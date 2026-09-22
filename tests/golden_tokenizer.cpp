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
// golden_tokenizer: compare the tokenizer against HF tokenizers.
//
// usage: golden_tokenizer <model.gguf> <cases.tsv>
//
// cases.tsv (tools/golden/make_tokenizer_golden.py): one case per line,
//   hex(text) TAB ids TAB hex(decode(ids)) TAB hex(pre-token pieces joined by NUL)
// A case passes when the pre-token pieces, encode(text) and decode(ids) all
// equal HF's exactly. The pieces check covers the split rules on their own,
// since BPE can hide a wrong split by producing the same ids.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gguf.h"
#include "tokenizer.h"

using namespace onebit;

namespace {

std::string unhex(const std::string& h) {
    std::string out;
    for (size_t i = 0; i + 1 < h.size(); i += 2) out += char(std::stoi(h.substr(i, 2), nullptr, 16));
    return out;
}

std::string show(const std::vector<int32_t>& ids) {
    std::string s;
    for (int32_t id : ids) s += std::to_string(id) + " ";
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <cases.tsv>\n", argv[0]);
        return 2;
    }
    auto file = GgufFile::open(argv[1]);
    if (!file) {
        std::fprintf(stderr, "%s\n", file.error().c_str());
        return 1;
    }
    auto tok = Tokenizer::from_gguf(*file);
    if (!tok) {
        std::fprintf(stderr, "tokenizer: %s\n", tok.error().c_str());
        return 1;
    }

    std::ifstream in(argv[2]);
    std::string line;
    size_t n = 0, split_fail = 0, enc_fail = 0, dec_fail = 0;
    while (std::getline(in, line)) {
        std::istringstream row(line);
        std::string text_hex, ids_str, dec_hex, pieces_hex;
        std::getline(row, text_hex, '\t');
        std::getline(row, ids_str, '\t');
        std::getline(row, dec_hex, '\t');
        std::getline(row, pieces_hex, '\t');
        const std::string text = unhex(text_hex), want_dec = unhex(dec_hex);
        std::vector<int32_t> want;
        std::istringstream ids(ids_str);
        for (int32_t id; ids >> id;) want.push_back(id);
        ++n;

        std::string got_pieces;
        if (auto p = tok->pre_tokenize(text)) {
            for (size_t k = 0; k < p->size(); ++k) got_pieces += (k ? std::string(1, '\0') : "") + (*p)[k];
        }
        if (got_pieces != unhex(pieces_hex)) {
            ++split_fail;
            std::printf("SPLIT MISMATCH case %zu text=%s\n", n, text_hex.c_str());
        }
        auto got = tok->encode(text);
        if (!got || *got != want) {
            ++enc_fail;
            std::printf("ENCODE MISMATCH case %zu text=%s\n  want %s\n  got  %s\n", n, text_hex.c_str(),
                        show(want).c_str(), got ? show(*got).c_str() : got.error().c_str());
        }
        auto dec = tok->decode(want);
        if (!dec || *dec != want_dec) {
            ++dec_fail;
            std::printf("DECODE MISMATCH case %zu ids=%s\n", n, show(want).c_str());
        }
    }
    if (n == 0) {
        std::fprintf(stderr, "no cases in %s\n", argv[2]);
        return 2;
    }
    const bool fail = split_fail || enc_fail || dec_fail;
    std::printf("%zu cases, split mismatches %zu, encode mismatches %zu, decode mismatches %zu\n%s\n", n, split_fail,
                enc_fail, dec_fail, fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
