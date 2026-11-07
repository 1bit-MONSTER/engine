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

// The WordPiece rules an embedding model's contract rests on (engine#1311), device-free: the BERT pipeline's
// cleaning, lowercasing, accent stripping, punctuation and CJK splitting, the "##" continuation and the
// whole-word [UNK] rule, plus the [CLS]/[SEP] framing and truncation. The measured proof that these agree with
// the reference tokenizer is the STS-B corpus row (2,552 sentences, 0 id mismatches); this test is what runs
// in CI.
#include "wordpiece.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using onebit::wordpiece::BertTokenizer;

static std::string write_vocab() {
    const char* path = "/tmp/embed_wordpiece_test.vocab.txt";
    std::ofstream f(path);
    const std::vector<std::string> pieces = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "##s",
                                             "the",  "cat",  "sat",  "on",   "mat",   ".",     ",",
                                             "index", "funds", "##ing", "e", "cafe", "caf", "##e"};
    for (const auto& p : pieces) f << p << "\n";
    f.close();
    return path;
}

static std::string ids_to_pieces(const BertTokenizer& t, const std::vector<int>& ids, const std::vector<std::string>& pieces) {
    std::string s;
    for (size_t i = 0; i < ids.size(); ++i) {
        s += (i ? " " : "");
        s += (ids[i] >= 0 && ids[i] < static_cast<int>(pieces.size())) ? pieces[size_t(ids[i])]
                                                                      : std::to_string(ids[i]);
    }
    return s;
}

int main() {
    const std::string vocab = write_vocab();
    const std::vector<std::string> pieces = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "##s", "the",
                                             "cat", "sat", "on", "mat", ".", ",", "index", "funds", "##ing",
                                             "e", "cafe", "caf", "##e"};
    BertTokenizer t;
    assert(t.load_vocab(vocab));

    // clean_text: NUL, control and U+FFFD dropped, every whitespace collapsed to one space, trimmed
    assert(BertTokenizer::clean_text(std::string("  a\0b\x01 c\t\uFFFDd  ", 16)) == "ab c d");
    // ... each whitespace becomes one space and the ends are trimmed; runs are NOT collapsed here (the
    // reference's whitespace split does that), and non-ASCII text survives untouched
    assert(BertTokenizer::clean_text("Caf\u00e9  au lait ") == "Caf\u00e9  au lait");

    // lowercase + strip accents ("Café" -> "cafe"), punctuation split out ("mat." -> "mat" ".")
    assert(ids_to_pieces(t, t.encode("Caf\u00e9 mat.", 16), pieces) == "[CLS] cafe mat . [SEP]");
    // the "##" continuation ("worlds" -> "world" "##s")
    assert(ids_to_pieces(t, t.encode("worlds", 16), pieces) == "[CLS] world ##s [SEP]");
    // Unknown punctuation-bearing words: a word whose first character has no piece is [UNK] for the WHOLE word
    assert(ids_to_pieces(t, t.encode("zzz", 16), pieces) == "[CLS] [UNK] [SEP]");
    assert(ids_to_pieces(t, t.encode("xyzcat", 16), pieces) == "[CLS] [UNK] [SEP]");
    // CJK: one token per character (each unknown here, so one [UNK] each), as the reference splits them
    const auto cjk = t.encode("\u4e2d\u6587\u5b57", 16);
    assert(cjk.size() == 5 && cjk[0] == t.cls_id() && cjk[4] == t.sep_id());
    assert(cjk[1] == t.unk_id() && cjk[2] == t.unk_id() && cjk[3] == t.unk_id());
    // [CLS]/[SEP] framing and the truncation that keeps them: max_tokens counts both specials
    const auto full = t.encode("the cat sat on the mat", 16);
    assert(full.front() == t.cls_id() && full.back() == t.sep_id());
    const auto cut = t.encode("the cat sat on the mat", 4);
    assert(cut.size() == 4 && cut.front() == t.cls_id() && cut.back() == t.sep_id());
    // empty input is a framed empty sequence, not an error
    const auto empty = t.encode("   ", 8);
    assert(empty.size() == 2 && empty[0] == t.cls_id() && empty[1] == t.sep_id());

    std::puts("embed_wordpiece: ok");
    return 0;
}
