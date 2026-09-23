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
// Hugging Face `tokenizers` for the engine (docs/tokenizers.md): the C ABI of
// hf_tokenizers/ plus a small C++ owner. Loads any model's tokenizer.json.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OnebitTokenizer OnebitTokenizer;
OnebitTokenizer* onebit_tok_from_json(const uint8_t* json, size_t len);
void onebit_tok_free(OnebitTokenizer* tok);
int64_t onebit_tok_encode(const OnebitTokenizer* tok, const uint8_t* text, size_t len, bool add_special,
                          uint32_t* out, size_t cap);
int64_t onebit_tok_decode(const OnebitTokenizer* tok, const uint32_t* ids, size_t n, bool skip_special,
                          uint8_t* out, size_t cap);
int64_t onebit_tok_vocab_size(const OnebitTokenizer* tok);
const char* onebit_tok_last_error(void);

#ifdef __cplusplus
}

#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace onebit {

class HfTokenizer {
public:
    explicit HfTokenizer(const std::string& tokenizer_json_path) {
        std::ifstream f(tokenizer_json_path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + tokenizer_json_path);
        const std::string json{std::istreambuf_iterator<char>(f), {}};
        tok_.reset(onebit_tok_from_json(reinterpret_cast<const uint8_t*>(json.data()), json.size()));
        if (!tok_) throw std::runtime_error(tokenizer_json_path + ": " + onebit_tok_last_error());
    }

    std::vector<uint32_t> encode(const std::string& text, bool add_special = false) const {
        std::vector<uint32_t> ids(text.size() + 8);
        auto n = onebit_tok_encode(tok_.get(), reinterpret_cast<const uint8_t*>(text.data()), text.size(),
                                   add_special, ids.data(), ids.size());
        if (n < 0) throw std::runtime_error(onebit_tok_last_error());
        if (size_t(n) > ids.size()) {
            ids.resize(size_t(n));
            n = onebit_tok_encode(tok_.get(), reinterpret_cast<const uint8_t*>(text.data()), text.size(),
                                  add_special, ids.data(), ids.size());
        }
        ids.resize(size_t(n));
        return ids;
    }

    std::string decode(const std::vector<uint32_t>& ids, bool skip_special = false) const {
        std::string s(ids.size() * 8 + 16, '\0');
        auto n = onebit_tok_decode(tok_.get(), ids.data(), ids.size(), skip_special,
                                   reinterpret_cast<uint8_t*>(s.data()), s.size());
        if (n < 0) throw std::runtime_error(onebit_tok_last_error());
        if (size_t(n) > s.size()) {
            s.resize(size_t(n));
            n = onebit_tok_decode(tok_.get(), ids.data(), ids.size(), skip_special,
                                  reinterpret_cast<uint8_t*>(s.data()), s.size());
        }
        s.resize(size_t(n));
        return s;
    }

    size_t vocab_size() const { return size_t(onebit_tok_vocab_size(tok_.get())); }

private:
    struct Free {
        void operator()(OnebitTokenizer* t) const { onebit_tok_free(t); }
    };
    std::unique_ptr<OnebitTokenizer, Free> tok_;
};

}  // namespace onebit
#endif
