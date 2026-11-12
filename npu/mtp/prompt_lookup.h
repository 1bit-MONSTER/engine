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

// prompt_lookup.h -- a drafter with no model: propose what followed the latest earlier occurrence of the stream's
// last tokens (engine#1781).
//
// The stream is the prompt plus every committed token, the last emitted token (t1) included. To draft, take its last n
// tokens (n from max_ngram down to min_ngram, the first length that matches wins), find their latest earlier
// occurrence, and propose the tokens that followed it. A reply that repeats its prompt -- code being edited, a quoted
// passage, a list that follows a pattern -- is then drafted at no cost but the verify batch, and a reply that does not
// gets no proposal at all, so the round falls back to a plain step. It never reads the base's hidden, so mtp::run()
// takes the one-pass round (needs_last_hidden() is false), and mtp::run's fallback bounds what a run of misses costs.
#pragma once

#include <algorithm>
#include <vector>

#include "draft.h"

namespace onebit::npu::mtp {

class PromptLookupHead : public DraftHead {
public:
    explicit PromptLookupHead(std::vector<int> prompt, int max_ngram = 3, int min_ngram = 1)
        : prompt_(std::move(prompt)), stream_(prompt_), max_n_(max_ngram), min_n_(min_ngram < 1 ? 1 : min_ngram) {}

    bool needs_last_hidden() const override { return false; }

    int draft(const std::vector<float>& /*hidden*/, int last_token, int pos, int k, std::vector<int>& out,
              std::vector<float>* conf = nullptr) override {
        out.clear();
        if (conf) conf->clear();   // no probabilities: mtp::run skips its tau gate
        // last_token runs at pos - 1, right after the committed stream; anything else means the stream is not the
        // base's, and a proposal from it would be a guess about some other text.
        if (k <= 0 || pos != int(stream_.size()) + 1) return 0;
        stream_.push_back(last_token);
        const int len = int(stream_.size());
        for (int n = std::min(max_n_, len - 1); n >= min_n_; n--) {
            const int* suf = stream_.data() + (len - n);
            for (int i = len - n - 1; i >= 0; i--) {   // latest earlier occurrence first
                bool hit = true;
                for (int j = 0; j < n && hit; j++) hit = stream_[size_t(i + j)] == suf[j];
                if (!hit) continue;
                for (int j = i + n; j < len && int(out.size()) < k; j++) out.push_back(stream_[size_t(j)]);
                break;
            }
            if (!out.empty()) break;
        }
        stream_.pop_back();
        return int(out.size());
    }

    void commit_rows(const std::vector<int>& tokens, const std::vector<float>& /*rows*/, int M) override {
        for (int i = 0; i < M && i < int(tokens.size()); i++) stream_.push_back(tokens[size_t(i)]);
    }

    void reset() override { stream_ = prompt_; }

    const std::vector<int>& stream() const { return stream_; }

private:
    std::vector<int> prompt_, stream_;
    int max_n_, min_n_;
};

}  // namespace onebit::npu::mtp
