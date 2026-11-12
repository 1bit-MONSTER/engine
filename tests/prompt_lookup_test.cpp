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

// prompt_lookup_test.cpp -- the prompt-lookup drafter (npu/mtp/prompt_lookup.h, engine#1781). Host only, no model.
//   [1] the lookup itself, by hand: the longest suffix that matches wins, its latest earlier occurrence wins, the
//       continuation runs up to k tokens and may run into the suffix, nothing matches -> no proposal, a position the
//       stream does not end at -> no proposal;
//   [2] through mtp::run() on a toy base: the text is plain decode's whatever the base does; on a base whose answer
//       repeats, drafts are accepted and the run takes fewer passes than plain decode; the head's stream is the
//       committed text.

#include "mtp/prompt_lookup.h"

#include <cstdio>
#include <string>
#include <vector>

using onebit::npu::mtp::BaseModel;
using onebit::npu::mtp::Options;
using onebit::npu::mtp::PromptLookupHead;
using onebit::npu::mtp::Result;

namespace {

int fails = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) fails++;
}

std::vector<int> draft(PromptLookupHead& h, int last, int k) {
    std::vector<int> out;
    h.draft({}, last, int(h.stream().size()) + 1, k, out);
    return out;
}

constexpr int V = 64;

// next = rule(token, pos): `period` > 0 makes the answer a loop of that period, 0 makes every position different.
class Base : public BaseModel {
public:
    explicit Base(int period) : period_(period) {}
    int vocab() const override { return V; }
    int hidden_size() const override { return 1; }
    int max_batch_rows() const override { return 8; }
    int next(int token, int pos) const { return period_ > 0 ? (token + 1) % period_ + 10 : (pos * 7 + 3) % V; }
    bool step(int t, int p, std::vector<float>& l) override {
        if (p != consumed_) return false;
        consumed_++;
        steps_++;
        l.assign(V, 0.0f);
        l[size_t(next(t, p))] = 1.0f;
        return true;
    }
    bool step_batch_all(const std::vector<int>& t, int s, std::vector<float>& all) override {
        if (s != consumed_ || int(t.size()) > 8) return false;
        all.assign(t.size() * size_t(V), 0.0f);
        for (size_t m = 0; m < t.size(); m++) all[m * size_t(V) + size_t(next(t[m], s + int(m)))] = 1.0f;
        pending_ = int(t.size());
        consumed_ += pending_;
        batches_++;
        return true;
    }
    bool commit_verify(int a) override {
        if (a < 0 || a > pending_) return false;
        consumed_ -= pending_ - a;
        pending_ = 0;
        return true;
    }
    const std::vector<float>& last_hidden() const override { return h_; }
    int consumed() const { return consumed_; }
    int passes() const { return steps_ + batches_; }

private:
    int period_, consumed_ = 0, pending_ = 0, steps_ = 0, batches_ = 0;
    std::vector<float> h_{0.0f};
};

}  // namespace

int main() {
    {
        PromptLookupHead h({1, 2, 3, 4, 5, 1, 2});
        check(draft(h, 3, 4) == std::vector<int>({4, 5, 1, 2}), "suffix 1 2 3 found at the start: 4 5 1 2 follow");
        check(draft(h, 3, 9) == std::vector<int>({4, 5, 1, 2, 3}), "k past the stream: the continuation stops at its end");
    }
    {
        PromptLookupHead h({7, 8, 1, 7, 8, 2, 7});
        check(draft(h, 8, 3) == std::vector<int>({2, 7, 8}), "two earlier 7 8: the latest one's continuation (2 7 8)");
    }
    {
        PromptLookupHead h({3, 9, 6, 1, 9, 7, 3});
        check(draft(h, 9, 3) == std::vector<int>({6, 1, 9}),
              "3 9 matches at 0 (6 1 9), 9 alone last at 4 (7 3 9): the longer suffix wins");
    }
    {
        PromptLookupHead h({1, 2, 3});
        check(draft(h, 4, 4).empty(), "a token never seen before: no proposal");
        std::vector<int> out;
        check(h.draft({}, 3, 7, 4, out) == 0 && out.empty(), "a position the stream does not end at: no proposal");
        check(h.stream() == std::vector<int>({1, 2, 3}), "drafting leaves the stream as it was");
        h.commit_rows({4, 5}, {}, 2);
        check(h.stream() == std::vector<int>({1, 2, 3, 4, 5}), "commit_rows appends the committed tokens");
        h.reset();
        check(h.stream() == std::vector<int>({1, 2, 3}), "reset goes back to the prompt");
    }
    const std::vector<int> prompt = {10, 11, 12, 13, 14, 15, 10, 11};
    for (int period : {6, 0}) {
        for (int k : {1, 3, 7}) {
            Options o;
            o.max_tokens = 40;
            o.k = k;
            o.tau = 0.0f;
            Base pb(period), b(period);
            const Result plain = onebit::npu::mtp::run_plain(pb, prompt, o);
            PromptLookupHead h(prompt);
            const Result r = onebit::npu::mtp::run(b, &h, prompt, o);
            std::vector<int> stream = prompt;
            stream.insert(stream.end(), r.tokens.begin(), r.tokens.end());
            const std::string tag = "period " + std::to_string(period) + " k=" + std::to_string(k) + ": ";
            check(r.ok && r.tokens == plain.tokens && b.consumed() == int(prompt.size()) + 40,
                  tag + "text and positions are plain decode's");
            check(h.stream() == stream, tag + "the head's stream is the committed text");
            if (period > 0)
                check(r.stats.accepted > 0 && b.passes() < pb.passes(),
                      tag + "a repeating answer: " + std::to_string(r.stats.accepted) + " drafts accepted, " +
                          std::to_string(b.passes()) + " passes against plain's " + std::to_string(pb.passes()));
            else
                std::printf("INFO %saccepted %d of %d drafts, %d passes against plain's %d\n", tag.c_str(),
                            r.stats.accepted, r.stats.drafted, b.passes(), pb.passes());
        }
    }
    std::printf(fails ? "\n%d CHECK(S) FAILED\n" : "\nall checks passed\n", fails);
    return fails ? 1 : 0;
}
