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

// mtp_draft_test.cpp -- the MTP draft loop against a toy base model.  Host only, no NPU, no XRT, no weights.
//
// The toy model has a closed-form "truth": the token at position p is (p*7+3) % V, so a head that knows the
// position can be exactly right and a head that guesses can be exactly wrong.  Everything the loop claims is
// then checkable without a device:
//   [1] emitted tokens are identical to plain greedy decode -- for a perfect head, a garbage head, and k=0;
//   [2] a perfect head accepts every draft and needs ~1 pass per (k+1) tokens;
//   [3] a garbage head accepts nothing, still emits the same tokens, and is honestly reported as a cost;
//   [4] the base state after the loop is the same as plain decoding would leave (rollback is not "close");
//   [5] k is clamped to max_batch_rows(), never silently widened past what one batch can hold.
#include "mtp/draft.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using onebit::npu::mtp::BaseModel;
using onebit::npu::mtp::DraftHead;
using onebit::npu::mtp::Options;
using onebit::npu::mtp::Result;

namespace {

constexpr int V = 32;          // vocabulary
constexpr int HD = 1;          // hidden: [position] -- enough for a head to be right if it wants to be

int truth(int pos) { return (pos * 7 + 3) % V; }

class ToyBase : public BaseModel {
public:
    explicit ToyBase(int batch_rows = 8) : batch_rows_(batch_rows) {}

    int vocab() const override { return V; }
    int hidden_size() const override { return HD; }
    int max_batch_rows() const override { return batch_rows_; }

    bool step(int token, int pos, std::vector<float>& logits) override {
        if (pos != consumed_) { bad_pos_++; return false; }   // the loop must never skip a position
        consumed_++;
        steps_++;
        logits.assign((size_t)V, 0.0f);
        fill_logits(pos, logits, 0);
        hidden_.assign(HD, (float)consumed_);                 // the head reads the position from here
        return true;
    }

    bool step_batch_all(const std::vector<int>& tokens, int start_pos,
                        std::vector<float>& logits_all) override {
        if (start_pos != consumed_) { bad_pos_++; return false; }
        if ((int)tokens.size() > batch_rows_) { over_batch_++; return false; }
        logits_all.assign(tokens.size() * (size_t)V, 0.0f);
        for (size_t m = 0; m < tokens.size(); m++)
            fill_logits(start_pos + (int)m, logits_all, m * (size_t)V);
        consumed_ += (int)tokens.size();
        pending_ = (int)tokens.size();     // these rows are a verify batch until commit_verify() decides
        batches_++;
        return true;
    }

    // Toy base: no GDN deferral, so committing a verify prefix is the position bookkeeping a pure-attention
    // model needs (drop the rejected rows).  It records the call so the test can check the loop commits
    // exactly the accepted count and never more.
    bool commit_verify(int accepted) override {
        if (accepted < 0 || accepted > pending_) return false;
        consumed_ = consumed_ - pending_ + accepted;
        pending_ = 0;
        commits_++;
        last_commit_ = accepted;
        return true;
    }

    int commits() const { return commits_; }
    int last_commit() const { return last_commit_; }
    int pending() const { return pending_; }

    const std::vector<float>& last_hidden() const override { return hidden_; }

    int consumed() const { return consumed_; }
    int steps() const { return steps_; }
    int batches() const { return batches_; }
    int bad_pos() const { return bad_pos_; }
    int over_batch() const { return over_batch_; }

private:
    // Writes into a caller-sized buffer (a batch is rows*V; resizing here would wipe the earlier rows).
    // The model consumes the token AT `pos` and then predicts the token at pos+1 -- the same convention as
    // the real forward (logits after a step are the next token's logits).
    static void fill_logits(int pos, std::vector<float>& out, size_t off) {
        out[off + (size_t)truth(pos + 1)] = 1.0f;
    }

    int batch_rows_;
    int consumed_ = 0, steps_ = 0, batches_ = 0, commits_ = 0, last_commit_ = -1, pending_ = 0;
    int bad_pos_ = 0, over_batch_ = 0;
    std::vector<float> hidden_{0.0f};
};

// Knows the truth (reads the position out of the hidden state): every draft should be accepted.
class PerfectHead : public DraftHead {
public:
    int draft(const std::vector<float>& hidden, int, int k, std::vector<int>& out,
              std::vector<float>* conf = nullptr) override {
        if (conf) conf->assign((size_t)k, 1.0f);
        const int have = (int)hidden[0];                      // positions consumed so far
        out.clear();
        for (int i = 0; i < k; i++) out.push_back(truth(have + i));
        return (int)out.size();
    }
};

// Proposes the token AFTER the true one at every position: the base must refuse all of it.
class GarbageHead : public DraftHead {
public:
    int draft(const std::vector<float>& hidden, int, int k, std::vector<int>& out,
              std::vector<float>* conf = nullptr) override {
        if (conf) conf->assign((size_t)k, 0.1f);
        const int have = (int)hidden[0];
        out.clear();
        for (int i = 0; i < k; i++) out.push_back((truth(have + i) + 1) % V);
        return (int)out.size();
    }
};

// Right for the first draft, wrong afterwards: exercises the longest-prefix accept AND the rollback.
class HalfHead : public DraftHead {
public:
    int draft(const std::vector<float>& hidden, int, int k, std::vector<int>& out,
              std::vector<float>* conf = nullptr) override {
        if (conf) { conf->assign((size_t)k, 0.1f); if (!conf->empty()) (*conf)[0] = 1.0f; }
        const int have = (int)hidden[0];
        out.clear();
        if (k > 0) out.push_back(truth(have));                 // accepted
        for (int i = 1; i < k; i++) out.push_back((truth(have + i) + 1) % V);   // refused
        return (int)out.size();
    }
};

bool same(const std::vector<int>& a, const std::vector<int>& b) { return a == b; }

int fails = 0;
void check(bool cond, const char* what) {
    std::printf("%s  %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) fails++;
}

std::vector<int> prompt() { return {1, 5, 9, 13}; }

}  // namespace

int main() {
    const auto P = prompt();
    const int WANT = 24;
    Options base_opt;
    base_opt.max_tokens = WANT;

    // [1]/[4] the reference: plain greedy decode, and the state it leaves behind.
    ToyBase ref_model;
    const Result ref = onebit::npu::mtp::run_plain(ref_model, P, base_opt);
    check(ref.ok && (int)ref.tokens.size() == WANT, "plain decode produces the requested tokens");

    for (int k : {0, 1, 2, 4, 8}) {
        for (int which = 0; which < 3; which++) {
            ToyBase m(8);
            PerfectHead ph; GarbageHead gh; HalfHead hh;
            DraftHead* h = which == 0 ? (DraftHead*)&ph : which == 1 ? (DraftHead*)&gh : (DraftHead*)&hh;
            Options o = base_opt;
            o.k = k;
            const Result r = onebit::npu::mtp::run(m, h, P, o);
            char msg[160];
            std::snprintf(msg, sizeof msg, "k=%d head=%s: tokens identical to plain decode",
                          k, which == 0 ? "perfect" : which == 1 ? "garbage" : "half");
            check(r.ok && same(r.tokens, ref.tokens), msg);
            std::snprintf(msg, sizeof msg, "k=%d head=%s: final state == prompt+emitted (%d), no skipped positions",
                          k, which == 0 ? "perfect" : which == 1 ? "garbage" : "half", m.consumed());
            check(m.bad_pos() == 0 && m.consumed() == (int)P.size() + (int)r.tokens.size(), msg);
            std::snprintf(msg, sizeof msg, "k=%d head=%s: draft rows never exceeded max_batch_rows (over=%d)",
                          k, which == 0 ? "perfect" : which == 1 ? "garbage" : "half", m.over_batch());
            check(m.over_batch() == 0, msg);
            std::snprintf(msg, sizeof msg,
                          "k=%d head=%s: acceptance %d/%d, mean accept len %.2f, passes/token %.2f",
                          k, which == 0 ? "perfect" : which == 1 ? "garbage" : "half", r.stats.accepted,
                          r.stats.drafted, r.stats.mean_accept_len(), r.stats.passes_per_token());
            if (which == 0 && k > 0)
                check(r.stats.accepted == r.stats.drafted, msg);         // perfect head: nothing refused
            else if (which == 1 && k > 0)
                check(r.stats.accepted == 0, msg);                       // garbage head: nothing accepted
            else if (which == 2 && k > 1)
                check(r.stats.accepted > 0 && r.stats.accepted < r.stats.drafted, msg);  // prefix only
            else
                std::printf("%s\n", msg);                                // informational
        }
        // [1] k=0 must be a bit-for-bit plain decode
        { ToyBase m(8); PerfectHead ph; Options o = base_opt; o.k = 0;
          const Result r = onebit::npu::mtp::run(m, &ph, P, o);
          check(r.ok && same(r.tokens, ref.tokens) && m.batches() == 0 && m.commits() == 0,
                "k=0: no batched pass, no commit, same tokens"); }
    }

    // [2] the win condition, stated as a number: a perfect head at k=8 amortises the pass
    {
        ToyBase m(8); PerfectHead ph; Options o = base_opt; o.k = 8;
        const Result r = onebit::npu::mtp::run(m, &ph, P, o);
        const double ppt = r.stats.passes_per_token();
        check(ppt < 0.40, "k=8 perfect head: passes per token well under 1 (the whole point)");
        std::printf("INFO k=8 perfect: iterations=%d steps=%d batches=%d emitted=%d accepted=%d passes/token=%.3f\n",
                    r.stats.iterations, r.stats.steps, r.stats.batches, r.stats.emitted, r.stats.accepted, ppt);
    }

    // [3] the cost condition: a bad head must not be free, and must not be silent about it
    {
        ToyBase m(8); GarbageHead gh; Options o = base_opt; o.k = 8;
        const Result r = onebit::npu::mtp::run(m, &gh, P, o);
        check(r.stats.first_reject == r.stats.iterations && r.stats.accepted == 0,
              "garbage head: every iteration is a first-position reject");
        check(r.stats.passes_per_token() > 1.0, "garbage head: more passes per token than plain decode");
        std::printf("INFO k=8 garbage: passes/token=%.3f (plain decode = 1.000), rows verified=%d\n",
                    r.stats.passes_per_token(), r.stats.rows_verified);
    }

    // [6] tau gate: a head whose first draft is below tau gets verified but not followed, so exactly one
    //     row goes through the batch every iteration -- and the text still matches plain decode.
    {
        ToyBase m(8); GarbageHead gh; Options o = base_opt; o.k = 8; o.tau = 0.6f;
        const Result r = onebit::npu::mtp::run(m, &gh, P, o);
        check(r.ok && same(r.tokens, ref.tokens), "tau gate: text still identical to plain decode");
        check(r.stats.rows_verified == r.stats.iterations,
              "tau gate: a below-tau first draft is verified alone (1 row per iteration)");
        std::printf("INFO tau=0.6 garbage head: rows verified=%d over %d iterations, passes/token=%.3f\n",
                    r.stats.rows_verified, r.stats.iterations, r.stats.passes_per_token());
    }
    // [7] the loop commits exactly the accepted prefix, never the whole batch when the tail is refused
    {
        ToyBase m(8); HalfHead hh; Options o = base_opt; o.k = 4;
        const Result r = onebit::npu::mtp::run(m, &hh, P, o);
        check(r.ok && m.pending() == 0, "commit: no verify rows left uncommitted at the end");
        check(m.last_commit() <= r.stats.drafted && m.last_commit() >= 0,
              "commit: the last commit was a prefix of the verified rows");
        std::printf("INFO half head k=4: commits=%d last_commit=%d accepted=%d drafted=%d\n", m.commits(),
                    m.last_commit(), r.stats.accepted, r.stats.drafted);
    }

    std::printf(fails ? "\n%d CHECK(S) FAILED\n" : "\nall checks passed\n", fails);
    return fails ? 1 : 0;
}
