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
#include "mtp/mtp_head.h"

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
        // Per-row hidden, the same quantity the real forward exposes for a drafter that must advance too: row m is
        // the hidden after the token at start_pos+m (this toy's hidden is "positions consumed so far").
        rows_.assign(tokens.size(), 0.0f);
        for (size_t m = 0; m < tokens.size(); m++) rows_[m] = (float)(consumed_ + (int)m + 1);
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
        rows_.resize((size_t)accepted);     // the accepted prefix only, as the engine's commit_verify_rows does now
        pending_ = 0;
        commits_++;
        last_commit_ = accepted;
        return true;
    }

    int commits() const { return commits_; }
    int last_commit() const { return last_commit_; }
    int pending() const { return pending_; }

    const std::vector<float>& last_hidden() const override { return hidden_; }
    const std::vector<float>& last_hidden_rows() const override { return rows_; }

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
    std::vector<float> rows_;
};

// Knows the truth (reads the position out of the hidden state): every draft should be accepted.
class PerfectHead : public DraftHead {
public:
    int draft(const std::vector<float>& hidden, int, int /*pos*/, int k, std::vector<int>& out,
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
    int draft(const std::vector<float>& hidden, int, int /*pos*/, int k, std::vector<int>& out,
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
    int draft(const std::vector<float>& hidden, int, int /*pos*/, int k, std::vector<int>& out,
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

// ---- the drafter's own stream (MtpDrafter's cursor) ------------------------------------------------------------
//
// The MTP head attends over its own states, so a draft is worth something only when the drafter's cache IS the base
// model's committed stream, row by row.  RealMtpDrafter below is the production class with its one device-shaped
// piece (the head row) replaced by a toy: it knows the truth exactly when its cache length equals the position it is
// drafting from, i.e. when it has seen the whole stream and nothing else.  That makes the cursor testable with no
// weights and no device -- and it makes the failure silent in production *and* visible here.
static onebit::npu::mtp::Weights g_no_weights;   // the toy never touches the head's matrices

class StreamDrafter : public onebit::npu::mtp::MtpDrafter {
public:
    StreamDrafter()
        : onebit::npu::mtp::MtpDrafter(
              g_no_weights, [](int, std::vector<float>& e) { e.assign(1, 0.0f); return true; }, nullptr, 1) {
        set_tau(0.0f);          // measure every depth, like the offline protocol
        set_lm_head(false);
    }

    // The old live path, reproduced: rows land in the cache but the cursor is not advanced.  Kept so the check below
    // states the bug instead of assuming it (npu/mtp/mtp_head.cpp prime() is the fixed version).
    void legacy_prime(int pos) {
        cache_.k.resize((size_t)pos + 1, 0.0f);
        cache_.v.resize((size_t)pos + 1, 0.0f);
        cache_.t = pos + 1;
    }
    int cache_rows() const { return cache_.t; }

protected:
    bool next_row(const float* hidden, int, int pos, std::vector<float>& next, std::vector<float>& logits) override {
        const int rows = cache_.t + 1;                 // this row lands at index cache_.t
        cache_.k.resize((size_t)rows, 0.0f);
        cache_.v.resize((size_t)rows, 0.0f);
        cache_.t = rows;
        logits.assign((size_t)V, 0.0f);
        // The toy reads two things a real head needs and a broken loop gets wrong: which position this row predicts
        // (the base's own hidden says it) and whether the cache it attends over is the complete stream up to here.
        const int pred = (int)hidden[0];
        const int t = truth(pred);
        (void)pos;
        logits[(size_t)(rows == pred + 1 ? t : (t + 1) % V)] = 1.0f;   // right only if the stream is complete
        next.assign(1, (float)(pred + 1));
        return true;
    }
};

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

    // [8] the drafter's stream: a chain must start where the base model's committed stream ends, and a committed
    //     row must replace the open chain's provisional rows.  Two runs of the same loop, differing only in whether
    //     the drafter's cursor knows about the primed prompt:
    //       * primed (what the serve route does): every draft is accepted, because the head's cache is the base's
    //         stream -- 0 stream mismatches;
    //       * legacy_prime (the old live path, rows in the cache and the cursor at 0): almost nothing is accepted and
    //         the mismatch counter says why.  That is the 0.62-vs-2.19 collapse, reproduced without weights.
    {
        const int P0 = (int)P.size();
        {   // primed: the cursor is what makes the offline protocol's acceptance reachable in the live loop
            ToyBase m(8); StreamDrafter sd;
            for (int i = 0; i < P0; i++) sd.prime(std::vector<float>{(float)(i + 1)}, 0, i);
            Options o = base_opt; o.k = 4; o.tau = 0.0f;      // every depth measured, like the offline protocol
            const Result r = onebit::npu::mtp::run(m, &sd, P, o);
            check(r.ok && same(r.tokens, ref.tokens), "stream: primed drafter still emits plain-decode text");
            check(sd.stream_mismatches() == 0, "stream: a primed drafter is never opened at the wrong position");
            check(sd.stream().keep() == (int)P.size() + r.stats.emitted,
                  "stream: the drafter's committed rows == prompt + emitted (its cache IS the base's stream)");
            check(r.stats.accepted == r.stats.drafted,
                  "stream: a drafter that has seen the stream accepts every draft (no bookkeeping loss)");
            std::printf("INFO stream primed k=4: rows=%d, mismatches=%d, accepted %d/%d, passes/token %.3f\n",
                        sd.stream().keep(), sd.stream_mismatches(), r.stats.accepted, r.stats.drafted,
                        r.stats.passes_per_token());
        }
        {   // the old path: rows land, the cursor does not
            ToyBase m(8); StreamDrafter sd;
            for (int i = 0; i < P0; i++) sd.legacy_prime(i);
            Options o = base_opt; o.k = 4; o.tau = 0.0f;
            const Result r = onebit::npu::mtp::run(m, &sd, P, o);
            check(r.ok && same(r.tokens, ref.tokens), "stream: legacy path is text-correct too (drafts only cost time)");
            check(sd.stream_mismatches() > 0, "stream: legacy path is reported as out of step, not silently bad");
            check(r.stats.accepted * 4 < r.stats.drafted, "stream: legacy path collapses acceptance (the measured drift)");
            std::printf("INFO stream legacy k=4: rows=%d, mismatches=%d, accepted %d/%d, passes/token %.3f\n",
                        sd.stream().keep(), sd.stream_mismatches(), r.stats.accepted, r.stats.drafted,
                        r.stats.passes_per_token());
        }
        // per-depth accounting exists and adds up to the totals (the live-vs-offline comparison needs it)
        {
            ToyBase m(8); StreamDrafter sd;
            for (int i = 0; i < P0; i++) sd.prime(std::vector<float>{(float)(i + 1)}, 0, i);
            Options o = base_opt; o.k = 4; o.tau = 0.0f;
            const Result r = onebit::npu::mtp::run(m, &sd, P, o);
            int d = 0, a = 0;
            for (int i = 0; i < onebit::npu::mtp::Stats::kMaxDepth; i++) { d += r.stats.drafted_at[i]; a += r.stats.accepted_at[i]; }
            check(d == r.stats.drafted && a == r.stats.accepted, "stream: per-depth counters add up to the totals");
        }
    }

    std::printf(fails ? "\n%d CHECK(S) FAILED\n" : "\nall checks passed\n", fails);
    return fails ? 1 : 0;
}
