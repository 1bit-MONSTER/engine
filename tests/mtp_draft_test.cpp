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
#include <random>
#include <cstring>
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
        // The real forward updates last_hidden() from the batch's own last row (`last_hidden_ = fnorm`, i.e.
        // norm_all[M-1], npu/forward/src/q4nx_forward.cpp:4040) -- a batched pass advances the visible hidden too.
        // This toy did not, so it modelled a forward that cannot exist, and any test asking "what is the hidden
        // after a batch" would have read a stale value from the last single step (found 2026-11-05 while pricing a
        // folded draft loop, which has no plain step and therefore no other hidden source).
        hidden_.assign(HD, rows_.back());
        consumed_ += (int)tokens.size();
        pending_ = (int)tokens.size();     // these rows are a verify batch until commit_verify() decides
        batches_++;
        return true;
    }

    // Toy base: no GDN deferral, so committing a verify prefix is the position bookkeeping a pure-attention
    // model needs (drop the rejected rows).  It records the call so the test can check the loop commits
    // exactly the accepted count and never more.
    // The other documented shape: the base keeps its whole batch buffer and commit_verify() only truncates the
    // position bookkeeping.  Legal per BaseModel::last_hidden_rows(), and what a new target most naturally writes.
    void keep_batch_rows(bool on) { keep_batch_rows_ = on; }

    bool commit_verify(int accepted) override {
        if (accepted < 0 || accepted > pending_) return false;
        consumed_ = consumed_ - pending_ + accepted;
        if (!keep_batch_rows_)
            rows_.resize((size_t)accepted);  // the accepted prefix only, as the engine's commit_verify_rows does now
        // ... and hidden_ is deliberately NOT touched here, because the engine does not touch it either:
        // commit_verify_rows() truncates last_hidden_rows_ to the accepted prefix and leaves last_hidden_ as the
        // FULL batch's last row.  So after a partial accept the two disagree by construction, and a reader of
        // last_hidden() is reading a row that was rolled back -- last_hidden_rows() is the one that follows the
        // commit.  Kept faithful (and asserted below) rather than quietly "fixed" here.
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
    bool keep_batch_rows_ = false;
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

// Every draft is the true token, but the head is confident only in the first: tau cuts the chain at nd=2, so BOTH
// drafted tokens are accepted and the chain stops because the GATE stopped it, not because the base refused one.
// That is the shape of a per-depth denominator the loop never drafted (see [9]).
class PerfectGatedHead : public DraftHead {
public:
    int draft(const std::vector<float>& hidden, int, int /*pos*/, int k, std::vector<int>& out,
              std::vector<float>* conf = nullptr) override {
        if (conf) conf->assign((size_t)k, 0.1f);
        const int have = (int)hidden[0];
        out.clear();
        for (int i = 0; i < k; i++) out.push_back(truth(have + i));
        if (conf && !conf->empty()) (*conf)[0] = 1.0f;
        return (int)out.size();
    }
};

// A head with a drawn accuracy: right at each depth with probability `acc`, and it stops chaining after its first
// wrong proposal the way the real head's tau gate does.  Used only by the randomised sweep in [14], where the point
// is that no hand-picked accept shape (all, none, prefix, gated) is special.
class RandomHead : public DraftHead {
public:
    RandomHead(double acc, unsigned seed) : acc_(acc), rng_(seed) {}
    int draft(const std::vector<float>& hidden, int, int /*pos*/, int k, std::vector<int>& out,
              std::vector<float>* conf = nullptr) override {
        const int have = (int)hidden[0];
        std::uniform_real_distribution<double> u(0.0, 1.0);
        out.clear();
        if (conf) conf->clear();
        for (int i = 0; i < k; i++) {
            const double r = u(rng_);
            out.push_back(r < acc_ ? truth(have + i) : (truth(have + i) + 1 + (int)(r * 977.0)) % V);
            if (conf) conf->push_back((float)(r < acc_ ? 0.99 : 0.05));
            if (r >= acc_) break;                      // a wrong draft ends the chain, as a low-confidence one does
        }
        return (int)out.size();
    }

private:
    double acc_;
    std::mt19937 rng_;
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
            // The chain positions are what `--offline --positions-in` scores, so they have to be exactly the
            // loop's own starts: one per iteration, inside the committed stream, strictly advancing.
            check((int)r.stats.chain_at.size() == r.stats.iterations,
                  "positions: one chain start recorded per iteration");
            bool adv = true;
            for (size_t i = 0; i < r.stats.chain_at.size(); i++) {
                const int have = r.stats.chain_at[i];
                if (have < P0 || have > P0 + r.stats.emitted) adv = false;
                if (i && have <= r.stats.chain_at[i - 1]) adv = false;
            }
            check(adv, "positions: chain starts lie inside the stream and advance");
            int acc_sum = 0;
            for (int a : r.stats.chain_accepted) acc_sum += a;
            check((int)r.stats.chain_drafts.size() == r.stats.iterations && acc_sum == r.stats.accepted,
                  "positions: the recorded chains add up to the totals (drafts + accepted)");
            std::printf("INFO stream positions: %zu chain starts, first %d, last %d, stream %d rows\n",
                        r.stats.chain_at.size(), r.stats.chain_at.front(), r.stats.chain_at.back(),
                        sd.stream().keep());
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

    // [12] the toy base follows the forward's hidden contract, because a test double that drifts from it silently
    // certifies the wrong thing.  Two facts, both read from npu/forward/src/q4nx_forward.cpp:
    //   * step_batch_all() sets last_hidden() to the batch's own last row (`last_hidden_ = fnorm`, :4040), so a
    //     batched pass advances the visible hidden -- this toy used to leave it on the previous single step;
    //   * commit_verify_rows() truncates last_hidden_rows() to the accepted prefix and leaves last_hidden() as the
    //     FULL batch's last row (:4100).  After a partial accept the two therefore disagree by construction, and
    //     last_hidden() names a row that was rolled back.  That asymmetry is asserted, not smoothed over: a drafter
    //     that advanced from last_hidden() after a commit would commit a rejected row.
    {
        ToyBase m(8);
        std::vector<float> lg;
        for (int i = 0; i < 3; i++) check(m.step(1, i, lg), "toy contract: prefill step accepted");
        const int before = m.consumed();
        std::vector<float> all;
        check(m.step_batch_all({1, 1, 1}, before, all), "toy contract: batch accepted");
        check(m.last_hidden_rows().size() == 3, "toy contract: the batch exposes one hidden row per row");
        check(m.last_hidden()[0] == m.last_hidden_rows().back(),
              "toy contract: after a batch last_hidden() is the batch's last row, as the real forward's is");
        const float full_last = m.last_hidden()[0];
        check(m.commit_verify(1), "toy contract: a partial commit is accepted");
        check(m.last_hidden_rows().size() == 1, "toy contract: the commit truncates the rows to the accepted prefix");
        check(m.last_hidden()[0] == full_last,
              "toy contract: a partial commit leaves last_hidden() on the FULL batch's last row, as the engine does");
    }

    // [9] the live-vs-offline depth verdict: a depth the loop never drafted is UNSCORED, not 0.00 %.
    // Two defects are kept stated here.  (a) The old rule turned a missing denominator into 0.00 %, so two tables
    // that both measured nothing compared equal to each other and counted as a pass (legacy_within).  (b) The
    // denominator itself was `d ? accepted_at[d-1] : drafted_at[0]`, and under a tau gate that counts a chain that
    // stopped right after ACCEPTING depth d-1 -- so a depth the loop never drafted was scored live 0 % against an
    // offline rate that was measured, and inflated the bar's scored count.  The live side below is a REAL gated toy
    // run, read through the same Stats fields both print sites read (accepted_at / reached_at), and the offline side
    // is the same head on the same stream ungated -- the protocol always drafts every depth, which is why its
    // denominator is positive where the gated loop's is zero.
    {
        auto legacy_within = [](int lh, int lr, int oh, int orr) {
            const double lv = lr ? 100.0 * lh / lr : 0.0;
            const double of = orr ? 100.0 * oh / orr : 0.0;
            return (lv - of) <= 2.0 && (lv - of) >= -2.0;
        };
        check(legacy_within(0, 0, 0, 0), "verdict: the OLD rule calls an unscored depth a pass (the defect)");

        ToyBase gated_base(8);
        PerfectGatedHead gated;
        Options gopt = base_opt;
        gopt.k = 4;
        gopt.tau = 0.6f;
        const Result g = onebit::npu::mtp::run(gated_base, &gated, P, gopt);
        ToyBase open_base(8);                 // the same chain with the gate off: every depth drafted (the protocol)
        PerfectGatedHead open_head;
        Options oopt = gopt;
        oopt.tau = 0.0f;
        const Result o = onebit::npu::mtp::run(open_base, &open_head, P, oopt);
        const auto& gs = g.stats;
        const auto& os = o.stats;
        check(g.ok && o.ok && gs.iterations > 0, "verdict: the gated toy chain ran");
        check(gs.accepted_at[1] > 0 && gs.drafted_at[2] == 0 && gs.reached_at[2] == 0,
              "verdict: the gate stopped after accepting depth 2, so depth 3 was never drafted");
        check(gs.reached_at[0] == gs.drafted_at[0] && gs.reached_at[1] == gs.accepted_at[0],
              "verdict: a reached depth is a drafted depth once the prefix was accepted");

        // Read the live side through the SAME construction both print sites use, not through hand-typed counts:
        // if the report ever derives the denominator differently again, this stops matching the loop.
        onebit::npu::mtp::LiveDepths live;
        onebit::npu::mtp::add_live_depths(gs, live);
        onebit::npu::mtp::LiveDepths open_live;
        onebit::npu::mtp::add_live_depths(os, open_live);
        check(live.hit[2] == gs.accepted_at[2] && live.reached[2] == gs.reached_at[2],
              "verdict: the report's live counts are the loop's own accepted_at/reached_at");

        int scored = 0, within = 0, legacy_scored = 0;
        for (int d = 0; d < 4; d++) {
            const auto v = onebit::npu::mtp::depth_verdict(live.hit[d], live.reached[d], open_live.hit[d],
                                                          open_live.reached[d]);
            if (v.scored) scored++;
            if (v.scored && v.within) within++;
            const int legacy_reached = d ? gs.accepted_at[d - 1] : gs.drafted_at[0];
            if (onebit::npu::mtp::depth_verdict(gs.accepted_at[d], legacy_reached, os.accepted_at[d], os.drafted_at[d])
                    .scored)
                legacy_scored++;
        }
        check(scored == 2, "verdict: only the depths the gated loop drafted are scored");
        check(legacy_scored == 3, "verdict: the OLD denominator scored the undrafted depth (the defect, stated)");
        check(within == scored, "verdict: the gated chain agrees with the ungated one at the depths it reached");

        check(!onebit::npu::mtp::depth_verdict(0, gs.reached_at[2], 5, 9).scored,
              "verdict: live offered no draft at the depth -> n/a, whatever offline measured");
        check(!onebit::npu::mtp::depth_verdict(0, 0, 0, 0).scored, "verdict: 0/0 is n/a, not a pass");
        check(!onebit::npu::mtp::depth_verdict(0, 0, 5, 9).scored, "verdict: live reached nothing -> n/a");
        check(!onebit::npu::mtp::depth_verdict(5, 9, 0, 0).scored, "verdict: offline scored nothing -> n/a");
        const auto gap = onebit::npu::mtp::depth_verdict(40, 100, 60, 100);
        check(gap.scored && !gap.within && gap.delta < -19.9 && gap.delta > -20.1,
              "verdict: a real 20 pp gap still fails, by its own amount");
        // The number that must not move: at tau=0 the new denominator IS the old one, so every published table
        // (tau=0, 8 prompts x 64 tokens) reads exactly as it did before this change.
        int differ = 0;
        for (int d = 0; d < 4; d++)
            if (os.reached_at[d] != (d ? os.accepted_at[d - 1] : os.drafted_at[0])) differ++;
        check(differ == 0, "verdict: at tau=0 the denominator equals accepted_at[d-1] at every depth (no number moves)");
        std::printf("INFO verdict gated chain: drafted %d/%d/%d/%d accepted %d/%d/%d/%d, %d of 4 depths scored (old"
                    " denominator: %d)\n",
                    gs.drafted_at[0], gs.drafted_at[1], gs.drafted_at[2], gs.drafted_at[3], gs.accepted_at[0],
                    gs.accepted_at[1], gs.accepted_at[2], gs.accepted_at[3], scored, legacy_scored);
    }

    // [10] the head's matvec is n-blocked (npu/mtp/mtp_head.cpp): one column at a time walks the [k][n] code image
    // with a 64-byte cache line per 1-byte code, which measured 135-331 ms for the eight matrices of one head step
    // depending on box load (0.7-7 GF/s; tools/mtp_head_bench.cpp).  Blocking n fixes the access pattern and must NOT move a single float: the accumulation order
    // per column (g ascending, k ascending, then the additive-min pass) is identical, so this is a bit-identity
    // check, not a tolerance check.  The unblocked form is kept here as the reference, so a future rewrite has to
    // delete the statement of the defect to pass.
    {
        auto matvec_unblocked = [](const onebit::npu::mtp::Matrix& m, const float* a, float* y) {
            const int G = m.K / onebit::npu::mtp::kG;
            for (int n = 0; n < m.N; n++) {
                double acc_n = 0.0;
                for (int g = 0; g < G; g++) {
                    const int8_t* cg = &m.codes[(size_t)(g * onebit::npu::mtp::kG) * m.N + n];
                    double P = 0.0;
                    for (int k = 0; k < onebit::npu::mtp::kG; k++)
                        P += (double)a[g * onebit::npu::mtp::kG + k] * (double)cg[(size_t)k * m.N];
                    acc_n += P * ((double)m.ms[(size_t)g * m.N + n] * std::ldexp(1.0, (int)m.eref[(size_t)n]));
                }
                for (int g = 0; g < G; g++) {
                    double s = 0.0;
                    for (int k = 0; k < onebit::npu::mtp::kG; k++) s += a[g * onebit::npu::mtp::kG + k];
                    acc_n += s * (double)m.add[(size_t)g * m.N + n];
                }
                y[n] = (float)acc_n;
            }
        };
        // N covers the block tail (200 = 3 x 64 + 8), an exact block (64), one column, and a wide matrix; a
        // wrong tail bound or an off-by-one in the block loop shows up as a disagreeing column.
        const int shapes[][2] = {{64, 200}, {128, 64}, {32, 1}, {256, 127}, {96, 300}};
        int bad_cols = 0, cols = 0;
        for (const auto& sh : shapes) {
            const int K = sh[0], N = sh[1], G = K / onebit::npu::mtp::kG;
            onebit::npu::mtp::Matrix m;
            m.K = K;
            m.N = N;
            m.codes.assign((size_t)K * N, 0);
            m.ms.assign((size_t)G * N, 0);
            m.eref.resize((size_t)N);
            m.add.resize((size_t)G * N);
            uint32_t st = 12345u + (uint32_t)(K * 131 + N);
            for (auto& c : m.codes) { st = st * 1664525u + 1013904223u; c = (int8_t)((int)((st >> 16) % 16) - 8); }
            for (auto& v : m.ms) { st = st * 1664525u + 1013904223u; v = (int16_t)(1000 + (int)((st >> 16) % 30000)); }
            for (auto& e : m.eref) { st = st * 1664525u + 1013904223u; e = (int16_t)((int)((st >> 16) % 9) - 4); }
            for (auto& v : m.add) { st = st * 1664525u + 1013904223u; v = (float)((int)((st >> 16) % 2001) - 1000) * 0.5f; }
            std::vector<float> a((size_t)K), got((size_t)N), want((size_t)N);
            for (auto& x : a) { st = st * 1664525u + 1013904223u; x = (float)((int)((st >> 16) % 2001) - 1000) * 0.001f; }
            matvec_unblocked(m, a.data(), want.data());
            m.matvec(a.data(), got.data());
            for (int n = 0; n < N; n++) {
                cols++;
                if (std::memcmp(&got[(size_t)n], &want[(size_t)n], sizeof(float)) != 0) bad_cols++;
            }
        }
        check(bad_cols == 0, "matvec: the blocked form is BIT-identical to the unblocked form on every column");
        std::printf("INFO matvec: %d columns over 5 shapes, %d differ (K x N incl. a 200 = 3x64+8 block tail)\n", cols,
                    bad_cols);
    }

    // [11] a stop is a stop: the consumer's refusal and the caller's budget both end the run where they happen.
    // Both defects live in the same place -- the accepted-prefix loop of npu/mtp/draft.h run():
    //   (a) `if (!emit(d[i])) { r.tokens.pop_back(); break; }` broke the INNER loop, while run()'s own contract says
    //       a false from on_token stops the loop (the top-of-loop token honours it).  The refused draft is the base's
    //       own argmax at that position, so the next iteration emitted it again: the consumer got the same token a
    //       second time, and a streaming serve could not stop on EOS or on a gone client in one iteration.
    //   (b) the budget was only noticed inside that loop, so an iteration that entered with one token left paid a
    //       full step + verify batch, credited its accepted drafts, and then emitted none of them.
    {
        // (a) a consumer that refuses the 3rd delivery: iteration 1 emits t1 and then its 4 accepted drafts, so the
        // refusal has to land INSIDE the accepted-prefix loop -- that is the arm the top-of-loop check does not cover
        // (a refusal on t1 already stopped the run, which is why the original 6 never exercised the defect).
        const int refuse_at = 3;
        ToyBase m(8); StreamDrafter sd;
        for (int i = 0; i < (int)P.size(); i++) sd.prime(std::vector<float>{(float)(i + 1)}, 0, i);
        Options o = base_opt;
        o.k = 4;                      // a perfect stream drafter: every draft is accepted, so a refusal lands on one
        o.tau = 0.0f;
        o.max_tokens = 64;            // the budget is not what stops this run
        std::vector<int> seen;
        const Result r = onebit::npu::mtp::run(m, &sd, P, o, [&](int t) {
            seen.push_back(t);
            return (int)seen.size() < refuse_at;      // false on the refuse_at-th delivery
        });
        check(r.ok, "stop: the run still reports ok when the consumer stops it");
        check((int)seen.size() == refuse_at, "stop: no delivery after the consumer refused a token (the loop ends here)");
        check((int)r.tokens.size() == refuse_at,
              "stop: the refused token is in the output exactly once, as the top-of-loop token is");
        // The old behaviour, stated as the number it produced instead of described: one extra delivery of the SAME
        // token (it is the base's own choice at the position the loop would have re-entered), so the sequence the
        // consumer saw ended ..., X, X.
        int repeat = 0;
        for (size_t i = 1; i < seen.size(); i++) if (seen[i] == seen[i - 1]) repeat++;
        check(repeat == 0, "stop: the consumer is never handed the same token twice by one stop");
        std::printf("INFO stop: %zu deliveries (refusal at %d), tokens %zu, iterations %d, batches %d\n", seen.size(),
                    refuse_at, r.tokens.size(), r.stats.iterations, r.stats.batches);
    }
    {
        // (b) an ODD budget, so the token that fills it is a bare t1 and nothing can be drafted from a full chain.
        ToyBase m(8); HalfHead hh;
        Options o = base_opt;
        o.k = 4;                      // accepts exactly one draft per iteration: 2 tokens per iteration
        o.max_tokens = 7;             // 2, 4, 6, then the 7th is a bare t1 -> the budget is full before the batch
        const Result r = onebit::npu::mtp::run(m, &hh, P, o);
        check((int)r.tokens.size() == 7, "budget: the loop emits exactly the requested tokens");
        check(m.batches() == 3, "budget: no verify batch is paid for a chain that cannot emit a token");
        check(m.consumed() == (int)P.size() + 7,
              "budget: the base still consumed every emitted token (no position is skipped by the early stop)");
        std::printf("INFO budget: emitted %d, steps %d, batches %d, consumed %d for a prompt of %zu\n",
                    r.stats.emitted, m.steps(), m.batches(), m.consumed(), P.size());
    }

    // [13] both documented hidden-row shapes advance the drafter.  BaseModel::last_hidden_rows() documents M rows
    // (the whole batch); Q4nxNpuForward truncates them to the accepted prefix in commit_verify_rows (#820).  The
    // loop used to demand the second shape exactly, so a base that keeps its batch buffer -- the first shape, and
    // the one the header's own wording describes -- failed the run outright with "base returned a partial batch
    // hidden" on every partial accept, naming a partial buffer it had not received.  Found by randomising the
    // loop's inputs (k, tau, budget, batch width, head accuracy) over a toy base that keeps its rows: 680 of 2000
    // such cases aborted, while the same 2000 with the accepted-prefix shape all passed on text, state and
    // position.  The drafter must also actually ADVANCE here, not just survive: an advance silently skipped is the
    // same 1-token/pass drift in a new disguise.
    {
        ToyBase m(8);
        m.keep_batch_rows(true);
        StreamDrafter sd;
        for (int i = 0; i < (int)P.size(); i++) sd.prime(std::vector<float>{(float)(i + 1)}, 0, i);
        Options o = base_opt;
        o.k = 4;
        o.tau = 0.0f;
        const Result r = onebit::npu::mtp::run(m, &sd, P, o);
        check(r.ok, "whole-batch rows: the run completes instead of refusing the base's row buffer");
        if (!r.ok) std::printf("INFO whole-batch rows: %s\n", r.err.c_str());
        check(r.ok && same(r.tokens, ref.tokens), "whole-batch rows: text identical to plain decode");
        check(r.ok && sd.cache_rows() == (int)P.size() + r.stats.emitted,
              "whole-batch rows: the drafter's stream advanced over every committed row");
        check(r.ok && sd.stream_mismatches() == 0, "whole-batch rows: no chain opened off the cursor");
        std::printf("INFO whole-batch rows: stream %d rows, accepted %d/%d, passes/token %.3f\n", sd.cache_rows(),
                    r.stats.accepted, r.stats.drafted, r.stats.passes_per_token());
    }

    // [14] the same properties over randomised inputs, because [1]-[13] are hand-picked shapes and the defects that
    // survive them are the ones nobody chose.  600 deterministic cases over k, tau, budget, prompt, batch width,
    // head accuracy and BOTH hidden-row shapes ([13]); every one must match plain decode, consume exactly
    // prompt+emitted positions, never exceed max_batch_rows, and leave no verify rows uncommitted.
    {
        std::mt19937 rng(20261105u);
        int cases = 0, bad = 0;
        for (int it = 0; it < 600; it++) {
            const int batch_rows = 1 + (int)(rng() % 6);
            const int plen = 1 + (int)(rng() % 6);
            std::vector<int> pr((size_t)plen);
            for (auto& t : pr) t = (int)(rng() % (unsigned)V);
            Options o;
            o.k = (int)(rng() % 8);
            o.max_tokens = 1 + (int)(rng() % 40);
            o.tau = (it % 3 == 0) ? 0.0f : (it % 3 == 1 ? 0.6f : 0.95f);
            const double acc = (double)(rng() % 101) / 100.0;
            RandomHead h(acc, rng());
            ToyBase m(batch_rows);
            m.keep_batch_rows(it % 2 == 1);
            ToyBase plain_m(batch_rows);
            const Result live = onebit::npu::mtp::run(m, &h, pr, o);
            const Result plain = onebit::npu::mtp::run_plain(plain_m, pr, o);
            cases++;
            const bool ok = live.ok && same(live.tokens, plain.tokens) &&
                            (int)live.tokens.size() == o.max_tokens && m.bad_pos() == 0 &&
                            m.consumed() == plen + (int)live.tokens.size() && m.over_batch() == 0 && m.pending() == 0;
            if (!ok && ++bad <= 3)
                std::printf("INFO sweep case %d: k=%d tau=%.2f budget=%d plen=%d rows=%d full_rows=%d acc=%.2f err=%s\n",
                            it, o.k, (double)o.tau, o.max_tokens, plen, batch_rows, (int)(it % 2 == 1), acc,
                            live.err.c_str());
        }
        check(bad == 0, "randomised: 600 cases keep text, position, batch width and commit invariants");
        std::printf("INFO randomised: %d cases, %d violated an invariant\n", cases, bad);
    }

    std::printf(fails ? "\n%d CHECK(S) FAILED\n" : "\nall checks passed\n", fails);
    return fails ? 1 : 0;
}
