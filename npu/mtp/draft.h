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

// The MTP draft loop: draft k tokens, verify them as ONE batch, keep the longest prefix the base model
// agrees with, roll the base state back, repeat.
//
// Why this shape on this hardware: one weight pass over the 27B container is ~0.44 s of decode, and the
// M=16 twins verify up to 16 rows for the price of one row's weight stream.  So the draft head is worth
// its cost exactly when it turns "one pass per emitted token" into "one pass per (accepted + 1) tokens".
//
// Rollback is a commit, not a snapshot, and not a rollback at all where it can be avoided.
//
// A snapshot of a 400-token context is ~100 MB and one per iteration would cost more than the drafts save, and
// attention state needs no rollback whatsoever: it is position-indexed, the caller supplies `pos`, so
// "forget everything past N" is a length bump.  The one state that cannot be re-indexed is a GDN layer's
// recurrence.  So instead of rolling it back, the verify pass DEFERS it: the rows of a verification batch
// store their recurrence inputs rather than applying them, and BaseModel::commit_verify(accepted) then
// applies exactly the accepted rows and drops the rest.  Nothing is recomputed and nothing is snapshotted;
// the cost is the storage of k rows of update inputs (per layer: the conv-ring rows plus the rank-1 update
// inputs), which is kilobytes, not the 100 MB a state copy would take.
//
// A base that has not implemented deferral reports it from commit_verify() and the loop falls back to a
// synchronous commit + position truncation -- correct, just not free.  Either way the emitted text is the
// same; the difference is only what a rejected draft costs.
//
// The loop is policy only -- it owns no weights and no device state.  The base model arrives through
// BaseModel (Q4nxNpuForward satisfies it through the adapter in npu/mtp/forward_base.h) and the draft head
// through DraftHead (the MTP head; npu/mtp/mtp_head.h), which is what makes it testable on the host with a
// toy model (tests/mtp_draft_test.cpp) before any device is involved.
//
// Invariants the test pins down, because each one is a way to ship wrong text:
//   * the emitted tokens are IDENTICAL to plain greedy decode, whatever the head proposes -- a wrong
//     draft costs time, never output;
//   * the base state after the loop equals the state plain decoding would have produced (rollback);
//   * every verified position is checked against the model's own choice at THAT position (not against
//     the head's next guess), so an accepted draft is accepted only because the base model agreed;
//   * k is clamped to what the batched path can actually hold (max_batch_rows), never silently widened.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace onebit::npu::mtp {

// Greedy choice with a deterministic tie-break (first maximum), so a draft that reproduces the model
// exactly is always accepted and two runs of the same prompt cannot disagree.
inline int argmax(const float* p, int n) {
    if (!p || n <= 0) return -1;
    int best = 0;
    for (int i = 1; i < n; i++)
        if (p[i] > p[best]) best = i;
    return best;
}

// The bookkeeping of a drafter that keeps its own stream (the MTP layer attends over its own states, so its
// cache must be the base model's committed stream, row by row).  There are only two numbers -- how many rows are
// committed, and therefore which position the next row belongs at -- and getting them wrong is silent: the drafts
// get worse and nothing crashes.  So the cursor is a thing the loop can check rather than a loose int: `at(pos)` is
// the assertion a drafter makes before opening a chain, and `keep()` is the row count its cache must be cut back to.
//
// The two mistakes this exists to catch, both seen in the live path (#811): opening a chain at a position the
// cursor does not agree with (a primed stream truncated to 0), and appending committed rows AFTER the open chain's
// provisional rows instead of over them.
struct DraftStream {
    int committed = 0;                                  // committed rows == position of the next committed row
    bool at(int pos) const { return pos == committed; }
    int keep() const { return committed; }               // truncate the cache to this many rows
    void commit(int pos, int keep_rows) { committed = pos + keep_rows; }
    void reset() { committed = 0; }
};

class BaseModel {
public:
    virtual ~BaseModel() = default;
    virtual int vocab() const = 0;
    virtual int hidden_size() const = 0;
    // Rows one batched launch can hold (Q4nxNpuForward::max_batch_rows()); 0 = no batching available,
    // which turns the loop into plain decode.
    virtual int max_batch_rows() const = 0;

    // One token at absolute position pos; fills logits (vocab()).
    virtual bool step(int token, int pos, std::vector<float>& logits) = 0;
    // One batch: token m is consumed at position start_pos + m.  logits_all is rows * vocab(), row-major,
    // row m being the logits AT position start_pos + m -- i.e. the model's own choice for the next
    // position.  Verification needs every row, not just the last one; that is the one thing the plain
    // batched prefill entry point does not return.
    virtual bool step_batch_all(const std::vector<int>& tokens, int start_pos,
                               std::vector<float>& logits_all) = 0;

    // After a verification batch of `rows` rows, keep the first `accepted` of them and discard the rest.
    //   * attention: pure bookkeeping -- drop the positions of the rejected rows (they are position-indexed,
    //     so nothing has to be undone);
    //   * GDN/hybrid: apply the deferred recurrence updates of the accepted rows, drop the rejected ones.
    //     That is the design this interface exists for: the updates were stored during the verify pass, not
    //     applied, so committing a prefix is cheap and no state copy is involved.
    // Returns false if this base cannot defer (no GDN deferral implemented); the loop then stops rather than
    // pretending the state is consistent.
    virtual bool commit_verify(int accepted) = 0;

    // The last row's hidden state (H floats) after the last step/step_batch_all: the MTP head's input.
    virtual const std::vector<float>& last_hidden() const = 0;
    // Every row's hidden of the last batched pass (M x hidden_size, row-major), or empty when the base does not
    // keep it -- then the drafter simply cannot be advanced and its commit_rows() is not called.
    //
    // TWO row counts are legal after commit_verify(accepted), and the loop reads only the first `accepted` rows of
    // either: the whole batch's M rows (what "M x hidden_size" above says, and what a base that keeps its batch
    // buffer most naturally returns), or the accepted prefix alone (what Q4nxNpuForward::commit_verify_rows does to
    // last_hidden_rows_ since #820).  Any other count is a bug in the base and fails the run rather than silently
    // skipping the advance -- that silence is what cost the live loop 1 token/pass for a day (#811).
    virtual const std::vector<float>& last_hidden_rows() const {
        static const std::vector<float> none;
        return none;
    }
};

class DraftHead {
public:
    virtual ~DraftHead() = default;
    // The base model has just consumed `last_token` and `hidden` is the state after it.  Propose up to k
    // tokens; return how many were written to out (0 = no proposal, the iteration then costs nothing).
    // `conf`, when given, receives each proposal's own probability (softmax of the head's logits at that
    // position) so the loop can apply the tau gate; a head that cannot report it leaves `conf` empty and the
    // gate is skipped for that iteration rather than guessed.
    // `pos` is the absolute position of the FIRST drafted token (the chain is pos, pos+1, ...): a drafter
    // with rope cannot guess it and the loop knows it exactly.
    virtual int draft(const std::vector<float>& hidden, int last_token, int pos, int k,
                      std::vector<int>& out, std::vector<float>* conf = nullptr) = 0;
    virtual void reset() {}
    // Whether draft() reads `hidden`, the base's state after `last_token`. true (the MTP head): the loop has to run
    // `last_token` through the base first, so a round is two base passes (that step, then the verify batch).
    // false (a DFlash draft, which drafts from the base's committed rows plus the last token's embedding): the loop
    // verifies [last_token, d1..dk] as ONE batch of k + 1 rows, so a round is one base pass, and draft() is handed an
    // empty `hidden`. engine#1781; docs/npu.md, "The draft loop, and whether to serve it".
    virtual bool needs_last_hidden() const { return true; }
    // Called once per iteration with the rows that were committed: `hidden_rows` is M x hidden_size and holds the
    // base's post-norm hidden of each accepted position (row 0 = the first accepted draft).  A drafter keeps its
    // own stream (the MTP layer attends over its own states), so it advances here -- without this it would see
    // only the current chain and its drafts would be worth very little.
    virtual void commit_rows(const std::vector<int>& /*tokens*/, const std::vector<float>& /*hidden_rows*/,
                             int /*M*/) {}
};

struct Options {
    int k = 4;                 // drafts per iteration; clamped to max_batch_rows()
    int max_tokens = 64;       // tokens to emit, prompt excluded
    // The chain is gated on the drafter's own confidence: keep drafting while the token's probability is at
    // least `tau` (0.6 is the published setting for this head), stop the chain otherwise -- a low-confidence
    // draft is verified anyway (it is already in the batch), it just does not get followed by more of them.
    // tau <= 0 disables the gate.
    float tau = 0.6f;
};

struct Stats {
    static constexpr int kMaxDepth = 16;
    int iterations = 0;        // speculative iterations (each = one step + one batch)
    int steps = 0;             // single-token passes
    int batches = 0;           // batched verification passes
    int rows_verified = 0;     // draft rows fed to the base model
    int emitted = 0;
    int drafted = 0;
    int accepted = 0;
    int first_reject = 0;      // iterations where the very first draft was refused
    // Per draft depth: rows offered at depth d, and how many of them the base model agreed with.  This is the
    // shape the offline protocol reports, and the only form in which "live equals offline" can be checked.
    int drafted_at[kMaxDepth] = {0}, accepted_at[kMaxDepth] = {0};
    // The denominator of the per-depth acceptance RATE: chains that accepted drafts 0..d-1 AND offered a draft at
    // d.  It is not accepted_at[d-1]: under a tau gate a chain can stop exactly after accepting depth d-1, which
    // leaves accepted_at[d-1] positive with nothing drafted at d -- and scoring that depth reads a phantom 0 %
    // against an offline rate that was measured.  At tau=0 every reached chain offers draft d, so this equals
    // accepted_at[d-1] (drafted_at[0] at d=0) and no reported number moves.
    int reached_at[kMaxDepth] = {0};
    // The stream position of every chain this loop opened, and what that chain produced.  The offline protocol
    // chains from EVERY position, so its per-depth table and this one are only the same measurement when it is
    // scored at these positions -- and the drafts themselves are the only way to tell a feeding difference from a
    // verification difference when the two disagree.
    std::vector<int> chain_at;
    std::vector<std::vector<int>> chain_drafts;
    std::vector<int> chain_accepted;
    double hit_at(int d) const {
        return (d >= 0 && d < kMaxDepth && drafted_at[d]) ? double(accepted_at[d]) / double(drafted_at[d]) : 0.0;
    }
    double acceptance() const { return drafted ? double(accepted) / double(drafted) : 0.0; }
    double mean_accept_len() const { return iterations ? double(accepted) / double(iterations) : 0.0; }
    // Passes per emitted token: the number that has to fall for speculation to be a speed win at all.
    double passes_per_token() const { return emitted ? double(steps + batches) / double(emitted) : 0.0; }
    // Where the draft phase's wall clock actually goes, in milliseconds.  "Two passes per iteration" only prices
    // a loop change if the passes carry the cost: the head's own chain and each pass' host work are separate
    // terms, and the arithmetic that decides a change (fold a pass, widen the batch, move the head to the array)
    // needs all four measured in ONE window rather than one of them subtracted across two.
    double ms_step = 0.0;      // single-token passes (the pass this loop pays for the token past the prefix)
    double ms_batch = 0.0;     // verify batches, array + host
    double ms_draft = 0.0;     // the head's chain: its matvecs and the lm head per draft
    double ms_commit = 0.0;    // commit_verify (snapshot restore + replay) and the drafter's row advance
    double phase_ms() const { return ms_step + ms_batch + ms_draft + ms_commit; }
};

// The live loop's side of the per-depth comparison, built in ONE place so the report sites and the tests cannot
// read it differently: hit[d] = chains that accepted depth d, reached[d] = chains that offered a draft at d after
// accepting the prefix.  reached[] is Stats::reached_at and NOT accepted_at[d-1]: under a tau gate the two differ
// exactly where a chain stopped right after accepting a depth, and using the second one scores a depth the loop
// never drafted (see tests/mtp_draft_test.cpp [9]).
struct LiveDepths {
    int hit[Stats::kMaxDepth] = {0};
    int reached[Stats::kMaxDepth] = {0};
};

inline void add_live_depths(const Stats& s, LiveDepths& t) {
    for (int d = 0; d < Stats::kMaxDepth; d++) {
        t.hit[d] += s.accepted_at[d];
        t.reached[d] += s.reached_at[d];
    }
}

struct Result {
    std::vector<int> tokens;
    Stats stats;
    bool ok = false;
    std::string err;
};

// Draft, verify, accept the longest prefix, roll back.  `on_token` returning false stops the loop.
inline Result run(BaseModel& base, DraftHead* head, const std::vector<int>& prompt, const Options& opt,
                  const std::function<bool(int)>& on_token = {}) {
    Result r;
    const int V = base.vocab();
    if (V <= 0) { r.err = "no vocabulary"; return r; }
    if (prompt.empty()) { r.err = "empty prompt"; return r; }
    const int k = head ? std::min(std::max(0, opt.k), base.max_batch_rows()) : 0;

    // Phase timing: always accumulated (steady_clock::now() is ~20 ns against a ~0.5 s pass), printed by the
    // caller when it wants the split.  Measurement only -- nothing here is read back into the loop's decisions.
    using Clock = std::chrono::steady_clock;
    auto lap = [](Clock::time_point& t0) {
        const auto t1 = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
        return ms;
    };
    Clock::time_point tp = Clock::now();

    std::vector<float> logits, all;
    int pos = 0;
    for (int t : prompt)
        if (!base.step(t, pos++, logits)) { r.err = "prefill: step failed"; return r; }

    auto emit = [&](int t) { r.tokens.push_back(t); return !on_token || on_token(t); };

    while ((int)r.tokens.size() < opt.max_tokens) {
        // The base model's own choice for the next position: always emitted, never speculative.
        const int t1 = argmax(logits.data(), V);
        if (t1 < 0) { r.err = "no logits"; return r; }
        const bool keep_going = emit(t1);
        if (!keep_going) break;

        if (k <= 0) {                       // plain decode
            if (!base.step(t1, pos++, logits)) { r.err = "step failed"; return r; }
            r.stats.steps++;
            r.stats.ms_step += lap(tp);
            continue;
        }

        if (!head->needs_last_hidden()) {
            // The one-pass round: t1 is row 0 of the verify batch instead of a step of its own. Row i of the batch is
            // the base's choice after position pos + i, so draft i is checked against row i, and the round commits t1
            // plus the accepted drafts (take + 1 rows); the next round's t1 is the choice in row `take`.
            const int left = opt.max_tokens - (int)r.tokens.size();   // tokens this run may still emit
            const int room = std::min({k, base.max_batch_rows() - 1, left});
            std::vector<int> d;
            std::vector<float> conf;
            int nd = 0;
            if (room > 0) {
                static const std::vector<float> no_hidden;
                r.stats.chain_at.push_back(pos + 1);
                nd = head->draft(no_hidden, t1, pos + 1, room, d, &conf);
                r.stats.ms_draft += lap(tp);
                nd = std::min(nd, std::min(room, (int)d.size()));
            }
            if (nd <= 0) {
                // Nothing to verify (no proposal, or the budget is spent): consume t1 the plain way, so the base has
                // run every token it was handed.
                if (!base.step(t1, pos++, logits)) { r.err = "step failed"; return r; }
                r.stats.steps++;
                r.stats.ms_step += lap(tp);
                if (head) head->commit_rows(std::vector<int>{t1}, base.last_hidden(), 1);
                if ((int)r.tokens.size() >= opt.max_tokens) break;
                continue;
            }
            d.resize((size_t)nd);
            if (opt.tau > 0.0f && conf.size() == d.size()) {
                for (int i = 0; i < nd; i++)
                    if (conf[(size_t)i] < opt.tau) { nd = i + 1; break; }
                d.resize((size_t)nd);
            }
            std::vector<int> batch{t1};
            batch.insert(batch.end(), d.begin(), d.end());
            if (!base.step_batch_all(batch, pos, all)) { r.err = "verify batch failed"; return r; }
            r.stats.ms_batch += lap(tp);
            r.stats.iterations++;
            r.stats.batches++;
            r.stats.rows_verified += nd;
            r.stats.drafted += nd;
            for (int i = 0; i < nd && i < Stats::kMaxDepth; i++) r.stats.drafted_at[i]++;
            int a = 0;
            while (a < nd && argmax(&all[(size_t)a * (size_t)V], V) == d[(size_t)a]) a++;
            // nd <= left, so every accepted draft fits the budget; only the consumer can stop the prefix early.
            int take = 0;
            bool stopped = false;
            for (int i = 0; i < a; i++) {
                const bool keep = emit(d[(size_t)i]);
                take = i + 1;
                if (!keep) { stopped = true; break; }
            }
            r.stats.accepted += a;
            r.stats.chain_drafts.push_back(d);
            r.stats.chain_accepted.push_back(a);
            for (int i = 0; i < a && i < Stats::kMaxDepth; i++) r.stats.accepted_at[i]++;
            for (int i = 0; i <= a && i < nd && i < Stats::kMaxDepth; i++) r.stats.reached_at[i]++;
            if (a == 0) r.stats.first_reject++;
            if (!base.commit_verify(take + 1)) { r.err = "commit_verify refused"; return r; }
            const std::vector<float>& rows = base.last_hidden_rows();
            const size_t h = (size_t)base.hidden_size();
            const size_t got = (h && rows.size() % h == 0) ? rows.size() / h : 0;
            if (!rows.empty()) {
                if (got != (size_t)(take + 1) && got != (size_t)(nd + 1)) {
                    r.err = "drafter rows: base returned " + std::to_string(got) + " rows, want " +
                            std::to_string(take + 1) + " (committed prefix) or " + std::to_string(nd + 1) + " (batch)";
                    return r;
                }
                std::vector<int> kept{t1};
                kept.insert(kept.end(), d.begin(), d.begin() + take);
                head->commit_rows(kept, rows, take + 1);
            }
            r.stats.ms_commit += lap(tp);
            logits.assign(all.begin() + (size_t)take * (size_t)V, all.begin() + (size_t)(take + 1) * (size_t)V);
            pos += take + 1;
            if (stopped || (int)r.tokens.size() >= opt.max_tokens) break;
            continue;
        }

        // Consume t1 so the head sees the state after it; `have` is the position the drafts start at.
        if (!base.step(t1, pos, logits)) { r.err = "step failed"; return r; }
        r.stats.steps++;
        r.stats.ms_step += lap(tp);
        const int have = ++pos;
        // The drafter must advance over t1 too: it keeps its own stream (the MTP layer attends over its own
        // states), so a position the base consumed outside the batch still has to reach it -- otherwise its cache
        // misses one position per iteration and every draft after the first is drafted against a stale stream.
        if (head) head->commit_rows(std::vector<int>{t1}, base.last_hidden(), 1);
        // The budget can already be full here (t1 was the last token asked for): a chain drafted from this
        // position could not emit anything, so paying for it -- head steps plus a verify batch -- is pure cost.
        // The step above is still taken, so the base has consumed every token the caller was handed.
        if ((int)r.tokens.size() >= opt.max_tokens) break;

        std::vector<int> d;
        std::vector<float> conf;
        r.stats.chain_at.push_back(have);
        int nd = head->draft(base.last_hidden(), t1, have, k, d, &conf);
        r.stats.ms_draft += lap(tp);
        if (nd <= 0) continue;
        nd = std::min(nd, k);
        d.resize((size_t)nd);
        // tau gate: the chain only continues while the drafter is confident; the first draft below tau is
        // still verified (it is in the batch anyway) but nothing is drafted after it.
        if (opt.tau > 0.0f && conf.size() == d.size()) {
            for (int i = 0; i < nd; i++)
                if (conf[(size_t)i] < opt.tau) { nd = i + 1; break; }
            d.resize((size_t)nd);
        }

        if (!base.step_batch_all(d, have, all)) { r.err = "verify batch failed"; return r; }
        r.stats.ms_batch += lap(tp);
        r.stats.iterations++;
        r.stats.batches++;
        r.stats.rows_verified += nd;
        r.stats.drafted += nd;
        for (int i = 0; i < nd && i < Stats::kMaxDepth; i++) r.stats.drafted_at[i]++;

        // Draft i is accepted only if the model's own choice at that position is that token: draft 0 is
        // checked against the logits of the last step, draft i>0 against row i-1 of the batch.
        int a = 0;
        for (int i = 0; i < nd; i++) {
            const float* lg = (i == 0) ? logits.data() : &all[(size_t)(i - 1) * (size_t)V];
            if (argmax(lg, V) != d[(size_t)i]) break;
            a++;
        }
        // Emit the accepted prefix, capped by the caller's budget (the accepted drafts are a prefix of
        // plain decode, so a cap only drops the tail -- the text stays identical).
        int take = 0;
        bool stopped = false;
        for (int i = 0; i < a; i++) {
            if ((int)r.tokens.size() >= opt.max_tokens) break;
            // A false from on_token stops the RUN (run()'s contract, and what a streaming serve needs to stop on
            // EOS or on a client that went away): breaking only this prefix loop walked back out to the top of the
            // while, which then handed the consumer the base's own token again -- one delivery past its stop.
            // The refused token stays emitted and committed, exactly as a refusal on t1 at the top of the loop does.
            const bool keep = emit(d[(size_t)i]);
            take = i + 1;
            if (!keep) { stopped = true; break; }
        }
        r.stats.accepted += a;              // acceptance is a property of the model, not of the budget
        r.stats.chain_drafts.push_back(std::vector<int>(d.begin(), d.begin() + nd));
        r.stats.chain_accepted.push_back(a);
        for (int i = 0; i < a && i < Stats::kMaxDepth; i++) r.stats.accepted_at[i]++;
        // Depth d is scored only if the prefix was accepted (a >= d) and a draft at d actually existed (nd > d).
        // a <= nd always, so this counts i < a plus the one rejected draft at i == a when the chain went that far.
        for (int i = 0; i <= a && i < nd && i < Stats::kMaxDepth; i++) r.stats.reached_at[i]++;
        if (a == 0) r.stats.first_reject++;

        // Commit the accepted rows of the verify batch and discard the rest.  With a deferring base this
        // applies only the accepted rows' recurrence updates; without one it truncates by position.
        if (!base.commit_verify(take)) { r.err = "commit_verify refused"; return r; }
        if (head && take > 0) {
            const std::vector<float>& rows = base.last_hidden_rows();
            // Accepted prefix or the whole batch -- both are the base's own rows, and row m is the hidden after the
            // token at start_pos+m either way, so the first `take` of them are the ones the drafter advances over.
            const size_t h = (size_t)base.hidden_size();
            const size_t got = (h && rows.size() % h == 0) ? rows.size() / h : 0;
            if (rows.empty()) {
                // A base that keeps no rows at all is legal: the drafter simply cannot advance.
            } else if (got != (size_t)take && got != (size_t)nd) {
                // A base that keeps the WRONG number of them is a bug, and silently skipping the advance is how the
                // live loop lost 1 token/pass for a day.  Fail instead of drifting.
                r.err = "drafter rows: base returned " + std::to_string(got) + " rows, want " +
                        std::to_string(take) + " (committed prefix) or " + std::to_string(nd) + " (batch)";
                return r;
            } else {
                std::vector<int> accepted(d.begin(), d.begin() + take);
                head->commit_rows(accepted, rows, take);
            }
        }
        r.stats.ms_commit += lap(tp);
        if (take > 0)
            logits.assign(all.begin() + (size_t)(take - 1) * (size_t)V,
                          all.begin() + (size_t)take * (size_t)V);
        pos = have + take;
        if (stopped || (int)r.tokens.size() >= opt.max_tokens) break;
    }

    r.stats.emitted = (int)r.tokens.size();
    r.ok = true;
    return r;
}

// Plain greedy decode through the same interface, so the test (and a --mtp-k 0 run) can compare text.
inline Result run_plain(BaseModel& base, const std::vector<int>& prompt, const Options& opt,
                        const std::function<bool(int)>& on_token = {}) {
    Options o = opt;
    o.k = 0;
    return run(base, nullptr, prompt, o, on_token);
}

}  // namespace onebit::npu::mtp
