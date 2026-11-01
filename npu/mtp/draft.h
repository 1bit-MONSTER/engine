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
// Rollback is a truncation, not a snapshot.  A snapshot of a 400-token context is ~100 MB and one per
// iteration would cost more than the drafts save, while every cache here is position-indexed: the caller
// supplies `pos`, so "forget everything past N" is a length bump for attention.  A GDN layer's recurrence is
// the exception (it is not position-indexed), which is why BaseModel::truncate() may refuse -- the loop then
// falls back to plain decode rather than emitting wrong tokens.
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
#include <cstddef>
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

    // Forget every position at or past `tokens`, so the next step can run at `tokens`.  Cheap by design:
    // attention state is position-indexed (the caller passes `pos`), so for a pure-attention model this is
    // a length bump; a GDN/hybrid model must also roll its recurrent state back and returns false until it
    // can (see the note at the top of this file -- that is the one engine gap speculation has).
    virtual bool truncate(int tokens) = 0;

    // The last row's hidden state (H floats) after the last step/step_batch_all: the MTP head's input.
    virtual const std::vector<float>& last_hidden() const = 0;
};

class DraftHead {
public:
    virtual ~DraftHead() = default;
    // The base model has just consumed `last_token` and `hidden` is the state after it.  Propose up to k
    // tokens; return how many were written to out (0 = no proposal, the iteration then costs nothing).
    virtual int draft(const std::vector<float>& hidden, int last_token, int k,
                      std::vector<int>& out) = 0;
    virtual void reset() {}
};

struct Options {
    int k = 4;                 // drafts per iteration; clamped to max_batch_rows()
    int max_tokens = 64;       // tokens to emit, prompt excluded
};

struct Stats {
    int iterations = 0;        // speculative iterations (each = one step + one batch)
    int steps = 0;             // single-token passes
    int batches = 0;           // batched verification passes
    int rows_verified = 0;     // draft rows fed to the base model
    int emitted = 0;
    int drafted = 0;
    int accepted = 0;
    int first_reject = 0;      // iterations where the very first draft was refused
    double acceptance() const { return drafted ? double(accepted) / double(drafted) : 0.0; }
    double mean_accept_len() const { return iterations ? double(accepted) / double(iterations) : 0.0; }
    // Passes per emitted token: the number that has to fall for speculation to be a speed win at all.
    double passes_per_token() const { return emitted ? double(steps + batches) / double(emitted) : 0.0; }
};

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
            continue;
        }

        // Consume t1 so the head sees the state after it; `have` is the position the drafts start at.
        if (!base.step(t1, pos, logits)) { r.err = "step failed"; return r; }
        r.stats.steps++;
        const int have = ++pos;

        std::vector<int> d;
        int nd = head->draft(base.last_hidden(), t1, k, d);
        if (nd <= 0) continue;
        nd = std::min(nd, k);
        d.resize((size_t)nd);

        if (!base.step_batch_all(d, have, all)) { r.err = "verify batch failed"; return r; }
        r.stats.iterations++;
        r.stats.batches++;
        r.stats.rows_verified += nd;
        r.stats.drafted += nd;

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
        for (int i = 0; i < a; i++) {
            if ((int)r.tokens.size() >= opt.max_tokens) break;
            if (!emit(d[(size_t)i])) { r.tokens.pop_back(); break; }
            take++;
        }
        r.stats.accepted += a;              // acceptance is a property of the model, not of the budget
        if (a == 0) r.stats.first_reject++;

        if (take < nd) {
            // Forget the rows we are not using (refused drafts, or the budget cut us short), and take the
            // logits that predict the next position from the batch row we stopped at.
            if (!base.truncate(have + take)) { r.err = "truncate refused"; return r; }
            if (take > 0)
                logits.assign(all.begin() + (size_t)(take - 1) * (size_t)V,
                              all.begin() + (size_t)take * (size_t)V);
            pos = have + take;
        } else {
            // Everything landed: the batch state is already right, and the last row's logits predict the
            // next position (one free token per fully accepted iteration).
            logits.assign(all.begin() + (size_t)(nd - 1) * (size_t)V,
                          all.begin() + (size_t)nd * (size_t)V);
            pos = have + nd;
        }
        if ((int)r.tokens.size() >= opt.max_tokens) break;
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
