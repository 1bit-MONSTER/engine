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

// The NPU decode batcher (engine#2689 item 3b): several in-flight requests share one batched pass.
//
// One decode step is M rows of the M=16 design and one request fills one of them, while the step costs the same from 1
// to 16 rows (measured 11-16: Llama-3.2-1B about 146 ms, Qwen3-1.7B about 214 ms, exact at every N). The forward can
// already run several independent sequences in one pass (Q4nxNpuForward::step_users, item 3a), but a forward instance
// holds ONE sequence, so until now a serve serialised its requests and each paid a full step.
//
// This is the piece between the two: the forward is touched by ONE thread (this object's), and requests hand it work.
//   - acquire() gives a request a slot (its sequence), or says the batch is full;
//   - prefill(slot, tokens) runs that slot's prompt, and means "this slot starts a new sequence" (position 0);
//   - step(slot, token, pos) joins the next decode ROUND and returns that row's logits.
// The loop prefers pending decode rows over a prefill (a prefill is one long pass and would stall every decode), and it
// runs ONE round for every slot with a pending token, so K requests cost one pass instead of K.
//
// Everything that can refuse is refused by name here rather than left to the forward: a slot that was not acquired, a
// second row for a slot that still has one in flight (that would interleave one sequence's positions), a prefill for a
// slot that is mid-decode. The forward's own refusals (a model whose state is more than KV) reach the request as its
// error, unchanged.

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "q4nx_forward.h"

namespace onebit {

class NpuDecodeBatch {
  public:
    // `fwd` is touched only by this object's thread once constructed: the caller must not use it while the batch runs.
    // `max_users` is the slot count (the design's row width, 16).
    // `window_us` is how long a round waits for other requests to join it (0 = run as soon as one row is pending).
    NpuDecodeBatch(Q4nxNpuForward& fwd, int max_users, int window_us = 2000)
        : fwd_(fwd), free_(size_t(max_users > 0 ? max_users : 1), true), window_us_(window_us) {
        thread_ = std::thread([this] { loop(); });
    }
    ~NpuDecodeBatch() { stop(); }
    NpuDecodeBatch(const NpuDecodeBatch&) = delete;
    NpuDecodeBatch& operator=(const NpuDecodeBatch&) = delete;

    // A free slot, or -1 when every slot is taken. A request holds its slot from its prompt to its last token.
    // `wait_ms >= 0` makes it QUEUE instead of failing, which is what a serve wants: the batch is full because other
    // requests are being served, not because this one is wrong. -1 keeps the non-blocking form (and the test).
    int acquire(long wait_ms = -1) {
        std::unique_lock<std::mutex> lk(mu_);
        auto take = [&]() -> int {
            for (size_t i = 0; i < free_.size(); i++)
                if (free_[i]) {
                    free_[i] = false;
                    return int(i);
                }
            return -1;
        };
        if (int s = take(); s >= 0 || wait_ms < 0) return s;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
        while (!stopping_) {
            if (cv_.wait_until(lk, deadline, [&] { return stopping_ || std::any_of(free_.begin(), free_.end(),
                                                                                  [](bool f) { return f; }); }))
                break;
            if (std::chrono::steady_clock::now() >= deadline) break;
        }
        if (stopping_) return -1;
        return take();
    }

    // false when the slot is out of range, was ALREADY free, or still has a row in the running pass. A double release is a
    // caller bug and silently accepting it would let the next acquire() hand a slot to a second request while the first is
    // still using it (review of #2734).
    bool release(int slot) {
        if (slot < 0) return false;
        bool freed = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (size_t(slot) < free_.size() && !free_[size_t(slot)] &&
                !(inflight_.size() > size_t(slot) && inflight_[size_t(slot)] > 0)) {
                free_[size_t(slot)] = true;
                freed = true;
            }
        }
        if (freed) cv_.notify_all();   // a request queued in acquire(wait_ms) is waiting for exactly this
        return freed;
    }

    // Run `tokens` as the slot's prompt (position 0 = a new sequence, which the forward clears). Blocking.
    // `logits`, when given, receives the LAST row -- the answer's first token is drawn from it, exactly as a serve draws
    // it from the last row of its own prefill pass.
    bool prefill(int slot, const std::vector<int>& tokens, std::vector<float>* logits, std::string* why) {
        auto j = std::make_shared<Job>();
        j->slot = slot;
        j->prefill = true;
        j->tokens = tokens;
        j->pos.resize(tokens.size());
        for (size_t i = 0; i < tokens.size(); i++) j->pos[i] = int(i);
        return submit(j, why, logits);
    }

    // Join the next decode round with one token at `pos`; `logits` receives that row. Blocking.
    bool step(int slot, int token, int pos, std::vector<float>* logits, std::string* why) {
        auto j = std::make_shared<Job>();
        j->slot = slot;
        j->tokens = {token};
        j->pos = {pos};
        return submit(j, why, logits);
    }

    // Passes the loop has run: a caller comparing this with its own step count sees what sharing bought.
    long rounds() const {
        std::lock_guard<std::mutex> lk(mu_);
        return rounds_;
    }
    int max_users() const { return int(free_.size()); }

  private:
    struct Job {
        int slot = -1;
        bool prefill = false;
        std::vector<int> tokens, pos;
        std::vector<float> out;
        std::string err;
        bool done = false;
    };

    bool submit(const std::shared_ptr<Job>& j, std::string* why, std::vector<float>* logits) {
        if (j->slot < 0) {
            if (why) *why = "decode batch: no slot was acquired for this request";
            return false;
        }
        std::unique_lock<std::mutex> lk(mu_);
        if (size_t(j->slot) >= free_.size() || free_[size_t(j->slot)]) {
            if (why) *why = "decode batch: slot " + std::to_string(j->slot) + " is not held by this request";
            return false;
        }
        if (inflight_.size() <= size_t(j->slot)) inflight_.resize(free_.size(), 0);
        if (inflight_[size_t(j->slot)] > 0) {
            if (why)
                *why = "decode batch: slot " + std::to_string(j->slot) +
                       " already has a row in the pass that is running (two rows of one sequence would interleave "
                       "positions)";
            return false;
        }
        for (const auto& q : queued_)
            if (q->slot == j->slot) {
                if (why)
                    *why = "decode batch: slot " + std::to_string(j->slot) +
                           " already has a row waiting (two rows of one sequence would interleave positions)";
                return false;
            }
        queued_.push_back(j);
        cv_.notify_all();
        cv_.wait(lk, [&] { return j->done || stopping_; });
        if (!j->done) {
            if (why) *why = "decode batch: stopped";
            return false;
        }
        if (!j->err.empty()) {
            if (why) *why = j->err;
            return false;
        }
        if (logits) *logits = std::move(j->out);
        return true;
    }

    void loop() {
        std::unique_lock<std::mutex> lk(mu_);
        while (!stopping_) {
            if (queued_.empty()) {
                cv_.wait(lk);
                continue;
            }
            // FAIRNESS FIRST: a prefill that is older than every pending decode row runs now, or a stream of decodes (each
            // one blocking its own request) starves a new request's prompt indefinitely -- measured: three requests, and
            // the third's prompt waited until the other two had finished every round they asked for.
            const bool oldest_is_prefill = queued_.front()->prefill;
            if (!oldest_is_prefill) {
                // MICRO-BATCH: wait a short, bounded window for other requests to join this round. Without it each thread
                // re-queues the instant its row is ready and the rounds land one row wide -- batching by accident of
                // timing, which in practice means never.
                if (window_us_ > 0) {
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(window_us_);
                    cv_.wait_until(lk, deadline, [&] {
                        return stopping_ || std::count_if(queued_.begin(), queued_.end(),
                                                          [](const auto& q) { return !q->prefill; }) >= max_users();
                    });
                }
            }
            std::vector<std::shared_ptr<Job>> round_jobs;
            const bool any_decode = std::any_of(queued_.begin(), queued_.end(), [](const auto& q) { return !q->prefill; });
            if (!any_decode || oldest_is_prefill) {
                round_jobs.push_back(queued_.front());   // a prefill runs alone
            } else {
                for (const auto& q : queued_)
                    if (!q->prefill) round_jobs.push_back(q);
                // Ascending by slot: the forward's step_users contract, and it keeps one launch per projection.
                std::sort(round_jobs.begin(), round_jobs.end(),
                          [](const std::shared_ptr<Job>& a, const std::shared_ptr<Job>& b) { return a->slot < b->slot; });
            }
            // A decode job contributes exactly one row; a prefill job contributes ALL of its rows and therefore always
            // runs in a round of its own (the branch above guarantees it).
            std::vector<int> tokens, pos, slots;
            for (const auto& q : round_jobs) {
                if (q->prefill) {
                    for (size_t i = 0; i < q->tokens.size(); i++) {
                        tokens.push_back(q->tokens[i]);
                        pos.push_back(q->pos[i]);
                        slots.push_back(q->slot);
                    }
                } else {
                    tokens.push_back(q->tokens.empty() ? 0 : q->tokens[0]);
                    pos.push_back(q->pos.empty() ? 0 : q->pos[0]);
                    slots.push_back(q->slot);
                }
            }
            std::deque<std::shared_ptr<Job>> rest;
            for (const auto& q : queued_)
                if (std::find(round_jobs.begin(), round_jobs.end(), q) == round_jobs.end()) rest.push_back(q);
            queued_.swap(rest);
            if (inflight_.size() < free_.size()) inflight_.resize(free_.size(), 0);
            for (const auto& q : round_jobs) inflight_[size_t(q->slot)]++;
            if (std::getenv("NPU_BATCH_DBG")) {
                std::string line = "  [batch-dbg] round slots:";
                for (int sl : slots) line += " " + std::to_string(sl);
                line += "  rows=" + std::to_string(tokens.size()) + "  pos:";
                for (int pv : pos) line += " " + std::to_string(pv);
                std::fprintf(stderr, "%s\n", line.c_str());
            }
            lk.unlock();

            std::vector<float> all;
            std::string err;
            const bool ok = fwd_.step_users(tokens, pos, slots, all);
            if (!ok) err = fwd_.last_error();
            if (ok && (round_jobs.size() == 1 && !round_jobs[0]->prefill) && all.empty())
                err = "decode batch: the pass returned no logits";
            size_t width = 0;
            const bool all_decode =
                !round_jobs.empty() && std::all_of(round_jobs.begin(), round_jobs.end(), [](const auto& q) { return !q->prefill; });
            if (err.empty() && all_decode) {
                width = all.size() / round_jobs.size();
                if (width == 0 || all.size() % round_jobs.size() != 0)
                    err = "decode batch: the pass returned " + std::to_string(all.size()) + " logits for " +
                          std::to_string(round_jobs.size()) + " row(s)";
            }

            lk.lock();
            for (size_t i = 0; i < round_jobs.size(); i++) {
                auto& q = round_jobs[i];
                if (inflight_[size_t(q->slot)] > 0) inflight_[size_t(q->slot)]--;
                if (!err.empty()) {
                    q->err = err;
                } else if (q->prefill) {
                    // The request draws its first token from the last row of its own prompt pass.
                    const size_t pw = tokens.empty() ? 0 : all.size() / tokens.size();
                    if (pw > 0 && all.size() >= pw) q->out.assign(all.end() - long(pw), all.end());
                } else {
                    q->out.assign(all.begin() + long(i * width), all.begin() + long((i + 1) * width));
                }
                q->done = true;
            }
            rounds_++;
            cv_.notify_all();
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stopping_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    Q4nxNpuForward& fwd_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::vector<bool> free_;
    std::deque<std::shared_ptr<Job>> queued_;
    // A slot whose row is IN THE PASS THAT IS RUNNING. The queue alone is not enough: a round's jobs are swapped out of
    // `queued_` before the pass, so a second submit arriving mid-pass would not see them and would queue a second row for a
    // sequence whose first row is still being computed (review of #2734).
    std::vector<int> inflight_;
    long rounds_ = 0;
    int window_us_ = 2000;
    bool stopping_ = false;
    std::thread thread_;
};

}  // namespace onebit
