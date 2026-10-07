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

// app/bridge.h — the device bridge (docs/bridge.md): a draft model on one device
// (the NPU fast lane) proposes tokens for a target model on another (HRX), over
// the fork's `draft-external` speculative type. The only thing that crosses the
// bridge is token ids, and the target verifies every one of them, so a wrong,
// slow or missing drafter changes the speed and never the reply.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace onebit::bridge {

// One greedy step of a draft model. step() decodes `token` as position ctx-1
// (1-based, the lane's convention); argmax() is the model's next token after the
// last step, or -1 when it has no answer.
class DraftModel {
public:
    virtual ~DraftModel() = default;
    virtual void reset() = 0;
    virtual void step(int token, int ctx) = 0;
    virtual int argmax() = 0;
    virtual int max_context() const = 0;
    virtual int n_vocab() const = 0;
};

// A scripted drafter, one integer per line (`#` starts a comment): its answers in
// order, then -1. This is the bridge's test seam, and the way to exercise the
// whole path on a box with no NPU. n_vocab <= 0 or max_context <= 0 mean "no
// bound" (a test script does not know the model it stands in for).
std::unique_ptr<DraftModel> script_draft(const std::string& path, int n_vocab, int max_context, std::string& err);

#ifdef ONEBIT_NPU
// The NPU fast lane as a drafter (docs/npu.md): dir is a Q4NX model directory
// whose npu/ holds the lane's kernels. Throws on a model the lane cannot load.
std::unique_ptr<DraftModel> npu_draft(const std::string& dir, std::string& err);
#endif

// The bridge's rules, no I/O (docs/bridge.md): resynchronise the drafter to the
// committed sequence by replay, then propose the next k greedy tokens.
class Drafter {
public:
    explicit Drafter(std::unique_ptr<DraftModel> model, int k_cap = 4);

    // `prompt` is the committed sequence and `id_last` the target's newest token,
    // i.e. the draft model is synchronised to `prompt` + [id_last] (their
    // concatenation is what the fork hands over as `prompt` / `id_last`).
    // Returns at most `n_max` drafts (capped by k_cap and the model's context);
    // an empty list is always legal, and `note` says why when it is empty.
    std::vector<int> draft(const std::vector<int>& prompt, int id_last, int n_max, std::string& note);
    void reset();

    int k_cap() const { return k_cap_; }
    long long rounds() const { return rounds_; }
    long long drafted() const { return drafted_; }
    long long replayed() const { return replayed_; }
    long long resets() const { return resets_; }
    // The sequence the model's committed rows hold, for tests and diagnostics.
    const std::vector<int>& committed() const { return have_; }

private:
    std::unique_ptr<DraftModel> model_;
    int k_cap_;
    std::vector<int> have_;  // the sequence the model's committed rows hold
    long long rounds_ = 0, drafted_ = 0, replayed_ = 0, resets_ = 0;
    bool context_full_ = false;
};

// The loopback endpoint the fork's `draft-external` client posts to (docs/bridge.md
// "The protocol"): POST /v1/draft, JSON in and out. Binds 127.0.0.1 only; port 0
// picks a free one. Requests are serialised: the drafter is one model, one KV.
class Server {
public:
    Server(std::unique_ptr<Drafter> drafter, int port = 0);
    ~Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // Bind and serve in a background thread. Returns false and sets `err` on a bind
    // failure; the server is the caller's to hold for as long as the child runs.
    bool start(std::string& err);
    void stop();
    int port() const { return port_; }
    std::string addr() const;  // 127.0.0.1:<port>
    Drafter& drafter() { return *drafter_; }

private:
    struct Impl;
    std::unique_ptr<Drafter> drafter_;
    std::unique_ptr<Impl> p_;
    int port_ = 0;
};

}  // namespace onebit::bridge
