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

#include "bridge.h"

#include <httplib.h>  // before http_guard.h: it installs a pre-routing handler
#include "http_guard.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef ONEBIT_NPU
#include "lane.h"
#include "model.h"
#endif

namespace onebit::bridge {

namespace {

using json = nlohmann::json;

// Unbounded, for a drafter that does not know the model it stands in for (the script seam).
constexpr int kNoBound = 1 << 30;

size_t common_prefix(const std::vector<int>& a, const std::vector<int>& b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

// Answers from a file, one integer per line: the bridge's test seam (docs/bridge.md).
class ScriptDraft : public DraftModel {
public:
    ScriptDraft(std::vector<int> answers, int n_vocab, int max_context)
        : answers_(std::move(answers))
        , n_vocab_(n_vocab > 0 ? n_vocab : kNoBound)
        , max_context_(max_context > 0 ? max_context : kNoBound) {}

    void reset() override { next_ = 0; }
    void step(int /*token*/, int /*ctx*/) override {}
    int argmax() override { return next_ < answers_.size() ? answers_[next_++] : -1; }
    int max_context() const override { return max_context_; }
    int n_vocab() const override { return n_vocab_; }

private:
    std::vector<int> answers_;
    size_t next_ = 0;
    int n_vocab_;
    int max_context_;
};

#ifdef ONEBIT_NPU
// The NPU fast lane as a drafter (docs/npu.md): one whole-layer run per layer per
// token, on Q4NX weights. The lane holds one KV cache and a context is created per
// generate() call, so reset() ends the old one and begins a fresh one.
class LaneDraft : public DraftModel {
public:
    LaneDraft(const std::string& dir, const std::string& kernels)
        : model_(dir)
        , lane_(model_, kernels)
        , n_vocab_(model_.dims().vocab) {}

    void reset() override {
        lane_.end();
        lane_.begin();
    }
    void step(int token, int ctx) override { lane_.step(token, ctx); }
    int argmax() override { return lane_.argmax(); }
    int max_context() const override { return npu::Lane::kMaxContext; }
    int n_vocab() const override { return n_vocab_; }

private:
    npu::Model model_;
    npu::Lane lane_;
    int n_vocab_;
};
#endif

}  // namespace

std::unique_ptr<DraftModel> script_draft(const std::string& path, int n_vocab, int max_context, std::string& err) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot read " + path;
        return nullptr;
    }
    std::vector<int> answers;
    std::string line;
    while (std::getline(f, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        const char* p = line.c_str();
        while (*p == ' ' || *p == '\t' || *p == '\r') ++p;
        if (*p == '\0') continue;
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) {
            err = "not an integer on a line of " + path + ": " + line;
            return nullptr;
        }
        answers.push_back(int(v));
    }
    return std::make_unique<ScriptDraft>(std::move(answers), n_vocab, max_context);
}

#ifdef ONEBIT_NPU
std::unique_ptr<DraftModel> npu_draft(const std::string& dir, std::string& err) {
    const std::string kernels = dir + "/npu";
    if (!std::ifstream(dir + "/model.q4nx").good()) {
        err = dir + " is not an NPU model directory (no model.q4nx)";
        return nullptr;
    }
    if (!std::ifstream(kernels + "/layer_ctx1.elf").good()) {
        err = kernels + "/layer_ctx1.elf is missing: the lane needs the model's npu/ kernels (docs/npu.md)";
        return nullptr;
    }
    return std::make_unique<LaneDraft>(dir, kernels);
}
#endif

Drafter::Drafter(std::unique_ptr<DraftModel> model, int k_cap)
    : model_(std::move(model)), k_cap_(k_cap < 1 ? 1 : k_cap) {
    if (!model_) throw std::invalid_argument("bridge: the drafter needs a model");
}

void Drafter::reset() {
    if (model_) model_->reset();
    have_.clear();
    context_full_ = false;
    ++resets_;
}

std::vector<int> Drafter::draft(const std::vector<int>& prompt, int id_last, int n_max, std::string& note) {
    std::vector<int> out;
    note.clear();
    if (!model_) {
        note = "no draft model";
        return out;
    }
    ++rounds_;

    // The committed sequence is prompt + [id_last] (docs/bridge.md "The protocol").
    std::vector<int> seq = prompt;
    seq.push_back(id_last);

    const int vocab = model_->n_vocab();
    for (int t : seq) {
        if (t < 0 || t >= vocab) {
            note = "a token is outside the drafter's vocabulary";
            return out;
        }
    }

    // Past the drafter's own context there is no answer, and none is needed: an
    // empty draft is a normal single-token decode for the target.
    const int room = model_->max_context() - int(seq.size());
    if (room <= 0) {
        if (!context_full_) {
            // Drop what the drafter holds and remember why: the next rounds until the
            // sequence shrinks are answered without touching the model again.
            reset();
            context_full_ = true;
            note = "the draft context is full";
        }
        return out;
    }
    context_full_ = false;

    int k = n_max > 0 ? std::min(n_max, k_cap_) : k_cap_;
    k = std::min(k, room);

    try {
        // Resynchronise: the model holds `have_` as committed rows, and may hold
        // rejected drafts past it. Replaying from the longest common prefix overwrites
        // those rows, and the causal attention only reads rows below the current
        // position, so no rewind exists or is needed. The common prefix is usually all
        // of `have_` (the target accepted some drafts) and the replay is then just the
        // tokens since the last round.
        const size_t from = common_prefix(have_, seq);
        for (size_t i = from; i < seq.size(); ++i) {
            model_->step(seq[i], int(i) + 1);
            ++replayed_;
        }
        have_ = seq;

        // Greedy drafts, each fed back at its own position. They extend the model
        // past the committed rows; the next round's replay overwrites them.
        for (int i = 0; i < k; ++i) {
            const int t = model_->argmax();
            if (t < 0 || t >= vocab) break;
            out.push_back(t);
            model_->step(t, int(seq.size()) + i + 1);
        }
    } catch (const std::exception& e) {
        note = std::string("the draft model failed: ") + e.what();
        reset();
        return {};
    }

    drafted_ += (long long) out.size();
    return out;
}

struct Server::Impl {
    explicit Impl(Drafter& d) : drafter(d) {}
    Drafter& drafter;
    httplib::Server srv;
    std::thread th;
    std::mutex mu;  // the drafter is one model, one KV: one request at a time
};

namespace {
Drafter& require_drafter(Drafter* d) {
    if (!d) throw std::invalid_argument("bridge: the server needs a drafter");
    return *d;
}
}  // namespace

Server::Server(std::unique_ptr<Drafter> drafter, int port)
    : drafter_(std::move(drafter))
    , p_(std::make_unique<Impl>(require_drafter(drafter_.get())))
    , port_(port) {}

Server::~Server() { stop(); }

std::string Server::addr() const { return "127.0.0.1:" + std::to_string(port_); }

bool Server::start(std::string& err) {
    Impl* impl = p_.get();

    impl->srv.Post("/v1/draft", [impl](const httplib::Request& req, httplib::Response& res) {
        json in;
        try {
            in = json::parse(req.body);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", std::string("bad JSON: ") + e.what()}}.dump(), "application/json");
            return;
        }

        json out;
        try {
            const std::string op = in.value("op", std::string());
            const int seq = in.value("seq", 0);
            std::lock_guard<std::mutex> lock(impl->mu);

            // HRX runs a gated delta-net model on one slot and the lane holds one
            // KV cache, so the bridge serves one sequence (docs/bridge.md).
            if (seq != 0) throw std::runtime_error("the bridge serves one sequence (seq 0)");

            if (op == "draft") {
                std::vector<int> prompt = in.value("prompt", std::vector<int>{});
                const int id_last = in.value("id_last", -1);
                const int n_max = in.value("n_max", 0);
                std::string note;
                out["draft"] = impl->drafter.draft(prompt, id_last, n_max, note);
                if (!note.empty()) out["note"] = note;

                // The first round and then every 256th: enough to see that the drafter is
                // being used (and how it is doing) without a line per token.
                const long long round = impl->drafter.rounds();
                if (round == 1 || round % 256 == 0)
                    std::fprintf(stderr, "1bit bridge: draft round %lld (%zu tokens drafted, %zu committed)%s%s\n",
                                 round, out["draft"].size(), impl->drafter.committed().size(),
                                 note.empty() ? "" : ", ", note.c_str());
            } else if (op == "begin" || op == "reset") {
                impl->drafter.reset();
                out["ok"] = true;
            } else if (op == "accept") {
                // Advisory: every draft resynchronises from `prompt`, so a dropped
                // accept cannot desynchronise the pair (docs/bridge.md).
                out["ok"] = true;
            } else {
                throw std::runtime_error("unknown op: " + op);
            }
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            return;
        }

        res.set_content(out.dump(), "application/json");
    });
    install_request_guard(impl->srv, "127.0.0.1");

    if (port_ > 0) {
        if (!impl->srv.bind_to_port("127.0.0.1", port_)) {
            err = "cannot bind 127.0.0.1:" + std::to_string(port_);
            return false;
        }
    } else {
        port_ = impl->srv.bind_to_any_port("127.0.0.1");
        if (port_ <= 0) {
            err = "cannot bind a loopback port";
            return false;
        }
    }

    // The socket is bound, so a client that connects before the accept loop starts
    // queues rather than races.
    impl->th = std::thread([impl] { impl->srv.listen_after_bind(); });
    return true;
}

void Server::stop() {
    if (!p_) return;
    p_->srv.stop();
    if (p_->th.joinable()) p_->th.join();
}

}  // namespace onebit::bridge
