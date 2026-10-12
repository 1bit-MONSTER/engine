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

// `1bit serve`'s progress watchdog (docs/serve.md, "When a backend hangs").
//
// A hung GPU dispatch leaves llama-server's HTTP thread answering /health 200 while its main loop
// never returns, so neither serve's own /health nor Lemonade's probe sees it. The watchdog tracks
// every request serve forwards: progress is a byte from the backend on that request, or, for a
// backend whose main loop answers GET /slots (llama-server), that probe answering. A request with
// no progress for `secs` (plus prompt_tokens / prompt_tps before its first byte, so a long prompt
// is not mistaken for a hang) means the backend is wedged: serve kills it and exits non-zero, and
// Lemonade's BackendWatchdog loads it again.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace onebit {

struct WatchdogConfig {
    double secs = 120;        // no progress for this long: the backend is wedged (0: watchdog off)
    double prompt_tps = 50;   // before the first byte, allow prompt_tokens / prompt_tps more
};

class ProgressWatch {
public:
    using clock = std::chrono::steady_clock;

    struct Entry {
        int port = 0;              // the backend's loopback port
        std::string what;          // "POST /v1/chat/completions (streamed)"
        double grace = 0;          // extra seconds before the first byte
        bool needs_probe = false;  // a non-streamed request: only the backend's probe shows progress
        clock::time_point start;
        std::atomic<int64_t> last_ns{0};   // clock time of the last progress
        std::atomic<bool> first{false};    // a byte has arrived
        std::atomic<uint64_t> bytes{0};
    };

    explicit ProgressWatch(WatchdogConfig c = {}) : cfg_(c) {}
    const WatchdogConfig& config() const { return cfg_; }
    bool enabled() const { return cfg_.secs > 0; }

    // A request forwarded to the backend on `port`; it is tracked until the returned handle goes.
    std::shared_ptr<Entry> begin(int port, std::string what, size_t prompt_bytes, bool needs_probe) {
        auto e = std::shared_ptr<Entry>(new Entry, [this](Entry* p) { end(p); delete p; });
        e->port = port;
        e->what = std::move(what);
        e->grace = cfg_.prompt_tps > 0 ? double(prompt_bytes) / 4.0 / cfg_.prompt_tps : 0;  // ~4 bytes a token
        e->needs_probe = needs_probe;
        e->start = clock::now();
        e->last_ns = ns(e->start);
        std::lock_guard<std::mutex> lock(mu_);
        live_.push_back(e.get());
        return e;
    }

    // bytes arrived from the backend on this request
    static void progress(Entry& e, size_t n) {
        e.bytes += n;
        e.first = true;
        e.last_ns = ns(clock::now());
    }

    // the backend on `port` answered its liveness probe: its main loop is turning
    void alive(int port) {
        const int64_t now = ns(clock::now());
        std::lock_guard<std::mutex> lock(mu_);
        for (Entry* e : live_)
            if (e->port == port) e->last_ns = now;
    }

    // the ports with requests in flight
    std::vector<int> busy_ports() const {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<int> out;
        for (const Entry* e : live_)
            if (std::find(out.begin(), out.end(), e->port) == out.end()) out.push_back(e->port);
        return out;
    }

    struct Stall {
        int port = 0;
        std::string what;
        double idle = 0, limit = 0, age = 0;
        uint64_t bytes = 0;
    };

    // The first request past its limit, if any. probe_ok(port) says whether that backend answers a
    // liveness probe; a non-streamed request on a backend without one is not judged (a long answer
    // and a hang look the same from outside).
    template <class ProbeOk>
    std::optional<Stall> stalled(ProbeOk probe_ok, clock::time_point now = clock::now()) const {
        if (!enabled()) return std::nullopt;
        std::lock_guard<std::mutex> lock(mu_);
        for (const Entry* e : live_) {
            if (e->needs_probe && !probe_ok(e->port)) continue;
            const double idle = double(ns(now) - e->last_ns.load()) / 1e9;
            const double limit = cfg_.secs + (e->first ? 0.0 : e->grace);
            if (idle > limit)
                return Stall{e->port, e->what, idle, limit,
                             std::chrono::duration<double>(now - e->start).count(), e->bytes.load()};
        }
        return std::nullopt;
    }

private:
    static int64_t ns(clock::time_point t) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
    }
    void end(Entry* p) {
        std::lock_guard<std::mutex> lock(mu_);
        live_.erase(std::remove(live_.begin(), live_.end(), p), live_.end());
    }

    WatchdogConfig cfg_;
    mutable std::mutex mu_;
    std::vector<Entry*> live_;
};

// The last lines a backend wrote to its stderr (serve passes them through and keeps a few), so a
// watchdog kill can say what the backend said last (an HSA status, a fault).
class LineTail {
public:
    explicit LineTail(size_t keep = 64) : keep_(keep) {}
    // raw bytes from the backend, possibly mid-line; returns the lines completed by them
    std::vector<std::string> feed(const char* data, size_t n) {
        std::vector<std::string> done;
        std::lock_guard<std::mutex> lock(mu_);
        for (size_t i = 0; i < n; i++) {
            if (data[i] == '\n') {
                if (!partial_.empty() && partial_.back() == '\r') partial_.pop_back();
                done.push_back(partial_);
                lines_.push_back(std::move(partial_));
                partial_.clear();
                if (lines_.size() > keep_) lines_.pop_front();
            } else if (partial_.size() < 4096) {
                partial_ += data[i];
            }
        }
        return done;
    }
    std::vector<std::string> last(size_t n) const {
        std::lock_guard<std::mutex> lock(mu_);
        const size_t from = lines_.size() > n ? lines_.size() - n : 0;
        return {lines_.begin() + long(from), lines_.end()};
    }

private:
    size_t keep_;
    mutable std::mutex mu_;
    std::deque<std::string> lines_;
    std::string partial_;
};

}  // namespace onebit
