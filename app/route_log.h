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

// app/route_log.h — what `1bit serve --laya-models` decided, request by request (docs/laya-models.md):
// the recent decisions for GET /v1/route/decisions, and the totals for GET /metrics. Every number is
// one serve measured (the router's time, the request's wall time) or one the backend reported in the
// reply (usage and timings); nothing is estimated.
#pragma once

#include <chrono>
#include <cstdio>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

#include <nlohmann/json.hpp>

namespace onebit {

struct RouteDecision {
    long id = 0;
    double t_unix = 0;          // when the request arrived (seconds since the epoch)
    std::string cls;            // the class used (after the fallback); "pinned" when the request named a row
    std::string laya_class;     // what Laya answered ("" for pinned)
    float confidence = 0;
    // The text Laya classified (the last user message), so a decision log can become `--eval` cases
    // and fine-tune data instead of an untraceable list of verdicts (engine#1313, docs/laya-models.md).
    // Held for classified requests only: a pinned request was never a decision.  Bounded, with the
    // full length kept: a long prompt must not grow this 512-entry log without bound, and a reader
    // has to be able to see that what it got is a prefix.
    std::string state;
    long state_len = -1;        // the state's full length in bytes; -1 when no state was recorded
    std::string model, device;  // the row that answered
    double router_ms = 0;       // the Laya decision (0 when pinned or cached)
    bool cached = false;
    bool stream = false;
    int status = 0;             // 0 for a stream (the status leaves after the handler)
    double request_ms = 0;      // serve's wall time for a non-streamed reply
    long prompt_tokens = -1, completion_tokens = -1;  // the reply's usage, -1 when absent
    double prompt_tok_s = -1, decode_tok_s = -1;      // the reply's timings, -1 when absent

    nlohmann::json to_json() const {
        nlohmann::json j = {{"id", id},           {"t", t_unix},           {"class", cls},
                            {"laya_class", laya_class}, {"confidence", confidence}, {"model", model},
                            {"device", device},   {"router_ms", router_ms}, {"cached", cached},
                            {"stream", stream}};
        if (!stream) {
            j["status"] = status;
            j["request_ms"] = request_ms;
        }
        if (prompt_tokens >= 0) j["prompt_tokens"] = prompt_tokens;
        if (completion_tokens >= 0) j["completion_tokens"] = completion_tokens;
        if (prompt_tok_s >= 0) j["prompt_tok_s"] = prompt_tok_s;
        if (decode_tok_s >= 0) j["decode_tok_s"] = decode_tok_s;
        if (state_len >= 0) {          // >= 0: classified (an empty state is still a decision)
            j["state"] = state;
            j["state_len"] = state_len;   // > (long)state.size() means `state` is a prefix
        }
        return j;
    }
};

// The most of a classified state a decision keeps: prose from an agent step, not a document.  The
// demo's steps are a few hundred bytes; 4 KiB is ~8 of those, and 512 decisions of it is the log's
// own ceiling (2 MiB).  The full length is recorded either way, so a truncated state is visible as
// truncated rather than silently being "the state" (engine#1313).
constexpr size_t kRouteStateMax = 4096;

inline std::string bounded_state(const std::string& s, long& full_len) {
    full_len = static_cast<long>(s.size());
    if (s.size() <= kRouteStateMax) return s;
    return s.substr(0, kRouteStateMax);
}

// Reads usage and timings from a non-streamed reply body (llama-server and the NPU routes both send them).
inline void read_reply_stats(const std::string& body, RouteDecision& d) {
    if (body.empty() || body.front() != '{') return;
    const nlohmann::json out = nlohmann::json::parse(body, nullptr, false);
    if (!out.is_object()) return;
    if (out.contains("usage") && out["usage"].is_object()) {
        const auto& u = out["usage"];
        if (u.contains("prompt_tokens") && u["prompt_tokens"].is_number()) d.prompt_tokens = u["prompt_tokens"].get<long>();
        if (u.contains("completion_tokens") && u["completion_tokens"].is_number())
            d.completion_tokens = u["completion_tokens"].get<long>();
    }
    if (out.contains("timings") && out["timings"].is_object()) {
        const auto& t = out["timings"];
        if (t.contains("prompt_per_second") && t["prompt_per_second"].is_number())
            d.prompt_tok_s = t["prompt_per_second"].get<double>();
        if (t.contains("predicted_per_second") && t["predicted_per_second"].is_number())
            d.decode_tok_s = t["predicted_per_second"].get<double>();
    }
}

class RouteLog {
public:
    explicit RouteLog(size_t keep = 512) : keep_(keep) {}

    long next_id() {
        std::lock_guard<std::mutex> l(mu_);
        return ++last_id_;
    }

    void add(const RouteDecision& d) {
        std::lock_guard<std::mutex> l(mu_);
        recent_.push_back(d);
        while (recent_.size() > keep_) recent_.pop_front();
        Totals& t = totals_[{d.cls, d.model, d.device}];
        ++t.decisions;
        if (!d.cached && d.cls != "pinned") {
            t.router_ms += d.router_ms;
            ++t.routed;
        }
        if (!d.stream) {
            t.request_ms += d.request_ms;
            ++t.answered;
        }
        if (d.completion_tokens >= 0) t.completion_tokens += d.completion_tokens;
        if (d.prompt_tokens >= 0) t.prompt_tokens += d.prompt_tokens;
        if (d.decode_tok_s >= 0) last_decode_[{d.model, d.device}] = d.decode_tok_s;
    }

    // the decisions after `since` (an id), oldest first, at most `limit`
    nlohmann::json recent(long since, size_t limit) const {
        std::lock_guard<std::mutex> l(mu_);
        nlohmann::json arr = nlohmann::json::array();
        size_t skip = 0;
        size_t n = 0;
        for (const auto& d : recent_) n += d.id > since;
        if (n > limit) skip = n - limit;
        for (const auto& d : recent_) {
            if (d.id <= since) continue;
            if (skip) { --skip; continue; }
            arr.push_back(d.to_json());
        }
        return {{"object", "list"}, {"last_id", last_id_}, {"data", arr}};
    }

    // Prometheus text format 0.0.4, appended to serve's own metrics
    std::string render() const {
        std::lock_guard<std::mutex> l(mu_);
        std::string out;
        auto head = [&](const char* name, const char* type, const char* help) {
            out += std::string("# HELP ") + name + " " + help + "\n# TYPE " + name + " " + type + "\n";
        };
        auto labels = [](const Key& k) {
            return "{class=\"" + esc(std::get<0>(k)) + "\",model=\"" + esc(std::get<1>(k)) + "\",device=\"" +
                   esc(std::get<2>(k)) + "\"}";
        };
        head("onebit_route_decisions_total", "counter", "requests --laya-models routed, by class, model and device");
        for (const auto& [k, t] : totals_) out += "onebit_route_decisions_total" + labels(k) + " " + std::to_string(t.decisions) + "\n";
        head("onebit_route_router_ms_sum", "counter", "milliseconds spent in Laya decisions (not cached, not pinned)");
        for (const auto& [k, t] : totals_) out += "onebit_route_router_ms_sum" + labels(k) + " " + fmt(t.router_ms) + "\n";
        head("onebit_route_router_count", "counter", "Laya decisions taken (not cached, not pinned)");
        for (const auto& [k, t] : totals_) out += "onebit_route_router_count" + labels(k) + " " + std::to_string(t.routed) + "\n";
        head("onebit_route_request_ms_sum", "counter", "wall milliseconds of non-streamed routed requests");
        for (const auto& [k, t] : totals_) out += "onebit_route_request_ms_sum" + labels(k) + " " + fmt(t.request_ms) + "\n";
        head("onebit_route_request_count", "counter", "non-streamed routed requests answered");
        for (const auto& [k, t] : totals_) out += "onebit_route_request_count" + labels(k) + " " + std::to_string(t.answered) + "\n";
        head("onebit_route_completion_tokens_total", "counter", "completion tokens the backends reported (usage)");
        for (const auto& [k, t] : totals_)
            out += "onebit_route_completion_tokens_total" + labels(k) + " " + std::to_string(t.completion_tokens) + "\n";
        head("onebit_route_prompt_tokens_total", "counter", "prompt tokens the backends reported (usage)");
        for (const auto& [k, t] : totals_)
            out += "onebit_route_prompt_tokens_total" + labels(k) + " " + std::to_string(t.prompt_tokens) + "\n";
        head("onebit_route_last_decode_tok_s", "gauge", "decode tok/s of the last reply that reported timings, by model and device");
        for (const auto& [k, v] : last_decode_)
            out += "onebit_route_last_decode_tok_s{model=\"" + esc(k.first) + "\",device=\"" + esc(k.second) + "\"} " + fmt(v) + "\n";
        return out;
    }

private:
    using Key = std::tuple<std::string, std::string, std::string>;
    struct Totals {
        long decisions = 0, routed = 0, answered = 0, completion_tokens = 0, prompt_tokens = 0;
        double router_ms = 0, request_ms = 0;
    };
    static std::string esc(const std::string& s) {
        std::string o;
        for (char c : s) {
            if (c == '\\' || c == '"') { o += '\\'; o += c; }
            else if (c == '\n') o += "\\n";
            else o += c;
        }
        return o;
    }
    static std::string fmt(double v) {
        char b[64];
        std::snprintf(b, sizeof b, "%.6g", v);
        return b;
    }

    size_t keep_;
    mutable std::mutex mu_;
    long last_id_ = 0;
    std::deque<RouteDecision> recent_;
    std::map<Key, Totals> totals_;
    std::map<std::pair<std::string, std::string>, double> last_decode_;
};

}  // namespace onebit
