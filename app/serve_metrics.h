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

// `1bit serve`'s own Prometheus metrics (docs/serve.md, "Metrics"): only what the engine sees and
// Lemonade cannot. Backend starts and watchdog kills; what the backend's log says (CPU fallbacks,
// HRX graph-program cache flushes); the GTT pool and Tctl. GET /metrics renders
// these, then appends the backend's own /metrics where it has one.
#pragma once

#include "sysinfo.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#endif
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace onebit {

// What a backend's stderr says that is worth a metric. Lines that match nothing are ignored, and
// a metric whose line never came is left out rather than reported as zero.
struct BackendLogFacts {
    // ggml-hrx (fork #63): "HRX graph program cache over its limit (...): kept the current one, flush N"
    uint64_t program_cache_flushes = 0;
    // engine #390's load-time report of ops placed off HRX: a line naming "CPU fallback" (or
    // "off HRX") with OP=count, OP xcount or OP: count pairs, or only a total. llama-server drops
    // library INFO lines at its default verbosity (they map to its trace level), so the report
    // reaches serve only as a WARN (or above).
    std::optional<long> cpu_fallback_total;
    std::map<std::string, long> cpu_fallback_ops;
    // The same report names the graph's node count and split count (engine #390):
    // "placement_observer: placement: 512 nodes, 34 off HRX (MUL_MAT x 30), 7 graph splits".
    std::optional<long> placement_nodes;
    std::optional<long> placement_splits;
    size_t              placement_lines = 0;   // placement reports seen; serve prints the first

    void feed(const std::string& line) {
        // every backend line passes here: plain substring tests first, a regex only on a hit
        if (line.find("HRX graph program cache over its limit") != std::string::npos) {
            static const std::regex flush(R"(flush (\d+))");
            std::smatch m;
            const uint64_t n = std::regex_search(line, m, flush) ? std::stoull(m[1]) : 0;
            program_cache_flushes = std::max(program_cache_flushes + 1, n);
            return;
        }
        const size_t at = fallback_at(line);
        if (at == std::string::npos) return;
        static const std::regex op_count(R"(\b([A-Z][A-Z0-9_]{2,})\s*(?:=|:|x)\s*(\d+))");
        static const std::regex number(R"((\d+))");
        const std::string rest = line.substr(at);
        std::map<std::string, long> ops;
        long sum = 0;
        for (auto it = std::sregex_iterator(rest.begin(), rest.end(), op_count); it != std::sregex_iterator(); ++it) {
            ops[(*it)[1]] += std::stol((*it)[2]);
            sum += std::stol((*it)[2]);
        }
        std::smatch m;
        // "34 off HRX": the placement report states its own count, so the first number after the
        // phrase (which may be the split count) is not it.
        static const std::regex off_count(R"((\d+) off HRX\b)");
        const bool stated = std::regex_search(line, m, off_count);
        const long stated_off = stated ? std::stol(m[1]) : 0;
        if (!ops.empty()) {
            cpu_fallback_ops = ops;
            cpu_fallback_total = sum;
        } else if (!stated && std::regex_search(rest, m, number)) {
            cpu_fallback_total = std::stol(m[1]);   // a total alone, with no per-op counts
        }
        if (stated) {
            cpu_fallback_total = stated_off;
            cpu_fallback_ops = ops;   // empty for a clean graph ("none")
            ++placement_lines;
        }
        // the graph's size and its split count, when the line names them (engine #390)
        static const std::regex nodes_count(R"((\d+) nodes\b)");
        static const std::regex splits_count(R"((\d+) graph splits\b)");
        if (std::regex_search(line, m, nodes_count)) placement_nodes = std::stol(m[1]);
        if (std::regex_search(line, m, splits_count)) placement_splits = std::stol(m[1]);
    }

private:
    // where the text after "CPU fallback" / "cpu-fallback" / "off HRX" (any case) starts, or npos
    static size_t fallback_at(const std::string& line) {
        std::string low = line;
        for (char& c : low) c = char(std::tolower(static_cast<unsigned char>(c)));
        for (const char* key : {"cpu fallback", "cpu-fallback", "off hrx"})
            if (const size_t p = low.find(key); p != std::string::npos) return p + std::strlen(key);
        return std::string::npos;
    }
};

// engine #390's line for `1bit serve` at load, and what --strict-device prints instead. Both read
// what the backend's placement report said; an unstated count shows as "?".
inline std::string off_ops_detail(const BackendLogFacts& f) {
    std::string ops;
    for (const auto& [op, n] : f.cpu_fallback_ops) ops += (ops.empty() ? "" : ", ") + op + " x " + std::to_string(n);
    return ops.empty() ? "none" : ops;
}

inline std::string placement_summary(const BackendLogFacts& f) {
    const std::string nodes  = f.placement_nodes ? std::to_string(*f.placement_nodes) : "?";
    const std::string splits = f.placement_splits ? std::to_string(*f.placement_splits) : "?";
    const long        off    = f.cpu_fallback_total.value_or(0);
    if (off == 0) return "all " + nodes + " nodes on HRX, " + splits + " graph splits";
    return std::to_string(off) + " of " + nodes + " nodes off HRX (" + off_ops_detail(f) + "), " + splits +
           " graph splits";
}

inline std::string placement_refusal(const BackendLogFacts& f) {
    const std::string nodes  = f.placement_nodes ? std::to_string(*f.placement_nodes) : "?";
    const std::string splits = f.placement_splits ? std::to_string(*f.placement_splits) : "?";
    return "--strict-device: " + std::to_string(f.cpu_fallback_total.value_or(0)) + " of " + nodes +
           " nodes would run off HRX (" + off_ops_detail(f) + "), " + splits + " graph splits: refusing to start";
}

// Counts that outlive one serve process: Lemonade restarts a backend by starting `1bit serve`
// again, so starts, watchdog kills and backend exits are kept per model id in
// ~/.local/state/1bit/serve/<hash of the id>.json (the file names the model inside). Unwritable:
// counted in memory only.
class Lifecycle {
public:
    uint64_t starts = 0, watchdog_kills = 0, backend_exits = 0;

    explicit Lifecycle(const std::string& model_id) {
        // The directory comes from the user database, not the environment, and the file name is a
        // hash of the model id, so no outside string becomes a path (CodeQL cpp/path-injection).
        std::string home;
#ifndef _WIN32
        if (const passwd* pw = ::getpwuid(::geteuid()); pw && pw->pw_dir && *pw->pw_dir) home = pw->pw_dir;
#endif
        if (home.empty()) return;
        uint64_t h = 1469598103934665603ull;   // FNV-1a
        for (unsigned char c : model_id) { h ^= c; h *= 1099511628211ull; }
        char name[32];
        std::snprintf(name, sizeof name, "%016llx.json", static_cast<unsigned long long>(h));
        path_ = home + "/.local/state/1bit/serve/" + name;
        id_ = model_id;
        std::ifstream in(path_);
        const auto j = nlohmann::json::parse(in, nullptr, false);
        if (j.is_object()) {
            starts = j.value("starts", uint64_t(0));
            watchdog_kills = j.value("watchdog_kills", uint64_t(0));
            backend_exits = j.value("backend_exits", uint64_t(0));
        }
    }
    void count_start() { std::lock_guard<std::mutex> l(mu_); ++starts; save(); }
    void count_watchdog_kill() { std::lock_guard<std::mutex> l(mu_); ++watchdog_kills; save(); }
    void count_backend_exit() { std::lock_guard<std::mutex> l(mu_); ++backend_exits; save(); }
    const std::string& path() const { return path_; }

private:
    void save() {
        if (path_.empty()) return;
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path_).parent_path(), ec);
        const std::string tmp = path_ + ".tmp";
        {
            std::ofstream out(tmp, std::ios::trunc);
            if (!out) return;
            out << nlohmann::json{{"model", id_}, {"starts", starts}, {"watchdog_kills", watchdog_kills}, {"backend_exits", backend_exits}}.dump() << "\n";
            if (!out) return;
        }
        std::filesystem::rename(tmp, path_, ec);
    }
    std::string path_, id_;
    std::mutex mu_;
};

// The engine's own series. Lemonade's /metrics already reports requests, tokens, TTFT and tok/s per
// model (lemonade_model_*), CPU and memory use, and GTT + VRAM together as lemonade_vram_used_gb;
// these are what it cannot see from outside the engine.
class ServeMetrics {
public:
    explicit ServeMetrics(std::string device) : device_(std::move(device)) {}

    void backend_line(const std::string& line) {
        std::lock_guard<std::mutex> l(mu_);
        log_.feed(line);
    }

    // engine #390: the backend's load-time placement report, once a line named it
    std::optional<BackendLogFacts> placement() const {
        std::lock_guard<std::mutex> l(mu_);
        if (log_.placement_lines == 0) return std::nullopt;
        return log_;
    }

    // Prometheus text format 0.0.4
    std::string render(const Lifecycle& life, const std::string& sys_root = "") const {
        std::string out;
        auto line = [&](const std::string& s) { out += s; out += '\n'; };
        auto head = [&](const char* name, const char* type, const char* help) {
            line(std::string("# HELP ") + name + " " + help);
            line(std::string("# TYPE ") + name + " " + type);
        };
        std::lock_guard<std::mutex> l(mu_);
        head("onebit_up", "gauge", "1 while this 1bit serve runs; device names the route");
        line("onebit_up{device=\"" + esc(device_) + "\"} 1");
        head("onebit_serve_starts_total", "counter", "times 1bit serve loaded this model (kept across restarts)");
        line("onebit_serve_starts_total " + std::to_string(life.starts));
        head("onebit_watchdog_kills_total", "counter", "backends serve's progress watchdog killed (kept across restarts)");
        line("onebit_watchdog_kills_total " + std::to_string(life.watchdog_kills));
        head("onebit_backend_exits_total", "counter", "backends that exited on their own (kept across restarts)");
        line("onebit_backend_exits_total " + std::to_string(life.backend_exits));
        head("onebit_program_cache_flushes_total", "counter", "HRX graph-program cache flushes (backend log WARN)");
        line("onebit_program_cache_flushes_total " + std::to_string(log_.program_cache_flushes));
        if (log_.cpu_fallback_total) {
            head("onebit_cpu_fallback_ops", "gauge", "graph ops placed on the CPU instead of HRX (backend's load report, engine #390)");
            line("onebit_cpu_fallback_ops{op=\"all\"} " + std::to_string(*log_.cpu_fallback_total));
            for (const auto& [op, n] : log_.cpu_fallback_ops)
                line("onebit_cpu_fallback_ops{op=\"" + esc(op) + "\"} " + std::to_string(n));
        }
        const sysinfo::Memory mem = sysinfo::memory(sys_root);
        if (mem.has_gtt) {
            head("onebit_gtt_bytes", "gauge", "the amdgpu GTT pool (mem_info_gtt_*), which HRX allocates from");
            line("onebit_gtt_bytes{kind=\"total\"} " + std::to_string(mem.gtt_total));
            line("onebit_gtt_bytes{kind=\"used\"} " + std::to_string(mem.gtt_used));
            head("onebit_gtt_fault_zone", "gauge", "1 when GTT use is at or above 95 GiB, where engine #315's faults were seen");
            line(std::string("onebit_gtt_fault_zone ") + (mem.gtt_used >= (95ull << 30) ? "1" : "0"));
        }
        if (const auto t = sysinfo::tctl_celsius(sys_root)) {
            head("onebit_tctl_celsius", "gauge", "CPU Tctl (k10temp)");
            line("onebit_tctl_celsius " + fmt(*t));
        }
        return out;
    }

private:
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

    std::string device_;
    mutable std::mutex mu_;
    BackendLogFacts log_;
};

}  // namespace onebit
