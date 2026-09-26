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
#include "cache_bench.h"

#include "expert_cache.h"
#include "gguf_index.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace onebit::moe {

namespace {

struct Token {
    bool decode = false;
    std::map<int, std::vector<int>> routes, pred, pred2;  // layer -> experts
};

// tools/moe_trace.cpp traces: step \t layer \t kind \t experts [\t predicted]
std::vector<Token> load_trace(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<Token> out;
    std::string line;
    long cur = -1;
    struct Row { std::vector<int> routes, pred, pred2; };
    std::map<int, std::vector<Row>> rows;  // layer -> per-token
    char kind = 'P';
    auto split = [](const std::string& s) {
        std::vector<int> v;
        if (s == "-") return v;
        std::stringstream ss(s);
        std::string c;
        while (std::getline(ss, c, ',')) v.push_back(std::stoi(c));
        return v;
    };
    auto flush = [&] {
        if (rows.empty()) return;
        const size_t n = rows.begin()->second.size();
        for (size_t j = 0; j < n; ++j) {
            Token t;
            t.decode = kind == 'D';
            for (auto& [l, r] : rows)
                if (j < r.size()) { t.routes[l] = r[j].routes; t.pred[l] = r[j].pred; t.pred2[l] = r[j].pred2; }
            out.push_back(std::move(t));
        }
        rows.clear();
    };
    while (std::getline(f, line)) {
        std::vector<std::string> col;
        std::stringstream ss(line);
        std::string c;
        while (std::getline(ss, c, '\t')) col.push_back(c);
        if (col.size() < 4) continue;
        const long step = std::stol(col[0]);
        if (step != cur) { flush(); cur = step; kind = col[2][0]; }
        rows[std::stoi(col[1])].push_back({split(col[3]), col.size() > 4 ? split(col[4]) : std::vector<int>{},
                                           col.size() > 5 ? split(col[5]) : std::vector<int>{}});
    }
    flush();
    return out;
}

void usage() {
    std::printf(
        "usage: 1bit moe-cache --model <gguf> --trace <tsv> [options]\n"
        "  replays a tools/moe_trace.cpp trace through the expert cache with real O_DIRECT reads\n"
        "  from the model file and simulated compute, and prints the decode rate it allows\n"
        "  --slots N           experts held in RAM (default 1024)\n"
        "  --per-layer         slots / layers per layer instead of one shared LRU\n"
        "  --io N              reads in flight (default 8)\n"
        "  --prefetch none|k|2k|oracle  gate-ahead prefetch of the trace's predictions (default k);\n"
        "                      oracle queues every layer's real experts when a token starts: the\n"
        "                      bound for a perfect predictor with every read in flight at once\n"
        "  --lookahead 1|2     prefetch a layer's experts at its own start (1, from the previous\n"
        "                      layer's output) or one layer earlier (2, from two layers back)\n"
        "  --compute-ms X      compute per token when every expert is resident (default 20)\n"
        "  --attn-frac F       share of a layer's compute before its experts are needed (default 0.5)\n"
        "  --warm N            prompt tokens replayed to warm the cache, untimed (default 64)\n"
        "  --tokens N          decode tokens replayed (default all)\n"
        "  --no-pin            do not mlock the slots\n");
}

}  // namespace

int run_moe_cache(int argc, char** argv) {
    std::string model, trace, prefetch = "k";
    CacheOptions opt;
    double compute_ms = 20, attn_frac = 0.5;
    int warm = 64, max_tokens = 1 << 30, lookahead = 1;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--trace") trace = next();
        else if (a == "--slots") opt.slots = std::stoi(next());
        else if (a == "--per-layer") opt.per_layer = true;
        else if (a == "--io") opt.io_threads = std::stoi(next());
        else if (a == "--prefetch") prefetch = next();
        else if (a == "--lookahead") lookahead = std::stoi(next());
        else if (a == "--compute-ms") compute_ms = std::stod(next());
        else if (a == "--attn-frac") attn_frac = std::stod(next());
        else if (a == "--warm") warm = std::stoi(next());
        else if (a == "--tokens") max_tokens = std::stoi(next());
        else if (a == "--no-pin") opt.pin = false;
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else throw std::runtime_error("unknown option " + a);
    }
    if (model.empty() || trace.empty()) { usage(); return 2; }
    if (prefetch != "none" && prefetch != "k" && prefetch != "2k" && prefetch != "oracle")
        throw std::runtime_error("--prefetch none|k|2k|oracle");

    const GgufIndex index = GgufIndex::open(model);
    const auto tokens = load_trace(trace);
    ExpertCache cache(index, opt);
    const int n_moe = (int) index.experts.size();
    const double layer_ms = compute_ms / n_moe;
    double expert_bytes = 0;
    for (const auto& [l, parts] : index.experts) expert_bytes += index.expert_bytes(l);
    std::printf("%s: %d MoE layers, %d experts, %.2f MiB per expert on average; %d slots, %.1f GiB pinned (largest slot %.2f MiB)%s\n",
                model.c_str(), n_moe, index.n_expert, expert_bytes / n_moe / 1048576.0, cache.slots(),
                cache.pinned_bytes() / double(1u << 30), cache.slot_bytes() / 1048576.0, opt.per_layer ? ", per layer" : ", shared");

    auto sleep_ms = [](double ms) {  // compute stand-in: spin for sub-millisecond precision
        const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double, std::milli>(ms);
        while (std::chrono::steady_clock::now() < end) {}
    };
    auto issue = [&](const Token& t, int l, const std::map<int, std::vector<int>>& pred) {
        auto it = pred.find(l);
        auto rt = t.routes.find(l);
        if (it == pred.end() || it->second.empty() || rt == t.routes.end() || !index.experts.count(l)) return;
        const size_t k = prefetch == "k" ? rt->second.size() : it->second.size();
        cache.prefetch(l, std::vector<int>(it->second.begin(), it->second.begin() + std::min(k, it->second.size())));
    };
    auto run_token = [&](const Token& t, bool timed) {
        if (timed && prefetch == "oracle")
            for (const auto& [l, experts] : t.routes) if (index.experts.count(l)) cache.prefetch(l, experts);
        for (const auto& [l, experts] : t.routes) {
            if (!index.experts.count(l)) continue;
            if (timed && prefetch != "none" && prefetch != "oracle") {
                if (lookahead == 1) issue(t, l, t.pred);
                else issue(t, l + 1, t.pred2);  // the next layer's experts, while this layer runs
            }
            if (timed) sleep_ms(layer_ms * attn_frac);           // attention while the prefetch reads
            cache.acquire(l, experts);
            if (timed) sleep_ms(layer_ms * (1 - attn_frac));     // the experts
            cache.release(l, experts);
        }
    };

    // warm: the last prompt tokens before the first decode token, untimed
    size_t first_decode = 0;
    while (first_decode < tokens.size() && !tokens[first_decode].decode) ++first_decode;
    for (size_t i = first_decode > size_t(warm) ? first_decode - warm : 0; i < first_decode; ++i) run_token(tokens[i], false);
    cache.reset_stats();

    int n = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = first_decode; i < tokens.size() && n < max_tokens; ++i) {
        if (!tokens[i].decode) {  // a later prompt (chat turn): replayed untimed, like a prefill
            run_token(tokens[i], false);
            continue;
        }
        run_token(tokens[i], true);
        ++n;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const CacheStats st = cache.stats();
    const double acc = double(st.hits + st.prefetch_hits + st.misses);
    std::printf("decode %d tokens: %.2f tok/s (compute alone %.2f tok/s); stall %.1f ms/token\n", n, n / s,
                1000.0 / compute_ms, st.stall_ms / std::max(1, n));
    std::printf("  hits %.1f%%, prefetched %.1f%%, demand misses %.1f%%; prefetches %llu (%.1f%% wasted); read %.2f GB (%.1f MB/token)\n",
                100 * st.hits / acc, 100 * st.prefetch_hits / acc, 100 * st.misses / acc,
                (unsigned long long) st.prefetches, st.prefetches ? 100.0 * st.prefetch_wasted / st.prefetches : 0.0,
                st.bytes_read / 1e9, st.bytes_read / 1e6 / std::max(1, n));
    return 0;
}

}  // namespace onebit::moe
