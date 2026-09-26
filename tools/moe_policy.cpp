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
//
// moe_policy: replays a tools/moe_trace.cpp trace through expert cache policies and prints the
// decode hit rate per policy and cache size (docs/moe-streaming.md).
//
//   moe_policy trace.tsv [cap ...]      caps are experts cached per layer (a shared budget
//                                       holds cap x layers); default 32 64 96 128 192 256
//
// Prompt tokens warm the cache and are not counted; decode tokens are. Policies:
//   lru, lfu, slru     one cache per layer (slru: 20% probation, 80% protected)
//   g-lru, g-lfu       one budget shared by all layers (g-lfu: frequency halved every 64 tokens)
//   opt                Belady per layer: evict the expert used furthest in the future (bound)
// Prefetch: for each decode miss at layer l+1, whether it was among the experts layer l+1 used
// for the previous token, or among the next layer's most likely experts given this layer's
// (co-occurrence counted so far), P predictions per routed expert.
// Build: g++ -std=c++17 -O2 tools/moe_policy.cpp -o moe_policy
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct Access { int token, layer, expert; bool decode; };

struct Trace {
    int n_layer = 0, n_exp = 0, top_k = 0;
    // tokens in order: token -> layer -> experts
    std::vector<std::vector<std::vector<int>>> routes;
    std::vector<std::vector<std::vector<int>>> pred;   // gate-ahead prediction, decode tokens (may be empty)
    std::vector<std::vector<std::vector<int>>> pred2;  // the same from two layers back
    std::vector<bool> is_decode;
    bool has_pred = false;
};

static Trace load(const char* path) {
    Trace t;
    std::ifstream f(path);
    std::string line;
    // rows of one step appear layer by layer, token by token within the layer
    long cur_step = -1; std::map<int, std::vector<std::vector<int>>> step_rows, step_pred, step_pred2; char cur_kind = 'P';
    auto flush = [&]() {
        if (step_rows.empty()) return;
        size_t n_tok = step_rows.begin()->second.size();
        for (size_t j = 0; j < n_tok; j++) {
            std::vector<std::vector<int>> per_layer, per_pred, per_pred2;
            for (auto& [l, rows] : step_rows) {
                if ((int) per_layer.size() <= l) { per_layer.resize(l + 1); per_pred.resize(l + 1); per_pred2.resize(l + 1); }
                if (j < rows.size()) per_layer[l] = rows[j];
                auto pit = step_pred.find(l);
                if (pit != step_pred.end() && j < pit->second.size()) per_pred[l] = pit->second[j];
                auto pit2 = step_pred2.find(l);
                if (pit2 != step_pred2.end() && j < pit2->second.size()) per_pred2[l] = pit2->second[j];
            }
            t.routes.push_back(std::move(per_layer));
            t.pred.push_back(std::move(per_pred));
            t.pred2.push_back(std::move(per_pred2));
            t.is_decode.push_back(cur_kind == 'D');
        }
        step_rows.clear(); step_pred.clear(); step_pred2.clear();
    };
    while (std::getline(f, line)) {
        std::vector<std::string> col; std::stringstream ss(line); std::string c;
        while (std::getline(ss, c, '\t')) col.push_back(c);
        if (col.size() < 3) continue;
        const long step = atol(col[0].c_str());
        const int layer = atoi(col[1].c_str());
        char kind; std::string ex;
        if (col.size() >= 4) { kind = col[2][0]; ex = col[3]; }
        else { kind = step == 0 ? 'P' : 'D'; ex = col[2]; }  // old format: step 0 is the prompt
        if (step != cur_step) { flush(); cur_step = step; cur_kind = kind; }
        std::vector<int> e; std::stringstream es(ex);
        while (std::getline(es, c, ',')) { e.push_back(atoi(c.c_str())); t.n_exp = std::max(t.n_exp, e.back() + 1); }
        t.top_k = std::max<int>(t.top_k, e.size());
        t.n_layer = std::max(t.n_layer, layer + 1);
        step_rows[layer].push_back(e);
        std::vector<int> p;
        if (col.size() >= 5 && col[4] != "-") { std::stringstream ps(col[4]); while (std::getline(ps, c, ',')) p.push_back(atoi(c.c_str())); t.has_pred = true; }
        step_pred[layer].push_back(p);
        std::vector<int> p2;
        if (col.size() >= 6 && col[5] != "-") { std::stringstream ps(col[5]); while (std::getline(ps, c, ',')) p2.push_back(atoi(c.c_str())); }
        step_pred2[layer].push_back(p2);
    }
    flush();
    for (auto& r : t.routes) r.resize(t.n_layer);
    for (auto& r : t.pred) r.resize(t.n_layer);
    for (auto& r : t.pred2) r.resize(t.n_layer);
    return t;
}

// ---------------- per-layer policies ----------------
struct LRU {
    int cap; std::list<int> l; std::unordered_map<int, std::list<int>::iterator> pos;
    explicit LRU(int c) : cap(c) {}
    bool touch(int e) {
        auto it = pos.find(e);
        if (it != pos.end()) { l.splice(l.begin(), l, it->second); return true; }
        if ((int) l.size() >= cap) { pos.erase(l.back()); l.pop_back(); }
        l.push_front(e); pos[e] = l.begin(); return false;
    }
};
struct LFU {  // evicts the lowest count, oldest use on ties
    int cap; long clock = 0; std::unordered_map<int, std::pair<long, long>> in;  // e -> (count, last)
    std::unordered_map<int, long> count;  // counts survive eviction
    explicit LFU(int c) : cap(c) {}
    bool touch(int e) {
        clock++; long& n = count[e]; n++;
        auto it = in.find(e);
        if (it != in.end()) { it->second = { n, clock }; return true; }
        if ((int) in.size() >= cap) {
            auto v = std::min_element(in.begin(), in.end(), [](auto& a, auto& b) { return a.second < b.second; });
            in.erase(v);
        }
        in[e] = { n, clock }; return false;
    }
};
struct SLRU {
    LRU prob, prot;
    explicit SLRU(int c) : prob(std::max(1, c / 5)), prot(std::max(1, c - c / 5)) {}
    bool touch(int e) {
        if (prot.pos.count(e)) { prot.touch(e); return true; }
        auto it = prob.pos.find(e);
        if (it != prob.pos.end()) {  // promote; the protected tail drops back to probation
            prob.l.erase(it->second); prob.pos.erase(it);
            if ((int) prot.l.size() >= prot.cap) { int d = prot.l.back(); prot.pos.erase(d); prot.l.pop_back(); prob.touch(d); }
            prot.l.push_front(e); prot.pos[e] = prot.l.begin();
            return true;
        }
        prob.touch(e); return false;
    }
};

template <class P>
static double per_layer(const Trace& t, int cap) {
    long hit = 0, tot = 0;
    for (int l = 0; l < t.n_layer; l++) {
        P p(cap);
        for (size_t s = 0; s < t.routes.size(); s++)
            for (int e : t.routes[s][l]) { bool h = p.touch(e); if (t.is_decode[s]) { tot++; hit += h; } }
    }
    return tot ? 100.0 * hit / tot : 0;
}

static double opt_per_layer(const Trace& t, int cap) {
    long hit = 0, tot = 0;
    for (int l = 0; l < t.n_layer; l++) {
        std::vector<std::pair<int, int>> seq;  // (expert, token)
        for (size_t s = 0; s < t.routes.size(); s++) for (int e : t.routes[s][l]) seq.push_back({ e, (int) s });
        std::vector<int> next(seq.size(), 1 << 30); std::unordered_map<int, int> last;
        for (int i = (int) seq.size() - 1; i >= 0; i--) { auto it = last.find(seq[i].first); if (it != last.end()) next[i] = it->second; last[seq[i].first] = i; }
        std::unordered_map<int, int> in;  // expert -> next use
        std::set<std::pair<int, int>> by_next;  // (next use, expert)
        for (size_t i = 0; i < seq.size(); i++) {
            const int e = seq[i].first; const bool dec = t.is_decode[seq[i].second];
            auto it = in.find(e); bool h = it != in.end();
            if (h) by_next.erase({ it->second, e });
            else if ((int) in.size() >= cap) { auto v = std::prev(by_next.end()); in.erase(v->second); by_next.erase(v); }
            in[e] = next[i]; by_next.insert({ next[i], e });
            if (dec) { tot++; hit += h; }
        }
    }
    return tot ? 100.0 * hit / tot : 0;
}

// ---------------- shared-budget policies ----------------
static double global_lru(const Trace& t, int cap) {
    LRU p(cap * t.n_layer); long hit = 0, tot = 0;
    for (size_t s = 0; s < t.routes.size(); s++)
        for (int l = 0; l < t.n_layer; l++) for (int e : t.routes[s][l]) { bool h = p.touch(l * t.n_exp + e); if (t.is_decode[s]) { tot++; hit += h; } }
    return tot ? 100.0 * hit / tot : 0;
}
static double global_lfu_decay(const Trace& t, int cap) {
    const int budget = cap * t.n_layer; long hit = 0, tot = 0;
    std::unordered_map<int, double> score; std::set<std::pair<double, int>> order; std::unordered_map<int, double> in;
    for (size_t s = 0; s < t.routes.size(); s++) {
        if (s && s % 64 == 0) {  // age: halve every score, cached ones too
            for (auto& [k, v] : score) v *= 0.5;
            std::set<std::pair<double, int>> o2; for (auto& [k, v] : in) { v *= 0.5; o2.insert({ v, k }); } order.swap(o2);
        }
        for (int l = 0; l < t.n_layer; l++) for (int e : t.routes[s][l]) {
            const int key = l * t.n_exp + e; double& sc = score[key]; sc += 1.0;
            auto it = in.find(key); bool h = it != in.end();
            if (h) { order.erase({ it->second, key }); }
            else if ((int) in.size() >= budget) { auto v = order.begin(); in.erase(v->second); order.erase(v); }
            in[key] = sc; order.insert({ sc, key });
            if (t.is_decode[s]) { tot++; hit += h; }
        }
    }
    return tot ? 100.0 * hit / tot : 0;
}

// ---------------- prefetch coverage (with per-layer LRU at cap) ----------------
static void prefetch(const Trace& t, int cap, int per_expert) {
    long misses = 0, prev_cov = 0, cooc_cov = 0, prev_extra = 0, cooc_extra = 0;
    long ga_k = 0, ga_2k = 0, ga_extra_k = 0, ga_extra_2k = 0, routed = 0, ga_hit_k = 0;
    long g2_k = 0, g2_hit_k = 0, routed2 = 0;
    std::vector<LRU> c(t.n_layer, LRU(cap));
    std::vector<std::unordered_map<long, int>> co(t.n_layer);  // (e_l * n_exp + e_next) -> count
    for (size_t s = 0; s < t.routes.size(); s++) {
        for (int l = 0; l < t.n_layer; l++) {
            const auto& cur = t.routes[s][l];
            std::set<int> pred_prev, pred_co;
            if (l > 0 && t.is_decode[s]) {
                if (s > 0) for (int e : t.routes[s - 1][l]) if (!c[l].pos.count(e)) pred_prev.insert(e);
                for (int e : t.routes[s][l - 1]) {  // this token's previous layer routes
                    std::vector<std::pair<int, int>> cand;
                    for (int n = 0; n < t.n_exp; n++) { auto it = co[l - 1].find((long) e * t.n_exp + n); if (it != co[l - 1].end()) cand.push_back({ -it->second, n }); }
                    std::partial_sort(cand.begin(), cand.begin() + std::min<size_t>(per_expert, cand.size()), cand.end());
                    for (int i = 0; i < per_expert && i < (int) cand.size(); i++) if (!c[l].pos.count(cand[i].second)) pred_co.insert(cand[i].second);
                }
                prev_extra += pred_prev.size(); cooc_extra += pred_co.size();
            }
            const auto& ga = t.pred[s][l];
            std::set<int> ga1, ga2;
            if (t.is_decode[s] && !ga.empty()) {
                for (size_t i = 0; i < ga.size(); i++) (i < cur.size() ? ga1 : ga2).insert(ga[i]);
                for (int e : ga1) ga2.insert(e);
                std::set<int> used(cur.begin(), cur.end());
                for (int e : ga1) { if (!c[l].pos.count(e) && !used.count(e)) ga_extra_k++; }
                for (int e : ga2) { if (!c[l].pos.count(e) && !used.count(e)) ga_extra_2k++; }
                for (int e : cur) { routed++; ga_hit_k += ga1.count(e); }
            }
            std::set<int> g2;
            const auto& p2 = t.pred2[s][l];
            if (t.is_decode[s] && !p2.empty()) {
                for (size_t i = 0; i < p2.size() && i < cur.size(); i++) g2.insert(p2[i]);
                for (int e : cur) { routed2++; g2_hit_k += g2.count(e); }
            }
            for (int e : cur) {
                bool h = c[l].touch(e);
                if (!h && t.is_decode[s] && l > 0) g2_k += g2.count(e);
                if (!h && t.is_decode[s] && l > 0) {
                    misses++; prev_cov += pred_prev.count(e); cooc_cov += pred_co.count(e);
                    ga_k += ga1.count(e); ga_2k += ga2.count(e);
                }
            }
            if (l > 0) for (int a : t.routes[s][l - 1]) for (int b : cur) co[l - 1][(long) a * t.n_exp + b]++;
        }
    }
    auto pct = [&](long a) { return misses ? 100.0 * a / misses : 0.0; };
    size_t n_dec = std::count(t.is_decode.begin(), t.is_decode.end(), true);
    printf("prefetch @%d/layer: %ld decode misses (layers 1+); previous token's set covers %.1f%% (%.1f reads/token),"
           " co-occurrence top-%d covers %.1f%% (%.1f reads/token)\n",
           cap, misses, pct(prev_cov), (double) prev_extra / std::max<size_t>(1, n_dec), per_expert, pct(cooc_cov),
           (double) cooc_extra / std::max<size_t>(1, n_dec));
    if (t.has_pred)
        printf("  gate-ahead @%d/layer: routing accuracy (top-k) %.1f%%; covers %.1f%% of misses with top-k"
               " (%.1f wasted reads/token), %.1f%% with top-2k (%.1f wasted reads/token)\n",
               cap, routed ? 100.0 * ga_hit_k / routed : 0.0, pct(ga_k), (double) ga_extra_k / std::max<size_t>(1, n_dec),
               pct(ga_2k), (double) ga_extra_2k / std::max<size_t>(1, n_dec));
    if (routed2)
        printf("  two-ahead @%d/layer: routing accuracy (top-k) %.1f%%; covers %.1f%% of misses with top-k\n",
               cap, 100.0 * g2_hit_k / routed2, pct(g2_k));
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "moe_policy trace.tsv [cap ...]\n"); return 2; }
    Trace t = load(argv[1]);
    std::vector<int> caps;
    for (int i = 2; i < argc; i++) caps.push_back(atoi(argv[i]));
    if (caps.empty()) caps = { 32, 64, 96, 128, 192, 256 };
    size_t n_dec = std::count(t.is_decode.begin(), t.is_decode.end(), true);
    printf("%s: %d layers, %d experts, top-%d, %zu prompt + %zu decode tokens\n", argv[1], t.n_layer, t.n_exp, t.top_k,
           t.routes.size() - n_dec, n_dec);
    printf("%6s %7s %7s %7s %7s %7s %7s\n", "cap", "lru", "lfu", "slru", "g-lru", "g-lfu", "opt");
    for (int cap : caps) {
        if (cap >= t.n_exp) { printf("%6d  (all %d experts fit)\n", cap, t.n_exp); continue; }
        printf("%6d %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%%\n", cap, per_layer<LRU>(t, cap), per_layer<LFU>(t, cap),
               per_layer<SLRU>(t, cap), global_lru(t, cap), global_lfu_decay(t, cap), opt_per_layer(t, cap));
        fflush(stdout);
    }
    for (int cap : caps) if (cap < t.n_exp) prefetch(t, cap, 2);
    return 0;
}
