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
//   g-arc              ARC over the shared budget
//   g-lru2             LRU-2 over the shared budget (oldest second-to-last use goes first)
//   g-tlfu1, g-tlfu10  W-TinyLFU over the shared budget, window 1% / 10%: frequency-gated admission
//   opt                Belady per layer: evict the expert used furthest in the future (bound)
// Prefetch: for each decode miss at layer l+1, whether it was among the experts layer l+1 used
// for the previous token, or among the next layer's most likely experts given this layer's
// (co-occurrence counted so far), P predictions per routed expert.
// Drive ceiling (env MOE_EXPERT_MIB, MOE_COMPUTE_MS, MOE_DRIVE_GBS default 3.7): for the shared
// LRU at each cap, the decode rate when every miss is read from the drive, serially after the
// compute (no overlap) and fully hidden behind it (perfect overlap).
// MTP (env MOE_MTP="n accepted", e.g. "3 2.2"): each verification step routes a window of n+1
// consecutive decoded tokens, so a layer reads the union of their experts; the window then
// advances by the accepted tokens per step. Rejected drafts are approximated by the real next
// tokens. Prints experts read per generated token with and without MTP (shared LRU).
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
#include <tuple>
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

// Replays every access through one shared-budget policy P(budget) with bool touch(key).
template <class P>
static double global_policy(const Trace& t, int cap, P p) {
    long hit = 0, tot = 0;
    for (size_t s = 0; s < t.routes.size(); s++)
        for (int l = 0; l < t.n_layer; l++) for (int e : t.routes[s][l]) { bool h = p.touch(l * t.n_exp + e); if (t.is_decode[s]) { tot++; hit += h; } }
    (void) cap;
    return tot ? 100.0 * hit / tot : 0;
}

// ARC (Megiddo & Modha): recency list T1 and frequency list T2, with ghost lists B1/B2 steering
// the split p between them.
struct ARC {
    int c; double p = 0; LRU t1, t2, b1, b2;
    explicit ARC(int cap) : c(cap), t1(1 << 30), t2(1 << 30), b1(1 << 30), b2(1 << 30) {}
    static void drop(LRU& x, int e) { auto it = x.pos.find(e); x.l.erase(it->second); x.pos.erase(it); }
    static int pop_back(LRU& x) { int e = x.l.back(); x.pos.erase(e); x.l.pop_back(); return e; }
    void replace(bool in_b2) {
        if (!t1.l.empty() && ((int) t1.l.size() > p || (in_b2 && (int) t1.l.size() == (int) p))) b1.touch(pop_back(t1));
        else if (!t2.l.empty()) b2.touch(pop_back(t2));
        else b1.touch(pop_back(t1));
    }
    bool touch(int e) {
        if (t1.pos.count(e)) { drop(t1, e); t2.touch(e); return true; }
        if (t2.pos.count(e)) { t2.touch(e); return true; }
        if (b1.pos.count(e)) {
            p = std::min<double>(c, p + std::max(1.0, (double) b2.l.size() / std::max<size_t>(1, b1.l.size())));
            replace(false); drop(b1, e); t2.touch(e); return false;
        }
        if (b2.pos.count(e)) {
            p = std::max(0.0, p - std::max(1.0, (double) b1.l.size() / std::max<size_t>(1, b2.l.size())));
            replace(true); drop(b2, e); t2.touch(e); return false;
        }
        const int l1 = (int) (t1.l.size() + b1.l.size()), tot = l1 + (int) (t2.l.size() + b2.l.size());
        if (l1 == c) {
            if ((int) t1.l.size() < c) { pop_back(b1); replace(false); } else pop_back(t1);
        } else if (tot >= c) {
            if (tot == 2 * c) pop_back(b2);
            replace(false);
        }
        t1.touch(e); return false;
    }
};

// LRU-2: evicts the expert whose second-to-last use is oldest (one use so far: oldest last use first).
struct LRU2 {
    int cap; long clock = 0;
    std::unordered_map<int, std::pair<long, long>> hist;  // e -> (second-to-last, last); kept after eviction
    std::set<std::tuple<long, long, int>> order; std::unordered_map<int, std::pair<long, long>> in;
    explicit LRU2(int c) : cap(c) {}
    bool touch(int e) {
        clock++;
        auto& h = hist[e]; h = { h.second ? h.second : -1, clock };
        if (h.first < 0) h.first = -(1L << 40) + clock;  // one use: before every twice-used expert
        auto it = in.find(e); bool hit = it != in.end();
        if (hit) order.erase({ it->second.first, it->second.second, e });
        else if ((int) in.size() >= cap) { auto v = order.begin(); in.erase(std::get<2>(*v)); order.erase(v); }
        in[e] = h; order.insert({ h.first, h.second, e });
        return hit;
    }
};

// W-TinyLFU (Einziger et al., as in Caffeine): a small LRU window takes new experts; one leaving
// the window enters the main cache only if it was used more often than the main cache's victim.
// Main is a segmented LRU: protected (at most 80%) and probation (the rest); a hit in probation
// moves to protected. Counts are halved every 10 x budget accesses.
struct WTinyLFU {
    LRU win, prob, prot; int main_cap, prot_cap; long n = 0, sample; std::unordered_map<int, int> freq;
    WTinyLFU(int cap, double w)
        : win(std::max(1, (int) (cap * w))), prob(1 << 30), prot(1 << 30), main_cap(std::max(1, cap - std::max(1, (int) (cap * w)))),
          prot_cap(main_cap * 4 / 5), sample(10L * cap) {}
    int f(int e) { auto it = freq.find(e); return it == freq.end() ? 0 : it->second; }
    static int pop_back(LRU& x) { int e = x.l.back(); x.pos.erase(e); x.l.pop_back(); return e; }
    bool touch(int e) {
        if (++n % sample == 0) for (auto it = freq.begin(); it != freq.end();) { if ((it->second /= 2) == 0) it = freq.erase(it); else ++it; }
        freq[e]++;
        if (prot.pos.count(e)) { prot.touch(e); return true; }
        if (auto it = prob.pos.find(e); it != prob.pos.end()) {  // promote; protected's tail falls back to probation
            prob.l.erase(it->second); prob.pos.erase(it);
            prot.touch(e);
            if ((int) prot.l.size() > prot_cap) prob.touch(pop_back(prot));
            return true;
        }
        if (win.pos.count(e)) { win.touch(e); return true; }
        if ((int) win.l.size() < win.cap) { win.touch(e); return false; }
        const int cand = win.l.back();
        win.touch(e);  // pushes cand out of the window
        if ((int) (prob.l.size() + prot.l.size()) < main_cap) { prob.touch(cand); return false; }
        LRU& vl = prob.l.empty() ? prot : prob;
        if (f(cand) > f(vl.l.back())) { pop_back(vl); prob.touch(cand); }
        return false;
    }
};

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
    printf("%6s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s\n", "cap", "lru", "lfu", "slru", "g-lru", "g-lfu", "g-arc", "g-lru2",
           "g-tlfu1", "g-tlfu10", "opt");
    for (int cap : caps) {
        if (cap >= t.n_exp) { printf("%6d  (all %d experts fit)\n", cap, t.n_exp); continue; }
        const int budget = cap * t.n_layer;
        printf("%6d %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%%\n", cap, per_layer<LRU>(t, cap),
               per_layer<LFU>(t, cap), per_layer<SLRU>(t, cap), global_lru(t, cap), global_lfu_decay(t, cap),
               global_policy(t, cap, ARC(budget)), global_policy(t, cap, LRU2(budget)),
               global_policy(t, cap, WTinyLFU(budget, 0.01)), global_policy(t, cap, WTinyLFU(budget, 0.10)), opt_per_layer(t, cap));
        fflush(stdout);
    }
    for (int cap : caps) if (cap < t.n_exp) prefetch(t, cap, 2);
    if (getenv("MOE_MTP")) {
        int n = 3; double acc = 2.0;
        sscanf(getenv("MOE_MTP"), "%d %lf", &n, &acc);
        std::vector<size_t> dec;
        for (size_t s = 0; s < t.routes.size(); s++) if (t.is_decode[s]) dec.push_back(s);
        for (int cap : caps) {
            if (cap >= t.n_exp) continue;
            LRU plain(cap * t.n_layer), mtp(cap * t.n_layer);
            long miss_plain = 0, miss_mtp = 0, steps = 0;
            // warm both with the prompt
            for (size_t s = 0; s < t.routes.size() && !t.is_decode[s]; s++)
                for (int l = 0; l < t.n_layer; l++) for (int e : t.routes[s][l]) { plain.touch(l * t.n_exp + e); mtp.touch(l * t.n_exp + e); }
            for (size_t s : dec) for (int l = 0; l < t.n_layer; l++) for (int e : t.routes[s][l]) miss_plain += !plain.touch(l * t.n_exp + e);
            double pos = 0;
            while (pos < dec.size()) {
                const size_t i0 = (size_t) pos, i1 = std::min(dec.size(), i0 + n + 1);
                for (int l = 0; l < t.n_layer; l++) {
                    std::set<int> u;
                    for (size_t i = i0; i < i1; i++) for (int e : t.routes[dec[i]][l]) u.insert(e);
                    for (int e : u) miss_mtp += !mtp.touch(l * t.n_exp + e);
                }
                steps++;
                pos += acc;
            }
            const double toks = dec.size();
            printf("MTP n=%d, %.2f accepted/step @%d/layer: %.1f expert reads per generated token (%.1f without MTP), %.2f per step\n",
                   n, acc, cap, miss_mtp / toks, miss_plain / toks, (double) miss_mtp / std::max(1L, steps));
        }
    }
    if (getenv("MOE_EXPERT_MIB") && getenv("MOE_COMPUTE_MS")) {
        const double mib = atof(getenv("MOE_EXPERT_MIB")), comp = atof(getenv("MOE_COMPUTE_MS"));
        const double gbs = getenv("MOE_DRIVE_GBS") ? atof(getenv("MOE_DRIVE_GBS")) : 3.7;
        const double per_tok = double(t.n_layer) * t.top_k;  // routed experts per token (MoE layers)
        for (int cap : caps) {
            if (cap >= t.n_exp) continue;
            const double miss = 1.0 - global_lru(t, cap) / 100.0;
            const double io_ms = per_tok * miss * mib * 1048576.0 / (gbs * 1e9) * 1e3;
            printf("drive ceiling @%d/layer (g-lru, %.2f MiB/expert, %.1f GB/s): %.1f MB/token read; %.1f tok/s serial, %.1f tok/s overlapped (compute alone %.1f)\n",
                   cap, mib, gbs, per_tok * miss * mib * 1.048576, 1000.0 / (comp + io_ms), 1000.0 / std::max(comp, io_ms), 1000.0 / comp);
        }
    }
    return 0;
}
