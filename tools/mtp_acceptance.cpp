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

// mtp_acceptance.cpp -- the MTP drafter measured against the real base model: draft, verify, commit, repeat.
//
//   mtp_acceptance --model <dir> --head-exact <pb16i-exact dir> --head-q4k <q4_k dir>
//                  [--tokens 64] [--k 4] [--tau 0.6] [--prompts 8] [--residual] [--dry-head]
//                  [--probe-lm N]
//                  [--offline [--positions-in F]] [--live [--positions-out F] [--match]]
//
// --match is the one that settles "live equals offline": the live loop records where it opened its chains
// (`--positions-out` writes them; `--offline --positions-in` scores those positions only), and --match scores the
// offline protocol at exactly those positions inside the same process, then prints live vs offline per depth with
// the 2 pp bar.  Without it the two tables are different samples of the same drafter and the difference is mostly
// which positions each one happened to reach.
//
// The base runs the exact route (pb16i container, kernels-m16) through Q4nxNpuForward; the drafter is the MTP
// head on the host (npu/mtp/mtp_head.h, the pb16i-exact arithmetic: the same codes, magnitudes and host
// additive table the NPU head would use).  Verification goes through step_batch_all(), so every draft is
// checked against the model's OWN choice at that position -- which is what makes "accepted" mean anything.
//
// Reported per run: tokens emitted, passes (steps + batches), passes/token, drafts, accepted, acceptance
// length (accepted/iterations), hit rate (accepted/drafted), and the drafter's first-draft acceptance.
//
// --dry-head runs the head with lm_head disabled (no vocab logits): it checks the plumbing and the head's
// timing without touching the model's head -- useful before the device window.
#include "mtp/draft.h"
#include "mtp/forward_base.h"
#include "mtp/mtp_head.h"
#include "rt.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace onebit::npu::mtp;

namespace {

double now_ms() {
    using clk = std::chrono::steady_clock;
    static const auto t0 = clk::now();
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

}  // namespace

// --offline: measure the DRAFTER alone, without needing a rollback for the base's recurrent state.
// Plain greedy decode records every position's hidden state and the token the model chose; then, offline, the
// head chains k drafts from each position and they are compared with the tokens the model actually produced.
// This is the same quantity the published numbers report (acceptance per draft depth, hit rate), and it is the
// only honest protocol available while the engine has no GDN deferral -- with a snapshot-based rollback refused
// by restore_state() (fixed-size conv/rec arrays), a live verify loop would drift the recurrence.
// --offline: measure the DRAFTER alone, without needing a rollback for the base's recurrent state.
//
// Plain greedy decode records every position's hidden state and the token the model chose.  The head is then run
// over the SAME stream in order -- its own KV cache primed position by position, exactly as the MTP layer is
// trained: at position t it attends over the head's own states 0..t, not over itself alone.  From each position
// the chain is continued k times and compared with the tokens the base actually produced.  That reports the
// published quantity (acceptance per draft depth, hit rate) and needs no base-state surgery at all -- which is
// what makes it the honest protocol while the engine has no GDN deferral.
static int argmax_n(const std::vector<float>& v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); i++) if (v[i] > v[best]) best = (int)i;
    return best;
}

// Per-depth acceptance counts: unconditional (d_*: every position, the estimator the published 3.19 uses) and
// conditional on the earlier drafts having been accepted (c_*: what a correctly fed loop actually converts).
struct DepthCounts {
    int d_hit[16] = {0}, d_tot[16] = {0};
    int c_hit[16] = {0}, c_tot[16] = {0};
    void add(const DepthCounts& o) {
        for (int i = 0; i < 16; i++) {
            d_hit[i] += o.d_hit[i];
            d_tot[i] += o.d_tot[i];
            c_hit[i] += o.c_hit[i];
            c_tot[i] += o.c_tot[i];
        }
    }
};

// The offline protocol as one function: prime the head's own cache position by position over a stream (that priming
// is the whole point -- the MTP layer attends over its own states, so it must be fed the stream in order), then
// chain k drafts from each position in `positions` (empty = every position) and score them against the stream.
// `positions` entries are stream indices whose hidden feeds the chain, i.e. the drafts predict position i+1.
// The live loop calls this too, with the positions IT opened chains at, so "live vs offline per depth" is one
// computation sampled two ways rather than two computations that quietly differ.
bool chain_stream(ForwardBase& base, const Weights& w, const std::vector<int>& stream,
                  const std::vector<std::vector<float>>& hidden, int k, const std::vector<int>& positions,
                  DepthCounts& out, double* cos_sum = nullptr, int* cos_n = nullptr,
                  std::vector<int>* drafts = nullptr, std::vector<uint8_t>* hits = nullptr) {
    auto lm = [&base](const std::vector<float>& x, std::vector<float>& lg) { return base.lm_head(x, lg); };
    KvCache hc;                       // the head's own stream, primed position by position
    std::vector<float> emb, next, lg;
    std::vector<KvCache> cache_at(hidden.size());
    for (size_t i = 0; i < hidden.size(); i++) {
        base.embed_into(stream[i], emb);
        if (!head_step(w, hidden[i].data(), emb.data(), (int)i, hc, next,
                       (i + 1 < hidden.size())
                           ? std::function<bool(const std::vector<float>&, std::vector<float>&)>(lm)
                           : nullptr,
                       lg)) {
            std::fprintf(stderr, "head step failed at %zu\n", i);
            return false;
        }
        cache_at[i] = hc;
        if (cos_sum && i + 1 < hidden.size() && !lg.empty()) {
            // diagnostic: the head's hidden should approximate the base's NEXT hidden
            double dot = 0, na = 0, nb = 0;
            for (int q = 0; q < kH; q++) {
                dot += (double)next[(size_t)q] * hidden[i + 1][(size_t)q];
                na += (double)next[(size_t)q] * next[(size_t)q];
                nb += (double)hidden[i + 1][(size_t)q] * hidden[i + 1][(size_t)q];
            }
            *cos_sum += dot / std::sqrt(na * nb + 1e-30);
            (*cos_n)++;
        }
    }
    std::vector<int> which = positions;
    if (which.empty())
        for (size_t i = 0; i + 1 < hidden.size(); i++) which.push_back((int)i);
    if (drafts) drafts->assign(which.size() * (size_t)k, -1);
    if (hits) hits->assign(which.size() * (size_t)k, 0);
    for (size_t n = 0; n < which.size(); n++) {
        const int i0 = which[n];
        if (i0 < 0 || (size_t)i0 + 1 >= hidden.size()) continue;
        const size_t i = (size_t)i0;
        KvCache c = cache_at[i];
        std::vector<float> h = hidden[i], e2, nx;
        int tok = stream[i];
        int pos = i0 + 1;
        // Depth j is scored unconditionally (the chain runs off-distribution after a rejection, exactly like the
        // published table) and conditionally (only while every earlier draft was accepted, which is the ceiling).
        bool prefix_ok = true;
        for (int j = 0; j < k; j++) {
            if (i + j + 1 >= hidden.size()) break;
            base.embed_into(tok, e2);
            if (!head_step(w, h.data(), e2.data(), pos, c, nx, lm, lg)) break;
            const int d = argmax_n(lg);
            const bool hit = (d == stream[i + j + 1]);
            if (drafts) (*drafts)[n * (size_t)k + (size_t)j] = d;
            if (hits) (*hits)[n * (size_t)k + (size_t)j] = hit ? 1 : 0;
            out.d_tot[j]++;
            if (hit) out.d_hit[j]++;
            if (prefix_ok) {
                out.c_tot[j]++;
                if (hit) out.c_hit[j]++;
                if (!hit) prefix_ok = false;
            }
            tok = d;
            h = nx;
            pos++;
        }
    }
    return true;
}

int run_offline(Q4nxNpuForward& fwd, ForwardBase& base, MtpDrafter& drafter, const Weights& w, int prompts,
                int tokens, int k, int prompt_base, bool residual,
                const std::map<int, std::vector<int>>& only = {}) {
    DepthCounts all;
    double cos_sum = 0, rel_sum = 0;
    int cos_n = 0;
    for (int p = 0; p < prompts; p++) {
        const int pi = prompt_base + p;
        std::vector<int> ids;
        for (int i = 0; i < 8; i++) ids.push_back((pi * 977 + i * 131 + 17) % (base.vocab() - 1) + 1);
        fwd.reset();
        std::vector<float> logits;
        std::vector<int> emitted;
        std::vector<std::vector<float>> hid;
        for (size_t i = 0; i < ids.size(); i++) {
            if (!fwd.step(ids[i], (int)i, logits)) { std::fprintf(stderr, "prefill failed\n"); return 1; }
            hid.push_back(residual ? fwd.last_residual() : fwd.last_hidden());
        }
        int pos = (int)ids.size();
        for (int t = 0; t < tokens; t++) {
            int best = 0;
            for (size_t i = 1; i < logits.size(); i++) if (logits[i] > logits[best]) best = (int)i;
            emitted.push_back(best);
            if (!fwd.step(best, pos++, logits)) { std::fprintf(stderr, "step failed\n"); return 1; }
            hid.push_back(residual ? fwd.last_residual() : fwd.last_hidden());
        }
        std::vector<int> stream = ids;
        stream.insert(stream.end(), emitted.begin(), emitted.end());

        // --positions-in: score only the positions the live loop opened chains at.  Live's chain starts at `have`,
        // the position of its first draft, so the equivalent chain here starts on stream index `have-1` (that
        // hidden is what feeds the drafts).  Restricting the SAME computation to those positions is the only way
        // the two per-depth tables measure the same thing; with 8 prompts, the sampled positions are what the
        // 9-18 pp cluster error is.
        std::vector<int> pos_list;
        if (!only.empty()) {
            auto it = only.find(pi);
            if (it != only.end())
                for (int have : it->second) pos_list.push_back(have - 1);
            std::printf("  scored at the live loop's positions: %zu of %zu\n", pos_list.size(), hid.size());
        }
        DepthCounts pp;
        if (!chain_stream(base, w, stream, hid, k, pos_list, pp, &cos_sum, &cos_n)) return 1;
        all.add(pp);
        std::printf("prompt %d: %zu tokens, %zu positions primed\n", pi, hid.size(), hid.size());
        std::printf("  per-depth p%d:", pi);
        for (int j = 0; j < k; j++) std::printf(" %d %d/%d", j + 1, pp.d_hit[j], pp.d_tot[j]);
        std::printf("\n");
    }
    double expected = 1.0;
    std::printf("\n== offline drafter acceptance, %d prompts x %d tokens, k=%d, hidden=%s%s\n", prompts, tokens, k,
                residual ? "residual" : "post-norm",
                only.empty() ? "" : ", only at the live loop's chain positions");
    std::printf("  [diag] cos(head_hidden(t), base_hidden(t+1)) = %.4f over %d positions\n",
                cos_n ? cos_sum / cos_n : 0.0, cos_n);
    for (int j = 0; j < k; j++) {
        const double r = all.d_tot[j] ? (double)all.d_hit[j] / all.d_tot[j] : 0.0;
        std::printf("  draft %d: %6.3f %% (%d/%d)\n", j + 1, 100.0 * r, all.d_hit[j], all.d_tot[j]);
        expected += r;
    }
    // A conditional rate is not a marginal, so the chain is NOT the sum of the rates printed below: by linearity
    // E[tokens per pass] = 1 + sum_j P(draft j accepted) = 1 + sum_j c_hit[j]/c_tot[0] (c_tot[0] = chains scored),
    // because P(accepted at depth j) = P(accepted at j | reached j) * P(reached j).
    double c_expected = 1.0, c_mean_rate = 0.0;
    const int chains = all.c_tot[0];
    std::printf("  conditional on an accepted prefix (the live loop's estimator):\n");
    for (int j = 0; j < k; j++) {
        const double r = all.c_tot[j] ? (double)all.c_hit[j] / all.c_tot[j] : 0.0;
        std::printf("    draft %d: %6.3f %% (%d/%d)\n", j + 1, 100.0 * r, all.c_hit[j], all.c_tot[j]);
        c_mean_rate += r;
        c_expected += chains ? (double)all.c_hit[j] / chains : 0.0;
    }
    std::printf("    mean conditional rate %.2f (rates at different depths are not additive, so this is not a chain)\n",
                c_mean_rate);
    std::printf("    expected tokens per pass (chain) %.2f = 1 + sum_j c_hit[j]/c_tot[0]\n", c_expected);
    std::printf("  hit rate %d/%d = %.3f; expected tokens per pass (chain) %.2f\n", all.d_hit[0], all.d_tot[0],
                all.d_tot[0] ? (double)all.d_hit[0] / all.d_tot[0] : 0.0, expected);
    return 0;
}

int main(int argc, char** argv) {
    // Line-buffered: the engine's own chatter is unbuffered, and a run that is killed mid-init (device hang, hold
    // limit) must still say how far it got -- that is the only evidence a post-mortem has.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string model_dir, head_exact, head_q4k;
    int tokens = 64, k = 4, prompts = 8, prompt_base = 0;
    float tau = 0.6f;
    bool residual = false, dry = false, selftest = false, offline = false, live = false, batchcheck = false;
    int probe_lm = 0;
    bool match_mode = false;
    std::string gold, positions_out, positions_in;
    // prompt index -> the stream positions where the live loop opened its chains
    std::map<int, std::vector<int>> only_positions;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto nexts = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--model") model_dir = nexts();
        else if (a == "--head-exact") head_exact = nexts();
        else if (a == "--head-q4k") head_q4k = nexts();
        else if (a == "--tokens") tokens = std::atoi(nexts().c_str());
        else if (a == "--k") k = std::atoi(nexts().c_str());
        else if (a == "--tau") tau = (float)std::atof(nexts().c_str());
        else if (a == "--prompts") prompts = std::atoi(nexts().c_str());
        else if (a == "--prompt-base") prompt_base = std::atoi(nexts().c_str());
        else if (a == "--residual") residual = true;
        else if (a == "--dry-head") dry = true;
        else if (a == "--selftest-head") selftest = true;
        else if (a == "--gold") gold = nexts();
        else if (a == "--offline") offline = true;
        else if (a == "--live") live = true;
        else if (a == "--batchcheck") batchcheck = true;
        else if (a == "--probe-lm") { const std::string v = (i + 1 < argc) ? argv[++i] : "10"; probe_lm = std::atoi(v.c_str()); }
        else if (a == "--match") match_mode = true;
        else if (a == "--positions-out") positions_out = nexts();
        else if (a == "--positions-in") positions_in = nexts();
        else if (a == "--fc-hidden-first") g_fc_hidden_first = true;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    // --selftest-head: no device.  Runs one head step on a frozen input and compares the hidden output with
    // the Python reference's (tools/mtp_head_ref.py head_step) so the C++ arithmetic is checked before any
    // window opens.
    if (selftest) {
        Weights hw;
        std::string e2;
        if (!load_weights(head_exact, head_q4k, hw, e2)) { std::fprintf(stderr, "head: %s\n", e2.c_str()); return 1; }
        auto rd = [](const std::string& p, std::vector<float>& v, size_t n) {
            std::ifstream f(p, std::ios::binary);
            if (!f) return false;
            v.resize(n);
            f.read((char*)v.data(), (std::streamsize)(n * sizeof(float)));
            return (size_t)f.gcount() == n * sizeof(float);
        };
        std::vector<float> hin, emb, want, got;
        if (!rd(gold + "/hidden.bin", hin, kH) || !rd(gold + "/emb.bin", emb, kH) ||
            !rd(gold + "/out_hidden.bin", want, kH)) {
            std::fprintf(stderr, "cannot read the golden files under %s\n", gold.c_str());
            return 1;
        }
        KvCache cache;
        std::vector<float> logits;
        if (!head_step(hw, hin.data(), emb.data(), 3, cache, got, nullptr, logits)) return 1;
        double worst = 0, num = 0, den = 0, dot = 0, na = 0, nb = 0;
        for (int i = 0; i < kH; i++) {
            const double d = std::fabs((double)got[(size_t)i] - (double)want[(size_t)i]);
            worst = std::max(worst, d);
            num += d;
            den += std::fabs((double)want[(size_t)i]);
            dot += (double)got[(size_t)i] * want[(size_t)i];
            na += (double)got[(size_t)i] * got[(size_t)i];
            nb += (double)want[(size_t)i] * want[(size_t)i];
        }
        std::printf("head selftest vs Python reference: rel-L1 %.3e, max|d| %.3e, cosine %.9f\n", num / den, worst,
                    dot / std::sqrt(na * nb));
        return (num / den < 5e-3 && dot / std::sqrt(na * nb) > 0.9999) ? 0 : 1;
    }
    if (model_dir.empty() || head_exact.empty() || head_q4k.empty()) {
        std::fprintf(stderr, "usage: --model DIR --head-exact DIR --head-q4k DIR [--tokens N] [--k N] [--tau F]\n");
        return 2;
    }
    Weights w;
    std::string err;
    const double t_load0 = now_ms();
    if (!load_weights(head_exact, head_q4k, w, err)) {
        std::fprintf(stderr, "head: %s\n", err.c_str());
        return 1;
    }
    std::printf("head loaded in %.1f s: fc %dx%d, q %dx%d, gate %dx%d\n", (now_ms() - t_load0) / 1000.0, w.fc.K,
                w.fc.N, w.q.K, w.q.N, w.gate.K, w.gate.N);

    xrt::device dev(0);
    Q4nxNpuForward fwd;
    if (!fwd.init(dev, (model_dir + "/model.q4nx").c_str(), model_dir + "/kernels-m16")) {
        std::fprintf(stderr, "forward init failed: %s\n", fwd.last_error());
        return 1;
    }
    ForwardBase base(fwd);
    base.use_residual(residual);
    std::vector<float> probe;
    fwd.reset();
    if (!fwd.step(1, 0, probe)) { std::fprintf(stderr, "probe step failed: %s\n", fwd.last_error()); return 1; }
    base.set_vocab((int)probe.size());
    MtpDrafter drafter(w, [&base](int t, std::vector<float>& e) { base.embed_into(t, e); return true; },
                      [&base](const std::vector<float>& x, std::vector<float>& lg) { return base.lm_head(x, lg); });
    drafter.set_vocab(base.vocab());
    drafter.set_lm_head(!dry);
    drafter.set_tau(tau);          // the chain gate the run was asked for (offline forces its own)
    std::printf("forward ready: vocab %d, max_batch_rows %d, hidden source %s\n", base.vocab(),
                base.max_batch_rows(), residual ? "pre-norm residual" : "post-final-norm");

    // The window's identity, and the LIVE path's only statement of it: `--offline` prints the same line in its summary,
    // but a live log stated k and tau only in the harness' argv echo (or not at all), so the pool tool could not tell a
    // tau=0 ticket from a tau=0.6 one -- and tau is not a detail: at tau>0 the chain gate stops the chain after the
    // first low-confidence draft (npu/mtp/mtp_head.cpp:373), so the deeper depths are never offered.  Printed at init,
    // not in the summary, so a ticket killed at its hold still says what window it was.
    if (live)
        std::printf("== %d prompts x %d tokens, k=%d tau=%.2f, hidden=%s\n", prompts, tokens, k, (double)tau,
                    residual ? "residual" : "post-norm");

    if (probe_lm > 0) {
        // What ONE full-vocab logits call costs on the array, measured rather than fitted.  The draft loop makes
        // k of them for its k drafts (plus 1 + accepted more that the logits-skip fix drops), so this single number
        // decides whether the draft route's cost is the head's maths (host, tools/mtp_head_bench.cpp) or its
        // vocabulary projection (array).  The hidden is the probe step's own, i.e. exactly what the drafter hands
        // the head; a repeat call is the same work every time.
        std::vector<float> h = residual ? fwd.last_residual() : fwd.last_hidden(), lg2;
        std::vector<double> ms;
        for (int r = 0; r < probe_lm; r++) {
            const double t0 = now_ms();
            if (!base.lm_head(h, lg2)) { std::fprintf(stderr, "lm head probe failed\n"); return 1; }
            ms.push_back(now_ms() - t0);
        }
        std::sort(ms.begin(), ms.end());
        std::printf("lm head probe: %d calls over vocab %d, median %.1f ms, min %.1f, max %.1f\n", probe_lm,
                    (int)lg2.size(), ms[ms.size() / 2], ms.front(), ms.back());
    }

    if (batchcheck) {
        // Does the deferred batch verify give the same answer as plain decode?
        //
        // This is the one engine property the draft loop rests on and the only one never checked directly: acceptance
        // compares a draft against the BASE's own choice at that position, and it reads that choice from the M=nd
        // verify batch (row i-1) for draft i, while draft 1 is read from a plain step.  So a batch whose per-row argmax
        // differs from sequential decode would leave depth 1 untouched and depress every deeper depth -- exactly the
        // signature the live loop shows.  Identical TEXT does not check this: the loop only ever emits from the row at
        // the accepted-prefix boundary, and a wrongly REJECTED draft costs a pass, never a token.
        //
        // Pass A decodes greedily, recording the argmax at every position.  Pass B walks the same stream in nd-row
        // batches (committing every row, since all of them are the stream's own tokens) and compares each row's argmax
        // with pass A's.  Cost: one decode plus one pass at 1/nd the step count.
        const int nd = std::min(k, base.max_batch_rows());
        for (int p = 0; p < prompts; p++) {
            const int pi = prompt_base + p;
            std::vector<int> ids;
            for (int i = 0; i < 8; i++) ids.push_back((pi * 977 + i * 131 + 17) % (base.vocab() - 1) + 1);
            fwd.reset();
            std::vector<float> lg;
            std::vector<int> stream = ids, seq;
            std::vector<std::vector<float>> hid_ref;   // pass A's hidden at every position
            for (size_t i = 0; i < ids.size(); i++) {
                if (!fwd.step(ids[i], (int)i, lg)) { std::fprintf(stderr, "prefill failed\n"); return 1; }
                hid_ref.push_back(residual ? fwd.last_residual() : fwd.last_hidden());
            }
            for (int t = 0; t < tokens; t++) {
                int best = 0;
                for (size_t i = 1; i < lg.size(); i++) if (lg[i] > lg[best]) best = (int)i;
                seq.push_back(best);
                stream.push_back(best);
                if (!fwd.step(best, (int)ids.size() + t, lg)) { std::fprintf(stderr, "step failed\n"); return 1; }
                hid_ref.push_back(residual ? fwd.last_residual() : fwd.last_hidden());
            }
            fwd.reset();
            const int n = (int)stream.size();
            int checked = 0, differ = 0, first_differ = -1;
            int hid_rows = 0, worst_pos = -1;
            double min_cos = 2.0, max_abs = 0;
            int pos = 0;
            // prefill the prompt one position at a time, so pass B starts from the same state pass A did
            for (size_t i = 0; i < ids.size(); i++)
                if (!fwd.step(ids[i], (int)i, lg)) { std::fprintf(stderr, "prefill failed\n"); return 1; }
            pos = (int)ids.size();
            std::vector<float> all;
            while (pos + nd <= n) {
                std::vector<int> rows(stream.begin() + pos, stream.begin() + pos + nd);
                if (!base.step_batch_all(rows, pos, all)) { std::fprintf(stderr, "batch failed\n"); return 1; }
                for (int m = 0; m < nd; m++) {
                    int best = 0;
                    const float* l = &all[(size_t)m * (size_t)base.vocab()];
                    for (int i = 1; i < base.vocab(); i++) if (l[i] > l[best]) best = i;
                    // row m's logits predict the token AFTER position pos+m; pass A recorded that as seq[pos+m-7]
                    const int want_at = pos + m - (int)ids.size() + 1;
                    if (want_at < 0 || want_at >= (int)seq.size()) continue;
                    checked++;
                    if (best != seq[(size_t)want_at]) {
                        differ++;
                        if (first_differ < 0) first_differ = pos + m + 1;   // the position row m predicts
                    }
                }
                {
                    // The other half of "is the batch the same as decode": the batch's per-row HIDDEN, which is what
                    // the drafter commits into its stream (last_hidden_rows()[m]).  The same argmax over a slightly
                    // different hidden would still poison every later draft, and only this comparison sees it.
                    const std::vector<float>& rows_h = base.last_hidden_rows();
                    const size_t H = (size_t)base.hidden_size();
                    for (int m = 0; m < nd && (pos + m) < (int)hid_ref.size(); m++) {
                        if (rows_h.size() < (size_t)(m + 1) * H) break;
                        const float* a = &rows_h[(size_t)m * H];
                        const std::vector<float>& b = hid_ref[(size_t)pos + (size_t)m];
                        double dot = 0, na = 0, nb = 0, md = 0;
                        for (size_t q = 0; q < H && q < b.size(); q++) {
                            dot += (double)a[q] * b[q];
                            na += (double)a[q] * a[q];
                            nb += (double)b[q] * b[q];
                            const double d = std::fabs((double)a[q] - b[q]);
                            if (d > md) md = d;
                        }
                        const double cos = dot / std::sqrt(na * nb + 1e-30);
                        hid_rows++;
                        if (cos < min_cos) { min_cos = cos; worst_pos = pos + m; }
                        if (md > max_abs) max_abs = md;
                    }
                }
                if (!base.commit_verify(nd)) { std::fprintf(stderr, "commit refused\n"); return 1; }
                pos += nd;
            }
            std::printf("prompt %d: batch argmax vs sequential argmax: %d of %d rows differ (first at position %d)\n",
                        pi, differ, checked, first_differ);
            std::printf("  batch hidden rows vs sequential hidden: %d rows, min cos %.6f (position %d), max |diff| %.3e\n",
                        hid_rows, min_cos, worst_pos, max_abs);
            // How a verify batch's cost scales with its row count.  The loop's own accounting calls an iteration "two
            // passes" (one step + one M=nd batch), which is only true if a batch costs a step.  The kernels are sealed at
            // M=16, so either a 5-row batch is free (pad rows are idle) or it is linear in rows (the GDN/conv work is per
            // row) -- and that decides whether the draft route's cost is the base or the head.  (P6's "leaf fill", which
            // proposed spending spare rows on 2nd/3rd candidates, cannot help here: acceptance is the base's argmax at
            // that position, so a non-argmax candidate is refused with certainty and a deeper right candidate is never
            // scored under a refused parent -- tested in tests/mtp_draft_test.cpp [16].)
            // the head.  Measured here, on the same stream, at nd = 1, 2, k+1 and max_batch_rows.
            {
                const int widths[] = {1, 2, std::min(k + 1, base.max_batch_rows()), base.max_batch_rows()};
                std::printf("  verify-batch cost, same stream, --tokens %d:\n", tokens);
                std::vector<int> last_width;
                for (int w : widths) {
                    if (w < 1 || w > base.max_batch_rows()) continue;
                    if (!last_width.empty() && last_width.back() == w) continue;
                    last_width.push_back(w);
                    fwd.reset();
                    std::vector<float> lg3;
                    for (size_t i = 0; i < ids.size(); i++)
                        if (!fwd.step(ids[i], (int)i, lg3)) { std::fprintf(stderr, "prefill failed\n"); return 1; }
                    std::vector<float> all3;
                    int p3 = (int)ids.size(), batches = 0;
                    const double t0 = now_ms();
                    while (p3 + w <= n) {
                        std::vector<int> rows(stream.begin() + p3, stream.begin() + p3 + w);
                        if (!base.step_batch_all(rows, p3, all3)) { std::fprintf(stderr, "batch failed\n"); return 1; }
                        if (!base.commit_verify(w)) { std::fprintf(stderr, "commit refused\n"); return 1; }
                        p3 += w;
                        batches++;
                    }
                    const double ms = now_ms() - t0;
                    std::printf("    M %2d rows: %2d batches, %7.2f ms/batch, %5.2f ms/row, %6.1f rows/s\n", w, batches,
                                batches ? ms / batches : 0.0, batches ? ms / (double)(batches * w) : 0.0,
                                ms > 0 ? 1000.0 * (double)(batches * w) / ms : 0.0);
                }
            }
        }
        return 0;
    }

    if (live) {
        // The gate that matters for switching a serve over: the speculative loop must emit exactly the tokens
        // plain greedy decode does, at fewer passes per token.
        std::ofstream pout;
        if (!positions_out.empty()) {
            pout.open(positions_out, std::ios::trunc);
            if (!pout) { std::fprintf(stderr, "cannot write %s\n", positions_out.c_str()); return 1; }
        }
        int bad = 0, tot_emitted = 0, tot_steps = 0, tot_batches = 0, tot_acc = 0, tot_draft = 0, tot_iter = 0;
        int draft_live[Stats::kMaxDepth] = {0};
        LiveDepths live_tot;      // hit[] + reached[] for every prompt so far, from the loop's own counters
        double hit_live[Stats::kMaxDepth] = {0};
        double tot_plain_s = 0, tot_spec_s = 0;
        // Phase split totals (MTP_PHASE_SPLIT=1): summed across prompts so a multi-prompt ticket reports one table.
        double tot_ms_step = 0, tot_ms_batch = 0, tot_ms_draft = 0, tot_ms_commit = 0;
        const bool phase_split = std::getenv("MTP_PHASE_SPLIT") != nullptr;
        DepthCounts matched;      // --match: the offline protocol scored where THIS run opened its chains
        for (int p = 0; p < prompts; p++) {
            const int pi = prompt_base + p;
            std::vector<int> ids;
            for (int i = 0; i < 8; i++) ids.push_back((pi * 977 + i * 131 + 17) % (base.vocab() - 1) + 1);
            // reference: plain greedy
            fwd.reset();
            std::vector<float> lg;
            std::vector<int> plain;
            std::vector<std::vector<float>> hid_plain;   // --match: the stream the offline protocol is scored on
            const double t_plain0 = now_ms();
            for (size_t i = 0; i < ids.size(); i++) {
                if (!fwd.step(ids[i], (int)i, lg)) { std::fprintf(stderr, "prefill failed\n"); return 1; }
                if (match_mode) hid_plain.push_back(residual ? fwd.last_residual() : fwd.last_hidden());
            }
            int pos = (int)ids.size();
            for (int t = 0; t < tokens; t++) {
                int best = 0;
                for (size_t i = 1; i < lg.size(); i++) if (lg[i] > lg[best]) best = (int)i;
                plain.push_back(best);
                if (!fwd.step(best, pos++, lg)) { std::fprintf(stderr, "step failed\n"); return 1; }
                if (match_mode) hid_plain.push_back(residual ? fwd.last_residual() : fwd.last_hidden());
            }
            const double plain_s = (now_ms() - t_plain0) / 1000.0;
            // the speculative loop on the same prompt: reset the base, then prime the drafter over the prompt so it
            // attends over the whole stream (that is what the offline protocol showed matters: 3 % -> 69 %)
            fwd.reset();
            drafter.reset();
            {
                // Prime the drafter over every prompt position (its KV must cover the stream), then hand the base
                // back at position 0 -- the loop prefills itself.  prime() advances the drafter's cursor, so from
                // here on the chain the loop opens must start where the committed stream ends.
                std::vector<float> lg2;
                for (size_t i = 0; i < ids.size(); i++) {
                    if (!fwd.step(ids[i], (int)i, lg2)) { std::fprintf(stderr, "prime prefill failed\n"); return 1; }
                    if (!drafter.prime(fwd.last_hidden(), ids[i], (int)i)) {
                        std::fprintf(stderr, "prompt %d: drafter prime refused at %zu (committed %d)\n", p, i,
                                     drafter.stream().keep());
                        return 1;
                    }
                }
                fwd.reset();
            }
            Options o;
            o.k = k;
            o.tau = tau;
            o.max_tokens = tokens;
            const double t_spec0 = now_ms();
            const long lm0 = drafter.lm_calls();
            const Result r = run(base, &drafter, ids, o);
            const long lm_used = drafter.lm_calls() - lm0;
            const double spec_s = (now_ms() - t_spec0) / 1000.0;
            if (!r.ok) { std::fprintf(stderr, "prompt %d: %s\n", p, r.err.c_str()); return 1; }
            const bool same = (r.tokens == plain);
            if (!same) bad++;
            tot_emitted += r.stats.emitted;
            tot_steps += r.stats.steps;
            tot_batches += r.stats.batches;
            tot_acc += r.stats.accepted;
            tot_draft += r.stats.drafted;
            tot_iter += r.stats.iterations;
            add_live_depths(r.stats, live_tot);
            for (int d = 0; d < Stats::kMaxDepth; d++) {
                draft_live[d] += r.stats.drafted_at[d];
                hit_live[d] = draft_live[d] ? (double)live_tot.hit[d] / draft_live[d] : 0.0;
            }
            // The two numbers the serve switch needs, measured in ONE process on the SAME prompt: the draft route
            // against plain decode.  "tokens/pass (chain)" is 1 + accepted/iterations -- the quantity the offline
            // protocol prints; it is NOT accepted/iterations alone (that is accept-len, one lower by the committed
            // token), which is the factor-(1+x) trap that made the live and offline summaries look incomparable.
            // `pi`, not `p`: with --prompt-base the slice index is not the prompt index, and a log that says
            // "prompt 0" for prompt 5 is how two tickets' tables get compared with the wrong slice.
            std::printf("prompt %d: %s, %d tokens, accepted %d/%d, passes/token %.3f, stream %d rows, %d mismatch\n", pi,
                        same ? "TEXT IDENTICAL" : "TEXT DIVERGED", r.stats.emitted, r.stats.accepted,
                        r.stats.drafted, r.stats.passes_per_token(), drafter.stream().keep(),
                        drafter.stream_mismatches());
            std::printf("  chain %.2f tokens/pass (accept-len %.2f + the committed token); plain %.2f s = %.3f s/tok, "
                        "draft %.2f s = %.3f s/tok (speedup %.2fx)\n",
                        1.0 + r.stats.mean_accept_len(), r.stats.mean_accept_len(), plain_s,
                        tokens ? plain_s / tokens : 0.0, spec_s, tokens ? spec_s / tokens : 0.0,
                        spec_s > 0 ? plain_s / spec_s : 0.0);
            tot_plain_s += plain_s;
            tot_spec_s += spec_s;
            // Per-PROMPT depth counts, not just the run aggregate: the array is shared, so the 8-prompt table is
            // assembled from several tickets/tickets-of-slices, and a per-prompt breakdown is the only form that can
            // be summed correctly afterwards (adding percentages weights a 16-iteration prompt like a 36 one).
            // Calls, not a claim: a full-vocab logits call is ~0.3-0.5 s of array dispatch chain, so this count is
            // the loop's real cost per iteration and the thing a cheaper head has to reduce.
            // The millisecond figure divides the WHOLE draft phase (base steps, verify batches, priming) by the
            // logits count, so it is an amortised phase cost, not a head latency -- one call alone is ~0.3-0.5 s.
            std::printf("  logits calls %ld over %d iterations (%.2f/iter), %.1f ms of the whole draft phase per call\n", lm_used,
                        r.stats.iterations, r.stats.iterations ? (double)lm_used / r.stats.iterations : 0.0,
                        lm_used ? 1000.0 * spec_s / (double)lm_used : 0.0);
            // Where the draft phase's wall clock goes, all four terms measured in this same run.  A loop change is
            // priced on this split, not on a subtraction across windows: "two passes per iteration" is only the
            // reason the route is slow if step + batch own the milliseconds, and after #968 they do not obviously.
            tot_ms_step += r.stats.ms_step;
            tot_ms_batch += r.stats.ms_batch;
            tot_ms_draft += r.stats.ms_draft;
            tot_ms_commit += r.stats.ms_commit;
            if (phase_split) {
                const double tot = r.stats.phase_ms();
                const double pc = tot > 0 ? 100.0 / tot : 0.0;
                std::printf("  phase split: step %8.1f ms (%4.1f%%) + batch %8.1f ms (%4.1f%%) + head %8.1f ms (%4.1f%%) "
                            "+ commit %6.1f ms (%4.1f%%) = %8.1f ms over %d iterations (%.0f ms/iter)\n",
                            r.stats.ms_step, r.stats.ms_step * pc, r.stats.ms_batch, r.stats.ms_batch * pc,
                            r.stats.ms_draft, r.stats.ms_draft * pc, r.stats.ms_commit, r.stats.ms_commit * pc,
                            tot, r.stats.iterations,
                            r.stats.iterations ? tot / r.stats.iterations : 0.0);
            }
            std::printf("  per-depth p%d:", pi);
            for (int d = 0; d < k && d < Stats::kMaxDepth; d++)
                std::printf(" %d %d/%d", d + 1, r.stats.accepted_at[d], r.stats.drafted_at[d]);
            std::printf("\n");
            // --positions-out records where each prompt's chains opened, so `--offline --positions-in` scores the
            // same positions and the two per-depth tables stop being a clustered subsample of each other.
            // An ofstream that was never opened still tests *true* (a default-constructed one is in the good state,
            // no buffer and no failbit), so `if (pout)` printed "written to " with an empty path whenever --match ran
            // without --positions-out.  Test the flag that decides whether the file exists, and never leave a short
            // write behind: a truncated positions file would be scored silently by a later `--offline --positions-in`.
            if (!positions_out.empty()) {
                pout << pi;
                for (int have : r.stats.chain_at) pout << ' ' << have;
                pout << '\n';
                pout.flush();
                if (!pout) {
                    std::fprintf(stderr, "write failed: %s\n", positions_out.c_str());
                    return 1;
                }
                std::printf("  chain positions: %zu written to %s\n", r.stats.chain_at.size(),
                            positions_out.c_str());
            }
            // --match: the deliverable's actual clause.  Score the OFFLINE protocol -- same function, same weights,
            // same primed stream -- at exactly the positions this live loop opened chains at, in this process, and
            // print live vs offline per depth.  Anything left over is then neither prompt composition nor a
            // subsample: it is the loop's feeding or the verify batch, both already proven exact digit for digit.
            if (match_mode) {
                if (!same)
                    std::printf("  match p%d: TEXT DIVERGED, so this comparison is against a different stream\n", pi);
                std::vector<int> stream = ids;
                stream.insert(stream.end(), plain.begin(), plain.end());
                std::vector<int> pos_list;
                for (int have : r.stats.chain_at) pos_list.push_back(have - 1);
                DepthCounts m;
                std::vector<int> odrafts;
                std::vector<uint8_t> ohits;
                if (!chain_stream(base, w, stream, hid_plain, k, pos_list, m, nullptr, nullptr, &odrafts, &ohits))
                    return 1;
                matched.add(m);
                std::printf("  match p%d: %zu offline chains at this loop's own positions (stream %zu rows)\n", pi,
                            pos_list.size(), hid_plain.size());
                // Three estimators live in this file and mixing them has already cost this lane two false alarms
                // (13:40, 15:22), so name them here once:
                //   marginal      P(draft d is the model's choice) at every position  -- the published 3.19 (d_*);
                //   conditional   P(draft d | the prefix was accepted)               -- offline c_*, the drafter's
                //                                                                     quality per depth;
                //   joint         P(the whole chain 0..d was accepted)              -- the loop's accepted_at[d]/
                //                                                                     drafted_at[d], which is a
                //                                                                     survival curve, not a rate.
                // The loop's per-depth acceptance is CONDITIONAL on reaching the depth (that is what a serve wants:
                // how often does draft d get taken when the chain got there), so compare it with offline's c_*.
                LiveDepths live;
                add_live_depths(r.stats, live);
                for (int d = 0; d < k && d < Stats::kMaxDepth; d++) {
                    const double joint = r.stats.drafted_at[d] ? 100.0 * r.stats.accepted_at[d] / r.stats.drafted_at[d] : 0.0;
                    const int hits = live.hit[d], reached = live.reached[d];
                    const DepthVerdict v = depth_verdict(hits, reached, m.c_hit[d], m.c_tot[d]);
                    const double om = m.d_tot[d] ? 100.0 * m.d_hit[d] / m.d_tot[d] : 0.0;
                    if (!v.scored) {   // no chain on one side: 0/0 is not agreement (see depth_verdict)
                        std::printf("    depth %d: n/a (live %d/%d reached, offline %d/%d)  not scored"
                                    "   [live joint %6.2f %%, offline marginal %6.2f %%]\n",
                                    d + 1, hits, reached, m.c_hit[d], m.c_tot[d], joint, om);
                        continue;
                    }
                    std::printf("    depth %d: live %6.2f %% (%d/%d, reached)  offline conditional %6.2f %% (%d/%d)"
                                "  delta %+6.2f pp%s   [live joint %6.2f %%, offline marginal %6.2f %%]\n",
                                d + 1, v.live, hits, reached, v.offline, m.c_hit[d], m.c_tot[d],
                                v.delta, v.within ? "" : "  > 2 pp", joint, om);
                }
                // ...and the form that needs no estimator at all: per chain position, how many drafts were accepted
                // before the first refusal.  Identical machinery must give identical prefix lengths at every
                // position, and a difference names its position and both lengths.
                int pref_same = 0, pref_diff = 0, ppos = -1, plive = -1, poff = -1;
                for (size_t n = 0; n < r.stats.chain_at.size() && n < r.stats.chain_accepted.size(); n++) {
                    const int i = r.stats.chain_at[n] - 1;
                    const int cap = std::min<int>(k, (int)stream.size() - 1 - i);
                    if (cap <= 0) continue;
                    int run = 0;
                    for (int d = 0; d < cap; d++) {
                        const size_t idx = n * (size_t)k + (size_t)d;
                        if (idx >= ohits.size() || ohits[idx] == 0) break;
                        run++;
                    }
                    const int lv = std::min<int>(r.stats.chain_accepted[n], cap);
                    if (lv == run) pref_same++;
                    else { pref_diff++; if (ppos < 0) { ppos = r.stats.chain_at[n]; plive = lv; poff = run; } }
                }
                std::printf("    accepted-prefix length equal at %d of %d positions", pref_same, pref_same + pref_diff);
                if (ppos >= 0) std::printf(", first difference at stream position %d: live %d, offline %d", ppos,
                                           plive, poff);
                std::printf("\n");
                // Per-(position, depth) diff.  The aggregates can only say "less"; the drafts say WHICH:
                //   * same draft token, different verdict -> the verification/feeding path disagrees with the
                //     protocol on the same computation (a real loop defect, and here is its position);
                //   * different draft token at depths 1-4 -> the chain itself was computed from a different state,
                //     which points at the drafter's cache rows, not at the verdict.
                // Only pairs both paths can score are counted, because offline stops at the stream end and the
                // live loop keeps drafting (tau=0 always drafts k).  That tail alone is a few pairs, not a trend.
                int pairs = 0, tok_diff = 0, verdict_diff = 0, verdict_diff_same_tok = 0, live_hit = 0, off_hit = 0;
                int fpos = -1, fdep = -1, flive = -1, foff = -1, fstr = -1, vpos = -1, vdep = -1, vlive = -1, voff = -1;
                for (size_t n = 0; n < r.stats.chain_at.size() && n < r.stats.chain_drafts.size(); n++) {
                    const int i = r.stats.chain_at[n] - 1;
                    for (int d = 0; d < k && d < Stats::kMaxDepth; d++) {
                        const size_t idx = n * (size_t)k + (size_t)d;
                        if (i < 0 || (size_t)(i + d + 1) >= stream.size() || idx >= odrafts.size()) continue;
                        const int od = odrafts[idx];
                        if (od < 0 || d >= (int)r.stats.chain_drafts[n].size()) continue;   // offline/loop stopped here
                        const int ld = r.stats.chain_drafts[n][(size_t)d];
                        const bool lh = d < r.stats.chain_accepted[n];
                        const bool oh = ohits[idx] != 0;
                        pairs++;
                        if (lh) live_hit++;
                        if (oh) off_hit++;
                        if (ld != od && fpos < 0) { fpos = i + 1; fdep = d + 1; flive = ld; foff = od; fstr = stream[(size_t)(i + d + 1)]; }
                        if (ld != od) tok_diff++;
                        if (lh != oh) {
                            verdict_diff++;
                            if (ld == od) {
                                verdict_diff_same_tok++;
                                if (vpos < 0) { vpos = i + 1; vdep = d + 1; vlive = lh; voff = oh; }
                            }
                        }
                    }
                }
                std::printf("    pairs %d: accepted live %d, offline %d (marginal); draft tokens differ %d; verdicts differ %d"
                            " (%d of them on the SAME draft token)\n",
                            pairs, live_hit, off_hit, tok_diff, verdict_diff, verdict_diff_same_tok);
                if (fpos >= 0)
                    std::printf("    first draft-token difference: stream position %d, depth %d: live %d, offline %d, "
                                "stream says %d\n", fpos, fdep, flive, foff, fstr);
                if (vpos >= 0)
                    std::printf("    first verdict difference on the same token: stream position %d, depth %d: live %s, "
                                "offline %s\n", vpos, vdep, vlive ? "accept" : "reject", voff ? "accept" : "reject");
            }
            if (!same) {
                std::printf("  plain:");
                for (int t : plain) std::printf(" %d", t);
                std::printf("\n  spec :");
                for (int t : r.tokens) std::printf(" %d", t);
                std::printf("\n");
            }
        }
        std::printf("\n== live loop: %d/%d prompts text-identical, passes/token %.3f, accept-len %.2f, hit %.3f\n",
                    prompts - bad, prompts, tot_emitted ? (double)(tot_steps + tot_batches) / tot_emitted : 0.0,
                    tot_iter ? (double)tot_acc / tot_iter : 0.0, tot_draft ? (double)tot_acc / tot_draft : 0.0);
        std::printf("   chain %.2f tokens/pass (= 1 + accept-len, the offline quantity; %.3f tokens/base-pass); "
                    "wall plain %.1f s, draft %.1f s (speedup %.2fx)\n",
                    tot_iter ? 1.0 + (double)tot_acc / tot_iter : 0.0,
                    tot_emitted ? (double)tot_emitted / (tot_steps + tot_batches) : 0.0, tot_plain_s, tot_spec_s,
                    tot_spec_s > 0 ? tot_plain_s / tot_spec_s : 0.0);
        // The whole run's split, so the serve switch and any loop change are priced on one window's measurement.
        if (phase_split && tot_ms_step + tot_ms_batch + tot_ms_draft + tot_ms_commit > 0) {
            const double tot = tot_ms_step + tot_ms_batch + tot_ms_draft + tot_ms_commit;
            const double pc = 100.0 / tot;
            std::printf("   phase split (all prompts): step %.1f%% + batch %.1f%% + head %.1f%% + commit %.1f%%"
                        "  = %.1f s of draft phase; of that, base passes %.1f%%\n",
                        tot_ms_step * pc, tot_ms_batch * pc, tot_ms_draft * pc, tot_ms_commit * pc, tot / 1000.0,
                        (tot_ms_step + tot_ms_batch) * pc);
        }
        // The comparison the deliverable asks for: the same quantity the offline protocol prints, per depth.
        std::printf("   per-depth acceptance:\n");
        for (int d = 0; d < k && d < Stats::kMaxDepth; d++)
            std::printf("     draft %d: %6.3f %% (%d/%d)\n", d + 1, 100.0 * hit_live[d], live_tot.hit[d], draft_live[d]);
        if (match_mode) {
            // The brief's clause, measured instead of argued: the same drafter on the same stream at the same
            // positions, so the only difference left is how the live loop feeds it.
            int over = 0, scored = 0;
            std::printf("   == live vs offline at the SAME positions, per-depth acceptance CONDITIONAL on reaching"
                        " the depth (the 2 pp bar):\n");
            for (int d = 0; d < k && d < Stats::kMaxDepth; d++) {
                const int hits = live_tot.hit[d], reached = live_tot.reached[d];
                const DepthVerdict v = depth_verdict(hits, reached, matched.c_hit[d], matched.c_tot[d]);
                if (!v.scored) {   // 0/0 used to read as "delta +0.00 pp PASS" and inflate the bar
                    std::printf("     depth %d: n/a (live %d/%d reached, offline %d/%d)  not scored\n", d + 1,
                                hits, reached, matched.c_hit[d], matched.c_tot[d]);
                    continue;
                }
                scored++;
                if (!v.within) over++;
                std::printf("     depth %d: live %6.2f %% (%d/%d)  offline %6.2f %% (%d/%d)  delta %+6.2f pp  %s\n",
                            d + 1, v.live, hits, reached, v.offline, matched.c_hit[d], matched.c_tot[d], v.delta,
                            v.within ? "PASS" : "FAIL");
            }
            const int depths = std::min(k, Stats::kMaxDepth);
            std::printf("   %d of %d scored depths within 2 pp at the same positions%s\n", scored - over, scored,
                        scored < depths ? " (a depth with no chain on either side is n/a, never a pass)" : "");
        }
        return bad ? 1 : 0;
    }

    if (offline) {
        drafter.set_tau(tau);      // tau=0 measures every depth unconditionally, which is what the offline
                                   // protocol reports (the chain gate is a speed choice, not a quality one)
        if (!positions_in.empty()) {
            std::ifstream in(positions_in);
            if (!in) { std::fprintf(stderr, "cannot read %s\n", positions_in.c_str()); return 1; }
            std::string line;
            while (std::getline(in, line)) {
                std::istringstream ls(line);
                int pi;
                if (!(ls >> pi)) continue;
                int have;
                while (ls >> have) only_positions[pi].push_back(have);
            }
            if (only_positions.empty()) {
                std::fprintf(stderr, "%s holds no positions\n", positions_in.c_str());
                return 1;
            }
            std::printf("scoring only the live loop's chain positions from %s (%zu prompts)\n", positions_in.c_str(),
                        only_positions.size());
        }
        return run_offline(fwd, base, drafter, w, prompts, tokens, k, prompt_base, residual, only_positions);
    }

    // Prompts are token ids: acceptance does not depend on the text, and this keeps the run reproducible
    // without dragging the tokenizer into a measurement harness.
    int total_emitted = 0, total_drafted = 0, total_accepted = 0, total_steps = 0, total_batches = 0;
    int total_iters = 0, first_accept = 0;
    const double t0 = now_ms();
    for (int p = 0; p < prompts; p++) {
        Options o;
        o.k = k;
        o.tau = tau;
        o.max_tokens = tokens;
        std::vector<int> prompt_ids;
        const int pi = prompt_base + p;      // tickets run a slice of the 8 prompts
        for (int i = 0; i < 8; i++) prompt_ids.push_back((pi * 977 + i * 131 + 17) % (base.vocab() - 1) + 1);
        fwd.reset();
        drafter.reset();
        const Result r = run(base, &drafter, prompt_ids, o);
        if (!r.ok) { std::fprintf(stderr, "prompt %d: %s\n", p, r.err.c_str()); return 1; }
        total_emitted += r.stats.emitted;
        total_drafted += r.stats.drafted;
        total_accepted += r.stats.accepted;
        total_steps += r.stats.steps;
        total_batches += r.stats.batches;
        total_iters += r.stats.iterations;
        first_accept += r.stats.iterations - r.stats.first_reject;
        std::printf("prompt %d: %d tokens, %d iters, accepted %d/%d, accept-len %.2f, passes/token %.3f\n", p,
                    r.stats.emitted, r.stats.iterations, r.stats.accepted, r.stats.drafted,
                    r.stats.mean_accept_len(), r.stats.passes_per_token());
    }
    const double wall = (now_ms() - t0) / 1000.0;
    std::printf("\n== %d prompts x %d tokens, k=%d tau=%.2f, hidden=%s\n", prompts, tokens, k, (double)tau,
                residual ? "residual" : "post-norm");
    std::printf("emitted %d in %d iterations; steps %d + batches %d = %d passes (%.3f passes/token)\n",
                total_emitted, total_iters, total_steps, total_batches, total_steps + total_batches,
                total_emitted ? (double)(total_steps + total_batches) / total_emitted : 0.0);
    std::printf("acceptance length %.2f tokens/pass; hit rate %.3f (%d/%d drafts); first-draft hit %.3f\n",
                total_iters ? (double)total_accepted / total_iters : 0.0,
                total_drafted ? (double)total_accepted / total_drafted : 0.0, total_accepted, total_drafted,
                total_iters ? (double)first_accept / total_iters : 0.0);
    std::printf("wall %.1f s = %.3f s/token (base %.3f s/token equivalent)\n", wall,
                total_emitted ? wall / total_emitted : 0.0, total_emitted ? wall / total_emitted : 0.0);
    return 0;
}
