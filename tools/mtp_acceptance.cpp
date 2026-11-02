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

#include <chrono>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

int run_offline(Q4nxNpuForward& fwd, ForwardBase& base, MtpDrafter& drafter, const Weights& w, int prompts,
                int tokens, int k, int prompt_base, bool residual) {
    int d_hit[16] = {0}, d_tot[16] = {0};
    double cos_sum = 0, rel_sum = 0;
    int cos_n = 0;
    std::vector<int> all_tokens;        // the token stream: prompt + emitted
    std::vector<std::vector<float>> hidden;   // hidden[i] = base hidden after all_tokens[i]
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
        all_tokens = stream;
        hidden = hid;

        auto lm = [&base](const std::vector<float>& x, std::vector<float>& lg) { return base.lm_head(x, lg); };
        KvCache hc;                       // the head's own stream, primed position by position
        std::vector<float> emb, next, lg;
        std::vector<KvCache> cache_at(hidden.size());
        for (size_t i = 0; i < hidden.size(); i++) {
            base.embed_into(all_tokens[i], emb);
            if (!head_step(w, hidden[i].data(), emb.data(), (int)i, hc, next,
                           (i + 1 < hidden.size()) ? std::function<bool(const std::vector<float>&,
                                                                        std::vector<float>&)>(lm)
                                                   : nullptr,
                           lg)) {
                std::fprintf(stderr, "head step failed at %zu\n", i);
                return 1;
            }
            cache_at[i] = hc;
            if (i + 1 < hidden.size() && !lg.empty()) {
                // diagnostic: the head's hidden should approximate the base's NEXT hidden
                double dot = 0, na = 0, nb = 0;
                for (int q = 0; q < kH; q++) {
                    dot += (double)next[(size_t)q] * hidden[i + 1][(size_t)q];
                    na += (double)next[(size_t)q] * next[(size_t)q];
                    nb += (double)hidden[i + 1][(size_t)q] * hidden[i + 1][(size_t)q];
                }
                cos_sum += dot / std::sqrt(na * nb + 1e-30);
                cos_n++;
            }
        }
        // from each position, chain k drafts off the primed cache and compare with the real continuation
        for (size_t i = 0; i + 1 < hidden.size(); i++) {
            KvCache c = cache_at[i];
            std::vector<float> h = hidden[i], e2, nx;
            int tok = all_tokens[i];
            int pos = (int)i + 1;
            for (int j = 0; j < k; j++) {
                if (i + j + 1 >= hidden.size()) break;
                base.embed_into(tok, e2);
                if (!head_step(w, h.data(), e2.data(), pos, c, nx, lm, lg)) break;
                int d = argmax_n(lg);
                d_tot[j]++;
                if (d == all_tokens[i + j + 1]) d_hit[j]++;
                tok = d;
                h = nx;
                pos++;
            }
        }
        std::printf("prompt %d: %zu tokens, %zu positions primed\n", p, hidden.size(), hidden.size());
    }
    double expected = 1.0;
    std::printf("\n== offline drafter acceptance, %d prompts x %d tokens, k=%d, hidden=%s\n", prompts, tokens, k,
                residual ? "residual" : "post-norm");
    std::printf("  [diag] cos(head_hidden(t), base_hidden(t+1)) = %.4f over %d positions\n",
                cos_n ? cos_sum / cos_n : 0.0, cos_n);
    for (int j = 0; j < k; j++) {
        const double r = d_tot[j] ? (double)d_hit[j] / d_tot[j] : 0.0;
        std::printf("  draft %d: %6.3f %% (%d/%d)\n", j + 1, 100.0 * r, d_hit[j], d_tot[j]);
        expected += r;
    }
    std::printf("  hit rate %d/%d = %.3f; expected tokens per pass (chain) %.2f\n", d_hit[0], d_tot[0],
                d_tot[0] ? (double)d_hit[0] / d_tot[0] : 0.0, expected);
    return 0;
}

int main(int argc, char** argv) {
    // Line-buffered: the engine's own chatter is unbuffered, and a run that is killed mid-init (device hang, hold
    // limit) must still say how far it got -- that is the only evidence a post-mortem has.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string model_dir, head_exact, head_q4k;
    int tokens = 64, k = 4, prompts = 8, prompt_base = 0;
    float tau = 0.6f;
    bool residual = false, dry = false, selftest = false, offline = false, live = false;
    std::string gold;
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

    if (live) {
        // The gate that matters for switching a serve over: the speculative loop must emit exactly the tokens
        // plain greedy decode does, at fewer passes per token.
        int bad = 0, tot_emitted = 0, tot_steps = 0, tot_batches = 0, tot_acc = 0, tot_draft = 0, tot_iter = 0;
        int acc_live[Stats::kMaxDepth] = {0}, draft_live[Stats::kMaxDepth] = {0};
        double hit_live[Stats::kMaxDepth] = {0};
        for (int p = 0; p < prompts; p++) {
            const int pi = prompt_base + p;
            std::vector<int> ids;
            for (int i = 0; i < 8; i++) ids.push_back((pi * 977 + i * 131 + 17) % (base.vocab() - 1) + 1);
            // reference: plain greedy
            fwd.reset();
            std::vector<float> lg;
            std::vector<int> plain;
            for (size_t i = 0; i < ids.size(); i++)
                if (!fwd.step(ids[i], (int)i, lg)) { std::fprintf(stderr, "prefill failed\n"); return 1; }
            int pos = (int)ids.size();
            for (int t = 0; t < tokens; t++) {
                int best = 0;
                for (size_t i = 1; i < lg.size(); i++) if (lg[i] > lg[best]) best = (int)i;
                plain.push_back(best);
                if (!fwd.step(best, pos++, lg)) { std::fprintf(stderr, "step failed\n"); return 1; }
            }
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
            const Result r = run(base, &drafter, ids, o);
            if (!r.ok) { std::fprintf(stderr, "prompt %d: %s\n", p, r.err.c_str()); return 1; }
            const bool same = (r.tokens == plain);
            if (!same) bad++;
            tot_emitted += r.stats.emitted;
            tot_steps += r.stats.steps;
            tot_batches += r.stats.batches;
            tot_acc += r.stats.accepted;
            tot_draft += r.stats.drafted;
            tot_iter += r.stats.iterations;
            for (int d = 0; d < Stats::kMaxDepth; d++) {
                acc_live[d] += r.stats.accepted_at[d];
                draft_live[d] += r.stats.drafted_at[d];
                hit_live[d] = draft_live[d] ? (double)acc_live[d] / draft_live[d] : 0.0;
            }
            std::printf("prompt %d: %s, %d tokens, accepted %d/%d, passes/token %.3f, stream %d rows, %d mismatch\n", p,
                        same ? "TEXT IDENTICAL" : "TEXT DIVERGED", r.stats.emitted, r.stats.accepted,
                        r.stats.drafted, r.stats.passes_per_token(), drafter.stream().keep(),
                        drafter.stream_mismatches());
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
        // The comparison the deliverable asks for: the same quantity the offline protocol prints, per depth.
        std::printf("   per-depth acceptance:\n");
        for (int d = 0; d < k && d < Stats::kMaxDepth; d++)
            std::printf("     draft %d: %6.3f %% (%d/%d)\n", d + 1, 100.0 * hit_live[d], acc_live[d], draft_live[d]);
        return bad ? 1 : 0;
    }

    if (offline) {
        drafter.set_tau(tau);      // tau=0 measures every depth unconditionally, which is what the offline
                                   // protocol reports (the chain gate is a speed choice, not a quality one)
        return run_offline(fwd, base, drafter, w, prompts, tokens, k, prompt_base, residual);
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
