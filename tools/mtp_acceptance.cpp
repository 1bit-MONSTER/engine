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

// argmax + the max probability under a softmax over the whole vocabulary (the tau gate needs the probability,
// not just the token).
int argmax_p(const std::vector<float>& logits, float* prob) {
    int best = 0;
    for (size_t i = 1; i < logits.size(); i++)
        if (logits[i] > logits[best]) best = (int)i;
    if (prob) {
        double mx = logits[(size_t)best];
        double sum = 0;
        for (float v : logits) sum += std::exp((double)v - mx);
        *prob = (float)(1.0 / sum);
    }
    return best;
}

// The MTP drafter: one head step per proposed token, chained on its own hidden state and KV cache.
class MtpDrafter : public DraftHead {
public:
    MtpDrafter(const Weights& w, ForwardBase& base) : w_(w), base_(base) {}

    void reset() override { cache_.reset(); }

    int draft(const std::vector<float>& hidden, int last_token, int pos, int k, std::vector<int>& out,
              std::vector<float>* conf) override {
        out.clear();
        if (conf) conf->clear();
        std::vector<float> emb, h = hidden, logits, next;
        cache_.reset();                       // the chain starts from the accepted token each iteration
        pos_ = pos;
        int tok = last_token;
        for (int i = 0; i < k; i++) {
            base_.embed_into(tok, emb);
            if (!head_step(w_, h.data(), emb.data(), pos_, cache_, next,
                           [this](const std::vector<float>& x, std::vector<float>& lg) {
                               return base_lm_head(x, lg);
                           },
                           logits)) {
                break;
            }
            float p = 1.0f;
            const int t = argmax_p(logits, conf ? &p : nullptr);
            out.push_back(t);
            if (conf) conf->push_back(p);
            if (p < tau_) break;              // stop the chain here; the loop still verifies this draft
            tok = t;
            h = next;
            pos_++;
        }
        return (int)out.size();
    }

    void set_lm_head(bool on) { use_lm_head_ = on; }

    double head_ms = 0;

private:
    bool base_lm_head(const std::vector<float>& x, std::vector<float>& lg) {
        if (!use_lm_head_) { lg.assign(base_.vocab(), 0.0f); return true; }
        return base_.lm_head(x, lg);
    }
    const Weights& w_;
    ForwardBase& base_;
    KvCache cache_;
    int pos_ = 0;
    bool use_lm_head_ = true;
    float tau_ = 0.6f;
};

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir, head_exact, head_q4k;
    int tokens = 64, k = 4, prompts = 8;
    float tau = 0.6f;
    bool residual = false, dry = false, selftest = false;
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
        else if (a == "--residual") residual = true;
        else if (a == "--dry-head") dry = true;
        else if (a == "--selftest-head") selftest = true;
        else if (a == "--gold") gold = nexts();
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
    MtpDrafter drafter(w, base);
    drafter.set_lm_head(!dry);
    std::printf("forward ready: vocab %d, max_batch_rows %d, hidden source %s\n", base.vocab(),
                base.max_batch_rows(), residual ? "pre-norm residual" : "post-final-norm");

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
        for (int i = 0; i < 8; i++) prompt_ids.push_back((p * 977 + i * 131 + 17) % (base.vocab() - 1) + 1);
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
