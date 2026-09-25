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
// npu_spec_bench: speculative decoding with the drafter on the NPU and the target on the GPU.
//
// The NPU fast lane (Qwen3-0.6B) proposes k tokens; the target (any GGUF with Qwen3's
// tokenizer, through libllama on Vulkan) scores them in one batch; the longest prefix the
// target agrees with is kept, plus the target's own next token, and both KV caches roll back
// to it. Both sides decode greedily, so the output must equal plain greedy decoding of the
// target token for token: the bench runs both and checks that, then reports speed and how many
// drafts were accepted.
//
// usage: npu_spec_bench <target.gguf> <npu model dir> <npu kernel dir> <k> <tokens> "<prompt>"
#include "lane.h"
#include "model.h"

#include <llama.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

struct Target {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    int n_vocab = 0;

    Target(const std::string& path, int n_ctx) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 999;
        model = llama_model_load_from_file(path.c_str(), mp);
        if (!model) throw std::runtime_error("cannot load " + path);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx;
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("cannot create a llama context");
        vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
    }
    ~Target() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }

    std::vector<llama_token> tokenize(const std::string& text) const {
        std::vector<llama_token> t(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), int(text.size()), t.data(), int(t.size()), false, true);
        if (n < 0) throw std::runtime_error("tokenize failed");
        t.resize(size_t(n));
        return t;
    }

    // decodes tokens at positions pos.., asking for logits at every one (or only the last);
    // returns the argmax at each position that has logits
    std::vector<llama_token> decode(const std::vector<llama_token>& toks, int pos, bool all_logits) {
        llama_batch b = llama_batch_init(int(toks.size()), 0, 1);
        for (size_t i = 0; i < toks.size(); i++) {
            b.token[i] = toks[i];
            b.pos[i] = pos + int(i);
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = all_logits || i + 1 == toks.size();
        }
        b.n_tokens = int(toks.size());
        if (llama_decode(ctx, b) != 0) { llama_batch_free(b); throw std::runtime_error("llama_decode failed"); }
        std::vector<llama_token> out;
        for (size_t i = 0; i < toks.size(); i++) {
            if (!b.logits[i]) continue;
            const float* lg = llama_get_logits_ith(ctx, int(i));
            out.push_back(llama_token(std::max_element(lg, lg + n_vocab) - lg));
        }
        llama_batch_free(b);
        return out;
    }

    void drop_from(int pos) { llama_memory_seq_rm(llama_get_memory(ctx), 0, pos, -1); }
    void clear() { llama_memory_clear(llama_get_memory(ctx), true); }
};

// Feeds `feed` (at positions npos, npos+1, ...) and then drafts k tokens greedily, each fed back.
// Like npu::generate, the next step's host work (RoPE rows, runlist) is prepared on the other
// slot while the current step runs. Returns the k drafts; npos ends past the last fed token.
std::vector<llama_token> npu_draft(onebit::npu::Lane& lane, const std::vector<llama_token>& feed, int k, int& npos) {
    std::vector<llama_token> drafts;
    const int total = int(feed.size()) + k - 1;   // steps: every fed token, then k-1 drafts
    int cur = 0;
    lane.prepare(cur, npos + 1);
    lane.launch(cur, int(feed[0]));
    for (int s = 0; s < total; s++) {
        const int nxt = 1 - cur;
        const bool more = s + 1 < total;
        if (more) lane.prepare(nxt, npos + 2);    // overlaps the run in flight
        lane.wait(cur);
        npos++;
        llama_token tok;
        if (s + 1 < int(feed.size())) {
            tok = feed[size_t(s + 1)];            // still catching up: the argmax is not needed
        } else {
            drafts.push_back(llama_token(lane.argmax()));
            tok = drafts.back();
        }
        if (more) lane.launch(nxt, int(tok));
        cur = nxt;
    }
    if (int(drafts.size()) < k) drafts.push_back(llama_token(lane.argmax()));   // k == 1 or feed-only edge
    return drafts;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: %s <target.gguf> <npu model dir> <npu kernel dir> <k> <tokens> \"<prompt>\"\n", argv[0]);
        return 2;
    }
    const int K = std::atoi(argv[4]), N = std::atoi(argv[5]);
    try {
        llama_backend_init();
        Target tgt(argv[1], 4096);
        onebit::npu::Model npu_model(argv[2]);
        if (npu_model.dims().vocab > tgt.n_vocab) throw std::runtime_error("the drafter's vocabulary is larger than the target's");

        // Qwen3 chat, no thinking
        const std::string text = std::string("<|im_start|>user\n") + argv[6] +
                                 "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        const std::vector<llama_token> prompt = tgt.tokenize(text);
        const int P = int(prompt.size());
        auto is_eog = [&](llama_token t) { return llama_vocab_is_eog(tgt.vocab, t); };

        // 1. plain greedy decoding of the target: the reference output and the baseline speed
        std::vector<llama_token> plain;
        auto t0 = clk::now();
        llama_token next = tgt.decode(prompt, 0, false)[0];
        const double plain_prefill = ms_since(t0);
        t0 = clk::now();
        for (int i = 0; i < N && !is_eog(next); i++) {
            plain.push_back(next);
            next = tgt.decode({next}, P + i, false)[0];
        }
        const double plain_ms = ms_since(t0);
        tgt.clear();

        // 2. speculative: NPU drafts, GPU verifies. The lane is created now, right before its first
        // run: a lane left idle for more than a few seconds times out on its next run (the NPU's
        // idle handling; docs/npu.md), and the plain run above takes longer than that.
        onebit::npu::Lane lane(npu_model, argv[3]);
        std::vector<llama_token> out;
        long drafted = 0, accepted = 0, rounds = 0;
        double npu_ms = 0, gpu_ms = 0;
        t0 = clk::now();
        next = tgt.decode(prompt, 0, false)[0];       // target prefill
        auto tn = clk::now();
        for (int i = 0; i < P; i++) lane.step(prompt[size_t(i)], i + 1);   // NPU prefill
        const double npu_prefill = ms_since(tn);
        const double spec_prefill = ms_since(t0);
        t0 = clk::now();
        int pos = P;            // target: positions < pos are in its cache; `next` is not yet
        int npos = P;           // NPU: positions < npos are in its cache
        std::vector<llama_token> npu_pending;   // accepted tokens the NPU has not consumed yet
        while (int(out.size()) < N && !is_eog(next)) {
            // draft k tokens: catch the NPU up, then feed `next` and each draft
            auto td = clk::now();
            npu_pending.push_back(next);
            const std::vector<llama_token> drafts = npu_draft(lane, npu_pending, K, npos);
            npu_pending.clear();
            npu_ms += ms_since(td);
            // verify: the target reads next, d1..dk and predicts after each
            auto tv = clk::now();
            std::vector<llama_token> batch{next};
            batch.insert(batch.end(), drafts.begin(), drafts.end());
            const std::vector<llama_token> a = tgt.decode(batch, pos, true);
            gpu_ms += ms_since(tv);
            int j = 0;
            while (j < K && drafts[size_t(j)] == a[size_t(j)]) j++;
            // keep next, d1..dj; the target's own a[j] becomes the next `next`
            out.push_back(next);
            for (int i = 0; i < j && int(out.size()) < N; i++) out.push_back(drafts[size_t(i)]);
            rounds++; drafted += K; accepted += j;
            pos += 1 + j;
            tgt.drop_from(pos);
            // The NPU consumed next, d1..d(k-1). Its rows for next, d1..dj stay valid; rows past
            // them held rejected drafts and are overwritten later. If every draft was accepted, dk
            // was never fed to the NPU, so it waits in npu_pending and the NPU is one row behind.
            if (j == K) npu_pending.push_back(drafts[size_t(K - 1)]);
            npos = pos - (j == K ? 1 : 0);
            next = a[size_t(j)];
        }
        const double spec_ms = ms_since(t0);
        out.resize(std::min(out.size(), plain.size()));

        const bool same = out == plain;
        size_t diverge = 0;
        while (diverge < out.size() && diverge < plain.size() && out[diverge] == plain[diverge]) diverge++;
        std::printf("target %s, drafter Qwen3-0.6B on the NPU, k = %d, prompt %d tokens\n", argv[1], K, P);
        std::printf("plain greedy:    %zu tokens, %.1f tok/s (prefill %.0f ms)\n", plain.size(),
                    plain.size() * 1000.0 / plain_ms, plain_prefill);
        std::printf("NPU speculative: %zu tokens, %.1f tok/s (prefill %.0f ms, of it NPU %.0f ms); %ld rounds, "
                    "%.1f%% of drafts accepted, %.2f tokens per round\n",
                    out.size(), out.size() * 1000.0 / spec_ms, spec_prefill, npu_prefill, rounds,
                    100.0 * accepted / std::max(1L, drafted), double(accepted + rounds) / std::max(1L, rounds));
        std::printf("  time: NPU drafting %.0f ms, GPU verifying %.0f ms (sequential)\n", npu_ms, gpu_ms);
        std::printf("speed-up %.2fx; output %s plain greedy\n", (out.size() * 1000.0 / spec_ms) / (plain.size() * 1000.0 / plain_ms),
                    same ? "IDENTICAL to" : "DIFFERS from");
        if (!same)
            std::printf("  first difference at token %zu of %zu (a batched and a one-token GPU decode can break a "
                        "near-tie differently)\n", diverge, plain.size());
        llama_backend_free();
        return same ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
