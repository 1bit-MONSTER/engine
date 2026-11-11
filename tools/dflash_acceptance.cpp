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

// dflash_acceptance.cpp -- a DFlash draft scored against its target model, both on the host (engine#1780).
//
//   dflash_acceptance --model <target dir> --draft <draft dir> --prompt-ids <file> [--tokens 32] [--window 0]
//                     [--all-positions]
//
// <file> holds one prompt per line as token ids (the tokenizer stays out of the harness; ids for a chat-formatted
// prompt are made once with the pack's tokenizer.json).
//
// What it does, per prompt:
//   1. the target runs the prompt and then `--tokens` greedy tokens through Q4nxNpuForward::step(), with
//      set_layer_taps() on the draft's target_layer_ids, so every position has its tapped row;
//   2. the draft is then asked, at a position s, for the block after token s: context = the tapped rows of
//      positions 0..s-1, block = the target's embeddings of [token s, mask, mask, ...] times the draft's
//      embedding_scale, and each block row after the first goes through the target's own head;
//   3. a proposal is accepted while it equals the target's greedy token at that position. The run walks the
//      positions the way a speculative loop would (after a pass that accepted a, the next pass starts a + 1 later),
//      or every position with --all-positions.
//
// Reported: passes, accepted tokens per pass, the first proposal's hit rate, and the tokens a pass would emit
// (accepted + 1, the target's own next token). This is the acceptance half of the issue's measured row; it says
// nothing about tok/s -- both models run on the host here, one thread, as references.
//
// Host-GEMM only: it sets NPU_INFER_HOST_GEMM and never opens the device.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "dflash_draft.h"
#include "dflash_forward.h"
#include "q4nx_forward.h"

namespace {

using namespace onebit::npu::dflash;

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int argmax(const std::vector<float>& v) {
    return int(std::max_element(v.begin(), v.end()) - v.begin());
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir, draft_dir, ids_path;
    int tokens = 32, window = 0;
    bool all_positions = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--model") model_dir = next();
        else if (a == "--draft") draft_dir = next();
        else if (a == "--prompt-ids") ids_path = next();
        else if (a == "--tokens") tokens = std::atoi(next().c_str());
        else if (a == "--window") window = std::atoi(next().c_str());
        else if (a == "--all-positions") all_positions = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model_dir.empty() || draft_dir.empty() || ids_path.empty() || tokens < 2) {
        std::fprintf(stderr, "usage: dflash_acceptance --model <target dir> --draft <draft dir> --prompt-ids <file> "
                             "[--tokens 32] [--window 0] [--all-positions]\n");
        return 2;
    }

    std::vector<std::vector<int>> prompts;
    {
        std::ifstream in(ids_path);
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::vector<int> ids;
            int v;
            while (ls >> v) ids.push_back(v);
            if (!ids.empty()) prompts.push_back(ids);
        }
    }
    if (prompts.empty()) { std::fprintf(stderr, "%s holds no prompt\n", ids_path.c_str()); return 2; }

    DraftWeights w;
    DraftConfig cfg;
    try {
        const DraftModel m = load_draft(draft_dir);
        cfg = m.cfg;
        w = decode_draft(m);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "draft: %s\n", e.what());
        return 1;
    }
    const int block = cfg.block_size, T = cfg.taps(), H = cfg.hidden;

    setenv("NPU_INFER_HOST_GEMM", "1", 1);
    Q4nxNpuForward fwd;
    if (!fwd.init_host((model_dir + "/model.q4nx").c_str(), model_dir + "/kernels")) {
        std::fprintf(stderr, "target: %s\n", fwd.last_error());
        return 1;
    }
    std::vector<float> row;
    fwd.embed_row(0, row);
    if (int(row.size()) != H) {
        std::fprintf(stderr, "the target is %zu wide and the draft %d: not a pair\n", row.size(), H);
        return 1;
    }
    std::vector<float> taps;
    if (!fwd.set_layer_taps(cfg.target_layer_ids, &taps)) {
        std::fprintf(stderr, "the draft's target layers do not fit a %d-layer target\n", fwd.layer_count());
        return 1;
    }
    std::printf("pair: target %d layers, H %d; draft %zu layers, block %d, taps", fwd.layer_count(), H, w.layers.size(), block);
    for (int l : cfg.target_layer_ids) std::printf(" %d", l);
    std::printf(", mask %lld, embedding scale %.5f, window %d\n", (long long)cfg.mask_token_id, cfg.embedding_scale, window);

    long passes = 0, accepted = 0, first_hits = 0, proposals = 0;
    std::vector<long> hist(size_t(block), 0);   // hist[a] = passes that accepted exactly a
    double t_target = 0.0, t_draft = 0.0;
    for (size_t p = 0; p < prompts.size(); p++) {
        std::vector<int> seq = prompts[p];
        const int P = int(seq.size());
        fwd.reset();
        taps.clear();
        std::vector<float> logits;
        const double t0 = now_s();
        for (int pos = 0; pos < P + tokens; pos++) {
            if (!fwd.step(seq[size_t(pos)], pos, logits)) {
                std::fprintf(stderr, "target step at %d: %s\n", pos, fwd.last_error());
                return 1;
            }
            if (pos >= P - 1) seq.push_back(argmax(logits));   // greedy: the token for position pos + 1
        }
        t_target += now_s() - t0;
        if (taps.size() != size_t(P + tokens) * size_t(T) * size_t(H)) {
            std::fprintf(stderr, "the taps hold %zu floats, expected %zu\n", taps.size(), size_t(P + tokens) * size_t(T) * size_t(H));
            return 1;
        }
        std::printf("prompt %zu: %d prompt tokens, greedy:", p, P);
        for (int i = P; i < P + std::min(tokens, 12); i++) std::printf(" %d", seq[size_t(i)]);
        std::printf(" ...\n");

        // seq[s] is known for s <= P + tokens; a block after s needs seq[s + 1 .. s + block - 1] to score against.
        const int last_start = P + tokens - (block - 1);
        const double t1 = now_s();
        long pa = 0, pp = 0;
        for (int s = P; s <= last_start;) {
            std::vector<float> noise(size_t(block) * size_t(H));
            for (int i = 0; i < block; i++) {
                fwd.embed_row(i == 0 ? seq[size_t(s)] : int(cfg.mask_token_id), row);
                for (int k = 0; k < H; k++) noise[size_t(i) * size_t(H) + size_t(k)] = row[size_t(k)] * cfg.embedding_scale;
            }
            std::vector<float> out;
            try {
                out = draft_forward(w, taps.data(), s, noise.data(), block, 0, window);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "draft at %d: %s\n", s, e.what());
                return 1;
            }
            int a = 0;
            bool open = true;
            for (int i = 1; i < block; i++) {
                std::vector<float> h(out.begin() + ptrdiff_t(size_t(i) * size_t(H)), out.begin() + ptrdiff_t(size_t(i + 1) * size_t(H)));
                std::vector<float> lg;
                if (!fwd.lm_head_logits(h, lg)) { std::fprintf(stderr, "head: %s\n", fwd.last_error()); return 1; }
                const bool hit = argmax(lg) == seq[size_t(s + i)];
                if (i == 1) { proposals++; first_hits += hit ? 1 : 0; }
                if (open && hit) a++;
                else open = false;
                if (!open && i >= 1) break;
            }
            passes++; pp++;
            accepted += a; pa += a;
            hist[size_t(a)]++;
            s += all_positions ? 1 : a + 1;
        }
        t_draft += now_s() - t1;
        std::printf("prompt %zu: %ld passes, %ld accepted, %.2f accepted per pass\n", p, pp, pa, pp ? double(pa) / double(pp) : 0.0);
        std::fflush(stdout);
    }
    std::printf("TOTAL %s: %ld passes, %ld accepted, %.3f accepted per pass, %.3f tokens per pass, first proposal right %ld of %ld (%.1f %%)\n",
                all_positions ? "every position" : "speculative walk", passes, accepted,
                passes ? double(accepted) / double(passes) : 0.0, passes ? double(accepted + passes) / double(passes) : 0.0,
                first_hits, proposals, proposals ? 100.0 * double(first_hits) / double(proposals) : 0.0);
    std::printf("accepted-per-pass histogram:");
    for (int a = 0; a < block; a++) std::printf(" %d:%ld", a, hist[size_t(a)]);
    std::printf("\nhost time: target %.1f s, draft %.1f s (references, one thread; not a speed result)\n", t_target, t_draft);
    return 0;
}
