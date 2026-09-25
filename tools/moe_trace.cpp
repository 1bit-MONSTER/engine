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
// moe_trace: records every token's routed experts per layer (llama.cpp's ffn_moe_topk
// tensor, through the eval callback) and simulates a per-layer LRU expert cache: the
// decode hit rate for several cache sizes (docs/moe-streaming.md). Env: MOE_NGL (layers
// on the GPU, default all), MOE_BATCH (prompt batch, default 2048).
// Build against a libllama: g++ -std=c++17 -O2 -I<llama.cpp>/include -I<llama.cpp>/ggml/include
//   tools/moe_trace.cpp -L<build>/bin -lllama -lggml -lggml-base -Wl,-rpath,<build>/bin
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <cstdlib>

struct Trace { std::vector<std::vector<std::vector<int>>> steps; };  // step -> layer -> experts (per token rows flattened)
static std::map<int, std::vector<std::vector<int>>> g_cur;           // layer -> tokens -> experts

static bool cb(struct ggml_tensor* t, bool ask, void*) {
    if (strncmp(t->name, "ffn_moe_topk", 12) != 0) return ask ? false : true;
    if (ask) return true;
    const int il = atoi(strrchr(t->name, '-') + 1);
    const int k = t->ne[0], n = t->ne[1];
    std::vector<int32_t> v(size_t(k) * n);
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
    auto& rows = g_cur[il];
    for (int j = 0; j < n; j++) rows.emplace_back(v.begin() + j * k, v.begin() + (j + 1) * k);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "moe_trace model.gguf prompt.txt n_gen [out.tsv]\n"); return 2; }
    std::ifstream pf(argv[2]); std::stringstream ss; ss << pf.rdbuf(); const std::string prompt = ss.str();
    const int n_gen = atoi(argv[3]);
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = getenv("MOE_NGL") ? atoi(getenv("MOE_NGL")) : 999;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    auto cp = llama_context_default_params(); cp.n_ctx = 8192; cp.n_batch = getenv("MOE_BATCH") ? atoi(getenv("MOE_BATCH")) : 2048; cp.n_ubatch = std::min<int>(cp.n_batch, 512); cp.cb_eval = cb; cp.cb_eval_user_data = nullptr;
    llama_context* ctx = llama_init_from_model(model, cp);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true);
    toks.resize(n);
    auto* smpl = llama_sampler_init_greedy();
    std::vector<std::map<int, std::vector<std::vector<int>>>> per_step;  // step: layer -> token rows
    for (size_t i = 0; i < toks.size(); i += cp.n_batch) {  // the prompt, one batch at a time
        llama_batch b = llama_batch_get_one(toks.data() + i, std::min<size_t>(cp.n_batch, toks.size() - i));
        if (llama_decode(ctx, b)) return 1;
    }
    per_step.push_back(g_cur); g_cur.clear();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_gen; i++) {
        llama_token t = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, t)) break;
        llama_batch bb = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, bb)) return 1;
        per_step.push_back(g_cur); g_cur.clear();
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("decode %.2f tok/s (%zu tokens in %.1f s)\n", (per_step.size() - 1) / dt, per_step.size() - 1, dt);
    const int n_layer = per_step[0].size();
    int n_exp = 0; for (auto& [l, rows] : per_step[0]) for (auto& r : rows) for (int e : r) n_exp = std::max(n_exp, e + 1);
    printf("prompt %d tokens, decoded %zu, layers %d, experts seen up to %d\n", n, per_step.size() - 1, n_layer, n_exp);
    if (argc > 4) {  // token-level trace: step \t layer \t e1,e2,...
        FILE* f = fopen(argv[4], "w");
        for (size_t s = 0; s < per_step.size(); s++)
            for (auto& [l, rows] : per_step[s])
                for (auto& r : rows) { fprintf(f, "%zu\t%d\t", s, l); for (size_t i = 0; i < r.size(); i++) fprintf(f, i ? ",%d" : "%d", r[i]); fprintf(f, "\n"); }
        fclose(f);
    }
    // decode-only LRU per layer, cache warmed by the prompt
    for (int cap : {16, 32, 64, 96, 128, 192, 256, 384}) {
        long hit = 0, tot = 0;
        for (int l = 0; l < n_layer; l++) {
            std::list<int> lru; std::unordered_map<int, std::list<int>::iterator> pos;
            auto touch = [&](int e, bool count) {
                auto it = pos.find(e);
                if (it != pos.end()) { if (count) hit++; lru.erase(it->second); }
                else if ((int)lru.size() >= cap) { pos.erase(lru.back()); lru.pop_back(); }
                lru.push_front(e); pos[e] = lru.begin();
                if (count) tot++;
            };
            for (size_t s = 0; s < per_step.size(); s++) {
                auto it = per_step[s].find(l); if (it == per_step[s].end()) continue;
                for (auto& r : it->second) for (int e : r) touch(e, s > 0);
            }
        }
        printf("LRU %3d experts/layer: decode hit rate %.1f%%\n", cap, 100.0 * hit / std::max(1L, tot));
    }
    llama_sampler_free(smpl); llama_free(ctx); llama_model_free(model);
    return 0;
}
