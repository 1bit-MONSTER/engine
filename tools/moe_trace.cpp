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
// tensor, through the eval callback) and writes them as a trace for tools/moe_policy.cpp
// (docs/moe-streaming.md).
//
//   moe_trace model.gguf prompt.txt n_gen out.tsv          one prompt, then n_gen tokens
//   moe_trace --chat model.gguf turns.txt n_gen out.tsv    a conversation: turns.txt holds the
//                                                          user turns, separated by "===TURN===" lines;
//                                                          each gets up to n_gen reply tokens
//
// Trace rows: step \t layer \t kind \t e1,e2,... [\t p1,... \t q1,...]   kind P = a prompt token (a
// whole prompt batch is one step), D = a decoded token (one step each), in execution order.
// Decode rows carry a fifth column: the gate-ahead prediction for this layer, 2 x top-k experts
// best first, from layer l+1's router applied to layer l's output (l_out-l, normed with layer
// l+1's FFN norm). That is what an engine can compute while layer l+1's attention runs, to
// prefetch its experts. The sixth column is the same prediction from two layers back (layer
// l+1's router on l_out-(l-1)), available a whole layer earlier. "-" where there is no such
// layer or no router weights.
//
// Env: MOE_NGL (layers on the GPU, default all), MOE_BATCH (prompt batch, default 2048),
// MOE_CTX (context, default 8192), MOE_DEV (a single device, e.g. Vulkan0), MOE_CPU_EXPS=1
// (routed experts stay in the mmap'd file on the CPU, not repacked: a model larger than free
// memory then traces without loading its experts into RAM).
// Build against a libllama: g++ -std=c++17 -O2 -I<llama.cpp>/include -I<llama.cpp>/ggml/include
//   tools/moe_trace.cpp -L<build>/bin -lllama -lggml -lggml-base -Wl,-rpath,<build>/bin
// (add -DMOE_HAVE_LAZY for a llama.cpp with llama_model_params::lazy_mode)
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <cerrno>
#include <fcntl.h>

static std::map<int, std::vector<std::vector<int>>> g_cur;  // layer -> token rows of the step running now
static std::map<int, std::vector<float>> g_lout;               // layer -> l_out of a one-token step

// Router weights per layer, read from the GGUF (all shards of a split file), as f32.
struct Router { std::vector<float> w, norm, bias; int n_exp = 0; };
struct Predictor {
    int n_embd = 0;
    std::map<int, Router> layer;
    bool load(const std::string& path) {
        std::vector<std::string> files = { path };
        const auto pos = path.find("-00001-of-");
        if (pos != std::string::npos) {
            const int n = atoi(path.c_str() + pos + 10);
            files.clear();
            for (int i = 1; i <= n; i++) { char b[32]; snprintf(b, sizeof b, "-%05d-of-", i); files.push_back(path.substr(0, pos) + b + path.substr(pos + 10)); }
        }
        for (auto& fname : files) {
            ggml_context* meta = nullptr;
            gguf_context* g = gguf_init_from_file(fname.c_str(), { true, &meta });
            if (!g) continue;
            FILE* f = fopen(fname.c_str(), "rb");
            auto read = [&](const std::string& name, std::vector<float>& out, int64_t* ne0, int64_t* ne1) {
                const int64_t id = gguf_find_tensor(g, name.c_str());
                if (id < 0) return false;
                ggml_tensor* t = ggml_get_tensor(meta, name.c_str());
                std::vector<uint8_t> raw(gguf_get_tensor_size(g, id));
                fseeko(f, gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id), SEEK_SET);
                if (fread(raw.data(), 1, raw.size(), f) != raw.size()) return false;
                out.resize(ggml_nelements(t));
                if (t->type == GGML_TYPE_F32) memcpy(out.data(), raw.data(), out.size() * 4);
                else {
                    auto tr = ggml_get_type_traits(t->type);
                    if (!tr->to_float) return false;
                    tr->to_float(raw.data(), out.data(), out.size());
                }
                if (ne0) *ne0 = t->ne[0];
                if (ne1) *ne1 = t->ne[1];
                return true;
            };
            for (int il = 0; il < 512; il++) {
                const std::string b = "blk." + std::to_string(il) + ".";
                int64_t ne0 = 0, ne1 = 0;
                std::vector<float> w;
                if (!read(b + "ffn_gate_inp.weight", w, &ne0, &ne1)) continue;
                Router& r = layer[il];
                r.w = std::move(w); r.n_exp = (int) ne1; n_embd = (int) ne0;
                if (!read(b + "ffn_norm.weight", r.norm, nullptr, nullptr)) read(b + "post_attention_norm.weight", r.norm, nullptr, nullptr);
                read(b + "exp_probs_b.bias", r.bias, nullptr, nullptr);
            }
            fclose(f); gguf_free(g); ggml_free(meta);
        }
        return !layer.empty();
    }
    // best 2 x top_k experts of layer il for hidden state h (the previous layer's output)
    std::vector<int> predict(int il, const std::vector<float>& h, int top_k) const {
        auto it = layer.find(il);
        if (it == layer.end() || (int) h.size() != n_embd) return {};
        const Router& r = it->second;
        double ss = 0; for (float v : h) ss += double(v) * v;
        const float inv = 1.0f / std::sqrt(float(ss / n_embd) + 1e-6f);
        std::vector<float> x(n_embd);
        for (int i = 0; i < n_embd; i++) x[i] = h[i] * inv * (r.norm.empty() ? 1.0f : r.norm[i]);
        std::vector<std::pair<float, int>> sc(r.n_exp);
        for (int e = 0; e < r.n_exp; e++) {
            const float* w = r.w.data() + size_t(e) * n_embd;
            float acc = 0; for (int i = 0; i < n_embd; i++) acc += w[i] * x[i];
            if (!r.bias.empty()) acc = 1.0f / (1.0f + std::exp(-acc)) + r.bias[e];  // sigmoid router + selection bias
            sc[e] = { -acc, e };
        }
        const int k = std::min<int>(2 * top_k, r.n_exp);
        std::partial_sort(sc.begin(), sc.begin() + k, sc.end());
        std::vector<int> out; for (int i = 0; i < k; i++) out.push_back(sc[i].second);
        return out;
    }
};
static Predictor g_pred;

static bool cb(struct ggml_tensor* t, bool ask, void*) {
    if (strncmp(t->name, "l_out-", 6) == 0) {
        if (ask) return !g_pred.layer.empty();
        if (t->ne[1] == 1 && t->type == GGML_TYPE_F32) {  // one-token steps only
            auto& v = g_lout[atoi(t->name + 6)];
            v.resize(t->ne[0]);
            ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
        }
        return true;
    }
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

struct Writer {
    FILE* f = nullptr;
    long step = 0, prompt_tokens = 0, decode_tokens = 0;
    void flush(char kind) {  // write the step the callback just collected
        for (auto& [l, rows] : g_cur)
            for (auto& r : rows) {
                fprintf(f, "%ld\t%d\t%c\t", step, l, kind);
                for (size_t i = 0; i < r.size(); i++) fprintf(f, i ? ",%d" : "%d", r[i]);
                if (kind == 'D') {
                    for (int back : {1, 2}) {
                        std::vector<int> p;
                        auto prev = g_lout.find(l - back);
                        if (prev != g_lout.end()) p = g_pred.predict(l, prev->second, (int) r.size());
                        fputc('\t', f);
                        if (p.empty()) fputc('-', f);
                        for (size_t i = 0; i < p.size(); i++) fprintf(f, i ? ",%d" : "%d", p[i]);
                    }
                }
                fputc('\n', f);
            }
        if (!g_cur.empty()) {
            const long n = (long) g_cur.begin()->second.size();
            (kind == 'P' ? prompt_tokens : decode_tokens) += n;
            step++;
        }
        g_cur.clear();
        g_lout.clear();
    }
};

static std::string read_file(const char* path) {
    std::ifstream f(path); std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

int main(int argc, char** argv) {
    const bool chat = argc > 1 && std::string(argv[1]) == "--chat";
    char** a = argv + (chat ? 1 : 0);
    if (argc - (chat ? 1 : 0) < 5) {
        fprintf(stderr, "moe_trace [--chat] model.gguf prompt.txt|turns.txt n_gen out.tsv\n");
        return 2;
    }
    const std::string input = read_file(a[2]);
    const int n_gen = atoi(a[3]);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = getenv("MOE_NGL") ? atoi(getenv("MOE_NGL")) : 999;
    ggml_backend_dev_t devs[2] = { nullptr, nullptr };
    if (getenv("MOE_DEV")) {
        devs[0] = ggml_backend_dev_by_name(getenv("MOE_DEV"));
        if (!devs[0]) { fprintf(stderr, "no device %s\n", getenv("MOE_DEV")); return 1; }
        mp.devices = devs;
    }
    static llama_model_tensor_buft_override ovr[2] = {};
    if (getenv("MOE_CPU_EXPS")) {
        ovr[0] = { "_exps\\.", ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) };
        ovr[1] = { nullptr, nullptr };
        mp.tensor_buft_overrides = ovr;
        mp.use_extra_bufts = false;  // no repacking: the experts stay file-backed
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    }
#ifdef MOE_HAVE_LAZY  // llama.cpp builds with lazy tensors (qwen4exp's per-layer token embeddings)
    mp.lazy_mode = LLAMA_LAZY_MODE_AUTO;
#endif
    llama_model* model = llama_model_load_from_file(a[1], mp);
    if (!model) return 1;
    if (!getenv("MOE_NO_PREDICT") && g_pred.load(a[1]))
        fprintf(stderr, "gate-ahead predictor: %zu router layers, n_embd %d\n", g_pred.layer.size(), g_pred.n_embd);
    auto cp = llama_context_default_params();
    cp.n_ctx = getenv("MOE_CTX") ? atoi(getenv("MOE_CTX")) : 8192;
    cp.n_batch = getenv("MOE_BATCH") ? atoi(getenv("MOE_BATCH")) : 2048;
    cp.n_ubatch = std::min<int>(cp.n_batch, 512);
    cp.cb_eval = cb;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    const llama_vocab* vocab = llama_model_get_vocab(model);
    auto* smpl = llama_sampler_init_greedy();
    Writer w;
    // the trace is data for the owner: 0644, not fopen's 0666 left to the umask
    const int tfd = open(a[4], O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    w.f = tfd < 0 ? nullptr : fdopen(tfd, "w");
    if (!w.f) { fprintf(stderr, "cannot write %s: %s\n", a[4], strerror(errno)); return 1; }
    int n_past = 0;

    auto tokenize = [&](const std::string& text, bool bos) {
        std::vector<llama_token> t(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), text.size(), t.data(), t.size(), bos, true);
        t.resize(std::max(0, n));
        return t;
    };
    auto feed = [&](std::vector<llama_token>& toks) {  // a prompt, one batch at a time
        for (size_t i = 0; i < toks.size(); i += cp.n_batch) {
            const int n = std::min<size_t>(cp.n_batch, toks.size() - i);
            if (n_past + n > (int) cp.n_ctx) return false;
            llama_batch b = llama_batch_get_one(toks.data() + i, n);
            if (llama_decode(ctx, b)) return false;
            n_past += n;
            w.flush('P');
        }
        return true;
    };
    double decode_s = 0;
    auto generate = [&](std::string* out) {  // up to n_gen tokens, greedy
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n_gen && n_past < (int) cp.n_ctx; i++) {
            llama_token t = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, t)) break;
            if (out) { char buf[256]; int n = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, true); if (n > 0) out->append(buf, n); }
            llama_batch bb = llama_batch_get_one(&t, 1);
            if (llama_decode(ctx, bb)) return false;
            n_past++;
            w.flush('D');
        }
        decode_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return true;
    };

    if (!chat) {
        auto toks = tokenize(input, true);
        if (!feed(toks) || !generate(nullptr)) { fprintf(stderr, "decode failed (context %d)\n", cp.n_ctx); return 1; }
    } else {
        std::vector<std::string> turns;
        std::stringstream ss(input); std::string line, cur;
        while (std::getline(ss, line)) {
            if (line == "===TURN===") { if (!cur.empty()) turns.push_back(cur); cur.clear(); }
            else cur += (cur.empty() ? "" : "\n") + line;
        }
        if (!cur.empty()) turns.push_back(cur);
        const char* tmpl = llama_model_chat_template(model, nullptr);
        std::vector<std::string> roles, contents;
        std::string rendered_prev;
        for (size_t t = 0; t < turns.size(); t++) {
            roles.push_back("user"); contents.push_back(turns[t]);
            std::vector<llama_chat_message> msgs;
            for (size_t i = 0; i < roles.size(); i++) msgs.push_back({ roles[i].c_str(), contents[i].c_str() });
            std::vector<char> buf(1 << 20);
            int n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, buf.data(), buf.size());
            std::string rendered;
            if (n < 0) {  // not a built-in template: ChatML
                for (size_t i = 0; i < msgs.size(); i++) rendered += std::string("<|im_start|>") + msgs[i].role + "\n" + msgs[i].content + "<|im_end|>\n";
                rendered += "<|im_start|>assistant\n";
            } else rendered.assign(buf.data(), n);
            // feed only what this turn adds to the conversation so far
            std::string delta = rendered.compare(0, rendered_prev.size(), rendered_prev) == 0 ? rendered.substr(rendered_prev.size()) : rendered;
            auto toks = tokenize(delta, t == 0);
            if (!feed(toks)) { fprintf(stderr, "context full at turn %zu\n", t); break; }
            std::string reply;
            if (!generate(&reply)) break;
            roles.push_back("assistant"); contents.push_back(reply);
            // the rendered conversation including the reply, as the next turn's prefix
            std::vector<llama_chat_message> m2;
            for (size_t i = 0; i < roles.size(); i++) m2.push_back({ roles[i].c_str(), contents[i].c_str() });
            n = llama_chat_apply_template(tmpl, m2.data(), m2.size(), false, buf.data(), buf.size());
            if (n < 0) {
                rendered_prev.clear();
                for (size_t i = 0; i < m2.size(); i++) rendered_prev += std::string("<|im_start|>") + m2[i].role + "\n" + m2[i].content + "<|im_end|>\n";
            } else rendered_prev.assign(buf.data(), n);
            // the model's own end-of-turn tokens were not decoded; feed the closing text of the reply
            std::string closing = rendered_prev.substr(std::min(rendered_prev.size(), rendered.size() + reply.size()));
            if (!closing.empty()) { auto ct = tokenize(closing, false); if (!feed(ct)) break; }
            fprintf(stderr, "turn %zu: %zu reply chars, context %d\n", t + 1, reply.size(), n_past);
        }
    }
    fclose(w.f);
    printf("prompt tokens %ld, decoded %ld, steps %ld, context %d, decode %.2f tok/s\n", w.prompt_tokens,
           w.decode_tokens, w.step, n_past, w.decode_tokens / std::max(1e-9, decode_s));
    llama_sampler_free(smpl); llama_free(ctx); llama_model_free(model);
    return 0;
}
