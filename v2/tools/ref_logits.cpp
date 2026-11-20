// The golden that tests/v2_cpu_e2e checks V2's CPU reference executor against: llama.cpp's own CPU
// logits for a prompt and its greedy continuation (docs/v2.md, "The first end-to-end check").
//
// Not part of the engine build: it links a CPU-only llama.cpp build (the oracle, D1), so it is
// built by hand when the golden is regenerated:
//
//   cmake -S llama.cpp -B build -DGGML_NATIVE=ON && cmake --build build --target llama
//   g++ -O2 -std=c++17 v2/tools/ref_logits.cpp -Illama.cpp/include -Illama.cpp/ggml/include \
//       -Lbuild/bin -lllama -lggml -lggml-base -Wl,-rpath,build/bin -o ref_logits
//   ./ref_logits model.gguf "prompt text" 16 > tests/golden/v2/qwen3-0.6b-q4_k_m.json
//
// For every position of the prompt and of the greedy continuation it writes the log-sum-exp of
// the logits and the top 128 (id, logit) pairs: enough for top-1 agreement and for a KLD over the
// partition {top 128, the rest}, a lower bound of the full KLD that is tight when the top 128 hold
// nearly all the mass (the file records that mass).
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int kTop = 128;

void dump_position(const float* logits, int n_vocab, int pos, int token_in, bool first) {
    std::vector<int> idx(static_cast<size_t>(n_vocab));
    for (int i = 0; i < n_vocab; ++i) idx[size_t(i)] = i;
    std::partial_sort(idx.begin(), idx.begin() + kTop, idx.end(), [&](int a, int b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
    });
    double mx = logits[idx[0]], s = 0;
    for (int i = 0; i < n_vocab; ++i) s += std::exp(double(logits[i]) - mx);
    const double lse = mx + std::log(s);
    double top_mass = 0;
    for (int k = 0; k < kTop; ++k) top_mass += std::exp(double(logits[idx[size_t(k)]]) - lse);
    std::printf("%s    {\"pos\": %d, \"token_in\": %d, \"lse\": %.9g, \"top_mass\": %.9g, \"top\": [", first ? "" : ",\n", pos,
                token_in, lse, top_mass);
    for (int k = 0; k < kTop; ++k)
        std::printf("%s[%d, %.7g]", k ? ", " : "", idx[size_t(k)], double(logits[idx[size_t(k)]]));
    std::printf("]}");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: ref_logits model.gguf \"prompt\" n_generate [threads]\n");
        return 2;
    }
    const std::string path = argv[1], prompt = argv[2];
    const int n_gen = std::atoi(argv[3]);
    const int threads = argc > 4 ? std::atoi(argv[4]) : 8;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) return 1;
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> toks(prompt.size() + 16);
    const int n = llama_tokenize(vocab, prompt.c_str(), int(prompt.size()), toks.data(), int(toks.size()), true, true);
    if (n <= 0) return 1;
    toks.resize(size_t(n));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    std::printf("{\n  \"generator\": \"v2/tools/ref_logits.cpp (llama.cpp CPU)\",\n  \"prompt\": \"");
    for (char c : prompt) {
        if (c == '"' || c == '\\') std::printf("\\%c", c);
        else if (c == '\n') std::printf("\\n");
        else std::printf("%c", c);
    }
    std::printf("\",\n  \"prompt_tokens\": [");
    for (int i = 0; i < n; ++i) std::printf("%s%d", i ? ", " : "", toks[size_t(i)]);
    std::printf("],\n  \"positions\": [\n");

    llama_batch batch = llama_batch_init(512, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.token[i] = toks[size_t(i)];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    if (llama_decode(ctx, batch) != 0) return 1;
    std::vector<llama_token> gen;
    llama_token next = 0;
    for (int i = 0; i < n; ++i) {
        const float* lg = llama_get_logits_ith(ctx, i);
        dump_position(lg, n_vocab, i, toks[size_t(i)], i == 0);
        if (i == n - 1) next = llama_token(std::max_element(lg, lg + n_vocab) - lg);
    }
    for (int g = 0; g < n_gen; ++g) {
        gen.push_back(next);
        batch.n_tokens = 1;
        batch.token[0] = next;
        batch.pos[0] = n + g;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        if (llama_decode(ctx, batch) != 0) return 1;
        const float* lg = llama_get_logits_ith(ctx, 0);
        dump_position(lg, n_vocab, n + g, next, false);
        next = llama_token(std::max_element(lg, lg + n_vocab) - lg);
    }
    std::printf("\n  ],\n  \"greedy\": [");
    for (size_t i = 0; i < gen.size(); ++i) std::printf("%s%d", i ? ", " : "", gen[i]);
    std::printf("]\n}\n");
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
