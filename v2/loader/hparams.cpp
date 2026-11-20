#include "hparams.h"

#include <cstdio>
#include <stdexcept>

namespace onebit::v2 {

std::string HParams::summary() const {
    char b[320];
    std::snprintf(b, sizeof b,
                  "%s: %d layers, embd %d, heads %d/%d kv, head dim %d/%d, ffn %d, vocab %d, ctx %lld, rope base %.0f, "
                  "rms eps %g%s",
                  arch.c_str(), n_layer, n_embd, n_head, n_head_kv, head_dim_k, head_dim_v, n_ff, n_vocab,
                  (long long) n_ctx_train, rope_base, rms_eps, tied_embeddings ? ", tied embeddings" : "");
    return b;
}

HParams read_hparams(const GgufFile& g) {
    HParams h;
    h.arch = g.arch();
    if (h.arch.empty()) throw std::runtime_error("GGUF has no general.architecture");
    auto need = [&](const char* key) {
        const int64_t v = g.arch_int(key, -1);
        if (v <= 0 || v > (int64_t(1) << 30)) throw std::runtime_error("GGUF " + h.arch + "." + key + " missing or out of range");
        return int(v);
    };
    h.n_layer = need("block_count");
    h.n_embd = need("embedding_length");
    h.n_head = need("attention.head_count");
    h.n_head_kv = int(g.arch_int("attention.head_count_kv", h.n_head));
    if (h.n_head_kv <= 0 || h.n_head % h.n_head_kv) throw std::runtime_error("head_count_kv must divide head_count");
    h.head_dim_k = int(g.arch_int("attention.key_length", h.n_embd / h.n_head));
    h.head_dim_v = int(g.arch_int("attention.value_length", h.head_dim_k));
    if (h.head_dim_k <= 0 || h.head_dim_v <= 0 || h.head_dim_k % 2) throw std::runtime_error("bad attention key/value length");
    h.n_ff = need("feed_forward_length");
    h.n_ctx_train = g.arch_int("context_length", 4096);
    h.rope_base = g.arch_float("rope.freq_base", 10000.0);
    h.rms_eps = g.arch_float("attention.layer_norm_rms_epsilon", 1e-6);
    const TensorInfo* emb = g.tensor("token_embd.weight");
    if (!emb || emb->ne.size() != 2) throw std::runtime_error("no 2-D token_embd.weight");
    if (emb->ne[0] != uint64_t(h.n_embd)) throw std::runtime_error("token_embd.weight width " + std::to_string(emb->ne[0]) + " != embedding_length");
    h.n_vocab = int(emb->ne[1]);
    h.tied_embeddings = g.tensor("output.weight") == nullptr;
    return h;
}

}  // namespace onebit::v2
