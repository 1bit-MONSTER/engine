#pragma once

// [F] The hyperparameters an architecture description needs, read from GGUF metadata and checked
// against the tensor table.

#include "gguf.h"

#include <cstdint>
#include <string>

namespace onebit::v2 {

struct HParams {
    std::string arch;
    int n_layer = 0;
    int n_embd = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int head_dim_k = 0;
    int head_dim_v = 0;
    int n_ff = 0;
    int n_vocab = 0;
    int64_t n_ctx_train = 0;
    double rope_base = 10000.0;
    double rms_eps = 1e-6;
    bool tied_embeddings = false;   // no output.weight: the LM head reads token_embd
    // KV bytes per token for one sequence at elem_bytes per element (all layers, K and V)
    uint64_t kv_bytes_per_token(double elem_bytes) const {
        return uint64_t(double(n_layer) * double(n_head_kv) * double(head_dim_k + head_dim_v) * elem_bytes);
    }
    std::string summary() const;
};

// Throws std::runtime_error naming the missing or inconsistent key.
HParams read_hparams(const GgufFile& g);

}  // namespace onebit::v2
