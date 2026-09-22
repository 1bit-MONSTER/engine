// cpu_model.h: fp32 reference decoder on the CPU.
//
// This is the correctness oracle for the NPU and GPU backends, so it favours
// plain, checkable code over speed: every weight is dequantized to fp32 at load
// time and every op runs in fp32, one token at a time.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

#include "gguf.h"
#include "model_config.h"
#include "parallel.h"

namespace onebit {

class CpuModel {
public:
    // Loads every tensor of the GGUF. Fails if a tensor is missing, has the
    // wrong shape, has an unsupported type, or is present but unused (an
    // unused tensor means the model has semantics this code does not model).
    static std::expected<CpuModel, std::string> load(const std::string& gguf_path, size_t n_threads = 0);

    const ModelConfig& config() const { return cfg_; }

    // Clears the KV cache.
    void reset();

    // Number of tokens in the KV cache.
    uint32_t n_past() const { return n_past_; }

    // Runs one token at position n_past() and returns the next-token logits
    // (config().n_vocab floats). Fails when the context is full.
    std::expected<const std::vector<float>*, std::string> forward(int32_t token);

    CpuModel(CpuModel&&) = default;
    CpuModel& operator=(CpuModel&&) = default;

private:
    CpuModel() = default;

    struct Layer {
        std::vector<float> attn_norm, wq, wk, wv, wo, q_norm, k_norm;
        std::vector<float> ffn_norm, w_gate, w_up, w_down;
    };

    void matvec(const std::vector<float>& w, const float* x, float* y, uint32_t rows, uint32_t cols);

    ModelConfig cfg_;
    uint32_t n_ctx_ = 0;
    std::vector<float> tok_embd_, output_norm_, output_;  // output_ empty when tied to tok_embd_
    std::vector<Layer> layers_;
    std::vector<float> k_cache_, v_cache_;  // [layer][pos][n_head_kv * head_dim]
    uint32_t n_past_ = 0;

    // Scratch.
    std::vector<float> x_, xn_, q_, k_, v_, attn_, scores_, gate_, up_, logits_;
    std::unique_ptr<ThreadPool> pool_;
};

}  // namespace onebit
