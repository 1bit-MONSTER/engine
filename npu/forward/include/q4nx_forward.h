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
// Copyright (c) 2026 bong-water-water-bong
// q4nx_forward.h — NPU forward pass for Qwen3-0.6B from model.q4nx.
//
// Drives this repo's four per-projection GEMM designs through I8Ctx:
//
//   QKV : A[1,1024]  x B[1024,4096] -> C[1,4096]   = q(2048) | k(1024) | v(1024)
//   O   : A[1,2048]  x B[2048,1024] -> C[1,1024]
//   GU  : A[1,1024]  x B[1024,6144] -> C[1,6144]   = gate(3072) | up(3072)
//   D   : A[1,3072]  x B[3072,1024] -> C[1,1024]
//
// plus a tiled LM head GEMM (K=1024 over the tied embedding) reusing the QKV
// geometry.  The non-GEMM work (RMSNorm, RoPE, GQA attention with KV cache,
// SiLU) runs on the host in float32, exactly as tools/q4nx_forward_ref.py does,
// so the two can be compared element-wise.
//
// Weights come from the verified Q4NX dequantizer (src/q4nx_dequant.c) and are
// packed per output column by I8Ctx::packB (src/q4nx_pack.c documents the
// numerics).
#ifndef NPU_INFER_Q4NX_FORWARD_H
#define NPU_INFER_Q4NX_FORWARD_H

#include <memory>
#include <string>
#include <vector>

#include "model.h"
#include "q4nx_pack.h"        // Q4nxDesignGeom / Q4nxModelDims (derived design table)
#include "q4nx_semantics.h"   // Q4nxSemantics (config-derived architecture flags)
#include <memory>
#include <unordered_map>

struct I8Ctx;

namespace xrt { class device; class bo; }

struct Q4nxForwardConfig {
    int H = 1024, NH = 16, NKV = 8, HD = 128, IM = 3072, NL = 28, NV = 151936;
    float eps = 1e-6f, theta = 1e6f;
};

class Q4nxNpuForward {
public:
    Q4nxNpuForward();
    ~Q4nxNpuForward();
    Q4nxNpuForward(const Q4nxNpuForward&) = delete;
    Q4nxNpuForward& operator=(const Q4nxNpuForward&) = delete;

    // artifact_dir holds the full ELFs (full_i8_*.elf).
    bool init(xrt::device& dev, const char* model_path, const std::string& artifact_dir);
    // Host-GEMM mode (NPU_INFER_HOST_GEMM=1): never opens the device.  Design
    // derivation, ELF mapping, weight decode and the layer maths all run, but
    // npu_gemm() does a plain float matmul.  This validates the whole forward
    // against the Python reference offline, which is how the C++ GDN port is
    // checked without the shared NPU (see docs/qwen35-4b-weight-format.md).
    bool init_host(const char* model_path, const std::string& artifact_dir);
    bool host_gemm() const { return host_gemm_; }
    bool ready() const { return ready_; }
    const char* last_error() const { return err_.c_str(); }

    // Clear the KV cache (start a new sequence).
    void reset();

    // Grow the KV cache by n_tokens empty positions, in the exact per-layer
    // pos-major [NKV][HD] float layout attention() appends (2 x NKV x HD x NL
    // floats per token).  This is the "extend context" path of the capacity
    // probe: the same KV cache the real forward grows during step(), extended
    // without running the GEMM.  Returns false if the forward isn't ready.
    bool extend_context(int n_tokens);

    // Run one token at absolute position pos; fills logits (size NV).
    bool step(int token, int pos, std::vector<float>& logits);

    // Batched prefill: run the whole prompt in one pass so each projection's
    // weight stream is read once per batch instead of once per token. The kernels
    // are M=128 wide, so a per-token step leaves ~99% of the array idle. Only the
    // simple dense path is batched; every other architecture falls back to
    // per-token step() (see the guard in the implementation).
    bool step_batch(const std::vector<int>& tokens, int start_pos,
                    std::vector<float>& logits);

    // Rows per batched launch. The activation BO holds exactly MD rows and the
    // per-op kernels are built M=MD wide, so a batch must not exceed this (the
    // batched quantizer refuses batches wider than MD rather than corrupting).
    int max_batch_rows() const;

    // Which artifacts were used, for reporting.
    const std::string& backend() const { return backend_; }

private:
    bool load_norms();
    bool init_impl(xrt::device* dev, const char* model_path,
                   const std::string& artifact_dir);
    bool host_gemm_ = false;
    bool load_layer(int l);
    void npu_gemm(I8Ctx& ctx, const float* A, int K, int N,
                  const float* Bmat, std::vector<float>& out, int wkey = -1);
    // Batched (M>1) GEMM: M rows in one launch, with per-row activation scales.
    void npu_gemm_rows(I8Ctx& ctx, const float* A, int M, int K, int N,
                       const float* Bmat, std::vector<float>& out, int wkey = -1);
    // True when this layer's projection `wkey` is already packed in its own BO,
    // so the caller can skip both the Q4NX dequant (load_layer) and the f32->B
    // assembly for the rest of the run.
    bool weight_cached(int key) const {
        return weight_cache_ && cur_layer_ >= 0 && cur_layer_ < (int)wcache_.size() &&
               wcache_[(size_t)cur_layer_].find(key) != wcache_[(size_t)cur_layer_].end();
    }
    bool attention(const std::vector<float>& qkv, int pos, std::vector<float>& ctx,
                   int head_dim, int n_kv_heads, float theta, int rotary_dim,
                   bool proportional, int window, int kv_layer, bool append_kv);
    bool mlp(const std::vector<float>& gu, std::vector<float>& h, int im);
    float rmsnorm(const float* x, const float* w, int n, float* out) const;
    float rmsnorm_noweight(const float* x, int n, float* out) const;
    bool run_lm_head(const std::vector<float>& h, std::vector<float>& logits);

    Q4nxForwardConfig c_;
    ModelWeights* mw_ = nullptr;
    xrt::device* dev_ = nullptr;

    // Architecture semantics derived from the model's own config.json.  Models
    // whose config asks for layer types this forward does not implement (hybrid
    // linear/sliding attention, per-layer-type RoPE, mRoPE) are REFUSED in
    // init() rather than silently run with the wrong math.
    Q4nxSemantics sem_;

    // Design set DERIVED from c_ (no model-specific geometry constant): four
    // kernels for fused gate/up, five when gate and up are split.
    Q4nxDesignGeom designs_[Q4NX_MAX_DESIGNS];
    int ndesigns_ = 0;
    std::unique_ptr<I8Ctx> ctx_[Q4NX_MAX_DESIGNS];
    // Index of the design playing a role, or -1 when the model has no such
    // kernel (e.g. GU is absent on split models, G/U absent on fused ones).
    int idx_of(Q4nxDesign role) const;

    // BF16 host-side tensors
    std::vector<uint16_t> embed_;                 // [NV*H] bf16
    // Some models store the embedding QUANTIZED (Gemma4-E2B/E4B: dtype I8), and
    // reading it as bf16 yields ~zero - which invalidated the whole forward and
    // the tied LM head (all-zero logits).  When the dtype is not BF16 the
    // embedding is dequantized once into embed_f_ and read from there.
    std::vector<float> embed_f_;
    bool embed_f32_ = false;
    float embed_at(int token, int k) const;
    // A model with tie_word_embeddings = false (Nanbeige4.1-3B) has a SEPARATE LM
    // head; decoding its logits from the embedding gives a different
    // distribution entirely, so the real head is dequantized once into lm_head_f_.
    std::vector<float> lm_head_f_;
    // The NPU LM head wants the head transposed into per-tile [H][T] blocks. That
    // gather transposes the whole NV x H matrix with a stride of T floats, so it
    // is built once here instead of on every token (it dominated decode time).
    std::vector<float> lm_head_tiles_;
    int                lm_head_tiles_T_ = 0;
    float lm_at(int n, int k) const;
    std::vector<std::vector<float>> in_norm_, post_norm_;   // [NL][H]
    std::vector<std::vector<float>> q_norm_, k_norm_;       // [NL][HD]
    std::vector<std::vector<float>> q_bias_, k_bias_, v_bias_;  // [NL][NH*HD]/[NL][NKV*HD] attention biases
    std::vector<float> rope_scale_;  // LongRoPE short_factor (per-half-dim), empty when none
    // MiniCPM4 scales (config "embedding_scale" / "residual_scale"); 1.0 for
    // every other architecture.  MiniCPM4 scales the embedding by 12 and each
    // residual branch by scale_depth/sqrt(NL) (0.2475 for 32 layers); skipping
    // them sends the hidden state to the wrong magnitude and the logits explode.
    float embed_scale_ = 1.0f;
    float residual_scale_ = 1.0f;
    // MiniCPM4 logit_scale (config "logit_scale", 16.0 for GLM's sibling
    // MiniCPM4).  The final logits are DIVIDED by it.  1.0 = no-op everywhere
    // else.  A constant positive divisor cannot move an argmax, so this only
    // matters for the probability distribution (sampling/temperature).
    float logit_scale_ = 1.0f;

    // ---- MLA + MoE (deepseek2 / GLM-4.7-Flash) ----
    // Set from config.json in init_impl; mla_ == 0 leaves every path below
    // untouched.  Each layer's dequantized weights are held here and refilled by
    // load_layer(), sized from the model's own dims.
    int mla_ = 0;
    int q_lora_ = 0, kv_lora_ = 0, rope_dim_ = 0, qk_nope_ = 0, v_head_ = 0;
    int n_expert_ = 0, top_k_ = 0, expert_im_ = 0, dense_im_ = 0;
    int leading_dense_ = 0;
    int expert_gating_sigmoid_ = 1;   // llama.cpp deepseek2: 2 = sigmoid, 1 = softmax
    float expert_w_scale_ = 1.0f;     // applied after the top-k weight normalisation
    std::vector<float> w_qa_, w_qb_, w_kva_, w_o_;      // per layer, [out][in]
    std::vector<float> mla_qan_, mla_kvan_;             // q_a / kv_a layernorms
    std::vector<float> w_kb_, w_vb_;                    // [NH][N][K], N padded for k_b
    int kb_rows_ = 0;                                   // k_b's padded row count (256)
    std::vector<std::vector<float>> mla_kn_, mla_kr_, mla_v_;  // per-layer MLA caches
    std::vector<float> moe_router_, moe_router_b_, share_g_, share_u_, share_d_, share_router_;
    std::vector<float> dense_g_, dense_u_, dense_d_;    // leading dense layer only
    // Routed experts stay packed until selected: per layer the Q4NX pools them as
    // [n_expert][tiles][5120].
    const uint8_t* exp_g_ = nullptr; const uint8_t* exp_u_ = nullptr; const uint8_t* exp_d_ = nullptr;
    size_t exp_g_bytes_ = 0, exp_u_bytes_ = 0, exp_d_bytes_ = 0;

    bool mla_attn(int l, const std::vector<float>& x, std::vector<float>& o, int pos);
    bool moe_mlp(int l, const std::vector<float>& xn, std::vector<float>& d);
    bool dense_mlp(int l, const std::vector<float>& xn, std::vector<float>& d);
    bool mla_rope(std::vector<float>& q, int NH, int qh, int nope, std::vector<float>& kr, int pos);
    bool mla_expert(const uint8_t* pool, size_t per_expert, int e, int N, int K,
                    std::vector<float>& out);
    std::vector<float> final_norm_;                         // [H]

    // Per-layer dequantized projections (reused each layer, chunked by design)
    std::vector<float> wq_, wk_, wv_, wo_, wg_, wu_, wd_;

    // KV cache: [layer] -> pos-major [max_pos][NKV][HD]
    std::vector<std::vector<float>> kcache_, vcache_;

    std::vector<float> B_;        // packed weight scratch  (max K*N floats)
    // Packed-int8 weight cache, one weight BO per (layer, wkey), packed on first
    // use.  Without it every token re-dequantizes the Q4NX (load_layer) AND
    // re-quantizes it (I8Ctx::packB) for every GEMM -- ~2 min/token on a 7B and
    // ~8 min/token on the 8B, which makes a served answer impractical.  Only
    // token-INVARIANT weights may be cached (dense projections, GDN, MLA, the
    // shared expert); routed-expert weights vary per token and pass wkey < 0.
    struct PackedWeight {
        std::unique_ptr<xrt::bo> bo;
        std::vector<float>       scales;
    };
    std::vector<std::unordered_map<int, PackedWeight>> wcache_;   // [layer][wkey]
    bool weight_cache_ = true;    // NPU_NO_WCACHE=1 disables (A/B)
    int cur_layer_ = 0;           // layer currently being processed

    // ── Gemma4 (per-layer embeddings + KV sharing + extra norms) ──
    // The per-layer token table is I8 [vocab, NL*ple_dim] = e.g. 262144 x 8960,
    // i.e. 9.4 GB if dequantized whole, so it is dequantized in 32-row blocks and
    // cached (ple_block_, starting at ple_block_row_).
    std::vector<float> ple_block_;      // [32 * NL*ple_dim] floats
    int ple_block_row_ = -1;
    std::vector<float> ple_norm_;       // per_layer_proj_norm  [ple_dim]
    std::vector<float> ple_model_proj_; // per_layer_model_proj [NL*ple_dim][H]
    std::vector<float> ple_input_;      // this token's per-layer inputs [NL][ple_dim]
    std::vector<int>    kv_owner_;      // per layer: layer whose K/V it reuses (itself)
    // per-layer Gemma4 tensors
    std::vector<std::vector<float>> pre_ff_norm_, post_ff_norm_, post_ln_;  // [H]
    std::vector<float> layer_scale_;                                        // [NL]
    std::vector<std::vector<float>> inp_gate_, ple_proj_;  // [ple][H], [H][ple]
    bool gemma4_layer_ = false;
    int  ple_dim_ = 0;
    bool load_ple_block(int row);
    void build_ple_inputs(int token);
    /* Per-layer attention geometry: Gemma4's full_attention layers use
     * global_head_dim (512) where its sliding layers use head_dim (256). */
    void layer_attn_dims(int l, int& hd, int& nkv) const;

    // ── GDN linear attention (qwen3_5 / Qwen3.5-4B) ──────────────────────
    // Derived from the config's linear_* dims; lin_kd_ == 0 when the model has
    // no linear_attention layers.  The fused q|k|v and the z projection run on
    // the NPU through the derived QKVLIN / Z designs; alpha/beta (N = nvh < 128)
    // and the conv are host-side; the recurrence is host-side like attention.
    // Layout and maths: docs/qwen35-4b-weight-format.md.
    int lin_kd_ = 0, lin_vd_ = 0;        // key_dim = nkh*khd, value_dim = nvh*vhd
    int attn_gate_ = 0;                   // qwen3_5 attn_output_gate
    int lin_nkh_ = 0, lin_nvh_ = 0;      // head counts
    int lin_khd_ = 0, lin_vhd_ = 0;      // head dims
    int lin_ck_ = 4;                     // depthwise conv kernel
    std::vector<float> wqkvlin_, wz_, woutlin_;          // [CD][H], [VD][H], [H][VD]
    std::vector<float> wa_, wb_;                          // [nvh][H] alpha/beta
    std::vector<float> lin_conv_w_, lin_norm_w_, lin_ssm_a_, lin_dt_;  // [CK][CD],[VHD],[nvh],[nvh]
    std::vector<std::vector<float>> conv_state_, rec_state_;  // [CD][CK-1], [nvh][khd][vhd]
    // One GDN token step: xn is the normed input; out gets the out_proj result.
    bool gdn_layer(int l, const std::vector<float>& xn, std::vector<float>& out);
    bool load_gdn_layer(int l);
    std::string err_;
    std::string backend_ = "uninitialised";
    bool ready_ = false;
};

#endif  // NPU_INFER_Q4NX_FORWARD_H
