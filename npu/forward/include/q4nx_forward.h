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

struct I8Ctx;

namespace xrt { class device; }

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

    // artifact_dir holds the full ELFs / xclbins; use_elf selects the backend.
    bool init(xrt::device& dev, const char* model_path, const std::string& artifact_dir,
              bool use_elf);
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

    // Which artifacts were used, for reporting.
    const std::string& backend() const { return backend_; }

private:
    bool load_norms();
    bool init_impl(xrt::device* dev, const char* model_path,
                   const std::string& artifact_dir, bool use_elf);
    bool host_gemm_ = false;
    bool load_layer(int l);
    void npu_gemm(I8Ctx& ctx, const float* A, int K, int N,
                  const float* Bmat, std::vector<float>& out);
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
    float lm_at(int n, int k) const;
    std::vector<std::vector<float>> in_norm_, post_norm_;   // [NL][H]
    std::vector<std::vector<float>> q_norm_, k_norm_;       // [NL][HD]
    std::vector<std::vector<float>> q_bias_, k_bias_, v_bias_;  // [NL][NH*HD]/[NL][NKV*HD] attention biases
    std::vector<float> final_norm_;                         // [H]

    // Per-layer dequantized projections (reused each layer, chunked by design)
    std::vector<float> wq_, wk_, wv_, wo_, wg_, wu_, wd_;

    // KV cache: [layer] -> pos-major [max_pos][NKV][HD]
    std::vector<std::vector<float>> kcache_, vcache_;

    std::vector<float> B_;        // packed weight scratch  (max K*N floats)
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
