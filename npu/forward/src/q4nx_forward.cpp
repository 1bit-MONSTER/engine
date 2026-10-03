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
// q4nx_forward.cpp — see q4nx_forward.h.
//
// The host does RMSNorm / RoPE / GQA attention / SiLU in float32; the GEMMs run
// on the NPU through I8Ctx.  The design set is DERIVED from the model's own
// architecture dims (q4nx_derive_designs): four kernels when gate/up are fused,
// five when they are split.  No model name or tag appears in this file; the
// full-ELF names are geometry-derived.
//
// Numerics follow tools/q4nx_forward_ref.py so the two can be compared
// element-wise.
#include "q4nx_forward.h"
#include <fcntl.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include <xrt/xrt_device.h>

#include "npu_engine_i8ctx_inc.h"   // I8Ctx
#include "q4nx_dequant.h"
#include "q4nx_pack.h"

#ifdef NPU_INFER_DEBUG_DUMPS
// Debug dumps to a path named by an environment variable (NPU_INFER_DUMP_ATTN, _LN/_L0,
// _HEAD): compiled only into debug builds (cmake -DONEBIT_NPU_DEBUG_DUMPS=ON), written
// owner-only (0600) and never through a symlink.
static FILE* open_debug_dump(const char* path) {
    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
    return fd >= 0 ? fdopen(fd, "wb") : nullptr;
}
#endif

static inline float bf16_to_f32(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// softplus with the same overflow guard the Python reference uses.
static inline float softplus_f(float x) {
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

// A tensor's data as n bf16 values, or null when the tensor holds fewer than n
// (config.json's dims decide n; the file decides the size).
static const uint16_t* bf16_data(ModelWeights* mw, const TensorDesc* d, size_t n) {
    if (!d || (size_t)d->data_size / 2 < n) return nullptr;
    return (const uint16_t*)model_tensor_data(mw, const_cast<TensorDesc*>(d));
}

// A tensor's data when it holds at least `bytes`, else null.
static const uint8_t* data_of(ModelWeights* mw, const TensorDesc* d, size_t bytes) {
    if (!d || (size_t)d->data_size < bytes) return nullptr;
    return (const uint8_t*)model_tensor_data(mw, const_cast<TensorDesc*>(d));
}

static bool file_exists(const std::string& p, long min_size = 1) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return sz >= min_size;
}

// Decode a quantized [rows, cols] tensor whatever chunk size it is stored in:
// 5120 = q4_1, 4736 = OFLM Q4_K (flm_dtype_t 8, Qwen3.5-4B's projections),
// 8704 = Q8 (Qwen3.5's tied table / ssm_alpha / ssm_beta).  The chunk size is
// derived from data_size rather than the declared shape so a 2-D [ntiles, N]
// and a 3-D [ntr, ntc, N] tensor both work.  (src/q4nx_dequant.c; the Q4_K/Q8
// layouts are in docs/qwen35-4b-weight-format.md.)
static int dequant_any(const void* data, size_t nbytes, int rows, int cols, float* out) {
    if (!data || !out || rows <= 0 || cols <= 0) return -1;
    if (rows % Q4NX_TILE_ROWS || cols % Q4NX_TILE_COLS) return -1;
    const size_t n = (size_t)(rows / Q4NX_TILE_ROWS) * (size_t)(cols / Q4NX_TILE_COLS);
    if (n == 0) return -1;
    const int cb = (int)(nbytes / n);
    if (cb == Q4NX_TILE_BYTES || cb == Q4NX_CHUNK_Q4K || cb == Q4NX_CHUNK_Q8)
        return q4nx_dequant_tensor_chunked((const uint8_t*)data, nbytes, rows, cols, cb, out);
    return q4nx_dequant_tensor((const uint8_t*)data, nbytes, rows, cols, out);
}

// LongRoPE short_factor from config.json's rope_scaling (MiniCPM4): the
// per-half-dim frequency multipliers.  Empty when the config declares none.
// Ported from the engine route's npu/forward (commit 599ac77) so both copies of
// the model-generic forward agree.
static std::vector<float> read_rope_short_factor(const std::string& cfg_path) {
    std::vector<float> out;
    std::ifstream f(cfg_path);
    if (!f) return out;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* p = strstr(s.c_str(), "short_factor");
    if (!p) return out;
    const char* b = strchr(p, '[');
    if (!b) return out;
    const char* e = strchr(b, ']');
    if (!e) return out;
    const char* q = b + 1;
    while (q < e) {
        char* end = nullptr;
        const float v = strtof(q, &end);
        if (end == q) break;
        out.push_back(v);
        q = end;
    }
    return out;
}

// A scalar from config.json (MiniCPM4's embedding_scale / residual_scale).
static float read_config_float(const std::string& cfg_path, const char* key, float dflt) {
    std::ifstream f(cfg_path);
    if (!f) return dflt;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* p = strstr(s.c_str(), key);
    if (!p) return dflt;
    const char* c = strchr(p, ':');
    if (!c) return dflt;
    return (float)strtod(c + 1, nullptr);
}

Q4nxNpuForward::Q4nxNpuForward() {}
Q4nxNpuForward::~Q4nxNpuForward() = default;

void Q4nxNpuForward::reset() {
    for (auto& v : kcache_) v.clear();
    for (auto& v : vcache_) v.clear();
}

bool Q4nxNpuForward::extend_context(int n_tokens) {
    if (!ready_ || n_tokens <= 0) return false;
    if (kcache_.empty() || vcache_.empty()) return false;
    const size_t per = (size_t)c_.NKV * (size_t)c_.HD;   // floats per layer per token
    for (int l = 0; l < c_.NL; l++) {
        kcache_[l].reserve(kcache_[l].size() + (size_t)n_tokens * per);
        vcache_[l].reserve(vcache_[l].size() + (size_t)n_tokens * per);
        kcache_[l].insert(kcache_[l].end(), (size_t)n_tokens * per, 0.0f);
        vcache_[l].insert(vcache_[l].end(), (size_t)n_tokens * per, 0.0f);
    }
    return true;
}

int Q4nxNpuForward::idx_of(Q4nxDesign role) const {
    for (int i = 0; i < ndesigns_; i++)
        if (designs_[i].role == role) return i;
    return -1;
}

static const char* role_tag(Q4nxDesign r) {
    switch (r) {
        case Q4NX_DESIGN_QKV: return "QKV";
        case Q4NX_DESIGN_QKV2: return "QKV2";
        case Q4NX_DESIGN_O:   return "O";
        case Q4NX_DESIGN_O2:  return "O2";
        case Q4NX_DESIGN_GU:  return "GU";
        case Q4NX_DESIGN_G:   return "G";
        case Q4NX_DESIGN_U:   return "U";
        case Q4NX_DESIGN_PLEGATE: return "PLEGATE";
        case Q4NX_DESIGN_PLEPROJ: return "PLEPROJ";
        case Q4NX_DESIGN_GU2: return "GU2";
        case Q4NX_DESIGN_D2:  return "D2";
        case Q4NX_DESIGN_QKVLIN: return "QKV";
        case Q4NX_DESIGN_Z:   return "Z";
        default:              return "D";
    }
}

// --------------------------------------------------------------------------
// init
// --------------------------------------------------------------------------
bool Q4nxNpuForward::init(xrt::device& dev, const char* model_path,
                          const std::string& dir) {
    return init_impl(&dev, model_path, dir);
}

bool Q4nxNpuForward::init_host(const char* model_path, const std::string& dir) {
    host_gemm_ = true;
    return init_impl(nullptr, model_path, dir);
}

bool Q4nxNpuForward::init_impl(xrt::device* dev, const char* model_path,
                               const std::string& dir) {
    if (getenv("NPU_INFER_HOST_GEMM")) host_gemm_ = true;
    if (getenv("NPU_NO_WCACHE")) weight_cache_ = false;   // A/B: disable the packed-weight cache
    dev_ = dev;

    // ---- config.json FIRST, then model_load ----
    // model_load allocates `layers` from the ModelConfig it is handed, so the
    // model's real depth has to be known before loading.  (Passing the built-in
    // 0.6B config made model_load allocate 28 layers while the rest of init used
    // the model's declared depth, so load_norms indexed past the end of the
    // array - a general-protection fault for Qwen3-VL-4B, which declares 36.)
    std::string mdir, cfg_path;
    {
        std::string mp(model_path);
        size_t slash = mp.find_last_of('/');
        mdir = slash == std::string::npos ? std::string(".") : mp.substr(0, slash);
        cfg_path = mdir + "/config.json";
    }
    ModelConfig cfg = QWEN3_0_6B_CONFIG;
    cfg.hidden_size = q4nx_config_int(cfg_path.c_str(), "hidden_size", cfg.hidden_size);
    cfg.num_layers = q4nx_config_int(cfg_path.c_str(), "num_hidden_layers", cfg.num_layers);
    cfg.num_attention_heads = q4nx_config_int(cfg_path.c_str(), "num_attention_heads", cfg.num_attention_heads);
    cfg.num_key_value_heads = q4nx_config_int(cfg_path.c_str(), "num_key_value_heads", cfg.num_key_value_heads);
    cfg.intermediate_size = q4nx_config_int(cfg_path.c_str(), "intermediate_size", cfg.intermediate_size);
    cfg.vocab_size = q4nx_config_int(cfg_path.c_str(), "vocab_size", cfg.vocab_size);
    {
        int hd = q4nx_config_int(cfg_path.c_str(), "head_dim", 0);
        if (hd <= 0 && cfg.num_attention_heads > 0)
            hd = cfg.hidden_size / cfg.num_attention_heads;
        if (hd > 0) cfg.head_dim = hd;
    }
    rope_scale_ = read_rope_short_factor(cfg_path);  // LongRoPE (MiniCPM4)
    embed_scale_ = read_config_float(cfg_path, "\"embedding_scale\"", 1.0f);
    residual_scale_ = read_config_float(cfg_path, "\"residual_scale\"", 1.0f);
    logit_scale_ = read_config_float(cfg_path, "\"logit_scale\"", 1.0f);
    if (logit_scale_ == 0.0f) logit_scale_ = 1.0f;
    mw_ = model_load(model_path, cfg);
    if (!mw_) { err_ = "model_load failed"; return false; }
    // model_load sized its layer table from the tensors; config.json may not promise more.
    if (cfg.num_layers <= 0 || cfg.num_layers > mw_->config.num_layers) {
        err_ = "config.json num_hidden_layers (" + std::to_string(cfg.num_layers) + ") does not match the " +
               std::to_string(mw_->config.num_layers) + " layers in model.q4nx";
        return false;
    }

    // forward dims mirror the config the weights were loaded with
    c_.H = cfg.hidden_size;
    c_.NL = cfg.num_layers;
    c_.NH = cfg.num_attention_heads;
    c_.NKV = cfg.num_key_value_heads;
    c_.HD = cfg.head_dim;
    c_.IM = cfg.intermediate_size;
    c_.NV = cfg.vocab_size;
    fprintf(stderr, "  [forward] dims from %s: H=%d NL=%d NH=%d NKV=%d HD=%d IM=%d NV=%d\n",
            cfg_path.c_str(), c_.H, c_.NL, c_.NH, c_.NKV, c_.HD, c_.IM, c_.NV);

    const int H = c_.H, NL = c_.NL, HD = c_.HD, NV = c_.NV;
    // Embedding: BF16 for most models, but Gemma4 stores it QUANTIZED (dtype I8).
    // Reading a quantized tensor as bf16 yields ~zero, which invalidated the whole
    // forward and the tied LM head (all-zero logits); dequantize it once instead.
    embed_.resize((size_t)NV * H);
    {
        /* A tied model may store the table ONLY as `lm_head.weight` (Qwen3.5-4B's
         * container has no `model.embed_tokens.weight`); it doubles as both the
         * token embedding and the head.  Reading the unset descriptor instead
         * memcpy'd 1.3 GB from a bogus offset and the whole forward went NaN. */
        TensorDesc* etd = &mw_->embed_tokens;
        if (etd->data_size == 0 || etd->name[0] == '\0') {
            TensorDesc* alt = model_tensor_by_name(mw_, "lm_head.weight");
            if (alt && alt->data_size) etd = alt;
        }
        const void* e = model_tensor_data(mw_, etd);
        if (!e || etd->data_size == 0) { err_ = "no embed_tokens/lm_head.weight"; return false; }
        const char* dt = etd->dtype;
        const size_t en = (size_t)NV * (size_t)H;
        if (dt && dt[0] && strncmp(dt, "BF16", 4) != 0) {
            embed_f_.resize(en);
            const size_t esz = (size_t)etd->data_size;
            int rc;
            if (esz == en) {
                /* Plain int8 with a per-32-column scale (Gemma4).  The tensor is
                 * stored PRE-SCALED by the architecture's embedding scale, so
                 * the forward applies none here; the LM head divides it back
                 * out (see run_lm_head). */
                TensorDesc* sd = model_tensor_by_name(mw_, "model.embed_tokens.weight.scale");
                const float* sc = sd ? (const float*)data_of(mw_, sd, en / 32 * sizeof(float)) : nullptr;
                if (sd && !sc) { err_ = "model.embed_tokens.weight.scale is too small"; return false; }
                rc = q4nx_dequant_plain_i8((const uint8_t*)e, sc, NV, H, embed_f_.data());
            } else {
                rc = dequant_any((const uint8_t*)e, esz, NV, H, embed_f_.data());
            }
            if (rc != 0) {
                err_ = std::string("embedding dtype ") + dt + " but dequant failed";
                return false;
            }
            embed_f32_ = true;
            fprintf(stderr, "  [forward] embedding dtype %s -> dequantized (%zu floats, %.2f GiB)\n",
                    dt, embed_f_.size(),
                    (double)embed_f_.size() * sizeof(float) / (1024.0 * 1024.0 * 1024.0));
        } else {
            if (etd->data_size < (size_t)NV * H * 2) {
                err_ = "embedding too small for NV*H bf16";
                return false;
            }
            memcpy(embed_.data(), e, (size_t)NV * H * sizeof(uint16_t));
        }
    }

    // ---- architecture semantics from the model's own config.json ----
    {
        std::string mp(model_path);
        size_t slash = mp.find_last_of('/');
        std::string mdir = slash == std::string::npos ? std::string(".") : mp.substr(0, slash);
        std::string cfg = mdir + "/config.json";
        if (q4nx_semantics_from_config(cfg.c_str(), mdir.c_str(), &sem_) != 0) {
            err_ = "cannot read " + cfg;
            return false;
        }
        if (sem_.eps <= 0) sem_.eps = c_.eps;
        q4nx_semantics_dump(&sem_);
        // Refuse rather than silently apply full-attention math to a model whose
        // config declares layer types (or RoPE variants) this forward lacks.
        // Refusal is now precise.  Only layers whose attention MATH we do not
        // have (linear_attention / SSM) and interleaved/multimodal RoPE are
        // refused.  Sliding-window layers and per-layer-type RoPE are executed:
        // a sliding window is a mask and per-layer-type RoPE is a per-layer
        // selection, both host-side, so Gemma4 needs no new kernel.
        const int n_linear = q4nx_semantics_linear_layers(&sem_);
        // GDN (qwen3_5 linear_attention) IS implemented: the fused q|k|v and the
        // z projection are derived from the config's linear_* dims and run through
        // the same full-ELF GEMM path; conv/alpha/beta/recurrence are host-side.
        // Text-only mRoPE reduces to the ordinary partial-RoPE angles (all three
        // position components are equal), so it is executed too.  Refuse only when
        // the linear geometry is underivable from the config.
        lin_kd_ = lin_vd_ = lin_nkh_ = lin_nvh_ = lin_khd_ = lin_vhd_ = 0;
        if (n_linear > 0) {
            lin_nkh_ = q4nx_config_int(cfg_path.c_str(), "linear_num_key_heads", 0);
            lin_nvh_ = q4nx_config_int(cfg_path.c_str(), "linear_num_value_heads", 0);
            lin_khd_ = q4nx_config_int(cfg_path.c_str(), "linear_key_head_dim", 0);
            lin_vhd_ = q4nx_config_int(cfg_path.c_str(), "linear_value_head_dim", 0);
            lin_ck_ = q4nx_config_int(cfg_path.c_str(), "linear_conv_kernel_dim", 4);
            lin_kd_ = lin_nkh_ * lin_khd_;
            lin_vd_ = lin_nvh_ * lin_vhd_;
            if (lin_kd_ <= 0 || lin_vd_ <= 0 || lin_ck_ < 1) {
                char mix[128];
                q4nx_semantics_layer_mix(&sem_, mix, sizeof(mix));
                char msg[384];
                snprintf(msg, sizeof(msg),
                         "unsupported attention layers (%s): %d layer(s) are "
                         "linear_attention but the config has no usable "
                         "linear_num_*/linear_*_head_dim dims",
                         mix, n_linear);
                err_ = msg;
                return false;
            }
            fprintf(stderr, "  [forward] GDN linear attention: %d layer(s), "
                    "key_dim=%d value_dim=%d heads=%d/%d hd=%d/%d conv_k=%d\n",
                    n_linear, lin_kd_, lin_vd_, lin_nkh_, lin_nvh_, lin_khd_,
                    lin_vhd_, lin_ck_);
        }
        attn_gate_ = q4nx_config_int(cfg_path.c_str(), "attn_output_gate", 0);
        if (attn_gate_) fprintf(stderr, "  [forward] attn_output_gate=1 (q_proj emits q|gate)\n");
        // The config's RoPE base supersedes the built-in default.
        c_.theta = sem_.rope_theta;
    }

    kcache_.assign(NL, {});
    vcache_.assign(NL, {});
    // Unconditional: this runs BEFORE the MLA detection below sets mla_, so
    // gating it on mla_ left the MLA caches empty and mla_attn segfaulted.
    mla_kn_.assign(NL, {}); mla_kr_.assign(NL, {}); mla_v_.assign(NL, {});

    // ---- derive the design set from the architecture dims ----
    // fused vs split gate/up is an artifact property, so probe for it rather
    // than keying on a model name: a GU artifact means fused, G/U means split.
    // Zero-initialised: the optionally-derived fields (ple_dim, double_wide_mlp,
    // NKV2/HD2, lin_kd/lin_vd, attn_output_gate) must be 0 when absent, not
    // garbage from the stack.
    /* MLA + MoE (deepseek2 / GLM-4.7-Flash).  A deepseek2 config's
     * head_dim/key_length describe the KV LATENT, not a per-head width, so the
     * MLA geometry comes from its own keys; then HD is re-pointed at the per-head
     * VALUE width so the standard O derivation gives K=NH*HD=NH*v_head, and IM at
     * the per-EXPERT intermediate so G/U/D ARE the expert MLP. */
    mla_ = q4nx_config_int(cfg_path.c_str(), "mla", 0);
    if (mla_) {
        q_lora_ = q4nx_config_int(cfg_path.c_str(), "q_lora_rank", 0);
        kv_lora_ = q4nx_config_int(cfg_path.c_str(), "kv_lora_rank", 0);
        rope_dim_ = q4nx_config_int(cfg_path.c_str(), "rope_dim", 0);
        qk_nope_ = q4nx_config_int(cfg_path.c_str(), "qk_nope_head_dim", 0);
        v_head_ = q4nx_config_int(cfg_path.c_str(), "v_head_dim", 0);
        if (q_lora_ <= 0 || kv_lora_ <= 0 || rope_dim_ <= 0 || qk_nope_ <= 0 ||
            v_head_ <= 0) {
            fprintf(stderr, "  [mla] config incomplete (q_lora=%d kv_lora=%d rope=%d nope=%d "
                            "v_head=%d)\n", q_lora_, kv_lora_, rope_dim_, qk_nope_, v_head_);
            err_ = "MLA config incomplete";
            return false;
        }
        // k_b's Q4NX slice is [kv_lora][nope] with the nope axis padded to the
        // 256-column block (192 -> 256), which is what its 16 tiles/head show.
        kb_rows_ = qk_nope_;   /* the re-converted Q4NX packs k_b at [192][512], 12 tiles/head (verified corr 0.996926 vs the GGUF) */
        c_.HD = v_head_;
        fprintf(stderr, "  [mla] MLA: q_lora=%d kv_lora=%d rope=%d nope=%d v_head=%d NH=%d\n",
                q_lora_, kv_lora_, rope_dim_, qk_nope_, v_head_, c_.NH);
    }
    // MoE params are parsed INDEPENDENTLY of mla_ so a non-MLA MoE arch (qwen3moe /
    // qwen35moe) gets its expert geometry too; reading them only inside `if (mla_)`
    // left n_expert_/top_k_/expert_im_ zero and the standard MoE path dead.
    // expert_im_ is the per-EXPERT intermediate, which c_.IM already holds for a MoE
    // config (the converter writes the per-expert FFN width as intermediate_size).
    n_expert_ = q4nx_config_int(cfg_path.c_str(), "n_expert", 0);
    if (n_expert_ > 0) {
        top_k_ = q4nx_config_int(cfg_path.c_str(), "top_k", 0);
        expert_im_ = c_.IM;
        dense_im_ = q4nx_config_int(cfg_path.c_str(), "dense_intermediate_size", 0);
        leading_dense_ = q4nx_config_int(cfg_path.c_str(), "leading_dense_block_count", 0);
        expert_gating_sigmoid_ =
            q4nx_config_int(cfg_path.c_str(), "expert_gating_func", 1) == 2;  // 2 = sigmoid
        expert_w_scale_ = read_config_float(cfg_path, "\"expert_weights_scale\"", 1.0f);
        if (top_k_ <= 0 || top_k_ > n_expert_ || expert_im_ <= 0) {
            fprintf(stderr, "  [moe] config incomplete (n_expert=%d top_k=%d im=%d)\n",
                    n_expert_, top_k_, expert_im_);
            err_ = "MoE config incomplete";
            return false;
        }
        fprintf(stderr, "  [moe] %sMoE: NH=%d n_expert=%d top%d im=%d dense_im=%d gating=%s\n",
                mla_ ? "MLA+" : "", c_.NH, n_expert_, top_k_, expert_im_, dense_im_,
                expert_gating_sigmoid_ ? "sigmoid" : "softmax");
    }

    // attention() shares each KV head across NH / NKV query heads (kh = h / (NH / NKV)), so the
    // KV heads must divide the query heads; Gemma4's global layers have their own KV count.
    if (!mla_) {
        auto bad_kv = [&](int nkv) {
            if (nkv > 0 && nkv <= c_.NH && c_.NH % nkv == 0) return false;
            err_ = "config.json has " + std::to_string(c_.NH) + " attention heads and " + std::to_string(nkv) +
                   " key/value heads; the key/value heads must divide the attention heads";
            return true;
        };
        if (bad_kv(c_.NKV)) return false;
        if (sem_.num_global_kv_heads > 0 && bad_kv(sem_.num_global_kv_heads)) return false;
    }

    Q4nxModelDims dims{};
    dims.mla = mla_;
    dims.q_lora = q_lora_; dims.kv_lora = kv_lora_; dims.rope_dim = rope_dim_;
    dims.qk_nope = qk_nope_; dims.dense_im = dense_im_;
    dims.H = c_.H; dims.NH = c_.NH; dims.NKV = c_.NKV; dims.HD = c_.HD;
    dims.IM = c_.IM; dims.M = 128;
    dims.ple_dim = sem_.hidden_size_per_layer_input;
    dims.double_wide_mlp = sem_.double_wide_mlp;
    /* A second attention geometry when the head dim varies by layer type. */
    dims.NKV2 = sem_.num_global_kv_heads;
    dims.HD2 = sem_.global_head_dim;
    /* GDN linear-attention projections (qwen3_5): fused q|k|v and z. */
    dims.lin_kd = lin_kd_;
    dims.lin_vd = lin_vd_;
    dims.attn_output_gate = attn_gate_;
    {
        char gu[512], g[512];
        snprintf(gu, sizeof(gu), "%s/full_i8_GU_K%d_N%d.elf", dir.c_str(), dims.H, 2 * dims.IM);
        snprintf(g, sizeof(g), "%s/full_i8_G_K%d_N%d.elf", dir.c_str(), dims.H, dims.IM);
        dims.fused_gu = file_exists(gu) || !file_exists(g) ? 1 : 0;
    }
    ndesigns_ = q4nx_derive_designs(&dims, designs_);
    if (ndesigns_ <= 0) { err_ = "design derivation failed"; return false; }

    // Batched expert gate/up (MoE): one GEMM per projection over ALL top-k experts
    // at once, instead of one per expert.  Derived here rather than in
    // q4nx_derive_designs because it depends on top_k, which the dims struct does
    // not carry.  The design's N is a G/U kernel width that holds top_k*IM; the
    // caller passes the real N = top_k*IM, so only those columns are packed.
    if (n_expert_ > 0 && top_k_ > 1 && expert_im_ > 0 && !mla_ && ndesigns_ + 2 <= Q4NX_MAX_DESIGNS) {
        const int need = top_k_ * expert_im_;
        const int nb = 8192;   // the only K=H G/U kernel width >= 4096 in the set
        char gb[512], ub[512];
        snprintf(gb, sizeof(gb), "%s/full_i8_G_K%d_N%d.elf", dir.c_str(), c_.H, nb);
        snprintf(ub, sizeof(ub), "%s/full_i8_U_K%d_N%d.elf", dir.c_str(), c_.H, nb);
        if (need <= nb && file_exists(gb) && file_exists(ub)) {
            memset(&designs_[ndesigns_], 0, sizeof(designs_[ndesigns_]));
            designs_[ndesigns_].M = dims.M; designs_[ndesigns_].K = c_.H; designs_[ndesigns_].N = nb;
            designs_[ndesigns_].role = Q4NX_DESIGN_GB;
            snprintf(designs_[ndesigns_].elf_name, Q4NX_ELF_NAME_MAX, "full_i8_G_K%d_N%d.elf", c_.H, nb);
            ndesigns_++;
            memset(&designs_[ndesigns_], 0, sizeof(designs_[ndesigns_]));
            designs_[ndesigns_].M = dims.M; designs_[ndesigns_].K = c_.H; designs_[ndesigns_].N = nb;
            designs_[ndesigns_].role = Q4NX_DESIGN_UB;
            snprintf(designs_[ndesigns_].elf_name, Q4NX_ELF_NAME_MAX, "full_i8_U_K%d_N%d.elf", c_.H, nb);
            ndesigns_++;
            fprintf(stderr, "  [forward] batched expert gate/up: N=%d (top_k=%d IM=%d) via K%d_N%d kernels\n",
                    need, top_k_, expert_im_, c_.H, nb);
        }
    }

    size_t maxB = 0;
    for (int i = 0; i < ndesigns_; i++)
        maxB = std::max(maxB, (size_t)designs_[i].K * (size_t)designs_[i].N);
    B_.resize(maxB);

    // Per-layer projection scratch, sized from the model's own dims (the earlier
    // version hardcoded 0.6B's NH*HD=2048, and the rewrite dropped the sizing
    // entirely, leaving these vectors empty: wq_.data() was null, so
    // q4nx_dequant_tensor's !out guard returned -1 and every step reported
    // "dequant failed").
    // Size the per-layer scratch for the WIDEST attention type (Gemma4's full
    // layers are 2x wider than its sliding ones) so one buffer serves both.
    const int max_hd = (sem_.global_head_dim > c_.HD) ? sem_.global_head_dim : c_.HD;
    const int max_nkv = (sem_.num_global_kv_heads > c_.NKV) ? sem_.num_global_kv_heads : c_.NKV;
    const size_t q_dim = (size_t)c_.NH * (size_t)max_hd;
    const size_t kv_dim = (size_t)max_nkv * (size_t)max_hd;
    wq_.resize(q_dim * (size_t)H * (attn_gate_ ? 2 : 1));   // q|gate doubles it
    wk_.resize(kv_dim * (size_t)H);
    wv_.resize(kv_dim * (size_t)H);
    wo_.resize((size_t)H * q_dim);
    wg_.resize((size_t)c_.IM * (size_t)H);
    wu_.resize((size_t)c_.IM * (size_t)H);
    wd_.resize((size_t)H * (size_t)c_.IM);

    // GDN scratch + per-layer recurrence state (qwen3_5 linear_attention).
    if (lin_kd_ > 0) {
        const size_t CD = 2 * (size_t)lin_kd_ + (size_t)lin_vd_;
        wqkvlin_.resize(CD * (size_t)H);
        wz_.resize((size_t)lin_vd_ * (size_t)H);
        woutlin_.resize((size_t)H * (size_t)lin_vd_);
        wa_.resize((size_t)lin_nvh_ * (size_t)H);
        wb_.resize((size_t)lin_nvh_ * (size_t)H);
        lin_conv_w_.resize((size_t)lin_ck_ * CD);
        lin_norm_w_.resize((size_t)lin_vhd_);
        lin_ssm_a_.resize((size_t)lin_nvh_);
        lin_dt_.resize((size_t)lin_nvh_);
        conv_state_.assign(NL, std::vector<float>((size_t)CD * (size_t)(lin_ck_ - 1), 0.0f));
        rec_state_.assign(NL, std::vector<float>((size_t)lin_nvh_ * (size_t)lin_khd_ * (size_t)lin_vhd_, 0.0f));
    }

    fprintf(stderr, "  [forward] derived %d designs (fused_gu=%d, H=%d NH=%d NKV=%d HD=%d IM=%d)\n",
            ndesigns_, dims.fused_gu, dims.H, dims.NH, dims.NKV, dims.HD, dims.IM);

    for (int i = 0; i < ndesigns_; i++) {
        const Q4nxDesignGeom* g = &designs_[i];
        ctx_[i] = std::make_unique<I8Ctx>();
        ctx_[i]->MD = g->M; ctx_[i]->KD = g->K; ctx_[i]->ND = g->N;

        if (host_gemm_) {
            // Offline mode: the geometry->artifact mapping is still asserted, but
            // no device is opened (the ELFs themselves are checked separately by
            // tests/verify_design_elfs + tests/verify_full_elfs.sh).
            std::string elf = dir + "/" + g->elf_name;
            if (!file_exists(elf)) { err_ = "missing full ELF " + elf; return false; }
            fprintf(stderr, "  [forward] %-3s M=%d K=%d N=%d  %s (host-GEMM)\n",
                    role_tag(g->role), g->M, g->K, g->N, elf.c_str());
        } else {
            std::string elf = dir + "/" + g->elf_name;
            if (!file_exists(elf)) {
                err_ = "missing full ELF " + elf;
                return false;
            }
            if (!ctx_[i]->init_elf(*dev, elf.c_str(), g->M, g->K, g->N, 1)) {
                err_ = "I8Ctx init_elf failed for " + elf;
                return false;
            }
            fprintf(stderr, "  [forward] %-3s M=%d K=%d N=%d  %s\n", role_tag(g->role),
                    g->M, g->K, g->N, elf.c_str());
        }
    }

    // ---- Gemma4 per-layer embeddings (PLE) + KV sharing ----
    // Every tensor here is read out of the model's own index (name + dtype checked
    // rather than assumed); the shapes are BF16 except the per-layer token table,
    // which is I8 [vocab, NL*ple_dim].
    ple_dim_ = sem_.hidden_size_per_layer_input;
    gemma4_layer_ = (ple_dim_ > 0) || sem_.extra_layer_norms;
    // A model with tie_word_embeddings = false ships a SEPARATE lm_head.weight
    // (packed I8); dequantize it once here.  Nanbeige4.1-3B is such a model and
    // its head is unrelated to the embedding (corr 0.013), so decoding logits
    // from the embedding returns '的' instead of 'Paris'.
    if (!sem_.tie_embeddings) {
        TensorDesc* ltd = model_tensor_by_name(mw_, "lm_head.weight");
        if (ltd) {
            lm_head_f_.assign((size_t)NV * (size_t)H, 0.0f);
            const uint8_t* lp = (const uint8_t*)model_tensor_data(mw_, ltd);
            if (!lp || dequant_any(lp, (size_t)ltd->data_size, NV, H,
                                   lm_head_f_.data()) != 0) {
                err_ = "lm_head dequant failed";
                return false;
            }
            fprintf(stderr, "  [forward] separate lm_head.weight -> dequantized "
                            "(%zu floats)\n", lm_head_f_.size());
        }
    }
    // Norms are loaded HERE, once the semantics and ple_dim_/gemma4_layer_ are
    // known: the Gemma4 extras (pre/post feedforward norms, post-PLE norm, layer
    // scalar) are only read when gemma4_layer_ is set, so loading them before the
    // config was parsed left those vectors empty and crashed the first step.
    if (!load_norms()) return false;
    if (ple_dim_ > 0) {
        const int H = c_.H, NL = c_.NL;
        auto read_named = [&](const char* nm, float* dst, int n) -> bool {
            TensorDesc* td = model_tensor_by_name(mw_, nm);
            if (!td || n < 0) return false;
            const uint16_t* p = bf16_data(mw_, td, (size_t)n);
            if (!p) return false;
            for (int i = 0; i < n; i++) dst[i] = bf16_to_f32(p[i]);
            return true;
        };

        ple_norm_.assign((size_t)ple_dim_, 0.0f);
        if (!read_named("model.per_layer_proj_norm.weight", ple_norm_.data(), ple_dim_)) {
            err_ = "missing model.per_layer_proj_norm.weight";
            return false;
        }
        // per_layer_model_proj.weight_layer{l} is [ple_dim][H], stored
        // ROW-GROUPED as [ple_dim/32, H, 32] (the converter's
        // reshape_matrix_to_block_matrix_for_mvm); stacked over the layers it is
        // the full [NL*ple_dim][H] context projection.
        ple_model_proj_.assign((size_t)NL * (size_t)ple_dim_ * (size_t)H, 0.0f);
        for (int l = 0; l < NL; l++) {
            char nm[160];
            snprintf(nm, sizeof(nm), "model.per_layer_model_proj.weight_layer%d", l);
            TensorDesc* td = model_tensor_by_name(mw_, nm);
            if (!td) { err_ = std::string("missing ") + nm; return false; }
            const uint16_t* p = bf16_data(mw_, td, (size_t)ple_dim_ * (size_t)H);
            if (!p || (ple_dim_ % 32) != 0) {
                err_ = std::string("bad ") + nm;
                return false;
            }
            float* dst = &ple_model_proj_[(size_t)l * ple_dim_ * H];
            for (int a = 0; a < ple_dim_ / 32; a++)
                for (int c = 0; c < H; c++)
                    for (int g = 0; g < 32; g++)
                        dst[(size_t)(a * 32 + g) * H + c] =
                            bf16_to_f32(p[((size_t)a * H + c) * 32 + g]);
        }
        ple_input_.assign((size_t)NL * (size_t)ple_dim_, 0.0f);

        // KV sharing: layers from (NL - num_kv_shared_layers) on carry no k/v
        // weights and reuse the K/V of the last non-shared layer of their own
        // attention type (HF's `store_full_length_kv`).
        const int first_shared = NL - sem_.num_kv_shared_layers;
        kv_owner_.assign((size_t)NL, 0);
        for (int l = 0; l < NL; l++) {
            if (l < first_shared || first_shared <= 0) { kv_owner_[l] = l; continue; }
            const int lt = (l < sem_.n_layer_types) ? sem_.layer_types[l] : Q4NX_LAYER_FULL;
            int owner = l;
            for (int p = first_shared - 1; p >= 0; p--) {
                const int pt = (p < sem_.n_layer_types) ? sem_.layer_types[p]
                                                       : Q4NX_LAYER_FULL;
                if (pt == lt) { owner = p; break; }
            }
            kv_owner_[l] = owner;
        }
        fprintf(stderr, "  [forward] gemma4 PLE: ple_dim=%d, %d layers KV-shared "
                        "(from layer %d), double_wide_mlp=%d\n",
                ple_dim_, sem_.num_kv_shared_layers, first_shared, sem_.double_wide_mlp);
    }

    backend_ = "full ELF";
    ready_ = true;
    return true;
}

float Q4nxNpuForward::embed_at(int token, int k) const {
    if (token < 0 || token >= c_.NV || k < 0 || k >= c_.H) return 0.0f;
    if (embed_f32_) return embed_f_[(size_t)token * (size_t)c_.H + (size_t)k];
    return bf16_to_f32(embed_[(size_t)token * (size_t)c_.H + (size_t)k]);
}

/* LM-head weight for output n, column k: the separate lm_head.weight when the
 * model does not tie it, otherwise the (pre-scaled) token embedding. */
float Q4nxNpuForward::lm_at(int n, int k) const {
    if (n < 0 || n >= c_.NV || k < 0 || k >= c_.H) return 0.0f;
    if (!lm_head_f_.empty())
        return lm_head_f_[(size_t)n * (size_t)c_.H + (size_t)k];
    return embed_at(n, k);
}

bool Q4nxNpuForward::load_norms() {
    const int H = c_.H, HD = c_.HD, NL = c_.NL;
    // Gemma4's q/k norms are per-head and sized by THAT layer type's head_dim
    // (256 sliding / 512 full), so every slot is sized for the widest type and
    // each layer reads only its own head_dim.
    const int max_hd = (sem_.global_head_dim > c_.HD) ? sem_.global_head_dim : c_.HD;
    in_norm_.assign(NL, std::vector<float>(H));
    post_norm_.assign(NL, std::vector<float>(H));
    q_norm_.assign(NL, std::vector<float>(max_hd));
    k_norm_.assign(NL, std::vector<float>(max_hd));
    const int max_nkv = (sem_.num_global_kv_heads > c_.NKV) ? sem_.num_global_kv_heads : c_.NKV;
    q_bias_.assign(NL, std::vector<float>((size_t)c_.NH * max_hd));
    k_bias_.assign(NL, std::vector<float>((size_t)max_nkv * max_hd));
    v_bias_.assign(NL, std::vector<float>((size_t)max_nkv * max_hd));
    final_norm_.assign(H, 0.0f);

    auto read_bf16 = [&](const TensorDesc* d, float* dst, int n) -> bool {
        if (n < 0) return false;
        // A tensor the model does not have (no name, no data) reads as zeros, as before
        // it read whatever sat at the start of the data; its slot is unused.
        if (d->name[0] == '\0' && d->data_size == 0) {
            std::fill(dst, dst + n, 0.0f);
            return true;
        }
        const uint16_t* p = bf16_data(mw_, d, (size_t)n);
        if (!p) {
            err_ = std::string("tensor ") + d->name + " is smaller than config.json's dims";
            return false;
        }
        for (int i = 0; i < n; i++) dst[i] = bf16_to_f32(p[i]);
        return true;
    };

    auto absent = [](const TensorDesc* d) {
        return d->name[0] == '\0' || d->data_size == 0;
    };
    for (int l = 0; l < NL; l++) {
        int hd_l = HD, nkv_l = c_.NKV;
        layer_attn_dims(l, hd_l, nkv_l);
        if (!read_bf16(&mw_->layers[l].input_layernorm_weight, in_norm_[l].data(), H)) return false;
        if (!read_bf16(&mw_->layers[l].post_attention_layernorm_weight, post_norm_[l].data(), H)) return false;
        if (!read_bf16(&mw_->layers[l].q_norm_weight, q_norm_[l].data(), hd_l)) return false;
        // Qwen2.5-style attention biases (q/k/v projection biases); absent on the
        // bias-free families, so a miss is not an error.  Loaded BEFORE the k_norm
        // continue below, or a k_norm-free model (Qwen2.5, Llama) would skip them.
        if (!absent(&mw_->layers[l].q_proj_bias))
            read_bf16(&mw_->layers[l].q_proj_bias, q_bias_[l].data(), (int)c_.NH * hd_l);
        if (!absent(&mw_->layers[l].k_proj_bias))
            read_bf16(&mw_->layers[l].k_proj_bias, k_bias_[l].data(), nkv_l * hd_l);
        if (!absent(&mw_->layers[l].v_proj_bias))
            read_bf16(&mw_->layers[l].v_proj_bias, v_bias_[l].data(), nkv_l * hd_l);
        // Gemma4 KV-shared layers carry no k_norm at all; that is structural, not
        // an error, so it must not fail the load.
        if (absent(&mw_->layers[l].k_norm_weight)) continue;
        if (!read_bf16(&mw_->layers[l].k_norm_weight, k_norm_[l].data(), hd_l)) return false;
    }
    if (!read_bf16(&mw_->norm_weight, final_norm_.data(), H)) return false;

    // ---- Gemma4 extras: two feedforward norms, the post-PLE norm, the layer
    // scalar, and the per-layer gate/projection weights (all BF16). ----
    if (gemma4_layer_) {
        pre_ff_norm_.assign(NL, std::vector<float>(H));
        post_ff_norm_.assign(NL, std::vector<float>(H));
        post_ln_.assign(NL, std::vector<float>(H));
        layer_scale_.assign(NL, 1.0f);
        inp_gate_.assign(NL, std::vector<float>());
        ple_proj_.assign(NL, std::vector<float>());
        for (int l = 0; l < NL; l++) {
            char nm[160];
            auto rd = [&](const char* fmt, float* dst, int n) -> bool {
                snprintf(nm, sizeof(nm), fmt, l);
                TensorDesc* td = model_tensor_by_name(mw_, nm);
                if (!td || n < 0) return false;
                const uint16_t* p = bf16_data(mw_, td, (size_t)n);
                if (!p) return false;
                for (int i = 0; i < n; i++) dst[i] = bf16_to_f32(p[i]);
                return true;
            };
            // The PLE gate / projection are stored ROW-GROUPED as
            // [rows/32, cols, 32] (the converter's
            // reshape_matrix_to_block_matrix_for_mvm), so a flat read would
            // scramble every row: logical(a*32+g, c) = stored(a, c, g).
            auto rd_grouped = [&](const char* fmt, float* dst, int rows, int cols) -> bool {
                snprintf(nm, sizeof(nm), fmt, l);
                TensorDesc* td = model_tensor_by_name(mw_, nm);
                if (!td || rows < 0 || cols < 0) return false;
                const uint16_t* p = bf16_data(mw_, td, (size_t)rows * (size_t)cols);
                if (!p || (rows % 32) != 0) return false;
                for (int a = 0; a < rows / 32; a++)
                    for (int c = 0; c < cols; c++)
                        for (int g = 0; g < 32; g++)
                            dst[(size_t)(a * 32 + g) * cols + c] =
                                bf16_to_f32(p[((size_t)a * cols + c) * 32 + g]);
                return true;
            };
            if (sem_.extra_layer_norms) {
                if (!rd("model.layers.%d.pre_feedforward_layernorm.weight",
                        pre_ff_norm_[l].data(), H)) return false;
                if (!rd("model.layers.%d.post_feedforward_layernorm.weight",
                        post_ff_norm_[l].data(), H)) return false;
                if (!rd("model.layers.%d.post_layernorm.weight", post_ln_[l].data(), H)) return false;
                if (!rd("model.layers.%d.layer_output_scale.weight", &layer_scale_[l], 1)) return false;
            }
            if (ple_dim_ > 0) {
                inp_gate_[l].assign((size_t)ple_dim_ * (size_t)H, 0.0f);
                ple_proj_[l].assign((size_t)H * (size_t)ple_dim_, 0.0f);
                if (!rd_grouped("model.layers.%d.inp_gate.weight", inp_gate_[l].data(),
                                ple_dim_, H)) return false;
                if (!rd_grouped("model.layers.%d.per_layer_projection.weight",
                                ple_proj_[l].data(), H, ple_dim_)) return false;
            }
        }
    }
    return true;
}

void Q4nxNpuForward::layer_attn_dims(int l, int& hd, int& nkv) const {
    const int lt = (l < sem_.n_layer_types) ? sem_.layer_types[l] : Q4NX_LAYER_FULL;
    if (lt != Q4NX_LAYER_SLIDING && sem_.global_head_dim > 0) {
        hd = sem_.global_head_dim;
        nkv = (sem_.num_global_kv_heads > 0) ? sem_.num_global_kv_heads : c_.NKV;
    } else {
        hd = c_.HD;
        nkv = c_.NKV;
    }
}

bool Q4nxNpuForward::load_layer(int l) {
    // MLA + MoE (deepseek2) has an entirely different tensor set.
    if (mla_) {
        const int H = c_.H, NH = c_.NH;
        const int qh = qk_nope_ + rope_dim_;
        const int kv_real = kv_lora_ + rope_dim_;
        char nm[160];
        auto td_of = [&](const char* fmt) -> TensorDesc* {
            snprintf(nm, sizeof(nm), fmt, l);
            return model_tensor_by_name(mw_, nm);
        };
        // A 2D packed tensor -> [rows][cols] floats.
        auto load2 = [&](const char* fmt, std::vector<float>& v, int rows, int cols) -> bool {
            TensorDesc* d = td_of(fmt);
            if (!d) { fprintf(stderr, "  [mla] missing " ); snprintf(nm,sizeof(nm),fmt,l); fprintf(stderr,"%s\n", nm); return false; }
            v.resize((size_t)rows * (size_t)cols);
            const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
            return raw && dequant_any(raw, (size_t)d->data_size, rows, cols, v.data()) == 0;
        };
        // A bf16 tensor (norms, router) is stored raw, not in 5120-byte tiles, so
        // it bypasses dequant_any.
        auto load_bf16 = [&](const char* fmt, std::vector<float>& v, int n) -> bool {
            TensorDesc* d = td_of(fmt);
            if (!d) { snprintf(nm, sizeof(nm), fmt, l); fprintf(stderr, "  [mla] missing %s\n", nm); return false; }
            const uint16_t* p = n < 0 ? nullptr : bf16_data(mw_, d, (size_t)n);
            if (!p) { snprintf(nm, sizeof(nm), fmt, l); fprintf(stderr, "  [mla] %s is too small\n", nm); return false; }
            v.resize((size_t)n);
            for (int i = 0; i < n; i++) v[i] = bf16_to_f32(p[i]);
            return true;
        };
        if (!load2("model.layers.%d.self_attn.q_a_proj.weight", w_qa_, q_lora_, H)) return false;
        if (!load2("model.layers.%d.self_attn.q_b_proj.weight", w_qb_, NH * qh, q_lora_)) return false;
        if (!load2("model.layers.%d.self_attn.kv_a_proj_with_mqa.weight", w_kva_, kv_real, H)) return false;
        if (!load2("model.layers.%d.self_attn.o_proj.weight", w_o_, H, NH * v_head_)) return false;
        if (!load_bf16("model.layers.%d.self_attn.q_a_layernorm.weight", mla_qan_, q_lora_)) return false;
        if (!load_bf16("model.layers.%d.self_attn.kv_a_layernorm.weight", mla_kvan_, kv_lora_)) return false;
        // k_b/v_b are pooled per head: [NH][tiles][5120].
        auto load_heads3 = [&](const char* fmt, std::vector<float>& v, int rows, int cols) -> bool {
            TensorDesc* d = td_of(fmt);
            if (!d) { snprintf(nm,sizeof(nm),fmt,l); fprintf(stderr, "  [mla] missing %s\n", nm); return false; }
            const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
            if (!raw) return false;
            const size_t per = (size_t)d->data_size / (size_t)NH;
            v.resize((size_t)NH * (size_t)rows * (size_t)cols);
            for (int h = 0; h < NH; h++)
                if (dequant_any(raw + (size_t)h * per, per, rows, cols,
                                &v[(size_t)h * (size_t)rows * (size_t)cols]) != 0) return false;
            return true;
        };
        if (!load_heads3("model.layers.%d.self_attn.k_b_proj.weight", w_kb_, kb_rows_, kv_lora_)) return false;
        // v_b's Q4NX grid is [kv_lora][v_head] = [512][256] (NOT [256][512]).  Pinned
        // empirically against llama.cpp's own tensor dump: reading it as a [512][256]
        // grid and indexing flat[k*vh + n] reproduces the oracle's layer-0 post-attention
        // sum to 5 decimals (-0.227843 vs -0.228092), where the [256][512] grid gave
        // -0.454611.  The tile GRID (not just the flat order) matters: [512][256] and
        // [256][512] have the same tile count but a different tile-to-element mapping.
        if (!load_heads3("model.layers.%d.self_attn.v_b_proj.weight", w_vb_, kv_lora_, v_head_)) return false;
        // Router + bias (bf16).
        // Router, routed experts and the shared expert exist ONLY on the MoE
        // layers: with leading_dense_block_count = 1 the leading dense layer(s)
        // carry a plain MLP instead (46 of GLM-4.7-Flash's 47 layers have them).
        if (l >= leading_dense_) {
        if (!load_bf16("model.layers.%d.moe_router.weight", moe_router_, n_expert_ * H)) return false;
        if (!load_bf16("model.layers.%d.moe_router.bias", moe_router_b_, n_expert_)) return false;
        // Routed experts stay packed until selected.  The Q4NX pools them per
        // layer as [n_expert][tiles][5120], so hold the pointer plus the
        // per-expert byte count and dequant one slice at a time.
        auto pool = [&](const char* fmt, const uint8_t** p, size_t* bytes) -> bool {
            TensorDesc* d = td_of(fmt);
            if (!d) { snprintf(nm,sizeof(nm),fmt,l); fprintf(stderr, "  [mla] missing %s\n", nm); return false; }
            const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
            if (!raw) return false;
            *p = raw;
            *bytes = (size_t)d->data_size / (size_t)n_expert_;
            return true;
        };
        if (!pool("model.layers.%d.mlp.gate_exps_proj.weight", &exp_g_, &exp_g_bytes_)) return false;
        if (!pool("model.layers.%d.mlp.up_exps_proj.weight", &exp_u_, &exp_u_bytes_)) return false;
        if (!pool("model.layers.%d.mlp.down_exps_proj.weight", &exp_d_, &exp_d_bytes_)) return false;
        // Shared expert ([tiles][5120], a plain 2D [im][H] / [H][im] matrix).
        if (!load2("model.layers.%d.mlp.share_gate_exps_proj.weight", share_g_, expert_im_, H)) return false;
        if (!load2("model.layers.%d.mlp.share_up_exps_proj.weight", share_u_, expert_im_, H)) return false;
        if (!load2("model.layers.%d.mlp.share_down_exps_proj.weight", share_d_, H, expert_im_)) return false;
        }   // end MoE-layer tensors
        // The leading dense layer has no experts.
        if (l < leading_dense_) {
            if (!load2("model.layers.%d.mlp.gate_proj.weight", dense_g_, dense_im_, H)) return false;
            if (!load2("model.layers.%d.mlp.up_proj.weight", dense_u_, dense_im_, H)) return false;
            if (!load2("model.layers.%d.mlp.down_proj.weight", dense_d_, H, dense_im_)) return false;
        }
        return true;
    }
    // A GDN (qwen3_5 linear_attention) layer has a different tensor set.
    if (lin_kd_ > 0 && l < sem_.n_layer_types &&
        sem_.layer_types[l] == Q4NX_LAYER_LINEAR)
        return load_gdn_layer(l);
    auto deq = [&](TensorDesc* d, float* out, int rows, int cols) -> bool {
        const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
        if (!raw) return false;
        return dequant_any(raw, (size_t)d->data_size, rows, cols, out) == 0;
    };
    const int H = c_.H, IM = c_.IM, NH = c_.NH;
    // Gemma4 varies the attention width by layer type, so the projection rows
    // come from THIS layer's head_dim (the scratch buffers are sized for the
    // widest type at init).
    int hd = 0, nkv = 0;
    layer_attn_dims(l, hd, nkv);
    LayerWeights* lw = &mw_->layers[l];
    // A Gemma4 KV-shared layer has no k/v weights and reuses another layer's cache.
    const bool kv_shared = (l < (int)kv_owner_.size() && kv_owner_[l] != l);
    // Gemma4 with use_double_wide_mlp runs a 2*IM-wide MLP on the shared layers.
    const bool wide = kv_shared && sem_.double_wide_mlp;
    const int im = wide ? 2 * IM : IM;
    if (wide) {
        wg_.resize((size_t)im * (size_t)H);
        wu_.resize((size_t)im * (size_t)H);
        wd_.resize((size_t)H * (size_t)im);
    }
    if (!deq(&lw->q_proj_weight, wq_.data(),
             NH * hd * (attn_gate_ ? 2 : 1), H)) return false;   // q|gate rows
    if (!kv_shared) {
        if (!deq(&lw->k_proj_weight, wk_.data(), nkv * hd, H)) return false;
        if (!deq(&lw->v_proj_weight, wv_.data(), nkv * hd, H)) return false;
    }
    if (!deq(&lw->o_proj_weight, wo_.data(), H, NH * hd)) return false;
    if (n_expert_ > 0) {
        // Qwen MoE layer (qwen3moe / qwen35moe): a router + top-k routed experts
        // instead of a dense gate/up/down.  The router is bf16; its bias is OPTIONAL
        // (Qwen MoE has none, deepseek2 does).  The experts stay pooled -- one
        // [N][K] slice is dequantized per selected expert at decode.
        char nm2[160];
        auto td2 = [&](const char* fmt) -> TensorDesc* {
            snprintf(nm2, sizeof(nm2), fmt, l);
            return model_tensor_by_name(mw_, nm2);
        };
        auto load_bf16_opt = [&](const char* fmt, std::vector<float>& v, int n) -> bool {
            TensorDesc* d = td2(fmt);
            if (!d || n < 0) return false;
            const uint16_t* p = bf16_data(mw_, d, (size_t)n);
            if (!p) return false;
            v.resize((size_t)n);
            for (int i = 0; i < n; i++) v[i] = bf16_to_f32(p[i]);
            return true;
        };
        if (!load_bf16_opt("model.layers.%d.moe_router.weight", moe_router_, n_expert_ * H))
            { err_ = "moe router"; return false; }
        if (!load_bf16_opt("model.layers.%d.moe_router.bias", moe_router_b_, n_expert_))
            moe_router_b_.clear();
        auto pool = [&](const char* fmt, const uint8_t** p, size_t* bytes) -> bool {
            TensorDesc* d = td2(fmt);
            if (!d) { fprintf(stderr, "  [moe] missing %s\n", nm2); return false; }
            const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
            if (!raw) return false;
            *p = raw; *bytes = (size_t)d->data_size / (size_t)n_expert_;
            return true;
        };
        if (!pool("model.layers.%d.mlp.gate_exps_proj.weight", &exp_g_, &exp_g_bytes_)) return false;
        if (!pool("model.layers.%d.mlp.up_exps_proj.weight", &exp_u_, &exp_u_bytes_)) return false;
        if (!pool("model.layers.%d.mlp.down_exps_proj.weight", &exp_d_, &exp_d_bytes_)) return false;
        // Optional shared expert (qwen35moe): a 2D [expert_im][H] / [H][expert_im] pair
        // plus its [H] gate.  Cleared when absent (qwen3moe has none).
        auto load_share = [&](const char* fmt, std::vector<float>& v, int rows, int cols) -> bool {
            TensorDesc* d = td2(fmt);
            if (!d) return false;
            v.resize((size_t)rows * (size_t)cols);
            const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
            return raw && dequant_any(raw, (size_t)d->data_size, rows, cols, v.data()) == 0;
        };
        if (!load_share("model.layers.%d.mlp.share_gate_exps_proj.weight", share_g_, expert_im_, H)) share_g_.clear();
        if (!load_share("model.layers.%d.mlp.share_up_exps_proj.weight", share_u_, expert_im_, H)) { share_g_.clear(); share_u_.clear(); }
        if (!load_share("model.layers.%d.mlp.share_down_exps_proj.weight", share_d_, H, expert_im_)) { share_g_.clear(); share_u_.clear(); }
        if (!load_bf16_opt("model.layers.%d.mlp.share_router.weight", share_router_, H)) share_router_.clear();
        return true;
    }
    if (!deq(&lw->gate_proj_weight, wg_.data(), im, H)) return false;
    if (!deq(&lw->up_proj_weight, wu_.data(), im, H)) return false;
    if (!deq(&lw->down_proj_weight, wd_.data(), H, im)) return false;
    return true;
}

// ── GDN (qwen3_5 linear_attention) ─────────────────────────────────────────
// Weight layouts are in docs/qwen35-4b-weight-format.md; the maths mirror
// tools/q4nx_forward_ref.py's forward_qwen35 (which predicts ' Paris').
bool Q4nxNpuForward::load_gdn_layer(int l) {
    const int H = c_.H;
    const int CD = 2 * lin_kd_ + lin_vd_;
    char nm[192];
    auto tensor = [&](const char* suffix) -> TensorDesc* {
        snprintf(nm, sizeof(nm), "model.layers.%d.%s", l, suffix);
        return model_tensor_by_name(mw_, nm);
    };
    auto deq = [&](const char* suffix, float* out, int rows, int cols) -> bool {
        TensorDesc* d = tensor(suffix);
        if (!d) { err_ = std::string("GDN: no tensor ") + nm; return false; }
        const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
        if (!raw) { err_ = std::string("GDN: no data ") + nm; return false; }
        if (dequant_any(raw, (size_t)d->data_size, rows, cols, out) != 0) {
            char m[64];
            snprintf(m, sizeof(m), " (rows=%d cols=%d bytes=%llu)", rows, cols,
                     (unsigned long long)d->data_size);
            err_ = std::string("GDN: dequant ") + nm + m;
            return false;
        }
        return true;
    };
    auto bf16v = [&](const char* suffix, float* out, int n) -> bool {
        TensorDesc* d = tensor(suffix);
        if (!d) { err_ = std::string("GDN: no tensor ") + nm; return false; }
        const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
        if (!raw || (size_t)d->data_size < (size_t)n * 2) {
            err_ = std::string("GDN: bad bf16 ") + nm;
            return false;
        }
        for (int i = 0; i < n; i++) out[i] = bf16_to_f32(((const uint16_t*)raw)[i]);
        return true;
    };
    if (!deq("linear_attn.qkv_proj.weight", wqkvlin_.data(), CD, H)) return false;
    if (!deq("self_attn.gate_proj.weight", wz_.data(), lin_vd_, H)) return false;
    if (!deq("linear_attn.ssm_out_proj.weight", woutlin_.data(), H, lin_vd_)) return false;
    if (!deq("linear_attn.ssm_alpha_proj.weight", wa_.data(), lin_nvh_, H)) return false;
    if (!deq("linear_attn.ssm_beta_proj.weight", wb_.data(), lin_nvh_, H)) return false;
    if (!bf16v("linear_attn.ssm_conv1d.weight", lin_conv_w_.data(), lin_ck_ * CD)) return false;
    if (!bf16v("linear_attn.ssm_norm.weight", lin_norm_w_.data(), lin_vhd_)) return false;
    {
        TensorDesc* da = tensor("linear_attn.ssm_a");
        TensorDesc* dd = tensor("linear_attn.ssm_dt.bias");
        if (!da || !dd) { err_ = std::string("GDN: no ssm_a/ssm_dt ") + nm; return false; }
        if ((size_t)da->data_size < (size_t)lin_nvh_ * 4 ||
            (size_t)dd->data_size < (size_t)lin_nvh_ * 4) {
            err_ = std::string("GDN: bad ssm_a/ssm_dt size ") + nm;
            return false;
        }
        memcpy(lin_ssm_a_.data(), model_tensor_data(mw_, da), (size_t)lin_nvh_ * 4);
        memcpy(lin_dt_.data(), model_tensor_data(mw_, dd), (size_t)lin_nvh_ * 4);
    }
    // The MLP runs on EVERY layer, but this path replaces load_layer()'s normal
    // tail for GDN layers -- without these three the MLP silently reused the
    // previous layer's (or uninitialised) weights, which is why the GDN
    // intermediates matched the reference exactly while the layer OUTPUT did not.
    LayerWeights* lw = &mw_->layers[l];
    auto deqT = [&](TensorDesc* d, float* out, int rows, int cols) -> bool {
        const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
        if (!raw || d->data_size == 0) { err_ = std::string("GDN: missing MLP tensor ") + d->name; return false; }
        if (dequant_any(raw, (size_t)d->data_size, rows, cols, out) != 0) {
            err_ = std::string("GDN: MLP dequant ") + d->name;
            return false;
        }
        return true;
    };
    if (n_expert_ > 0) {
        // qwen35moe: a GDN layer's FFN is MoE, not a dense gate/up/down.  The
        // loading mirrors the standard path's MoE branch; the FFN itself is run by
        // the common MLP section (which sees n_expert_ > 0 and calls moe_mlp).
        char nm2[160];
        auto td2 = [&](const char* fmt) -> TensorDesc* {
            snprintf(nm2, sizeof(nm2), fmt, l);
            return model_tensor_by_name(mw_, nm2);
        };
        auto load_bf16_opt = [&](const char* fmt, std::vector<float>& v, int n) -> bool {
            TensorDesc* d = td2(fmt);
            if (!d || n < 0) return false;
            const uint16_t* p = bf16_data(mw_, d, (size_t)n);
            if (!p) return false;
            v.resize((size_t)n);
            for (int i = 0; i < n; i++) v[i] = bf16_to_f32(p[i]);
            return true;
        };
        if (!load_bf16_opt("model.layers.%d.moe_router.weight", moe_router_, n_expert_ * H))
            { err_ = "moe router"; return false; }
        if (!load_bf16_opt("model.layers.%d.moe_router.bias", moe_router_b_, n_expert_))
            moe_router_b_.clear();
        auto pool = [&](const char* fmt, const uint8_t** p, size_t* bytes) -> bool {
            TensorDesc* d = td2(fmt);
            if (!d) { fprintf(stderr, "  [moe] missing %s\n", nm2); return false; }
            const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
            if (!raw) return false;
            *p = raw; *bytes = (size_t)d->data_size / (size_t)n_expert_;
            return true;
        };
        if (!pool("model.layers.%d.mlp.gate_exps_proj.weight", &exp_g_, &exp_g_bytes_)) return false;
        if (!pool("model.layers.%d.mlp.up_exps_proj.weight", &exp_u_, &exp_u_bytes_)) return false;
        if (!pool("model.layers.%d.mlp.down_exps_proj.weight", &exp_d_, &exp_d_bytes_)) return false;
        // Optional shared expert (qwen35moe): a 2D [expert_im][H] / [H][expert_im] pair
        // plus its [H] gate.  Cleared when absent.
        auto load_share = [&](const char* fmt, std::vector<float>& v, int rows, int cols) -> bool {
            TensorDesc* d = td2(fmt);
            if (!d) return false;
            v.resize((size_t)rows * (size_t)cols);
            const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
            return raw && dequant_any(raw, (size_t)d->data_size, rows, cols, v.data()) == 0;
        };
        if (!load_share("model.layers.%d.mlp.share_gate_exps_proj.weight", share_g_, expert_im_, H)) share_g_.clear();
        if (!load_share("model.layers.%d.mlp.share_up_exps_proj.weight", share_u_, expert_im_, H)) { share_g_.clear(); share_u_.clear(); }
        if (!load_share("model.layers.%d.mlp.share_down_exps_proj.weight", share_d_, H, expert_im_)) { share_g_.clear(); share_u_.clear(); }
        if (!load_bf16_opt("model.layers.%d.mlp.share_router.weight", share_router_, H)) share_router_.clear();
        return true;
    }
    if (!deqT(&lw->gate_proj_weight, wg_.data(), c_.IM, H)) return false;
    if (!deqT(&lw->up_proj_weight, wu_.data(), c_.IM, H)) return false;
    if (!deqT(&lw->down_proj_weight, wd_.data(), H, c_.IM)) return false;
    return true;
}

bool Q4nxNpuForward::gdn_layer(int l, const std::vector<float>& xn,
                               std::vector<float>& out) {
    const int H = c_.H;
    const int KD = lin_kd_, VD = lin_vd_, CD = 2 * lin_kd_ + lin_vd_;
    const int NKH = lin_nkh_, NVH = lin_nvh_, KHD = lin_khd_, VHD = lin_vhd_;
    const int CK = lin_ck_;
    const int i_qkv = idx_of(Q4NX_DESIGN_QKVLIN);
    const int i_z = idx_of(Q4NX_DESIGN_Z);
    const int i_o = idx_of(Q4NX_DESIGN_O);
    if (i_qkv < 0 || i_z < 0 || i_o < 0) {
        err_ = "GDN designs missing (QKVLIN/Z/O)";
        return false;
    }
    std::vector<float> qkv, z;
    {
        const Q4nxDesignGeom* g = &designs_[i_qkv];
        Q4nxProjections p{};
        p.ple = wqkvlin_.data();
        if (!weight_cached(0) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble gdn qkv"; return false; }
        npu_gemm(*ctx_[i_qkv], xn.data(), g->K, g->N, B_.data(), qkv, 0);
    }
    {
        const Q4nxDesignGeom* g = &designs_[i_z];
        Q4nxProjections p{};
        p.ple = wz_.data();
        if (!weight_cached(1) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble gdn z"; return false; }
        npu_gemm(*ctx_[i_z], xn.data(), g->K, g->N, B_.data(), z, 1);
    }
    // depthwise causal conv1d (kernel CK) + SiLU over CD channels;
    // full = [state (CK-1) | current], new state = full[1:]
    std::vector<float> co(CD);
    float* st = conv_state_[l].data();
    for (int c = 0; c < CD; c++) {
        const float* cs = st + (size_t)c * (CK - 1);
        float s = 0;
        for (int j = 0; j < CK - 1; j++) s += cs[j] * lin_conv_w_[(size_t)j * CD + c];
        s += qkv[c] * lin_conv_w_[(size_t)(CK - 1) * CD + c];
        co[c] = s / (1.0f + expf(-s));
    }
    for (int c = 0; c < CD; c++) {
        float* cs = st + (size_t)c * (CK - 1);
        for (int j = 0; j + 1 < CK - 1; j++) cs[j] = cs[j + 1];
        if (CK > 1) cs[CK - 2] = qkv[c];
    }
    // alpha/beta on the host (N = NVH < 128, below the GEMM tile width)
    std::vector<float> ah(NVH), bh(NVH);
    for (int h = 0; h < NVH; h++) {
        const float* ra = wa_.data() + (size_t)h * H;
        const float* rb = wb_.data() + (size_t)h * H;
        float sa = 0, sb = 0;
        for (int i = 0; i < H; i++) { sa += xn[i] * ra[i]; sb += xn[i] * rb[i]; }
        ah[h] = sa;
        bh[h] = sb;
    }
    // q/k L2-normalised with NVH/NKH head repetition; v per v-head
    const int rep = NVH / NKH;
    std::vector<float> ql((size_t)NVH * KHD), kl((size_t)NVH * KHD), vv((size_t)NVH * VHD);
    for (int h = 0; h < NVH; h++) {
        const int src = h / rep;
        for (int d = 0; d < KHD; d++) ql[(size_t)h * KHD + d] = co[(size_t)src * KHD + d];
        for (int d = 0; d < KHD; d++) kl[(size_t)h * KHD + d] = co[KD + (size_t)src * KHD + d];
        for (int d = 0; d < VHD; d++) vv[(size_t)h * VHD + d] = co[2 * KD + (size_t)h * VHD + d];
    }
    for (int h = 0; h < NVH; h++) {
        float sq = 0, sk = 0;
        for (int d = 0; d < KHD; d++) { sq += ql[(size_t)h*KHD+d]*ql[(size_t)h*KHD+d]; sk += kl[(size_t)h*KHD+d]*kl[(size_t)h*KHD+d]; }
        const float iq = 1.0f / sqrtf(sq + 1e-6f), ik = 1.0f / sqrtf(sk + 1e-6f);
        for (int d = 0; d < KHD; d++) { ql[(size_t)h*KHD+d] *= iq; kl[(size_t)h*KHD+d] *= ik; }
    }
    // recurrent gated delta rule, then scale-free gated RMSNorm, then out_proj
    float* rs = rec_state_[l].data();
    const float scale = 1.0f / sqrtf((float)KHD);
    std::vector<float> core((size_t)NVH * VHD), kvm(VHD);
    for (int h = 0; h < NVH; h++) {
        const float eg = expf(lin_ssm_a_[h] * softplus_f(ah[h] + lin_dt_[h]));
        const float beta = 1.0f / (1.0f + expf(-bh[h]));
        float* sh = rs + (size_t)h * KHD * VHD;
        const float* kh = &kl[(size_t)h * KHD];
        const float* qh = &ql[(size_t)h * KHD];
        const float* vh = &vv[(size_t)h * VHD];
        for (int i = 0; i < KHD * VHD; i++) sh[i] *= eg;
        for (int vd = 0; vd < VHD; vd++) {
            float s = 0;
            for (int kd = 0; kd < KHD; kd++) s += sh[kd * VHD + vd] * kh[kd];
            kvm[vd] = s;
        }
        for (int vd = 0; vd < VHD; vd++) {
            const float delta = (vh[vd] - kvm[vd]) * beta;
            for (int kd = 0; kd < KHD; kd++) sh[kd * VHD + vd] += kh[kd] * delta;
        }
        float ss = 0;
        for (int vd = 0; vd < VHD; vd++) {
            float s = 0;
            for (int kd = 0; kd < KHD; kd++) s += sh[kd * VHD + vd] * qh[kd];
            core[(size_t)h * VHD + vd] = s * scale;
            ss += core[(size_t)h * VHD + vd] * core[(size_t)h * VHD + vd];
        }
        const float inv = 1.0f / sqrtf(ss / (float)VHD + sem_.eps);
        for (int vd = 0; vd < VHD; vd++) {
            const float zz = z[(size_t)h * VHD + vd];
            core[(size_t)h * VHD + vd] = core[(size_t)h * VHD + vd] * inv *
                lin_norm_w_[vd] * (zz / (1.0f + expf(-zz)));
        }
    }
    if (getenv("NPU_INFER_DUMP_LAYERS")) {
        auto dmp = [&](const char* nm, const float* v, int n) {
            double s = 0;
            float mx = 0;
            for (int i = 0; i < n; i++) { float a = fabsf(v[i]); s += a; if (a > mx) mx = a; }
            fprintf(stderr, "  [G] l=%d %-5s %.6f %.4f\n", l, nm, s / n, mx);
        };
        dmp("qkv", qkv.data(), CD);
        dmp("co", co.data(), CD);
        dmp("z", z.data(), VD);
        dmp("a", ah.data(), NVH);
        dmp("b", bh.data(), NVH);
        dmp("core", core.data(), (int)core.size());
    }
    {
        const Q4nxDesignGeom* g = &designs_[i_o];
        Q4nxProjections p{};
        p.o = woutlin_.data();
        if (!weight_cached(2) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble gdn out"; return false; }
        npu_gemm(*ctx_[i_o], core.data(), g->K, g->N, B_.data(), out, 2);
    }
    return true;
}

// --------------------------------------------------------------------------
// Gemma4 per-layer embeddings (PLE)
// --------------------------------------------------------------------------
// The per-layer token table is I8 [vocab, NL*ple_dim] = 262144 x 8960 for
// Gemma4-E2B, i.e. 9.4 GB if dequantized whole.  The packed grid is row-major
// over 32-row bands (each band of a [32, cols] tensor is a contiguous run of
// cols/256 tiles), so exactly the band holding the requested row is dequantized
// and cached; row-within-band is then a plain offset.
bool Q4nxNpuForward::load_ple_block(int row) {
    const int block = row / Q4NX_TILE_ROWS;
    if (block == ple_block_row_) return true;
    TensorDesc* td = model_tensor_by_name(mw_, "model.per_layer_token_embd.weight");
    if (!td) { err_ = "missing model.per_layer_token_embd.weight"; return false; }
    const int cols = c_.NL * ple_dim_;
    const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, td);
    if (!raw || block < 0) { err_ = "per_layer_token_embd has no data"; return false; }
    ple_block_.assign((size_t)Q4NX_TILE_ROWS * (size_t)cols, 0.0f);
    const size_t plain_bytes =
        (size_t)sem_.vocab_size_per_layer_input * (size_t)cols;
    if ((size_t)td->data_size == plain_bytes) {
        // PLAIN int8 + per-32-column scale (Gemma4): only the requested band of
        // 32 rows is needed, so it is dequantized in place rather than the whole
        // [vocab, NL*ple_dim] table.
        TensorDesc* sd =
            model_tensor_by_name(mw_, "model.per_layer_token_embd.weight.scale");
        // the band and its scales must lie inside their tensors
        if (((size_t)block + 1) * Q4NX_TILE_ROWS * (size_t)cols > plain_bytes) {
            err_ = "per_layer_token_embd row out of range";
            return false;
        }
        const float* sc = sd ? (const float*)data_of(mw_, sd, ((size_t)block + 1) * Q4NX_TILE_ROWS *
                                                                  (size_t)(cols / 32) * sizeof(float))
                             : nullptr;
        if (sd && !sc) { err_ = "per_layer_token_embd scale is too small"; return false; }
        const uint8_t* band = raw + (size_t)block * Q4NX_TILE_ROWS * (size_t)cols;
        const float* band_sc =
            sc ? sc + (size_t)block * Q4NX_TILE_ROWS * (size_t)(cols / 32) : nullptr;
        if (q4nx_dequant_plain_i8(band, band_sc, Q4NX_TILE_ROWS, cols,
                                  ple_block_.data()) != 0) {
            err_ = "per_layer_token_embd dequant failed";
            return false;
        }
    } else {
        const size_t band = q4nx_tensor_bytes(Q4NX_TILE_ROWS, cols);
        if (band == 0 || ((size_t)block + 1) * band > (size_t)td->data_size) {
            err_ = "per_layer_token_embd row out of range";
            return false;
        }
        if (q4nx_dequant_tensor(raw + (size_t)block * band, band, Q4NX_TILE_ROWS,
                                cols, ple_block_.data()) != 0) {
            err_ = "per_layer_token_embd dequant failed";
            return false;
        }
    }
    ple_block_row_ = block;
    return true;
}

// per_layer_input[l] = (context[l] + token_identity[l]) * (1/sqrt(2)) where
//   token_identity[l] = table[token][l*P:(l+1)*P] * sqrt(P)
//   context[l]        = rmsnorm( (emb @ W_l) * H**-0.5, per_layer_proj_norm )
// i.e. project -> scale by 1/sqrt(hidden) -> normalise -> combine.
void Q4nxNpuForward::build_ple_inputs(int token) {
    const int H = c_.H, NL = c_.NL, P = ple_dim_;
    if (P <= 0) return;
    if (!load_ple_block(token)) return;
    const int r = token % Q4NX_TILE_ROWS;
    const float* tok_row = &ple_block_[(size_t)r * (size_t)(NL * P)];
    // The stored table is already scaled by sqrt(P) (the converter bakes it in),
    // so no extra scale is applied here.
    const float tok_scale = 1.0f;
    const float inv_sqrt2 = 0.70710678118654752f;

    std::vector<float> emb((size_t)H);
    for (int k = 0; k < H; k++) emb[k] = embed_at(token, k);

    for (int l = 0; l < NL; l++) {
        const float* W = &ple_model_proj_[(size_t)l * P * H];   // [P][H]
        float* dst = &ple_input_[(size_t)l * P];
        for (int o = 0; o < P; o++) {
            const float* wr = &W[(size_t)o * H];
            double acc = 0.0;
            for (int k = 0; k < H; k++) acc += (double)wr[k] * (double)emb[k];
            dst[o] = (float)acc * (1.0f / std::sqrt((float)H));
        }
        rmsnorm(dst, ple_norm_.data(), P, dst);
        for (int i = 0; i < P; i++) {
            const float ti = tok_row[(size_t)l * P + i] * tok_scale;
            dst[i] = (dst[i] + ti) * inv_sqrt2;
        }
    }
}

// --------------------------------------------------------------------------
// one NPU GEMM: C[1,N] = A[1,K] x B[K,N]  (row-major B)
// --------------------------------------------------------------------------
void Q4nxNpuForward::npu_gemm(I8Ctx& ctx, const float* A, int K, int N,
                              const float* Bmat, std::vector<float>& out, int wkey) {
    if (host_gemm_) {
        // Bmat is B[K][N] row-major (ld = N); a plain float matvec.
        out.assign((size_t)N, 0.0f);
        for (int k = 0; k < K; k++) {
            const float a = A[k];
            if (a == 0.0f) continue;
            const float* row = Bmat + (size_t)k * N;
            for (int n = 0; n < N; n++) out[(size_t)n] += a * row[n];
        }
        return;
    }
    float amax = 0.0f;
    for (int i = 0; i < K; i++) {
        float a = std::fabs(A[i]);
        if (a > amax) amax = a;
    }
    const float as = amax > 0.0f ? amax / 127.0f : 1.0f;
    const bool prof = getenv("NPU_PROFILE") != nullptr;
    const auto tp = std::chrono::steady_clock::now();

    // Token-invariant weights (wkey >= 0) get their own BO, packed once and
    // reused; every other call (wkey < 0: routed experts, whose weights change
    // per token) re-packs into layerB[0] as before.
    PackedWeight*             slot = nullptr;
    const std::vector<float>* gs = nullptr;
    if (weight_cache_ && wkey >= 0 && cur_layer_ >= 0 && cur_layer_ < c_.NL && dev_) {
        if (int(wcache_.size()) <= cur_layer_) wcache_.resize(size_t(cur_layer_) + 1);
        auto& m = wcache_[size_t(cur_layer_)];
        auto  it = m.find(wkey);
        if (it == m.end()) {
            PackedWeight pw;
            pw.bo = ctx.make_weight_bo(*dev_);
            float sout = 1.0f;
            ctx.packB_into(*pw.bo, Bmat, K, N, sout, pw.scales);
            it = m.emplace(wkey, std::move(pw)).first;
        }
        slot = &it->second;
        gs = &slot->scales;
    } else {
        float sout = 1.0f;
        ctx.packB(0, Bmat, K, N, sout);
        ctx.quantize_async(A, 1, K, as);
        gs = &ctx.group_scales[0];
    }

    const auto tl = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    xrt::run r = slot ? ctx.launch_async_with_bo(*slot->bo, A, 1, K, as)
                      : ctx.sync_and_launch(0);
    const auto tq = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ctx.wait_kernel(r);
    const auto tw = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ctx.readback();
    const auto tr = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    out.resize((size_t)N);
    for (int n = 0; n < N; n++)
        out[(size_t)n] = (float)ctx.Cm[n] * as * (*gs)[(size_t)n];

    if (prof) {
        static double sp = 0.0, sq = 0.0, sw = 0.0, sr = 0.0;
        static long sn = 0;
        sp += std::chrono::duration<double, std::milli>(tl - tp).count();
        sq += std::chrono::duration<double, std::milli>(tq - tl).count();
        sw += std::chrono::duration<double, std::milli>(tw - tq).count();
        sr += std::chrono::duration<double, std::milli>(tr - tw).count();
        if (++sn % 50 == 0)
            std::fprintf(stderr, "[prof] npu_gemm n=%ld pack=%.1f quant+launch=%.1f wait=%.1f readback=%.1f ms (N=%d)\n",
                         sn, sp / (double)sn, sq / (double)sn, sw / (double)sn, sr / (double)sn, N);
    }
}

void Q4nxNpuForward::npu_gemm_rows(I8Ctx& ctx, const float* A, int M, int K, int N,
                                   const float* Bmat, std::vector<float>& out, int wkey) {
    if (host_gemm_) {
        // Bmat is B[K][N] row-major (ld = N): a plain float matmul of A[M][K].
        out.assign((size_t)M * N, 0.0f);
        for (int m = 0; m < M; m++) {
            float* o = out.data() + (size_t)m * N;
            for (int k = 0; k < K; k++) {
                const float a = A[(size_t)m * K + k];
                if (a == 0.0f) continue;
                const float* row = Bmat + (size_t)k * N;
                for (int n = 0; n < N; n++) o[n] += a * row[n];
            }
        }
        return;
    }
    // Per-row activation scales: row m is quantized with its own amax/127 so a
    // batch's rows each keep their dynamic range (issue #1699: a single shared
    // scale zeroed low-magnitude rows when one token's activations dwarfed the
    // rest).
    std::vector<float> ascales((size_t)M);
    for (int m = 0; m < M; m++) {
        const float* row = A + (size_t)m * K;
        float amax = 0.0f;
        for (int k = 0; k < K; k++) {
            const float a = std::fabs(row[k]);
            if (a > amax) amax = a;
        }
        ascales[(size_t)m] = amax > 0.0f ? amax / 127.0f : 1.0f;
    }
    PackedWeight*             slot = nullptr;
    const std::vector<float>* gs = nullptr;
    if (weight_cache_ && wkey >= 0 && cur_layer_ >= 0 && cur_layer_ < c_.NL && dev_) {
        if (int(wcache_.size()) <= cur_layer_) wcache_.resize(size_t(cur_layer_) + 1);
        auto& m = wcache_[size_t(cur_layer_)];
        auto  it = m.find(wkey);
        if (it == m.end()) {
            PackedWeight pw;
            pw.bo = ctx.make_weight_bo(*dev_);
            float sout = 1.0f;
            ctx.packB_into(*pw.bo, Bmat, K, N, sout, pw.scales);
            it = m.emplace(wkey, std::move(pw)).first;
        }
        slot = &it->second;
        gs = &slot->scales;
    } else {
        float sout = 1.0f;
        ctx.packB(0, Bmat, K, N, sout);
        gs = &ctx.group_scales[0];
    }
    xrt::run r = slot
                     ? ctx.launch_async_with_bo_rows(*slot->bo, A, M, K, ascales.data())
                     : ctx.launch_async_rows(0, A, M, K, ascales.data());
    ctx.wait_kernel(r);
    ctx.readback();
    out.resize((size_t)M * N);
    for (int m = 0; m < M; m++) {
        const int32_t* cm = ctx.Cm + (size_t)m * ctx.ND;
        const float asc = ascales[(size_t)m];
        float* o = out.data() + (size_t)m * N;
        for (int n = 0; n < N; n++)
            o[n] = (float)cm[n] * asc * (*gs)[(size_t)n];
    }
}

bool Q4nxNpuForward::step_batch(const std::vector<int>& tokens, int start_pos,
                                std::vector<float>& logits) {
    if (!ready_) { err_ = "not initialised"; return false; }
    const int H = c_.H;
    const int M = (int)tokens.size();
    if (M <= 0) { err_ = "empty batch"; return false; }
    for (int t : tokens)
        if (t < 0 || t >= c_.NV) { err_ = "token out of range"; return false; }

    // A new request starts at position 0: clear the KV cache.  The cache is
    // append-only (kcache_[l].insert(end, ...)) but the attention indexes it by
    // `pos` (t in 0..pos), so without this a second request reads the PREVIOUS
    // request's K/V for positions 0..prompt_len and appends its own after them --
    // visible as cross-request echo (MiniCPM5-1B answered Japan, then echoed
    // "Tokyo" for France/Germany).  forward_serve.cpp's "each request resets it
    // from position 0" only ever held in the NPU_BATCH_CHECK debug path.
    if (start_pos == 0) reset();

    // Anything the batched body does not model falls back to per-token step():
    // MLA/MoE, GDN linear attention, Gemma4 (PLE + per-layer-type attention widths
    // + KV sharing), attn_output_gate, and multi-layer-type RoPE.
    bool kv_self = true;
    for (int l = 0; l < (int)kv_owner_.size() && l < c_.NL; l++)
        if (kv_owner_[l] != l) kv_self = false;
    // Two attention geometries (QKV2) and multiple layer types are fine: the
    // batched body selects the per-layer design and RoPE exactly as step() does.
    // MoE is fine too: its router/top-k are data-dependent per row and stay
    // per-token, while the attention GEMMs around them are batched. MLA and GDN
    // still fall back (their attention is not modelled here).
    const bool simple_dense =
        !mla_ && lin_kd_ == 0 && ple_dim_ == 0 && !gemma4_layer_ &&
        attn_gate_ == 0 && kv_self;
    if (!simple_dense || getenv("NPU_NO_BATCH")) {
        if (getenv("NPU_BATCH_DBG"))
            std::fprintf(stderr,
                         "[batch] M=%d fallback: mla=%d lin=%d ple=%d gemma4=%d experts=%d "
                         "gate=%d kv_self=%d no_batch=%d\n",
                         M, mla_, lin_kd_, ple_dim_, (int)gemma4_layer_, n_expert_,
                         attn_gate_, (int)kv_self, (int)(getenv("NPU_NO_BATCH") != nullptr));
        std::vector<float> one;
        for (int m = 0; m < M; m++)
            if (!step(tokens[m], start_pos + m, one)) return false;
        logits = one;
        return true;
    }

    // Debug oracle (NPU_BATCH_CHECK=1): run the same tokens sequentially first,
    // capture the next-token argmax, clear the KV the sequential pass appended,
    // then let the batched pass run and compare. One process, both paths, same
    // device state -- localises a batched-vs-sequential divergence immediately.
    static const bool batch_check = getenv("NPU_BATCH_CHECK") != nullptr;
    int seq_argmax = -1;
    if (batch_check) {
        std::vector<float> seq;
        for (int m = 0; m < M; m++)
            if (!step(tokens[m], start_pos + m, seq)) return false;
        reset();
        for (size_t i = 0; i < seq.size(); i++)
            if (seq_argmax < 0 || seq[i] > seq[(size_t)seq_argmax]) seq_argmax = (int)i;
    }

    const int i_qkv = idx_of(Q4NX_DESIGN_QKV);
    const int i_o   = idx_of(Q4NX_DESIGN_O);
    const int i_gu  = idx_of(Q4NX_DESIGN_GU);
    const int i_g   = idx_of(Q4NX_DESIGN_G);
    const int i_u   = idx_of(Q4NX_DESIGN_U);
    const int i_d   = idx_of(Q4NX_DESIGN_D);
    // A model with two attention geometries (e.g. MiniCPM4's sliding/full layers)
    // uses a different QKV/O design per layer type, as step() does.
    const int i_qkv2 = idx_of(Q4NX_DESIGN_QKV2);
    const int i_o2   = idx_of(Q4NX_DESIGN_O2);
    const bool two_widths = (i_qkv2 >= 0);
    const int im    = c_.IM;

    std::vector<float> X((size_t)M * H), XN((size_t)M * H);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < H; k++)
            X[(size_t)m * H + k] = embed_at(tokens[m], k) * embed_scale_;

    std::vector<float> QKV, CTX, O, ACT, HH, D, qkv_row, ctx_row, act_row, hh;
    for (int l = 0; l < c_.NL; l++) {
        cur_layer_ = l;
        const bool cached = weight_cached(0);
        if (!cached && !load_layer(l)) { if (err_.empty()) err_ = "dequant failed"; return false; }

        const int lt = (l < sem_.n_layer_types) ? sem_.layer_types[l] : Q4NX_LAYER_FULL;
        const bool l_full = (lt != Q4NX_LAYER_SLIDING);
        const int l_theta = l_full ? sem_.rope_theta : sem_.rope_theta_sliding;
        const int l_rot   = l_full ? sem_.rotary_dim : sem_.rotary_dim_sliding;
        const bool l_prop = l_full && sem_.proportional_rope;
        const int l_win   = l_full ? 0 : sem_.sliding_window;
        const int i_qkv_l = (two_widths && l_full) ? i_qkv2 : i_qkv;
        const int i_o_l   = (two_widths && l_full) ? i_o2 : i_o;

        // ---- attention ----
        for (int m = 0; m < M; m++)
            rmsnorm(X.data() + (size_t)m * H, in_norm_[l].data(), H,
                    XN.data() + (size_t)m * H);
        {
            const Q4nxDesignGeom* g = &designs_[i_qkv_l];
            Q4nxProjections p{};
            p.q = wq_.data(); p.k = wk_.data(); p.v = wv_.data();
            if (!cached && q4nx_assemble_B(g, &p, B_.data()) != 0) {
                err_ = "assemble qkv"; return false;
            }
            npu_gemm_rows(*ctx_[i_qkv_l], XN.data(), M, g->K, g->N, B_.data(), QKV, 0);
            const int l_hd = c_.NH > 0 ? g->q_rows / c_.NH : 0;
            const int l_nkv = (l_hd > 0 && g->k_rows > 0) ? g->k_rows / l_hd : c_.NKV;
            const int ctx_w = c_.NH * l_hd;
            CTX.assign((size_t)M * (size_t)ctx_w, 0.0f);
            for (int m = 0; m < M; m++) {
                qkv_row.assign(QKV.begin() + (size_t)m * g->N,
                               QKV.begin() + (size_t)(m + 1) * g->N);
                if (!attention(qkv_row, start_pos + m, ctx_row, l_hd, l_nkv, l_theta,
                               l_rot, l_prop, l_win, l, true)) {
                    err_ = "attention failed";
                    return false;
                }
                memcpy(CTX.data() + (size_t)m * ctx_w, ctx_row.data(),
                       (size_t)ctx_w * sizeof(float));
            }
        }
        {
            const Q4nxDesignGeom* g = &designs_[i_o_l];
            Q4nxProjections p{};
            p.o = wo_.data();
            if (!cached && q4nx_assemble_B(g, &p, B_.data()) != 0) {
                err_ = "assemble o"; return false;
            }
            npu_gemm_rows(*ctx_[i_o_l], CTX.data(), M, g->K, g->N, B_.data(), O, 1);
        }
        for (int m = 0; m < M; m++)
            for (int k = 0; k < H; k++)
                X[(size_t)m * H + k] += O[(size_t)m * H + k] * residual_scale_;

        // ---- MLP ----
        for (int m = 0; m < M; m++)
            rmsnorm(X.data() + (size_t)m * H, post_norm_[l].data(), H,
                    XN.data() + (size_t)m * H);
        if (n_expert_ > 0) {
            // MoE FFN: the router/top-k pick different experts per row, so this
            // stays per-token (the attention GEMMs above are already batched).
            // Grouping rows by expert so each expert's weight read serves the
            // whole batch is the next step -- go_rows is the primitive for it.
            std::vector<float> xn_row, d_row;
            for (int m = 0; m < M; m++) {
                xn_row.assign(XN.begin() + (size_t)m * H,
                              XN.begin() + (size_t)(m + 1) * H);
                if (!moe_mlp(l, xn_row, d_row)) { err_ = "moe failed"; return false; }
                for (int k = 0; k < H; k++)
                    X[(size_t)m * H + k] += d_row[(size_t)k] * residual_scale_;
            }
            continue;
        }
        if (i_gu >= 0) {
            const Q4nxDesignGeom* g = &designs_[i_gu];
            Q4nxProjections p{};
            p.gate = wg_.data(); p.up = wu_.data();
            if (!cached && q4nx_assemble_B(g, &p, B_.data()) != 0) {
                err_ = "assemble gu"; return false;
            }
            npu_gemm_rows(*ctx_[i_gu], XN.data(), M, g->K, g->N, B_.data(), ACT, 2);
        } else if (i_g >= 0 && i_u >= 0) {
            std::vector<float> gv, uv, av((size_t)M * (size_t)im * 2);
            const Q4nxDesignGeom* gg = &designs_[i_g];
            Q4nxProjections pg{}; pg.gate = wg_.data();
            if (!cached && q4nx_assemble_B(gg, &pg, B_.data()) != 0) {
                err_ = "assemble g"; return false;
            }
            npu_gemm_rows(*ctx_[i_g], XN.data(), M, gg->K, gg->N, B_.data(), gv, 2);
            const Q4nxDesignGeom* gu = &designs_[i_u];
            Q4nxProjections pu{}; pu.up = wu_.data();
            if (!cached && q4nx_assemble_B(gu, &pu, B_.data()) != 0) {
                err_ = "assemble u"; return false;
            }
            npu_gemm_rows(*ctx_[i_u], XN.data(), M, gu->K, gu->N, B_.data(), uv, 3);
            for (int m = 0; m < M; m++)
                for (int i = 0; i < im; i++) {
                    av[(size_t)m * 2 * im + i] = gv[(size_t)m * im + i];
                    av[(size_t)m * 2 * im + im + i] = uv[(size_t)m * im + i];
                }
            ACT.swap(av);
        } else {
            err_ = "no gate/up design";
            return false;
        }
        HH.assign((size_t)M * (size_t)im, 0.0f);
        for (int m = 0; m < M; m++) {
            act_row.assign(ACT.begin() + (size_t)m * 2 * im,
                           ACT.begin() + (size_t)(m + 1) * 2 * im);
            if (!mlp(act_row, hh, im)) { err_ = "mlp failed"; return false; }
            memcpy(HH.data() + (size_t)m * im, hh.data(), (size_t)im * sizeof(float));
        }
        {
            const Q4nxDesignGeom* g = &designs_[i_d];
            Q4nxProjections p{};
            p.down = wd_.data();
            if (!cached && q4nx_assemble_B(g, &p, B_.data()) != 0) {
                err_ = "assemble d"; return false;
            }
            npu_gemm_rows(*ctx_[i_d], HH.data(), M, g->K, g->N, B_.data(), D, 4);
        }
        for (int m = 0; m < M; m++)
            for (int k = 0; k < H; k++)
                X[(size_t)m * H + k] += D[(size_t)m * H + k] * residual_scale_;
    }

    // The next-token logits come from the last row of the batch, after the final
    // norm -- step() applies final_norm_ before run_lm_head, and feeding the raw
    // residual here produced garbage logits (the M=1 self-check caught it).
    std::vector<float> last(X.begin() + (size_t)(M - 1) * H, X.end());
    std::vector<float> fnorm(H);
    rmsnorm(last.data(), final_norm_.data(), H, fnorm.data());
    const bool ok = run_lm_head(fnorm, logits);
    if (batch_check) {
        int ba = 0;
        for (size_t i = 1; i < logits.size(); i++)
            if (logits[i] > logits[(size_t)ba]) ba = (int)i;
        std::fprintf(stderr, "[batchcheck] M=%d start=%d seq_argmax=%d batch_argmax=%d %s\n",
                     M, start_pos, seq_argmax, ba, seq_argmax == ba ? "MATCH" : "MISMATCH");
    } else if (getenv("NPU_BATCH_DBG")) {
        std::fprintf(stderr, "[batch] M=%d start=%d batched (%d layers, two_widths=%d)\n",
                     M, start_pos, c_.NL, (int)two_widths);
    }
    return ok;
}

int Q4nxNpuForward::max_batch_rows() const {
    for (int i = 0; i < ndesigns_; i++)
        if (ctx_[i]) return ctx_[i]->MD;
    return 1;  // host_gemm_ / not initialised: keep batches trivially small
}

// --------------------------------------------------------------------------
// host helpers (float32, matching the Python reference)
// --------------------------------------------------------------------------
float Q4nxNpuForward::rmsnorm(const float* x, const float* w, int n, float* out) const {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float inv = 1.0f / std::sqrt((float)(ss / (double)n) + sem_.eps);
    if (sem_.rmsnorm_offset) {
        // Gemma convention: y = x/rms * (1 + w)
        for (int i = 0; i < n; i++) out[i] = x[i] * inv * (1.0f + w[i]);
    } else {
        for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
    }
    return inv;
}

/* Scale-free RMSNorm: Gemma4's v_norm is built with_scale=False. */
float Q4nxNpuForward::rmsnorm_noweight(const float* x, int n, float* out) const {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float inv = 1.0f / std::sqrt((float)(ss / (double)n) + sem_.eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv;
    return inv;
}

// ---------------------------------------------------------------------------
// MLA + MoE (deepseek2 / GLM-4.7-Flash)
// ---------------------------------------------------------------------------
// Dequantize one expert's slice out of the pooled [n_expert][tiles][5120] tensor.
// MEASURED (not inferred): the converter's unpack TRANSPOSES, so although the GGUF
// stores {K, N, n_expert} (llama.cpp's create_tensor_gate_up_exps), the Q4NX slice
// is [N][K] -- corr 0.996969 against the GGUF read as [N][K] vs 0.0157 as [K][N].
// So dequant straight into [N][K], which is what assemble_B wants.
bool Q4nxNpuForward::mla_expert(const uint8_t* pool, size_t per_expert, int e,
                                int N, int K, std::vector<float>& out) {
    if (!pool || per_expert == 0 || e < 0) return false;
    // Dequantize straight into `out`: the old path materialized a `kn` temporary
    // and then copied it -- an extra (K*N) allocation plus a full copy per expert,
    // ~960 experts/token.
    out.resize((size_t)N * (size_t)K);
    // Do NOT transpose.  The tile decode at (rows=K, cols=N) already reproduces the
    // GGUF's flat order, and that flat sequence IS the logical [N][K] = [out][in]
    // weight.  Verified elementwise against the GGUF per expert: reshape(N,K) gives
    // corr 0.999198 (gate) / 0.996949 (down), while transposing gives -0.001079 /
    // 0.000417 -- i.e. the transpose destroyed the expert entirely, making every
    // routed expert's contribution ~10-20x wrong and flipping the top-k selection.
    if (dequant_any(pool + (size_t)e * per_expert, per_expert, K, N, out.data()) != 0)
        return false;
    return true;
}

// MLA attention for one layer.  q_a -> norm -> q_b, kv_a -> norm -> k_b/v_b, with
// RoPE applied to the LAST rope_dim_ of each q head and to the single shared
// rope slice carried inside kv_a (LLM_ARCH_DEEPSEEK2's MLA).
bool Q4nxNpuForward::mla_attn(int l, const std::vector<float>& x, std::vector<float>& o,
                              int pos) {
    const int H = c_.H, NH = c_.NH;
    const int qh = qk_nope_ + rope_dim_;         // per-head q width (192+64)
    const int kv_real = kv_lora_ + rope_dim_;    // 576
    const int i_qa = idx_of(Q4NX_DESIGN_QA), i_qb = idx_of(Q4NX_DESIGN_QB);
    const int i_kva = idx_of(Q4NX_DESIGN_KVA), i_o = idx_of(Q4NX_DESIGN_O);
    if (i_qa < 0 || i_qb < 0 || i_kva < 0 || i_o < 0) { err_ = "MLA designs missing"; return false; }

    std::vector<float> qa, qan, q, kva;
    {   // q_a: [H] -> [q_lora]
        const Q4nxDesignGeom* g = &designs_[i_qa];
        Q4nxProjections p{}; p.q = w_qa_.data();
        if (!weight_cached(0) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble q_a"; return false; }
        npu_gemm(*ctx_[i_qa], x.data(), g->K, g->N, B_.data(), qa, 0);
    }
    qan.resize(q_lora_);
    rmsnorm(qa.data(), mla_qan_.data(), q_lora_, qan.data());
    {   // q_b: [q_lora] -> [NH*qh]
        const Q4nxDesignGeom* g = &designs_[i_qb];
        Q4nxProjections p{}; p.k = w_qb_.data();
        if (!weight_cached(1) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble q_b"; return false; }
        npu_gemm(*ctx_[i_qb], qan.data(), g->K, g->N, B_.data(), q, 1);
    }
    {   // kv_a: [H] -> [kv_real] (+ zero pad to the kernel's n-tile)
        const Q4nxDesignGeom* g = &designs_[i_kva];
        Q4nxProjections p{}; p.v = w_kva_.data();
        if (!weight_cached(2) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble kv_a"; return false; }
        npu_gemm(*ctx_[i_kva], x.data(), g->K, g->N, B_.data(), kva, 2);
    }
    if ((int)kva.size() < kv_real) { err_ = "kv_a short"; return false; }
    std::vector<float> kl(kv_lora_);
    rmsnorm(kva.data(), mla_kvan_.data(), kv_lora_, kl.data());

    // RoPE on the shared slice and on each q head's tail.
    std::vector<float> kr(rope_dim_);
    for (int i = 0; i < rope_dim_; i++) kr[i] = kva[(size_t)kv_lora_ + i];
    if (!mla_rope(q, NH, qh, qk_nope_, kr, pos)) { err_ = "mla rope"; return false; }

    // Cache the reconstructed k_nope / shared k_rope / v for this position.
    const int kh = qk_nope_, vh = v_head_;
    std::vector<float>& kn = mla_kn_[l];
    std::vector<float>& krc = mla_kr_[l];
    std::vector<float>& vc = mla_v_[l];
    const size_t old = kn.size() / ((size_t)NH * kh);
    kn.resize((old + 1) * (size_t)NH * kh);
    vc.resize((old + 1) * (size_t)NH * vh);
    krc.resize((old + 1) * (size_t)rope_dim_);
    // k_b[h] : [kh][kv_lora] (stored padded to a 32-row tile, so read kh real rows)
    for (int h = 0; h < NH; h++) {
        // k_b: Q4NX tile grid [kv_lora][nope padded to 256], indexed flat[n*kv_lora+k].
        // v_b: the Q4NX tiles v_b with cols=256 while its LOGICAL row is 512 wide, so
        // the per-head [512][256] tile grid is NOT the logical matrix.  Decoding the
        // whole tensor at (rows=10240, cols=256) reproduces the GGUF's flat order
        // exactly (corr 0.996860 vs the 0.996870 a faithful Q4_1 requant scores),
        // i.e. the logical index is (h*256+n)*512 + k = out-major.  Indexing flat
        // [k*vh+n] instead permuted the whole MLA value path; the earlier check that
        // "pinned" it compared a single SUM, which a permutation can still match.
        const float* wk = &w_kb_[(size_t)h * (size_t)kv_lora_ * (size_t)kb_rows_];
        const float* wv = &w_vb_[(size_t)h * (size_t)kv_lora_ * (size_t)vh];
        float* dk = &kn[old * (size_t)NH * kh + (size_t)h * kh];
        float* dv = &vc[old * (size_t)NH * vh + (size_t)h * vh];
        for (int n = 0; n < kh; n++) {
            float s = 0.0f;
            for (int k = 0; k < kv_lora_; k++) s += wk[(size_t)n * kv_lora_ + k] * kl[k];
            dk[n] = s;
        }
        for (int n = 0; n < vh; n++) {
            float s = 0.0f;
            for (int k = 0; k < kv_lora_; k++) s += wv[(size_t)n * kv_lora_ + k] * kl[k];
            dv[n] = s;
        }
    }
    for (int i = 0; i < rope_dim_; i++) krc[old * (size_t)rope_dim_ + i] = kr[i];

    // Scores and softmax over the cache, then the weighted sum of v.
    const int T = pos + 1;
    const float scale = 1.0f / std::sqrt((float)(qh));
    std::vector<float> ctx((size_t)NH * vh, 0.0f);
    std::vector<float> score(T);
    for (int h = 0; h < NH; h++) {
        const float* qn = &q[(size_t)h * qh];
        const float* qr = qn + qk_nope_;
        float mx = -1e30f;
        for (int t = 0; t < T; t++) {
            const float* kt = &kn[(size_t)t * NH * kh + (size_t)h * kh];
            const float* rt = &krc[(size_t)t * rope_dim_];
            float s = 0.0f;
            for (int i = 0; i < kh; i++) s += qn[i] * kt[i];
            for (int i = 0; i < rope_dim_; i++) s += qr[i] * rt[i];
            score[t] = s * scale;
            if (score[t] > mx) mx = score[t];
        }
        float sum = 0.0f;
        for (int t = 0; t < T; t++) { score[t] = std::exp(score[t] - mx); sum += score[t]; }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        float* out = &ctx[(size_t)h * vh];
        for (int t = 0; t < T; t++) {
            const float pm = score[t] * inv;
            const float* vt = &vc[(size_t)t * NH * vh + (size_t)h * vh];
            for (int i = 0; i < vh; i++) out[i] += pm * vt[i];
        }
    }
    {   // o_proj: [NH*v_head] -> [H]
        const Q4nxDesignGeom* g = &designs_[i_o];
        Q4nxProjections p{}; p.o = w_o_.data();
        if (!weight_cached(3) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble o"; return false; }
        npu_gemm(*ctx_[i_o], ctx.data(), g->K, g->N, B_.data(), o, 3);
    }
    return true;
}

// RoPE for MLA.  llama.cpp's llama_model_rope_type() returns LLAMA_ROPE_TYPE_NORM
// for LLM_ARCH_DEEPSEEK2 ("normal RoPE, operating on pairs of CONSECUTIVE head
// values"), so the rotation pairs (2i, 2i+1) -- NOT (i, i+half) as the paired/NeoX
// convention would.  Using the paired form here was a bug of exactly the class
// that broke MiniCPM4/MiniCPM5.
bool Q4nxNpuForward::mla_rope(std::vector<float>& q, int NH, int qh, int nope,
                              std::vector<float>& kr, int pos) {
    const int half = rope_dim_ / 2;
    std::vector<float> cs(half), sn(half);
    for (int i = 0; i < half; i++) {
        const double inv = 1.0 / std::pow((double)sem_.rope_theta, (double)i / (double)half);
        const double ang = (double)pos * inv;
        cs[i] = (float)std::cos(ang); sn[i] = (float)std::sin(ang);
    }
    auto rot = [&](float* v) {
        for (int i = 0; i < half; i++) {
            const float a = v[2 * i], b = v[2 * i + 1];
            v[2 * i] = a * cs[i] - b * sn[i];
            v[2 * i + 1] = a * sn[i] + b * cs[i];
        }
    };
    for (int h = 0; h < NH; h++) rot(&q[(size_t)h * qh + nope]);
    rot(kr.data());
    return true;
}

// MoE FFN: router (host, N=64 cannot tile) -> gating -> top-k -> per-expert G/U/D
// on the NPU -> weighted sum, plus the shared expert.
bool Q4nxNpuForward::moe_mlp(int l, const std::vector<float>& xn, std::vector<float>& d) {
    const int H = c_.H;
    const int i_g = idx_of(Q4NX_DESIGN_G), i_u = idx_of(Q4NX_DESIGN_U), i_d = idx_of(Q4NX_DESIGN_D);
    if (i_g < 0 || i_u < 0 || i_d < 0) { err_ = "MoE designs missing"; return false; }
    d.assign(H, 0.0f);

    // Router.  llama.cpp: probs = sigmoid(logits) UNBIASED; selection_probs =
    // probs + exp_probs_b (the bias is added AFTER the sigmoid and used ONLY for
    // the top-k selection -- "leave probs unbiased as it's later used to get
    // expert weights").  Adding the bias to the logit before the sigmoid, as I
    // had, corrupts both the selection and the weights.
    std::vector<float> probs(n_expert_), selprobs(n_expert_), logits(n_expert_);
    for (int e = 0; e < n_expert_; e++) {
        const float* row = &moe_router_[(size_t)e * H];
        float s = 0.0f;
        for (int i = 0; i < H; i++) s += row[i] * xn[i];
        logits[e] = s;
        // llama.cpp's deepseek2 expert_gating_func: 2 = sigmoid, 1 = softmax.
        probs[e] = expert_gating_sigmoid_ ? (1.0f / (1.0f + std::exp(-s))) : s;
    }
    if (!expert_gating_sigmoid_) {
        float mx = *std::max_element(probs.begin(), probs.end());
        float sum = 0.0f;
        for (int e = 0; e < n_expert_; e++) { probs[e] = std::exp(probs[e] - mx); sum += probs[e]; }
        for (int e = 0; e < n_expert_; e++) probs[e] /= sum;
    }
    for (int e = 0; e < n_expert_; e++)
        selprobs[e] = probs[e] + (moe_router_b_.empty() ? 0.0f : moe_router_b_[e]);
    std::vector<int> idx(n_expert_);
    for (int e = 0; e < n_expert_; e++) idx[e] = e;
    std::partial_sort(idx.begin(), idx.begin() + top_k_, idx.end(),
                      [&](int a, int b) { return selprobs[a] > selprobs[b]; });
    std::vector<int> sel(idx.begin(), idx.begin() + top_k_);
    // Weights come from the UNBIASED probs, sum-normalised, then scaled.
    std::vector<float> r(n_expert_, 0.0f);
    for (int e : sel) r[e] = probs[e];
    // expert_weights_norm: llama.cpp's deepseek2 normalises the SELECTED weights by
    // their SUM (ggml_div(weights, ggml_sum_rows(weights))), not by their L2 norm,
    // and only then applies expert_weights_scale.  Using L2 here was a bug.
    float wsum = 0.0f;
    for (int e : sel) wsum += r[e];
    if (wsum > 0.0f) for (int e : sel) r[e] = r[e] / wsum * expert_w_scale_;

    // Per-layer MoE dump, laid out to compare directly against llama.cpp's
    // llama-eval-callback sums (ffn_moe_logits/probs/weights, ffn_moe_*).
    const bool dump = getenv("NPU_INFER_DUMP_MOE") && std::atoi(getenv("NPU_INFER_DUMP_MOE")) == l;
    if (dump) {
        auto sm = [](const std::vector<float>& v) { double s = 0; for (float f : v) s += f; return s; };
        double rawsum = 0; for (float f : logits) rawsum += f;
        double xnsq = 0; for (float f : xn) xnsq += (double)f * f;
        fprintf(stderr, "  [moe] L%d xn_sum=%.6f xn_rms=%.6f router_rms=%.6f\n",
                l, sm(xn), std::sqrt(xnsq / H),
                [&] { double s = 0; for (float f : moe_router_) s += (double)f * f;
                      return std::sqrt(s / moe_router_.size()); }());
        fprintf(stderr, "  [moe] L%d logits=%.6f probs=%.6f biased=%.6f sel=[%d,%d,%d,%d] "
                        "w=[%.6f,%.6f,%.6f,%.6f] wsum_raw=%.6f scaled=%.6f\n",
                l, rawsum, sm(probs), sm(selprobs), sel[0], sel[1], sel[2], sel[3],
                r[sel[0]], r[sel[1]], r[sel[2]], r[sel[3]], wsum, wsum * expert_w_scale_);
    }
    std::vector<float> g, u, dd, go, uo;
    double moe_g = 0, moe_u = 0, moe_sw = 0, moe_dn = 0;
    const int i_gb = idx_of(Q4NX_DESIGN_GB), i_ub = idx_of(Q4NX_DESIGN_UB);
    if (i_gb >= 0 && i_ub >= 0 && !getenv("NPU_NO_EXPERT_BATCH")) {
        // One gate GEMM and one up GEMM over ALL selected experts (they share xn),
        // instead of 2*top_k dispatches.  Expert j's weights occupy columns
        // [j*IM, (j+1)*IM) of a [K][top_k*IM] B; the design's N is wider, so
        // npu_gemm packs only the real columns and the tail stays zero.
        const int K = H, tk = (int)sel.size(), N = tk * expert_im_;
        static std::vector<float> Bg, Bu;   // reused across layers; fully written below
        Bg.resize((size_t)K * N);
        Bu.resize((size_t)K * N);
        std::vector<std::vector<float>> gs(tk), us(tk), ds(tk);
        for (int j = 0; j < tk; j++) {
            if (!mla_expert(exp_g_, exp_g_bytes_, sel[j], expert_im_, K, gs[j])) { err_ = "expert gate"; return false; }
            if (!mla_expert(exp_u_, exp_u_bytes_, sel[j], expert_im_, K, us[j])) { err_ = "expert up"; return false; }
            if (!mla_expert(exp_d_, exp_d_bytes_, sel[j], K, expert_im_, ds[j])) { err_ = "expert down"; return false; }
        }
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < tk; j++) {
            for (int n = 0; n < expert_im_; n++) {
                const float* rg = &gs[j][(size_t)n * K];
                const float* ru = &us[j][(size_t)n * K];
                for (int i = 0; i < K; i++) {
                    Bg[(size_t)i * N + (size_t)j * expert_im_ + n] = rg[i];
                    Bu[(size_t)i * N + (size_t)j * expert_im_ + n] = ru[i];
                }
            }
        }
        std::vector<float> go_all, uo_all;
        npu_gemm(*ctx_[i_gb], xn.data(), designs_[i_gb].K, N, Bg.data(), go_all);
        npu_gemm(*ctx_[i_ub], xn.data(), designs_[i_ub].K, N, Bu.data(), uo_all);
        for (int j = 0; j < tk; j++) {
            const int e = sel[j];
            std::vector<float> act((size_t)expert_im_);
            for (int i = 0; i < expert_im_; i++) {
                const float gv = go_all[(size_t)j * expert_im_ + i];
                act[i] = (gv / (1.0f + std::exp(-gv))) * uo_all[(size_t)j * expert_im_ + i];
            }
            const Q4nxDesignGeom* gd = &designs_[i_d];
            std::vector<float> ed;
            Q4nxProjections p{}; p.down = ds[j].data();
            if (q4nx_assemble_B(gd, &p, B_.data()) != 0) { err_ = "assemble expert down"; return false; }
            npu_gemm(*ctx_[i_d], act.data(), gd->K, gd->N, B_.data(), ed);
            if (dump) {
                for (int i = 0; i < expert_im_; i++) { moe_g += go_all[(size_t)j * expert_im_ + i]; moe_u += uo_all[(size_t)j * expert_im_ + i]; }
                for (float f : act) moe_sw += f;
                for (float f : ed) moe_dn += f;
            }
            for (int i = 0; i < H; i++) d[i] += r[e] * ed[i];
        }
    } else {
    for (int e : sel) {
        if (!mla_expert(exp_g_, exp_g_bytes_, e, expert_im_, H, g)) { err_ = "expert gate"; return false; }
        if (!mla_expert(exp_u_, exp_u_bytes_, e, expert_im_, H, u)) { err_ = "expert up"; return false; }
        if (!mla_expert(exp_d_, exp_d_bytes_, e, H, expert_im_, dd)) { err_ = "expert down"; return false; }
        // gate/up are GEMMs against xn (the dequantized matrices are weights, not
        // outputs): run them on the NPU through the G/U designs.
        {
            const Q4nxDesignGeom* gg = &designs_[i_g];
            Q4nxProjections p{}; p.gate = g.data();
            if (q4nx_assemble_B(gg, &p, B_.data()) != 0) { err_ = "assemble expert gate"; return false; }
            npu_gemm(*ctx_[i_g], xn.data(), gg->K, gg->N, B_.data(), go);
        }
        {
            const Q4nxDesignGeom* gu = &designs_[i_u];
            Q4nxProjections p{}; p.up = u.data();
            if (q4nx_assemble_B(gu, &p, B_.data()) != 0) { err_ = "assemble expert up"; return false; }
            npu_gemm(*ctx_[i_u], xn.data(), gu->K, gu->N, B_.data(), uo);
        }
        std::vector<float> act((size_t)expert_im_);
        for (int i = 0; i < expert_im_; i++)
            act[i] = (go[i] / (1.0f + std::exp(-go[i]))) * uo[i];
        const Q4nxDesignGeom* gd = &designs_[i_d];
        std::vector<float> ed;
        Q4nxProjections p{}; p.down = dd.data();
        if (q4nx_assemble_B(gd, &p, B_.data()) != 0) { err_ = "assemble expert down"; return false; }
        npu_gemm(*ctx_[i_d], act.data(), gd->K, gd->N, B_.data(), ed);
        if (dump) {
            for (float f : go) moe_g += f;
            for (float f : uo) moe_u += f;
            for (float f : act) moe_sw += f;
            for (float f : ed) moe_dn += f;
        }
        for (int i = 0; i < H; i++) d[i] += r[e] * ed[i];
    }
    }
    double moe_d_sum = 0; for (float f : d) moe_d_sum += f;
    // Shared expert, always on.
    double shexp_sum = 0;
    if (!share_g_.empty()) {
        std::vector<float> act((size_t)expert_im_);
        for (int i = 0; i < expert_im_; i++) {
            const float* rg = &share_g_[(size_t)i * H];
            const float* ru = &share_u_[(size_t)i * H];
            float sg = 0.0f, su = 0.0f;
            for (int k = 0; k < H; k++) { sg += rg[k] * xn[k]; su += ru[k] * xn[k]; }
            act[i] = (sg / (1.0f + std::exp(-sg))) * su;
        }
        const Q4nxDesignGeom* gd = &designs_[i_d];
        Q4nxProjections p{}; p.down = share_d_.data();
        if (!weight_cached(5) && q4nx_assemble_B(gd, &p, B_.data()) != 0) { err_ = "assemble shared down"; return false; }
        std::vector<float> sd;
        npu_gemm(*ctx_[i_d], act.data(), gd->K, gd->N, B_.data(), sd, 5);
        if (dump) {
            double ss = 0; for (float f : act) ss += f;
            fprintf(stderr, "  [moe] L%d expert gate=%.6f up=%.6f swiglu=%.6f down=%.6f weighted=%.6f | "
                            "shexp_swiglu=%.6f\n", l, moe_g, moe_u, moe_sw, moe_dn, moe_d_sum, ss);
        }
        for (float f : sd) shexp_sum += f;
        // Qwen MoE (qwen35moe) gates the shared expert with sigmoid(x . g), a [H]
        // vector; deepseek2 has no such gate (share_router_ empty -> scale 1).
        float sgate = 1.0f;
        if (!share_router_.empty()) {
            float g = 0.0f;
            for (int i = 0; i < H; i++) g += share_router_[i] * xn[i];
            sgate = 1.0f / (1.0f + std::exp(-g));
        }
        for (int i = 0; i < H; i++) d[i] += sgate * sd[i];
    }
    if (dump) {
        double tot = 0; for (float f : d) tot += f;
        fprintf(stderr, "  [moe] L%d moe_out=%.6f shexp=%.6f ffn_out=%.6f\n", l, moe_d_sum, shexp_sum, tot);
    }
    return true;
}

// The leading dense layer (leading_dense_block_count=1): DG/DU/DD designs.
bool Q4nxNpuForward::dense_mlp(int l, const std::vector<float>& xn, std::vector<float>& d) {
    const int H = c_.H;
    const int i_dg = idx_of(Q4NX_DESIGN_DG), i_du = idx_of(Q4NX_DESIGN_DU), i_dd = idx_of(Q4NX_DESIGN_DD);
    if (i_dg < 0 || i_du < 0 || i_dd < 0) { err_ = "dense designs missing"; return false; }
    std::vector<float> g, u;
    {
        const Q4nxDesignGeom* gd = &designs_[i_dg];
        Q4nxProjections p{}; p.gate = dense_g_.data();
        if (q4nx_assemble_B(gd, &p, B_.data()) != 0) { err_ = "assemble dense gate"; return false; }
        npu_gemm(*ctx_[i_dg], xn.data(), gd->K, gd->N, B_.data(), g);
    }
    {
        const Q4nxDesignGeom* gd = &designs_[i_du];
        Q4nxProjections p{}; p.up = dense_u_.data();
        if (q4nx_assemble_B(gd, &p, B_.data()) != 0) { err_ = "assemble dense up"; return false; }
        npu_gemm(*ctx_[i_du], xn.data(), gd->K, gd->N, B_.data(), u);
    }
    std::vector<float> act((size_t)dense_im_);
    for (int i = 0; i < dense_im_; i++)
        act[i] = (g[i] / (1.0f + std::exp(-g[i]))) * u[i];
    if (getenv("NPU_INFER_DUMP_DENSE")) {
        auto sm = [](const std::vector<float>& v) {
            double s = 0; for (float f : v) s += f; return s;
        };
        fprintf(stderr, "  [dense] gate n=%zu sum=%.6f | up n=%zu sum=%.6f | act n=%zu sum=%.6f\n",
                g.size(), sm(g), u.size(), sm(u), act.size(), sm(act));
    }
    const Q4nxDesignGeom* gd = &designs_[i_dd];
    Q4nxProjections p{}; p.down = dense_d_.data();
    if (q4nx_assemble_B(gd, &p, B_.data()) != 0) { err_ = "assemble dense down"; return false; }
    npu_gemm(*ctx_[i_dd], act.data(), gd->K, gd->N, B_.data(), d);
    return true;
}

bool Q4nxNpuForward::attention(const std::vector<float>& qkv, int pos,
                               std::vector<float>& ctx, int head_dim,
                               int n_kv_heads, float theta, int rotary_dim,
                               bool proportional, int window,
                               int kv_layer, bool append_kv) {
    const int NH = c_.NH;
    const int HD = head_dim;      /* per layer type (Gemma4: 256 or 512) */
    const int NKV = n_kv_heads;
    const int gqa = NH / NKV;

    std::vector<float> q((size_t)NH * HD), k((size_t)NKV * HD), v((size_t)NKV * HD);
    for (int i = 0; i < NH * HD; i++) q[i] = qkv[i];
    for (int i = 0; i < NKV * HD; i++) k[i] = qkv[NH * HD + i];
    for (int i = 0; i < NKV * HD; i++) v[i] = qkv[NH * HD + NKV * HD + i];
    // Qwen2.5-style attention biases: the q/k/v projections carry a per-output
    // bias (config attention_bias=true). Added before qk-norm/RoPE.
    if (sem_.attention_bias) {
        for (int i = 0; i < NH * HD; i++) q[i] += q_bias_[cur_layer_][i];
        for (int i = 0; i < NKV * HD; i++) k[i] += k_bias_[cur_layer_][i];
        for (int i = 0; i < NKV * HD; i++) v[i] += v_bias_[cur_layer_][i];
    }

    // per-head RMSNorm on q and k, only for architectures that have it.  A KV-shared
    // layer (Gemma4) computed no k at all, so its k is never normalized or used.
    if (sem_.qk_norm) {
        std::vector<float> t(HD);
        for (int h = 0; h < NH; h++) {
            rmsnorm(&q[(size_t)h * HD], q_norm_[cur_layer_].data(), HD, t.data());
            memcpy(&q[(size_t)h * HD], t.data(), sizeof(float) * HD);
        }
        if (append_kv) {
            for (int h = 0; h < NKV; h++) {
                rmsnorm(&k[(size_t)h * HD], k_norm_[cur_layer_].data(), HD, t.data());
                memcpy(&k[(size_t)h * HD], t.data(), sizeof(float) * HD);
            }
        }
    }
    // Gemma4 normalises v with a SCALE-FREE RMSNorm (its v_norm has no weight).
    if (sem_.v_norm && append_kv) {
        std::vector<float> tv(HD);
        for (int h = 0; h < NKV; h++) {
            rmsnorm_noweight(&v[(size_t)h * HD], HD, tv.data());
            memcpy(&v[(size_t)h * HD], tv.data(), sizeof(float) * HD);
        }
    }

    // RoPE: half-split over the first `rot` channels (rot == HD means full).
    // Gemma4's full layers use "proportional" RoPE: the pairing spans the WHOLE
    // head_dim, but the frequencies past rot/2 are zero, so those pairs are the
    // identity (_compute_proportional_rope_parameters).
    const int rot = (rotary_dim > 0 && rotary_dim <= HD) ? rotary_dim : HD;
    {
        const int half = proportional ? HD / 2 : rot / 2;
        std::vector<float> cos_(half), sin_(half);
        for (int i = 0; i < half; i++) {
            double inv;
            if (proportional) {
                inv = 1.0 / std::pow((double)theta, 2.0 * (double)i / (double)HD);
                if (i >= rot / 2) inv = 0.0;
            } else {
                inv = 1.0 / std::pow((double)theta, (double)i / (double)half);
                // LongRoPE (MiniCPM4): ggml's rope applies the factor as
                // `rope_yarn(theta / ff, ...)` (ggml-cpu/ops.cpp
                // ggml_rope_cache_init), i.e. the angle is DIVIDED by
                // freq_factors.  So divide the inv_freq here.  Multiplying (the
                // engine route's 599ac77) blows the high dims up by up to 31x
                // and yields garbage logits.  NPU_INFER_ROPE_FACTOR_MUL=1 keeps
                // the old multiply for A/B.
                if (i < (int)rope_scale_.size()) {
                    const char* mul = getenv("NPU_INFER_ROPE_FACTOR_MUL");
                    if (mul && mul[0] == '1') inv *= (double)rope_scale_[i];
                    else                       inv /= (double)rope_scale_[i];
                }
            }
            const double ang = (double)pos * inv;
            cos_[i] = (float)std::cos(ang);
            sin_[i] = (float)std::sin(ang);
        }
        auto rot_half = [&](float* x) {
            // pairs (i, i + half) inside the rotated block; the rest passes through
            for (int i = 0; i < half; i++) {
                const float x1 = x[i], x2 = x[i + half];
                x[i] = x1 * cos_[i] - x2 * sin_[i];
                x[i + half] = x1 * sin_[i] + x2 * cos_[i];
            }
        };
        for (int h = 0; h < NH; h++) rot_half(&q[(size_t)h * HD]);
        for (int h = 0; h < NKV; h++) rot_half(&k[(size_t)h * HD]);
    }

    // append to cache (a KV-shared layer reuses its owner's, which the owner has
    // already appended earlier in this same step).
    if (append_kv) {
        kcache_[kv_layer].insert(kcache_[kv_layer].end(), k.begin(), k.end());
        vcache_[kv_layer].insert(vcache_[kv_layer].end(), v.begin(), v.end());
    }

    const int T = pos + 1;
    // Gemma4 sets attention scaling to 1.0 (q/k carry an RMSNorm); everyone else
    // uses the usual 1/sqrt(head_dim).
    const float scale = sem_.attn_scaling_one ? 1.0f : 1.0f / std::sqrt((float)HD);
    // Sliding-window attention: ignore positions older than the window.
    const int t0 = (window > 0 && pos + 1 > window)
                       ? pos + 1 - window : 0;
    ctx.assign((size_t)NH * HD, 0.0f);
    std::vector<float> score(T);
    for (int h = 0; h < NH; h++) {
        const int kh = h / gqa;
        const float* qh = &q[(size_t)h * HD];
        float mx = -1e30f;
        for (int t = 0; t < T; t++) {
            if (t < t0) { score[t] = -1e30f; continue; }
            const float* kt = &kcache_[kv_layer][((size_t)t * NKV + kh) * HD];
            float d = 0.0f;
            for (int i = 0; i < HD; i++) d += qh[i] * kt[i];
            score[t] = d * scale;
            if (score[t] > mx) mx = score[t];
        }
        float sum = 0.0f;
        for (int t = 0; t < T; t++) {
            if (t < t0) { score[t] = 0.0f; continue; }
            score[t] = std::exp(score[t] - mx);
            sum += score[t];
        }
        const float inv = sum > 0 ? 1.0f / sum : 0.0f;
        float* out = &ctx[(size_t)h * HD];
        for (int t = t0; t < T; t++) {
            const float p = score[t] * inv;
            const float* vt = &vcache_[kv_layer][((size_t)t * NKV + kh) * HD];
            for (int i = 0; i < HD; i++) out[i] += p * vt[i];
        }
    }
    return true;
}

// SiLU(gate) * up from either a fused GU result or separate G and U results.
bool Q4nxNpuForward::mlp(const std::vector<float>& gu, std::vector<float>& h, int im) {
    h.resize(im);
    const int gelu = (sem_.mlp_act == Q4NX_ACT_GELU);
    for (int i = 0; i < im; i++) {
        const float g = gu[i];
        const float u = gu[im + i];
        float a;
        if (gelu) {
            // tanh approximation (gelu_pytorch_tanh), as HF uses for Gemma
            const float c = 0.7978845608028654f;   // sqrt(2/pi)
            a = 0.5f * g * (1.0f + std::tanh(c * (g + 0.044715f * g * g * g)));
        } else {
            a = g / (1.0f + std::exp(-g));         // SiLU
        }
        h[i] = a * u;
    }
    return true;
}

// --------------------------------------------------------------------------
// step
// --------------------------------------------------------------------------
bool Q4nxNpuForward::step(int token, int pos, std::vector<float>& logits) {
    if (!ready_) { err_ = "not initialised"; return false; }
    const int H = c_.H;
    if (token < 0 || token >= c_.NV) { err_ = "token out of range"; return false; }

    const int i_qkv = idx_of(Q4NX_DESIGN_QKV);
    const int i_o = idx_of(Q4NX_DESIGN_O);
    const int i_gu = idx_of(Q4NX_DESIGN_GU);
    const int i_g = idx_of(Q4NX_DESIGN_G);
    const int i_u = idx_of(Q4NX_DESIGN_U);
    const int i_d = idx_of(Q4NX_DESIGN_D);
    const int i_qkv2 = idx_of(Q4NX_DESIGN_QKV2);   /* 2nd attention type (Gemma4) */
    const int i_o2 = idx_of(Q4NX_DESIGN_O2);

    std::vector<float> x(H), xn(H), o(H), d(H), h, ctx;
    // The stored embedding is PRE-SCALED by the architecture's embedding scale
    // (Gemma4 bakes sqrt(hidden_size) into the tensor), so no scale is applied
    // here; the LM head divides it back out (see run_lm_head).
    for (int k = 0; k < H; k++)
        x[k] = embed_at(token, k) * embed_scale_;
    if (pos == 0 && getenv("NPU_INFER_DUMP_LAYERS")) {
        double s = 0;
        float mx = 0;
        for (int k = 0; k < H; k++) {
            const float a = std::fabs(x[(size_t)k]);
            s += a;
            if (a > mx) mx = a;
        }
        fprintf(stderr, "  [L] -1 %.6f %.4f\n", s / H, mx);
    }

    // Gemma4 needs this token's per-layer inputs before the layer loop.
    if (ple_dim_ > 0) build_ple_inputs(token);

    for (int l = 0; l < c_.NL; l++) {
        cur_layer_ = l;
        // Dense models: once this layer's projections are packed (after the
        // first token), the f32 weights are no longer read, so skip the per-token
        // Q4NX dequant.  GDN/MLA/PLE load members used outside the GEMMs, so they
        // keep loading every token.
        const bool dense_weights_cached = lin_kd_ == 0 && !mla_ && ple_dim_ == 0 && n_expert_ == 0 && weight_cached(0);
        if (!dense_weights_cached && !load_layer(l)) { if (err_.empty()) err_ = "dequant failed"; return false; }
        // MLA + MoE layers have no plain q/k/v/o or single-MLP structure, so they
        // take their own path through this loop body and skip the rest.
        if (mla_) {
            // The attention input is the NORMALISED hidden state (input_layernorm),
            // exactly as the standard path does; feeding the raw residual here was
            // a bug.
            rmsnorm(x.data(), in_norm_[l].data(), H, xn.data());
            if (!mla_attn(l, xn, o, pos)) return false;
            for (int k = 0; k < H; k++) x[k] += o[k] * residual_scale_;
#ifdef NPU_INFER_DEBUG_DUMPS
            if (l == 0 && pos == 0 && getenv("NPU_INFER_DUMP_ATTN")) {
                FILE* f = open_debug_dump(getenv("NPU_INFER_DUMP_ATTN"));
                if (f) { fwrite(x.data(), sizeof(float), (size_t)H, f); fclose(f); }
            }
#endif
            rmsnorm(x.data(), post_norm_[l].data(), H, xn.data());
            if (l < leading_dense_) {
                if (!dense_mlp(l, xn, d)) return false;
            } else {
                if (!moe_mlp(l, xn, d)) return false;
            }
            for (int k = 0; k < H; k++) x[k] += d[k] * residual_scale_;
            // Dump one layer's post-residual hidden state for the Python reference
            // comparison (NPU_INFER_DUMP_LN names the layer).
#ifdef NPU_INFER_DEBUG_DUMPS
            if (getenv("NPU_INFER_DUMP_LN") && l == atoi(getenv("NPU_INFER_DUMP_LN")) && pos == 0) {
                const char* p = getenv("NPU_INFER_DUMP_L0");
                FILE* f = open_debug_dump(p ? p : "/tmp/npu_layer.bin");
                if (f) { fwrite(x.data(), sizeof(float), (size_t)H, f); fclose(f); }
                fprintf(stderr, "  [mla] dumped layer-%d x (%d floats)\n", l, H);
            }
#endif
            if (pos == 0 && getenv("NPU_INFER_DUMP_LAYERS")) {
                auto am = [&](const std::vector<float>& v) {
                    float m = 0.0f;
                    for (float f : v) m = std::max(m, std::fabs(f));
                    return m;
                };
                fprintf(stderr, "  [mla] L%d attn_absmax=%.4f mlp_absmax=%.4f x_absmax=%.4f\n",
                        l, am(o), am(d), am(x));
            }
            continue;
        }
        // KV sharing (Gemma4): a layer past num_kv_shared_layers carries no k/v
        // weights and reuses the K/V of the last non-shared layer of its own
        // attention type, which that layer already appended earlier this step.
        const bool kv_shared = (l < (int)kv_owner_.size() && kv_owner_[l] != l);
        const int kv_layer = (l < (int)kv_owner_.size()) ? kv_owner_[l] : l;
        // use_double_wide_mlp: the shared layers run a 2*IM-wide MLP (GU2/D2).
        const bool wide = kv_shared && sem_.double_wide_mlp;
        const int i_gu_l = wide ? idx_of(Q4NX_DESIGN_GU2) : i_gu;
        const int i_d_l = wide ? idx_of(Q4NX_DESIGN_D2) : i_d;
        const int im_l = wide ? 2 * c_.IM : c_.IM;

        // ---- per-layer-type attention geometry ----
        // Gemma4's full_attention layers use global_head_dim (512) where its
        // sliding layers use head_dim (256), so BOTH the attention width and the
        // QKV/O geometry differ by layer type; every other model has one type.
        const int lt = (l < sem_.n_layer_types) ? sem_.layer_types[l]
                                                : Q4NX_LAYER_FULL;
        const bool l_full = (lt != Q4NX_LAYER_SLIDING);
        const bool two_widths = (i_qkv2 >= 0);
        const int i_qkv_l = (two_widths && l_full) ? i_qkv2 : i_qkv;
        const int i_o_l = (two_widths && l_full) ? i_o2 : i_o;
        const int l_qrows_all = designs_[i_qkv_l].q_rows;   // q_proj rows (x2 if gated)
        const int l_hd = (attn_gate_ ? l_qrows_all / 2 : l_qrows_all) / c_.NH;
        const int l_nkv_heads = (l_hd > 0) ? designs_[i_qkv_l].k_rows / l_hd : c_.NKV;
        const float l_theta = l_full ? sem_.rope_theta : sem_.rope_theta_sliding;
        const int l_rot = l_full ? sem_.rotary_dim : sem_.rotary_dim_sliding;
        const bool l_prop = l_full && sem_.proportional_rope;
        const int l_win = l_full ? 0 : sem_.sliding_window;
        // Debug aid: with NPU_INFER_NANCHECK set, stop at the first layer that
        // produces a non-finite hidden state (the offline host-GEMM run made
        // NaN-hunting cheap this way).
        auto nancheck = [&](const char* where) -> bool {
            if (!getenv("NPU_INFER_NANCHECK")) return true;
            for (int k = 0; k < H; k++) {
                if (!std::isfinite(x[(size_t)k])) {
                    fprintf(stderr, "  [nan] pos=%d layer=%d (%s) after %s\n", pos, l,
                            (lin_kd_ > 0 && lt == Q4NX_LAYER_LINEAR) ? "linear" : "full", where);
                    return false;
                }
            }
            return true;
        };

        // ---- attention ---- (or, for a GDN layer, gdn_layer below)
        rmsnorm(x.data(), in_norm_[l].data(), H, xn.data());
        if (lin_kd_ > 0 && lt == Q4NX_LAYER_LINEAR) {
            if (!gdn_layer(l, xn, o)) return false;
        } else {
        {
            const Q4nxDesignGeom* g = &designs_[i_qkv_l];
            Q4nxProjections p{};
            p.q = wq_.data(); p.k = wk_.data(); p.v = wv_.data();
            // A shared layer has no k/v weights at all: only its q is real and the
            // k/v half of the GEMM output is discarded.  Reusing wq_ for those
            // slots keeps assemble_B's row counts in bounds (q_rows >= k_rows).
            if (kv_shared) { p.k = wq_.data(); p.v = wq_.data(); }
            if (!weight_cached(0) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble qkv"; return false; }
            std::vector<float> qkv;
            npu_gemm(*ctx_[i_qkv_l], xn.data(), g->K, g->N, B_.data(), qkv, 0);
            // attn_output_gate: the fused QKV is [q | gate | k | v], so attention()
            // must not see the gate half; the gate multiplies ctx afterwards.
            std::vector<float> qkv_use;
            const int nhd = c_.NH * l_hd;
            const float* gate_ptr = nullptr;
            if (attn_gate_) {
                const int kvsz = g->k_rows + g->v_rows;
                qkv_use.resize((size_t)nhd + (size_t)kvsz);
                memcpy(qkv_use.data(), qkv.data(), (size_t)nhd * sizeof(float));
                memcpy(qkv_use.data() + nhd, qkv.data() + 2 * nhd,
                       (size_t)kvsz * sizeof(float));
                gate_ptr = qkv.data() + nhd;
            } else {
                qkv_use = qkv;
            }
            if (!attention(qkv_use, pos, ctx, l_hd, l_nkv_heads, l_theta, l_rot, l_prop,
                           l_win, kv_layer, !kv_shared)) {
                err_ = "attention failed";
                return false;
            }
            if (gate_ptr)
                for (int i = 0; i < nhd; i++)
                    ctx[i] *= 1.0f / (1.0f + expf(-gate_ptr[i]));
        }
        {
            const Q4nxDesignGeom* g = &designs_[i_o_l];
            Q4nxProjections p{};
            p.o = wo_.data();
            if (!weight_cached(1) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble o"; return false; }
            npu_gemm(*ctx_[i_o_l], ctx.data(), g->K, g->N, B_.data(), o, 1);
        }
        }  // end non-GDN attention
        if (gemma4_layer_) {
            // Gemma4 normalises the attention output BEFORE the residual add.
            rmsnorm(o.data(), post_norm_[l].data(), H, xn.data());
            for (int k = 0; k < H; k++) x[k] += xn[k];
        } else {
            for (int k = 0; k < H; k++) x[k] += o[k] * residual_scale_;
        }
        if (!nancheck("attention")) return false;

        // ---- MLP ----
        // Gemma4 normalises with pre_feedforward_layernorm and applies
        // post_feedforward_layernorm to the down-projection output; the plain
        // decoder normalises x with post_attention_layernorm instead.
        if (gemma4_layer_) {
            rmsnorm(x.data(), pre_ff_norm_[l].data(), H, xn.data());
        } else {
            rmsnorm(x.data(), post_norm_[l].data(), H, xn.data());
        }
        if (n_expert_ > 0) {
            // Qwen MoE layer (qwen3moe / qwen35moe): the FFN is router -> top-k
            // experts, not a dense gate/up/down.  moe_mlp writes the whole FFN
            // delta into d.  (deepseek2/GLM MoE layers never reach here -- the MLA
            // branch above handles attn+MoE and continues.)
            if (!moe_mlp(l, xn, d)) return false;
            for (int k = 0; k < H; k++) x[k] += d[k] * residual_scale_;
            if (!nancheck("moe")) return false;
            continue;
        }
        std::vector<float> act;
        if (i_gu_l >= 0) {
            const Q4nxDesignGeom* g = &designs_[i_gu_l];
            Q4nxProjections p{};
            p.gate = wg_.data(); p.up = wu_.data();
            if (!weight_cached(2) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble gu"; return false; }
            npu_gemm(*ctx_[i_gu_l], xn.data(), g->K, g->N, B_.data(), act, 2);
        } else if (i_g >= 0 && i_u >= 0) {
            // split gate/up: two kernels, then SiLU(gate) * up
            std::vector<float> gv, uv;
            const Q4nxDesignGeom* gg = &designs_[i_g];
            Q4nxProjections pg{}; pg.gate = wg_.data();
            if (!weight_cached(2) && q4nx_assemble_B(gg, &pg, B_.data()) != 0) { err_ = "assemble g"; return false; }
            npu_gemm(*ctx_[i_g], xn.data(), gg->K, gg->N, B_.data(), gv, 2);

            const Q4nxDesignGeom* gu = &designs_[i_u];
            Q4nxProjections pu{}; pu.up = wu_.data();
            if (!weight_cached(3) && q4nx_assemble_B(gu, &pu, B_.data()) != 0) { err_ = "assemble u"; return false; }
            npu_gemm(*ctx_[i_u], xn.data(), gu->K, gu->N, B_.data(), uv, 3);

            act.resize((size_t)im_l * 2);
            for (int i = 0; i < im_l; i++) {
                act[i] = gv[(size_t)i];
                act[im_l + i] = uv[(size_t)i];
            }
        } else {
            err_ = "no gate/up design";
            return false;
        }
        mlp(act, h, im_l);
        {
            const Q4nxDesignGeom* g = &designs_[i_d_l];
            Q4nxProjections p{};
            p.down = wd_.data();
            if (!weight_cached(4) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble d"; return false; }
            npu_gemm(*ctx_[i_d_l], h.data(), g->K, g->N, B_.data(), d, 4);
        }
        if (!gemma4_layer_) {
            for (int k = 0; k < H; k++) x[k] += d[k] * residual_scale_;
            if (!nancheck("mlp")) return false;
            if (pos == 0 && getenv("NPU_INFER_DUMP_LAYERS")) {
                double s = 0;
                float mx = 0;
                for (int k = 0; k < H; k++) {
                    const float a = std::fabs(x[(size_t)k]);
                    s += a;
                    if (a > mx) mx = a;
                }
                fprintf(stderr, "  [L] %d %.6f %.4f\n", l, s / H, mx);
            }
            continue;
        }
        // post_feedforward_layernorm, then the residual add.
        rmsnorm(d.data(), post_ff_norm_[l].data(), H, xn.data());
        for (int k = 0; k < H; k++) x[k] += xn[k];

        // ---- per-layer embedding block (PLE) ----
        // x = x + post_per_layer_input_norm( per_layer_projection(
        //         act_fn(per_layer_input_gate(x)) * per_layer_input[l] ) )
        if (ple_dim_ > 0) {
            const int i_pg = idx_of(Q4NX_DESIGN_PLEGATE);
            const int i_pp = idx_of(Q4NX_DESIGN_PLEPROJ);
            std::vector<float> pg, pp;
            {
                const Q4nxDesignGeom* g = &designs_[i_pg];
                Q4nxProjections p{};
                p.ple = inp_gate_[l].data();
                if (!weight_cached(0) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble plegate"; return false; }
                npu_gemm(*ctx_[i_pg], x.data(), g->K, g->N, B_.data(), pg, 0);
            }
            const int gelu = (sem_.mlp_act == Q4NX_ACT_GELU);
            for (int i = 0; i < ple_dim_; i++) {
                float a;
                if (gelu) {
                    const float cc = 0.7978845608028654f;
                    a = 0.5f * pg[i] *
                        (1.0f + std::tanh(cc * (pg[i] + 0.044715f * pg[i] * pg[i] * pg[i])));
                } else {
                    a = pg[i] / (1.0f + std::exp(-pg[i]));
                }
                pg[i] = a * ple_input_[(size_t)l * ple_dim_ + i];
            }
            {
                const Q4nxDesignGeom* g = &designs_[i_pp];
                Q4nxProjections p{};
                p.ple = ple_proj_[l].data();
                if (!weight_cached(1) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble pleproj"; return false; }
                npu_gemm(*ctx_[i_pp], pg.data(), g->K, g->N, B_.data(), pp, 1);
            }
            rmsnorm(pp.data(), post_ln_[l].data(), H, xn.data());
            for (int k = 0; k < H; k++) x[k] += xn[k];
        }
        // layer_scalar (layer_output_scale) closes the Gemma4 layer.
        const float ls = (l < (int)layer_scale_.size()) ? layer_scale_[l] : 1.0f;
        if (ls != 1.0f)
            for (int k = 0; k < H; k++) x[k] *= ls;
    }

    rmsnorm(x.data(), final_norm_.data(), H, xn.data());
    if (getenv("NPU_INFER_DUMP_HIDDEN")) {
        char hp[256];
        snprintf(hp, sizeof(hp), "/tmp/q4nx_hidden_%d.bin", pos);
        // private to the user, and never through a pre-planted link in /tmp
        const int fd = open(hp, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
        FILE* fp = fd >= 0 ? fdopen(fd, "wb") : nullptr;
        if (fp) { fwrite(xn.data(), sizeof(float), (size_t)H, fp); fclose(fp); }
        fprintf(stderr, "  [debug] hidden pos %d mean=%.6f absmax=%.6f -> %s\n",
                pos, [&]{ double s=0; for(float v:xn) s+=std::fabs((double)v); return s/H; }(),
                [&]{ float m=0; for(float v:xn) m=std::max(m, std::fabs(v)); return m; }(), hp);
    }
#ifdef NPU_INFER_DEBUG_DUMPS
    if (getenv("NPU_INFER_DUMP_HEAD")) {
        FILE* f = open_debug_dump(getenv("NPU_INFER_DUMP_HEAD"));
        if (f) { fwrite(xn.data(), sizeof(float), (size_t)xn.size(), f); fclose(f); }
    }
#endif
    if (!run_lm_head(xn, logits)) return false;
    if (getenv("NPU_INFER_DUMP_LAYERS")) {
        double s = 0; float mx = 0;
        for (float f : xn) { s += std::fabs(f); if (std::fabs(f) > mx) mx = std::fabs(f); }
        fprintf(stderr, "  [mla] final xn: mean|x|=%.6f absmax=%.6f (n=%zu)\n",
                s / (double)xn.size(), mx, xn.size());
        // top-3 before the final norm, to separate a collapsed hidden state from a
        // broken head
        double s2 = 0; float mx2 = 0;
        for (float f : x) { s2 += std::fabs(f); if (std::fabs(f) > mx2) mx2 = std::fabs(f); }
        fprintf(stderr, "  [mla] pre-norm x: mean|x|=%.6f absmax=%.6f\n",
                s2 / (double)x.size(), mx2);
    }
    // MiniCPM4 logit_scale: fitted against the llama.cpp oracle, this forward's
    // logits were ~16x the oracle's (oracle top-1 prob 0.928 vs effectively
    // one-hot here) and the best-fit temperature was 0.060 ~ 1/16.7 against a
    // GGUF logit_scale of 16.  Constant divisor -> cannot move the argmax, so
    // the captured anchor answers are unchanged.
    if (logit_scale_ != 1.0f)
        for (size_t i = 0; i < logits.size(); i++) logits[i] /= logit_scale_;
    return true;
}

// tiled LM head: logits[K*t + n] = dot(embed[K*t + n], h); K = H, tile = QKV N
bool Q4nxNpuForward::run_lm_head(const std::vector<float>& h, std::vector<float>& logits) {
    const int H = c_.H, NV = c_.NV;
    if (host_gemm_) {
        // Offline mode: plain float matvec against the (possibly dequantized)
        // tied table, matching the Python reference.  No I8Ctx is touched.
        const float inv_es0 = (sem_.tie_embeddings && sem_.embed_scale > 0)
                                  ? 1.0f / sem_.embed_scale : 1.0f;
        logits.assign((size_t)NV, 0.0f);
        for (int n = 0; n < NV; n++) {
            float s = 0;
            for (int k = 0; k < H; k++) s += h[(size_t)k] * inv_es0 * lm_at(n, k);
            if (sem_.logit_softcap > 0)
                s = sem_.logit_softcap * std::tanh(s / sem_.logit_softcap);
            logits[(size_t)n] = s;
        }
        return true;
    }
    // Reuse an existing GEMM geometry for the head (the variable keeps its old name
    // for the block below).  QKV when the arch has one; the MLA/deepseek2 arch
    // (GLM-4.7-Flash) has QA/QB/KVA/O instead, so idx_of(QKV) returned -1 and T was
    // read out of designs_[-1] -- garbage, and the ntiles division below raised
    // SIGFPE.  Only this NPU path uses T (the host path is a plain matvec), which is
    // why GLM crashed here the first time the ELF path was exercised.
    // Pick the WIDEST design with K=H for the head.  Each tile is a separate
    // synchronous launch (~one NPU round-trip each, ~constant regardless of N),
    // so fewer/larger tiles mean fewer launches for the same math.
    int i_qkv = -1;
    for (int d = 0; d < ndesigns_; d++) {
        if (designs_[d].K != H || designs_[d].N <= 0) continue;
        if (i_qkv < 0 || designs_[d].N > designs_[i_qkv].N) i_qkv = d;
    }
    if (i_qkv < 0) {
        err_ = "LM head: no GEMM design with K=H available";
        return false;
    }
    const int T = designs_[i_qkv].N;
    const int ntiles = (NV + T - 1) / T;
    logits.assign(NV, 0.0f);

    // The stored embedding is PRE-SCALED by the architecture's embedding scale
    // (Gemma4: sqrt(hidden_size)), so the LM head must divide that scale back out
    // to recover the raw tied weight.  embed_scale is 1 for every other model.
    const float inv_es = (sem_.tie_embeddings && sem_.embed_scale > 0)
                             ? 1.0f / sem_.embed_scale : 1.0f;
    std::vector<float> hs((size_t)H);
    for (int k = 0; k < H; k++) hs[(size_t)k] = h[(size_t)k] * inv_es;

    const size_t tile_elems = (size_t)H * (size_t)T;
    if (lm_head_tiles_.size() != (size_t)ntiles * tile_elems || lm_head_tiles_T_ != T) {
        // Once per model: [tile][H][T], so each token copies sequentially instead
        // of gathering the whole head with a T-float stride.
        lm_head_tiles_.assign((size_t)ntiles * tile_elems, 0.0f);
        for (int t = 0; t < ntiles; t++) {
            const int n0 = t * T, n1 = std::min(NV, n0 + T);
            float* dst = lm_head_tiles_.data() + (size_t)t * tile_elems;
            for (int n = n0; n < n1; n++)
                for (int k = 0; k < H; k++)
                    dst[(size_t)k * T + (n - n0)] = lm_at(n, k);
        }
        lm_head_tiles_T_ = T;
    }
    std::vector<float> Bt(tile_elems);
    for (int t = 0; t < ntiles; t++) {
        const int n0 = t * T, n1 = std::min(NV, n0 + T);
        std::memcpy(Bt.data(), lm_head_tiles_.data() + (size_t)t * tile_elems,
                    tile_elems * sizeof(float));
        float sout = 1.0f;
        ctx_[i_qkv]->packB(0, Bt.data(), H, T, sout);
        float amax = 0.0f;
        for (int k = 0; k < H; k++) { float a = std::fabs(hs[(size_t)k]); if (a > amax) amax = a; }
        const float as = amax > 0 ? amax / 127.0f : 1.0f;
        ctx_[i_qkv]->quantize_async(hs.data(), 1, H, as);
        xrt::run r = ctx_[i_qkv]->sync_and_launch(0);
        ctx_[i_qkv]->wait_kernel(r);
        ctx_[i_qkv]->readback();
        const std::vector<float>& gs = ctx_[i_qkv]->group_scales[0];
        for (int n = n0; n < n1; n++) {
            float v = (float)ctx_[i_qkv]->Cm[n - n0] * as * gs[(size_t)(n - n0)];
            if (sem_.logit_softcap > 0)
                v = sem_.logit_softcap * std::tanh(v / sem_.logit_softcap);
            logits[(size_t)n] = v;
        }
    }
    return true;
}
