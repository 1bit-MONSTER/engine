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
// five when they are split.  No model name or tag appears in this file; artifact
// names are geometry-derived, with an optional tag fallback for xclbin-only
// model directories (NPU_INFER_TAG).
//
// Numerics follow tools/q4nx_forward_ref.py so the two can be compared
// element-wise.
#include "q4nx_forward.h"
#include <fcntl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include <xrt/xrt_device.h>

#include "npu_engine_i8ctx_inc.h"   // I8Ctx
#include "q4nx_dequant.h"
#include "q4nx_pack.h"

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

// LongRoPE short_factor from config.json's rope_scaling (MiniCPM4). Returns the
// per-half-dim frequency multipliers (empty when the config has none).
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
                          const std::string& dir, bool use_elf) {
    return init_impl(&dev, model_path, dir, use_elf);
}

bool Q4nxNpuForward::init_host(const char* model_path, const std::string& dir) {
    host_gemm_ = true;
    return init_impl(nullptr, model_path, dir, true);
}

bool Q4nxNpuForward::init_impl(xrt::device* dev, const char* model_path,
                               const std::string& dir, bool use_elf) {
    if (getenv("NPU_INFER_HOST_GEMM")) host_gemm_ = true;
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
                const float* sc = sd ? (const float*)model_tensor_data(mw_, sd) : nullptr;
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

    // ---- derive the design set from the architecture dims ----
    // fused vs split gate/up is an artifact property, so probe for it rather
    // than keying on a model name: a GU artifact means fused, G/U means split.
    // Zero-initialised: the optionally-derived fields (ple_dim, double_wide_mlp,
    // NKV2/HD2, lin_kd/lin_vd, attn_output_gate) must be 0 when absent, not
    // garbage from the stack.
    Q4nxModelDims dims{};
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

    const char* tag = getenv("NPU_INFER_TAG");

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
        } else if (use_elf) {
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
        } else {
            // Artifact resolution.  The geometry-suffixed names are preferred by
            // default (tag-free), but several of them are degenerate 59-byte
            // stubs while the model-tagged twins are real, so when NPU_INFER_TAG
            // is set explicitly it wins; otherwise the geometry name is tried and
            // the tag used as fallback.
            char xc[512], ip[512];
            snprintf(xc, sizeof(xc), "%s/final_i8_%s_K%d_N%d.xclbin", dir.c_str(),
                     role_tag(g->role), g->K, g->N);
            snprintf(ip, sizeof(ip), "%s/insts_i8_%s_K%d_N%d.txt", dir.c_str(),
                     role_tag(g->role), g->K, g->N);
            if (tag) {
                snprintf(xc, sizeof(xc), "%s/final_i8_%s_%s.xclbin", dir.c_str(),
                         role_tag(g->role), tag);
                snprintf(ip, sizeof(ip), "%s/insts_i8_%s_%s.txt", dir.c_str(),
                         role_tag(g->role), tag);
                if (!file_exists(xc, 4096)) {
                    snprintf(xc, sizeof(xc), "%s/final_i8_%s_K%d_N%d.xclbin",
                             dir.c_str(), role_tag(g->role), g->K, g->N);
                    snprintf(ip, sizeof(ip), "%s/insts_i8_%s_K%d_N%d.txt",
                             dir.c_str(), role_tag(g->role), g->K, g->N);
                }
            }
            if (!file_exists(xc)) {
                err_ = std::string("missing xclbin for ") + role_tag(g->role) + ": " + xc;
                return false;
            }
            if (!ctx_[i]->init(*dev, xc, ip, 4, 1)) {
                err_ = std::string("I8Ctx init failed for ") + xc;
                return false;
            }
            fprintf(stderr, "  [forward] %-3s M=%d K=%d N=%d  %s\n", role_tag(g->role),
                    g->M, g->K, g->N, xc);
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
            if (!td) return false;
            const uint16_t* p = (const uint16_t*)model_tensor_data(mw_, td);
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
            const uint16_t* p = (const uint16_t*)model_tensor_data(mw_, td);
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

    backend_ = use_elf ? "full ELF" : "xclbin";
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
        const uint16_t* p = (const uint16_t*)model_tensor_data(mw_, const_cast<TensorDesc*>(d));
        if (!p) return false;
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
                if (!td) return false;
                const uint16_t* p = (const uint16_t*)model_tensor_data(mw_, td);
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
                if (!td) return false;
                const uint16_t* p = (const uint16_t*)model_tensor_data(mw_, td);
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
        if (q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble gdn qkv"; return false; }
        npu_gemm(*ctx_[i_qkv], xn.data(), g->K, g->N, B_.data(), qkv);
    }
    {
        const Q4nxDesignGeom* g = &designs_[i_z];
        Q4nxProjections p{};
        p.ple = wz_.data();
        if (q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble gdn z"; return false; }
        npu_gemm(*ctx_[i_z], xn.data(), g->K, g->N, B_.data(), z);
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
        if (q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble gdn out"; return false; }
        npu_gemm(*ctx_[i_o], core.data(), g->K, g->N, B_.data(), out);
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
    if (!raw) { err_ = "per_layer_token_embd has no data"; return false; }
    ple_block_.assign((size_t)Q4NX_TILE_ROWS * (size_t)cols, 0.0f);
    const size_t plain_bytes =
        (size_t)sem_.vocab_size_per_layer_input * (size_t)cols;
    if ((size_t)td->data_size == plain_bytes) {
        // PLAIN int8 + per-32-column scale (Gemma4): only the requested band of
        // 32 rows is needed, so it is dequantized in place rather than the whole
        // [vocab, NL*ple_dim] table.
        TensorDesc* sd =
            model_tensor_by_name(mw_, "model.per_layer_token_embd.weight.scale");
        const float* sc = sd ? (const float*)model_tensor_data(mw_, sd) : nullptr;
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
                              const float* Bmat, std::vector<float>& out) {
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
    float sout = 1.0f;
    ctx.packB(0, Bmat, K, N, sout);

    float amax = 0.0f;
    for (int i = 0; i < K; i++) {
        float a = std::fabs(A[i]);
        if (a > amax) amax = a;
    }
    const float as = amax > 0.0f ? amax / 127.0f : 1.0f;
    ctx.quantize_async(A, 1, K, as);
    xrt::run r = ctx.sync_and_launch(0);
    ctx.wait_kernel(r);
    ctx.readback();

    out.resize((size_t)N);
    const std::vector<float>& gs = ctx.group_scales[0];
    for (int n = 0; n < N; n++)
        out[(size_t)n] = (float)ctx.Cm[n] * as * gs[(size_t)n];
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
                // LongRoPE (MiniCPM4): ggml applies the factor as
                // rope_yarn(theta / ff, ...) (ggml-cpu/ops.cpp
                // ggml_rope_cache_init) -- the angle is DIVIDED by freq_factors.
                // Multiplying inflates the high dims by up to 31x and yields
                // garbage.  NPU_INFER_ROPE_FACTOR_MUL=1 keeps the old multiply.
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
        if (!load_layer(l)) { if (err_.empty()) err_ = "dequant failed"; return false; }
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
            if (q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble qkv"; return false; }
            std::vector<float> qkv;
            npu_gemm(*ctx_[i_qkv_l], xn.data(), g->K, g->N, B_.data(), qkv);
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
            if (q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble o"; return false; }
            npu_gemm(*ctx_[i_o_l], ctx.data(), g->K, g->N, B_.data(), o);
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
        std::vector<float> act;
        if (i_gu_l >= 0) {
            const Q4nxDesignGeom* g = &designs_[i_gu_l];
            Q4nxProjections p{};
            p.gate = wg_.data(); p.up = wu_.data();
            if (q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble gu"; return false; }
            npu_gemm(*ctx_[i_gu_l], xn.data(), g->K, g->N, B_.data(), act);
        } else if (i_g >= 0 && i_u >= 0) {
            // split gate/up: two kernels, then SiLU(gate) * up
            std::vector<float> gv, uv;
            const Q4nxDesignGeom* gg = &designs_[i_g];
            Q4nxProjections pg{}; pg.gate = wg_.data();
            if (q4nx_assemble_B(gg, &pg, B_.data()) != 0) { err_ = "assemble g"; return false; }
            npu_gemm(*ctx_[i_g], xn.data(), gg->K, gg->N, B_.data(), gv);

            const Q4nxDesignGeom* gu = &designs_[i_u];
            Q4nxProjections pu{}; pu.up = wu_.data();
            if (q4nx_assemble_B(gu, &pu, B_.data()) != 0) { err_ = "assemble u"; return false; }
            npu_gemm(*ctx_[i_u], xn.data(), gu->K, gu->N, B_.data(), uv);

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
            if (q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble d"; return false; }
            npu_gemm(*ctx_[i_d_l], h.data(), g->K, g->N, B_.data(), d);
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
                if (q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble plegate"; return false; }
                npu_gemm(*ctx_[i_pg], x.data(), g->K, g->N, B_.data(), pg);
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
                if (q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble pleproj"; return false; }
                npu_gemm(*ctx_[i_pp], pg.data(), g->K, g->N, B_.data(), pp);
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
    if (!run_lm_head(xn, logits)) return false;
    // MiniCPM4 logit_scale: fitted against the llama.cpp oracle, this forward's
    // logits were ~16x the oracle's (oracle top-1 prob 0.928 vs effectively
    // one-hot here) and the best-fit temperature was 0.060 ~ 1/16.7 against a
    // GGUF logit_scale of 16.  Constant divisor -> cannot move the argmax.
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
    const int i_qkv = idx_of(Q4NX_DESIGN_QKV);
    const int T = designs_[i_qkv].N;                 // reuse the QKV geometry
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
