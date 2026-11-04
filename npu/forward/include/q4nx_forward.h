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
#include "plan/plan.h"  // plan.json (engine #799 P1): the stream plan replaces the filename probes
#include "q4nx_pack.h"        // Q4nxDesignGeom / Q4nxModelDims (derived design table)
#include "q4nx_semantics.h"   // Q4nxSemantics (config-derived architecture flags)
#include <memory>
#include <unordered_map>

struct I8Ctx;

#include "rt.h"  // xrt::device, xrt::bo: the own NPU runtime (ONEBIT_NPU_RT_DIR)

#include <chrono>
#include <cstdlib>

// ---- phase attribution (NPU_PHASE_PROF=1) --------------------------------
// Named RAII timers, one accumulator per slot.  Header-resident (inline
// variables) so that app/forward_serve.cpp -- which owns the request lifecycle
// -- times the *request* phases (accept, tokenisation, reset, prefill, decode,
// detokenisation, response) with the same machinery the layer loop uses, and a
// request can print the delta of this array so its rows sum to its wall time.
//
// Slot order matters: the first LH_LAYER_N entries are the layer-loop rows the
// per-token [lh] table ranks, the rest are request-level rows.
namespace q4nx_lh {
enum LhSlot {
    LH_LOAD = 0,        // load_layer(): per-token f32 re-dequant
    LH_LOAD_PROJ,       //   of which: GDN qkv/z/out projections
    LH_LOAD_FFN,        //   of which: dense FFN tail (gate/up/down)
    LH_LOAD_GDN_SMALL,  //   of which: alpha/beta, conv1d, norm, ssm_a/dt
    LH_ASM,             // q4nx_assemble_B()
    LH_GEMM,            // npu_gemm() total outside gdn_layer, incl. amax scan + out deq
    LH_GEMM_GDN,        //   of which: npu_gemm() time spent inside gdn_layer()
    LH_ATTN,            // attention(): rope, softmax, KV append
    LH_GDN,             // gdn_layer() total
    LH_GDN_AB,          //   of which: host alpha/beta dot products
    LH_GDN_CORE,        //   of which: delta-rule recurrence + gated rmsnorm
    LH_RMS,             // rmsnorm()
    LH_RES,             // residual adds / layer_scale
    LH_MLP,             // mlp(): SiLU/GELU * up
    LH_LAYER_N,         // ---- end of the layer-loop slots ----
    // Request-level rows (forward_serve.cpp handle_generate).
    LH_RQ_ACCEPT = LH_LAYER_N,  // body parse + validate + options/grammar
    LH_RQ_TOKENIZE,     // prompt -> token ids
    LH_RQ_LOCK,         // wait for the engine mutex
    LH_RQ_RESET,        //   of which (inside prefill): KV/conv/recurrent reset
    LH_RQ_PREFILL,      // prefill chunk loop (contains reset + weight packs)
    LH_RQ_DECODE,       // decode step() total
    LH_RQ_SAMPLE,       //   of which: grammar mask + sample_token
    LH_RQ_EMIT,         //   of which: detokenise the emitted token
    LH_RQ_RESP,         // response serialise + write
    LH_PACK_UPLOAD,     // weight BO create + pack + upload (wcache_ miss only)
    LH_HEAD_ROWS,       // run_lm_head_rows(): the head for a batched verify pass
    LH_N
};
inline bool lh_on() {
    static const bool on = std::getenv("NPU_PHASE_PROF") != nullptr;
    return on;
}
inline double g_lh[LH_N];
inline long   g_lh_n[LH_N];
inline int    g_gdn_d = 0;   // >0 while inside a gdn_layer() scope
inline bool lh_in_gdn() { return g_gdn_d > 0; }
struct LhScope {
    int  s; bool on;
    std::chrono::steady_clock::time_point t;
    explicit LhScope(int s_)
        : s(s_), on(lh_on()),
          t(on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {
        if (on && s == LH_GDN) g_gdn_d++;
    }
    void stop() {
        if (!on) return;
        g_lh[s] += std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t).count();
        g_lh_n[s]++;
        if (s == LH_GDN) g_gdn_d--;
        on = false;
    }
    ~LhScope() { stop(); }
};
// Table names, indexed by slot (shared by the per-token and per-request tables).
inline const char* lh_name(int s) {
    static const char* const kn[LH_N] = {
        "load_layer", "  of-which: GDN proj deq", "  of-which: FFN tail deq",
        "  of-which: GDN small deq", "assemble_B", "npu_gemm (total)",
        "  of-which: gemm in gdn", "attention", "gdn_layer",
        "  of-which: gdn alpha/beta", "  of-which: gdn recurrence", "rmsnorm",
        "residual adds", "mlp (act*up)",
        "request accept/validate", "tokenisation", "engine lock wait",
        "  of-which: KV/rec state reset", "prefill (chunks)",
        "decode (step loop)", "  of-which: sample+grammar",
        "  of-which: detokenise token", "response write",
        "  of-which: weight pack+upload (wcache miss)",
        "lm head (batch rows)"};
    return (s >= 0 && s < LH_N) ? kn[s] : "?";
}
}  // namespace q4nx_lh

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
    // derivation, weight decode and the layer maths all run, but npu_gemm() does a
    // plain float matmul, and **no ELF is required** — a geometry whose kernel set
    // has not been built yet can still load and answer.  This validates the whole
    // forward against the Python reference offline, which is how the C++ GDN port is
    // checked without the shared NPU (see docs/qwen35-4b-weight-format.md).
    bool init_host(const char* model_path, const std::string& artifact_dir);
    bool host_gemm() const { return host_gemm_; }
    bool ready() const { return ready_; }
    const char* last_error() const { return err_.c_str(); }

    // Clear the KV cache (start a new sequence).
    void reset();

    // A prefix of the sequence in host memory, so a request that shares it can skip its prefill
    // (app/npu_prefix_cache.h, issue #650).  Attention KV is per layer, per position [NKV][HD]; a
    // GDN layer additionally carries a conv window and a recurrent state across positions, which is
    // why a snapshot is not just kcache_/vcache_ (reset() had the same bug once).
    struct StateSnapshot {
        int tokens = 0;                     // positions covered, counted from 0
        size_t per_pos = 0;                 // NKV*HD floats per layer per position
        std::vector<std::vector<float>> k, v, conv, rec;  // [layer] -> the used part
        size_t bytes() const;
    };
    // Copy the state of the first `tokens` positions out.  False when the forward is not ready or
    // `tokens` exceeds what it holds.
    bool snapshot_state(int tokens, StateSnapshot& out) const;
    // Replace the state with the first `tokens` positions of `in`: the caller then continues at
    // position `tokens`.  Restoring fewer positions than the snapshot holds is the normal case (a
    // request resumes to just short of its own prompt and re-runs the last token), and restoring
    // more than asked would append a duplicate of that token.  False when the snapshot does not fit
    // this forward's geometry.
    bool restore_state(const StateSnapshot& in, int tokens);

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

    // The same batched pass, but returning the logits of EVERY row (M x NV, row-major) instead of only
    // the last one.  Speculative decoding needs them: a draft token is accepted only if the model's own
    // choice AT THAT POSITION is that token, so the verifier has to see each row's logits, not just the
    // tail of the batch.  The layer maths is the batched body step_batch already runs -- only the final
    // norm and the LM head are applied per row, and the head is launched once for all M rows (the M=16
    // twin streams its weights once for the batch, exactly like the layers).
    bool step_batch_all(const std::vector<int>& tokens, int start_pos,
                        std::vector<float>& logits_all);

    // Public: a speculative driver outside this class begins a verify batch, then commits a prefix of it.
    void begin_verify(int rows, int start_pos);
    bool commit_verify_rows(int accepted);
    bool gdn_apply_rows(int from, int to);      // recurrence-only replay of stored rows [from, to)


    // The last position's hidden state after the final norm, H floats: what a d3 readout head pools
    // (`pooling: last` in a pack's decision_config.json, npu/decision.h). Filled by the last
    // successful step() or step_batch(), empty before that. This is the vector the LM head reads, not
    // the raw residual, because that is what a packed decision head was trained against.
    const std::vector<float>& last_hidden() const { return last_hidden_; }
    // One norm earlier: the last row's residual before the final RMSNorm.  The MTP drafter takes one of the
    // two as its hidden input; which one it wants is measured, not assumed.
    const std::vector<float>& last_residual() const { return last_residual_; }
    // Logits of an arbitrary hidden vector through the SAME lm head this forward uses -- the drafter has no
    // lm_head of its own, so this is how it turns its own hidden state into a token.
    bool lm_head_logits(const std::vector<float>& h, std::vector<float>& logits) { return run_lm_head(h, logits); }
    // The post-final-norm hidden of EVERY row of the last batched pass (M x H, row-major): a speculative drafter
    // has to advance its own stream over the rows that were actually accepted, and those are exactly these.
    const std::vector<float>& last_hidden_rows() const { return last_hidden_rows_; }
    // One row of the embedding table (what a token means to the model): the MTP drafter's `pre_fc_norm_embedding`
    // input.  It reads the same table the forward uses, so the drafter cannot drift from the model's embeddings.
    void embed_row(int token, std::vector<float>& out) const {
        out.resize((size_t)c_.H);
        for (int k = 0; k < c_.H; k++) out[(size_t)k] = embed_at(token, k);
    }

    // Rows per batched launch. The activation BO holds exactly MD rows and the
    // per-op kernels are built M=MD wide, so a batch must not exceed this (the
    // batched quantizer refuses batches wider than MD rather than corrupting).
    int max_batch_rows() const;

    // Which artifacts were used, for reporting.
    const std::string& backend() const { return backend_; }

private:
    bool load_norms();
    std::vector<float> last_hidden_;  // last_hidden(): the post-final-norm state of the last position
    std::vector<float> last_residual_;  // and one norm earlier: the raw residual of the last position
    std::vector<float> last_hidden_rows_;  // [M][H] normed hidden of the last batched pass
    // Set only while step_batch_all() runs: the batched epilogue then norms every row and writes M x NV
    // logits there instead of only the last row's.  Null on every other path, so nothing else changes.
    std::vector<float>* batch_all_out_ = nullptr;
    bool init_impl(xrt::device* dev, const char* model_path,
                   const std::string& artifact_dir);
    bool host_gemm_ = false;
    bool load_layer(int l);
    // `a_k` is the activation's REAL width per row (the design's real_k, i.e. the model's hidden or
    // intermediate size).  It is K whenever the model is already on the container's 256-column K
    // grid, and smaller when it is not (Qwen2.5-0.5B: 896 of 1024), in which case the activation is
    // zero-padded to K before the kernel sees it -- see q4nx_pad_a_f32() for why that is exact.
    void npu_gemm(I8Ctx& ctx, const float* A, int K, int N,
                  const float* Bmat, std::vector<float>& out, int wkey = -1,
                  const char* sc = nullptr, int a_k = 0);
    // Batched (M>1) GEMM: M rows in one launch, with per-row activation scales.
    void npu_gemm_rows(I8Ctx& ctx, const float* A, int M, int K, int N,
                       const float* Bmat, std::vector<float>& out, int wkey = -1,
                       const char* sc = nullptr, int a_k = 0);
    // pb16i sidecar stem for (layer, tensor tag): <pb16i_dir>/blk_<l>_<tag>_weight
    std::string pb16i_stem(int layer, const char* tag) const;
    // Phase-selected activation tile.  A one-row launch is decode, so it runs
    // this design's M=16 twin (cheap single-row kernel); anything wider is
    // prefill and keeps the M=128 kernel so a chunk still lands in one array
    // pass.  Returns `prefill` unchanged when the twin is unavailable, so the
    // callers never have to test for it.
    I8Ctx* ctx_for(I8Ctx* prefill, int rows);
    // Which role a context belongs to (prefill or decode twin); -1 if unknown.
    int design_index(I8Ctx* c) const;
    // Bit-equality gate for the decode family (NPU_M16_CHECK=1): run the same
    // quantized activation and the same weight BO through both kernels and
    // compare the int32 outputs, which must match exactly (int32 accumulation is
    // exact, so a difference is a bug, never rounding).  Returns the int32
    // maxdiff; records the worst case across the run.
    int32_t m16_check_gemm(I8Ctx& ref, I8Ctx& fast, const float* A, int K, int N,
                           const float* Bmat, xrt::bo* wbo, float as);
    // The same gate for a multi-row pass (a prefill chunk of <= 16 rows also runs
    // the twin): row m of both kernels must agree exactly.
    int32_t m16_check_gemm_rows(I8Ctx& ref, I8Ctx& fast, const float* A, int M, int K,
                                int N, xrt::bo* wbo, const std::vector<float>& ascales);
    // NPU_PROFILE per-role dispatch + timing accumulators, one per phase.
    struct ProfAcc {
        double pack = 0, quant = 0, wait = 0, rb = 0;
        long   n = 0;
    };
    void prof_add(ProfAcc* acc, int design, double pack, double quant, double wait,
                  double rb);
    void prof_dump();
    bool load_hadamard(const std::string& path);
    // True when this layer's projection `wkey` is already packed in its own BO,
    // so the caller can skip both the Q4NX dequant (load_layer) and the f32->B
    // assembly for the rest of the run.
    bool weight_cached(int key) const {
        return weight_cache_ && cur_layer_ >= 0 && cur_layer_ < (int)wcache_.size() &&
               wcache_[(size_t)cur_layer_].find(key) != wcache_[(size_t)cur_layer_].end();
    }
    // True when this layer's dense FFN tail (the f32 wg_/wu_/wd_ buffers) can be
    // skipped: they are read only by q4nx_assemble_B(), which the MLP sections
    // skip once the layer's packed weights are resident.  See the definition.
    bool ffn_tail_cached() const;
    // The default for the above when NPU_FFN_CACHE says nothing: on for a GDN
    // model on the NPU route, with a size guard.  See the definition.
    bool ffn_cache_default() const;
    bool attention(const std::vector<float>& qkv, int pos, std::vector<float>& ctx,
                   int head_dim, int n_kv_heads, float theta, int rotary_dim,
                   bool proportional, int window, int kv_layer, bool append_kv,
                   int kv_len = 0);
    bool mlp(const std::vector<float>& gu, std::vector<float>& h, int im);
    float rmsnorm(const float* x, const float* w, int n, float* out) const;
    float rmsnorm_noweight(const float* x, int n, float* out) const;
    bool run_lm_head(const std::vector<float>& h, std::vector<float>& logits);
    // M rows of hidden state -> M x NV logits, row-major.  Same head, same packed tiles: the rows ride
    // one launch (per-row activation scales, which the batched launcher and the dequantiser already carry).
    bool run_lm_head_rows(const std::vector<float>& h_rows, int M, std::vector<float>& logits_all);

    // ── the gemv (forge q4_1 GEMV) weight image ────────────────────────────────────────────────
    // I8Ctx detects that family from each ELF's declared argument widths; when it is in use the
    // weight BO holds the container's own q4_1 tiles instead of an int8 packing, because the kernel
    // dequantises with each tile's bf16 scale/zero and repacking would throw those away.
    static int role_from_tag(const char* sc);
    bool gemv_tiles(int role, std::vector<uint8_t>& out);
    bool run_lm_head_gemv(const std::vector<float>& h, std::vector<float>& logits);

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
    // M=16 decode twins of the same roles (empty/skipped where no ELF or no legal
    // geometry).  Index-aligned with designs_/ctx_ so ctx_for() can pair them.
    Q4nxDesignGeom designs16_[Q4NX_MAX_DESIGNS];
    std::unique_ptr<I8Ctx> ctx16_[Q4NX_MAX_DESIGNS];
    bool m16_ = false;          // at least one role has a working twin
    bool m16_check_ = false;    // NPU_M16_CHECK=1: cross-check every decode launch
    // Weight image "pb16i" (container config q4nx_weight_layout): the decode family
    // reads sidecar images instead of packing int8 from the f32 weights, so no
    // f32 weight buffer is needed for a covered projection at all.  pb16i_dir_ is
    // where blk_<layer>_<tag>_weight.{pb16i,ref} live (config q4nx_pb16i_dir, env
    // Q4NX_PB16I_DIR wins).  Empty = the layout is off and nothing changes.
    bool pb16i_ = false;
    // Weight stream "t2s" (container config q4nx_weight_layout "t2s"): pb16i's integer-exact
    // ternary contract at 2.125 bits/weight -- 2-bit codes expanded on the core, one multiplier per
    // 128 K instead of four.  The image is `<stem>.t2s` at q4nx_t2s's size, the side table is pb16i's
    // `.ref` unchanged, and the kernels carry the `_t2s` suffix.  pb16i_ is set too: everything
    // downstream of the load is the same path, and only the file it reads differs.
    bool t2s_ = false;
    // Which t2s wire order the container's images are in (q4nx_weight_layout_version 5 = v1, 6 = v2).  The two
    // hold the same weights in different byte orders and are the same size, so they are told apart by the
    // *artifact family*: v1 decode kernels are `_t2s_m16`, v2's are `_t2s2_m16`, and pairing one with the other
    // is a silent permutation of the matrix rather than an error.  v2 is the one to ship: it is the same
    // weights with the ordering chosen so the core's two contiguous stores are exact, i.e. no lane shuffle
    // (measured on the array: 0.589 ms against the pb16i phase's 1.037 ms, 40.2 GB/s).
    bool t2s_v2_ = false;
    std::string pb16i_dir_;
    // The artifact suffix and sidecar extension of the container's weight family, in one place so a
    // kernel name and the image it reads cannot drift apart.
    const char* family_tag() const {
        if (t2s_) return t2s_v2_ ? "_t2s2" : "_t2s";
        return pb16i_ ? "_pb16i" : (weight_layout_b8_ ? "_b8" : "");
    }
    const char* image_ext() const { return t2s_ ? ".t2s" : ".pb16i"; }
    // A sealed ELF carries every ternary instance in ONE full ELF (one PDI, one
    // hw_context, 7 kernel instances), so changing phase costs nothing: measured on this
    // box, a phase change inside the sealed context is 0 ms against ~2.7 ms between two
    // hw_contexts.  Config `q4nx_pb16i_sealed_elf` names it in the kernels dir; without
    // it the per-role `_pb16i.elf` files are used as before.
    std::string pb16i_sealed_;
    // plan.json (engine #799 P1), when the model directory carries one.  It states the three things
    // this class used to decide by looking at which files exist in --kernels: fused gate/up, the
    // artifact family (""/_b8/_pb16i), and where the M=16 decode twins live.  Absent = the legacy
    // probes below, unchanged -- that is what makes the plan additive.  plan_path_ is kept for the
    // refusals, which name the plan that asked for a kernel that is not there.
    onebit::npu::plan::Plan plan_;
    std::string plan_path_;
    // attention.mode from the plan: a noncausal pack's full-attention layers see the whole prompt, so a
    // prompt-encoding forward (step_batch at start_pos 0) attends bidirectionally there and the GDN
    // layers keep their recurrence.  Off unless the plan says so, so a causal model is untouched.
    bool noncausal_attn_ = false;
    // The refusal for a kernel a design needs but the directory does not have.  With a plan it names
    // the plan file and the role, so "which description asked for this" needs no log reading.
    std::string missing_kernel(const std::string& dir, const char* elf_name, Q4nxDesign role) const;
    //  [[hadamard]]
    // The PTQ1 fold: a folded container's weights are W_true rotated by a block-`had_block_`
    // normalized Hadamard with a per-input-width sign vector, so an activation entering one of
    // those GEMMs is rotated first, A' = H (A . s).  Config key `q4nx_hadamard` names the binary
    // sign table (magic "HAD1", u32 count, count x (u32 width, u32 offset), f32 signs).
    bool hadamard_ = false;
    int had_block_ = 1024;
    std::vector<float> had_buf_;
    double had_ms_ = 0.0;      // host cost of the rotation, accumulated for the profile line
    long had_n_ = 0;
    std::unordered_map<int, std::vector<float>> had_signs_;   // input width -> signs
    // Full-attention QKV as pb16i: the one 14336-wide fused launch has no kernel
    // (its int64 q needs 448 columns per core), so it runs as the two q halves plus
    // k and v -- the four launches the arena measured at 1.49 ms per layer.
    int i_qhalf_ = -1, i_kv_ = -1;
    // The lm head is not a PTQ1_0 tensor (the container stores it I8), so it has no
    // pb16i image and keeps an int8 context of its own: the M=16 twin of the widest
    // K=H design when one is on disk, else the M=128 ELF.  Without this, a pb16i
    // container ran the head on the M=128 kernel and paid 30 ms a launch (~0.55 s a
    // token, measured 2026-10-31 08:00).
    std::unique_ptr<I8Ctx> ctx_head_;
    int32_t m16_worst_ = 0;     // worst int32 maxdiff seen by the check
    long m16_bad_ = 0, m16_checked_ = 0;
    std::vector<int32_t> m16_ref_;   // reference C row, reused per check
    bool prof_ = false;              // NPU_PROFILE=1
    ProfAcc prof_dec_[Q4NX_MAX_DESIGNS], prof_pre_[Q4NX_MAX_DESIGNS];
    // The lm head has no design of its own (it borrows a K=H geometry and, under pb16i, an
    // int8 context of its own), so it gets its own row: attributing it to a design put
    // 15-16 launches of ~27 ms into the G row and made any per-role budget wrong by 0.4 s.
    ProfAcc prof_head_;
    int     prof_head_M_ = 0;
    long prof_dec_n_ = 0;            // decode launches (dispatches/token)
    long prof_pre_n_ = 0;
    long prof_tok_n_ = 0;
    int  prof_tick_ = 0;
    // Index of the design playing a role, or -1 when the model has no such
    // kernel (e.g. GU is absent on split models, G/U absent on fused ones).
    int idx_of(Q4nxDesign role) const;

    // The container stores every q4_1/Q8 tensor on the chunk's 256-column K grid (npu/q4nx.h
    // kCols), so the projection width a container tensor is dequantised at is the PADDED hidden
    // size (engine#845: Qwen2.5-0.5B's 896 is stored as 1024 columns, zeros in the pad).  The
    // pad columns are zero in the container, so they add exactly 0 to every dot product.  The
    // designs' K comes out of the same grid (q4nx_pad_k), which is what makes assemble_B's source
    // stride and this width the same number.
    int kpad_ = 0;        // c_.H on the container's K grid
    int impad_ = 0;       // c_.IM on the container's K grid
    // Row stride of the f32 table embed_at()/lm_at() read: the padded width when the table was
    // dequantised from a container tensor, c_.H when it is the raw BF16 embedding.
    int lm_w_ = 0;
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
    // Container-declared device weight layout (q4nx_weight_layout == "b8").
    // False is the row-major layout every earlier container has.
    bool weight_layout_b8_ = false;    std::vector<float> lm_head_tiles_;
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
        // pb16i: the sidecar's host side table.  Empty eref = an int8 pack.
        std::vector<int16_t>     eref;
        int                      pb16i_mode = 0;
    };
    std::vector<std::unordered_map<int, PackedWeight>> wcache_;   // [layer][wkey]
    bool weight_cache_ = true;    // NPU_NO_WCACHE=1 disables (A/B)
    // Skip the per-token re-dequant of a layer's dense FFN tail once its packed
    // weights are resident (see ffn_tail_cached()).  -1 asks
    // ffn_cache_default(), which is on for a GDN model on the NPU route; init
    // resolves it to 0 or 1.  NPU_FFN_CACHE=0|1 forces it either way.
    int ffn_cache_ = -1;
    // The NPU LM head packs its B tiles ONCE into their own weight BOs, one per
    // tile ([H][T] int8 + T per-column scales), instead of re-copying a 356 MB f32
    // tile and re-running packB for every tile of every token.  With these
    // resident the f32 head (lm_head_f_, 5.09 GB) and the f32 tile gather
    // (lm_head_tiles_, 5.35 GB) are both dead on the NPU path and are released.
    std::vector<PackedWeight> lm_head_packed_;
    int                       lm_head_packed_T_ = 0;
    // gemv family: the head's tile stream is the whole image in one BO, copied once.
    std::vector<uint8_t>      gemv_blob_;   // one design's tile stream, reused per upload
    std::vector<uint8_t>      gemv_head_;
    // The plan's resolved decode directories, when it names any. The head ELF lives in the same place as
    // the projections' ELFs, so it is looked for there rather than in the built-in guesses only.
    std::vector<std::string>  plan_decode_dirs_;
    std::unique_ptr<xrt::bo>  head_bo_;
    // The lm head on ternary pb16i images (conv/pack_head_pb16i.py): output.weight is PTQ1_0 in
    // the folded GGUF, so the image is exact, it is half the int8 codes' bytes, and it runs the
    // same kernel the MLP's GU launch runs.  Off -- and the head keeps the int8 family -- when
    // the images or the (K=H, N=<tile>) pb16i ELF are not on disk.
    bool                      head_pb16i_ = false;
    std::string head_pb16i_stem(int tile) const {
        return pb16i_dir_ + "/head_t" + std::to_string(tile) + "_weight";
    }
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
    // GDN members the HOST reads every token: the alpha/beta projections (the
    // per-head dot products), the conv window, the gate norm and ssm_a/ssm_dt
    // (the recurrence).  They are container constants, but the recurrence consumes
    // a different one per layer, so a single shared buffer forced a re-dequant of
    // all of them for every layer of every token: 299 ms/token on the 27B, 53% of
    // all layer host time, for ~2 MB/layer.  Per layer, dequantised once.
    // NPU_GDN_CACHE=0 restores the per-token reload.
    struct GdnSmall {
        std::vector<float> wa, wb;        // [nvh][H]
        std::vector<float> conv;          // [CK][CD]
        std::vector<float> norm;          // [VHD]
        std::vector<float> ssm_a, dt;     // [nvh]
        bool loaded = false;
    };
    std::vector<GdnSmall> gdn_small_;     // [layer]
    bool gdn_cache_ = true;               // NPU_GDN_CACHE=0 disables
    std::vector<std::vector<float>> conv_state_, rec_state_;  // [CD][CK-1], [nvh][khd][vhd]
    // One GDN token step: xn is the normed input; out gets the out_proj result.
    bool gdn_layer(int l, const std::vector<float>& xn, std::vector<float>& out);
    // ---- deferred GDN commit (speculative verify) ------------------------------------------------
    // A verify batch runs k draft rows through the layers, so the recurrence is advanced over rows the
    // verifier may reject.  A snapshot is not affordable (per layer the recurrent state is
    // NVH*KHD*VHD floats; for the 27B that is 3 MB x 48 layers) and attention needs no rollback at all
    // (it is position-indexed).  So: snapshot once per verify batch, keep every row's conv + gate inputs
    // as the batch runs, and on commit() restore and re-apply ONLY the accepted rows.  The re-apply is the
    // recurrence alone -- no projections, no GEMMs -- which is why it is cheap compared with the passes it
    // saves.
    struct GdnRowIn {
        std::vector<float> qkv;   // [CD] the conv input of this row (to advance the conv ring)
        std::vector<float> ah, bh; // [NVH] the gate inputs, so the decay/beta are reproduced exactly
    };
    // verify-batch deferral (see begin_verify): the snapshots are taken once per batch and reused
    bool verify_open_ = false;
    int verify_rows_ = 0, verify_pos_ = 0;
    std::vector<std::vector<float>> snap_conv_, snap_rec_;
    std::vector<std::vector<GdnRowIn>> row_in_;   // [layer][row]

    bool gdn_layer_batch(int l, const std::vector<float>& xn, int M,
                         std::vector<float>& out);
    bool load_gdn_layer(int l);
    std::string err_;
    std::string backend_ = "uninitialised";
    bool ready_ = false;
};

#endif  // NPU_INFER_Q4NX_FORWARD_H
