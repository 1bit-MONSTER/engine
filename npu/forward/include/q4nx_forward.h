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

#include "moe_container.h"  // the two MoE container conventions (engine#1035)
#include "model.h"
#include "plan/plan.h"  // plan.json (engine #799 P1): the stream plan replaces the filename probes
#include "q4nx_pack.h"        // Q4nxDesignGeom / Q4nxModelDims (derived design table)
#include "q4nx_rope.h"        // rope_type "llama3" frequency scaling (engine#1048)
#include "q4nx_semantics.h"   // Q4nxSemantics (config-derived architecture flags)
// The launch layer: this class owns the maths and the model state, the target owns the device, the kernels
// and the weight residency (engine#799 P4).  A future non-XDNA NPU is a different implementation of the same
// interface (npu/target/target.h) and nothing else in this file changes.
#include "target/xdna/xdna_target.h"  // npu-relative: CI's syntax-only target has no tree root on its path
#include "model/npu_model.h"   // NpuModel owns the target (engine#803 A3)
#include <memory>
#include <unordered_map>

struct I8Ctx;


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
#include "q4nx_lh.h"   // the phase accounting both the forward and a target write

// Q4nxForwardConfig now lives with the model that owns it: npu/model/npu_model.h (engine#803 A3).

// engine#1740 W2: the rope_type "dynamic" reader, exposed so a host test can pin it against the exact block the
// Hunyuan 1.9 packs carry ({"type": "dynamic", "factor": 1.0} with max_position_embeddings 262144) without a device.
Q4nxRopeDynamic read_rope_dynamic(const std::string& cfg_path);

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
    // attention.mode as the loaded plan states it: false unless the plan says
    // noncausal_full_attention, so a caller can check the served pack against its own declaration
    // (app/forward_serve.cpp, engine#796).
    bool noncausal() const { return noncausal_attn_; }
    // engine#2102: what the decode family turned out to be, so a caller can say it out loud. `decode_roles()` is the number of
    // roles that got an M=16 kernel, `design_roles()` the number of roles the model has. 0 of N is the state that serves at a
    // fraction of the fused route's rate while printing nothing an operator would notice at the ready line: every decode step
    // then runs prefill-width launches. The values are recorded where the family is opened (open_decode_family_local), not
    // inferred later, because the derived designs are not kept per role.
    int decode_roles() const { return decode_roles_; }
    // The host-GEMM arm loads no decode family, so it reports no roles rather than "0 of N" (engine#2102).
    int design_roles() const { return host_gemm_ ? 0 : ndesigns_; }
    bool decode_disabled() const { return decode_disabled_; }   // NPU_DECODE_M16=0
    bool decode_image() const { return pb16i_; }                // a ternary (pb16i/t2s) container's decode family
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
    // `pos` is the ROTARY position; `cache_pos` is the slot the new K/V lands in, which is the only thing that
    // decides how many cached rows attention reads.  They are the same number only while positions equal indices:
    // after an image they are not (an image's rows walk a small h/w grid, so the MRoPE position is far below the
    // cache length), and using the rotary position as the window dropped the prompt's tail -- the question -- and
    // every token after the first never saw what it had generated.  -1 keeps the old behaviour for text prompts.
    bool step(int token, int pos, std::vector<float>& logits, int cache_pos = -1);

    // Batched prefill: run the whole prompt in one pass so each projection's
    // weight stream is read once per batch instead of once per token. The kernels
    // are M=128 wide, so a per-token step leaves ~99% of the array idle. Only the
    // simple dense path is batched; every other architecture falls back to
    // per-token step() (see the guard in the implementation).
    bool step_batch(const std::vector<int>& tokens, int start_pos,
                    std::vector<float>& logits);

    // The same batched pass, fed **precomputed embeddings** instead of token ids (engine#1311/TQVL, and the
    // shape AMD's reference pack exposes as `embedding.onnx` -> `inputs_embeds` -> `text.onnx`).
    //
    // `embeds` is rows x H row-major, each row the model's own embedding for that position **before**
    // `embed_scale` (the forward applies its own scale, as it does on the token path), so a caller that builds
    // rows with embed_row() gets byte-identical logits to step_batch(ids) -- npu/forward/embeds_equiv_probe.cpp
    // (target `embeds_equiv_probe`, run under npu-hold or with NPU_INFER_HOST_GEMM=1) checks exactly that. This exists so a hybrid can put image-feature rows where the image tokens are, without the
    // decoder needing to know what a vision encoder is; it is NOT a way to skip the embedding table for text.
    // The per-token fallback path (MLA/MoE/GDN/... models) needs ids and is refused by name.
    bool step_batch_embeds(const std::vector<float>& embeds, int start_pos,
                           std::vector<float>& logits);
    // What a vision-conditioned prompt adds on top of the embeddings (engine#1311/TQVL; plan P4's `rope.planes`
    // and a per-layer feature injection):
    //   position_ids 3 x rows, plane-major (t, h, w) — the layout the reference's `get_rope_index` produces and
    //                AMD's `text.onnx` takes. Text rows have all three equal, which is bit-identical to plain RoPE
    //                (measured: 0 difference over positions 0..262143; image rows differ per axis and that is the
    //                whole point). A pack that declares mrope without sections is refused by name.
    //   deepstack   sets x rows x H, one set per `vision_config.deepstack_visual_indexes` entry (Qwen3-VL: 5/11/17);
    //                set s is added to the hidden state when the layer loop reaches deepstack_layers[s]. This is how
    //                the reference injects the vision tower's extra features, and an answer cannot match it without.
    struct HybridInputs {
        const std::vector<int>* position_ids = nullptr;
        const float* deepstack = nullptr;
        int deepstack_sets = 0;
        // The rows the deepstack sets belong to: the image span. They are added THERE, not everywhere -- a set has
        // rows_img rows, so adding it to every row of the prompt would both be the wrong function and read past the
        // buffer (the first version of this did exactly that and produced garbage tokens).
        int image_lo = -1;
        int image_hi = -1;
    };
    const int* deepstack_layers() const { return sem_.deepstack_layers; }
    int n_deepstack_layers() const { return sem_.n_deepstack; }
    bool step_batch_embeds(const std::vector<float>& embeds, int start_pos, std::vector<float>& logits,
                           const HybridInputs& in);

    // The same batched pass, but returning the logits of EVERY row (M x NV, row-major) instead of only
    // the last one.  Speculative decoding needs them: a draft token is accepted only if the model's own
    // choice AT THAT POSITION is that token, so the verifier has to see each row's logits, not just the
    // tail of the batch.  The layer maths is the batched body step_batch already runs -- only the final
    // norm and the LM head are applied per row, and the head is launched once for all M rows (the M=16
    // twin streams its weights once for the batch, exactly like the layers).
    //   A noncausal pack (`attention.mode noncausal_full_attention`) is refused by name: in such a batch
    // every row attends to the rows after it, so these per-row logits are not the prefix-conditioned ones a
    // draft is checked against.  See docs/npu.md, "attention.mode".
    bool step_batch_all(const std::vector<int>& tokens, int start_pos,
                        std::vector<float>& logits_all);

    // Public: a speculative driver outside this class begins a verify batch, then commits a prefix of it.
    void begin_verify(int rows, int start_pos);
    bool commit_verify_rows(int accepted);
    // How many commits took the full-accept path (kept the batch's own state instead of restoring and replaying it).
    // A test that compares the two paths reads it to know the path it meant to test is the one that ran.
    long full_accepts_kept() const { return full_accepts_kept_; }
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
    // engine#1780: the residual stream after the listed layers, for a draft that reads the target's hidden states
    // (a DFlash draft's target_layer_ids). `layers` must be ascending and inside the model; every pass then appends
    // one row per position to *sink, [taps * H] wide, the listed layers side by side in the order given -- the
    // layout dflash_forward.h's `taps` argument takes. Layer l's row is the stream as layer l + 1 receives it (after
    // layer l's own closing scale). Works for step() and for the batched passes. An empty list or a null sink turns
    // it off; the sink must outlive the passes. Returns false, changing nothing, on a list that is not usable.
    // A pass that returns false appends nothing: whatever it had written is taken back, so the sink only ever holds
    // rows of passes that succeeded. A call refused earlier on the same forward does not affect later passes.
    // Not for use under NPU_BATCH_CHECK: that oracle replays positions through step(), and the replays would be
    // appended as if they were new positions.
    bool set_layer_taps(const std::vector<int>& layers, std::vector<float>* sink);
    int layer_count() const { return c_.NL; }   // layers of the loaded model; 0 before init
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
    long                full_accepts_kept_ = 0;   // commit_verify_rows(): see full_accepts_kept()
    std::vector<int>    tap_layers_;           // engine#1780: set_layer_taps()
    std::vector<float>* tap_sink_ = nullptr;
    size_t              tap_base_ = 0;         // where the current pass's rows start in *tap_sink_
    bool                tap_fault_ = false;    // a tapped layer arrived with a different row count than the pass opened with
    void tap_layer(int layer, const float* X, int M);
    // The bodies of step() and step_batch(). The public functions wrap them so that a failed pass leaves the tap
    // sink as it found it (layer_taps.h, LayerTapPass).
    bool step_impl(int token, int pos, std::vector<float>& logits, int cache_pos);
    bool step_batch_impl(const std::vector<int>& tokens, int start_pos, std::vector<float>& logits);
    std::vector<float> last_hidden_rows_;  // [M][H] normed hidden of the last batched pass
    // Set only while step_batch_all() runs: the batched epilogue then norms every row and writes M x NV
    // logits there instead of only the last row's.  Null on every other path, so nothing else changes.
    std::vector<float>* batch_all_out_ = nullptr;
    // engine#803 A2b: the model's half of the decode family (design names, summary lines, the wide and narrow
    // sealed families); the per-role ELF resolution and context creation is XdnaTarget::open_decode_family().
    bool open_decode_family_local(bool want_m16, xrt::device* dev, const std::string& mdir,
                                  const std::string& dir);

    bool init_impl(xrt::device* dev, const char* model_path,
                   const std::string& artifact_dir);
    bool host_gemm_ = false;
    // NPU_INFER_HOST_GEMM=i8 (engine#1808): the host arm accumulates int8 x int4 in EXACT int32, which is
    // the only arm a kernel family can be bit-exact against -- f32 cannot match an int8 device by
    // construction. It cannot see a wrong weight layout (the f32 arm covers that); it can see a wrong
    // quantisation, a wrong 128-K group boundary, an int8 intermediate or a saturated accumulator.
    bool host_gemm_i8_ = false;
    // Weight codes + per-128-K multipliers for the integer host arm, quantised once per tensor (engine#1809).
    // `blocked` is the layout `codes` was packed in: blocked only when K x N is whole tiles (engine#1994).
    struct HostI8Tensor { int groups = 0; int N = 0; bool blocked = true; std::vector<int8_t> codes; std::vector<float> d; };
    // Keyed on (LAYER, wkey, K, N): `wkey` alone is a per-role slot that repeats in every layer and `Bmat` is the one
    // scratch buffer each layer is decoded into, so keying on it served layer 0's weights to every later layer
    // (engine#1830 review). Routed experts pass wkey < 0 and are never cached: their weights change per token.
    std::map<std::tuple<int, int, int, int>, HostI8Tensor> host_i8_cache_;
    std::vector<HostI8Tensor> lm_head_i8_;   // integer arm's head image, one per vocabulary block
    void host_gemm_i8(const float* A, int a_k, int M, int K, int N, const float* Bmat, int wkey, float* out);
    bool load_layer(int l);
    // `a_k` is the activation's REAL width per row (the design's real_k, i.e. the model's hidden or
    // intermediate size).  It is K whenever the model is already on the container's 256-column K
    // grid, and smaller when it is not (Qwen2.5-0.5B: 896 of 1024), in which case the activation is
    // zero-padded to K before the kernel sees it -- see q4nx_pad_a_f32() for why that is exact.
    // One projection GEMM. `role` indexes the design table; the launch itself (phase-selected M=16/M=128
    // kernel, packed-weight BO cache, pb16i images, the Hadamard fold, the a_k pad) lives in the target
    // (npu/target/xdna/xdna_target.h).  The host-GEMM path of init_host() stays here: it is the model's
    // fallback, not the target's op.
    void npu_gemm(int role, const float* A, int K, int N, const float* Bmat,
                  std::vector<float>& out, int wkey = -1, const char* sc = nullptr, int a_k = 0);
    // Batched (M>1) GEMM: M rows in one launch, with per-row activation scales.
    void npu_gemm_rows(int role, const float* A, int M, int K, int N, const float* Bmat,
                       std::vector<float>& out, int wkey = -1, const char* sc = nullptr,
                       int a_k = 0, bool decode_twin = false);
    // True when this layer's projection `wkey` is already resident in the target.
    bool weight_cached(int key) const { return xd_ && xd_->weight_cached(cur_layer_, key); }
    // True when this layer's dense FFN tail (the f32 wg_/wu_/wd_ buffers) can be
    // skipped: they are read only by q4nx_assemble_B(), which the MLP sections
    // skip once the layer's packed weights are resident.  See the definition.
    bool ffn_tail_cached() const;
    // The default for the above when NPU_FFN_CACHE says nothing: on for a GDN
    // model on the NPU route, with a size guard.  See the definition.
    bool ffn_cache_default() const;
    bool attention(const std::vector<float>& qkv, int pos, const int* pos3, std::vector<float>& ctx,
                   int head_dim, int n_kv_heads, float theta, int rotary_dim,
                   bool proportional, int window, int kv_layer, bool append_kv,
                   int kv_len = 0, std::vector<float>* q_out = nullptr);
    bool mlp(const std::vector<float>& gu, std::vector<float>& h, int im);
    // The same elementwise gate*up on caller memory, so a batched pass can read its ACT row and write its HH
    // row where they already are instead of staging both through a vector (a copy on each side).
    bool mlp(const float* gu, float* h, int im);
    // `tag` is the NPU_INFER_DUMP_OPS label (ops-dump.h); the per-head q/k norms pass their own.
    float rmsnorm(const float* x, const float* w, int n, float* out, const char* tag = "rmsnorm") const;
    float rmsnorm_noweight(const float* x, int n, float* out) const;
    bool run_lm_head(const std::vector<float>& h, std::vector<float>& logits);
    // M rows of hidden state -> M x NV logits, row-major.  Same head, same packed tiles: the rows ride
    // one launch (per-row activation scales, which the batched launcher and the dequantiser already carry).
    bool run_lm_head_rows(const std::vector<float>& h_rows, int M, std::vector<float>& logits_all);

    // ── the gemv (forge q4_1 GEMV) weight image ────────────────────────────────────────────────
    // I8Ctx detects that family from each ELF's declared argument widths; when it is in use the
    // weight BO holds the container's own q4_1 tiles instead of an int8 packing, because the kernel
    // dequantises with each tile's bf16 scale/zero and repacking would throw those away.
    //
    // The image is container state (the chunks live in `mw_`), so it is built here and handed to the
    // target through GemvArgs::tiles on a residency miss; the launch is the target's.  main's
    // role_from_tag() is gone with the move: GemvArgs::role already is the design index, which is the
    // one thing that mapping was producing.
    bool gemv_tiles(int layer, int role, std::vector<uint8_t>& out, std::string& err);
    static bool gemv_tile_cb(void* self, int layer, int role, std::vector<uint8_t>& out,
                             std::string& err);
    bool run_lm_head_gemv(const std::vector<float>& h, std::vector<float>& logits);

    ModelWeights* mw_ = nullptr;
    xrt::device* dev_ = nullptr;

    // The model (engine#803 A3) owns its state; the members below that it owns are references into it, so every use
    // site is unchanged. Declared FIRST among the owned state: a reference member is initialised from something that
    // already exists.
    npu::NpuModel model_;

    // Architecture semantics derived from the model's own config.json.  Models
    // whose config asks for layer types this forward does not implement (hybrid
    // linear/sliding attention, per-layer-type RoPE, mRoPE) are REFUSED in
    // init() rather than silently run with the wrong math.
    Q4nxSemantics & sem_ = model_.sem;
    // The model owns its config too; bound here rather than where the value member used to sit, because a reference
    // member is initialised in DECLARATION order — above `model_` this bound a subobject of an object whose lifetime
    // had not started (latent UB that only worked because the address is fixed).
    Q4nxForwardConfig & c_ = model_.c;

    // Design set DERIVED from c_ (no model-specific geometry constant): four
    // kernels for fused gate/up, five when gate and up are split.
    Q4nxDesignGeom designs_[Q4NX_MAX_DESIGNS];
    int ndesigns_ = 0;
    bool decode_disabled_ = false;   // engine#2102: NPU_DECODE_M16=0 turned the family off
    int decode_roles_ = 0;   // engine#2102: roles with an M=16 decode kernel, as open_decode_family_local() found them
    // M=16 decode twins of the same roles (empty/skipped where no ELF or no legal geometry).  The geometry
    // stays here because the model needs it (max_batch_rows, the layer loop's row count); the contexts
    // themselves live in the target.
    Q4nxDesignGeom designs16_[Q4NX_MAX_DESIGNS];
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
    std::string wide_sealed_;       // q4nx_<layout>_sealed_wide_elf: a wide prefill family (engine#1990)
    int wide_sealed_rows_ = 32;    // q4nx_sealed_wide_rows
    std::string narrow_sealed_;     // q4nx_<layout>_sealed_narrow_elf: a narrow decode family
    int narrow_sealed_rows_ = 4;    // q4nx_sealed_narrow_rows
    std::string sealed_dir_;        // the kernels dir the 16-row sealed ELF was found in
    // ---- the NPU target (engine#799 P4) ----
    // Everything that talks to a device lives behind this: the ELF contexts (M=128 prefill and the M=16
    // decode twin), the phase-selected launch, the packed-weight BO cache, the pb16i sidecar images, the
    // Hadamard fold of the activation, the M=16 bit-equality gate and the NPU_PROFILE accumulators.
    // A1 keeps the design table and the container parsing here; A2 (docs/npu-p4-target-split.md) hands the
    // target a plan instead.  `pb16i_dir_` above stays because the lm head's own images are named from it.
    // The model (engine#803 A3) owns the target; this is a reference to it, so the ~94 `xd_->…` call sites and every
    // expression in this class stay exactly as they were.
    std::unique_ptr<npu::target::xdna::XdnaTarget> & xd_ = model_.xd;
    // plan.json (engine #799 P1), when the model directory carries one.  It states the three things
    // this class used to decide by looking at which files exist in --kernels: fused gate/up, the
    // artifact family (""/_b8/_pb16i), and where the M=16 decode twins live.  Absent = the legacy
    // probes below, unchanged -- that is what makes the plan additive.  plan_path_ is kept for the
    // refusals, which name the plan that asked for a kernel that is not there.
    //
    // P4 (this branch) hands the same plan to the target: XdnaTarget::open(plan, dir) checks the
    // geometry it was given against it, so a plan and a launch layer cannot disagree silently.
    onebit::npu::plan::Plan plan_;
    std::string plan_path_;
    // attention.mode from the plan: a noncausal pack's full-attention layers see the whole prompt, so a
    // prompt-encoding forward (step_batch at start_pos 0) attends bidirectionally there and the GDN
    // layers keep their recurrence.  Off unless the plan says so, so a causal model is untouched.
    bool noncausal_attn_ = false;
    // engine#1959: measured wall of one batched launch per design, narrow (the 16-row twin) and wide (the prefill
    // design), for cached weights only. npu_gemm_rows() reads them to decide whether a 17..wide-row chunk runs as one
    // wide launch or as several narrow ones (batch_split.h). 0 = not measured yet. `split_mode_` is NPU_PREFILL_SPLIT:
    // -1 unset (decide from the measurements), 0 never split (the behaviour before), 1 split whenever a twin exists.
    double split_narrow_ms_[Q4NX_MAX_DESIGNS] = {0};
    double split_wide_ms_[Q4NX_MAX_DESIGNS] = {0};
    int    split_wide_n_[Q4NX_MAX_DESIGNS] = {0};   // wide launches timed so far (batch_split.h: kSplitWideSamples)
    int    split_mode_ = -2;      // -2 = environment not read yet
    bool   split_said_ = false;
    // The refusal for a kernel a design needs but the directory does not have.  With a plan it names
    // the plan file and the role, so "which description asked for this" needs no log reading.
    std::string missing_kernel(const std::string& dir, const char* elf_name, Q4nxDesign role) const;
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
    std::vector<uint16_t> embed_;                 // unused since engine#1784 (the table is read in place); always empty
    /* engine#1784: the token embedding table, read in place from the container's read-only mapping, one row per token.
     * These point INTO that mapping, which is owned by `model_` and outlives the forward -- they are views, never
     * owners, and must not be freed or written. Empty/null means the dtype goes through embed_f_ (the chunked formats
     * that have no element-wise read yet). */
    const uint16_t* embed_bf16_   = nullptr;      // bf16 table, row stride c_.H
    const int8_t*   embed_i8_     = nullptr;      // plain int8 table, row stride c_.H
    const float*    embed_sc_     = nullptr;      // per-32-column scales, H/32 per row (may be null: unscaled)
    int             embed_groups_ = 0;            // H / Q4NX_GROUP_SIZE
    /* The chunked formats (tiled), read one row per token out of the same mapping: `embed_row_buf_` holds the row
     * currently gathered and `embed_row_tok_` which token it is for (mutable: embed_at() is const and the callers walk
     * k innermost, so a row is gathered once and served H times). engine#1784. */
    const uint8_t*  embed_tile_        = nullptr;
    size_t          embed_tile_bytes_  = 0;
    int             embed_tile_rows_   = 0;
    int             embed_tile_cols_   = 0;
    int             embed_tile_cb_     = 0;
    mutable std::vector<float>   embed_row_buf_;
    mutable int                  embed_row_tok_ = -1;
    // Set only while step_batch_embeds() runs: the row source for the batched body's embedding fill.
    const float* inject_embeds_ = nullptr;
    // Set only while step_batch_embeds(..., position_ids, ...) runs: 3 x rows, plane-major, or null.
    const int* pos3_ = nullptr;
    int pos3_rows_ = 0;
    // Set only while a step_batch_embeds() call carries deepstack sets: sets x rows x H.
    const float* deepstack_ = nullptr;
    int deepstack_rows_ = 0;
    int deepstack_lo_ = -1, deepstack_hi_ = -1;
    // Which plane a rotary frequency belongs to under interleaved MRoPE.  The rule lives in q4nx_semantics.h as
    // q4nx_mrope_axis() so the C unit test can pin it without building a forward pass.
    int mrope_plane(int freq) const {
        return sem_.mrope_sectioned ? q4nx_mrope_axis_sectioned(freq, sem_.n_mrope_section, sem_.mrope_section)
                                    : q4nx_mrope_axis(freq, sem_.n_mrope_section, sem_.mrope_section);
    }
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
    // The model owns these (A3); the references keep every use site unchanged, `assign` included.
    std::vector<std::vector<float>> & in_norm_ = model_.in_norm;
    std::vector<std::vector<float>> & post_norm_ = model_.post_norm;
    std::vector<std::vector<float>> & q_norm_ = model_.q_norm;   // [NL][HD], owned by the model (A3)
    std::vector<std::vector<float>> & k_norm_ = model_.k_norm;
    std::vector<std::vector<float>> q_bias_, k_bias_, v_bias_;  // [NL][NH*HD]/[NL][NKV*HD] attention biases
    std::vector<float> rope_scale_;  // LongRoPE short_factor (per-half-dim), empty when none
    Q4nxRopeLlama3 rope_llama3_;     // rope_type "llama3" (Llama-3.1/3.2/3.3); on == 0 for everything else
    Q4nxRopeDynamic rope_dynamic_;   // rope_type "dynamic" (NTK base rule, engine#1740 W2); on == 0 elsewhere
    double rope_linear_factor_ = 0.0; // rope_type "linear" factor (Gemma3's full layers); 0 elsewhere
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
    // Which MoE container convention this container uses, decided from its own tensors in
    // init_impl (moe_container.h): the engine converter's `mlp.experts.*` names with a plain
    // [N][K] per-expert tile grid, or the FLM-style `*_exps_proj` names whose writer transposes
    // each expert's tiles.  The wrong choice is silent, so it is read, not assumed.
    MoeFormat moe_format_ = MoeFormat::Flm;
    int leading_dense_ = 0;
    int expert_gating_sigmoid_ = 1;   // llama.cpp deepseek2: 2 = sigmoid, 1 = softmax
    float expert_w_scale_ = 1.0f;     // applied after the top-k weight normalisation
    std::vector<float> w_qa_, w_qb_, w_kva_, w_o_;      // per layer, [out][in]
    std::vector<float> mla_qan_, mla_kvan_;             // q_a / kv_a layernorms
    std::vector<float> w_kb_, w_vb_;                    // [NH][N][K], N padded for k_b
    int kb_rows_ = 0;                                   // k_b's padded row count (256)
    // The model owns the state the launch path writes (A3); references keep every use site unchanged.
    std::vector<std::vector<float>> & mla_kn_ = model_.mla_kn;
    std::vector<std::vector<float>> & mla_kr_ = model_.mla_kr;
    std::vector<std::vector<float>> & mla_v_ = model_.mla_v;
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
    std::vector<std::vector<float>> & kcache_ = model_.kcache;   // per-layer attention K/V, owned by the model (A3)
    std::vector<std::vector<float>> & vcache_ = model_.vcache;

    std::vector<float> B_;        // packed weight scratch  (max K*N floats)
    // Packed-int8 weight cache, one weight BO per (layer, wkey), packed on first
    // use.  Without it every token re-dequantizes the Q4NX (load_layer) AND
    // re-quantizes it (I8Ctx::packB) for every GEMM -- ~2 min/token on a 7B and
    // ~8 min/token on the 8B, which makes a served answer impractical.  Only
    // token-INVARIANT weights may be cached (dense projections, GDN, MLA, the
    // shared expert); routed-expert weights vary per token and pass wkey < 0.
    using PackedWeight = npu::target::xdna::PackedWeight;
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
    // gemv family: the head's tile stream is the whole image in one BO, copied once.  (The per-projection
    // tile image is not kept here: the target asks for it on a residency miss and uploads it straight into
    // the weight BO.)
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
    std::vector<int> & kv_owner_ = model_.kv_owner;   // per layer: layer whose K/V it reuses (itself)
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
    bool mrope_declared_ = false;         // rope_scaling type "mrope" (Qwen2-VL form), validated after semantics
    // LFM3's per-head output gate, [NH][H] per attention layer (empty for conv layers and other models):
    // ctx of head h *= sigmoid(head_gate_[l][h] . xn), xn the operator-normed layer input.
    std::vector<std::vector<float>> head_gate_;
    void apply_head_gate(int l, const float* xn, float* ctx, int hd) const;
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
    std::vector<std::vector<float>> & conv_state_ = model_.conv_state;   // [CD][CK-1]
    std::vector<std::vector<float>> & rec_state_ = model_.rec_state;     // [nvh][khd][vhd]
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

    // ── LFM2 gated short-convolution layers ──────────────────────────────
    // A layer that carries shortconv.* tensors has no attention: in_proj (H -> 3H, the QKVLIN design, weight
    // key 0) and out_proj (H -> H, the O design, key 1, the slot an attention layer's o_proj has) run on the
    // device, and everything between them is lfm2_shortconv.h on the host. The only state is the last L-1
    // values of B*X per channel, kept in conv_state_[l] so snapshot/restore/reset cover it with no new code.
    bool shortconv_ = false;                       // the model has at least one such layer
    int sc_taps_ = 0;                              // L, from shortconv.conv.weight's own shape
    std::vector<uint8_t> sc_layer_;                // [NL] 1 = conv layer
    std::vector<std::vector<float>> sc_w_;         // [NL][H * L] the depthwise taps, bf16-decoded once
    std::vector<std::vector<float>> sc_rows_;      // [NL][rows * H] B*X of a verify batch's rows (see begin_verify)
    bool is_conv_layer(int l) const { return shortconv_ && l >= 0 && l < (int)sc_layer_.size() && sc_layer_[(size_t)l]; }
    bool conv_layer(int l, const std::vector<float>& xn, std::vector<float>& out);
    bool conv_layer_batch(int l, const std::vector<float>& xn, int M, std::vector<float>& out);
    void conv_apply_rows(int rows);                // re-apply the first `rows` recorded rows after a restore
    std::string err_;
    std::string backend_ = "uninitialised";
    bool ready_ = false;
};

#endif  // NPU_INFER_Q4NX_FORWARD_H
