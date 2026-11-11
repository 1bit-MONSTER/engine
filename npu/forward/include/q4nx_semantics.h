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
// q4nx_semantics.h — architecture semantics for the model-generic forward pass.
//
// Every flag is derived from a field in the model's own config.json, so the
// forward pass stops assuming Qwen3.  q4nx_semantics_field() returns the config
// field each flag comes from, which is what makes the mapping auditable (and is
// what the docs semantics table is generated from).
//
// Verified against the config field named in the table below; see
// tests/test_q4nx_semantics.c.
#ifndef NPU_INFER_Q4NX_SEMANTICS_H
#define NPU_INFER_Q4NX_SEMANTICS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* Maximum layer count we can hold a per-layer attention type for. */
#define Q4NX_MAX_LAYERS 64

/* MLP activation. */
typedef enum {
    Q4NX_ACT_SILU = 0,   /* hidden_act: "silu"            */
    Q4NX_ACT_GELU,       /* hidden_act: "gelu*"/"geglu"   */
} Q4nxAct;

/* Attention layer kinds, from layer_types.  FULL and SLIDING run the attention block, LINEAR runs the
 * GDN recurrence (lin_kd_/conv_state_); OTHER is anything else, which nothing implements - the forward
 * refuses a container whose layer_types contains one, by name (unknown_layer_type). */
typedef enum {
    Q4NX_LAYER_FULL = 0,
    Q4NX_LAYER_SLIDING,
    Q4NX_LAYER_LINEAR,
    Q4NX_LAYER_OTHER
} Q4nxLayerType;

typedef struct {
    /* RMSNorm convention: 0 -> y = x/rms * w ; 1 -> y = x/rms * (1 + w)
     * Source: model_type/architectures (gemma* uses the offset form), or, for a qwen3_5 pack, the
     * layer-0 input_layernorm weight itself (see rmsnorm_offset_probe). */
    int rmsnorm_offset;
    /* qwen3_5 declares ONE model_type for TWO norm conventions: the official release is zero-centered
     * (Qwen3_5RMSNorm: y = x/rms * (1 + w), weight initialised to 0) while hybrid packs on this box keep
     * the plain form (weight initialised to 1). The name cannot decide it, so the forward probes the
     * loaded layer-0 input_layernorm weight once. 1 = probe, 0 = the convention is already known.
     * Source: model_type/architectures == qwen3_5*, resolved against the weights. */
    int rmsnorm_offset_probe;
    /* Per-head RMSNorm applied to q and k before RoPE.
     * Source: config.json's `use_qk_norm` when the config states it (engine#1889: Hunyuan), else the
     * model_type/architectures table (qwen3*, gemma*). The forward refuses a model whose flag is set and whose
     * layers carry no self_attn.q_norm.weight, since normalising with an absent gain zeroes q. */
    int qk_norm;
    /* The q/k RMSNorm runs AFTER RoPE instead of before it (engine#1889). Qwen3 and Gemma normalise, then rotate;
     * Hunyuan rotates, then normalises. With a per-dimension gain the two orders differ wherever a rotary pair's two
     * gains differ, so the order is part of the model. Source: model_type/architectures (hunyuan*) -- no config key
     * states it; the evidence is the pack's own graph, where every layer is
     * q_proj -> RotaryEmbedding -> q_norm (SimplifiedLayerNormalization) -> GroupQueryAttention. */
    int qk_norm_after_rope;
    /* RoPE base.  Source: rope_theta. */
    float rope_theta;
    /* Channels actually rotated; 0 means the full head_dim.  Partial rotary
     * applies rotary_dim channels and passes the rest through.
     * Source: partial_rotary_factor * head_dim (phi3). */
    int rotary_dim;
    /* MLP activation.  Source: hidden_act. */
    int mlp_act;
    /* Linear biases on q/k/v/o and on the MLP projections.
     * Source: attention_bias / mlp_bias (presence of corresponding tensors). */
    int attention_bias;
    int mlp_bias;
    /* Multiplier applied to token embeddings before layer 0.
     * Source: model_type (gemma* uses sqrt(hidden_size)); else 1.0. */
    float embed_scale;
    /* Final-logit soft cap: logits = c * tanh(logits / c). 0 disables.
     * Source: final_logit_softcapping (gemma2/3). */
    float logit_softcap;
    /* Sliding-window attention size; 0 disables.
     * Source: sliding_window (+ use_sliding_window). */
    int sliding_window;
    /* RMSNorm epsilon.  Source: rms_norm_eps. */
    float eps;
    /* Number of layers whose attention is NOT full attention.  Hybrid models
     * (qwen3_5_text) interleave linear_attention with full_attention; a plain
     * decoder forward pass cannot serve those layers.
     * Source: layer_types (count of entries != "full_attention"). */
    int hybrid_layers;
    /* Interleaved/multimodal RoPE variant that this forward does not implement.
     * Source: rope_parameters.mrope_interleaved (or rope_type != "default"). */
    int mrope;
    /* MVROPE sections: how many of the rotary frequencies belong to each of the (t, h, w) axes.  Read from
     * rope_scaling.mrope_section (Hugging Face) or rope_parameters.mrope_section (OGA/Qwen3.5); zeros and
     * n_mrope_section == 0 mean "declared mrope without sections", which a consumer must refuse rather than
     * guess a layout for. */
    int mrope_section[3];
    int n_mrope_section;
    /* DEEPSTACK: which decoder layers take one of the vision tower's extra feature sets.  Read from
     * vision_config.deepstack_visual_indexes (Qwen3-VL: [5, 11, 17]); the consumer adds set s to the hidden state at
     * layer deepstack_layers[s].  Empty means the pack has none. */
    int deepstack_layers[8];
    int n_deepstack;
    /* RoPE is configured PER LAYER TYPE (Gemma4 nests rope_parameters.
     * full_attention and .sliding_attention with different thetas and types).
     * Source: presence of rope_parameters.{full,sliding}_attention sub-blocks. */
    int per_layer_rope;
    /* Per-layer attention type, index = layer, from layer_types.  Entries are
     * Q4NX_LAYER_* ; when the config declares no layer_types every layer is
     * Q4NX_LAYER_FULL.  A model whose layers include Q4NX_LAYER_LINEAR needs
     * attention math (a recurrence/SSM) that exists nowhere in our artifact set,
     * so it is refused rather than approximated. */
    int layer_types[Q4NX_MAX_LAYERS];
    int n_layer_types;
    /* The first layer_types entry whose kind this forward does not implement ("" if none).
     * Source: layer_types.  The parser below maps anything it does not recognise to Q4NX_LAYER_OTHER and the
     * forward treats OTHER as full attention, so without this name a config that names, say, a conv attention
     * would be served as a plain decoder and be wrong everywhere - the same defect shape as the rope-scaling
     * refusal (engine#1087).  Kept as the spelling found, so the refusal can say which kind it is. */
    char unknown_layer_type[40];
    /* Per-layer-type RoPE, used by a layer whose type matches.  rotary_dim 0
     * means "full head_dim".  Source: rope_parameters.sliding_attention. */
    float rope_theta_sliding;
    int rotary_dim_sliding;
    /* A named rope_type other than "default" was declared (gemma4 uses
     * "proportional").  This is NOT mrope: the proportional partial-rotary form is
     * what this forward already implements (exponents normalised over the rotated
     * half), so it is recorded for the write-up but does not gate execution.
     * Source: rope_parameters.rope_type / rope_parameters.<type>.rope_type. */
    int rope_type_named;
    /* ── Gemma4-style per-layer embeddings (PLE) ──
     * `hidden_size_per_layer_input` > 0 switches on the PLE block, which needs a
     * per-layer model projection plus a separate per-layer token embedding table.
     * Sources: hidden_size_per_layer_input, vocab_size_per_layer_input. */
    int hidden_size_per_layer_input;
    int vocab_size_per_layer_input;
    /* Layers whose K/V is reused from an earlier layer of the same attention type
     * (`first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers`).
     * Source: num_kv_shared_layers. */
    int num_kv_shared_layers;
    /* Gemma4 adds pre/post feedforward norms and a post-PLE norm that the plain
     * layer does not have.  Source: presence of those tensors / model_type. */
    int extra_layer_norms;
    /* Gemma4 use_double_wide_mlp: KV-shared layers run a 2*IM-wide MLP.
     * Source: use_double_wide_mlp. */
    int double_wide_mlp;
    /* ── Gemma4 per-layer-type attention width and semantics ──
     * full_attention layers use `global_head_dim` where sliding layers use
     * `head_dim`, so the QKV/O geometries differ per layer type; with no
     * `num_global_key_value_heads` in the config the head COUNT is the same for
     * both.  Sources: global_head_dim, num_global_key_value_heads. */
    int global_head_dim;
    int num_global_kv_heads;
    /* Gemma4 sets attention scaling to 1.0 (q/k carry a per-head RMSNorm),
     * normalises v with a SCALE-FREE RMSNorm, and uses "proportional" RoPE on
     * its full layers.  Sources: model_type/architectures, rope_type. */
    int attn_scaling_one;
    int v_norm;
    int proportional_rope;
    /* Whether the LM head is the (possibly embedding-scaled) token embedding
     * (`tie_word_embeddings = true`, the default) or a SEPARATE
     * `lm_head.weight` tensor.  Nanbeige4.1-3B is the latter, and decoding its
     * logits from the embedding gives a completely different distribution
     * (corr 0.013 with the real head).  Source: tie_word_embeddings. */
    int tie_embeddings;
} Q4nxSemantics;

/* Identifier for each flag, for the auditable field mapping. */
typedef enum {
    Q4NX_SEM_RMSNORM_OFFSET = 0,
    Q4NX_SEM_QK_NORM,
    Q4NX_SEM_ROPE_THETA,
    Q4NX_SEM_ROTARY_DIM,
    Q4NX_SEM_MLP_ACT,
    Q4NX_SEM_ATTENTION_BIAS,
    Q4NX_SEM_MLP_BIAS,
    Q4NX_SEM_EMBED_SCALE,
    Q4NX_SEM_LOGIT_SOFTCAP,
    Q4NX_SEM_SLIDING_WINDOW,
    Q4NX_SEM_EPS,
    Q4NX_SEM_HYBRID_LAYERS,
    Q4NX_SEM_MROPE,
    Q4NX_SEM_PER_LAYER_ROPE,
    Q4NX_SEM_LAYER_TYPES,
    Q4NX_SEM_COUNT
} Q4nxSemFlag;

/* The config.json field a flag is derived from (never NULL). */
const char* q4nx_semantics_field(int flag);

/* Neutral defaults = the plain decoder-transformer convention. */
void q4nx_semantics_default(Q4nxSemantics* s);

/* Derive semantics from a config.json.  `model_dir` is used to detect
 * q_norm/k_norm tensors when the model file itself is not inspected; pass NULL
 * to rely on model_type and the scalar fields only.  Returns 0 on success,
 * -1 if the file cannot be read.  Unparseable or absent fields keep their
 * default value, so a missing field degrades to the neutral convention. */
int q4nx_semantics_from_config(const char* json_path, const char* model_dir,
                               Q4nxSemantics* s);

/* Which axis a rotary frequency belongs to under INTERLEAVED MRoPE.
 *
 * The three `mrope_section` values are not contiguous runs of frequencies.  The reference builds cos/sin for all three
 * planes and recomposes them: start from the t plane and, for each of h and w, overwrite every third frequency with that
 * plane's value over a stride-3 window of length `section[dim] * 3` (transformers
 * `Qwen3VLTextRotaryEmbedding.recomposition_frequencies`: `idx = slice(offset, mrope_section[dim] * 3, 3)`).
 *
 *   axis(i) = 1 if i % 3 == 1 && i < 3 * section[1]
 *           = 2 if i % 3 == 2 && i < 3 * section[2]
 *           = 0 otherwise
 *
 * For [24, 20, 20] that is 24 t / 20 h / 20 w frequencies, and 43 of the 64 sit where a contiguous split would not put
 * them -- which is why treating the sections as contiguous runs left every image prompt's answer bad while all
 * equal-plane (text-only) prompts stayed exact.  Frequency index is the index within head_dim/2.
 */
static inline int q4nx_mrope_axis(int freq, int n_section, const int* section) {
    if (n_section < 3 || !section) return 0;
    const int m = freq % 3;
    if (m == 1 && freq < 3 * section[1]) return 1;
    if (m == 2 && freq < 3 * section[2]) return 2;
    return 0;
}

/* Human-readable one-line dump of every flag with its source field. */
void q4nx_semantics_dump(const Q4nxSemantics* s);

/* Number of layers whose type needs attention math this forward does not have
 * (linear_attention / SSM).  0 means the model is executable. */
int q4nx_semantics_linear_layers(const Q4nxSemantics* s);

/* True when the model needs Gemma4-style per-layer embeddings (PLE). */
int q4nx_semantics_uses_ple(const Q4nxSemantics* s);

/* Zero-centered vs plain RMSNorm, decided from the loaded layer-0 input_layernorm weight itself: a
 * checkpoint trained with the (1 + w) form stores deltas around 0 (the initialiser is 0, so the weight
 * carries negatives and a mean well below 1), a plain one stores gains around 1. Returns 1 for the
 * (1 + w) form, 0 for the plain w form. Exported so the rule is testable with no model present. */
int q4nx_rmsnorm_offset_from_norm(const float* w, int n);

/* Per-head q/k RMSNorm, decided by the container when config.json's name does not decide it (engine#1788).
 * q4nx_semantics_from_config() turns qk_norm on for the families its table knows by model_type (qwen3*, gemma*). A
 * pack of the same architecture under another name -- FastFlowLM's MiniCPM-V 4.7 is `minicpmv4_7` around a qwen3_5
 * text model -- came out as "no q/k norm" while shipping self_attn.q_norm and k_norm in every full-attention layer;
 * those layers then ran without the norm, and the model answered short prompts and drifted on long ones.
 * `has_q[l]` / `has_k[l]` say whether layer l carries a non-empty q_norm / k_norm tensor. A layer carrying BOTH is the
 * container saying it has the norm: qk_norm becomes 1 and that layer's index is returned. Returns -1, changing
 * nothing, when qk_norm is already on, when no layer carries both, or on null/empty input. It never turns the norm
 * off: a family the table knows keeps its answer whatever the tensors are. Exported so the rule is testable. */
int q4nx_semantics_qk_norm_from_tensors(Q4nxSemantics* s, const unsigned char* has_q, const unsigned char* has_k, int layers);

/* attention.mode for one container, from the two places that can state it (engine#796).

 * A d3 pack declares the mask it was trained with in its own `decision_config.json`
 * (`attention_mode: noncausal_full_attention`), and `tools/npu_plan.py` copies that declaration into
 * `plan.json`, which is where this forward reads `attention.mode`.  A directory converted with
 * `pack` + `decision` and no plan therefore used to be answered causally -- measured on the real d3
 * pack, worst |dp| 0.3702 and a choice flip against 0.0246 with the plan -- with nothing in the log
 * saying so.  So the declaration decides when there is no plan, and a plan that disagrees with it is
 * refused rather than obeyed (a plan written before the pack declared its mask, or hand-edited).

 *   declared        the pack's attention_mode, or NULL when the directory has no decision_config.json
 *   plan_loaded     1 when plan.json was read
 *   plan_noncausal  1 when that plan says noncausal_full_attention

 * Returns 1 (noncausal_full_attention), 0 (causal), or -1 when the two disagree, with the reason in
 * `why`.  Exported so the rule is testable with no model and no device. */
int q4nx_attention_mode(const char* declared, int plan_loaded, int plan_noncausal, char* why, size_t why_len);

/* Why a wide prefill was blocked below the prefill design's width, or "" when it was not (engine#1582).

 * npu_gemm_rows() cannot launch more rows than the loaded activation BO holds, and that width is the
 * *loaded context's* MD -- 128 for a prefill set, 16 for a decode-only ("_m16") set, 1 when no context
 * is loaded at all. A 219-row prefill at 16 rows/launch is 14 launches per projection instead of 2, and
 * the cost lands in per-launch host packing, which is what a phase profile then reports as "host". A
 * reader looking at such a log has no way to tell "the device ran it in blocks" from "the device did not
 * run it", so the block width is said out loud, once, with the launch count it implies.

 * Returns "" when the batch fits the prefill width (nothing to say), else a one-line note. Pure, so
 * tests/npu_rmsnorm_offset_test.c pins it with no model and no device. */
const char* q4nx_block_note(int rows, int block, int prefill_rows);

/* Description of the layer-type mix, for error messages. */
void q4nx_semantics_layer_mix(const Q4nxSemantics* s, char* out, size_t out_sz);

/* Read an integer field from a config.json (top-level keys only); returns def
 * when the file or field is absent.  Lets the forward take its architecture dims
 * from the model's own config instead of a built-in model. */
int q4nx_config_int(const char* json_path, const char* key, int def);

#ifdef __cplusplus
}
#endif
#endif /* NPU_INFER_Q4NX_SEMANTICS_H */
