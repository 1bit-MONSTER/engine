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

/* Attention layer kinds, from layer_types (and the only kinds this forward can
 * execute; Q4NX_LAYER_LINEAR is detected but not implemented). */
typedef enum {
    Q4NX_LAYER_FULL = 0,
    Q4NX_LAYER_SLIDING,
    Q4NX_LAYER_LINEAR,
    Q4NX_LAYER_OTHER
} Q4nxLayerType;

typedef struct {
    /* RMSNorm convention: 0 -> y = x/rms * w ; 1 -> y = x/rms * (1 + w)
     * Source: model_type/architectures (gemma* uses the offset form). */
    int rmsnorm_offset;
    /* Per-head RMSNorm applied to q and k before RoPE.
     * Source: presence of self_attn.q_norm.weight / k_norm.weight. */
    int qk_norm;
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

/* Human-readable one-line dump of every flag with its source field. */
void q4nx_semantics_dump(const Q4nxSemantics* s);

/* Number of layers whose type needs attention math this forward does not have
 * (linear_attention / SSM).  0 means the model is executable. */
int q4nx_semantics_linear_layers(const Q4nxSemantics* s);

/* True when the model needs Gemma4-style per-layer embeddings (PLE). */
int q4nx_semantics_uses_ple(const Q4nxSemantics* s);

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
