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
// q4nx_pack.h — per-projection GEMM design table + weight packing for Q4NX models.
//
// The design set is DERIVED from the model's architecture dims, not hardcoded.
// A kernel computes C[M,N] = A[M,K] x B[K,N], and the checkpoint stores each
// projection as [out_features, in_features], so B is the transposed projection
// (B[i][n] = W[n][i]), with projections concatenated along N where one kernel
// fuses them:
//
//   QKV : K = H,            N = NH*HD + 2*NKV*HD        (q | k | v)
//   O   : K = NH*HD,        N = H
//   GU  : K = H,            N = 2*IM                    (gate | up, fused)
//   G   : K = H,            N = IM                      (split models)
//   U   : K = H,            N = IM
//   D   : K = IM,           N = H
//
// Whether gate+up are fused into one kernel (GU) or split (G + U) is an artifact
// property, not an architecture property: engine/npu/build_xclbins.sh fuses for
// Qwen3-0.6B and Gemma4, and splits for Llama / Qwen3-8B / Qwen3-VL-4B. It is
// therefore an explicit input (`fused_gu`), and the artifact resolver can also
// choose it by which full ELF exists on disk.
//
// Quantization matches I8Ctx::packB_into() exactly (n1_engine_i8ctx_inc.h):
// per-OUTPUT-COLUMN scale ts_j = max_i|w_ij| / 127, codes roundf(w_ij / ts_j)
// clamped to +/-127.  Per-column (rather than per-tensor) scales matter: Qwen3
// v_proj has a much smaller rms than q/k, and a single scale quantized those
// columns onto ~10 int8 levels.
#ifndef NPU_INFER_Q4NX_PACK_H
#define NPU_INFER_Q4NX_PACK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q4NX_MAX_DESIGNS 12
#define Q4NX_ELF_NAME_MAX 64

typedef enum {
    Q4NX_DESIGN_QKV = 0,
    Q4NX_DESIGN_O,
    Q4NX_DESIGN_GU,   /* fused gate|up  */
    Q4NX_DESIGN_G,    /* gate only      */
    Q4NX_DESIGN_U,    /* up only        */
    Q4NX_DESIGN_D,
    Q4NX_DESIGN_PLEGATE, /* Gemma4 per-layer input gate:  K=H,      N=ple_dim */
    Q4NX_DESIGN_PLEPROJ, /* Gemma4 per-layer projection:  K=ple_dim, N=H      */
    Q4NX_DESIGN_GU2,     /* Gemma4 double-wide gate|up:   K=H,   N=4*IM      */
    Q4NX_DESIGN_D2,      /* Gemma4 double-wide down:      K=2*IM,   N=H      */
    Q4NX_DESIGN_QKV2,    /* 2nd attention type's QKV (Gemma4 full layers)    */
    Q4NX_DESIGN_O2,      /* 2nd attention type's O   (Gemma4 full layers)    */
    Q4NX_DESIGN_QKVLIN,  /* GDN linear attention's fused q|k|v: K=H, N=2*KD+VD */
    Q4NX_DESIGN_Z,       /* GDN in_proj_z (the gated-norm gate): K=H, N=VD   */
    /* MLA (deepseek2, e.g. GLM-4.7-Flash).  These replace QKV; O/G/U/D are still
     * used, with HD set to the per-head VALUE width so the O design's K = NH*HD =
     * NH*v_head, and IM set to the per-expert intermediate so G/U/D are the
     * expert MLP.  The dense leading layer gets its own DG/DU/DD.  Set
     * dims->mla to derive this set. */
    Q4NX_DESIGN_QA,      /* MLA q_a_proj:            K=H,       N=q_lora           */
    Q4NX_DESIGN_QB,      /* MLA q_b_proj:            K=q_lora,  N=NH*(qk_nope+rope) */
    Q4NX_DESIGN_KVA,     /* MLA kv_a_proj_with_mqa:  K=H,       N=kv_lora+rope     */
    Q4NX_DESIGN_DG,      /* dense leading layer gate: K=H,       N=dense_im        */
    Q4NX_DESIGN_DU,      /* dense leading layer up:   K=H,       N=dense_im        */
    Q4NX_DESIGN_DD,      /* dense leading layer down: K=dense_im, N=H             */
    Q4NX_DESIGN_COUNT
} Q4nxDesign;

typedef struct {
    int M, K, N;                       /* GEMM dims */
    Q4nxDesign role;                   /* which projection(s) fill N */
    char elf_name[Q4NX_ELF_NAME_MAX];  /* geometry-derived artifact name */
    int q_rows, k_rows, v_rows;        /* row counts along N for QKV */
    int gate_rows, up_rows;            /* row counts along N for GU/G/U */
    /* Real N before the kernel's n-tile pad; 0 means "same as N".  Only KVA
     * (kv_lora+rope_dim = 576, padded to 640) needs it: assemble_B scatters
     * real_n columns and zero-fills the rest, so the weight is never read out
     * of bounds. */
    int real_n;
} Q4nxDesignGeom;

/* Architecture dims a design set is derived from. */
typedef struct {
    int H;        /* hidden_size */
    int NH;       /* num_attention_heads */
    int NKV;      /* num_key_value_heads */
    int HD;       /* head_dim */
    int IM;       /* intermediate_size */
    int fused_gu; /* 1 = one GU kernel, 0 = separate G and U kernels */
    int M;        /* activation tile (128) */
    int ple_dim;  /* Gemma4 hidden_size_per_layer_input; 0 = no per-layer block.
                   * Appended last so positional initialisers keep their meaning. */
    int double_wide_mlp; /* Gemma4 use_double_wide_mlp: KV-shared layers run a
                          * 2*IM-wide MLP, so a second GU/D pair is needed. */
    /* A model whose attention width varies BY LAYER TYPE needs a second QKV and
     * O geometry.  Gemma4's full_attention layers use global_head_dim (512)
     * where its sliding layers use head_dim (256): HD2 == 0 (or == HD) means
     * every layer shares one width. */
    int NKV2;     /* 2nd type's num_key_value_heads (0 = same as NKV) */
    int HD2;      /* 2nd type's head_dim (0 = same as HD) */
    /* GDN (qwen3_5 linear_attention) projections.  The fused q|k|v is
     * (K=H, N=2*KD+VD) with KD = linear_num_key_heads*linear_key_head_dim and
     * VD = linear_num_value_heads*linear_value_head_dim; in_proj_z (stored as
     * self_attn.gate_proj) is (K=H, N=VD).  lin_kd == 0 means the model has no
     * linear layers.  alpha/beta are (K=H, N=NVH) -- N < 128 -- so they are
     * computed on the host, not as a design. */
    int lin_kd;
    int lin_vd;
    /* qwen3_5 sets attn_output_gate: q_proj emits BOTH q and an output gate, so
     * the QKV design's q_rows (and therefore N) are 2*NH*HD while the attention
     * width and the O design's K stay NH*HD.  The .q4nx stores it as
     * [q(all heads) | gate(all heads)] (converter reorder).  Source:
     * attn_output_gate in the model's config.json. */
    int attn_output_gate;
    /* MLA + MoE (deepseek2 / GLM-4.7-Flash).  mla == 0 for every other
     * architecture, leaving the standard derivation untouched.  When set, HD must
     * be the per-head VALUE width (so the O design's K = NH*HD = NH*v_head) and IM
     * the per-expert intermediate (so G/U/D are the expert MLP). */
    int mla;
    int q_lora;      /* q_a_proj output width                       */
    int kv_lora;     /* kv_a_proj latent width (excluding rope)      */
    int rope_dim;    /* rope width inside kv_a and each q head       */
    int qk_nope;     /* per-head qk width excluding rope             */
    int dense_im;    /* leading dense layer's intermediate width     */
    int mla_kva_pad; /* KVA design N pad: ceil((kv_lora+rope_dim)/128)*128 */
} Q4nxModelDims;

/* No model-specific geometry constant lives here by design: every caller
 * derives Q4nxModelDims from its own model configuration. */

/* Number of designs a model needs: 4 when fused_gu, 5 when split. */
int q4nx_num_designs(const Q4nxModelDims* d);

/* Fill designs[0..q4nx_num_designs(d)) with the derived geometry.  Artifact
 * names are geometry-derived (full_i8_<ROLE>_K<K>_N<N>.elf) so nothing is keyed
 * on a model name or tag.  Returns the design count, or -1 on bad dims. */
int q4nx_derive_designs(const Q4nxModelDims* d, Q4nxDesignGeom* designs);

/* Look up the entry for a role in a derived set; NULL if absent. */
const Q4nxDesignGeom* q4nx_find_design(const Q4nxDesignGeom* designs, int count,
                                       Q4nxDesign role);

/* Floating-point weight matrices per projection ([out][in], row-major).
 * Row counts are implied by the dims. */
typedef struct {
    const float* q, *k, *v, *o, *gate, *up, *down;
    const float* ple;   /* single matrix for PLEGATE / PLEPROJ */
} Q4nxProjections;

/* Assemble B[K][N] (row-major, ld = N) for a derived design.  B must hold
 * K*N floats.  For a QKV design the q/k/v row counts come from the design
 * entry, so this works for any head configuration.  Returns 0, or -1. */
int q4nx_assemble_B(const Q4nxDesignGeom* g, const Q4nxProjections* p, float* B);

/* Per-column int8 quantization of B[K][N] into out with leading dimension ld,
 * plus per-column scales.  Identical numerics to I8Ctx::packB_into().
 * Returns the mean column scale (I8Ctx's `sout`). */
float q4nx_pack_B_int8(const float* B, int K, int N, int8_t* out, int ld,
                       float* col_scales);

#ifdef __cplusplus
}
#endif
#endif /* NPU_INFER_Q4NX_PACK_H */
