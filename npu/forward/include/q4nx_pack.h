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
// property, not an architecture property: the design build fuses for
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

#define Q4NX_MAX_DESIGNS 14
#define Q4NX_ELF_NAME_MAX 64

/* Activation tile (M) each design FAMILY is built for.  Two families share one
 * role geometry, so the artifact name carries M only when it is not the
 * prefill one:
 *
 *   full_i8_<ROLE>_K<K>_N<N>[_m16].elf
 *
 * PREFILL (128) is the family the engine shipped before the decode fix: one
 * launch covers 128 activation rows, which suits a prefill chunk.  Decode is
 * M=1, and streaming that through a 128-row kernel costs 128 A rows and 128 C
 * rows for one useful row -- 8 of 32 compute tiles carry data and 1 of 128
 * baked MAC rows is useful (0.78%).  DECODE (16) is the same role geometry at
 * the smallest vectorized mmul M: one core row, all 8 columns, the column
 * groups still looped inside the ELF (so one launch per role stays the
 * contract).  Measured cost of an M=16 launch is ~0.34 ms against 3.4-4.9 ms
 * for the M=128 path. */
#define Q4NX_DESIGN_M_PREFILL 128
#define Q4NX_DESIGN_M_DECODE 16

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
    /* MoE: one gate (and up) GEMM over ALL top-k selected experts at once.  They
     * share the input xn, so their weights concatenate along N (expert j occupies
     * columns [j*IM, (j+1)*IM)); the caller passes the real N = top_k*IM, which is
     * <= the design's N, and reads back the first top_k*IM outputs. */
    Q4NX_DESIGN_GB,      /* batched expert gate: K=H, N>=top_k*IM                 */
    Q4NX_DESIGN_UB,      /* batched expert up:   K=H, N>=top_k*IM                 */
    /* Only a pb16i container derives these: its kernels have no wide enough launch
     * for a fused full-attention QKV, so the role runs as the two q halves and the k/v
     * pair.  K=H, N=NH*HD for the halves (q and the output gate) and N=NKV*HD for k/v.
     * Appended last so no existing enum value changes meaning. */
    Q4NX_DESIGN_QKVFA_Q,
    Q4NX_DESIGN_QKVFA_KV,
    Q4NX_DESIGN_COUNT
} Q4nxDesign;

typedef struct {
    int M, K, N;                       /* GEMM dims */
    Q4nxDesign role;                   /* which projection(s) fill N */
    char elf_name[Q4NX_ELF_NAME_MAX];  /* geometry-derived artifact name */
    int q_rows, k_rows, v_rows;        /* row counts along N for QKV */
    int gate_rows, up_rows;            /* row counts along N for GU/G/U */
    /* Real N before the kernel's n-tile pad; 0 means "same as N".  assemble_B scatters only
     * real_n columns and zero-fills the rest, so the weight is never read out of bounds.  A
     * non-zero real_n also means the design's N was padded up to q4nx_pad_n(): Qwen2.5-0.5B's O
     * projection is 896 rows of output in a 1024-column kernel (engine#845). */
    int real_n;
    /* Real K before the container's chunk grid pad; equal to K when the model is already on the
     * grid.  The pad rows are ZERO in the container (npu/pack_import.cpp writes scale 0, zero 0,
     * codes 0), so a padded K adds exactly 0 to every dot product and the K=K design is an exact
     * reading of the container rather than an approximation of it.  Callers that dequantise a
     * container tensor must use K as the width, not the model's hidden/intermediate size. */
    int real_k;
} Q4nxDesignGeom;

/* The two grids a design's K and N are expressed on, and the only padding in this file.  Both
 * come from the kernels and the container, not from a model:
 *
 *   q4nx_pad_k  the Q4NX q4_1/Q8 chunk is 32 rows x 256 columns (npu/q4nx.h `kCols`), so a
 *               container stores every tensor with K padded up to 256 -- Qwen2.5-0.5B's hidden
 *               896 is stored as 1024 columns of zeros-and-weights.
 *   q4nx_pad_n  a full ELF's N is built on the gemv family's column pass, 32 rows x 4 x cols(4)
 *               = 512 columns (addons/npu-dx/forge pad_rows in the private kernel tree), and it
 *               loops whole 512-column groups inside one ELF, so one launch still covers N.
 *
 * Both are identity for every model validated before this existed (hidden and N were already on
 * both grids: 5120/6144/2048/14336/17408 ...), which is exactly why the mismatch went unseen
 * until a pack with hidden 896 (engine#845). */
int q4nx_pad_k(int k);
int q4nx_pad_n(int n);

/* Pad an activation batch from `real_k` columns per row up to the design's `K`, zero-filling the
 * tail into `out` (M x K floats).  A kernel launched at a padded K reads A at stride K, while the
 * forward's activation vectors hold only the model's real width (896 of 1024 columns for
 * Qwen2.5-0.5B), so without this every A read runs past its buffer -- UB, and the device path also
 * bakes it in by taking the int8 activation scale from the amax over the overrun.  The container's
 * pad columns are zero, so the padded read is the same dot product, not an approximation
 * (engine#845 review).  `real_k <= 0` or `real_k >= K` copies nothing extra, which keeps the MLA
 * designs (they do not set real_k) on their existing path. */
void q4nx_pad_a_f32(const float* A, int M, int real_k, int K, float* out);

/* The model's own fallback GEMM: C[M,N] = A[M,a_k] x B[K,N], with the activation rows READ AT STRIDE
 * `a_k` and PADDED to the design's K before the multiply.  `out` gets M*N floats (written, not
 * accumulated into).
 *
 * It exists because the padding above and this loop are one decision, not two: a caller that pads for
 * the kernel but not for the host fallback runs the `for (k < K) A[k]` read past the end of the
 * model's real-width activation buffer -- and in the rows form at stride K from a buffer laid out at
 * stride a_k, so every row after the first is misaligned as well as overrun (engine#845's contract;
 * found in review of engine#808, where the split moved the pad into the XDNA target and left the host
 * branch behind on the raw A).  Keeping the pad and the loop in one function is what makes that class
 * of mistake unavailable to a future caller. */
void q4nx_host_gemm_f32_padded(const float* A, int a_k, int M, int K, int N, const float* Bmat,
                              float* out);

/* The integer arm of the same host path (engine#1808): int8 activation x int4 weight codes -> exact int32,
 * which is what the kernel computes. Written for the bit-exactness gate, because the f32 host path cannot
 * be bit-exact with an int8 device arm by construction; scales are applied by the caller on both sides so
 * a scale mistake cannot hide inside the integer comparison. `codes` is [K][N], one int8 per element
 * holding an int4 value in [-8, 7], exactly as the importer's packB lays it down. */
void q4nx_host_gemm_i8(const int8_t* a8, int K, const int8_t* codes, int N, int32_t* out32);

/* engine#1839: the int4 code range and its scale, together, because they are only correct as a pair.
 *
 * A symmetric int4 code spans [-8, 7], so the scale that maps a group's `amax` onto that range is `amax / 7`, and the
 * code is `round(v * 7 / amax)` clamped to [-8, 7]. The two places that quantise float weights into the container's
 * int4 codes used to pair `amax / 127` (an INT8 scale) with the int4 clamp, so every code but the very largest
 * saturated at +-7 and the dequantised weight was ~18x too small -- in the host arm the gate compares the device
 * against, which is the worst place for it. 7 rather than 8 so that no code saturates: round(v * 7 / amax) lands in
 * [-7, 7] for |v| <= amax, and the -8 slot stays a guard rather than an asymmetry. */
float q4nx_quant_int4_scale(float amax);
int8_t q4nx_quant_int4_code(float v, float amax);

/* The whole GEMM in int8 x int4 for the gate's `i8` host arm, quantising both sides here with the engine's
 * own rule. It cannot see a wrong weight LAYOUT (the f32 arm covers that); what it can see is a wrong int4
 * quantisation, a wrong 128-K group boundary, an int8 intermediate or a saturated accumulator -- i.e. the
 * kernel's arithmetic, exactly. int32 out, M*N, no scaling applied. */
void q4nx_host_gemm_i8_from_f32(const float* A, int a_k, int M, int K, int N, const float* Bmat,
                                int32_t* out32);

/* The reusable form: `codes` and `d` are quantised once per tensor by the caller (a per-call quantisation of
 * the weights costs ~7.5 G scalar operations per 7B token, measured — it never finished a token in a
 * 9-minute ticket), and this applies the kernel's epilogue: ascale * sum_g d[g][n] * sum_{k in g} code*a8,
 * with the inner sum exact int32 and no float inside it. */
void q4nx_host_gemm_i8_grouped(const int8_t* a8, int K, const int8_t* codes, const float* d,
                               int groups, int N, float ascale, float* out);

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
    /* LFM2's gated short-convolution layers: in_proj is one fused (K=H, N=3*H)
     * projection [B | C | X] and out_proj is (K=H, N=H), which is the O design
     * whenever NH*HD == H.  shortconv_n is in_proj's N; 0 means the model has no
     * such layer.  It derives one QKVLIN design and nothing else (no Z), and is
     * exclusive with lin_kd: a model has GDN layers or conv layers, not both.
     * Appended last so positional initialisers keep their meaning. */
    int shortconv_n;
} Q4nxModelDims;

/* No model-specific geometry constant lives here by design: every caller
 * derives Q4nxModelDims from its own model configuration. */

/* Number of designs a model needs: 4 when fused_gu, 5 when split. */
int q4nx_num_designs(const Q4nxModelDims* d);

/* Fill designs[0..q4nx_num_designs(d)) with the derived geometry.  Artifact
 * names are geometry-derived (full_i8_<ROLE>_K<K>_N<N>.elf) so nothing is keyed
 * on a model name or tag.  Returns the design count, or -1 on bad dims. */
int q4nx_derive_designs(const Q4nxModelDims* d, Q4nxDesignGeom* designs);

/* M=16 (decode) twins of an already-derived design set.  out[i] is the twin of
 * designs[i], or zeroed (out[i].M == 0) when this geometry cannot be covered by
 * the m16 kernel (m=16, k=64, n=128 over 8 columns, so K%64, N%128 and
 * (N/128)%8 must all be 0).  The artifact name is the input's with "_m16"
 * inserted before ".elf"; the bool result is the same test WITHOUT asking the
 * file system, so the caller decides whether a missing ELF is fatal (it is
 * not: the role then simply stays on its M=128 kernel).  Returns the number of
 * usable twins. */
int q4nx_derive_decode_designs(const Q4nxDesignGeom* designs, int count,
                               Q4nxDesignGeom* out);

/* The UNIQUE name of a design role -- not the display tag the forward logs.
 *
 * They differ where a model carries two roles that read the same in a log: a qwen3_5 pack derives an
 * attention QKV and a GDN QKVLIN, which the forward both prints as "QKV".  A plan's `designs[].role`
 * is a key (engine#803 A2), so it uses these; the log keeps the short tag. */
const char* q4nx_design_role_name(Q4nxDesign role);

/* The LOG tag for a design role -- deliberately NOT the unique name above, and deliberately the
 * historical mapping (QKVLIN prints as "QKV", every role the old table did not name prints as "D").
 *
 * It exists so a diagnostic a reader has already learned does not change when code moves: the setup
 * side moved into the target (engine#803 A2b) and the log lines had to stay byte-identical, which the
 * #1380 review caught them not doing.  The mapping is arguably misleading for the MLA and MoE roles --
 * that is a change worth making on purpose, in a PR that says so, not as a side effect of a move. */
const char* q4nx_design_log_tag(Q4nxDesign role);

// Which instance of a SEALED pb16i ELF runs a role, or NULL when the sealed ELF does not cover it.  The names
// are the contract with the kernel lane (TROSS-GU-STATE.md 08:30).  Here rather than in the forward because the
// target opens the decode family (engine#803 A2b) and a table the two sides must agree on cannot live on one
// side of the split: a duplicate would be two contracts that drift silently.
const char* q4nx_pb16i_instance(Q4nxDesign role);

// The same for a SEALED int8 b8 ELF (engine#2208): one hw_context for the decode roles of a b8 container whose
// full-attention QKV is one fused role (QKV, with the output gate) and whose gate/up is one fused role (GU).
// The split G and U roles have no instance: the sealed GU instance is the fused width, so they keep their own
// kernels. NULL means the sealed ELF does not cover the role.
const char* q4nx_i8_sealed_instance(Q4nxDesign role);

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
 * Returns the mean column scale (I8Ctx's `sout`), and **0.0 with nothing written** when the per-column scale array cannot
 * be allocated (the row-major traversal added in engine #1416 needs N floats; the old j-outer loop did not allocate). The
 * only caller today is a test, but a 0 return must not be read as "quantised to nothing". */
/* The PRODUCTION weight packer: I8Ctx::packB_blocked, extracted so it can be tested without the device runtime
 * (engine #1416). Writes Bm at the blocked (or row-major, `blocked == 0`) index for this geometry, per-column scales into
 * col_scales[N], and the sum of the scales into *ssum_out. `KD`/`ND` are the container's padded geometry the layout is
 * defined on. Behaviour-preserving extraction: the body is the class's own, with `bpos()` passed in. */
int q4nx_pack_B_int8_blocked(int8_t* Bm, const float* w, int K, int N, int KD, int ND, int blocked,
                             float* col_scales, double* ssum_out);

/* Chunks -> int8 without materialising the float matrix (engine #1416): the q4_1 tile kind only.
 * Dequantises one 32x256 tile into scratch, folds it into the per-column amax, then writes the codes on a second pass.
 * Byte-identical to `q4nx_dequant_tensor() -> q4nx_pack_B_int8()` (same tile decoder, same arithmetic, same amax floor),
 * which tests/npu_design_grid_test.c asserts by running both. Returns 0, -1 on a bad geometry, -2 when the container is
 * not the q4_1 tile kind (caller keeps the two-step path). */
/* Fold a q4_1 tile grid straight into the int8 image q4nx_pack_B_int8 produces, with no K x N float buffer:
 * dequantise one 32x256 tile into 32 KB of scratch, fold it into the per-column amax, then a second pass over the
 * same tiles emits the codes. Returns 0 on success, -2 when `nbytes` is not a q4_1 tile grid (the check a caller
 * uses to keep its existing path), -1 on a bad argument or allocation failure.
 *
 * NOT CALLED YET (engine#1416 / #1713): nothing in the forward path invokes it. `load_layer` still runs
 * deq() into the K x N float buffer and `I8Ctx::packB_into` -> packB_blocked(). The next slice is to call this
 * from load_layer's q4_1 branch (fused when rc == 0, deq + pack when rc == -2), and gate that on a byte-identity
 * comparison against packB_blocked's image on the production quantiser -- until then the traffic and timing
 * numbers measure this primitive, not the path it will replace. */
int q4nx_pack_q4_1_tiles_to_int8(const uint8_t* tiles, size_t nbytes, int K, int N, int8_t* out, int ld,
                                 float* col_scales);

float q4nx_pack_B_int8(const float* B, int K, int N, int8_t* out, int ld,
                       float* col_scales);

/* The three buffer widths a full ELF's runtime sequence declares (its .dynsym STT_OBJECT symbols
 * named "0".."2"), read from the artifact instead of assumed.  A launcher must pass buffers at
 * least that wide: the kernel's shim DMA touches exactly these widths, and a narrower buffer makes
 * the DMA walk past the mapping (IO_PAGE_FAULTs, garbage results, no error).  Measured 2026-11-02:
 * a pack whose ELF family is (weights, activations, output) = (655,360, 2,048, 4,096) B driven by a
 * launcher that passed (16,384, 1,048,576, 65,536) B.  Returns the number of widths found (0..3) or
 * -1 when the file is not a 32-bit LE ELF / has no .dynsym; undeclared entries stay 0. */
int q4nx_elf_arg_sizes(const void* elf, size_t n, uint64_t out[3]);
/* Does a declaration describe the gemv (q4_1 GEMV) family's launch?  Another family, or a
 * declaration that does not fit K/N, is 0.  The per-core split of the .dynsym symbols is handled
 * here: arg0/arg2 are 1/cores of the stream/output, arg1 is the whole activation row. */
int q4nx_gemv_family(const uint64_t declared[3], int K, int N);

#ifdef __cplusplus

}
#endif
#endif /* NPU_INFER_Q4NX_PACK_H */
