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
// q4nx_pack.c — see q4nx_pack.h.  The design set is derived from architecture
// dims; nothing here is keyed on a model name or tag, and no geometry literal
// for any specific model exists in this file.
#include "q4nx_pack.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* A model whose attention width varies by layer type (Gemma4 full vs sliding)
 * needs a second QKV and O geometry derived from its other head_dim. */
static int has_second_attn(const Q4nxModelDims* d) {
    return d->HD2 > 0 && d->HD2 != d->HD;
}

int q4nx_num_designs(const Q4nxModelDims* d) {
    if (!d || d->H <= 0 || d->NH <= 0 || d->NKV <= 0 || d->HD <= 0 || d->IM <= 0)
        return -1;
    /* MLA + MoE: QA, QB, KVA, O, G, U, D, DG, DU, DD.  The standard QKV is not
     * used (MLA has no plain q/k/v projection), the expert MLP reuses G/U/D with
     * IM = the per-expert intermediate, and the leading dense layer gets DG/DU/DD. */
    if (d->mla)
        return 10;
    return (d->fused_gu ? 4 : 5) + (d->ple_dim > 0 ? 2 : 0) +
           (d->double_wide_mlp ? 2 : 0) + (has_second_attn(d) ? 2 : 0) +
           (d->lin_kd > 0 ? 2 : 0);
}

/* MLA + MoE design set (see q4nx_num_designs).  dims->HD must be the per-head
 * VALUE width so O comes out K=NH*HD=NH*v_head, and dims->IM the per-expert
 * intermediate so G/U/D are the expert MLP. */
static void set_elf_name(Q4nxDesignGeom* g, const char* role);

static int derive_mla(const Q4nxModelDims* d, Q4nxDesignGeom* out) {
    int i = 0;
    const int q_head = d->qk_nope + d->rope_dim;      /* q's per-head width */
    const int kv_real = d->kv_lora + d->rope_dim;
    const int kv_pad = d->mla_kva_pad > 0
                           ? d->mla_kva_pad
                           : ((kv_real + 127) / 128) * 128;

    memset(&out[i], 0, sizeof(out[i]));
    out[i].M = d->M; out[i].K = d->H; out[i].N = d->q_lora;
    out[i].role = Q4NX_DESIGN_QA; set_elf_name(&out[i], "QA");

    i++; memset(&out[i], 0, sizeof(out[i]));
    out[i].M = d->M; out[i].K = d->q_lora; out[i].N = d->NH * q_head;
    out[i].role = Q4NX_DESIGN_QB; set_elf_name(&out[i], "QB");

    i++; memset(&out[i], 0, sizeof(out[i]));
    out[i].M = d->M; out[i].K = d->H; out[i].N = kv_pad; out[i].real_n = kv_real;
    out[i].role = Q4NX_DESIGN_KVA; set_elf_name(&out[i], "KVA");

    i++; memset(&out[i], 0, sizeof(out[i]));
    out[i].M = d->M; out[i].K = d->NH * d->HD; out[i].N = d->H;
    out[i].role = Q4NX_DESIGN_O; set_elf_name(&out[i], "O");

    i++; memset(&out[i], 0, sizeof(out[i]));
    out[i].M = d->M; out[i].K = d->H; out[i].N = d->IM;
    out[i].role = Q4NX_DESIGN_G; set_elf_name(&out[i], "G");

    i++; memset(&out[i], 0, sizeof(out[i]));
    out[i].M = d->M; out[i].K = d->H; out[i].N = d->IM;
    out[i].role = Q4NX_DESIGN_U; set_elf_name(&out[i], "U");

    i++; memset(&out[i], 0, sizeof(out[i]));
    out[i].M = d->M; out[i].K = d->IM; out[i].N = d->H;
    out[i].role = Q4NX_DESIGN_D; set_elf_name(&out[i], "D");

    i++; memset(&out[i], 0, sizeof(out[i]));
    out[i].M = d->M; out[i].K = d->H; out[i].N = d->dense_im;
    out[i].role = Q4NX_DESIGN_DG; set_elf_name(&out[i], "DG");

    i++; memset(&out[i], 0, sizeof(out[i]));
    out[i].M = d->M; out[i].K = d->H; out[i].N = d->dense_im;
    out[i].role = Q4NX_DESIGN_DU; set_elf_name(&out[i], "DU");

    i++; memset(&out[i], 0, sizeof(out[i]));
    out[i].M = d->M; out[i].K = d->dense_im; out[i].N = d->H;
    out[i].role = Q4NX_DESIGN_DD; set_elf_name(&out[i], "DD");

    return i + 1;
}

static void set_elf_name(Q4nxDesignGeom* g, const char* role) {
    /* The prefill family is the one that shipped un-suffixed.  Decode twins are
     * named in q4nx_derive_decode_designs() (a copy of this name with "_m16"
     * inserted before ".elf"); keeping the "_m%d" branch here as well would be
     * the same string built two ways, and every caller of this function is a
     * prefill design. */
    snprintf(g->elf_name, Q4NX_ELF_NAME_MAX, "full_i8_%s_K%d_N%d.elf", role,
             g->K, g->N);
}

/* The container's K grid (npu/q4nx.h `kCols`: the q4_1/Q8 chunk is 32 x 256) and the kernel's
 * column pass (32 x 4 x cols, cols=4 -> 512).  See q4nx_pack.h for why both exist. */
#define Q4NX_K_GRID 256
#define Q4NX_N_GRID 512

int q4nx_pad_k(int k) {
    return k <= 0 ? k : ((k + Q4NX_K_GRID - 1) / Q4NX_K_GRID) * Q4NX_K_GRID;
}

int q4nx_pad_n(int n) {
    return n <= 0 ? n : ((n + Q4NX_N_GRID - 1) / Q4NX_N_GRID) * Q4NX_N_GRID;
}

void q4nx_pad_a_f32(const float* A, int M, int real_k, int K, float* out) {
    if (!A || !out || M <= 0 || K <= 0) return;
    const int w = (real_k <= 0 || real_k > K) ? K : real_k;
    for (int m = 0; m < M; m++) {
        float* d = out + (size_t)m * (size_t)K;
        memcpy(d, A + (size_t)m * (size_t)w, sizeof(float) * (size_t)w);
        if (w < K) memset(d + w, 0, sizeof(float) * (size_t)(K - w));
    }
}

void q4nx_host_gemm_f32_padded(const float* A, int a_k, int M, int K, int N, const float* Bmat,
                              float* out) {
    if (!A || !Bmat || !out || M <= 0 || K <= 0 || N <= 0) return;
    /* Same rule as the pad helper: real_k <= 0 (MLA designs) or >= K means nothing to pad. */
    const int w = (a_k <= 0 || a_k > K) ? K : a_k;
    float* pad = NULL;
    const float* src = A;
    if (w < K) {
        pad = (float*)malloc(sizeof(float) * (size_t)M * (size_t)K);
        if (!pad) return;   /* out stays as the caller left it; the forward reports a failed init */
        q4nx_pad_a_f32(A, M, w, K, pad);
        src = pad;
    }
    for (int m = 0; m < M; m++) {
        const float* arow = src + (size_t)m * (size_t)K;
        float* orow = out + (size_t)m * (size_t)N;
        memset(orow, 0, sizeof(float) * (size_t)N);
        for (int k = 0; k < K; k++) {
            const float a = arow[k];
            if (a == 0.0f) continue;
            const float* brow = Bmat + (size_t)k * (size_t)N;
            for (int n = 0; n < N; n++) orow[n] += a * brow[n];
        }
    }
    free(pad);
}

const char* q4nx_design_log_tag(Q4nxDesign role) {
    switch (role) {
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
        case Q4NX_DESIGN_QKVFA_Q:  return "QKVFAQ";
        case Q4NX_DESIGN_QKVFA_KV: return "QKVFAKV";
        default:              return "D";
    }
}

const char* q4nx_design_role_name(Q4nxDesign role) {
    switch (role) {
        case Q4NX_DESIGN_QKV: return "QKV";
        case Q4NX_DESIGN_QKV2: return "QKV2";
        case Q4NX_DESIGN_O: return "O";
        case Q4NX_DESIGN_O2: return "O2";
        case Q4NX_DESIGN_GU: return "GU";
        case Q4NX_DESIGN_G: return "G";
        case Q4NX_DESIGN_U: return "U";
        case Q4NX_DESIGN_D: return "D";
        case Q4NX_DESIGN_PLEGATE: return "PLEGATE";
        case Q4NX_DESIGN_PLEPROJ: return "PLEPROJ";
        case Q4NX_DESIGN_GU2: return "GU2";
        case Q4NX_DESIGN_D2: return "D2";
        case Q4NX_DESIGN_QKVLIN: return "QKVLIN";
        case Q4NX_DESIGN_Z: return "Z";
        case Q4NX_DESIGN_QA: return "QA";
        case Q4NX_DESIGN_QB: return "QB";
        case Q4NX_DESIGN_KVA: return "KVA";
        case Q4NX_DESIGN_DG: return "DG";
        case Q4NX_DESIGN_DU: return "DU";
        case Q4NX_DESIGN_DD: return "DD";
        case Q4NX_DESIGN_GB: return "GB";
        case Q4NX_DESIGN_UB: return "UB";
        case Q4NX_DESIGN_QKVFA_Q: return "QKVFAQ";
        case Q4NX_DESIGN_QKVFA_KV: return "QKVFAKV";
        default: return "?";
    }
}

/* One standard design: the model's real K and N in, the kernel's grids out, the real widths kept
 * beside them.  Role-specific row counts (q_rows/k_rows/v_rows, gate_rows/up_rows) are the
 * caller's, because only it knows what fills N.  MLA keeps its own derivation: its KVA pad is the
 * kernel's 128-column n-tile, not this grid. */
static void geom(Q4nxDesignGeom* g, int M, int K, int N, Q4nxDesign role, const char* tag) {
    memset(g, 0, sizeof(*g));
    g->M = M;
    g->real_k = K;
    g->K = q4nx_pad_k(K);
    g->real_n = (q4nx_pad_n(N) != N) ? N : 0;
    g->N = q4nx_pad_n(N);
    g->role = role;
    set_elf_name(g, tag);
}

int q4nx_derive_decode_designs(const Q4nxDesignGeom* designs, int count,
                               Q4nxDesignGeom* out) {
    int usable = 0;
    if (!out) return 0;
    for (int i = 0; i < count; i++) {
        memset(&out[i], 0, sizeof(out[i]));
        if (!designs) continue;
        const Q4nxDesignGeom* g = &designs[i];
        /* The m16 kernel is m=16 k=64 n=128 over whole column groups, and the groups are an
         * in-ELF loop, so one launch still covers the whole N.  What a geometry needs is the
         * column pass on the grid the family is built on, q4nx_pad_n() (512 columns: 32 rows x 4
         * x cols(4)).  This used to demand N % 1024 == 0 ("whole 8-column groups"), which the
         * family that was actually built contradicts: the oga-qwen25-0.5b set's QKV (N 1536),
         * GU (9728) and lm_head (152064) are all 512-grid ELFs that ran bit-exact over their full
         * width on the array (npu-kernels ogaload/oga-qwen25-0.5b-bench-20261101), and 1024 would
         * have refused three of the five.  A geometry whose twin is simply absent still falls
         * back to its M=128 kernel: the caller probes the filesystem. */
        if (g->K % 64 != 0 || g->N % Q4NX_N_GRID != 0) continue;
        const size_t len = strlen(g->elf_name);
        if (len < 5 || len + 6 >= Q4NX_ELF_NAME_MAX) continue;   /* +"_m16" */
        if (strcmp(g->elf_name + len - 4, ".elf") != 0) continue;
        out[i] = *g;
        out[i].M = Q4NX_DESIGN_M_DECODE;
        snprintf(out[i].elf_name, Q4NX_ELF_NAME_MAX, "%.*s_m16.elf", (int)(len - 4),
                 g->elf_name);
        usable++;
    }
    return usable;
}

int q4nx_derive_designs(const Q4nxModelDims* d, Q4nxDesignGeom* out) {
    int n = q4nx_num_designs(d);
    if (n < 0 || !out) return -1;

    if (d->mla) return derive_mla(d, out);

    const int q_rows = d->NH * d->HD;
    const int kv_rows = d->NKV * d->HD;
    int i = 0;

    /* QKV: K = H, N = q | k | v   (q doubles under attn_output_gate) */
    geom(&out[i], d->M, d->H, q_rows * (d->attn_output_gate ? 2 : 1) + 2 * kv_rows,
         Q4NX_DESIGN_QKV, "QKV");
    out[i].q_rows = q_rows * (d->attn_output_gate ? 2 : 1);
    out[i].k_rows = kv_rows;
    out[i].v_rows = kv_rows;

    /* O: K = NH*HD, N = H */
    i++;
    geom(&out[i], d->M, q_rows, d->H, Q4NX_DESIGN_O, "O");

    /* gate/up: fused into GU, or split into G and U */
    if (d->fused_gu) {
        i++;
        geom(&out[i], d->M, d->H, 2 * d->IM, Q4NX_DESIGN_GU, "GU");
        out[i].gate_rows = d->IM;
        out[i].up_rows = d->IM;
    } else {
        i++;
        geom(&out[i], d->M, d->H, d->IM, Q4NX_DESIGN_G, "G");
        out[i].gate_rows = d->IM;

        i++;
        geom(&out[i], d->M, d->H, d->IM, Q4NX_DESIGN_U, "U");
        out[i].up_rows = d->IM;
    }

    /* D: K = IM, N = H */
    i++;
    geom(&out[i], d->M, d->IM, d->H, Q4NX_DESIGN_D, "D");

    /* Gemma4 varies the attention width by layer type: its full_attention
     * layers use global_head_dim where its sliding layers use head_dim, so a
     * second QKV and O geometry is required (q rows scale with head_dim too). */
    if (has_second_attn(d)) {
        const int nkv2 = d->NKV2 > 0 ? d->NKV2 : d->NKV;
        const int q_rows2 = d->NH * d->HD2;
        const int kv_rows2 = nkv2 * d->HD2;

        i++;
        geom(&out[i], d->M, d->H, q_rows2 + 2 * kv_rows2, Q4NX_DESIGN_QKV2, "QKV");
        out[i].q_rows = q_rows2;
        out[i].k_rows = kv_rows2;
        out[i].v_rows = kv_rows2;

        i++;
        geom(&out[i], d->M, q_rows2, d->H, Q4NX_DESIGN_O2, "O");
    }

    /* Gemma4 per-layer embeddings: a gate H->ple and a projection ple->H. */
    if (d->ple_dim > 0) {
        i++;
        geom(&out[i], d->M, d->H, d->ple_dim, Q4NX_DESIGN_PLEGATE, "PLEGATE");

        i++;
        geom(&out[i], d->M, d->ple_dim, d->H, Q4NX_DESIGN_PLEPROJ, "PLEPROJ");
    }

    /* Gemma4 use_double_wide_mlp: the KV-shared layers carry an MLP that is twice
     * as wide (gate/up are 2*IM rows each), so a second GU/D pair is required.
     * The checkpoint confirms it: Gemma4-E2B's layer-20 mlp.gate_proj has twice the
     * packed rows of layer 0's, while its config sets use_double_wide_mlp=true. */
    if (d->double_wide_mlp) {
        i++;
        /* The ELF name is geometry-derived (N = 4*IM already distinguishes it
         * from the narrow GU), matching the artifacts emitted for Gemma4-E2B. */
        geom(&out[i], d->M, d->H, 4 * d->IM, Q4NX_DESIGN_GU2, "GU");
        out[i].gate_rows = 2 * d->IM;
        out[i].up_rows = 2 * d->IM;

        i++;
        geom(&out[i], d->M, 2 * d->IM, d->H, Q4NX_DESIGN_D2, "D");
    }

    /* GDN linear attention (qwen3_5): a fused q|k|v projection and the gated
     * norm's z projection.  Same kernel format, new N. */
    if (d->lin_kd > 0) {
        i++;
        geom(&out[i], d->M, d->H, 2 * d->lin_kd + d->lin_vd, Q4NX_DESIGN_QKVLIN, "QKV");

        i++;
        geom(&out[i], d->M, d->H, d->lin_vd, Q4NX_DESIGN_Z, "Z");
    }

    return i + 1;
}

const Q4nxDesignGeom* q4nx_find_design(const Q4nxDesignGeom* designs, int count,
                                       Q4nxDesign role) {
    if (!designs) return NULL;
    for (int i = 0; i < count; i++)
        if (designs[i].role == role) return &designs[i];
    return NULL;
}

/* Copy src[n][i] (row-major [rows][K]) into B[i][n0 + n].
 *
 * Tiled: writing one float at B[i*N + n0 + n] while walking n costs a 70 KB stride
 * (N = 17408 -> 68 KB) and one cache line per element, which measured 0.6-1.9 s for
 * a single 5120x17408 projection on strixhalo -- that is the whole reason a first
 * token took minutes, not the pack that follows it. 64x64 tiles make every inner
 * run 64 contiguous floats on both sides (16 KB source tile stays in L1/L2);
 * measured 20-27 ms for the same projection, byte-identical output. */
static void scatter_transposed(const float* src, int K, int rows, int n0, int N,
                               float* B) {
    const int T = 64;
    #pragma omp parallel for schedule(static)
    for (int jb = 0; jb < rows; jb += T) {
        const int jn = (rows - jb < T) ? (rows - jb) : T;
        for (int ib = 0; ib < K; ib += T) {
            const int in = (K - ib < T) ? (K - ib) : T;
            for (int i = 0; i < in; i++) {
                float* dst = B + (size_t)(ib + i) * (size_t)N + (size_t)(n0 + jb);
                for (int b = 0; b < jn; b++)
                    dst[b] = src[(size_t)(jb + b) * (size_t)K + (size_t)(ib + i)];
            }
        }
    }
}

/* Zero the B columns [from, N) of every row: the kernel's n-tile pad.  The pad columns are never
 * read as weights, but they are multiplied, so they must be zero rather than whatever the previous
 * design left in the shared scratch buffer. */
static void zero_cols(float* B, int K, int N, int from) {
    if (from >= N) return;
    const size_t n = (size_t)(N - from) * sizeof(float);
    for (int i = 0; i < K; i++) memset(B + (size_t)i * (size_t)N + from, 0, n);
}

int q4nx_assemble_B(const Q4nxDesignGeom* g, const Q4nxProjections* p, float* B) {
    if (!g || !p || !B || g->K <= 0 || g->N <= 0) return -1;
    const int K = g->K, N = g->N;
    /* real_n is the projection's real output width; the kernel's N may be wider (engine#845: a
     * 1024-column O kernel on an 896-row projection).  Each arm scatters real rows only and
     * reports how many columns it filled; the tail is zeroed once, below. */
    const int rn = g->real_n > 0 ? g->real_n : N;
    if (rn > N) return -1;
    int filled = -1;

    // attn_output_gate: q_rows counts q AND the output gate, but they are ONE
    // stored projection ([q(all heads) | gate(all heads)]); p->q already holds
    // all q_rows rows, so a plain scatter is correct (no gate pointer needed).
    switch (g->role) {
        case Q4NX_DESIGN_QKV:
        case Q4NX_DESIGN_QKV2: {
            if (!p->q || !p->k || !p->v) return -1;
            int n0 = 0;
            scatter_transposed(p->q, K, g->q_rows, n0, N, B); n0 += g->q_rows;
            scatter_transposed(p->k, K, g->k_rows, n0, N, B); n0 += g->k_rows;
            scatter_transposed(p->v, K, g->v_rows, n0, N, B); n0 += g->v_rows;
            filled = n0;
            break;
        }
        case Q4NX_DESIGN_O:
        case Q4NX_DESIGN_O2: {
            if (!p->o) return -1;
            scatter_transposed(p->o, K, rn, 0, N, B);
            filled = rn;
            break;
        }
        case Q4NX_DESIGN_G:
        case Q4NX_DESIGN_DG: {
            if (!p->gate) return -1;
            scatter_transposed(p->gate, K, rn, 0, N, B);
            filled = rn;
            break;
        }
        case Q4NX_DESIGN_GU: {
            if (!p->gate || !p->up) return -1;
            int n0 = 0;
            scatter_transposed(p->gate, K, g->gate_rows, n0, N, B); n0 += g->gate_rows;
            scatter_transposed(p->up, K, g->up_rows, n0, N, B); n0 += g->up_rows;
            filled = n0;
            break;
        }
        /* MLA projections.  QA/QB are plain single-matrix scatters; KVA is
         * stored at kv_lora+rope_dim (576) but the kernel's n-tile forces N=640,
         * so only real_n columns are scattered and the pad columns are zeroed
         * (the weight is never read out of bounds).  O/G/U/D are reused for MLA's
         * o_proj and the expert/dense MLPs, with DG/DU/DD aliased onto G/U/D for
         * the leading dense layer. */
        case Q4NX_DESIGN_QA: {
            if (!p->q) return -1;
            scatter_transposed(p->q, K, rn, 0, N, B);
            filled = rn;
            break;
        }
        case Q4NX_DESIGN_QB: {
            if (!p->k) return -1;
            scatter_transposed(p->k, K, rn, 0, N, B);
            filled = rn;
            break;
        }
        case Q4NX_DESIGN_KVA: {
            if (!p->v) return -1;
            scatter_transposed(p->v, K, rn, 0, N, B);
            filled = rn;
            break;
        }
        case Q4NX_DESIGN_U:
        case Q4NX_DESIGN_DU: {
            if (!p->up) return -1;
            scatter_transposed(p->up, K, rn, 0, N, B);
            filled = rn;
            break;
        }
        case Q4NX_DESIGN_D:
        case Q4NX_DESIGN_DD:
        case Q4NX_DESIGN_D2: {
            if (!p->down) return -1;
            scatter_transposed(p->down, K, rn, 0, N, B);
            filled = rn;
            break;
        }
        case Q4NX_DESIGN_GU2: {
            if (!p->gate || !p->up) return -1;
            int n0 = 0;
            scatter_transposed(p->gate, K, g->gate_rows, n0, N, B); n0 += g->gate_rows;
            scatter_transposed(p->up, K, g->up_rows, n0, N, B); n0 += g->up_rows;
            filled = n0;
            break;
        }
        case Q4NX_DESIGN_PLEGATE:
        case Q4NX_DESIGN_PLEPROJ:
        case Q4NX_DESIGN_QKVLIN:
        case Q4NX_DESIGN_Z: {
            /* All four are a single weight matrix; the caller puts it in
             * Q4nxProjections::ple.  QKVLIN is the GDN fused q|k|v (N=2*KD+VD),
             * Z is the gated norm's in_proj_z (N=VD). */
            if (!p->ple) return -1;
            scatter_transposed(p->ple, K, rn, 0, N, B);
            filled = rn;
            break;
        }
        default:
            return -1;
    }
    if (filled < 0 || filled > N) return -1;
    zero_cols(B, K, N, filled);
    return 0;
}

float q4nx_pack_B_int8(const float* B, int K, int N, int8_t* out, int ld,
                       float* col_scales) {
    double ssum = 0.0;
    for (int j = 0; j < N; j++) {
        // Per-output-column scale: amax_j / 127 (I8Ctx::packB_into).
        float amax = 0.0f;
        for (int i = 0; i < K; i++) {
            float a = fabsf(B[(size_t)i * N + j]);
            if (isfinite(a) && a > amax) amax = a;
        }
        if (amax < 1e-12f) amax = 1.0f;
        const float ts = amax / 127.0f;
        const float tis = 127.0f / amax;
        for (int i = 0; i < K; i++) {
            float v = B[(size_t)i * N + j];
            if (!isfinite(v)) v = 0.0f;
            int x = (int)roundf(v * tis);
            if (x > 127) x = 127;
            else if (x < -127) x = -127;
            out[(size_t)i * ld + j] = (int8_t)x;
        }
        if (col_scales) col_scales[j] = ts;
        ssum += (double)ts;
    }
    return (float)(ssum / (double)N);
}

/* ── The launch contract: the argument widths a full ELF declares ───────────────────────────────
 *
 * An aiecc full ELF carries one STT_OBJECT symbol per kernel argument in .dynsym, named "0", "1",
 * "2", whose st_size is the byte width the kernel's shim DMA touches — the driver patches those very
 * symbols, so this is the artifact's own contract and not a copy of one.  A launcher that passes a
 * narrower buffer does not fail loudly: the DMA runs past the mapping, the IOMMU logs IO_PAGE_FAULTs
 * and the kernel multiplies whatever follows (measured 2026-11-02 on the 896-hidden OGA pack, whose
 * ELF family is (weights, activations, output) = (655,360, 2,048, 4,096) B while the int8 launcher
 * passed (16,384, 1,048,576, 65,536) B: 639 KB of weight stream read past a 16 KB BO).  Reading the
 * widths here lets a load refuse instead of dispatching.
 *
 * Returns the number of argument widths found (0..3), or -1 when the file is not a 32-bit
 * little-endian ELF or has no .dynsym.  out[] entries that were not declared stay 0. */
/* Which launch family does an artifact's declaration describe?
 *
 * The forge-built q4_1 GEMV family takes (weights, activations, outputs) = (a whole tile stream,
 * K bf16, N f32) and its .dynsym symbols are PER CORE: measured 2026-11-02 on the OGA 0.5B pack's
 * five ELFs, arg0 and arg2 are 1/cores of the stream and of the output, while arg1 (the whole
 * activation row) is not split.  K=N=0 means "no opinion", -1 an unusable declaration.
 *
 * Kept here, as arithmetic over the declarations, so the rule that decides how a launch is shaped
 * can be tested without a device or an artifact (tests/npu_elf_args_test.c).
 */
int q4nx_gemv_family(const uint64_t declared[3], int K, int N) {
    if (!declared || K <= 0 || N <= 0) return 0;
    if (K % 256 || N % 32) return 0;
    if (!declared[0] || !declared[1] || !declared[2]) return 0;
    const uint64_t want_w = (uint64_t)(K / 256) * (uint64_t)(N / 32) * 5120u;
    if (want_w % declared[0]) return 0;
    const uint64_t cores = want_w / declared[0];
    if (cores < 1) return 0;
    return declared[1] == (uint64_t)K * 2u && declared[2] * cores == (uint64_t)N * 4u;
}

int q4nx_elf_arg_sizes(const void* elf, size_t n, uint64_t out[3]) {
    if (out) { out[0] = out[1] = out[2] = 0; }
    if (!elf || !out || n < 52) return -1;
    const uint8_t* b = (const uint8_t*)elf;
    if (memcmp(b, "\x7f" "ELF", 4) || b[4] != 1 || b[5] != 1) return -1;   /* ELF32, LSB */
#define RD16(p) ((uint32_t)(p)[0] | ((uint32_t)(p)[1] << 8))
#define RD32(p) ((uint64_t)(p)[0] | ((uint64_t)(p)[1] << 8) | ((uint64_t)(p)[2] << 16) | \
                 ((uint64_t)(p)[3] << 24))
    const uint64_t shoff = RD32(b + 0x20);
    const uint32_t shentsize = RD16(b + 0x2E), shnum = RD16(b + 0x30);
    if (shentsize < 40 || !shnum || shnum > 512) return -1;      /* real artifacts have ~13 */
    if (shoff + (uint64_t)shentsize * shnum > n) return -1;
    struct sh { uint64_t off, size, link, entsize, type; };
    struct sh sect[512];
    for (uint32_t i = 0; i < shnum; i++) {
        const uint8_t* p = b + shoff + (uint64_t)i * shentsize;
        sect[i].type = RD32(p + 4);
        sect[i].off = RD32(p + 16);
        sect[i].size = RD32(p + 20);
        sect[i].link = RD32(p + 24);
        sect[i].entsize = RD32(p + 36);
        if (sect[i].type != 8 /* SHT_NOBITS */ && sect[i].off + sect[i].size > n) return -1;
    }
    int found = 0;
    for (uint32_t i = 0; i < shnum; i++) {
        if (sect[i].type != 11 /* SHT_DYNSYM */ || sect[i].link >= shnum) continue;
        const struct sh* st = &sect[sect[i].link];              /* the symbols' string table */
        const uint32_t esz = sect[i].entsize ? (uint32_t)sect[i].entsize : 16;
        if (esz < 16) continue;
        for (uint64_t s = 1; s * esz + 16 <= sect[i].size; s++) {
            const uint8_t* sym = b + sect[i].off + s * esz;
            const uint64_t name = RD32(sym);
            const uint64_t size = RD32(sym + 8);
            const uint8_t info = sym[12];
            if ((info & 0xF) != 1 /* STT_OBJECT */ || !size) continue;
            if (name >= st->size) continue;
            const uint8_t* nm = b + st->off + name;
            const uint64_t room = st->size - name;
            if (room < 1) continue;
            /* the argument symbols are named by their index: "0", "1", "2" */
            if (room >= 2 && nm[1] != 0) continue;
            if (nm[0] < '0' || nm[0] > '2') continue;
            const int idx = nm[0] - '0';
            if (!out[idx]) { out[idx] = size; found++; }
        }
    }
#undef RD16
#undef RD32
    return found ? found : -1;
}
