// Copyright (c) 2026 bong-water-water-bong
// q4nx_pack.c — see q4nx_pack.h.  The design set is derived from architecture
// dims; nothing here is keyed on a model name or tag, and no geometry literal
// for any specific model exists in this file.
#include "q4nx_pack.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* A model whose attention width varies by layer type (Gemma4 full vs sliding)
 * needs a second QKV and O geometry derived from its other head_dim. */
static int has_second_attn(const Q4nxModelDims* d) {
    return d->HD2 > 0 && d->HD2 != d->HD;
}

int q4nx_num_designs(const Q4nxModelDims* d) {
    if (!d || d->H <= 0 || d->NH <= 0 || d->NKV <= 0 || d->HD <= 0 || d->IM <= 0)
        return -1;
    return (d->fused_gu ? 4 : 5) + (d->ple_dim > 0 ? 2 : 0) +
           (d->double_wide_mlp ? 2 : 0) + (has_second_attn(d) ? 2 : 0) +
           (d->lin_kd > 0 ? 2 : 0);
}

static void set_elf_name(Q4nxDesignGeom* g, const char* role) {
    snprintf(g->elf_name, Q4NX_ELF_NAME_MAX, "full_i8_%s_K%d_N%d.elf", role, g->K,
             g->N);
}

int q4nx_derive_designs(const Q4nxModelDims* d, Q4nxDesignGeom* out) {
    int n = q4nx_num_designs(d);
    if (n < 0 || !out) return -1;

    const int q_rows = d->NH * d->HD;
    const int kv_rows = d->NKV * d->HD;
    int i = 0;

    /* QKV: K = H, N = q | k | v   (q doubles under attn_output_gate) */
    memset(&out[i], 0, sizeof(out[i]));
    out[i].M = d->M;
    out[i].K = d->H;
    out[i].N = q_rows * (d->attn_output_gate ? 2 : 1) + 2 * kv_rows;
    out[i].role = Q4NX_DESIGN_QKV;
    out[i].q_rows = q_rows * (d->attn_output_gate ? 2 : 1);
    out[i].k_rows = kv_rows;
    out[i].v_rows = kv_rows;
    set_elf_name(&out[i], "QKV");

    /* O: K = NH*HD, N = H */
    i++;
    memset(&out[i], 0, sizeof(out[i]));
    out[i].M = d->M;
    out[i].K = q_rows;
    out[i].N = d->H;
    out[i].role = Q4NX_DESIGN_O;
    set_elf_name(&out[i], "O");

    /* gate/up: fused into GU, or split into G and U */
    if (d->fused_gu) {
        i++;
        memset(&out[i], 0, sizeof(out[i]));
        out[i].M = d->M;
        out[i].K = d->H;
        out[i].N = 2 * d->IM;
        out[i].role = Q4NX_DESIGN_GU;
        out[i].gate_rows = d->IM;
        out[i].up_rows = d->IM;
        set_elf_name(&out[i], "GU");
    } else {
        i++;
        memset(&out[i], 0, sizeof(out[i]));
        out[i].M = d->M;
        out[i].K = d->H;
        out[i].N = d->IM;
        out[i].role = Q4NX_DESIGN_G;
        out[i].gate_rows = d->IM;
        set_elf_name(&out[i], "G");

        i++;
        memset(&out[i], 0, sizeof(out[i]));
        out[i].M = d->M;
        out[i].K = d->H;
        out[i].N = d->IM;
        out[i].role = Q4NX_DESIGN_U;
        out[i].up_rows = d->IM;
        set_elf_name(&out[i], "U");
    }

    /* D: K = IM, N = H */
    i++;
    memset(&out[i], 0, sizeof(out[i]));
    out[i].M = d->M;
    out[i].K = d->IM;
    out[i].N = d->H;
    out[i].role = Q4NX_DESIGN_D;
    set_elf_name(&out[i], "D");

    /* Gemma4 varies the attention width by layer type: its full_attention
     * layers use global_head_dim where its sliding layers use head_dim, so a
     * second QKV and O geometry is required (q rows scale with head_dim too). */
    if (has_second_attn(d)) {
        const int nkv2 = d->NKV2 > 0 ? d->NKV2 : d->NKV;
        const int q_rows2 = d->NH * d->HD2;
        const int kv_rows2 = nkv2 * d->HD2;

        i++;
        memset(&out[i], 0, sizeof(out[i]));
        out[i].M = d->M;
        out[i].K = d->H;
        out[i].N = q_rows2 + 2 * kv_rows2;
        out[i].role = Q4NX_DESIGN_QKV2;
        out[i].q_rows = q_rows2;
        out[i].k_rows = kv_rows2;
        out[i].v_rows = kv_rows2;
        set_elf_name(&out[i], "QKV");

        i++;
        memset(&out[i], 0, sizeof(out[i]));
        out[i].M = d->M;
        out[i].K = q_rows2;
        out[i].N = d->H;
        out[i].role = Q4NX_DESIGN_O2;
        set_elf_name(&out[i], "O");
    }

    /* Gemma4 per-layer embeddings: a gate H->ple and a projection ple->H. */
    if (d->ple_dim > 0) {
        i++;
        memset(&out[i], 0, sizeof(out[i]));
        out[i].M = d->M;
        out[i].K = d->H;
        out[i].N = d->ple_dim;
        out[i].role = Q4NX_DESIGN_PLEGATE;
        set_elf_name(&out[i], "PLEGATE");

        i++;
        memset(&out[i], 0, sizeof(out[i]));
        out[i].M = d->M;
        out[i].K = d->ple_dim;
        out[i].N = d->H;
        out[i].role = Q4NX_DESIGN_PLEPROJ;
        set_elf_name(&out[i], "PLEPROJ");
    }

    /* Gemma4 use_double_wide_mlp: the KV-shared layers carry an MLP that is twice
     * as wide (gate/up are 2*IM rows each), so a second GU/D pair is required.
     * The checkpoint confirms it: Gemma4-E2B's layer-20 mlp.gate_proj has twice the
     * packed rows of layer 0's, while its config sets use_double_wide_mlp=true. */
    if (d->double_wide_mlp) {
        i++;
        memset(&out[i], 0, sizeof(out[i]));
        out[i].M = d->M;
        out[i].K = d->H;
        out[i].N = 4 * d->IM;
        out[i].role = Q4NX_DESIGN_GU2;
        out[i].gate_rows = 2 * d->IM;
        out[i].up_rows = 2 * d->IM;
        /* The ELF name is geometry-derived (N = 4*IM already distinguishes it
         * from the narrow GU), matching the artifacts emitted for Gemma4-E2B. */
        set_elf_name(&out[i], "GU");

        i++;
        memset(&out[i], 0, sizeof(out[i]));
        out[i].M = d->M;
        out[i].K = 2 * d->IM;
        out[i].N = d->H;
        out[i].role = Q4NX_DESIGN_D2;
        set_elf_name(&out[i], "D");
    }

    /* GDN linear attention (qwen3_5): a fused q|k|v projection and the gated
     * norm's z projection.  Same kernel format, new N. */
    if (d->lin_kd > 0) {
        i++;
        memset(&out[i], 0, sizeof(out[i]));
        out[i].M = d->M;
        out[i].K = d->H;
        out[i].N = 2 * d->lin_kd + d->lin_vd;
        out[i].role = Q4NX_DESIGN_QKVLIN;
        set_elf_name(&out[i], "QKV");

        i++;
        memset(&out[i], 0, sizeof(out[i]));
        out[i].M = d->M;
        out[i].K = d->H;
        out[i].N = d->lin_vd;
        out[i].role = Q4NX_DESIGN_Z;
        set_elf_name(&out[i], "Z");
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

/* Copy src[n][i] (row-major [rows][K]) into B[i][n0 + n]. */
static void scatter_transposed(const float* src, int K, int rows, int n0, int N,
                               float* B) {
    for (int n = 0; n < rows; n++) {
        const float* row = src + (size_t)n * (size_t)K;
        for (int i = 0; i < K; i++)
            B[(size_t)i * (size_t)N + (size_t)(n0 + n)] = row[i];
    }
}

int q4nx_assemble_B(const Q4nxDesignGeom* g, const Q4nxProjections* p, float* B) {
    if (!g || !p || !B || g->K <= 0 || g->N <= 0) return -1;
    const int K = g->K, N = g->N;

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
            return n0 == N ? 0 : -1;
        }
        case Q4NX_DESIGN_O:
        case Q4NX_DESIGN_O2: {
            if (!p->o) return -1;
            scatter_transposed(p->o, K, N, 0, N, B);
            return 0;
        }
        case Q4NX_DESIGN_GU: {
            if (!p->gate || !p->up) return -1;
            int n0 = 0;
            scatter_transposed(p->gate, K, g->gate_rows, n0, N, B); n0 += g->gate_rows;
            scatter_transposed(p->up, K, g->up_rows, n0, N, B); n0 += g->up_rows;
            return n0 == N ? 0 : -1;
        }
        case Q4NX_DESIGN_G: {
            if (!p->gate) return -1;
            scatter_transposed(p->gate, K, N, 0, N, B);
            return 0;
        }
        case Q4NX_DESIGN_U: {
            if (!p->up) return -1;
            scatter_transposed(p->up, K, N, 0, N, B);
            return 0;
        }
        case Q4NX_DESIGN_D: {
            if (!p->down) return -1;
            scatter_transposed(p->down, K, N, 0, N, B);
            return 0;
        }
        case Q4NX_DESIGN_GU2: {
            if (!p->gate || !p->up) return -1;
            int n0 = 0;
            scatter_transposed(p->gate, K, g->gate_rows, n0, N, B); n0 += g->gate_rows;
            scatter_transposed(p->up, K, g->up_rows, n0, N, B); n0 += g->up_rows;
            return n0 == N ? 0 : -1;
        }
        case Q4NX_DESIGN_D2: {
            if (!p->down) return -1;
            scatter_transposed(p->down, K, N, 0, N, B);
            return 0;
        }
        case Q4NX_DESIGN_PLEGATE:
        case Q4NX_DESIGN_PLEPROJ:
        case Q4NX_DESIGN_QKVLIN:
        case Q4NX_DESIGN_Z: {
            /* All four are a single weight matrix; the caller puts it in
             * Q4nxProjections::ple.  QKVLIN is the GDN fused q|k|v (N=2*KD+VD),
             * Z is the gated norm's in_proj_z (N=VD). */
            if (!p->ple) return -1;
            scatter_transposed(p->ple, K, N, 0, N, B);
            return 0;
        }
        default:
            return -1;
    }
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
