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

// npu_design_grid_test -- the GEMM designs are derived on the grids the container and the kernels
// actually use (engine#845): K on the q4_1 chunk's 256 columns, N on the full ELF's 512-column
// pass, with the model's real widths kept beside them.
//
// Host-only, no model, no device: it calls q4nx_derive_designs()/q4nx_assemble_B() and checks the
// numbers against geometry worked out by hand.
//
// The case that motivated it: Qwen2.5-0.5B (hidden 896, intermediate 4864, 14 heads / 2 KV heads,
// head_dim 64) is stored by `1bit npu-pack import` with K padded to 1024, and its five M=16 ELFs were
// built -- and timed on the array -- at full_i8_{QKV,O,GU,D}_K1024_N{1536,1024,9728,1024}_m16.
// Before this, the forward derived K=896/N=1152, probed names no kernel has, and dequantised a
// 1024-column tensor at 896 columns (which the chunk decoder refuses outright).
#include "q4nx_pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check(int ok, const char* what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

static void check_int(int got, int want, const char* what) {
    printf("%s %s (got %d, want %d)\n", got == want ? "ok  " : "FAIL", what, got, want);
    if (got != want) failures++;
}

static Q4nxDesignGeom* find(Q4nxDesignGeom* d, int n, Q4nxDesign role) {
    for (int i = 0; i < n; i++)
        if (d[i].role == role) return &d[i];
    return NULL;
}

/* Qwen2.5-0.5B-Instruct, the OGA pack geometry. */
static void oga_geometry(void) {
    Q4nxModelDims d;
    memset(&d, 0, sizeof d);
    d.H = 896; d.NH = 14; d.NKV = 2; d.HD = 64; d.IM = 4864;
    d.fused_gu = 1; d.M = Q4NX_DESIGN_M_PREFILL;

    check_int(q4nx_pad_k(896), 1024, "hidden 896 pads to 1024 on the container's K grid");
    check_int(q4nx_pad_k(4864), 4864, "intermediate 4864 is already on the K grid");
    check_int(q4nx_pad_n(1152), 1536, "QKV N 1152 pads to 1536");
    check_int(q4nx_pad_n(9728), 9728, "GU N 9728 is already on the N grid");
    check_int(q4nx_pad_n(151936), 152064, "lm_head N 151936 pads to 152064");

    Q4nxDesignGeom g[Q4NX_MAX_DESIGNS];
    const int n = q4nx_derive_designs(&d, g);
    check_int(n, 4, "four designs (fused GU)");

    Q4nxDesignGeom* qkv = find(g, n, Q4NX_DESIGN_QKV);
    check(qkv != NULL, "a QKV design is derived");
    if (qkv) {
        check_int(qkv->K, 1024, "QKV K is the container's padded K");
        check_int(qkv->real_k, 896, "QKV keeps the real K");
        check_int(qkv->N, 1536, "QKV N is on the kernel's N grid");
        check_int(qkv->real_n, 1152, "QKV keeps the real N");
        check_int(qkv->q_rows, 896, "QKV q rows are the real q width");
        check_int(qkv->k_rows, 128, "QKV k rows are the real kv width");
        check(strcmp(qkv->elf_name, "full_i8_QKV_K1024_N1536.elf") == 0,
              "the QKV artifact name is the one that was built");
    }
    Q4nxDesignGeom* o = find(g, n, Q4NX_DESIGN_O);
    check(o != NULL, "an O design is derived");
    if (o) {
        check_int(o->K, 1024, "O K is padded (NH*HD = 896 -> 1024)");
        check_int(o->N, 1024, "O N is padded (hidden 896 -> 1024)");
        check_int(o->real_n, 896, "O keeps the real N");
        check(strcmp(o->elf_name, "full_i8_O_K1024_N1024.elf") == 0, "the O artifact name is the one that was built");
    }
    Q4nxDesignGeom* gu = find(g, n, Q4NX_DESIGN_GU);
    check(gu != NULL, "a GU design is derived");
    if (gu) {
        check_int(gu->K, 1024, "GU K is padded");
        check_int(gu->N, 9728, "GU N is 2*IM, already on the grid");
        check_int(gu->real_n, 0, "GU needs no N pad");
        check(strcmp(gu->elf_name, "full_i8_GU_K1024_N9728.elf") == 0, "the GU artifact name is the one that was built");
    }
    Q4nxDesignGeom* dn = find(g, n, Q4NX_DESIGN_D);
    check(dn != NULL, "a D design is derived");
    if (dn) {
        check_int(dn->K, 4864, "D K is the intermediate size, already on the grid");
        check_int(dn->N, 1024, "D N is padded (hidden 896 -> 1024)");
        check(strcmp(dn->elf_name, "full_i8_D_K4864_N1024.elf") == 0, "the D artifact name is the one that was built");
    }

    /* The M=16 twins must be found for every role of this geometry: the five ELFs that exist are
     * _m16 ones, and the old guard (N % 1024 == 0) refused QKV, GU and the head. */
    Q4nxDesignGeom t[Q4NX_MAX_DESIGNS];
    const int nt = q4nx_derive_decode_designs(g, n, t);
    check_int(nt, 4, "every design of this geometry has an M=16 twin");
    for (int i = 0; i < n; i++) {
        if (!t[i].M) continue;
        char what[96];
        snprintf(what, sizeof what, "twin %d keeps K/N and is named _m16", i);
        check(t[i].K == g[i].K && t[i].N == g[i].N && t[i].M == Q4NX_DESIGN_M_DECODE &&
                  strstr(t[i].elf_name, "_m16.elf") != NULL,
              what);
    }

    /* assemble_B: a synthetic source whose every element encodes its (row, k) puts the weights where
     * the kernel reads them, and every pad column is zero. */
    if (o) {
        const int rows = 896, K = o->K, N = o->N;
        float* src = calloc((size_t)rows * K, sizeof(float));
        float* B = malloc((size_t)K * N * sizeof(float));
        check(src && B, "the assemble scratch allocates");
        if (src && B) {
            for (int r = 0; r < rows; r++)
                for (int k = 0; k < K; k++)
                    src[(size_t)r * K + k] = (k < 896) ? 1.0f : 0.0f;   // the container's pad is zero
            memset(B, 0x7f, (size_t)K * N * sizeof(float));             // poison, so a pad that is
                                                                       // not written shows up
            Q4nxProjections p;
            memset(&p, 0, sizeof p);
            p.o = src;
            check_int(q4nx_assemble_B(o, &p, B), 0, "assemble_B accepts the padded O design");
            int bad_weight = 0, bad_pad = 0;
            for (int k = 0; k < K; k++) {
                for (int col = 0; col < N; col++) {
                    const float want = (col < 896 && k < 896) ? 1.0f : 0.0f;
                    const float got = B[(size_t)k * N + col];
                    if (col >= 896) { if (got != 0.0f) bad_pad++; }
                    else if (got != want) bad_weight++;
                }
            }
            check_int(bad_weight, 0, "every real weight lands in its (k, column) slot");
            check_int(bad_pad, 0, "every column past real_n is zeroed");
        }
        free(src);
        free(B);
    }

    /* QKV: the three projections are stacked along N at their real row counts and the tail of the
     * 1536-wide kernel is zero. */
    if (qkv) {
        const int K = qkv->K, N = qkv->N;
        float* q = calloc((size_t)896 * K, sizeof(float));
        float* k = calloc((size_t)128 * K, sizeof(float));
        float* v = calloc((size_t)128 * K, sizeof(float));
        float* B = malloc((size_t)K * N * sizeof(float));
        if (q && k && v && B) {
            for (int r = 0; r < 896; r++) q[(size_t)r * K] = 1.0f;   // k column 0 marker
            for (int r = 0; r < 128; r++) k[(size_t)r * K] = 2.0f;
            for (int r = 0; r < 128; r++) v[(size_t)r * K] = 3.0f;
            memset(B, 0x7f, (size_t)K * N * sizeof(float));
            Q4nxProjections p;
            memset(&p, 0, sizeof p);
            p.q = q; p.k = k; p.v = v;
            check_int(q4nx_assemble_B(qkv, &p, B), 0, "assemble_B accepts the padded QKV design");
            check(B[0] == 1.0f && B[(size_t)896] == 2.0f && B[(size_t)1024] == 3.0f,
                  "q, k and v land at their real row offsets");
            int bad_tail = 0;
            for (int i = 0; i < K; i++)
                for (int col = 1152; col < N; col++)
                    if (B[(size_t)i * N + col] != 0.0f) bad_tail++;
            check_int(bad_tail, 0, "the 384 columns past q|k|v are zeroed");
        }
        free(q); free(k); free(v); free(B);
    }
}

/* Qwen3-0.6B, the geometry every earlier model had: both grids are already satisfied, so this is
 * the regression case -- the padding must be the identity. */
static void on_grid_geometry(void) {
    Q4nxModelDims d;
    memset(&d, 0, sizeof d);
    d.H = 1024; d.NH = 16; d.NKV = 8; d.HD = 128; d.IM = 3072;
    d.fused_gu = 1; d.M = Q4NX_DESIGN_M_PREFILL;
    Q4nxDesignGeom g[Q4NX_MAX_DESIGNS];
    const int n = q4nx_derive_designs(&d, g);
    check_int(n, 4, "four designs for an on-grid model");
    Q4nxDesignGeom* qkv = find(g, n, Q4NX_DESIGN_QKV);
    check(qkv && strcmp(qkv->elf_name, "full_i8_QKV_K1024_N4096.elf") == 0,
          "an on-grid model's artifact names are unchanged");
    check(qkv && qkv->real_n == 0 && qkv->real_k == 1024, "an on-grid model records no pad");
    const int nt = q4nx_derive_decode_designs(g, n, (Q4nxDesignGeom[Q4NX_MAX_DESIGNS]){{0}});
    check_int(nt, 4, "an on-grid model still derives every M=16 twin");
}

int main(void) {
    oga_geometry();
    on_grid_geometry();
    printf("%s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 1 : 0;
}
