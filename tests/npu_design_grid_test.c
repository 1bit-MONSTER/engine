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
#include "q4nx_dequant.h"
#include "q4nx_pack.h"
#include "q4nx_head_image.h"

#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;


static void truncate_file(const char* path, long bytes) {
    FILE* f = fopen(path, "r+b");
    if (!f) return;
    if (ftruncate(fileno(f), bytes) != 0) { /* best effort: the arm asserts the refusal either way */ }
    fclose(f);
}

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

    /* THE BYTE-EXACT TARGET for a fused weight packer (engine #1416). The non-folded route currently materialises
     * B[K][N] in float32 (356 MB per projection on a 27B model) and then quantises it; the fix is to produce the int8
     * image directly from the container's tiles without the float matrix. That fix can only be argued as "byte-exact"
     * if the two-step composition it replaces is pinned first, which is what this does: assemble from crafted
     * projections, quantise, and assert every code and scale against an independently computed expectation plus a
     * literal golden of the first codes. A fused implementation has to reproduce all three. */
    if (o) {
        const int K = o->K, N = o->N;
        float* src = calloc((size_t)896 * K, sizeof(float));
        float* B = malloc((size_t)K * N * sizeof(float));
        int8_t* out = malloc((size_t)K * N);
        float* scales = malloc((size_t)N * sizeof(float));
        check(src && B && out && scales, "the assemble/pack fixture allocates");
        if (src && B && out && scales) {
            for (int r = 0; r < 896; r++)
                for (int k = 0; k < K; k++)
                    src[(size_t)r * K + k] = (k < 896) ? (float)((r * 7 + k * 13) % 23) * 0.05f - 0.5f : 0.0f;
            Q4nxProjections p;
            memset(&p, 0, sizeof p);
            p.o = src;
            check_int(q4nx_assemble_B(o, &p, B), 0, "fixture: assemble_B accepts the crafted O design");
            /* ld is the ROW stride of the code image: q4nx_pack_B_int8 writes out[i * ld + j] with j < N, so it must
             * be >= N. It happens that K == N in this fixture; passing K would be the same bug #1713 hit as a silent
             * self-overwrite on a non-square grid. */
            const int ld = N;
            const float sout = q4nx_pack_B_int8(B, K, N, out, ld, scales);
            /* expectation, computed here rather than read back from the implementation */
            int bad_code = 0, bad_scale = 0;
            float sum_scale = 0.0f;
            for (int col = 0; col < N; col++) {
                float amax = 0.0f;
                for (int k = 0; k < K; k++) {
                    const float a = fabsf(B[(size_t)k * N + col]);
                    if (a > amax) amax = a;
                }
                /* The contract, not my assumption: an all-zero column (every pad column of this design) takes the
                 * implementation's amax floor of 1.0, so its scale is 1/127 rather than 0. A fused packer has to
                 * reproduce that floor too, which is why it is asserted rather than skipped. */
                if (amax < 1e-12f) amax = 1.0f;
                const float ws = amax / 127.0f;
                sum_scale += ws;
                if (fabsf(scales[col] - ws) > 1e-6f * (ws > 1.0f ? ws : 1.0f)) bad_scale++;
                for (int k = 0; k < K; k++) {
                    int q = (int)lroundf(B[(size_t)k * N + col] / ws);
                    if (q > 127) q = 127;
                    else if (q < -127) q = -127;
                    if (out[(size_t)k * K + col] != (int8_t)q) bad_code++;
                }
            }
            check_int(bad_scale, 0, "pack_B -> every column scale is max|B_col|/127");
            check_int(bad_code, 0, "pack_B -> every code is round(B / column scale), clamped to +-127");
            check(fabsf(sout - sum_scale / (float)N) < 1e-6f,
                  "pack_B -> the returned scale is the mean of the column scales");
            /* The literal golden, ASSERTED rather than printed. This is the one check that does not derive from what
             * q4nx_assemble_B happened to produce, so it is what a fused packer -- which never goes through assemble -- is
             * measured against; a `check(x, x)` here would pass whatever the composition did. */
            /* column 0, rows 0..3 (the codes are stored row-major at stride ld), not "codes[0..3]" as the label used to say */
            check(out[0] == -106 && out[ld] == 32 && out[2 * ld] == -74 && out[3 * ld] == 64,
                  "fixture: column 0 rows 0..3 are the pinned golden (-106 32 -74 64)");
            check(fabsf(scales[0] - 0.004724f) < 5e-7f && fabsf(scales[1] - 0.004724f) < 5e-7f,
                  "fixture: scales[0..1] are the pinned golden (0.004724)");
            check(fabsf(sout - 0.005118f) < 5e-7f, "fixture: the mean column scale is the pinned golden (0.005118)");
            printf("  assemble->pack fixture: K=%d N=%d  codes[0..3]=%d %d %d %d  scales[0..1]=%.6f %.6f  mean=%.6f\n",
                   K, N, (int)out[0], (int)out[K], (int)out[2 * K], (int)out[3 * K], (double)scales[0],
                   (double)scales[1], (double)sout);
        }
        free(src);
        free(B);
        free(out);
        free(scales);
    }

    /* THE FUSED PATH (engine #1416): chunks -> int8 without the K x N float buffer. It has to be byte-identical to
     * `q4nx_dequant_tensor() -> q4nx_pack_B_int8()`, so this runs BOTH on the same synthetic tiles and compares the codes,
     * the scales and the returned mean. A synthetic 32x256 q4_1 tile is 5120 B of (16 B scale/zero header + 128 nibble
     * pairs) x 8 groups; the encoder here is only required to be a *valid* q4_1 tile, not a meaningful weight. */
    {
        const int K = 64, N = 512;                     /* 2 x 2 tiles */
        const size_t ntr = (size_t)(K / 32), ntc = (size_t)(N / 256);
        uint8_t* tiles = malloc(ntr * ntc * 5120);
        float* B = malloc((size_t)K * N * sizeof(float));
        int8_t* ref = malloc((size_t)K * N);
        int8_t* fused = malloc((size_t)K * N);
        float* sc_ref = malloc((size_t)N * sizeof(float));
        float* sc_fused = malloc((size_t)N * sizeof(float));
        check(tiles && B && ref && fused && sc_ref && sc_fused, "fused fixture allocates");
        if (tiles && B && ref && fused && sc_ref && sc_fused) {
            for (size_t t = 0; t < ntr * ntc; t++)
                for (size_t i = 0; i < 5120; i++)
                    tiles[t * 5120 + i] = (uint8_t)((i * 37 + t * 91) & 0xFF);
            /* The reference is `dequant_any`'s own call for this chunk kind: q4nx_dequant_tensor_chunked with
             * chunk_bytes = 5120. Plain q4nx_dequant_tensor() dispatches differently (it sniffs the byte count for a
             * packed grid) and left part of B unwritten, which the first version of this test read as a mismatch in the
             * fused path -- it was the reference that was wrong for these tiles. */
            check_int(q4nx_dequant_tensor_chunked(tiles, ntr * ntc * 5120, K, N, 5120, B), 0,
                      "fused: the reference path dequantises the synthetic tiles");
            const int ld = N;   // q4nx_pack_B_int8 writes out[i * ld + j] with i < K, j < N, so ld must be >= N
            const float m_ref = q4nx_pack_B_int8(B, K, N, ref, ld, sc_ref);
            const int rc = q4nx_pack_q4_1_tiles_to_int8(tiles, ntr * ntc * 5120, K, N, fused, ld, sc_fused);
            check_int(rc, 0, "fused: q4nx_pack_q4_1_tiles_to_int8 accepts the same tiles");
            int bad_code = 0, bad_scale = 0;
            for (int i = 0; i < K * N; i++)
                if (fused[i] != ref[i]) {
                    if (bad_code < 6)
                        printf("    mismatch at row %d col %d: fused %d ref %d\n", i / N, i % N, (int)fused[i],
                               (int)ref[i]);
                    bad_code++;
                }
            for (int j = 0; j < N; j++) if (sc_fused[j] != sc_ref[j]) bad_scale++;
            check_int(bad_code, 0, "fused: every code is byte-identical to dequant -> pack");
            check_int(bad_scale, 0, "fused: every column scale is byte-identical as well");
            /* the mean the two-step path returns is the mean of the same scales, so it agrees by construction */
            double ssum = 0.0;
            for (int j = 0; j < N; j++) ssum += (double)sc_fused[j];
            check(fabsf((float)(ssum / N) - m_ref) < 1e-6f,
                  "fused: the mean column scale matches the two-step return value");
            /* a wrong chunk kind is refused by name rather than quantised from garbage */
            check_int(q4nx_pack_q4_1_tiles_to_int8(tiles, ntr * ntc * 5120 - 1, K, N, fused, ld, sc_fused), -2,
                      "fused: a non-q4_1 byte count returns -2 so the caller keeps the two-step path");
            printf("  fused chunks->int8: K=%d N=%d  %d codes and %d scales byte-identical to dequant->pack\n",
                   K, N, K * N, N);
        }
        free(tiles); free(B); free(ref); free(fused); free(sc_ref); free(sc_fused);
    }

    /* THE HEAD TILE'S LAYOUT CONTRACT (engine #1716, spec before implementation). The forward packs the lm head one
     * tile at a time from a scratch buffer it fills as
     *
     *     bt[k * T + (n - n0)] = lm_at(n, k)          // K = kpad rows, T columns per tile
     *     packB_into(bo, bt.data(), kpad, T, ...)     -> packB_blocked -> q4nx_pack_B_int8(bt, kpad, T, ...)
     *
     * A head-image cache has to reproduce that byte for byte, so this pins the contract it must match: one scale per
     * COLUMN of the tile (T of them), codes in row-major order, and the axis order load-bearing -- which is the property
     * a transposed or row/column-swapped builder would get wrong while still producing plausible numbers. */
    {
        const int kpad = 96, T = 64;                 /* a head tile: kpad along K, T along N */
        float* bt = malloc((size_t)kpad * T * sizeof(float));
        int8_t* codes = malloc((size_t)kpad * T);
        float* scales = malloc((size_t)T * sizeof(float));
        check(bt && codes && scales, "head tile: the layout fixture allocates");
        if (bt && codes && scales) {
            for (int k = 0; k < kpad; k++)
                for (int n = 0; n < T; n++) bt[(size_t)k * T + n] = (float)((k * 13 + n * 7) % 97) * 0.01f - 0.5f;
            const float m = q4nx_pack_B_int8(bt, kpad, T, codes, T, scales);
            /* (a) one scale per column, and it is that column's absmax/127 (with the all-zero floor) */
            int bad_scale = 0;
            for (int n = 0; n < T; n++) {
                float amax = 0.0f;
                for (int k = 0; k < kpad; k++) {
                    const float a = fabsf(bt[(size_t)k * T + n]);
                    if (a > amax) amax = a;
                }
                if (amax < 1e-12f) amax = 1.0f;
                if (fabsf(scales[n] - amax / 127.0f) > 1e-7f) bad_scale++;
            }
            check_int(bad_scale, 0, "head tile: one scale per COLUMN of the tile (T of them), absmax/127");
            double ssum = 0.0;
            for (int n = 0; n < T; n++) ssum += (double)scales[n];
            check(fabsf((float)(ssum / T) - m) < 1e-6f, "head tile: the returned mean is over the T columns");
            /* (b) codes are row-major over (k, n) and match the column's scale */
            int bad_code = 0;
            for (int k = 0; k < kpad; k++)
                for (int n = 0; n < T; n++) {
                    int want = (int)lroundf(bt[(size_t)k * T + n] / scales[n]);
                    if (want > 127) want = 127;
                    else if (want < -127) want = -127;
                    if (codes[(size_t)k * T + n] != (int8_t)want) bad_code++;
                }
            check_int(bad_code, 0, "head tile: codes are row-major over (k, n) with the column's scale");
            /* (c) the axis order is load-bearing: a transposed scratch is a DIFFERENT image, so a builder that swaps
             *     rows and columns cannot pass by producing plausible numbers. */
            float* tr = malloc((size_t)kpad * T * sizeof(float));
            int8_t* codes_t = malloc((size_t)kpad * T);
            float* scales_t = malloc((size_t)kpad * sizeof(float));   /* K/N swap: the transposed image has kpad columns */
            if (tr && codes_t && scales_t) {
                for (int k = 0; k < kpad; k++)
                    for (int n = 0; n < T; n++) tr[(size_t)n * kpad + k] = bt[(size_t)k * T + n];
                q4nx_pack_B_int8(tr, T, kpad, codes_t, kpad, scales_t);
                int differ = 0;
                for (int i = 0; i < kpad * T; i++) if (codes_t[i] != codes[i]) differ++;
                check(differ > kpad * T / 4, "head tile control: the transposed scratch is a different image");
                free(tr); free(codes_t); free(scales_t);
            }
            printf("  head tile: kpad %d x T %d, %d codes, %d column scales, layout pinned\n", kpad, T, kpad * T, T);
        }
        free(bt); free(codes); free(scales);
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


/* The A side of the same pad (engine#845 review).  The kernel is launched at the padded K while the
 * forward's activation rows hold only real_k floats, so without q4nx_pad_a_f32() every read of
 * A[m*K + k] for k >= real_k runs past the caller's buffer -- UB on the host path and, on the device
 * path, the 128 overrun floats feed the int8 activation scale.  The source here is allocated at
 * exactly M * real_k floats with a canary parked immediately after it, so a pad that reads past its
 * input (rather than memcpy-ing the real width) shows up as a canary change and fails the suite. */
static void activation_pad(void) {
    const int M = 3, real_k = 896, K = 1024;
    const size_t guard = 64;
    float* src = (float*)malloc(sizeof(float) * (size_t)M * real_k + sizeof(float) * guard);
    float* dst = (float*)malloc(sizeof(float) * (size_t)M * K);
    check(src && dst, "activation pad: buffers allocated");
    if (!src || !dst) { free(src); free(dst); return; }
    for (int i = 0; i < M * real_k; i++) src[i] = (float)(i % 251) * 0.001f - 0.1f;
    for (size_t i = 0; i < guard; i++) src[(size_t)M * real_k + i] = 1234.5f;   /* canary */
    for (int i = 0; i < M * K; i++) dst[i] = -7.0f;                            /* poison */

    q4nx_pad_a_f32(src, M, real_k, K, dst);

    int bad_real = 0, bad_tail = 0, bad_canary = 0;
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < real_k; k++)
            if (dst[(size_t)m * K + k] != src[(size_t)m * real_k + k]) bad_real++;
        for (int k = real_k; k < K; k++)
            if (dst[(size_t)m * K + k] != 0.0f) bad_tail++;
    }
    for (size_t i = 0; i < guard; i++)
        if (src[(size_t)M * real_k + i] != 1234.5f) bad_canary++;
    check_int(bad_real, 0, "activation pad: every real column is copied verbatim");
    check_int(bad_tail, 0, "activation pad: the 128 pad columns of every row are zero");
    check_int(bad_canary, 0, "activation pad: nothing is read past the source buffer");

    /* the two cases that must stay the identity: an on-grid model (real_k == K) and an MLA design
     * that never sets real_k at all (0 -> treat as K) */
    for (int i = 0; i < K; i++) dst[i] = -7.0f;
    q4nx_pad_a_f32(src, 1, K, K, dst);
    check_int(dst[K - 1] == src[K - 1] && dst[0] == src[0] ? 0 : 1, 0,
              "activation pad: real_k == K is the identity");
    for (int i = 0; i < K; i++) dst[i] = -7.0f;
    q4nx_pad_a_f32(src, 1, 0, K, dst);
    check_int(dst[K - 1] == src[K - 1] ? 0 : 1, 0, "activation pad: real_k 0 (MLA) copies K");
    free(src); free(dst);
}


static void test_head_image(void) {
    /* THE HEAD IMAGE ROUND TRIP (engine#1716, the cache's correctness bar). The image a serve copies into the weight BO
     * must be byte-identical to what the packer produces for the same tile, and a file that cannot prove it is the same
     * image must be REFUSED rather than copied -- a stale or damaged image copied into a BO answers wrongly with no
     * message at all, which is the failure this whole contract exists to make impossible. */
    {
        const int kd = 96, nd = 64;                 /* the same head-tile shape the layout contract above uses */
        float* bt = malloc((size_t)kd * nd * sizeof(float));
        int8_t* codes = malloc((size_t)kd * nd);
        int8_t* back = malloc((size_t)kd * nd);
        float* scales = malloc((size_t)nd * sizeof(float));
        float* sc_back = malloc((size_t)nd * sizeof(float));
        check(bt && codes && back && scales && sc_back, "head image: the fixture allocates");
        if (bt && codes && back && scales && sc_back) {
            for (int k = 0; k < kd; k++)
                for (int n = 0; n < nd; n++) bt[(size_t)k * nd + n] = (float)((k * 29 + n * 11) % 101) * 0.013f - 0.6f;
            q4nx_pack_B_int8(bt, kd, nd, codes, nd, scales);

            const char* path = "head_image_roundtrip.bin";
            remove(path);
            check(q4nx_head_image_write(path, codes, scales, kd, nd) == 1, "head image: the image is written");
            Q4nxHeadImageInfo info;
            char why[256];
            check(q4nx_head_image_read(path, back, kd * nd, sc_back, nd, &info, why, (int)sizeof why) == 1,
                  "head image: it reads back");
            check_int((int)info.kd, kd, "head image: the geometry in the header is the geometry written");
            check_int((int)info.nd, nd, "head image: and so is the column count");
            /* THE BAR: byte-identical to a fresh pack, not merely close */
            check(memcmp(codes, back, (size_t)kd * nd) == 0, "head image: the codes are byte-identical to a fresh pack");
            check(memcmp(scales, sc_back, (size_t)nd * sizeof(float)) == 0,
                  "head image: and so are the scales");

            /* a caller that rebuilt the image itself can compare hashes rather than bytes */
            check(q4nx_head_image_hash(codes, scales, kd, nd) == info.hash,
                  "head image: the file's hash is the hash of the image the caller would build");

            /* (a) a corrupted byte must refuse */
            {
                FILE* f = fopen(path, "r+b");
                if (f) {
                    /* the payload starts after the 24-byte header this writer uses */
                    fseek(f, 24 + 5, SEEK_SET);
                    int c = fgetc(f);
                    fseek(f, 24 + 5, SEEK_SET);
                    fputc((c ^ 0x5a) & 0xff, f);
                    fclose(f);
                }
                why[0] = 0;
                check(q4nx_head_image_read(path, back, kd * nd, sc_back, nd, &info, why, (int)sizeof why) == 0,
                      "head image: a single corrupted payload byte is REFUSED, not copied");
                check(why[0] != 0 && strstr(why, "hash") != NULL, "... and the refusal says the hash did not reproduce");
            }
            /* (b) the wrong geometry must refuse even when the file is intact */
            {
                check(q4nx_head_image_write(path, codes, scales, kd, nd) == 1, "head image: rewritten intact");
                why[0] = 0;
                check(q4nx_head_image_read(path, back, kd * nd, sc_back, nd - 1, &info, why, (int)sizeof why) == 0,
                      "head image: a container with a different column count refuses it");
            }
            /* (c) absence is not an error at this layer: it is a refusal with a reason the caller can treat as "pack" */
            {
                remove(path);
                why[0] = 0;
                check(q4nx_head_image_read(path, back, kd * nd, sc_back, nd, &info, why, (int)sizeof why) == 0,
                      "head image: no image reads as a refusal");
                check(why[0] != 0 && strstr(why, "not there") != NULL, "... whose reason is that it is not there");
            }
            /* (d) a truncated file refuses rather than returning a partial image */
            {
                check(q4nx_head_image_write(path, codes, scales, kd, nd) == 1, "head image: written for the truncation arm");
                FILE* f = fopen(path, "r+b");
                if (f) { fseek(f, 0, SEEK_END); const long sz = ftell(f); fclose(f); if (sz > 40) truncate_file(path, sz - 40); }
                why[0] = 0;
                check(q4nx_head_image_read(path, back, kd * nd, sc_back, nd, &info, why, (int)sizeof why) == 0,
                      "head image: a truncated image is refused");
                remove(path);
            }
        }
        free(bt); free(codes); free(back); free(scales); free(sc_back);
    }
}

int main(void) {
    test_head_image();
    oga_geometry();
    on_grid_geometry();
    activation_pad();
    printf("%s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 1 : 0;
}
