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
#include <sys/types.h>
#include <sys/wait.h>
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
                /* Per-column spread is load-bearing: the plain ramp reaches -0.5 in every column, so every column's
                 * absmax is 0.5 and all T scales are equal (measured: 1 distinct scale of 64) -- an arm comparing two
                 * packers on that fixture cannot see a per-column mistake. Scaling by (1 + 0.05n) gives T distinct
                 * absmax values (measured: 64 of 64), and the check below keeps the fixture from going flat again. */
                for (int n = 0; n < T; n++)
                    bt[(size_t)k * T + n] = ((float)((k * 13 + n * 7) % 97) * 0.01f - 0.5f) * (1.0f + 0.05f * (float)n);
            {
                /* the fixture must exercise per-column arithmetic, or every arm below compares on a flat input */
                float seen[64];
                int distinct = 0;
                for (int n = 0; n < T; n++) {
                    float amax = 0.0f;
                    for (int k = 0; k < kpad; k++) {
                        const float a = fabsf(bt[(size_t)k * T + n]);
                        if (a > amax) amax = a;
                    }
                    int dup = 0;
                    for (int i = 0; i < distinct; i++) if (seen[i] == amax) dup = 1;
                    if (!dup) seen[distinct++] = amax;
                }
                check_int(distinct, T, "head tile: the fixture has T distinct column absmax (not a flat input)");
            }
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
            /* The reference has to use the packer's own arithmetic -- absmax with the 1e-12 floor, then
             * roundf(v * (127/amax)) -- and not v/scales[n]: once the columns differ the two disagree on 26 of 6144
             * codes, because 127/amax and 1/(amax/127) are different floats. */
            float tis[64];
            for (int n = 0; n < T; n++) {
                float amax = 0.0f;
                for (int k = 0; k < kpad; k++) {
                    const float a = fabsf(bt[(size_t)k * T + n]);
                    if (a > amax) amax = a;
                }
                if (amax < 1e-12f) amax = 1.0f;
                tis[n] = 127.0f / amax;
            }
            int bad_code = 0;
            for (int k = 0; k < kpad; k++)
                for (int n = 0; n < T; n++) {
                    int want = (int)roundf(bt[(size_t)k * T + n] * tis[n]);
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
            /* The forward does NOT pack the head through q4nx_pack_B_int8: it goes
             * lctx.packB_into -> packB_blocked -> q4nx_pack_B_int8_blocked, a separate row-major loop in
             * q4nx_pack_blocked.cpp that never calls the function pinned above. A head-image cache (the fix
             * for the 10-17 s rebuild, engine#1716) stands in for THAT call, so the layout arms above are
             * only its contract if the two implementations agree byte for byte -- and nothing asserted it.
             * Measured on main (engine#1716 comment, 2026-11-10): identical codes, scales and mean at this
             * geometry; this arm is what keeps that true, since a change to either loop would otherwise
             * break the cache contract silently. */
            {
                int8_t* oc = calloc((size_t)kpad * T, 1);
                float* osc = calloc((size_t)T, sizeof(float));
                double osum = 0.0;
                if (oc && osc) {
                    q4nx_pack_B_int8_blocked(oc, bt, kpad, T, kpad, T, 0, osc, &osum);
                    int code_diff = 0;
                    for (int i = 0; i < kpad * T; i++) if (oc[i] != codes[i]) code_diff++;
                    int scale_diff = 0;
                    for (int j = 0; j < T; j++) if (osc[j] != scales[j]) scale_diff++;
                    check_int(code_diff, 0, "head tile: the blocked packer (the one the forward calls) is byte-identical");
                    check_int(scale_diff, 0, "head tile: and returns the same column scales");
                    check(fabs(ssum - osum) <= 1e-9 * (fabs(osum) + 1.0),
                          "head tile: and the same mean column scale");
                    free(oc); free(osc);
                }
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
            /* (e) THE KEY: each of the four facts a head image depends on must change it. kpad, T and blocked come from the
             * kernel designs and head_hash from the container, so any one of them differing means a DIFFERENT image -- and
             * keying on fewer would serve one kernel set's image to another while the correlation still looked plausible. */
            {
                Q4nxHeadImageKey base = {96, 64, 1, q4nx_head_image_hash(codes, scales, kd, nd)};
                const uint64_t d0 = q4nx_head_image_key_digest(&base);
                Q4nxHeadImageKey k[4];
                for (int i = 0; i < 4; i++) k[i] = base;
                k[0].kpad++;              k[1].t++;           k[2].blocked = 0;     k[3].head_hash ^= 1ull;
                int changed = 0;
                for (int i = 0; i < 4; i++)
                    if (q4nx_head_image_key_digest(&k[i]) != d0) changed++;
                check_int(changed, 4, "head image key: kpad, T, blocked and the head hash each change the digest");
                /* and the name is stable for the same key, so a cache written by one run is found by the next */
                char n1[64], n2[64], n3[64];
                check(q4nx_head_image_key_name(&base, n1, (int)sizeof n1) == 1, "head image key: it names a file");
                check(q4nx_head_image_key_name(&base, n2, (int)sizeof n2) == 1 && strcmp(n1, n2) == 0,
                      "... the same name for the same key");
                check(q4nx_head_image_key_name(&k[1], n3, (int)sizeof n3) == 1 && strcmp(n1, n3) != 0,
                      "... and a different name when T changes, so two kernel sets cannot share one cached image");
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
            /* (e) TWO WRITERS, ONE IMAGE. Two serves that start cold on the same pack write the same file at the same
             * time. Each must end with a successful write and the path must read back intact after every one of them:
             * with one shared temp name the second open truncated the first writer's file and one rename found nothing. */
            {
                const int bk = 2048, bn = 2048;     /* 4 MiB, so a write is long enough for the two to overlap */
                int8_t* big = malloc((size_t)bk * bn);
                int8_t* big_back = malloc((size_t)bk * bn);
                float* bsc = malloc((size_t)bn * sizeof(float));
                float* bsc_back = malloc((size_t)bn * sizeof(float));
                check(big && big_back && bsc && bsc_back, "head image: the two-writer fixture allocates");
                if (big && big_back && bsc && bsc_back) {
                    for (size_t i = 0; i < (size_t)bk * bn; i++) big[i] = (int8_t)((i * 31u + 7u) % 251u - 125);
                    for (int n = 0; n < bn; n++) bsc[n] = 0.001f * (float)(n + 1);
                    const char* shared = "head_image_two_writers.bin";
                    remove(shared);
                    const int rounds = 40;
                    fflush(stdout);
                    const pid_t child = fork();
                    if (child == 0) {
                        int bad = 0;
                        for (int i = 0; i < rounds; i++) bad += q4nx_head_image_write(shared, big, bsc, bk, bn) != 1;
                        _exit(bad > 255 ? 255 : bad);
                    }
                    int bad_w = 0, bad_r = 0;
                    for (int i = 0; i < rounds; i++) {
                        bad_w += q4nx_head_image_write(shared, big, bsc, bk, bn) != 1;
                        bad_r += q4nx_head_image_read(shared, big_back, bk * bn, bsc_back, bn, &info, why, (int)sizeof why) != 1;
                    }
                    int st = 0, bad_c = -1;
                    if (child > 0 && waitpid(child, &st, 0) == child && WIFEXITED(st)) bad_c = WEXITSTATUS(st);
                    check_int(bad_w, 0, "head image: with a second writer on the same path, every write of ours lands");
                    check_int(bad_c, 0, "... and so does every write of the other process");
                    check_int(bad_r, 0, "... and the path reads back intact after each of them");
                    check(q4nx_head_image_read(shared, big_back, bk * bn, bsc_back, bn, &info, why, (int)sizeof why) == 1 &&
                              memcmp(big, big_back, (size_t)bk * bn) == 0,
                          "... and what is left is the image both wrote");
                    remove(shared);
                }
                free(big); free(big_back); free(bsc); free(bsc_back);
            }
        }
        free(bt); free(codes); free(back); free(scales); free(sc_back);
    }
}

/* ── THE HEAD TILE CACHE (engine#1716): what a serve calls instead of packing ─────────────────────────────────────────
 * The issue is "the packed lm-head is rebuilt on every serve". The arms below are that sentence and its two dangerous
 * neighbours, with a COUNTER on the gather so they are about work done and not about a log line:
 *   - a second pass over the same tiles gathers only the sample, and returns the bytes a fresh pack returns;
 *   - a changed key (layout, head bytes) packs again rather than borrowing the other image;
 *   - a damaged file, and an intact file written for OTHER weights, are both replaced rather than used. */
typedef struct {
    int nv, k;
    int variant;     /* 0: the fixture; 1: row 3 negated (same column scales, different codes); 2: everything x1.25 */
    long calls;      /* columns gathered */
} HeadFixture;

static void head_fixture_col(void* ctx, int n, float* out) {
    HeadFixture* f = (HeadFixture*)ctx;
    f->calls++;
    for (int k = 0; k < f->k; k++) {
        float v = 0.0f;
        if (n < f->nv) {
            /* a distinct absmax per column (the lesson of #1778: equal column scales hide a per-column mistake) */
            v = ((float)((k * 37 + n * 11) % 211) - 105.0f) * 0.004f * (1.0f + 0.01f * (float)(n % 97));
            if (f->variant == 1 && k == 3) v = -v;
            if (f->variant == 2) v *= 1.25f;
        }
        out[k] = v;
    }
}

/* What I8Ctx::packB_into() writes for one tile: zero the image, gather the tile, pack it. */
static void head_reference(HeadFixture* f, int n0, int ncols, int k, int t, int kd, int nd, int blocked, int8_t* codes,
                           float* scales) {
    float* bt = calloc((size_t)k * t, sizeof(float));
    float* col = malloc((size_t)k * sizeof(float));
    const long keep = f->calls;
    for (int j = 0; j < ncols; j++) {
        head_fixture_col(f, n0 + j, col);
        for (int i = 0; i < k; i++) bt[(size_t)i * t + j] = col[i];
    }
    f->calls = keep;
    memset(codes, 0, (size_t)kd * nd);
    for (int j = 0; j < nd; j++) scales[j] = 0.0f;
    double ssum = 0.0;
    q4nx_pack_B_int8_blocked(codes, bt, k, t, kd, nd, blocked, scales, &ssum);
    free(bt); free(col);
}

static void head_tile_path(const char* dir, const Q4nxHeadImageKey* key, int tile, char* out, size_t out_n) {
    char name[64];
    q4nx_head_image_key_name(key, name, (int)sizeof name);
    char* dot = strrchr(name, '.');
    if (dot) *dot = 0;
    snprintf(out, out_n, "%s/%s-t%03d.q4hi", dir, name, tile);
}

static void test_head_tile_cache(void) {
    const int k = 128, t = 128, kd = 128, nd = 128, nv = 300, ntiles = 3;   /* the last tile holds 44 real columns */
    char dir[] = "head_tile_cache.XXXXXX";
    check(mkdtemp(dir) != NULL, "head tile cache: a scratch cache directory");
    int8_t* codes = malloc((size_t)kd * nd);
    int8_t* ref = malloc((size_t)kd * nd);
    float* scales = malloc((size_t)nd * sizeof(float));
    float* sref = malloc((size_t)nd * sizeof(float));
    check(codes && ref && scales && sref, "head tile cache: the fixture allocates");
    if (!codes || !ref || !scales || !sref) return;
    HeadFixture fx = {nv, k, 0, 0};
    char paths[8][512];
    int npaths = 0;

    for (int blocked = 0; blocked <= 1; blocked++) {
        Q4nxHeadImageKey key = {(uint32_t)k, (uint32_t)t, (uint32_t)blocked, 0x1716c0ffeeull};
        int cold_ok = 0, cold_same = 0, warm_hit = 0, warm_same = 0;
        long cold_calls = 0, warm_calls = 0;
        for (int pass = 0; pass < 2; pass++) {
            for (int tile = 0; tile < ntiles; tile++) {
                const int n0 = tile * t, ncols = (nv - n0 < t) ? nv - n0 : t;
                head_reference(&fx, n0, ncols, k, t, kd, nd, blocked, ref, sref);
                Q4nxHeadTileResult r;
                const long before = fx.calls;
                memset(codes, 0x55, (size_t)kd * nd);   /* the BO is not ours to assume zero on a hit */
                const int rc = q4nx_head_tile_image(dir, &key, tile, n0, ncols, k, t, kd, nd, head_fixture_col, &fx, codes,
                                                    scales, &r);
                const int same = memcmp(codes, ref, (size_t)kd * nd) == 0 && memcmp(scales, sref, (size_t)t * sizeof(float)) == 0;
                if (pass == 0) {
                    cold_ok += rc == 1 && r.hit == 0 && r.wrote == 1 && r.refused == 0;
                    cold_same += same;
                    cold_calls += fx.calls - before;
                    if (npaths < 8) head_tile_path(dir, &key, tile, paths[npaths++], sizeof paths[0]);
                } else {
                    warm_hit += rc == 1 && r.hit == 1 && r.wrote == 0 && r.refused == 0 && r.sampled > 0;
                    warm_same += same;
                    warm_calls += fx.calls - before;
                }
            }
        }
        check_int(cold_ok, ntiles, blocked ? "head tile cache (b8): a first pass packs and writes every tile"
                                           : "head tile cache (rm): a first pass packs and writes every tile");
        check_int(cold_same, ntiles, "... and its bytes are packB_into()'s bytes (codes and scales)");
        check_int((int)cold_calls, nv, "... having gathered every vocabulary column once");
        check_int(warm_hit, ntiles, "head tile cache: a SECOND pass takes every tile from the cache");
        check_int(warm_same, ntiles, "... byte-identical to a fresh pack, over a buffer that was not zero");
        check(warm_calls > 0 && warm_calls <= (long)ntiles * Q4NX_HEAD_TILE_SAMPLE,
              "... gathering only the verification sample, not the vocabulary (the head is NOT rebuilt)");
    }

    /* a different key must not borrow the image: the layout and the head bytes each name a different file */
    {
        Q4nxHeadImageKey other = {(uint32_t)k, (uint32_t)t, 0, 0x1716c0ffeeull ^ 1ull};
        Q4nxHeadTileResult r;
        const long before = fx.calls;
        const int rc = q4nx_head_tile_image(dir, &other, 0, 0, t, k, t, kd, nd, head_fixture_col, &fx, codes, scales, &r);
        check(rc == 1 && r.hit == 0 && r.wrote == 1, "head tile cache: a different head hash packs, it does not hit");
        check_int((int)(fx.calls - before), t, "... gathering the whole tile");
        if (npaths < 8) head_tile_path(dir, &other, 0, paths[npaths++], sizeof paths[0]);
    }

    Q4nxHeadImageKey key = {(uint32_t)k, (uint32_t)t, 1, 0x1716c0ffeeull};
    char path[512];
    head_tile_path(dir, &key, 0, path, sizeof path);

    /* (a) a damaged file is replaced, with the reason, and the next serve hits again */
    {
        FILE* f = fopen(path, "r+b");
        check(f != NULL, "head tile cache: the cached file is where the key says it is");
        if (f) { fseek(f, 24 + 777, SEEK_SET); int c = fgetc(f); fseek(f, 24 + 777, SEEK_SET); fputc((c ^ 0x21) & 0xff, f); fclose(f); }
        head_reference(&fx, 0, t, k, t, kd, nd, 1, ref, sref);
        Q4nxHeadTileResult r;
        int rc = q4nx_head_tile_image(dir, &key, 0, 0, t, k, t, kd, nd, head_fixture_col, &fx, codes, scales, &r);
        check(rc == 1 && r.hit == 0 && r.refused == 1 && r.wrote == 1, "head tile cache: a damaged image is REPLACED, not used");
        check(strstr(r.why, "hash") != NULL, "... and the reason names the hash");
        check(memcmp(codes, ref, (size_t)kd * nd) == 0, "... and the tile served is the freshly packed one");
        rc = q4nx_head_tile_image(dir, &key, 0, 0, t, k, t, kd, nd, head_fixture_col, &fx, codes, scales, &r);
        check(rc == 1 && r.hit == 1, "... after which the cache hits again");
    }
    /* (b) THE ARM THE PAYLOAD HASH CANNOT PASS FOR US: an intact, self-consistent image written for OTHER weights under
     * the same name. Variant 1 keeps every column scale and changes codes; variant 2 changes the scales. Both must be
     * caught by the sample and replaced -- copying either into a weight BO would answer plausibly and wrongly. */
    for (int variant = 1; variant <= 2; variant++) {
        fx.variant = variant;
        head_reference(&fx, 0, t, k, t, kd, nd, 1, ref, sref);
        Q4nxHeadTileResult r;
        int rc = q4nx_head_tile_image(dir, &key, 0, 0, t, k, t, kd, nd, head_fixture_col, &fx, codes, scales, &r);
        check(rc == 1 && r.hit == 0 && r.refused == 1,
              variant == 1 ? "head tile cache: an intact image for OTHER weights (same scales, other codes) is not used"
                           : "head tile cache: an intact image for OTHER weights (other scales) is not used");
        check(strstr(r.why, variant == 1 ? "holds code" : "has scale") != NULL, "... and the reason names what disagreed");
        check(memcmp(codes, ref, (size_t)kd * nd) == 0 && memcmp(scales, sref, (size_t)t * sizeof(float)) == 0,
              "... and the tile served is packed from the weights actually held");
        rc = q4nx_head_tile_image(dir, &key, 0, 0, t, k, t, kd, nd, head_fixture_col, &fx, codes, scales, &r);
        check(rc == 1 && r.hit == 1, "... and the replacement is what the next serve hits");
    }
    fx.variant = 0;
    /* (c) the cache is never required: no directory, or one that cannot be written, packs and says it wrote nothing */
    {
        head_reference(&fx, 256, 44, k, t, kd, nd, 1, ref, sref);
        Q4nxHeadTileResult r;
        int rc = q4nx_head_tile_image(NULL, &key, 2, 256, 44, k, t, kd, nd, head_fixture_col, &fx, codes, scales, &r);
        check(rc == 1 && r.hit == 0 && r.wrote == 0 && memcmp(codes, ref, (size_t)kd * nd) == 0,
              "head tile cache: with no cache directory the tile packs as it always did");
        rc = q4nx_head_tile_image("head_tile_cache.no-such-dir/x", &key, 2, 256, 44, k, t, kd, nd, head_fixture_col, &fx,
                                  codes, scales, &r);
        check(rc == 1 && r.hit == 0 && r.wrote == 0 && r.refused == 0 && memcmp(codes, ref, (size_t)kd * nd) == 0,
              "head tile cache: an unwritable directory is not an error either");
        /* and a geometry the key does not describe is refused outright rather than packed under the wrong name */
        rc = q4nx_head_tile_image(dir, &key, 0, 0, t, k - 64, t, kd, nd, head_fixture_col, &fx, codes, scales, &r);
        check(rc == 0 && r.why[0] != 0, "head tile cache: a K that is not the key's kpad is refused");
    }
    /* (d) the source hash moves with the bytes and with the facts beside them */
    {
        unsigned char bytes[64];
        for (int i = 0; i < 64; i++) bytes[i] = (unsigned char)(i * 7);
        uint32_t extra[2] = {300, 128};
        const uint64_t h0 = q4nx_head_source_hash(bytes, sizeof bytes, extra, 2);
        bytes[40] ^= 1;
        const uint64_t h1 = q4nx_head_source_hash(bytes, sizeof bytes, extra, 2);
        bytes[40] ^= 1;
        extra[0] = 301;
        const uint64_t h2 = q4nx_head_source_hash(bytes, sizeof bytes, extra, 2);
        check(h0 != 0 && h1 != h0 && h2 != h0 && h1 != h2, "head source hash: one byte, or one fact beside the bytes, changes it");
        check(q4nx_head_source_hash(NULL, 0, NULL, 0) == 0, "head source hash: no data is 0, which names no cache entry");
    }

    for (int i = 0; i < npaths; i++) remove(paths[i]);
    check(rmdir(dir) == 0, "head tile cache: only the files the keys name were written (the directory empties)");
    free(codes); free(ref); free(scales); free(sref);
}

/* engine#1999 / #2001 review, closed here: the head path on a geometry that is not a whole 128-column tile grid.
 * `blocked_fits(128, 96)` is false, so the forward packs the head ROW-MAJOR and the key carries blocked = 0.
 * `test_head_tile_cache()` loops blocked 0/1 over an ALIGNED 128x128, so neither of the two things that matter here is
 * covered: (a) a key that claims the blocked layout for an unaligned geometry must be REFUSED -- blocked_index()
 * addresses whole tiles, so serving one would read past the image or read foreign bytes -- and (b) a row-major image
 * still caches: cold writes, warm hits, and the bytes are packB_into()'s. */
static void test_head_tile_unaligned(void) {
    const int k = 128, t = 96, kd = 128, nd = 96, nv = 90, ntiles = 1;   /* 96 columns: 3/4 of a tile */
    char dir[] = "head_tile_unaligned.XXXXXX";
    check(mkdtemp(dir) != NULL, "unaligned head: a scratch cache directory");
    int8_t* codes = malloc((size_t)kd * nd);
    int8_t* ref = malloc((size_t)kd * nd);
    float* scales = malloc((size_t)nd * sizeof(float));
    float* sref = malloc((size_t)nd * sizeof(float));
    check(codes && ref && scales && sref, "unaligned head: the fixture allocates");
    if (!codes || !ref || !scales || !sref) return;
    HeadFixture fx = {nv, k, 0, 0};

    /* (a) a blocked key on a geometry with no blocked layout is refused, by name, and writes nothing */
    {
        Q4nxHeadImageKey key = {(uint32_t)k, (uint32_t)t, 1, 0xa11a11a11ull};
        Q4nxHeadTileResult r;
        const int rc = q4nx_head_tile_image(dir, &key, 0, 0, nv, k, t, kd, nd, head_fixture_col, &fx, codes, scales, &r);
        check(rc != 1, "unaligned head: a blocked key on a 128x96 geometry is REFUSED, not served");
        check(strstr(r.why, "no blocked layout") != NULL, "... and the reason names the missing layout");
        check(r.wrote == 0, "... and nothing was written for it");
    }

    /* (b) the row-major image caches: cold writes, warm hits, both byte-identical to a fresh pack */
    {
        Q4nxHeadImageKey key = {(uint32_t)k, (uint32_t)t, 0, 0xa11a11a11ull};
        int cold_ok = 0, cold_same = 0, warm_hit = 0, warm_same = 0;
        Q4nxHeadTileResult r;
        for (int pass = 0; pass < 2; pass++) {
            for (int tile = 0; tile < ntiles; tile++) {
                const int n0 = tile * t, ncols = (nv - n0 < t) ? nv - n0 : t;
                head_reference(&fx, n0, ncols, k, t, kd, nd, /*blocked=*/0, ref, sref);
                memset(codes, 0x5a, (size_t)kd * nd);
                const int rc = q4nx_head_tile_image(dir, &key, tile, n0, ncols, k, t, kd, nd, head_fixture_col, &fx, codes,
                                                    scales, &r);
                const int same = memcmp(codes, ref, (size_t)kd * nd) == 0 &&
                                 memcmp(scales, sref, (size_t)t * sizeof(float)) == 0;
                if (pass == 0) {
                    cold_ok += rc == 1 && r.hit == 0 && r.wrote == 1 && r.refused == 0;
                    cold_same += same;
                } else {
                    warm_hit += rc == 1 && r.hit == 1 && r.wrote == 0 && r.refused == 0 && r.sampled > 0;
                    warm_same += same;
                }
            }
        }
        check_int(cold_ok, ntiles, "unaligned head: a first pass packs and writes the row-major tile");
        check_int(cold_same, ntiles, "... and its bytes are packB_into()'s (codes and scales)");
        check_int(warm_hit, ntiles, "unaligned head: a SECOND pass takes it from the cache");
        check_int(warm_same, ntiles, "... byte-identical to a fresh pack, over a buffer that was not zero");
    }
    free(codes); free(ref); free(scales); free(sref);
}

int main(void) {
    test_head_image();
    test_head_tile_cache();
    test_head_tile_unaligned();
    oga_geometry();
    on_grid_geometry();
    activation_pad();
    printf("%s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 1 : 0;
}
