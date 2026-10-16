// Debug harness: repeat one MLA-shaped FLASH_ATTN_EXT (+ the v_mla de-absorption matmul) on HRX0 and report
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
// where runs differ. Adapted from the earlier investigation's fa-race.cpp, which used a Qwen-shaped graph
// (Hq=16/Hkv=8, head 128) - GLM-4.7-Flash is MLA (deepseek2): 20 query heads over a single latent KV head, a
// 576-wide key (512 kv_lora_rank + 64 rope) and a 512-wide value, with the de-absorption done afterwards by
// `ggml_mul_mat(v_mla, reshape_4d(cur, 512, 1, 20, T))` (llama-graph.cpp:2442-2454).
//
// Usage: glm-mla-race <T> <cells> <visible> <reps> [seed] [device]
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <tuple>
#include <vector>

int main(int argc, char ** argv) {
    const int64_t kVK     = 576;   // latent key: kv_lora_rank 512 + rope 64
    const int64_t kVL     = 512;   // latent value = kv_lora_rank
    const int64_t kVHead  = 256;   // per-head value width after de-absorption
    const int64_t Hq      = 20;    // query heads
    const int64_t T       = argc > 1 ? atoll(argv[1]) : 190;
    const int64_t cells   = argc > 2 ? atoll(argv[2]) : 2048;
    const int64_t visible = argc > 3 ? atoll(argv[3]) : 810;
    const int     reps    = argc > 4 ? atoi(argv[4]) : 250;
    const int     seed    = argc > 5 ? atoi(argv[5]) : 1;

    std::mt19937                    rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<ggml_fp16_t> mask(cells * T);
    for (int64_t r = 0; r < T; ++r) {
        const int64_t qp = visible - T + r;
        for (int64_t c = 0; c < cells; ++c) mask[r * cells + c] = ggml_fp32_to_fp16(c <= qp ? 0.0f : -INFINITY);
    }
    std::vector<float> q(kVK * Hq * T);
    for (float & x : q) x = 2.0f * nd(rng);
    std::vector<ggml_fp16_t> k(kVK * cells), v(kVL * cells);
    for (int64_t c = 0; c < cells; ++c) {
        for (int64_t i = 0; i < kVK; ++i) k[c * kVK + i] = ggml_fp32_to_fp16(c < visible ? nd(rng) : 0.0f);
        for (int64_t i = 0; i < kVL; ++i) v[c * kVL + i] = ggml_fp32_to_fp16(c < visible ? nd(rng) : 0.0f);
    }
    std::vector<float> vmla(kVL * kVHead);
    for (float & x : vmla) x = nd(rng);

    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_name(argc > 6 ? argv[6] : "HRX0", nullptr);
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }

    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * tq   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kVK, Hq, T);
    ggml_tensor * tk   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kVK, 1, cells);
    ggml_tensor * tv   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kVL, 1, cells);
    ggml_tensor * tm   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cells, T);
    ggml_tensor * wv   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kVL, kVHead);

    ggml_tensor * kv = ggml_view_3d(ctx, tk, kVK, cells, 1, tk->nb[2], tk->nb[1], 0);
    ggml_tensor * vv = ggml_view_3d(ctx, tv, kVL, cells, 1, tv->nb[2], tv->nb[1], 0);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, ggml_permute(ctx, tq, 0, 2, 1, 3), kv, vv, tm,
                                          1.0f / std::sqrt((float) kVK), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    // MLA de-absorption, exactly as llama-graph.cpp does it for deepseek2/GLM
    ggml_tensor * cur = ggml_reshape_4d(ctx, fa, kVL, 1, Hq, T);
    ggml_tensor * abs = ggml_mul_mat(ctx, wv, cur);
    ggml_tensor * out = ggml_reshape_2d(ctx, abs, kVHead * Hq, T);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (buf == nullptr) { fprintf(stderr, "alloc failed\n"); return 1; }

    ggml_backend_tensor_set(tq, q.data(), 0, ggml_nbytes(tq));
    ggml_backend_tensor_set(tk, k.data(), 0, ggml_nbytes(tk));
    ggml_backend_tensor_set(tv, v.data(), 0, ggml_nbytes(tv));
    ggml_backend_tensor_set(tm, mask.data(), 0, ggml_nbytes(tm));
    ggml_backend_tensor_set(wv, vmla.data(), 0, ggml_nbytes(wv));

    std::vector<float> ref, o(ggml_nelements(out));
    int                bad_runs = 0;
    for (int r = 0; r < reps; ++r) {
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
        ggml_backend_tensor_get(out, o.data(), 0, o.size() * sizeof(float));
        if (r == 0) {
            ref = o;
            double cs = 0;
            for (float x : o) cs += x;
            printf("checksum %.6f\n", cs);
            continue;
        }
        std::map<std::tuple<int64_t, int64_t, int64_t>, int> groups;
        int n = 0, nn = 0;
        for (size_t i = 0; i < o.size(); ++i) {
            if (memcmp(&o[i], &ref[i], sizeof(float)) == 0) continue;
            ++n;
            if (!std::isfinite(o[i])) ++nn;
            const int64_t tok = i / (kVHead * Hq);
            const int64_t ch  = i % kVHead;
            const int64_t hd  = (i / kVHead) % Hq;
            ++groups[{tok, hd, ch / 16}];
        }
        if (n != 0) {
            ++bad_runs;
            printf("run %d: %d differ (%d nonfinite):", r, n, nn);
            int shown = 0;
            for (const auto & g : groups) {
                if (shown++ == 3) break;
                printf(" [tok %lld head %lld ch %lld..%lld: %d]", (long long) std::get<0>(g.first),
                       (long long) std::get<1>(g.first), (long long) std::get<2>(g.first) * 16,
                       (long long) std::get<2>(g.first) * 16 + 15, g.second);
            }
            printf("\n");
        }
    }
    printf("T=%lld cells=%lld visible=%lld reps=%d bad_runs=%d\n", (long long) T, (long long) cells,
           (long long) visible, reps, bad_runs);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return bad_runs != 0 ? 2 : 0;
}
