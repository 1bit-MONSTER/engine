# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The gated delta-net prefill kernel, the target a KernelForge campaign optimizes.

A standalone copy of gated_delta_net_prefill_cuda from the engine's ROCmFPX pin
(ggml/src/ggml-cuda/gated_delta_net.cu, ROCmFPX#1, #4 and #5), for gfx1151 (Strix Halo). One call
runs a whole micro-batch of tokens through the delta rule for every value head:

    for each token t, head h, state column c (S is kept transposed: S[h][c][i]):
        g     = exp(g_log[t, h])
        kv    = sum_i S[h][c][i] * k[t, hk][i]            hk = h % Hk (q/k heads are shared)
        delta = (v[t, h, c] - g * kv) * beta[t, h]
        S[h][c][i] = g * S[h][c][i] + k[t, hk][i] * delta
        out[t, h, c] = scale * sum_i S[h][c][i] * q[t, hk][i]

and returns the output, the final state, and (K > 1) the state after each of the last K tokens:
slot 0 is the final state, slot s the state s tokens back (speculative decoding's rollback).
"""
from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load

S_V = 128
os.environ.setdefault("PYTORCH_ROCM_ARCH", os.environ.get("GPU_TARGET", "gfx1151"))

_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <cmath>
#include <type_traits>

// XOR-reduce across the 4 lanes of a quad and broadcast. __shfl_xor lowers to ds_bpermute on RDNA,
// which issues on the *LDS pipe* -- the same pipe the k_s/q_s reads contend for, and it needs an
// lgkmcnt drain that also retires those reads, dragging them onto the token recurrence. DPP
// quad_perm is a VALU modifier instead: no LDS traffic and no wait.
template <int WIDTH>
__device__ __forceinline__ float group_sum(float x) {
    static_assert(WIDTH == 4 || WIDTH == 8, "DPP reduction covers SPLIT 4 and 8");
    // row_half_mirror is the XOR-7 butterfly over each group of 8 lanes; composing it with the two
    // quad_perms covers {1,2,3,4,5,6,7}, i.e. a full 8-lane reduce-and-broadcast in three DPP adds.
    if constexpr (WIDTH == 8)
        x += __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(
                 0, __builtin_bit_cast(int, x), 0x141, 0xF, 0xF, true));
    // quad_perm[1,0,3,2] = XOR 1, quad_perm[2,3,0,1] = XOR 2.
    x += __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(
             0, __builtin_bit_cast(int, x), 0xB1, 0xF, 0xF, true));
    x += __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(
             0, __builtin_bit_cast(int, x), 0x4E, 0xF, 0xF, true));
    return x;
}

// Lane -> state-row map. The blocked map (row = part * ROWS) puts the SPLIT lanes of a column on
// LDS dwords 0, 32, 64, 96 of the same token row, i.e. all four on bank 0: a 4-way conflict on
// every k_s/q_s read. Interleaving in groups of 4 instead --
//     row(r) = (r >> 2) * (SPLIT * 4) + part * 4 + (r & 3)
// -- is still a bijection onto 0..S_v-1, still gives each lane four *consecutive* dwords (so the
// ds_load_b128 survives, base part*16, immediates 64 * (r >> 2)), but spreads the four lanes over
// banks 4 * part .. 4 * part + 3, which is conflict-free. Row padding cannot fix the blocked map:
// part * ROWS is a multiple of 32 for any row stride. The state tensor layout is unchanged --
// s[r] simply denotes a different row -- so only the partial-sum partition of the reduction moves.
#define GDN_ROW(r) ((((r) >> 2) * (SPLIT * 4)) + part * 4 + ((r) & 3))

// SPLIT lanes per state column *group*, CPL columns per lane, COLS columns per block, CHUNK-token
// chunks of k/q/v/g/beta in shared memory.
//
// Why CPL > 1: k_s/q_s do not depend on the column, so every column in the block re-reads the same
// dwords. A work unit of (column, token) costs 2 * S_v LDS dwords no matter how SPLIT is chosen --
// the only way to cut LDS traffic is to serve several columns from one set of reads. CPL = 2 halves
// the dwords per column-token; raising SPLIT to 8 at the same time keeps ROWS (and so the register
// footprint) and the total wave count -- H * (S_v / CPL) * SPLIT lanes -- exactly where they were.
template <int S_v, int SPLIT, int CPL, int COLS, int CHUNK, bool KEEP>
__global__ void __launch_bounds__((COLS / CPL) * SPLIT)
gdn_prefill(const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
            const float * __restrict__ g, const float * __restrict__ beta, const float * __restrict__ state_in,
            float * __restrict__ out, float * __restrict__ state_out,
            int H, int Hk, int n_tokens, int K, float scale) {
    constexpr int ROWS     = S_v / SPLIT;
    constexpr int NTHREADS = (COLS / CPL) * SPLIT;

    __shared__ float k_s[CHUNK][S_v];
    __shared__ float q_s[CHUNK][S_v];
    __shared__ float v_s[CHUNK][COLS];
    // (g, beta) for a token in one 16 B slot: one ds_load_b128 instead of two ds_load_b32.
    __shared__ float4 gb_s[CHUNK];

    const int h       = blockIdx.x;
    const int hk      = h % Hk;
    const int tid     = threadIdx.x;
    const int part    = tid % SPLIT;
    const int cloc    = (tid / SPLIT) * CPL;   // first of this lane's CPL columns, block-local
    const int col0    = blockIdx.z * COLS;
    const int col     = col0 + cloc;
    static_assert(ROWS % 4 == 0, "row map needs ROWS a multiple of 4");

    const float * s_in  = state_in + (size_t) h * S_v * S_v;
    float *       s_out = state_out + (size_t) h * S_v * S_v;
    const size_t  slot_stride = (size_t) H * S_v * S_v;

    float s[CPL][ROWS];
#pragma unroll
    for (int cc = 0; cc < CPL; cc++)
#pragma unroll
        for (int r = 0; r < ROWS; r++) s[cc][r] = s_in[(col + cc) * S_v + GDN_ROW(r)];

    for (int t0 = 0; t0 < n_tokens; t0 += CHUNK) {
        const int nt = min(CHUNK, n_tokens - t0);
        __syncthreads();
        // float4 staging: S_v and COLS are multiples of 4, so a vector never straddles a token row
        // and every base is 16 B aligned (torch bases are >= 256 B aligned).
        for (int idx = tid * 4; idx < nt * S_v; idx += NTHREADS * 4) {
            const int tt = idx / S_v, i = idx % S_v;
            const size_t b = ((size_t) (t0 + tt) * Hk + hk) * S_v + i;
            *reinterpret_cast<float4 *>(&k_s[tt][i]) = *reinterpret_cast<const float4 *>(k + b);
            *reinterpret_cast<float4 *>(&q_s[tt][i]) = *reinterpret_cast<const float4 *>(q + b);
        }
        for (int idx = tid * 4; idx < nt * COLS; idx += NTHREADS * 4) {
            const int tt = idx / COLS, c = idx % COLS;
            *reinterpret_cast<float4 *>(&v_s[tt][c]) =
                *reinterpret_cast<const float4 *>(v + ((size_t) (t0 + tt) * H + h) * S_v + col0 + c);
        }
        static_assert(NTHREADS >= CHUNK, "g/beta staging needs one thread per chunk token");
        if (tid < nt) {
            const float gv = expf(g[(size_t) (t0 + tid) * H + h]);
            gb_s[tid] = make_float4(gv, beta[(size_t) (t0 + tid) * H + h], 1.0f / gv, 0.0f);
        }
        __syncthreads();

        // The snapshot block fires on only the last K of n_tokens tokens, yet its `slot < K` guard
        // costs scalar-compare/branch slots -- and breaks up the VOPD pairs around it -- on every
        // token of every chunk. The predicate is wave-uniform and chunk-monotone, so hoist it to
        // chunk granularity: a chunk that cannot reach the snapshot window runs the byte-identical
        // K = 1 token loop. The per-token `slot < K` test stays inside the snapshotting copy, which
        // is what keeps partial chunks and K > CHUNK correct.
        auto token_loop = [&](auto snap_tag) {
            constexpr bool SNAP = decltype(snap_tag)::value;
        for (int tt = 0; tt < nt; tt++) {
            const float4 gb = gb_s[tt];
            const float beta_val = gb.y;
            const float g_val = gb.x;

            float kr[ROWS];
            float kv[CPL][4] = {};
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                kr[r] = k_s[tt][GDN_ROW(r)];
#pragma unroll
                for (int cc = 0; cc < CPL; cc++) kv[cc][r % 4] += s[cc][r] * kr[r];
            }
            float delta[CPL];
#pragma unroll
            for (int cc = 0; cc < CPL; cc++) {
                const float khat = group_sum<SPLIT>((kv[cc][0] + kv[cc][1]) + (kv[cc][2] + kv[cc][3]));
                delta[cc] = (v_s[tt][cloc + cc] - g_val * khat) * beta_val;
            }

            float at[CPL][4] = {};
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const float qr = q_s[tt][GDN_ROW(r)];
#pragma unroll
                for (int cc = 0; cc < CPL; cc++) {
                    s[cc][r]       = g_val * s[cc][r] + kr[r] * delta[cc];
                    at[cc][r % 4] += s[cc][r] * qr;
                }
            }
            const float gscale = scale;
            // One exec-masked region and one naturally aligned float2: col is a multiple of CPL = 2,
            // so the wave's active lanes write whole 32 B spans instead of two half-empty ones.
            float2 o;
            o.x = group_sum<SPLIT>((at[0][0] + at[0][1]) + (at[0][2] + at[0][3])) * gscale;
            o.y = group_sum<SPLIT>((at[1][0] + at[1][1]) + (at[1][2] + at[1][3])) * gscale;
            static_assert(CPL == 2, "the float2 out store pairs exactly two columns");
            if (part == 0) *reinterpret_cast<float2 *>(&out[((size_t) (t0 + tt) * H + h) * S_v + col]) = o;

            if constexpr (SNAP) {
                const int slot = n_tokens - 1 - (t0 + tt);
                if (slot < K) {
#pragma unroll
                    for (int cc = 0; cc < CPL; cc++)
#pragma unroll
                        for (int r = 0; r < ROWS; r++)
                            s_out[slot * slot_stride + (col + cc) * S_v + GDN_ROW(r)] = s[cc][r];
                }
            }
        }
        };
        if constexpr (KEEP) {
            if (t0 + nt > n_tokens - K) token_loop(std::true_type{});
            else                        token_loop(std::false_type{});
        } else {
            token_loop(std::false_type{});
        }
    }
    if constexpr (!KEEP) {
#pragma unroll
        for (int cc = 0; cc < CPL; cc++)
#pragma unroll
            for (int r = 0; r < ROWS; r++) s_out[(col + cc) * S_v + GDN_ROW(r)] = s[cc][r];
    }
}

// Raw pointers and the caller's stream: this file includes no torch headers (a pip ROCm torch
// ships without the rocThrust headers they need). state_out holds max(K, 1) slots.
extern "C" void gdn_prefill_hip(const float * q, const float * k, const float * v, const float * g,
                                const float * beta, const float * state, float * out, float * state_out,
                                int T, int H, int Hk, int K, hipStream_t stream) {
    constexpr int SPLIT = 8, CPL = 2, COLS = 64, CHUNK = 16;
    dim3 grid(H, 1, 128 / COLS), block((COLS / CPL) * SPLIT);
    const float scale = 1.0f / sqrtf(128.0f);
    if (K > 1) {
        hipLaunchKernelGGL((gdn_prefill<128, SPLIT, CPL, COLS, CHUNK, true>), grid, block, 0, stream,
            q, k, v, g, beta, state, out, state_out, H, Hk, T, K, scale);
    } else {
        hipLaunchKernelGGL((gdn_prefill<128, SPLIT, CPL, COLS, CHUNK, false>), grid, block, 0, stream,
            q, k, v, g, beta, state, out, state_out, H, Hk, T, 1, scale);
    }
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime_api.h>
extern "C" void gdn_prefill_hip(const float *, const float *, const float *, const float *, const float *,
                                const float *, float *, float *, int, int, int, int, hipStream_t);

// q, k: [T, Hk, 128]; v: [T, H, 128]; g, beta: [T, H]; state: [H, 128, 128] (transposed);
// returns out [T, H, 128] and states [max(K, 1), H, 128, 128] (slot 0 = final state).
std::vector<torch::Tensor> gdn_prefill_launch(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
                                              torch::Tensor beta, torch::Tensor state, int64_t K, int64_t stream) {
    const int T = v.size(0), H = v.size(1), Hk = q.size(1);
    auto out    = torch::empty_like(v);
    auto states = torch::empty({std::max<int64_t>(K, 1), H, 128, 128}, v.options());
    gdn_prefill_hip(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), g.data_ptr<float>(),
                    beta.data_ptr<float>(), state.data_ptr<float>(), out.data_ptr<float>(), states.data_ptr<float>(),
                    T, H, Hk, (int) K, reinterpret_cast<hipStream_t>(stream));
    return {out, states};
}
"""

_ext = None


def _device_lib_flags():
    # a pip ROCm torch (TheRock wheels) ships the device bitcode inside its SDK package, where its
    # hipcc does not look by itself
    import importlib.util
    candidates = [os.environ.get("ROCM_DEVICE_LIB_PATH")]
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec and spec.submodule_search_locations:
        candidates.append(os.path.join(spec.submodule_search_locations[0], "lib", "llvm", "amdgcn", "bitcode"))
    for path in candidates:
        if path and os.path.isdir(path):
            return [f"--rocm-device-lib-path={path}"]
    return []


def _module():
    # load(), not load_inline(): load_inline prepends torch/types.h to the GPU source
    global _ext
    if _ext is None:
        import hashlib
        import tempfile
        tag = hashlib.sha1((_HIP_SRC + _CPP_SRC).encode()).hexdigest()[:12]
        src = os.path.join(tempfile.gettempdir(), f"gdn_prefill_{tag}")
        os.makedirs(src, exist_ok=True)
        with open(os.path.join(src, "gdn_prefill.hip"), "w") as f:
            f.write(_HIP_SRC)
        with open(os.path.join(src, "gdn_prefill_bind.cpp"), "w") as f:
            f.write(_CPP_SRC + '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gdn_prefill_launch", &gdn_prefill_launch); }\n')
        _ext = load(
            name=f"gdn_prefill_{tag}",
            sources=[os.path.join(src, "gdn_prefill_bind.cpp"), os.path.join(src, "gdn_prefill.hip")],
            extra_cuda_cflags=["-O3", f"--offload-arch={os.environ.get('GPU_TARGET', 'gfx1151')}"] + _device_lib_flags(),
            verbose=False,
        )
    return _ext


def gdn_prefill(q, k, v, g, beta, state, K: int = 1):
    """Delta-rule prefill over T tokens. Public entry point: returns (out, states)."""
    return _module().gdn_prefill_launch(q, k, v, g, beta, state, int(K), torch.cuda.current_stream().cuda_stream)
