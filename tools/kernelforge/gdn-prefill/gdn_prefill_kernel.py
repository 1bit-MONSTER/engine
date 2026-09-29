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
(ggml/src/ggml-cuda/gated_delta_net.cu, ROCmFPX#1 and #4), for gfx1151 (Strix Halo). One call
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

template <int WIDTH>
__device__ __forceinline__ float group_sum(float x) {
#pragma unroll
    for (int o = WIDTH / 2; o > 0; o >>= 1) x += __shfl_xor(x, o, WIDTH);
    return x;
}

// 4 lanes per state column, 32 columns per block, 32-token chunks of k/q/v/g/beta in shared memory.
template <int S_v, int SPLIT, int COLS, int CHUNK, bool KEEP>
__global__ void __launch_bounds__(COLS * SPLIT)
gdn_prefill(const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
            const float * __restrict__ g, const float * __restrict__ beta, const float * __restrict__ state_in,
            float * __restrict__ out, float * __restrict__ state_out,
            int H, int Hk, int n_tokens, int K, float scale) {
    constexpr int ROWS     = S_v / SPLIT;
    constexpr int NTHREADS = COLS * SPLIT;

    __shared__ float k_s[CHUNK][S_v];
    __shared__ float q_s[CHUNK][S_v];
    __shared__ float v_s[CHUNK][COLS];
    __shared__ float g_s[CHUNK];
    __shared__ float b_s[CHUNK];

    const int h       = blockIdx.x;
    const int hk      = h % Hk;
    const int tid     = threadIdx.x;
    const int col_loc = tid / SPLIT;
    const int part    = tid % SPLIT;
    const int col0    = blockIdx.z * COLS;
    const int col     = col0 + col_loc;
    const int row0    = part * ROWS;

    const float * s_in  = state_in + (size_t) h * S_v * S_v;
    float *       s_out = state_out + (size_t) h * S_v * S_v;
    const size_t  slot_stride = (size_t) H * S_v * S_v;

    float s[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; r++) s[r] = s_in[col * S_v + row0 + r];

    for (int t0 = 0; t0 < n_tokens; t0 += CHUNK) {
        const int nt = min(CHUNK, n_tokens - t0);
        __syncthreads();
        for (int idx = tid; idx < nt * S_v; idx += NTHREADS) {
            const int tt = idx / S_v, i = idx % S_v;
            k_s[tt][i] = k[((size_t) (t0 + tt) * Hk + hk) * S_v + i];
            q_s[tt][i] = q[((size_t) (t0 + tt) * Hk + hk) * S_v + i];
        }
        for (int idx = tid; idx < nt * COLS; idx += NTHREADS) {
            const int tt = idx / COLS, c = idx % COLS;
            v_s[tt][c] = v[((size_t) (t0 + tt) * H + h) * S_v + col0 + c];
        }
        if (tid < nt) {
            g_s[tid] = expf(g[(size_t) (t0 + tid) * H + h]);
            b_s[tid] = beta[(size_t) (t0 + tid) * H + h];
        }
        __syncthreads();

        for (int tt = 0; tt < nt; tt++) {
            const float g_val = g_s[tt], beta_val = b_s[tt];
            float kv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int r = 0; r < ROWS; r++) kv[r % 4] += s[r] * k_s[tt][row0 + r];
            const float kv_col    = group_sum<SPLIT>((kv[0] + kv[1]) + (kv[2] + kv[3]));
            const float delta_col = (v_s[tt][col_loc] - g_val * kv_col) * beta_val;

            float at[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                s[r]       = g_val * s[r] + k_s[tt][row0 + r] * delta_col;
                at[r % 4] += s[r] * q_s[tt][row0 + r];
            }
            const float attn_col = group_sum<SPLIT>((at[0] + at[1]) + (at[2] + at[3]));
            if (part == 0) out[((size_t) (t0 + tt) * H + h) * S_v + col] = attn_col * scale;
            if constexpr (KEEP) {
                const int slot = n_tokens - 1 - (t0 + tt);
                if (slot < K) {
#pragma unroll
                    for (int r = 0; r < ROWS; r++) s_out[slot * slot_stride + col * S_v + row0 + r] = s[r];
                }
            }
        }
    }
    if constexpr (!KEEP) {
#pragma unroll
        for (int r = 0; r < ROWS; r++) s_out[col * S_v + row0 + r] = s[r];
    }
}

// Raw pointers and the caller's stream: this file includes no torch headers (a pip ROCm torch
// ships without the rocThrust headers they need). state_out holds max(K, 1) slots.
extern "C" void gdn_prefill_hip(const float * q, const float * k, const float * v, const float * g,
                                const float * beta, const float * state, float * out, float * state_out,
                                int T, int H, int Hk, int K, hipStream_t stream) {
    constexpr int SPLIT = 4, COLS = 32, CHUNK = 32;
    dim3 grid(H, 1, 128 / COLS), block(COLS * SPLIT);
    const float scale = 1.0f / sqrtf(128.0f);
    if (K > 1) {
        hipLaunchKernelGGL((gdn_prefill<128, SPLIT, COLS, CHUNK, true>), grid, block, 0, stream,
            q, k, v, g, beta, state, out, state_out, H, Hk, T, K, scale);
    } else {
        hipLaunchKernelGGL((gdn_prefill<128, SPLIT, COLS, CHUNK, false>), grid, block, 0, stream,
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
