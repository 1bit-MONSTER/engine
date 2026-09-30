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
"""Q4_0 weights x Q8_1 activations for a few columns: the target a KernelForge campaign optimizes.

A standalone copy of ggml's mul_mat_vec_q for GGML_TYPE_Q4_0 as the engine's ROCmFPX pin runs it
on gfx1151 (ggml/src/ggml-cuda/mmvq.cu + vecdotq.cuh): one warp per block, one row per block,
vdr = 2, one dp4a vec_dot per (column, block), plus, for 5-8 columns, the row-tiled body a first
campaign found (ROCmFPX#6 ships it for 8 columns). It is the matmul of speculative verification: a
drafter proposes up to 7 tokens and the model checks them, plus the last accepted one, as a
batch of up to 8 columns, reading all 15 GB of Qwen3.8-27B's weights once.

Formats (ggml's):
    block_q4_0 (18 B, 32 weights): half d; uint8 qs[16]. Weight j of the block is
        (qs[j] & 0xF) - 8 for j < 16 and (qs[j - 16] >> 4) - 8 for j >= 16, times d.
    block_q8_1 (36 B, 32 values): half2 ds = (d, d * sum(q)); int8 qs[32]. Value j is qs[j] * d.

    out[c][r] = sum over blocks b of row r:  d_w * (d_a * sum_j q4_j q8_j  -  8 * s_a)
where (d_a, s_a) = ds of column c's block b, so the "- 8" of every weight is applied once per
block through the precomputed sum.
"""
from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load

os.environ.setdefault("PYTORCH_ROCM_ARCH", os.environ.get("GPU_TARGET", "gfx1151"))

QK = 32
Q4_0_BYTES = 18
Q8_1_BYTES = 36

_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

struct block_q4_0 { __half d; uint8_t qs[16]; };
struct block_q8_1 { __half2 ds; int8_t qs[32]; };
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 is 18 bytes");
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 is 36 bytes");

static __device__ __forceinline__ int get_int_b2(const void * x, int i32) {   // 2-byte aligned
    const uint16_t * x16 = (const uint16_t *) x;
    return x16[2 * i32] | (x16[2 * i32 + 1] << 16);
}
static __device__ __forceinline__ int get_int_b4(const void * x, int i32) {   // 4-byte aligned
    return ((const int *) x)[i32];
}
static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);   // ggml_cuda_dp4a on RDNA3/4
}

constexpr int QI4_0 = 4;   // ints of quants per block_q4_0 half: 32 / (4 * 2)
constexpr int VDR   = 2;   // ints per vec_dot call (VDR_Q4_0_Q8_1_MMVQ)
constexpr int WARP  = 32;

static __device__ __forceinline__ float vec_dot_q4_0_q8_1(const block_q4_0 * bq4, const block_q8_1 * bq8, int iqs) {
    int v[VDR], u[2 * VDR];
#pragma unroll
    for (int i = 0; i < VDR; ++i) {
        v[i]         = get_int_b2(bq4->qs, iqs + i);
        u[2 * i + 0] = get_int_b4(bq8->qs, iqs + i);
        u[2 * i + 1] = get_int_b4(bq8->qs, iqs + i + QI4_0);
    }
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi = dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dp4a(vi1, u[2 * i + 1], sumi);
    }
    const float2 ds8 = __half22float2(bq8->ds);
    // the -8 offset of every weight, applied through the block sum: 8 * (vdr / QI4_0) of it per call
    return __half2float(bq4->d) * (sumi * ds8.x - (8 * VDR / QI4_0) * ds8.y);
}

// One warp per block, one row per block (the gfx1151 parameters for Q4_0).
template <int NCOLS>
__global__ void __launch_bounds__(WARP, 1)
gemv_q4_0(const block_q4_0 * __restrict__ x, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
          int ncols_x, int nrows) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int blocks_per_row = ncols_x / 32;
    constexpr int blocks_per_iter = VDR * WARP / QI4_0;   // 16

    float tmp[NCOLS] = {0.0f};
    const block_q4_0 * xr = x + (size_t) row * blocks_per_row;
    for (int kbx = tid / (QI4_0 / VDR); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kqs = VDR * (tid % (QI4_0 / VDR));
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            tmp[j] += vec_dot_q4_0_q8_1(&xr[kbx], &y[(size_t) j * blocks_per_row + kbx], kqs);
        }
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) tmp[j] += __shfl_xor(tmp[j], o, WARP);
    }
    if (tid < NCOLS) {
        float v = tmp[0];
#pragma unroll
        for (int j = 1; j < NCOLS; ++j) if (tid == j) v = tmp[j];
        dst[(size_t) tid * nrows + row] = v;
    }
}

// Row register-tiling: R consecutive rows per warp. Each k-block's activation operands (and the
// nibble unpack of each weight int) are loaded/computed ONCE and reused, instead of once per
// (row, column) pair as the one-row-per-warp body does. The arithmetic per (row, column, block)
// and its summation order are bit-identical to vec_dot_q4_0_q8_1 above.
// R=4 measured best on both 8-column cases (R=1 -> 2 -> 4 each improved, and R=4 holds 92 VGPRs
// with 0 spills at 16 waves/SIMD at 8 columns). R=4 is also the largest R that keeps one result per lane in
// the epilogue at 8 columns. At 8 columns R=1 and R=2 are far worse (0.265 and 0.239 against 0.226 on
// 5120x17408; 0.270 and 0.239 against 0.221 on 17408x5120) and R=8 needs tmp[8][8] = 64 accumulators,
// which does not fit under the 96-VGPR ceiling for 16 waves/SIMD. At 4 columns row tiling has nothing
// left to reuse and every R > 1 loses: R=2 0.228, R=4 0.222, R=8 0.240 against 0.216 for the one-row
// body, so 1..4 columns keep that body.
//
// This body is at the memory floor, and the floor is the weight stream, not the activation side:
//   - a 50 MB dwordx4 grid-stride read of the weight matrix runs at 0.217 ms / 241 GB/s, and the
//     one-column body reaches 0.213 ms -- 97% of that, with 100% DRAM byte efficiency (a row is
//     45 whole 64 B lines, 64 B-aligned, and every byte of every line is consumed);
//   - hoisting ALL activation loads out of the k-loop (wrong answers, ablation only) saves just
//     0.005 ms at 8 columns, so 2.2% on the two 8-column cases bounds EVERY activation-side idea.
// Measured and rejected against that bound, all slower than what is here: LDS staging of the
// activation k-tile (single- and double-buffered, 2/4/8 waves per block, 0.238-0.42 ms);
// multi-wave workgroups with independent row tiles and no LDS (0.230/0.231); splitting the lane
// pair by nibble half so each lane loads 16 contiguous activation bytes as one b128 (0.253);
// non-temporal and explicitly non-temporal-b64 weight loads (0.28-0.33 at 8 columns); -mcumode.
// Non-temporal weight loads DO beat this body when the machine is quiet -- partitioning the matrix
// into a cacheable and a streamed row range reached 0.194 ms, below the DRAM floor, by keeping part
// of the 50 MB resident in the 32 MB Infinity Cache instead of LRU-thrashing all of it. Do not
// pursue it: that gain lives entirely in the cache and inverts when anything else shares the bus.
// Canonically it measured 0.978x with a 3-run score spread of 0.29, against 1.173x for this body.
constexpr int ROWS_PER_WARP = 4;

template <int NCOLS, int R>
__global__ void __launch_bounds__(WARP, 1)
gemv_q4_0_rt(const block_q4_0 * __restrict__ x, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
             int ncols_x, int nrows) {
    static_assert(R * NCOLS <= WARP, "one lane per result in the epilogue");
    const int tid = threadIdx.x;
    const int blocks_per_row = ncols_x / 32;
    constexpr int blocks_per_iter = VDR * WARP / QI4_0;   // 16
    const int row0 = blockIdx.x * R;                      // consecutive rows: one narrow weight window
    const int kqs = VDR * (tid % (QI4_0 / VDR));

    float tmp[R][NCOLS] = {{0.0f}};
    const block_q4_0 * xr = x + (size_t) row0 * blocks_per_row;
    for (int kbx = tid / (QI4_0 / VDR); kbx < blocks_per_row; kbx += blocks_per_iter) {
        int   u[NCOLS][2 * VDR];      // this lane's slice of every column's k-block: read once
        float2 ds[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            const block_q8_1 * bq8 = &y[(size_t) j * blocks_per_row + kbx];
#pragma unroll
            for (int i = 0; i < VDR; ++i) {
                u[j][2 * i + 0] = get_int_b4(bq8->qs, kqs + i);
                u[j][2 * i + 1] = get_int_b4(bq8->qs, kqs + i + QI4_0);
            }
            ds[j] = __half22float2(bq8->ds);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const block_q4_0 * bq4 = &xr[(size_t) r * blocks_per_row + kbx];
            int vi0[VDR], vi1[VDR];
#pragma unroll
            for (int i = 0; i < VDR; ++i) {          // unpacked once, used by all NCOLS columns
                const int vv = get_int_b2(bq4->qs, kqs + i);
                vi0[i] = (vv >> 0) & 0x0F0F0F0F;
                vi1[i] = (vv >> 4) & 0x0F0F0F0F;
            }
            const float dw = __half2float(bq4->d);
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < VDR; ++i) {
                    sumi = dp4a(vi0[i], u[j][2 * i + 0], sumi);
                    sumi = dp4a(vi1[i], u[j][2 * i + 1], sumi);
                }
                tmp[r][j] += dw * (sumi * ds[j].x - (8 * VDR / QI4_0) * ds[j].y);
            }
        }
    }
    // Epilogue. Every lane holds a partial for each of the R*NCOLS results and the warp needs, per
    // result, the sum over lanes. When there is exactly one result per lane, a reduce-scatter
    // halves the live accumulator count at each xor stage instead of running an independent
    // 5-stage butterfly per accumulator: 31 shuffles for 32 results rather than 160, and lane i
    // ends holding result i, which also deletes the 32-way select chain before the store. The
    // pairing order (xor 16, 8, 4, 2, 1) is the butterfly's, so each result is summed by the same
    // balanced tree over the same partials -- bit for bit the same float.
    if constexpr (R * NCOLS == WARP) {
        float acc[WARP];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
            for (int r = 0; r < R; ++r) acc[j * R + r] = tmp[r][j];
        }
#pragma unroll
        for (int m = WARP / 2; m > 0; m >>= 1) {
            const bool hi = (tid & m) != 0;
#pragma unroll
            for (int k = 0; k < m; ++k) {
                const float a = acc[k], b = acc[k + m];
                acc[k] = (hi ? b : a) + __shfl_xor(hi ? a : b, m, WARP);
            }
        }
        dst[(size_t) (tid / R) * nrows + row0 + (tid % R)] = acc[0];
    } else {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int o = WARP / 2; o > 0; o >>= 1) tmp[r][j] += __shfl_xor(tmp[r][j], o, WARP);
            }
        }
        if (tid < R * NCOLS) {          // lane j*R + r holds column j of row row0 + r
            float v = tmp[0][0];
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int r = 0; r < R; ++r) if (tid == j * R + r) v = tmp[r][j];
            }
            dst[(size_t) (tid / R) * nrows + row0 + (tid % R)] = v;
        }
    }
}

// x: nrows * (ncols_x/32) block_q4_0; y: ncols * (ncols_x/32) block_q8_1; dst: [ncols][nrows] float.
extern "C" void gemv_q4_0_hip(const void * x, const void * y, float * dst, int ncols_x, int nrows, int ncols,
                              hipStream_t stream) {
    const dim3 grid(nrows), block(WARP);
    const block_q4_0 * bx = (const block_q4_0 *) x;
    const block_q8_1 * by = (const block_q8_1 *) y;
    // Row tiling only pays where the activation side is the scaling cost; 1..4 columns already run
    // at ~88% of LPDDR5X peak, so they keep the one-row body verbatim (no risk to plain decode).
    constexpr int R = ROWS_PER_WARP;
    if (ncols >= 5 && R > 1 && nrows % R == 0) {
        const dim3 grid_rt(nrows / R);
        switch (ncols) {
            case 5: hipLaunchKernelGGL((gemv_q4_0_rt<5, R>), grid_rt, block, 0, stream, bx, by, dst, ncols_x, nrows); return;
            case 6: hipLaunchKernelGGL((gemv_q4_0_rt<6, R>), grid_rt, block, 0, stream, bx, by, dst, ncols_x, nrows); return;
            case 7: hipLaunchKernelGGL((gemv_q4_0_rt<7, R>), grid_rt, block, 0, stream, bx, by, dst, ncols_x, nrows); return;
            case 8: hipLaunchKernelGGL((gemv_q4_0_rt<8, R>), grid_rt, block, 0, stream, bx, by, dst, ncols_x, nrows); return;
            default: break;
        }
    }
    switch (ncols) {
        case 1: hipLaunchKernelGGL(gemv_q4_0<1>, grid, block, 0, stream, bx, by, dst, ncols_x, nrows); break;
        case 2: hipLaunchKernelGGL(gemv_q4_0<2>, grid, block, 0, stream, bx, by, dst, ncols_x, nrows); break;
        case 3: hipLaunchKernelGGL(gemv_q4_0<3>, grid, block, 0, stream, bx, by, dst, ncols_x, nrows); break;
        case 4: hipLaunchKernelGGL(gemv_q4_0<4>, grid, block, 0, stream, bx, by, dst, ncols_x, nrows); break;
        case 5: hipLaunchKernelGGL(gemv_q4_0<5>, grid, block, 0, stream, bx, by, dst, ncols_x, nrows); break;
        case 6: hipLaunchKernelGGL(gemv_q4_0<6>, grid, block, 0, stream, bx, by, dst, ncols_x, nrows); break;
        case 7: hipLaunchKernelGGL(gemv_q4_0<7>, grid, block, 0, stream, bx, by, dst, ncols_x, nrows); break;
        case 8: hipLaunchKernelGGL(gemv_q4_0<8>, grid, block, 0, stream, bx, by, dst, ncols_x, nrows); break;
        default: break;
    }
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime_api.h>
extern "C" void gemv_q4_0_hip(const void *, const void *, float *, int, int, int, hipStream_t);

torch::Tensor gemv_q4_0_launch(torch::Tensor x, torch::Tensor y, int64_t ncols_x, int64_t nrows, int64_t ncols,
                               int64_t stream) {
    TORCH_CHECK(ncols >= 1 && ncols <= 8, "ncols must be 1..8");
    auto dst = torch::empty({ncols, nrows}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    gemv_q4_0_hip(x.data_ptr(), y.data_ptr(), dst.data_ptr<float>(), (int) ncols_x, (int) nrows, (int) ncols,
                  reinterpret_cast<hipStream_t>(stream));
    return dst;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemv_q4_0_launch", &gemv_q4_0_launch); }
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
    # load(), not load_inline(): load_inline prepends torch/types.h to the GPU source, and a pip ROCm
    # torch ships without the rocThrust headers that pulls in
    global _ext
    if _ext is None:
        import hashlib
        import tempfile
        tag = hashlib.sha1((_HIP_SRC + _CPP_SRC).encode()).hexdigest()[:12]
        src = os.path.join(tempfile.gettempdir(), f"gemv_q4_0_{tag}")
        os.makedirs(src, exist_ok=True)
        with open(os.path.join(src, "gemv_q4_0.hip"), "w") as f:
            f.write(_HIP_SRC)
        with open(os.path.join(src, "gemv_q4_0_bind.cpp"), "w") as f:
            f.write(_CPP_SRC)
        _ext = load(
            name=f"gemv_q4_0_{tag}",
            sources=[os.path.join(src, "gemv_q4_0_bind.cpp"), os.path.join(src, "gemv_q4_0.hip")],
            extra_cuda_cflags=["-O3", f"--offload-arch={os.environ.get('GPU_TARGET', 'gfx1151')}"] + _device_lib_flags(),
            verbose=False,
        )
    return _ext


def gemv_q4_0(x, y, ncols_x: int, nrows: int, ncols: int):
    """x: uint8 [nrows * ncols_x/32 * 18] (block_q4_0), y: uint8 [ncols * ncols_x/32 * 36] (block_q8_1).
    Returns float32 [ncols, nrows]. Public entry point."""
    return _module().gemv_q4_0_launch(x, y, ncols_x, nrows, ncols, torch.cuda.current_stream().cuda_stream)
