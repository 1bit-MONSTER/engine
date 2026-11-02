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

// mtp_head.h -- the MTP draft head on the host, in the arithmetic our containers actually hold.
//
// The head is one full-attention layer with an input-fusion GEMM in front (see tools/mtp_head_ref.py for the
// reference and npu-re/docs/FACTS-flm-q4nx-container-and-mtp-head.md for the container).  It runs here in the
// pb16i-exact form: per 32-K group a signed int4 code and an int16 magnitude, then the source's additive
// min as a host table --  y[n] = sum_g (sum_k a_k code_kn) ms_gn 2^e_ref[n]  +  sum_g A_g add_gn.
// That is the same decomposition the device does (npu-arena/HARVEST.md), so a draft measured here is a draft
// the NPU head can reproduce; and the final projection to vocabulary goes through the BASE model's lm head,
// because the drafter does not have one of its own.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace onebit::npu::mtp {

constexpr int kH = 5120, kIM = 17408, kHD = 256, kNH = 24, kNKV = 4, kRot = 64;
constexpr int kDQ = kNH * kHD;      // 6144 (q, and again the output gate)
constexpr int kKV = kNKV * kHD;     // 1024
constexpr int kG = 32;              // K values per scale group

// One matrix in the pb16i-exact form: codes [K][N] int8, magnitudes [K/32][N] int16, one exponent per column,
// plus the source's additive min as [K/32][N] float.
struct Matrix {
    int K = 0, N = 0;
    std::vector<int8_t> codes;      // K*N, code-major (k*N + n)
    std::vector<int16_t> ms;        // (K/32)*N, group-major (g*N + n)
    std::vector<int16_t> eref;      // N
    std::vector<float> add;         // (K/32)*N
    bool empty() const { return codes.empty(); }

    // y[N] = matvec(this, a[K]) exactly as the device plus the host table compute it.
    void matvec(const float* a, float* y) const;
};

struct Weights {
    Matrix fc, q, k, v, o, gate, up, down;
    std::vector<float> pre_fc_emb, pre_fc_hid, ln_in, ln_post, norm, q_norm, k_norm;  // BF16 -> float
};

struct KvCache {
    std::vector<float> k, v;   // [t][NKV][HD], grown per step
    int t = 0;
    void reset() { k.clear(); v.clear(); t = 0; }
};

// Load ~/models/npu/bonsai2-27b-mtp-head/pb16i-exact (manifest.json + <t>.pb16i/.add/.eref) and the q4_k norms.
bool load_weights(const std::string& exact_dir, const std::string& q4k_dir, Weights& w, std::string& err);

// One draft step: hidden (H) and the accepted token's embedding (H) in, the head's hidden out, and -- when
// `logits_out` is not null -- the vocabulary logits through `lm_head` (a callback into the base model).
// Returns false only on a shape mismatch.
bool head_step(const Weights& w, const float* hidden, const float* tok_emb, int pos, KvCache& cache,
               std::vector<float>& out_hidden, const std::function<bool(const std::vector<float>&,
                                                                      std::vector<float>&)>& lm_head,
               std::vector<float>& logits_out);

}  // namespace onebit::npu::mtp
