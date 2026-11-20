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
//
// audio_npu_gemm.h -- the audio executor's matrix products on an NpuTarget (engine#1783, the encoder route).
//
// The executor (npu/audio_exec.h, AudioGemm) offers every product whose right operand is a weight: the projections and the
// two Conv stems as im2col GEMMs. This file is the device side of that offer, and it is device-free: it speaks to
// `NpuTarget::gemv`, so it is tested against a fake target (tests/audio_npu_gemm_test.cpp).
//
// What it decides, per product:
//  * WHICH design. The caller passes the designs in the order it gave them to the target's open_prefill(); the target's
//    role index IS that order (xdna_target.cpp: make_ctx(i)). A product goes to the design whose (K, N) is its STAGED
//    shape -- K on the 256 grid, N on the 512 grid (audio_kernel_roles.h) -- so whisper-small's O (768 x 768) asks for
//    K768 N1024, conv1 (240 x 768) for K256 N1024, conv2 (2304 x 768) for K2304 N1024. A shape no design covers is
//    declined by name and runs on the host. Routing every launch to role 0 is the defect that crashed the first
//    standalone harness (engine#1783, 08:48Z), so the index is never a default.
//  * HOW MANY launches. One launch holds at most the design's M rows; more rows are cut into M-row chunks (conv1's 3000
//    frames at M=1536 are two launches).
//  * WHAT the weight is. The target packs `Bmat` as K_staged x N_staged f32, so the model's K x N weight is zero-padded
//    here once, under a stable residency key (layer 0, one wkey per weight). After the first launch the target serves
//    the packed weight from its cache. The padded host copy is dropped then, unless the caller says the target does
//    not cache.
//  * WHAT comes back. The target writes rows x N_staged; the executor's tensor is rows x N, so the padding columns are
//    dropped per row.
//
// It also keeps the device column of the issue's bar: the wall time inside gemv, the launches and the products, so the
// tool prints `device_gemm=<X> s + host_remainder=<Y> s` from a run rather than from arithmetic.

#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "audio_kernel_roles.h"
#include "audio_epilogue.h"
#include "target.h"

namespace onebit::audio {

struct NpuAudioDesign {
    int M = 0;   // rows per launch (the ELF's MD)
    int K = 0;   // staged K
    int N = 0;   // staged N
};

class NpuAudioGemm {
public:
    // `designs[i]` is role index i of `target` (the open_prefill() order). `target_caches` = the target was given a weight
    // cache with at least one layer (set_device(..., weight_cache=true, n_layers>=1)); without it the padded weights stay
    // on the host and are re-sent every launch.
    NpuAudioGemm(::npu::target::NpuTarget& target, std::vector<NpuAudioDesign> designs, bool target_caches = true)
        : t_(target), designs_(std::move(designs)), caches_(target_caches) {}

    // The AudioGemm::gemm contract: C (M x N) = A (M x K) x B (K x N); false + why to leave it to the host.
    bool gemm(const std::string& key, const float* A, int64_t M, int64_t K, const float* B, int64_t N,
              std::vector<float>* C, std::string* why) {
        return gemm_ep(key, A, M, K, B, N, C, nullptr, why);
    }

    // engine#1783: the same product with the projection's tail folded in (bias, GELU, residual; audio_epilogue.h). The target
    // applies it in its dequant when it can; otherwise it is applied here on the rows the target wrote, same order.
    bool gemm_ep(const std::string& key, const float* A, int64_t M, int64_t K, const float* B, int64_t N,
                 std::vector<float>* C, const Epilogue* ep, std::string* why) {
        // B may be null only for a weight the target already holds (cached(key)): the executor then skips loading it.
        if (A == nullptr || (B == nullptr && !cached(key)) || C == nullptr || M <= 0 || K <= 0 || N <= 0) {
            if (why) *why = "empty product";
            return false;
        }
        const ProjectionShape st0 = staged(ProjectionShape{int(K), int(N)});
        const int role = role_for(st0.k, st0.n);
        if (role < 0) {
            if (why)
                *why = "no design for staged K" + std::to_string(st0.k) + " N" + std::to_string(st0.n) + " (model " +
                       std::to_string(K) + " x " + std::to_string(N) + ")";
            return false;
        }
        const NpuAudioDesign& d = designs_[size_t(role)];
        // The shape the launch runs at: the design's, which is the staged shape or (role_for's fallback) a wider N.
        const ProjectionShape st{d.K, d.N};
        if (d.M <= 0) {
            if (why) *why = "design " + std::to_string(role) + " states no rows";
            return false;
        }

        Weight& w = weight(key, B, K, N, st);
        if (w.role < 0) { w.role = role; w.k = st.k; w.n = st.n; }
        const auto t0 = std::chrono::steady_clock::now();
        // engine#1783: with direct output (default) the target dequantises each chunk's rows straight into C, so C is sized
        // once and nothing is copied; ONEBIT_AUDIO_DIRECT_OUT=0 keeps the staged-width buffer and the per-row copy.
        static const bool direct = [] {
            const char* e = std::getenv("ONEBIT_AUDIO_DIRECT_OUT");
            return e == nullptr || e[0] != '0';
        }();
        if (direct) C->resize(size_t(M) * size_t(N));
        else C->assign(size_t(M) * size_t(N), 0.0f);
        std::vector<float> got;
        for (int64_t r0 = 0; r0 < M; r0 += d.M) {
            const int rows = int(std::min<int64_t>(d.M, M - r0));
            ::npu::target::GemvArgs a;
            a.role = role;
            a.layer = 0;
            a.M = rows;
            a.K = st.k;
            a.N = st.n;
            a.x = A + size_t(r0) * size_t(K);
            a.a_k = int(K);
            a.Bmat = w.padded.empty() ? nullptr : w.padded.data();
            a.wkey = caches_ ? w.wkey : -1;
            a.rows = true;   // the batched body, whatever the chunk's row count (target.h: M==1 with rows=true is batched)
            a.tag = "audio";
            if (direct) {
                a.dst = C->data() + size_t(r0) * size_t(N);
                a.dst_ld = int(N);
                a.dst_n = int(N);
            }
            Epilogue chunk_ep;
            int ep_done = 0;
            if (ep != nullptr && ep->any()) {
                chunk_ep = *ep;
                if (chunk_ep.res != nullptr) chunk_ep.res += size_t(r0) * size_t(chunk_ep.res_ld);
                a.epilogue = &chunk_ep;
                a.epilogue_done = &ep_done;
            }
            got.clear();
            std::string err;
            if (!t_.gemv(a, got, err)) {
                if (why) *why = "role " + std::to_string(role) + " (" + key + "): " + (err.empty() ? "gemv failed" : err);
                return false;
            }
            const bool wrote_c = direct && got.empty();   // the target honoured dst
            if (!wrote_c && got.size() != size_t(rows) * size_t(st.n)) {
                if (why)
                    *why = "role " + std::to_string(role) + " wrote " + std::to_string(got.size()) + " floats for " +
                           std::to_string(rows) + " x " + std::to_string(st.n);
                return false;
            }
            for (int r = 0; !wrote_c && r < rows; ++r)
                std::copy(got.begin() + size_t(r) * size_t(st.n), got.begin() + size_t(r) * size_t(st.n) + size_t(N),
                          C->begin() + size_t(r0 + r) * size_t(N));
            if (ep != nullptr && ep->any() && ep_done == 0)
                epilogue_rows(chunk_ep, 0, rows, N, C->data() + size_t(r0) * size_t(N), N);   // the target did not
            ++launches_;
            ++per_role_[role];
            // The target now holds the packed weight under (layer 0, wkey): later chunks and later calls send none.
            if (caches_ && !w.padded.empty()) w.padded = std::vector<float>();
        }
        device_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++products_;
        return true;
    }

    // True when `key`'s packed weight is in the target's cache, so gemm() needs no B for it (engine#1783).
    bool cached(const std::string& key) const {
        const auto it = w_.find(key);
        return caches_ && it != w_.end() && it->second.padded.empty();
    }

    // engine#1783: every weight the target holds, with what it takes to restore it in another process: `out(key, role, staged
    // K, staged N, image, scales)`. Returns how many were handed out.
    template <class Out>
    int export_weights(Out&& out) const {
        int n = 0;
        std::vector<uint8_t> img;
        std::vector<float> sc;
        for (const auto& kv : w_) {
            const Weight& w = kv.second;
            if (!caches_ || !w.padded.empty() || w.role < 0) continue;
            if (!t_.export_weight(w.role, 0, w.wkey, &img, &sc)) continue;
            out(kv.first, w.role, w.k, w.n, img, sc);
            ++n;
        }
        return n;
    }

    // engine#1783: make `key` resident from a persisted image, as if a gemm() had packed it: cached(key) is then true, the
    // executor skips loading the f32 weight, and gemm() launches with no Bmat. The design must still be role `role` at
    // the same staged shape; false + why otherwise (the caller then lets the weight be packed as usual).
    bool import_weight(const std::string& key, int role, int k, int n, size_t bytes, const std::function<bool(void*)>& fill,
                       const std::vector<float>& scales, std::string* why) {
        if (!caches_ || w_.count(key) != 0) {
            if (why) *why = "import: " + key + " is already known";
            return false;
        }
        if (role < 0 || size_t(role) >= designs_.size() || designs_[size_t(role)].K != k || designs_[size_t(role)].N != n) {
            if (why) *why = "import: " + key + " was saved for a design this set does not have at that role";
            return false;
        }
        std::string err;
        const int wkey = next_wkey_;
        if (!t_.import_weight(role, 0, wkey, bytes, fill, scales, err)) {
            if (why) *why = err;
            return false;
        }
        ++next_wkey_;
        Weight w;
        w.wkey = wkey;
        w.role = role;
        w.k = k;
        w.n = n;
        w_.emplace(key, std::move(w));
        return true;
    }

    double device_ms() const { return device_ms_; }
    int launches() const { return launches_; }
    int products() const { return products_; }
    const std::map<int, int>& launches_per_role() const { return per_role_; }
    int weights() const { return int(w_.size()); }

private:
    struct Weight {
        int wkey = -1;
        std::vector<float> padded;   // K_staged x N_staged, until the target has it
        int role = -1, k = 0, n = 0; // the design it launches on, and the staged shape (export_weights)
    };

    // The design for staged (k, n): the exact one, else the same-K design with the smallest N above n (engine#1783: a sealed
    // set carries one K768 N3072 instance for both fc1 and the fused q|k|v, whose staged N is 2560). A wider design is
    // exact too: the weight's extra columns are zero-padded like the staging pad, each output column depends only on its
    // own weight column and the row's activation scale, and the extra columns are never copied out (dst_n = N).
    int role_for(int k, int n) const {
        int best = -1;
        for (size_t i = 0; i < designs_.size(); ++i) {
            if (designs_[i].K != k || designs_[i].N < n) continue;
            if (designs_[i].N == n) return int(i);
            if (best < 0 || designs_[i].N < designs_[size_t(best)].N) best = int(i);
        }
        return best;
    }

    Weight& weight(const std::string& key, const float* B, int64_t K, int64_t N, const ProjectionShape& st) {
        // Padded once per key: either the copy is still here (not yet in the target, or the target does not cache), or
        // the target holds it and the copy was dropped after its first launch.
        auto it = w_.find(key);
        if (it != w_.end()) return it->second;
        Weight w;
        w.wkey = next_wkey_++;
        w.padded.assign(size_t(st.k) * size_t(st.n), 0.0f);
        for (int64_t k = 0; k < K; ++k)
            std::copy(B + size_t(k) * size_t(N), B + size_t(k) * size_t(N) + size_t(N),
                      w.padded.begin() + size_t(k) * size_t(st.n));
        return w_.emplace(key, std::move(w)).first->second;
    }

    ::npu::target::NpuTarget& t_;
    std::vector<NpuAudioDesign> designs_;
    bool caches_ = true;
    std::map<std::string, Weight> w_;
    int next_wkey_ = 0;
    double device_ms_ = 0.0;
    int launches_ = 0;
    int products_ = 0;
    std::map<int, int> per_role_;
};

}  // namespace onebit::audio
