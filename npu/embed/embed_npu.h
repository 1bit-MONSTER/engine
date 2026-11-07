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

// The embedding encoder's ternary linears, run on the XDNA2 through the engine's own int8 GEMM (engine#1311).
//
// The container already holds the encoder as int8: ternary codes in {-1,0,+1} with one f32 scale per output
// row. The array executes int8*int8 -> int32, which is exactly the product the reference forms, so the device
// path is not a re-quantisation: the codes go to the weight buffer as they are, the int8 activation the
// reference's `quant_act` produced goes to the activation buffer as it is, and the scales are applied to the
// int32 sums on the host. `gemm_q8_raw` (npu/ort/onebit_npu_gemm.h) is that call: it writes both buffers
// verbatim and hands back exact accumulators, and its own probe checks them against the host's integer sum.
//
// Two shapes of call, and the second one is a measured requirement rather than a convenience:
//   * widths 384 (q, k, v, o, fc1) are quantised per token, so one launch per linear;
//   * fc2's input is 1536 wide and the reference quantises it **per 256-channel block** -- replacing that with
//     one per-token scale moves the vectors enough to put 73 of 1,000 gate sentences below a 0.999 cosine
//     (measured), so fc2 runs as six K=256 launches and the per-block scales are applied to the partial sums.
//
// The design M is a property of the ELF (128 here), so a launch costs the same for a 6-token sentence as for a
// 128-token one; the ELFs are named for the engine's own convention and pointed at by ONEBIT_EP_NPU_ELFS:
//   128x384x384=<full_i8_Q_K384_N384.elf>,128x384x1536=<full_i8_GU_K384_N1536.elf>,128x256x384=<full_i8_D_K256_N384.elf>
#pragma once

#include "embed_model.h"
#include "onebit_npu_gemm.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace onebit::embed {

class NpuGemm {
public:
    // The hook for Model::set_gemm(). A device failure returns false and fills `why`, which makes the encoder
    // return an empty vector with that reason -- there is no silent host fallback: a caller that wants the host
    // dot must not install this hook (the probe's --host arm does exactly that). `status()` keeps the device's
    // own words for the report, and `used_device()` is false once a launch has failed, so a row cannot claim
    // the array when it did not run.
    Model::Gemm hook() {
        return [this](const Tern& t, const Quantised& q, std::vector<float>& x, int rows, std::string& why) {
            return run(t, q, x, rows, why);
        };
    }

    // Build every transposed weight block ONCE, before any request can run. A served route answers from
    // cpp-httplib's worker pool, so a lazily filled cache shared by those threads is a data race: the first
    // concurrent burst after start could insert into the map from two threads (UB) or launch with a block that
    // another thread had sized but not yet filled -- a silently wrong embedding, which is the one failure this
    // route must never have. Preparing here makes the map read-only for the life of the process; `m_` then
    // covers the counters below and keeps two launches from interleaving their host-side scaling.
    bool prepare(const Model& m, std::string& why) {
        std::lock_guard<std::mutex> g(m_);
        for (const Layer& L : m.layers()) {
            for (const Tern* t : {&L.q, &L.k, &L.v, &L.o, &L.fc1, &L.fc2}) {
                const int blocks = t->k > 256 && t->k % 256 == 0 ? t->k / 256 : 1;
                const int block = blocks == 1 ? t->k : 256;
                for (int b = 0; b < blocks; ++b) {
                    auto& dst = bt_[{t, b * block}];
                    dst.resize(size_t(block) * t->n);
                    for (int n = 0; n < t->n; ++n)
                        for (int i = 0; i < block; ++i)
                            dst[size_t(i) * t->n + n] = t->code[size_t(n) * t->k + b * block + i];
                }
            }
        }
        prepared_ = !bt_.empty();
        if (!prepared_) why = "npu gemm: the model has no ternary linears to pack";
        return prepared_;
    }

    bool used_device() const { return launches() > 0 && !failed(); }
    std::string status() const {
        std::lock_guard<std::mutex> g(m_);
        return status_;
    }
    long launches() const {
        std::lock_guard<std::mutex> g(m_);
        return launches_;
    }
    bool failed() const {
        std::lock_guard<std::mutex> g(m_);
        return failed_;
    }

private:
    // codes transposed to the kernel's [k][n] row-major, one entry per (weight, k-block). Filled by prepare()
    // only; a miss here means prepare() was skipped, which is a programming error and is reported as one.
    const std::vector<int8_t>* weight(const Tern& t, int k0) const {
        const auto it = bt_.find({&t, k0});
        return it == bt_.end() ? nullptr : &it->second;
    }

    bool run(const Tern& t, const Quantised& q, std::vector<float>& x, int rows, std::string& why) {
        std::lock_guard<std::mutex> g(m_);
        if (!prepared_) {
            why = "npu gemm: prepare() was not called before the first request";
            return false;
        }
        if (rows <= 0 || rows > 128) {  // the design's M; a longer prompt is refused, never truncated
            why = "npu gemm: " + std::to_string(rows) + " rows does not fit the 128-row design";
            return false;
        }
        std::vector<int32_t> c(size_t(rows) * t.n);
        x.assign(size_t(rows) * t.n, 0.0f);
        const int blocks = q.width / q.block;
        std::vector<int8_t> a_block;
        for (int b = 0; b < blocks; ++b) {
            const int8_t* a = nullptr;
            if (blocks == 1) {
                a = q.q.data();
            } else {
                // gather this 256-channel block's columns into contiguous rows: the GEMM takes one stride
                a_block.resize(size_t(rows) * q.block);
                for (int r = 0; r < rows; ++r)
                    std::copy_n(q.q.data() + size_t(r) * q.width + size_t(b) * q.block, q.block,
                                a_block.data() + size_t(r) * q.block);
                a = a_block.data();
            }
            const std::vector<int8_t>* w = weight(t, b * q.block);
            if (!w) {
                why = "npu gemm: no packed weight for this linear (prepare() did not cover it)";
                return false;
            }
            if (!onebit::npu::gemm_q8_raw(a, rows, q.block, t.n, w->data(), c.data(), &why)) {
                failed_ = true;
                status_ = why;
                return false;
            }
            launches_++;
            // int32 sums -> floats with the activation scale of this block/row and the weight's row scale
            for (int r = 0; r < rows; ++r) {
                const float as = q.scale[size_t(r) * size_t(blocks) + size_t(b)];
                for (int n = 0; n < t.n; ++n)
                    x[size_t(r) * t.n + n] += float(c[size_t(r) * t.n + n]) * as * t.scale[size_t(n)];
            }
        }
        for (int r = 0; r < rows; ++r)
            for (int n = 0; n < t.n; ++n) x[size_t(r) * t.n + n] += t.bias[size_t(n)];
        return true;
    }

    std::map<std::pair<const Tern*, int>, std::vector<int8_t>> bt_;
    bool prepared_ = false;
    mutable std::mutex m_;   // counters, status, and the launch sequence (bt_ is read-only after prepare)
    long launches_ = 0;
    bool failed_ = false;
    std::string status_;
};

}  // namespace onebit::embed
