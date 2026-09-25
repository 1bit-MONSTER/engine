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

#include "lane.h"

#include "full_elf.h"
#include "pack.h"

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>

namespace onebit::npu {

namespace {

constexpr size_t kSmallBo = 1 << 20;  // act, logits, final norm, i5, i6

std::unique_ptr<xrt::ext::bo> make_bo(xrt::device& dev, size_t bytes) {
    auto bo = std::make_unique<xrt::ext::bo>(dev, bytes);
    std::memset(bo->map(), 0, bytes);
    return bo;
}

}  // namespace

struct Lane::Impl {
    const Model& model;
    const Dims& d;
    xrt::device dev{0};
    std::unique_ptr<xrt::hw_context> ctx;
    std::unique_ptr<xrt::ext::kernel> lmhead;
    std::map<int, std::unique_ptr<xrt::ext::kernel>> layer_kernels;

    Bytes ctx1, pdi, head;
    ContextMap map;

    std::vector<std::unique_ptr<xrt::ext::bo>> weights, i5, kv;
    std::vector<std::unique_ptr<xrt::ext::bo>> i6[2];  // per runlist slot: RoPE rows differ
    std::unique_ptr<xrt::ext::bo> act, logits, fnorm, lmhead_w;

    struct Slot {
        std::vector<xrt::run> runs;
        std::unique_ptr<xrt::runlist> rl;
    } slots[2];

    Impl(const Model& m, const std::string& dir) : model(m), d(m.dims()) {
        ctx1 = read_file(dir + "/layer_ctx1.elf");
        pdi = read_file(dir + "/layer.pdi");
        map = ContextMap::derive(ctx1, read_file(dir + "/layer_ctx2.elf"), read_file(dir + "/layer_ctx17.elf"));

        // The lm-head ELF is stored now; the hw_context (init load_pdi config
        // + lm-head config) is created fresh in begin() so a context never sits
        // idle across the NPU's runtime-suspend.
        head = assemble_full_elf(read_file(dir + "/lmhead.elf"), pdi, "flhead", PdiMode::kNone);

        // KV: 4 regions (K and V, two head halves) of kMaxContext rows each.
        const size_t kv_bytes = std::max<size_t>(size_t(d.kv_heads / 2) * d.head_dim * 2 * kMaxContext * 4, 32u << 20);
        const size_t w_bytes = layer_weight_bytes(model);
        for (int L = 0; L < d.layers; ++L) {
            weights.push_back(make_bo(dev, w_bytes));
            pack_layer_weights(model, L, static_cast<uint8_t*>(weights.back()->map()));
            i5.push_back(make_bo(dev, kSmallBo));
            fill_i5(model, L, static_cast<uint8_t*>(i5.back()->map()));
            for (auto& s : i6) {
                s.push_back(make_bo(dev, kSmallBo));
                fill_i6(model, L, static_cast<uint8_t*>(s.back()->map()));
            }
            kv.push_back(make_bo(dev, kv_bytes));
        }
        act = make_bo(dev, kSmallBo);
        logits = make_bo(dev, kSmallBo);
        fnorm = make_bo(dev, kSmallBo);
        std::memcpy(fnorm->map(), model.tensor("model.norm.weight").data, size_t(d.hidden) * 2);
        lmhead_w = make_bo(dev, lmhead_weight_bytes(model));
        pack_lmhead_weights(model, static_cast<uint8_t*>(lmhead_w->map()));

        for (auto* v : {&weights, &i5, &kv, &i6[0], &i6[1]})
            for (auto& bo : *v) bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        for (auto* bo : {&act, &logits, &fnorm, &lmhead_w}) (*bo)->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    xrt::ext::kernel& layer_kernel(int n) {
        auto it = layer_kernels.find(n);
        if (it != layer_kernels.end()) return *it->second;
        if (n < 1 || n > kMaxContext) throw std::runtime_error("context length " + std::to_string(n) + " out of range");
        char name[16];
        std::snprintf(name, sizeof name, "fl%05d", n);
        const Bytes e = assemble_full_elf(derive_context(ctx1, map, n), pdi, name, PdiMode::kNone);
        ctx->add_config(xrt::elf(reinterpret_cast<const char*>(e.data()), e.size()));
        return *layer_kernels.emplace(n, std::make_unique<xrt::ext::kernel>(*ctx, name)).first->second;
    }

    void prepare(int slot, int n) {
        auto& rope = i6[slot];
        for (auto& bo : rope) {
            fill_rope(static_cast<uint16_t*>(bo->map()), n - 1, d.rope_theta);
            bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, 256, 0);
        }
        xrt::ext::kernel& k = layer_kernel(n);
        Slot& s = slots[slot];
        s.rl.reset();
        s.runs.clear();
        s.runs.reserve(size_t(d.layers) + 1);
        s.rl = std::make_unique<xrt::runlist>(*ctx);
        for (int L = 0; L < d.layers; ++L) {
            xrt::run& r = s.runs.emplace_back(k);
            r.set_arg(0, *act);
            r.set_arg(1, *weights[L]);
            r.set_arg(2, *i5[L]);
            r.set_arg(3, *rope[L]);
            r.set_arg(4, *kv[L]);
            s.rl->add(r);
        }
        xrt::run& r = s.runs.emplace_back(*lmhead);
        r.set_arg(0, *logits);
        r.set_arg(1, *lmhead_w);
        r.set_arg(2, *act);
        r.set_arg(3, *fnorm);
        s.rl->add(r);
    }

    void launch(int slot, int token) {
        if (token < 0 || token >= d.vocab) throw std::runtime_error("token id out of range");
        const size_t row = size_t(d.hidden) * 2;
        std::memcpy(act->map(), model.tensor("model.embed_tokens.weight").data + size_t(token) * row, row);
        act->sync(XCL_BO_SYNC_BO_TO_DEVICE);  // the whole buffer, as the reference runtime does
        slots[slot].rl->execute();
    }

    const uint16_t* logits_bf16() {
        logits->sync(XCL_BO_SYNC_BO_FROM_DEVICE, size_t(d.vocab) * 2, 0);
        return static_cast<const uint16_t*>(logits->map());
    }

    // Create a fresh hw_context carrying the init (load_pdi) config and the
    // lm-head config. Called at the start of each generate(): a context is
    // created, used for one generation, then destroyed while warm in end(), so
    // it never sits idle across the NPU's runtime-suspend (which leaves a
    // long-idle context stale and its next runlist execute() fails with
    // ERT_CMD_STATE_TIMEOUT).
    void begin() {
        const Bytes init = assemble_full_elf(ctx1, pdi, "flinit", PdiMode::kInitOnly);
        ctx = std::make_unique<xrt::hw_context>(dev, xrt::elf(reinterpret_cast<const char*>(init.data()), init.size()));
        {
            xrt::ext::kernel k(*ctx, "flinit");
            xrt::ext::bo unused(dev, 4096);
            xrt::run r(k);
            r.set_arg(0, unused);
            r.start();
            if (r.wait(std::chrono::milliseconds(5000)) != ERT_CMD_STATE_COMPLETED)
                throw std::runtime_error("NPU init run did not complete");
        }
        ctx->add_config(xrt::elf(reinterpret_cast<const char*>(head.data()), head.size()));
        lmhead = std::make_unique<xrt::ext::kernel>(*ctx, "flhead");
        layer_kernels.clear();
        slots[0].rl.reset(); slots[0].runs.clear();
        slots[1].rl.reset(); slots[1].runs.clear();
    }

    // Destroy the hw_context while it is warm (every run completed), so its
    // destructor never waits on a stuck command. Drop the runlists, runs,
    // layer kernels and lm-head kernel before the context they reference.
    void end() {
        slots[0].rl.reset(); slots[0].runs.clear();
        slots[1].rl.reset(); slots[1].runs.clear();
        layer_kernels.clear();
        lmhead.reset();
        ctx.reset();
    }
};

Lane::Lane(const Model& model, const std::string& kernel_dir) : p_(std::make_unique<Impl>(model, kernel_dir)) {}
Lane::~Lane() = default;

void Lane::prepare(int slot, int ctx) { p_->prepare(slot, ctx); }
void Lane::launch(int slot, int token) { p_->launch(slot, token); }
void Lane::wait(int slot) { p_->slots[slot].rl->wait(); }
void Lane::begin() { p_->begin(); }
void Lane::end() { p_->end(); }

void Lane::step(int token, int ctx) {
    prepare(0, ctx);
    launch(0, token);
    wait(0);
}

int Lane::argmax() {
    // Sign-magnitude bf16 to a monotonic unsigned key: flip the sign bit of
    // positives and every bit of negatives. Ties go to the lowest id.
    const uint16_t* lg = p_->logits_bf16();
    int best = 0;
    uint32_t best_key = 0;
    for (int i = 0; i < p_->d.vocab; ++i) {
        const uint32_t key = lg[i] ^ ((lg[i] & 0x8000u) ? 0xFFFFu : 0x8000u);
        if (i == 0 || key > best_key) {
            best_key = key;
            best = i;
        }
    }
    return best;
}

void Lane::logits(std::vector<float>& out) {
    const uint16_t* lg = p_->logits_bf16();
    out.resize(size_t(p_->d.vocab));
    for (int i = 0; i < p_->d.vocab; ++i) out[size_t(i)] = bf16_to_f32(lg[i]);
}

}  // namespace onebit::npu
