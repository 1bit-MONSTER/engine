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

// The full-ELF kernel set of the lax decode (npu/lax_kernels.h, docs/npu-lax.md): every
// kernel is PDI + control code, assembled in memory (npu/lax_elf.h) and opened through
// XRT's ELF flow. No xclbin is read.
//
// Layer context: created from laxinit; lxf and one axf<pos> per full-attention position
// reached are configs added to it. Every token's runlist starts with the laxinit run: ln
// and lm_head run in their own stand-alone contexts, and after them the layer context's
// array configuration is gone (an init-once runlist timed out at token 1).
//
// The position lives in the relocation addends on the ELF path, so each position is its
// own config (npu/lax_stream.h, elf_position_sites), added the first time the decode
// reaches it and kept for later sequences.
#include "lax_elf.h"
#include "lax_kernels.h"
#include "lax_pack.h"

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>

#include <chrono>
#include <cstdio>
#include <map>
#include <stdexcept>

namespace onebit::npu::lax {

namespace {

xrt::elf as_elf(const Bytes& b) { return xrt::elf(b.data(), b.size()); }

class FullElf final : public KernelSet {
public:
    FullElf(xrt::device& dev, const std::string& dir) : dir_(dir), in_(ElfInputs::read(dir)) {
        layer_ctx_ = xrt::hw_context(dev, as_elf(init_elf(in_)));
        init_ = xrt::ext::kernel(layer_ctx_, kInitKernel);
        layer_ctx_.add_config(as_elf(linear_elf(in_)));
        lxf_ = xrt::ext::kernel(layer_ctx_, kLinearKernel);
        ln_ctx_ = xrt::hw_context(dev, as_elf(norm_elf(in_)));
        ln_ = xrt::ext::kernel(ln_ctx_, kNormKernel);
        lm_ctx_ = xrt::hw_context(dev, as_elf(head_elf(in_)));
        lm_ = xrt::ext::kernel(lm_ctx_, kHeadKernel);
        // The stand-alone kernels are built; only the full-attention text is needed again.
        in_.ln.clear(), in_.ln_pdi.clear(), in_.lm.clear(), in_.lm_pdi.clear(), in_.lax_l.clear();

        dummy_ = xrt::ext::bo(dev, 4096);
        head_ = xrt::run(init_);
        head_.set_arg(0, dummy_);
    }

    xrt::hw_context& layer_context() override { return layer_ctx_; }

    xrt::run make_run(Stage s) override {
        switch (s) {
        case Stage::LayerLinear: return xrt::run(lxf_);
        case Stage::LayerFull: return xrt::run(axf_kernel(pos_));
        case Stage::Norm: return xrt::run(ln_);
        case Stage::Head: return xrt::run(lm_);
        }
        throw std::runtime_error("unknown stage");
    }

    int first_buffer_arg() const override { return 0; }

    void set_position(size_t pos) override {
        axf_kernel(pos);
        pos_ = pos;
    }

    bool position_in_run() const override { return true; }

    std::optional<xrt::run> head_run() override { return head_; }

    std::string describe() const override {
        char b[160];
        std::snprintf(b, sizeof b, "; %zu position configs, %.2f ms each to add", axf_.size(),
                      axf_.empty() ? 0.0 : config_ms_ / double(axf_.size()));
        return "full ELFs from " + dir_ + b;
    }

private:
    // The full-attention config for position pos, added to the layer context on first use.
    xrt::kernel& axf_kernel(size_t pos) {
        if (auto it = axf_.find(pos); it != axf_.end()) return it->second;
        const auto t0 = std::chrono::steady_clock::now();
        layer_ctx_.add_config(as_elf(full_elf(in_, pos)));
        auto& k = axf_.emplace(pos, xrt::ext::kernel(layer_ctx_, full_kernel_name(pos))).first->second;
        config_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return k;
    }

    std::string dir_;
    ElfInputs in_;
    xrt::hw_context layer_ctx_, ln_ctx_, lm_ctx_;
    xrt::kernel init_, lxf_, ln_, lm_;
    std::map<size_t, xrt::kernel> axf_;
    xrt::bo dummy_;
    xrt::run head_;
    size_t pos_ = 0;
    double config_ms_ = 0;
};

}  // namespace

std::unique_ptr<KernelSet> elf_kernels(xrt::device& dev, const std::string& dir) {
    return std::make_unique<FullElf>(dev, dir);
}

Transport parse_transport(const std::string& name) {
    if (name == "elf") return Transport::Elf;
    if (name == "classic") return Transport::Classic;
    throw std::runtime_error("kernel transport '" + name + "': elf or classic");
}

std::unique_ptr<KernelSet> make_kernels(xrt::device& dev, const std::string& dir, Transport t) {
    return t == Transport::Elf ? elf_kernels(dev, dir) : classic_kernels(dev, dir);
}

}  // namespace onebit::npu::lax
