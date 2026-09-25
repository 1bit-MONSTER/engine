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

// How the lax decode (npu/lax.h) gets its four kernels onto the device. The decoder only
// asks for runs with their non-buffer arguments set, the context the layer runlist is
// bound to, and a way to set the full-attention position; everything about xclbins,
// instruction buffers or ELFs stays behind this interface.
//
// Two sets:
//   elf_kernels      full ELFs (docs/npu-lax.md, "Full ELFs"): PDI + control code,
//                    assembled in memory by npu/full_elf; no xclbin. The default.
//   classic_kernels  xclbin + insts.bin, the path the design was brought up on; kept
//                    selectable for A/B.
#pragma once

#include <xrt/experimental/xrt_kernel.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <memory>
#include <optional>
#include <string>

namespace onebit::npu::lax {

enum class Stage { LayerLinear, LayerFull, Norm, Head };

class KernelSet {
public:
    virtual ~KernelSet() = default;
    // The context both layer kernels live in: one runlist carries all 40 layers.
    virtual xrt::hw_context& layer_context() = 0;
    // A run of that stage with every argument before the buffers set. A LayerFull run is
    // for the position set last.
    virtual xrt::run make_run(Stage s) = 0;
    // The argument index of the stage's first buffer.
    virtual int first_buffer_arg() const = 0;
    // Point the full-attention layers at cache position pos (their KV window, new row and
    // RoPE record); called before every token's runlist.
    virtual void set_position(size_t pos) = 0;
    // True: the position is baked into each LayerFull run (a full ELF config per
    // position); set_position only affects runs made after it, so the decoder makes new
    // ones, and may do so while an earlier runlist executes. False: set_position rewrites
    // what the existing runs execute (classic: the shared instruction buffer).
    virtual bool position_in_run() const = 0;
    // A run that must head every token's runlist, or none. Full ELFs: the init config's
    // load_pdi, because the norm and the lm head run in their own contexts, and a full-ELF
    // context does not get its array configuration back after another context ran.
    virtual std::optional<xrt::run> head_run() = 0;
    virtual std::string describe() const = 0;
};

// The classic path: dir/{lax_l,lax_a,ln,lm_head_q8}/{final.xclbin,insts.bin}, as
// scripts/build-lax.sh builds them. lax_l and lax_a share one xclbin (one context); the
// instruction words go in through an instruction buffer at argument 1.
std::unique_ptr<KernelSet> classic_kernels(xrt::device& dev, const std::string& dir);

// The full-ELF path: dir/{lax_l,lax_a,ln,lm_head_q8}/{insts.elf,main.pdi}, as
// scripts/build-lax.sh exports them (docs/npu-lax.md, "Kernel directory"). The layer
// context is created from an init config (lax_l's PDI, load_pdi only); the linear layer
// and each full-attention position reached are configs added to it; the norm and the lm
// head are stand-alone contexts. Buffers from argument 0.
std::unique_ptr<KernelSet> elf_kernels(xrt::device& dev, const std::string& dir);

enum class Transport { Elf, Classic };
Transport parse_transport(const std::string& name);  // "elf" or "classic"
std::unique_ptr<KernelSet> make_kernels(xrt::device& dev, const std::string& dir, Transport t);

}  // namespace onebit::npu::lax
