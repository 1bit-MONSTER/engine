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

// The lax decode's full ELFs (docs/npu-lax.md, "Full ELFs"), assembled in memory from the
// design's instruction ELFs and PDIs by npu/full_elf. Pure host code, no XRT: the kernel
// set (npu/lax_elf_kernels.cpp) opens what this builds, and the tests compare it byte for
// byte with the ELFs the reference harness ran.
//
// A kernel directory (scripts/build-lax.sh's <prefix>/kernels) holds per kind
// <kind>/insts.elf and <kind>/main.pdi (or an aiecc build tree's final.prj/main.pdi):
//   lax_l       the merged whole-layer design's linear-attention text; its PDI configures
//               the array for both layer texts
//   lax_a       the full-attention text (its PDI differs from lax_l's only in the build ID)
//   ln          the final RMSNorm, stand-alone
//   lm_head_q8  the q8 lm head, stand-alone
//
// Kernels (the names XRT looks them up by):
//   laxinit     lax_l's PDI and only load_pdi, one unreferenced buffer argument; the layer
//               context is created from it, and it heads every token's runlist
//   lxf         lax_l's text, no load_pdi (a config added to the layer context)
//   axf<pos>    lax_a's text written for position pos, no load_pdi (and no PDI)
//   ln, lm      load_pdi + text, each its own context
// Instruction ELFs from the default (folded) aiecc build carry +0x80000000 in the
// addends and DDR_PATCH words of arguments >= 5 (mlir-aie's kDDRAIEAddrOffset, for the
// classic firmware, which translates only the first 5 arguments); XRT's ELF flow adds
// the aperture offset to every argument itself, so the fold would land twice. unfold()
// clears it (assemble_full_elf clears bit 31 as well) and refreshes the UID note to the
// md5 of the control text XRT will run, as the reference harness does.
#pragma once

#include "full_elf.h"
#include "lax_stream.h"

#include <string>
#include <vector>

namespace onebit::npu::lax {

struct ElfInputs {
    Bytes lax_l, lax_a, pdi;      // pdi: lax_l's
    Bytes ln, ln_pdi, lm, lm_pdi;
    std::vector<ElfPosSite> sites;  // lax_a's position sites

    // Reads dir/<kind>/insts.elf and the PDIs; derives the position sites.
    static ElfInputs read(const std::string& dir);
};

// The instruction ELF without the fold, its UID note refreshed. Throws on a DDR_PATCH word
// with bit 31 on an argument below 5 (not a fold).
Bytes unfold(Bytes insts_elf);

// A kind's PDI path: <kind_dir>/main.pdi, else <kind_dir>/final.prj/main.pdi.
std::string pdi_path(const std::string& kind_dir);

Bytes init_elf(const ElfInputs& in);
Bytes linear_elf(const ElfInputs& in);
// with_pdi false: an empty .pdi.1. A position config has no load_pdi, so XRT never uploads
// its PDI, but it keeps every added ELF in memory: with lax_l's 257 KiB PDI in each, the
// configs would hold about 1 GB at 4096 positions; without it they hold a few MB.
Bytes full_elf(const ElfInputs& in, size_t pos, bool with_pdi = false);
Bytes norm_elf(const ElfInputs& in);
Bytes head_elf(const ElfInputs& in);
std::string full_kernel_name(size_t pos);

inline constexpr const char* kInitKernel = "laxinit";
inline constexpr const char* kLinearKernel = "lxf";
inline constexpr const char* kNormKernel = "ln";
inline constexpr const char* kHeadKernel = "lm";

}  // namespace onebit::npu::lax
