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

// Full ELFs for the NPU fast lane (docs/npu.md), built in memory at load time.
//
// Inputs are the model's kernel artifacts: the layer kernel's instruction ELF for
// context length 1 (plus the ones for lengths 2 and 17, which pin down how the
// control code depends on the length), the lm-head instruction ELF and the design
// PDI. Outputs are ELFs XRT opens directly: xrt::hw_context(dev, xrt::elf) and
// hw_context::add_config, with no xclbin anywhere.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace onebit::npu {

using Bytes = std::vector<uint8_t>;

Bytes read_file(const std::string& path);

// A section of an ELF32 little-endian file, by name: its byte offset and size in the
// file. Throws if the file is not an ELF32 LE file or has no such section.
struct ElfSection {
    uint32_t offset, size;
};
ElfSection elf_section(const Bytes& elf, const std::string& name);

// How the layer kernel's control code depends on the context length N.
// Every differing word follows one of two rules, found by comparing the
// captured contexts 1, 2 and 17:
//   linear  v(N) = v(1) + (N - 1) * step   (KV row offsets, their relocation
//                                           addends, and words equal to N)
//   block   v(N) = v(1) * ceil(N / 16)     (KV lengths in 16-row blocks)
// The UID note is md5 of .ctrltext and is recomputed, not mapped.
struct ContextMap {
    struct Word {
        uint32_t offset;  // byte offset in the instruction ELF
        uint32_t base;    // value at N = 1
        uint32_t step;    // linear step; 0 for a block word
    };
    std::vector<Word> words;
    static constexpr int kBlock = 16;

    // Throws if a differing word follows neither rule.
    static ContextMap derive(const Bytes& ctx1, const Bytes& ctx2, const Bytes& ctx17);
};

// The instruction ELF for context length n (n >= 1), byte-identical to the
// captured layer_ctx<n>.elf.
Bytes derive_context(const Bytes& ctx1, const ContextMap& map, int n);

// Rewrite an instruction ELF's UID note to the md5 of its .ctrltext, as aiebu-asm writes
// it, after the control code was edited.
void refresh_uid(Bytes& instruction_elf);

enum class PdiMode {
    kLoad,     // control code starts with load_pdi (a stand-alone kernel)
    kNone,     // no load_pdi: runs on an array an init config already configured
    kInitOnly  // only load_pdi: the init config; one unreferenced buffer argument
};

// Assemble a full ELF (PDI + control code) from an instruction ELF, following the
// layout aiecc --get-full-elf produces. Argument symbols move from the xclbin
// convention (first buffer is argument 3) to the full-ELF one (argument 0).
// kernel_name is the name XRT looks the kernel up by; config is the column count.
Bytes assemble_full_elf(const Bytes& instruction_elf, const Bytes& pdi, const std::string& kernel_name,
                        PdiMode mode, uint32_t config = 8);

// RFC 1321 MD5; exposed for the UID note and its test.
std::array<uint8_t, 16> md5(std::span<const uint8_t> data);

}  // namespace onebit::npu
