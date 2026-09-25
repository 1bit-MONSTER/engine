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
#include "lax_elf.h"

#include "lax_pack.h"

#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace onebit::npu::lax {

std::string pdi_path(const std::string& kind_dir) {
    for (const char* f : {"/main.pdi", "/final.prj/main.pdi"})
        if (std::filesystem::exists(kind_dir + f)) return kind_dir + f;
    throw std::runtime_error(kind_dir + ": no main.pdi (scripts/build-lax.sh exports it)");
}

Bytes unfold(Bytes e) {
    constexpr uint32_t kFold = 0x80000000u;
    constexpr uint32_t kTranslatedArgs = 5;  // the classic firmware translates these itself
    const ElfSection ct = elf_section(e, ".ctrltext"), rela = elf_section(e, ".rela.dyn");
    auto word = [&](size_t o) {
        uint32_t v;
        std::memcpy(&v, e.data() + o, 4);
        return v;
    };
    auto clear = [&](size_t o) {
        const uint32_t v = word(o) & ~kFold;
        std::memcpy(e.data() + o, &v, 4);
    };
    for (size_t i = 4; 4 * i + 4 <= ct.size;) {
        const size_t o = ct.offset + 4 * i;
        const uint32_t op = word(o);
        if (op == 0x81 && 4 * (i + 11) < ct.size && (word(o + 40) & kFold)) {
            if (word(o + 32) < kTranslatedArgs)
                throw std::runtime_error("unfold: a DDR_PATCH on argument " + std::to_string(word(o + 32)) +
                                         " carries bit 31, which is not the fold");
            clear(o + 40);
        }
        i += op_words(op);
    }
    for (size_t k = 0; k < rela.size / 12; ++k)
        if (word(rela.offset + 12 * k + 8) & kFold) clear(rela.offset + 12 * k + 8);
    refresh_uid(e);
    return e;
}

ElfInputs ElfInputs::read(const std::string& dir) {
    ElfInputs in;
    in.lax_l = unfold(read_file(dir + "/lax_l/insts.elf"));
    in.lax_a = read_file(dir + "/lax_a/insts.elf");  // unfolded per position, after the sites are written
    in.pdi = read_file(pdi_path(dir + "/lax_l"));
    in.ln = unfold(read_file(dir + "/ln/insts.elf"));
    in.ln_pdi = read_file(pdi_path(dir + "/ln"));
    in.lm = unfold(read_file(dir + "/lm_head_q8/insts.elf"));
    in.lm_pdi = read_file(pdi_path(dir + "/lm_head_q8"));
    in.sites = elf_position_sites(in.lax_a, kKvRow);
    return in;
}

Bytes init_elf(const ElfInputs& in) { return assemble_full_elf(in.lax_l, in.pdi, kInitKernel, PdiMode::kInitOnly); }
Bytes linear_elf(const ElfInputs& in) { return assemble_full_elf(in.lax_l, in.pdi, kLinearKernel, PdiMode::kNone); }

std::string full_kernel_name(size_t pos) { return "axf" + std::to_string(pos); }

Bytes full_elf(const ElfInputs& in, size_t pos, bool with_pdi) {
    if (pos >= size_t(kMaxContext)) throw std::runtime_error("position " + std::to_string(pos) + " past the context");
    Bytes e = in.lax_a;
    apply_elf_position(e, in.sites, pos, kKvRow, kPtabRow);
    return assemble_full_elf(unfold(std::move(e)), with_pdi ? in.pdi : Bytes{}, full_kernel_name(pos), PdiMode::kNone);
}

Bytes norm_elf(const ElfInputs& in) { return assemble_full_elf(in.ln, in.ln_pdi, kNormKernel, PdiMode::kLoad); }
Bytes head_elf(const ElfInputs& in) { return assemble_full_elf(in.lm, in.lm_pdi, kHeadKernel, PdiMode::kLoad); }

}  // namespace onebit::npu::lax
