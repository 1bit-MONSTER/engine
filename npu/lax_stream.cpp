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

#include "lax_stream.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace onebit::npu::lax {

size_t op_words(uint32_t w) {
    switch (w) {
    case 0x00: return 6;
    case 0x01: return 12;
    case 0x03: return 7;
    case 0x80: return 4;
    case 0x81: return 12;
    default: return 1;
    }
}

std::vector<PosPatch> position_patches(std::span<const uint32_t> w, size_t kv_row) {
    std::vector<PosPatch> out;
    size_t last_bd = 0;
    bool have_bd = false;
    for (size_t i = 4; i < w.size(); i += op_words(w[i])) {
        if (w[i] == 0x01) {
            last_bd = i;
            have_bd = true;
        }
        if (w[i] != 0x81 || i + 11 >= w.size()) continue;
        const uint32_t reg = w[i + 6], arg = w[i + 8];
        const uint32_t flags = w[i + 10] & 0x80000000u, off = w[i + 10] & 0x7FFFFFFFu;
        if (arg == 3 && off == 0) {
            // The fill's length is in the BD written just before: its register + 4 is the
            // register this patch rewrites (the BD's address word).
            if (!have_bd || last_bd + 4 >= w.size() || w[last_bd + 2] + 4 != reg)
                throw std::runtime_error("lax_a stream: no BD write before the KV window fill");
            out.push_back({last_bd + 4, PosPatch::WindowLength, 0});
            out.push_back({i + 10, PosPatch::WindowOffset, flags});
        } else if (arg == 3 && off == kv_row) {
            out.push_back({i + 10, PosPatch::RowDrain, flags});
        } else if (arg == 3) {
            throw std::runtime_error("lax_a stream: unexpected KV transfer at offset " + std::to_string(off));
        } else if (arg == 5) {
            out.push_back({i + 10, PosPatch::Record, flags});
        }
    }
    for (int k = 0; k < 4; ++k) {
        int n = 0;
        for (const auto& p : out) n += p.kind == k;
        if (n != 1)
            throw std::runtime_error("lax_a stream: " + std::to_string(n) + " position patches of kind " + std::to_string(k) +
                                     ", expected 1");
    }
    return out;
}

uint32_t position_value(PosPatch::Kind kind, size_t pos, size_t kv_row, size_t ptab_row) {
    const size_t nf = pos ? pos : 1;  // position 0 streams one dummy row the core masks
    switch (kind) {
    case PosPatch::WindowLength: return uint32_t(nf * kv_row / 4);
    case PosPatch::RowDrain: return uint32_t(pos * kv_row);
    case PosPatch::Record: return uint32_t(pos * ptab_row);
    case PosPatch::WindowOffset: return 0;
    }
    throw std::runtime_error("unknown position patch kind");
}

void apply_position(std::span<uint32_t> w, const std::vector<PosPatch>& patches, size_t pos, size_t kv_row,
                    size_t ptab_row) {
    for (const auto& p : patches) {
        if (p.word >= w.size()) throw std::runtime_error("position patch past the stream");
        w[p.word] = position_value(p.kind, pos, kv_row, ptab_row) | p.flags;
    }
}

namespace {

uint32_t le32(const Bytes& b, size_t o) {
    if (o + 4 > b.size()) throw std::runtime_error("lax_a ELF: read past the end");
    uint32_t v;
    std::memcpy(&v, b.data() + o, 4);
    return v;
}

}  // namespace

std::vector<ElfPosSite> elf_position_sites(const Bytes& e, size_t kv_row) {
    const ElfSection ct = elf_section(e, ".ctrltext"), rela = elf_section(e, ".rela.dyn"),
                     dsym = elf_section(e, ".dynsym"), dstr = elf_section(e, ".dynstr");
    if (ct.size % 4) throw std::runtime_error("lax_a ELF: .ctrltext is not whole words");
    std::vector<uint32_t> w(ct.size / 4);
    std::memcpy(w.data(), e.data() + ct.offset, ct.size);

    // The BD blockwrite each DDR_PATCH op follows (the last one before it).
    std::vector<size_t> bd_before(w.size(), SIZE_MAX);
    for (size_t i = 4, last = SIZE_MAX; i < w.size(); i += op_words(w[i])) {
        if (w[i] == 0x81) bd_before[i] = last;
        if (w[i] == 0x01) last = i;
    }
    auto symbol_name = [&](uint32_t sym) {
        if (uint64_t(sym + 1) * 16 > dsym.size) throw std::runtime_error("lax_a ELF: relocation symbol out of range");
        const size_t o = dstr.offset + le32(e, dsym.offset + 16u * sym);
        const auto* b = reinterpret_cast<const char*>(e.data());
        return std::string(b + o, strnlen(b + o, e.size() - o));
    };

    std::vector<ElfPosSite> out;
    for (const PosPatch& p : position_patches(w, kv_row)) {
        out.push_back({ct.offset + 4 * p.word, p.kind, p.flags, ElfPosSite::CtrlWord, p.word});
        if (p.kind == PosPatch::WindowLength) continue;
        // The DDR_PATCH op holding this offset word, and the BD it rewrites.
        const size_t op = p.word - 10, bd = bd_before[op];
        if (bd == SIZE_MAX || w[bd + 2] + 4 != w[op + 6])
            throw std::runtime_error("lax_a ELF: no BD write before the patch at word " + std::to_string(p.word));
        const uint32_t bd_offset = uint32_t(4 * (bd + 4));  // the BD's first word, in .ctrltext
        const uint32_t arg = w[op + 8];
        int found = 0;
        for (size_t k = 0; k < rela.size / 12; ++k) {
            const size_t ro = rela.offset + 12 * k;
            if (le32(e, ro) != bd_offset) continue;
            if (symbol_name(le32(e, ro + 4) >> 8) != std::to_string(arg + 3))  // xclbin numbering
                throw std::runtime_error("lax_a ELF: the relocation at word " + std::to_string(bd + 4) +
                                         " names another argument than its DDR_PATCH");
            if ((le32(e, ro + 8) & 0x7FFFFFFFu) != (w[p.word] & 0x7FFFFFFFu))
                throw std::runtime_error("lax_a ELF: addend and DDR_PATCH word disagree at word " + std::to_string(p.word));
            out.push_back({ro + 8, p.kind, le32(e, ro + 8) & 0x80000000u, ElfPosSite::Addend, k});
            ++found;
        }
        if (found != 1)
            throw std::runtime_error("lax_a ELF: " + std::to_string(found) + " relocations on the BD at word " +
                                     std::to_string(bd + 4) + ", expected 1");
    }
    return out;
}

void apply_elf_position(Bytes& e, const std::vector<ElfPosSite>& sites, size_t pos, size_t kv_row, size_t ptab_row) {
    for (const auto& s : sites) {
        if (s.offset + 4 > e.size()) throw std::runtime_error("position site past the ELF");
        const uint32_t v = position_value(s.kind, pos, kv_row, ptab_row) | s.flags;
        std::memcpy(e.data() + s.offset, &v, 4);
    }
}

std::array<uint32_t, 10> cfg_words(uint64_t pool_addr) {
    std::array<uint32_t, 10> c{};
    c[0] = uint32_t(pool_addr);
    c[1] = uint32_t(pool_addr >> 32);
    for (size_t i = 0; i < kQueues.size(); ++i) c[2 + i] = kQueues[i];
    return c;
}

}  // namespace onebit::npu::lax
