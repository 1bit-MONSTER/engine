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

// npu_partition_test -- engine#2689's host-side pieces, no device:
//   1. elf_partition_cols() reads the column count from a full ELF's .note.xrt.configuration, and returns 0 (not a guess)
//      for anything that is not one: wrong magic, 64-bit, truncated section table, note too short, no note at all.
//   2. q4nx_i8_sealed_prefill_instance() names every role differently from q4nx_i8_sealed_instance(), since one
//      hardware context refuses a second ELF with an instance key it already holds (measured: "code 501: main:d").
#include "elf_partition.h"
#include "q4nx_pack.h"

#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}
void put32(std::vector<uint8_t>& v, size_t at, uint32_t x) { std::memcpy(v.data() + at, &x, 4); }
void put16(std::vector<uint8_t>& v, size_t at, uint16_t x) { std::memcpy(v.data() + at, &x, 2); }

// A minimal 32-bit LSB ELF: header, [shstrtab], [note], then three section headers (null, shstrtab, note).
std::vector<uint8_t> elf_with_note(uint32_t cols, const char* section = ".note.xrt.configuration", bool note = true) {
    const std::string strtab = std::string("\0.shstrtab\0", 11) + section + std::string("\0", 1);
    std::vector<uint8_t> nt;
    if (note) {
        nt.resize(20);
        put32(nt, 0, 4); put32(nt, 4, 4); put32(nt, 8, 6);
        std::memcpy(nt.data() + 12, "XRT", 4);
        put32(nt, 16, cols);
    }
    const size_t str_off = 52, note_off = str_off + strtab.size(), sh_off = (note_off + nt.size() + 3) & ~size_t(3);
    std::vector<uint8_t> b(sh_off + 3 * 40, 0);
    std::memcpy(b.data(), "\x7f" "ELF", 4);
    b[4] = 1; b[5] = 1; b[6] = 1;
    put32(b, 32, uint32_t(sh_off));
    put16(b, 46, 40); put16(b, 48, 3); put16(b, 50, 1);
    std::memcpy(b.data() + str_off, strtab.data(), strtab.size());
    if (note) std::memcpy(b.data() + note_off, nt.data(), nt.size());
    put32(b, sh_off + 40 + 0, 1);  put32(b, sh_off + 40 + 4, 3);
    put32(b, sh_off + 40 + 16, uint32_t(str_off)); put32(b, sh_off + 40 + 20, uint32_t(strtab.size()));
    put32(b, sh_off + 80 + 0, 11); put32(b, sh_off + 80 + 4, 7);
    put32(b, sh_off + 80 + 16, uint32_t(note_off)); put32(b, sh_off + 80 + 20, uint32_t(nt.size()));
    return b;
}
}  // namespace

int main() {
    using onebit::npu::elf_partition_cols;
    {
        auto e4 = elf_with_note(4), e8 = elf_with_note(8);
        check(elf_partition_cols(e4.data(), e4.size()) == 4, "a 4-column note reads 4");
        check(elf_partition_cols(e8.data(), e8.size()) == 8, "an 8-column note reads 8");
    }
    {
        auto bad = elf_with_note(4);
        bad[0] = 0;
        check(elf_partition_cols(bad.data(), bad.size()) == 0, "wrong magic reads 0");
        auto c64 = elf_with_note(4);
        c64[4] = 2;
        check(elf_partition_cols(c64.data(), c64.size()) == 0, "a 64-bit ELF reads 0 (the full ELFs are 32-bit)");
        auto other = elf_with_note(4, ".note.xrt.UID");
        check(elf_partition_cols(other.data(), other.size()) == 0, "a note of another name reads 0");
        auto none = elf_with_note(4, ".note.xrt.configuration", false);
        check(elf_partition_cols(none.data(), none.size()) == 0, "a configuration section shorter than a note reads 0");
        auto e4 = elf_with_note(4);
        check(elf_partition_cols(e4.data(), e4.size() - 40) == 0, "a truncated section table reads 0, not out of bounds");
        check(elf_partition_cols(e4.data(), 20) == 0, "a file shorter than an ELF header reads 0");
    }
    {
        const Q4nxDesign roles[] = {Q4NX_DESIGN_QKVLIN, Q4NX_DESIGN_QKV, Q4NX_DESIGN_Z,
                                    Q4NX_DESIGN_O,      Q4NX_DESIGN_GU,  Q4NX_DESIGN_D};
        std::set<std::string> dec, pre;
        bool paired = true, prefixed = true;
        for (Q4nxDesign r : roles) {
            const char* d = q4nx_i8_sealed_instance(r);
            const char* p = q4nx_i8_sealed_prefill_instance(r);
            paired &= (d != nullptr) == (p != nullptr);
            if (d && p) {
                dec.insert(d);
                pre.insert(p);
                prefixed &= std::string(p) == "main:p_" + std::string(d).substr(5);
            }
        }
        bool disjoint = true;
        for (const auto& p : pre) disjoint &= dec.count(p) == 0;
        check(paired, "every role with a decode instance has a prefill instance, and only those");
        check(prefixed, "each prefill instance is its decode name with a p_ prefix");
        check(disjoint && pre.size() == dec.size(), "no prefill instance key equals any decode instance key");
        check(q4nx_i8_sealed_prefill_instance(Q4NX_DESIGN_PLEGATE) == nullptr, "a role with no decode instance has none");
    }
    std::printf("%s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
