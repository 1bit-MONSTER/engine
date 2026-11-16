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

// elf_partition.h -- how many NPU columns a full ELF asks the driver for (engine#2689).
//
// A full ELF carries its partition width in the `.note.xrt.configuration` note: one 32-bit value after the "XRT" name.
// The runtime turns it into the hardware context's num_tiles, so it decides how much of the array the context claims.
// A model that shares the NPU must open every context at ONE width (the driver shares a partition within a process only
// when width and start column match, and refuses a 4-column next to an 8-column context: measured 11-15), so the engine
// reads the width itself and refuses a mismatched ELF by name at load, instead of letting the driver fail halfway.
//
// elf_partition_cols(bytes, size) -> the column count, or 0 when the bytes are not a 32-bit little-endian ELF or carry
// no such note. Pure function over a byte range, so the test can feed it crafted ELFs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace onebit::npu {

inline uint32_t elf_rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t elf_rd16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

inline uint32_t elf_partition_cols(const uint8_t* b, size_t n) {
    if (n < 52 || std::memcmp(b, "\x7f" "ELF", 4) != 0 || b[4] != 1 /*ELFCLASS32*/ || b[5] != 1 /*LSB*/) return 0;
    const uint32_t shoff = elf_rd32(b + 32);
    const uint16_t shentsize = elf_rd16(b + 46), shnum = elf_rd16(b + 48), shstrndx = elf_rd16(b + 50);
    if (shentsize < 40 || shnum == 0 || shstrndx >= shnum) return 0;
    if (uint64_t(shoff) + uint64_t(shnum) * shentsize > n) return 0;
    auto sh = [&](uint16_t i) { return b + shoff + size_t(i) * shentsize; };
    const uint32_t stroff = elf_rd32(sh(shstrndx) + 16), strsz = elf_rd32(sh(shstrndx) + 20);
    if (uint64_t(stroff) + strsz > n) return 0;
    static const char kName[] = ".note.xrt.configuration";
    for (uint16_t i = 0; i < shnum; ++i) {
        const uint32_t name = elf_rd32(sh(i)), off = elf_rd32(sh(i) + 16), sz = elf_rd32(sh(i) + 20);
        if (name >= strsz || strsz - name < sizeof(kName)) continue;
        if (std::memcmp(b + stroff + name, kName, sizeof(kName)) != 0) continue;
        if (uint64_t(off) + sz > n || sz < 16) return 0;
        const uint8_t* note = b + off;
        const uint32_t namesz = elf_rd32(note), descsz = elf_rd32(note + 4);
        const uint64_t doff = 12ull + ((uint64_t(namesz) + 3) & ~3ull);
        if (descsz < 4 || doff + 4 > sz) return 0;
        return elf_rd32(note + doff);
    }
    return 0;
}

inline uint32_t elf_partition_cols_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return 0;
    std::vector<uint8_t> v((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return elf_partition_cols(v.data(), v.size());
}

}  // namespace onebit::npu
