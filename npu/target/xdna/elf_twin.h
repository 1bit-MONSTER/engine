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

// elf_twin.h -- engine#2599: is a prefill ELF the very same artifact as its own M=16 decode twin?
//
// bonsai2-27b-NPU shipped kernels/full_i8_<ROLE>_K*_N*.elf byte-identical to the matching _m16.elf: its "prefill" set
// was the 16-row design.  The artifact cannot say how many rows it computes (its declared widths are per tile), so a
// context sized from the design geometry (M=128) computed the first 16 rows of every 128-row chunk.  The evidence is
// equal size and equal bytes against the twin of the SAME stem -- two roles of one shape (G and U) are legitimately
// identical and are never compared with each other.  tests/npu_elf_twin_test.cpp pins both functions.
#pragma once

#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace onebit::npu {

// "<dir>/<stem>_m16.elf" for an ELF named "<stem>.elf" (a name without ".elf" keeps its whole name as the stem).
inline std::string m16_twin_path(const std::string& dir, const std::string& elf_name) {
    std::string stem(elf_name);
    if (stem.size() > 4 && stem.compare(stem.size() - 4, 4, ".elf") == 0) stem.resize(stem.size() - 4);
    return dir + "/" + stem + "_m16.elf";
}

// True only when both files open, are the same non-zero size and hold the same bytes.
inline bool same_artifact(const std::string& a, const std::string& b) {
    std::ifstream fa(a, std::ios::binary | std::ios::ate), fb(b, std::ios::binary | std::ios::ate);
    if (!fa || !fb) return false;
    const std::streamoff na = fa.tellg(), nb = fb.tellg();
    if (na <= 0 || na != nb) return false;
    fa.seekg(0);
    fb.seekg(0);
    std::vector<char> ba(1 << 16), bb(1 << 16);
    for (std::streamoff left = na; left > 0;) {
        const std::streamsize n = (std::streamsize)std::min<std::streamoff>(left, (std::streamoff)ba.size());
        if (!fa.read(ba.data(), n) || !fb.read(bb.data(), n)) return false;
        if (std::memcmp(ba.data(), bb.data(), (size_t)n) != 0) return false;
        left -= n;
    }
    return true;
}

}  // namespace onebit::npu
