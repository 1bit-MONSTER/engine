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

void apply_position(std::span<uint32_t> w, const std::vector<PosPatch>& patches, size_t pos, size_t kv_row,
                    size_t ptab_row) {
    const size_t nf = pos ? pos : 1;  // position 0 streams one dummy row the core masks
    for (const auto& p : patches) {
        size_t v = 0;
        switch (p.kind) {
        case PosPatch::WindowLength: v = nf * kv_row / 4; break;
        case PosPatch::RowDrain: v = pos * kv_row; break;
        case PosPatch::Record: v = pos * ptab_row; break;
        case PosPatch::WindowOffset: v = 0; break;
        }
        if (p.word >= w.size()) throw std::runtime_error("position patch past the stream");
        w[p.word] = uint32_t(v) | p.flags;
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
