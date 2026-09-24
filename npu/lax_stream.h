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

// Host-side words the lax decode writes to the device between tokens (docs/npu-lax.md).
// Pure host code, no XRT.
//
// Position patching. The full-attention control text (lax_a insts.bin) is compiled for
// one placeholder position, and all ten full-attention runs of a token share it. An
// mlir-aie instruction stream is 4 header words, then ops: op 1 is a BD blockwrite (12
// words, its register at +2 and the BD's length at +4), op 0x81 a DDR address patch (12
// words: register at +6, the kernel's buffer argument at +8, the byte offset into that
// buffer at +10); ops 0, 3 and 0x80 are 6, 7 and 4 words, anything else one. Per token
// four words change:
//   the KV window fill's length   (the blockwrite before the arg-3 patch at offset 0), in
//                                 32-bit words: nf rows, nf = max(pos, 1)
//   the KV window fill's offset   (that patch): the window's first row, always 0 here
//   the new row's drain offset    (the arg-3 patch at one row): pos rows
//   the position record's offset  (the arg-5 patch): ptab row pos
// Arguments past the first five carry the firmware's +0x80000000 address translation in
// the offset word; that bit is kept. Layout from the open kernels' harness
// (open_kernels/harness/stream_patch.hpp, attn_table / attn_apply; MIT), re-implemented.
//
// Layer config (cfg, 4 KiB per layer): words 0..1 the layer pool's device address
// (bo.address() + 0x80000000), which the on-device router forms the routed experts'
// addresses from; words 2..9 each column's weight-stream MM2S queue register (the merged
// lax design's shim allocation; open_kernels model/lax_decode_cfg.py QUEUES).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace onebit::npu::lax {

struct PosPatch {
    enum Kind : uint8_t { WindowLength, RowDrain, Record, WindowOffset };
    size_t word;     // index into the instruction words
    Kind kind;
    uint32_t flags;  // bits of the offset word to keep (0x80000000 or 0)
};

// Words of one instruction-stream op starting with opcode word w.
size_t op_words(uint32_t w);

// The four position patches of a full-attention stream; throws unless each is found
// exactly once. kv_row is the KV row in bytes.
std::vector<PosPatch> position_patches(std::span<const uint32_t> words, size_t kv_row);

// Write position pos into the stream.
void apply_position(std::span<uint32_t> words, const std::vector<PosPatch>& patches, size_t pos, size_t kv_row,
                    size_t ptab_row);

constexpr std::array<uint32_t, 8> kQueues = {0x1D21C, 0x1D214, 0x1D21C, 0x1D21C, 0x1D21C, 0x1D21C, 0x1D214, 0x1D214};

// cfg words 0..9 for a pool at device address pool_addr (bo.address() + 0x80000000).
std::array<uint32_t, 10> cfg_words(uint64_t pool_addr);

}  // namespace onebit::npu::lax
