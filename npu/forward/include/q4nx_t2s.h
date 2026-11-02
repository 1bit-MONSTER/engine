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
//
// q4nx_t2s.h -- the "t2s" weight stream: pb16i's exact ternary weights at 2.125 bits/weight.
//
// A ternary model needs 2 bits per weight (code in {-1,0,+1}) and one fp16 magnitude per 128 K
// rows per column.  The pb16i image streams 4.5 bits/weight because it carries int4 nibbles and
// repeats the magnitude into every 32-K group; t2s carries the codes at 2 bits and the magnitude
// once per 128-K block, and the decode kernel expands the codes back to int4 nibbles **inside the
// core** between the DMA and the mmul.  Nothing is requantised: the stream is derived from the
// pb16i image by dropping only duplicated bytes, which tools/q4nx_t2s.py checks and refuses
// (named error) when a tensor's 32-K multipliers are not shared inside their 128-K block.
//
// The engine's part of that contract is small -- the sidecar file's size, and the fact that the
// side table (e_ref int16[N] + mode byte) is byte-identical to pb16i's, because the compaction
// changes neither the magnitude values nor e_ref.  A tile is
//
//   2,176 B per k=256 x n=32 tile       (pb16i: 4,608 B)      27B decode: 6.11 GB/token, not 12.93
//   [0, 2048)     2-bit codes, 4 per byte, n-major blocks: block (nb, kb) at (nb*16 + kb)*64,
//                 and inside a block byte r*4 + j holds the 4 codes for k = kb*16 + r,
//                 n = nb*16 + 4j + {0,1,2,3}, code c at bits 2*c; 0 -> -1, 1 -> 0, 2 -> +1,
//                 3 reserved (a converter never writes it, and the kernel refuses it)
//   [2048, 2176)  int16 multipliers, [nb 2][gb 2][16 c]: one per 128-K block per column
//
// and the tiles are ordered exactly as pb16i orders them (c in 8, ki in K/256, nt in CW/32, r in 4),
// so a per-core stream stays one contiguous forward walk.  The full format, the expansion the
// kernel runs and the byte-exact gates live in tools/q4nx_t2s.py and docs/npu.md; this header is
// only what the host needs in order to size and read the file.
//
// tests/npu_t2s_layout_test.cpp pins the numbers below against the Python tool's measured ones.
#pragma once

#include <cstddef>
#include <cstdint>

namespace q4nx_t2s {

constexpr int KC = 256;                    // K rows per tile (pb16i's, unchanged)
constexpr int NW = 32;                     // N columns per tile
constexpr int CODE_BYTES = KC * NW / 4;    // 2 bits per weight
constexpr int MS_BYTES = (KC / 128) * NW * 2;   // one int16 per 128-K block per column
constexpr int TILE = CODE_BYTES + MS_BYTES;     // 2,176
constexpr int RESERVED = 3;

// A geometry that has a t2s stream at all.  N is finer than pb16i's: the tile's N unit is 32
// columns, and the core owns a contiguous run of whole tiles (the kernel partitions N across
// cores, not M).
inline bool fits(int K, int N) {
    return K > 0 && N > 0 && K % KC == 0 && N % NW == 0;
}

inline size_t tiles(int K, int N) {
    return (size_t)(K / KC) * (size_t)(N / NW);
}

inline size_t w_bytes(int K, int N) {
    return tiles(K, N) * (size_t)TILE;
}

// What the stream costs per weight: 2,176 B for 8,192 weights = 2.125 bits.
inline double bits_per_weight() {
    return 8.0 * (double)TILE / (double)(KC * NW);
}

}  // namespace q4nx_t2s
