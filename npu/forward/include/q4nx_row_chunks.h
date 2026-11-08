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
// q4nx_row_chunks.h -- how a batched verify pass is split into device launches.
//
// A hardware context has a fixed row capacity (I8Ctx::MD: 16 for the decode twins, 128 for the prefill
// design) and `quantize_async_rows()` REFUSES a wider batch -- it prints "am=... exceeds bA capacity"
// and returns the rows unquantised, after which the caller's launch still runs.  That is not a slow
// path, it is a read and a write past the end of the activation BO: engine#1372 was exactly that on the
// per-op prefill path, and engine#1386 is the same shape on the pb16i LM head, whose rows path launched
// ALL of M into a 16-row context.
//
// #1364 fixed the int8 head by launching in chunks of at most MD rows (a resident weight tile, one
// launch per chunk, and a row's activation scale is its own either way, so chunking cannot move a
// value).  This header is that arithmetic, in one place, with the invariant the tests pin:
//
//   * no chunk is wider than MD;
//   * the chunks are contiguous and cover [0, M) exactly once, in order;
//   * M == 0 is no chunks at all;
//   * an unknown capacity (MD <= 0) is no chunks, so the caller falls back to its per-row path instead
//     of launching blind.
//
// NPU_HEAD_NO_CHUNK=1 asks for the pre-chunking arm: one chunk per row.  That is the A/B arm for the
// claim (#1364's knob, reused so both heads answer the same way), never the default.
#pragma once

#include <algorithm>
#include <vector>

namespace onebit {
namespace npu {

struct RowChunk {
    int offset;  // first row of the chunk, in [0, M)
    int rows;    // 1 .. MD
};

// The chunks a batched launch of `rows` rows is made in, given a context capacity of `capacity` rows.
// `chunked == false` gives one chunk per row (the NPU_HEAD_NO_CHUNK arm).
inline std::vector<RowChunk> row_chunks(int rows, int capacity, bool chunked = true) {
    std::vector<RowChunk> out;
    if (rows <= 0) return out;
    if (capacity <= 0) return out;  // unknown capacity: the caller must not launch
    const int step = chunked ? std::min(rows, capacity) : 1;
    for (int r0 = 0; r0 < rows; r0 += step) out.push_back(RowChunk{r0, std::min(step, rows - r0)});
    return out;
}

}  // namespace npu
}  // namespace onebit
