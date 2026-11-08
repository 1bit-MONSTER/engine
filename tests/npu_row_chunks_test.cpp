// Copyright (c) 2026 bong-water-water-bong
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
// npu_row_chunks_test.cpp -- the invariant that keeps a batched verify pass inside its context.
//
// engine#1386: the pb16i LM-head rows path launched all M rows into a 16-row context; the quantiser
// refused the batch, returned the rows unquantised, and the launch ran anyway (engine#1372's failure on
// the head). The fix is the chunking #1364 shipped for the int8 head, and these are the properties that
// make it a fix rather than a rearrangement:
//
//   no chunk wider than the capacity, the chunks contiguous and in order, every row covered exactly
//   once, M == 0 as no chunks, an unknown capacity as no chunks (so the caller falls back instead of
//   launching blind), and the A/B arm (one row per chunk) as the same coverage.
//
// Device-free: the header is arithmetic only. A device A/B of the launch count is #1386's own note.
#include <cstdio>
#include <set>
#include <vector>

#include "q4nx_row_chunks.h"

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::printf("FAIL %s\n", what);
        ++g_failures;
    }
}

// Coverage: contiguous from 0, no gaps, no overlaps, total == rows, each chunk <= capacity, all >= 1.
void check_cover(int rows, int capacity) {
    const std::vector<onebit::npu::RowChunk> cs = onebit::npu::row_chunks(rows, capacity, true);
    int next = 0;
    std::set<int> seen;
    for (const auto& c : cs) {
        check(c.rows >= 1, "chunk is non-empty");
        check(c.rows <= capacity, "chunk is no wider than the capacity");
        check(c.offset == next, "chunk is contiguous with the previous one");
        for (int r = c.offset; r < c.offset + c.rows; ++r) check(seen.insert(r).second, "row covered once");
        next = c.offset + c.rows;
    }
    check(next == rows, "chunks cover every row");
    check((int)cs.size() == (rows + capacity - 1) / capacity, "chunk count is ceil(rows / capacity)");
}

}  // namespace

int main() {
    using onebit::npu::row_chunks;

    // 1. The regression itself: M above the 16-row twin capacity must become ceil(M/16) chunks, never one
    //    launch of M rows. The old code passed M straight through, which is what the device refused.
    for (int m : {17, 18, 31, 32, 33, 64, 128}) check_cover(m, 16);
    check((int)row_chunks(17, 16, true).size() == 2, "17 rows into a 16-row context is 2 launches");
    check((int)row_chunks(128, 16, true).size() == 8, "a 128-row verify batch is 8 launches");
    check(row_chunks(17, 16, true)[0].offset == 0 && row_chunks(17, 16, true)[1].offset == 16,
          "the second chunk starts at row 16");
    check(row_chunks(17, 16, true)[1].rows == 1, "the tail chunk carries the 1 leftover row");

    // 2. Small and exact batches are unchanged: M <= capacity is one launch of M rows (so M=16 is
    //    bit-for-bit what the path did before chunking).
    for (int m : {1, 2, 16}) {
        const auto cs = row_chunks(m, 16, true);
        check((int)cs.size() == 1 && cs[0].offset == 0 && cs[0].rows == m, "M <= capacity is one launch");
    }
    check((int)row_chunks(16, 16, true).size() == 1, "M == capacity is exactly one launch");
    check_cover(1, 16);
    check_cover(16, 16);

    // 3. The prefill design's capacity (128 rows) is where the head would never need chunking, but the
    //    arithmetic must still hold for it.
    check((int)row_chunks(128, 128, true).size() == 1, "128 rows into a 128-row context is one launch");
    check_cover(130, 128);

    // 4. The A/B arm (NPU_HEAD_NO_CHUNK=1): one row per chunk, same coverage - that is what the path did
    //    before chunking existed, so a numeric A/B compares artifacts, not coverage.
    const auto per_row = row_chunks(17, 16, false);
    check((int)per_row.size() == 17, "the no-chunk arm is one launch per row");
    int next = 0;
    for (const auto& c : per_row) {
        check(c.rows == 1 && c.offset == next, "no-chunk arm covers rows one at a time, in order");
        ++next;
    }

    // 5. Refusals, not guesses: no rows and an unknown capacity both produce no chunks, so the caller
    //    takes its per-row path rather than launching into a context it cannot size.
    check(row_chunks(0, 16, true).empty(), "0 rows is no chunks");
    check(row_chunks(-3, 16, true).empty(), "a negative row count is no chunks");
    check(row_chunks(64, 0, true).empty(), "capacity 0 is no chunks");
    check(row_chunks(64, -1, true).empty(), "a negative capacity is no chunks");

    if (g_failures == 0) {
        std::printf("npu_row_chunks: PASS\n");
        return 0;
    }
    std::printf("npu_row_chunks: %d failure(s)\n", g_failures);
    return 1;
}
