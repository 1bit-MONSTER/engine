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
// Proprietary and confidential. Distributed only as compiled binaries under the
// 1bit Binary License (LICENSE-BINARY).

// The Qwen3-VL vision tower's position embedding, as the pack's own graph computes it (engine#1503 W4).
//
// The graph interpolates a learned (48, 48, C) table to the patch grid and then reorders the tokens into 2x2
// merge-block order. Both halves were established against the graph itself, not inferred:
//
//   * the interpolation is bilinear with **align_corners = TRUE**, i.e. `src = c * 47 / (n - 1)`. The neighbouring
//     convention `c * 48 / n` reproduces the graph's corner indices for the first eight rows and columns and then
//     drifts, which is what made this look like a mysterious index map for a while: measured against the graph's own
//     gather, align_corners gives corr 1.00000000 / mean|d| 5e-8, and `c*48/n` gives 0.9959.
//   * the reorder is exactly one permutation: `perm = [0 1 gw gw+1 | 2 3 gw+2 gw+3 | ...]`, i.e. group g takes the
//     2x2 block at (2*(g/gw2), 2*(g%gw2)) as (h,w), (h,w+1), (h+1,w), (h+1,w+1). Verified: applying it to the
//     row-major resample reproduces what the graph adds to the patch embedding at corr 1.000000 / mean|d| 5e-8.
//
// Both facts are pinned by tests/q4nx_vision_pos_test.cpp against a fixture produced by the numpy reference that
// matches the fp32 oracle at corr 1.000000, so a regression here fails a test rather than a served image.
#ifndef ONEBIT_NPU_VISION_H
#define ONEBIT_NPU_VISION_H

#include <cstddef>

namespace q4nx_vision {

// Tokens the grid holds, i.e. gh * gw. The merge-block order only exists when both extents are even (merge size 2).
constexpr int kMergeSize = 2;
constexpr int kPosTableSide = 48;

// Index into the (side*side) table that position (h, w) of a (gh, gw) grid maps to, per axis: floor(c * (side-1)/(n-1))
// and the fraction towards the next index. `side` and `n` must be >= 2; the fraction is 0 when the next index clamps.
inline int axis_index(int c, int n, int side) {
    const int hi = side - 1;
    const int nm1 = n > 1 ? n - 1 : 1;
    int i = (c * hi) / nm1;                        // align_corners = true
    if (i > hi) i = hi;
    if (i < 0) i = 0;
    return i;
}
inline float axis_frac(int c, int n, int side) {
    const int hi = side - 1;
    const int nm1 = n > 1 ? n - 1 : 1;
    const float src = float(c) * float(hi) / float(nm1);
    int i = axis_index(c, n, side);
    float f = src - float(i);
    if (i >= hi) f = 0.0f;                          // no neighbour: the second tap is the same row/col
    return f;
}

// Token t of a (gh, gw) patch grid, in merge-block order, as (row-major) h and w.
inline void block_position(int t, int gh, int gw, int* h, int* w) {
    const int gw2 = gw / kMergeSize;
    const int b = t / (kMergeSize * kMergeSize);
    const int sub = t % (kMergeSize * kMergeSize);
    *h = kMergeSize * (b / gw2) + sub / kMergeSize;
    *w = kMergeSize * (b % gw2) + sub % kMergeSize;
    (void)gh;
}

// out[token * C + c] = bilinear(table, align_corners) at the token's (h, w), where `token` runs in merge-block order.
// `table` is row-major (side, side, C); `out` is (gh*gw, C). gh and gw must be even and >= kMergeSize.
void pos_embed(const float* table, int gh, int gw, int C, float* out);

// The tower's axial 2D rotary embedding, exactly as the graph applies it to q and k:
//   ang = concat([pos_h * inv_freq, pos_w * inv_freq])            (head_dim/2 wide)
//   cos = TILE(cos(ang), 2),  sin = TILE(sin(ang), 2)             (Lg: cat([a,a]) -- a tile, NOT an interleave)
//   out = x*cos + rotate_half(x)*sin,  rotate_half(u) = cat(-u[head_dim/2:], u[:head_dim/2])
// Established against the graph's own post-rope q: corr 0.999996, mean|d| 3.3e-3 on |ref| 1.15. Both details are load
// bearing and both were wrong in my first three attempts: an interleaved `repeat` gives corr 0.44 where the tile gives 0.84
// before the rest of the tower is even considered, and rotating each 32-dim half separately gives 0.99 but never 1.
// `inv_freq` is the pack's own `visual.rotary_pos_emb.inv_freq` table (head_dim/4 entries); `pos_h`/`pos_w` are the token
// positions in merge-block order (see block_position).
void rope_axial(float* x, int tokens, int n_heads, int head_dim, const float* inv_freq, const int* pos_h, const int* pos_w);

}  // namespace q4nx_vision

#endif  // ONEBIT_NPU_VISION_H
