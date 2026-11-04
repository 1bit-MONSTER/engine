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
// q4nx_state_planes.h -- the plane-copy half of a state snapshot, in isolation so it can be tested without a
// device (tests/q4nx_state_planes_test.cpp).  Two kinds of plane go into a snapshot and they are cut differently:
//
//   PER-POSITION (kcache_/vcache_): one layer holds [NKV][HD] floats for every position it has seen, so a
//   snapshot of the first `tokens` positions keeps `tokens * NKV * HD` floats of it and no more.
//
//   WHOLE (conv_state_/rec_state_): the GDN conv window and the recurrent state are one fixed block per layer
//   that is *updated* by each position, not appended to.  Its size has nothing to do with `tokens`, so it is
//   always carried and restored whole.
//
// Applying the per-position cut to a whole plane truncates it (a 786,432-float recurrent plane keeps only
// `tokens * 1024` of itself, 23 % of it at 180 tokens), and the restore then refuses the snapshot because the
// plane cannot cover the positions being restored -- which is why the prefix cache answered every request with
// `slot has no usable state` and re-prefilled, measured 2026-11-04 on the shipped Bonsai-2 serve.
#ifndef Q4NX_STATE_PLANES_H
#define Q4NX_STATE_PLANES_H

#include <algorithm>
#include <cstddef>
#include <vector>

namespace q4nx_state {

// Snapshot side: keep the first `want` floats of each per-position layer (empty layers stay empty).
inline void take_per_pos(const std::vector<std::vector<float>>& src, std::vector<std::vector<float>>& dst,
                         size_t want) {
    dst.resize(src.size());
    for (size_t l = 0; l < src.size(); l++) {
        const size_t n = std::min(want, src[l].size());
        dst[l].assign(src[l].begin(), src[l].begin() + long(n));
    }
}

// Snapshot side: carry a whole plane as it stands.
inline void take_whole(const std::vector<std::vector<float>>& src, std::vector<std::vector<float>>& dst) {
    dst = src;
}

// Restore side: a per-position layer must cover the positions being restored, and an empty layer means the
// plane is not this layer's to hold (a shared-KV layer), so the destination is left empty rather than filled
// with another layer's keys.
inline bool give_per_pos(const std::vector<std::vector<float>>& src, std::vector<std::vector<float>>& dst,
                         size_t want) {
    if (dst.size() < src.size()) return false;
    for (size_t l = 0; l < src.size(); l++) {
        if (src[l].empty()) {
            dst[l].clear();
            continue;
        }
        if (src[l].size() < want) return false;
        dst[l].assign(src[l].begin(), src[l].begin() + long(want));
    }
    return true;
}

// Restore side: a whole plane is accepted only at the size this forward allocated.  A truncated one is refused
// (the caller falls back to a full prefill) instead of being installed short, which would leave the GDN step
// reading past the end of the plane.
inline bool give_whole(const std::vector<std::vector<float>>& src, std::vector<std::vector<float>>& dst) {
    if (dst.size() < src.size()) return false;
    for (size_t l = 0; l < src.size(); l++) {
        if (src[l].size() != dst[l].size()) return false;
        dst[l] = src[l];
    }
    return true;
}

}  // namespace q4nx_state

#endif  // Q4NX_STATE_PLANES_H
