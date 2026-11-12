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

// weight_keys.h -- the keys a layer's projections are cached under on the device.
//
// The device target keeps one packed weight image per (layer, key) (XdnaTarget's weight cache), so two projections
// of the same layer that share a key share an image: the second one launches with the first one's weights. The host
// arms do not go through that cache, so a collision is invisible off the device -- a Gemma4 pack answered correctly
// in host-GEMM mode and returned noise on the NPU, with every kernel bit-exact, because its per-layer input gate and
// projection were launched under the attention projections' keys 0 and 1.
//
// The keys of step()'s dense layer are listed here and held pairwise distinct at compile time.
//
// Which call sites use the names today: only the two that were wrong, the Gemma4 gate and projection in step()
// (kWkeyPleGate, kWkeyPleProj). The older call sites still pass literals -- 0..4 and 10..13 in step() and in the
// batched body -- and the names above carry the same numbers, so the list describes them but does not yet bind
// them. Moving those to the names is a mechanical follow-up. Until then the rule for a new projection is: take a
// new name here, not a number there.

#pragma once

namespace onebit::npu {

enum : int {
    kWkeyQkv = 0,        // attention q|k|v (or the first half of a split design)
    kWkeyO = 1,          // attention output
    kWkeyGateUp = 2,     // fused gate|up, or gate alone on a split MLP
    kWkeyUp = 3,         // up alone on a split MLP
    kWkeyDown = 4,       // MLP down
    kWkeyPleGate = 8,    // Gemma4 per-layer input gate
    kWkeyPleProj = 9,    // Gemma4 per-layer projection
    kWkeySplitQ0 = 10,   // split full-attention halves
    kWkeySplitQ1 = 11,
    kWkeySplitK = 12,
    kWkeySplitV = 13,
};

constexpr int kLayerWkeys[] = {kWkeyQkv, kWkeyO, kWkeyGateUp, kWkeyUp, kWkeyDown, kWkeyPleGate, kWkeyPleProj,
                               kWkeySplitQ0, kWkeySplitQ1, kWkeySplitK, kWkeySplitV};

// The predicate on any list, so a test can hand it one that is NOT distinct and see it say so.
constexpr bool wkeys_distinct(const int* keys, int n) {
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (keys[i] == keys[j]) return false;
    return true;
}
constexpr bool layer_wkeys_distinct() {
    return wkeys_distinct(kLayerWkeys, int(sizeof(kLayerWkeys) / sizeof(kLayerWkeys[0])));
}
static_assert(layer_wkeys_distinct(), "two projections of one layer share a weight-cache key");

}  // namespace onebit::npu
