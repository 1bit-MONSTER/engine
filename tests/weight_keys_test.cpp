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

// weight_keys_test.cpp -- the per-layer weight-cache keys (npu/forward/include/weight_keys.h) are pairwise distinct,
// and the two Gemma4 per-layer-input projections do not use the attention projections' keys. The header asserts the
// first at compile time; this test is what makes CI compile it, and it names the pair that broke.

#include <cstdio>

#include "weight_keys.h"

int main() {
    using namespace onebit::npu;
    int fail = 0;
    const int n = int(sizeof(kLayerWkeys) / sizeof(kLayerWkeys[0]));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (kLayerWkeys[i] == kLayerWkeys[j]) {
                std::printf("FAIL keys %d and %d of the list are both %d\n", i, j, kLayerWkeys[i]);
                fail++;
            }
    if (kWkeyPleGate == kWkeyQkv || kWkeyPleGate == kWkeyO || kWkeyPleProj == kWkeyQkv || kWkeyPleProj == kWkeyO) {
        std::printf("FAIL a per-layer-input projection shares a key with an attention projection of the same layer\n");
        fail++;
    }
    // The predicate is not a constant: a list with a repeat is reported, wherever the repeat sits.
    const int dup_ends[] = {0, 1, 2, 0}, dup_mid[] = {0, 8, 8, 9}, one[] = {5};
    if (wkeys_distinct(dup_ends, 4) || wkeys_distinct(dup_mid, 4)) {
        std::printf("FAIL a list with a repeated key is called distinct\n");
        fail++;
    }
    if (!wkeys_distinct(one, 1) || !wkeys_distinct(one, 0)) {
        std::printf("FAIL a list of one key, or of none, is called not distinct\n");
        fail++;
    }
    if (fail) return 1;
    std::printf("weight_keys: ok (%d keys, all distinct; a repeated key is reported)\n", n);
    return 0;
}
