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

// layer_taps_test.cpp -- the three rules of npu/forward/include/layer_taps.h (engine#1780), without a model.

#include <cstdio>
#include <vector>

#include "layer_taps.h"

using namespace onebit::npu;

static int g_fail = 0, g_n = 0;
static void check(bool ok, const char* what) {
    ++g_n;
    if (!ok) { std::printf("FAIL %s\n", what); ++g_fail; }
}

int main() {
    {
        const int a[] = {5, 11, 17, 23, 35};
        check(layer_taps_usable(a, 5, 42), "the Bolt draft's five layers are usable on a 42-layer model");
        check(!layer_taps_usable(a, 5, 35), "a layer equal to the layer count is outside the model");
        check(layer_taps_usable(a, 5, 36), "the last layer of the model is usable");
        const int d[] = {5, 5}, r[] = {11, 5}, n[] = {-1, 3};
        check(!layer_taps_usable(d, 2, 42), "a repeated layer is refused");
        check(!layer_taps_usable(r, 2, 42), "a descending list is refused");
        check(!layer_taps_usable(n, 2, 42), "a negative layer is refused");
        check(!layer_taps_usable(a, 0, 42) && !layer_taps_usable(nullptr, 5, 42) && !layer_taps_usable(a, 5, 0),
              "an empty list, a null list and a model without layers are refused");
        check(layer_tap_index(a, 5, 5) == 0 && layer_tap_index(a, 5, 23) == 3 && layer_tap_index(a, 5, 35) == 4,
              "a listed layer owns the column block of its place in the list");
        check(layer_tap_index(a, 5, 6) == -1 && layer_tap_index(a, 5, 0) == -1, "an unlisted layer owns none");
    }
    {   // 2 positions, 3 taps, 2 wide: position m, tap j lands at (m * 3 + j) * 2, whatever order the layers arrive in
        std::vector<float> rows(2 * 3 * 2, -1.0f);
        for (int j = 2; j >= 0; j--)          // layer-major, last tap first, as a batched pass would not -- order-free
            for (int m = 0; m < 2; m++) {
                const float x[2] = {float(100 * m + 10 * j), float(100 * m + 10 * j + 1)};
                layer_tap_put(rows.data(), m, j, 3, 2, x);
            }
        bool ok = true;
        for (int m = 0; m < 2; m++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 2; k++)
                    ok = ok && rows[size_t((m * 3 + j) * 2 + k)] == float(100 * m + 10 * j + k);
        check(ok, "a row is one position with its taps side by side, in list order");
    }
    if (g_fail) { std::printf("layer_taps: %d of %d checks FAILED\n", g_fail, g_n); return 1; }
    std::printf("layer_taps: ok (%d checks)\n", g_n);
    return 0;
}
