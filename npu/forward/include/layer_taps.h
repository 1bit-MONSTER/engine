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

// layer_taps.h -- where a tapped layer's rows go (engine#1780).
//
// A DFlash draft reads the target model's hidden states at a few layers (its target_layer_ids). The forward collects
// them with Q4nxNpuForward::set_layer_taps(); the three rules that decide what lands where are here, header-only, so
// a test can hold them without a model:
//
//   - which lists are usable: ascending, no repeats, every layer inside the model;
//   - which column block a layer owns: its index in the list, or none;
//   - where a row goes: position m of a pass, tap j of T, H wide -> rows[(m * T + j) * H ...]. That is one row per
//     position with the tapped layers side by side, the layout dflash_forward.h takes as `taps`.
//
// And one about failure: LayerTapPass takes the sink back to the size it had when the pass began unless the pass is
// marked kept. A pass opens its rows when the first listed layer arrives, so a pass that fails later would otherwise
// leave rows whose remaining blocks are zeros, and zeros read as data.

#pragma once

#include <cstddef>
#include <cstring>
#include <vector>

namespace onebit::npu {

inline bool layer_taps_usable(const int* layers, int n, int model_layers) {
    if (!layers || n <= 0 || model_layers <= 0) return false;
    for (int i = 0; i < n; i++) {
        if (layers[i] < 0 || layers[i] >= model_layers) return false;
        if (i > 0 && layers[i] <= layers[i - 1]) return false;
    }
    return true;
}

inline int layer_tap_index(const int* layers, int n, int layer) {
    for (int i = 0; i < n; i++)
        if (layers[i] == layer) return i;
    return -1;
}

inline void layer_tap_put(float* rows, int m, int j, int taps, int hidden, const float* x) {
    std::memcpy(rows + (size_t(m) * size_t(taps) + size_t(j)) * size_t(hidden), x, size_t(hidden) * sizeof(float));
}

struct LayerTapPass {
    std::vector<float>* sink;
    size_t base;
    bool keep = false;
    explicit LayerTapPass(std::vector<float>* s) : sink(s), base(s ? s->size() : 0) {}
    ~LayerTapPass() {
        if (sink && !keep && sink->size() > base) sink->resize(base);
    }
    LayerTapPass(const LayerTapPass&) = delete;
    LayerTapPass& operator=(const LayerTapPass&) = delete;
};

}  // namespace onebit::npu
