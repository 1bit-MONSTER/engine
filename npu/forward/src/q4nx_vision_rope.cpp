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

#include "q4nx_vision.h"

#include <cmath>
#include <cstdio>

namespace q4nx_vision {

void rope_axial(float* x, int tokens, int n_heads, int head_dim, const float* inv_freq, const int* pos_h,
                const int* pos_w) {
    if (!x || !inv_freq || !pos_h || !pos_w || tokens <= 0 || n_heads <= 0 || head_dim < 4 || (head_dim % 4)) {
        std::fprintf(stderr, "vision rope_axial: bad geometry tokens=%d heads=%d hd=%d\n", tokens, n_heads, head_dim);
        return;
    }
    const int nf = head_dim / 4;   // frequencies per axis; the angle is 2*nf wide and is TILED to head_dim
    const int half = head_dim / 2;
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            float* v = x + (size_t(t) * n_heads + h) * head_dim;
            for (int d = 0; d < half; ++d) {
                // dims [0, nf) turn with the row position, [nf, half) with the column position
                const int axis = d < nf ? 0 : 1;
                const float pos = float(axis == 0 ? pos_h[t] : pos_w[t]);
                const float ang = pos * inv_freq[d < nf ? d : d - nf];
                const float c = std::cos(ang), s = std::sin(ang);
                const float a = v[d], b = v[half + d];
                v[d] = a * c - b * s;         // rotate_half over the whole head_dim
                v[half + d] = b * c + a * s;
            }
        }
    }
}

}  // namespace q4nx_vision
