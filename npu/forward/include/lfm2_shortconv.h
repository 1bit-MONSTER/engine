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
// lfm2_shortconv.h -- the gated short convolution of an LFM2 layer, as host arithmetic (engine#1786).
//
// LFM2 is a hybrid: ten of LFM2-1.2B's sixteen layers have no attention at all. In place of it they run
//
//     bcx = in_proj(norm(x))                 H -> 3H, the three thirds are B, C and X
//     bx  = B * X                            elementwise
//     conv[h] = sum_j taps[h][j] * seq[t + j][h]      depthwise, causal, L taps: seq is the last L-1 values of bx
//                                                     followed by this token's, so taps[h][L-1] meets the present
//     y   = out_proj(C * conv)               H -> H
//
// and then the same residual and feed-forward as an attention layer. The only state a layer carries from one token
// to the next is the last L-1 values of bx, per channel: (L-1) * H floats, where an attention layer carries a KV row.
//
// The two projections are ordinary GEMMs and belong to the device. What is here is everything between them, written
// once so that the three places that will need it (token-by-token decode, a batched prefill, and a snapshot of the
// state for the prefix cache) share one definition, and tested without a model: the batched form must equal the
// token-by-token form bit for bit, because a prefill that disagrees with decode is wrong at every later position.
//
// The definition follows the graph our own llama.cpp fork builds for this family (src/models/lfm2.cpp,
// build_shortconv_block), not any vendor's code. Layout of `taps`: [H][L] row-major, the container's
// `shortconv.conv.weight` after its bf16 decode.

#pragma once

#include <cstddef>
#include <cstring>
#include <vector>

namespace onebit::npu {

// The conv state of one layer: the last L-1 values of bx, oldest first, [L-1][H] row-major. All zeros at position 0.
struct ShortConvState {
    int hidden = 0;
    int taps = 0;                  // L, the kernel length (config.json conv_L_cache)
    std::vector<float> history;    // (L-1) * hidden

    void reset(int hidden_size, int kernel_len) {
        hidden = hidden_size;
        taps = kernel_len;
        history.assign(size_t(kernel_len > 1 ? kernel_len - 1 : 0) * size_t(hidden_size), 0.0f);
    }
    bool ready() const { return hidden > 0 && taps > 0 && history.size() == size_t(taps - 1) * size_t(hidden); }
};

// One token. `bcx` is in_proj's output, 3*H values [B | C | X]; `taps` is [H][L]; `y` receives H values, the input of
// out_proj. The state is advanced by one position. Returns false, touching nothing, when the state is not sized.
inline bool shortconv_step(ShortConvState& st, const float* bcx, const float* taps, float* y) {
    if (!st.ready() || bcx == nullptr || taps == nullptr || y == nullptr) return false;
    const int H = st.hidden, L = st.taps;
    const float* B = bcx;
    const float* C = bcx + H;
    const float* X = bcx + 2 * H;
    for (int h = 0; h < H; ++h) {
        const float bx = B[h] * X[h];
        const float* k = taps + size_t(h) * size_t(L);
        float acc = 0.0f;
        for (int j = 0; j < L - 1; ++j) acc += k[j] * st.history[size_t(j) * size_t(H) + size_t(h)];
        acc += k[L - 1] * bx;
        y[h] = C[h] * acc;
    }
    // slide the window: drop the oldest row, append this token's bx
    if (L > 1) {
        if (L > 2) std::memmove(st.history.data(), st.history.data() + H, size_t(L - 2) * size_t(H) * sizeof(float));
        float* last = st.history.data() + size_t(L - 2) * size_t(H);
        for (int h = 0; h < H; ++h) last[h] = B[h] * X[h];
    }
    return true;
}

// Advance the state by one position WITHOUT computing an output: `bx` is that position's B*X, H values. This is
// what a rolled-back batch re-applies for the rows it keeps (the outputs were already produced; only the window
// has to match the accepted prefix). Leaves exactly the state shortconv_step() leaves for the same token.
inline bool shortconv_push(ShortConvState& st, const float* bx) {
    if (!st.ready() || bx == nullptr) return false;
    const int H = st.hidden, L = st.taps;
    if (L > 1) {
        if (L > 2) std::memmove(st.history.data(), st.history.data() + H, size_t(L - 2) * size_t(H) * sizeof(float));
        std::memcpy(st.history.data() + size_t(L - 2) * size_t(H), bx, size_t(H) * sizeof(float));
    }
    return true;
}

// M tokens in order: `bcx` is [M][3H] row-major, `y` is [M][H]. Defined as M calls of shortconv_step(), in that
// order, so there is nothing for it to disagree with.
inline bool shortconv_rows(ShortConvState& st, const float* bcx, int rows, const float* taps, float* y) {
    if (rows < 0) return false;
    for (int r = 0; r < rows; ++r)
        if (!shortconv_step(st, bcx + size_t(r) * 3 * size_t(st.hidden), taps, y + size_t(r) * size_t(st.hidden))) return false;
    return true;
}

}  // namespace onebit::npu
