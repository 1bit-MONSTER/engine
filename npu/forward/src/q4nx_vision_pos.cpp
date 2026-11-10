// Copyright 2026 bong-water-water-bong. All rights reserved.
// Proprietary and confidential. Distributed only as compiled binaries under the
// 1bit Binary License (LICENSE-BINARY).

#include "q4nx_vision.h"

#include <cstdio>

namespace q4nx_vision {

void pos_embed(const float* table, int gh, int gw, int C, float* out) {
    const int side = kPosTableSide;
    if (!table || !out || C <= 0 || gh < kMergeSize || gw < kMergeSize ||
        (gh % kMergeSize) || (gw % kMergeSize)) {
        std::fprintf(stderr, "vision pos_embed: bad geometry gh=%d gw=%d C=%d\n", gh, gw, C);
        return;
    }
    const int tokens = gh * gw;
    for (int t = 0; t < tokens; ++t) {
        int h = 0, w = 0;
        block_position(t, gh, gw, &h, &w);
        const int y0 = axis_index(h, gh, side), x0 = axis_index(w, gw, side);
        const int y1 = y0 + 1 < side ? y0 + 1 : y0;
        const int x1 = x0 + 1 < side ? x0 + 1 : x0;
        const float fy = axis_frac(h, gh, side), fx = axis_frac(w, gw, side);
        const float* p00 = table + (size_t(y0) * side + x0) * C;
        const float* p01 = table + (size_t(y0) * side + x1) * C;
        const float* p10 = table + (size_t(y1) * side + x0) * C;
        const float* p11 = table + (size_t(y1) * side + x1) * C;
        float* o = out + size_t(t) * C;
        const float w00 = (1.0f - fy) * (1.0f - fx), w01 = (1.0f - fy) * fx;
        const float w10 = fy * (1.0f - fx), w11 = fy * fx;
        for (int c = 0; c < C; ++c)
            o[c] = w00 * p00[c] + w01 * p01[c] + w10 * p10[c] + w11 * p11[c];
    }
}

}  // namespace q4nx_vision
