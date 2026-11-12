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
// lfm2_shortconv_test: the gated short convolution (npu/forward/include/lfm2_shortconv.h) against the definition
// written out directly, with no model: a causal depthwise convolution over the whole sequence, zero history before
// position 0. Token-by-token, in one batch, and in uneven batches must all give the same bits.

#include "lfm2_shortconv.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using onebit::npu::ShortConvState;
using onebit::npu::shortconv_rows;
using onebit::npu::shortconv_step;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL %s\n", what.c_str());
        ++failures;
    }
}

// Deterministic inputs with a little of everything: both signs, zeros, different scales per channel.
static float bcx_of(int t, int i, int H) {
    const int third = i / H, h = i % H;
    float v = std::sin(0.37f * float(t + 1) * float(h + 1) + 1.3f * float(third));
    if ((t + h + third) % 7 == 0) v = 0.0f;
    return v * (1.0f + 0.25f * float(h % 4));
}
static float tap_of(int h, int j) { return 0.5f * std::cos(0.9f * float(h + 1) + 1.7f * float(j)) + (j == 0 ? 0.125f : 0.0f); }

// The definition, on the whole sequence at once: y[t][h] = C[t][h] * sum_j taps[h][j] * bx[t - (L-1) + j][h], bx = 0 before 0.
// The sum runs in the same order as the code under test, so equality can be exact.
static std::vector<float> reference(int T, int H, int L, const std::vector<float>& bcx, const std::vector<float>& taps) {
    std::vector<float> y(size_t(T) * size_t(H));
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < L; ++j) {
                const int src = t - (L - 1) + j;
                const float bx = src < 0 ? 0.0f : bcx[size_t(src) * 3 * H + h] * bcx[size_t(src) * 3 * H + 2 * H + h];
                acc += taps[size_t(h) * L + j] * bx;
            }
            y[size_t(t) * H + h] = bcx[size_t(t) * 3 * H + H + h] * acc;
        }
    return y;
}

int main() {
    for (int L : {1, 2, 3, 4}) {
        const int H = 37, T = 23;
        std::vector<float> bcx(size_t(T) * 3 * H), taps(size_t(H) * L);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < 3 * H; ++i) bcx[size_t(t) * 3 * H + i] = bcx_of(t, i, H);
        for (int h = 0; h < H; ++h)
            for (int j = 0; j < L; ++j) taps[size_t(h) * L + j] = tap_of(h, j);
        const std::vector<float> want = reference(T, H, L, bcx, taps);
        const std::string tag = " (L=" + std::to_string(L) + ")";

        ShortConvState one;
        one.reset(H, L);
        check(one.ready() && one.history.size() == size_t(L - 1) * H, "the state is L-1 rows of H" + tag);
        std::vector<float> got(size_t(T) * H);
        bool ok = true;
        for (int t = 0; t < T; ++t) ok = ok && shortconv_step(one, &bcx[size_t(t) * 3 * H], taps.data(), &got[size_t(t) * H]);
        check(ok && std::memcmp(got.data(), want.data(), want.size() * sizeof(float)) == 0,
              "token by token it is the causal depthwise convolution of B*X, gated by C, bit for bit" + tag);

        ShortConvState all;
        all.reset(H, L);
        std::vector<float> batched(size_t(T) * H);
        check(shortconv_rows(all, bcx.data(), T, taps.data(), batched.data()) &&
                  std::memcmp(batched.data(), want.data(), want.size() * sizeof(float)) == 0,
              "one batch of every token gives the same bits" + tag);
        check(all.history == one.history, "and leaves the same state behind" + tag);

        ShortConvState mixed;   // a prefill in uneven chunks, then decode: 5 + 1 + 9 + the rest one at a time
        mixed.reset(H, L);
        std::vector<float> m(size_t(T) * H);
        int at = 0;
        bool mok = true;
        for (int chunk : {5, 1, 9}) {
            mok = mok && shortconv_rows(mixed, &bcx[size_t(at) * 3 * H], chunk, taps.data(), &m[size_t(at) * H]);
            at += chunk;
        }
        for (; at < T; ++at) mok = mok && shortconv_step(mixed, &bcx[size_t(at) * 3 * H], taps.data(), &m[size_t(at) * H]);
        check(mok && std::memcmp(m.data(), want.data(), want.size() * sizeof(float)) == 0 && mixed.history == one.history,
              "uneven prefill chunks followed by decode steps give the same bits and state" + tag);

        // a snapshot of the state is all that a resume needs: copy it at position 11, replay from there
        ShortConvState first;
        first.reset(H, L);
        std::vector<float> scratch(size_t(T) * H);
        shortconv_rows(first, bcx.data(), 11, taps.data(), scratch.data());
        ShortConvState resumed = first;
        std::vector<float> tail(size_t(T - 11) * H);
        check(shortconv_rows(resumed, &bcx[size_t(11) * 3 * H], T - 11, taps.data(), tail.data()) &&
                  std::memcmp(tail.data(), &want[size_t(11) * H], tail.size() * sizeof(float)) == 0,
              "a copy of the state at position 11 resumes to the same bits" + tag);

        // a rolled-back batch: run all T rows from a copy, put the copy back, re-apply only the first 7 rows' B*X.
        // That must be the state 7 decode steps leave, and decode must carry on from it to the same bits.
        {
            ShortConvState base;
            base.reset(H, L);
            std::vector<float> junk(size_t(T) * H);
            shortconv_rows(base, bcx.data(), 4, taps.data(), junk.data());       // 4 tokens already committed
            const ShortConvState snap = base;
            shortconv_rows(base, &bcx[size_t(4) * 3 * H], T - 4, taps.data(), junk.data());   // the speculative batch
            base = snap;                                                           // rejected past row 7
            bool pok = true;
            std::vector<float> bx(H);
            for (int t = 4; t < 11; ++t) {
                for (int h = 0; h < H; ++h) bx[h] = bcx[size_t(t) * 3 * H + h] * bcx[size_t(t) * 3 * H + 2 * H + h];
                pok = pok && onebit::npu::shortconv_push(base, bx.data());
            }
            std::vector<float> rest(size_t(T - 11) * H);
            pok = pok && shortconv_rows(base, &bcx[size_t(11) * 3 * H], T - 11, taps.data(), rest.data());
            check(pok && base.history == one.history &&
                      std::memcmp(rest.data(), &want[size_t(11) * H], rest.size() * sizeof(float)) == 0,
                  "restore + push of the accepted rows' B*X equals having decoded them" + tag);
            ShortConvState unsized;
            check(!onebit::npu::shortconv_push(unsized, bx.data()) && !onebit::npu::shortconv_push(base, nullptr),
                  "push refuses an unsized state and a null row" + tag);
        }

        // the history really is the last L-1 values of B*X, oldest first
        bool hist = true;
        for (int j = 0; j < L - 1 && hist; ++j)
            for (int h = 0; h < H; ++h) {
                const int src = T - (L - 1) + j;
                const float bx = bcx[size_t(src) * 3 * H + h] * bcx[size_t(src) * 3 * H + 2 * H + h];
                if (one.history[size_t(j) * H + h] != bx) { hist = false; break; }
            }
        check(hist, "the state is the last L-1 values of B*X, oldest first" + tag);
    }
    {   // the present token meets the LAST tap, and the gate C multiplies the result, not the input
        ShortConvState st;
        st.reset(1, 3);
        const float taps[3] = {100.0f, 10.0f, 1.0f};
        float y = 0.0f;
        const float t0[3] = {2.0f, 5.0f, 3.0f};   // B, C, X -> bx 6
        check(shortconv_step(st, t0, taps, &y) && y == 5.0f * (1.0f * 6.0f), "at position 0 only the last tap sees anything");
        const float t1[3] = {1.0f, 2.0f, 4.0f};   // bx 4
        check(shortconv_step(st, t1, taps, &y) && y == 2.0f * (10.0f * 6.0f + 1.0f * 4.0f), "the token before meets the middle tap");
        const float t2[3] = {1.0f, 1.0f, 1.0f};   // bx 1
        check(shortconv_step(st, t2, taps, &y) && y == 1.0f * (100.0f * 6.0f + 10.0f * 4.0f + 1.0f * 1.0f),
              "and the one before that the first tap");
    }
    {   // refusals: an unsized state, null buffers, a negative row count
        ShortConvState st;
        float b[3] = {1, 1, 1}, k[1] = {1}, y = 7.0f;
        check(!shortconv_step(st, b, k, &y) && y == 7.0f, "an unsized state is refused and nothing is written");
        st.reset(1, 1);
        check(!shortconv_step(st, nullptr, k, &y) && !shortconv_step(st, b, nullptr, &y) && !shortconv_step(st, b, k, nullptr),
              "null buffers are refused");
        check(!shortconv_rows(st, b, -1, k, &y) && shortconv_rows(st, b, 0, k, &y), "a negative row count is refused; zero rows is a no-op");
    }
    if (failures == 0) std::printf("lfm2_shortconv: ok\n");
    return failures == 0 ? 0 : 1;
}
