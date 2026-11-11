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

// engine#1780: known answers for the draft block's acceptance arithmetic, taken from the OGA-route measurement rather
// than invented. The three rows are docs/oga-drafter-path.md's counts: k=4 over 54 rounds, and k=7 over two replays.
// Host-only: no container, no device, no NPU runtime -- so it runs wherever this file is built.
#include "dflash_accept.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace df = onebit::dflash;

static int failures = 0;

static void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

static bool near(double a, double b) { return std::fabs(a - b) < 5e-4; }

int main() {
    /* The three measured rows. `tokens` is what the tool printed; `accepted` is the drafts the target took. */
    struct Row { int k; std::int64_t offered, rounds, accepted, tokens; double want_train; };
    const Row rows[] = {
        {4, 216, 54, 108, 160, 2.963},
        {7, 322, 46, 116, 160, 3.478},
        {7, 329, 47, 117, 160, 3.404},
    };
    for (const Row& r : rows) {
        const std::int64_t bonus = r.tokens - r.accepted;
        const double tpr = df::tokens_per_round(r.accepted, bonus, r.rounds);
        check(near(tpr, r.want_train),
              "accept: k=" + std::to_string(r.k) + " reproduces the published tokens/round (" + std::to_string(tpr) + ")");
        /* tokens = accepted + bonus is exact, so the shortfall is exactly the rounds that ended the sequence. */
        check(r.accepted + bonus == r.tokens, "accept: tokens == accepted + bonus (the decomposition is exact)");
        const std::int64_t shortfall = df::bonus_shortfall(r.rounds, bonus);
        const double ideal = df::ideal_tokens_per_round(r.accepted, r.rounds);
        check(near(ideal - static_cast<double>(shortfall) / static_cast<double>(r.rounds), tpr),
              "accept: the gap from the ideal form is exactly shortfall/rounds, not rounding");
        check(shortfall > 0 && shortfall <= 4,
              "accept: and the shortfall is a handful of sequence-ending rounds (" + std::to_string(shortfall) + ")");
    }

    /* Per-position acceptance, the draft's own quality: 0.5 at k=4, ~0.36 at k=7. */
    check(near(df::per_position_acceptance(108, 216), 0.5), "accept: 108 of 216 offers is 0.5 per position");
    check(near(df::per_position_acceptance(116, 322), 0.3602), "accept: 116 of 322 is 0.3602");

    /* The break-even rule, and the case that makes it a function: the same acceptance passes one round shape and
     * fails the other. 2.963 tokens/round against 1.05 steps pays; against 2.05 it barely pays; a draft at 1.5 does not. */
    check(df::pays(3.000, 1.05), "accept: 3.0 tokens/round pays for a one-pass round (1.05 steps)");
    check(df::pays(3.000, 2.05), "accept: 3.0 tokens/round also pays for a two-pass round (2.05 steps)");
    check(!df::pays(1.500, 2.05), "accept: 1.5 tokens/round does NOT pay for a two-pass round -- the distinction");
    check(df::pays(1.500, 1.05), "accept: but it does pay for a one-pass round");
    /* A drafter that accepts nothing is the identity, 1.0 token/round, and must never be reported as a win. */
    check(near(df::tokens_per_round(0, 54, 54), 1.0), "accept: a drafter that accepts nothing gives 1.0 token/round");
    check(!df::pays(1.0, 1.0), "accept: and 1.0 does not pay for a round that costs a step (the rule is strict)");
    /* Degenerate inputs are answers, not crashes: a forward can be handed a zero-length run. */
    check(df::tokens_per_round(0, 0, 0) == 0.0 && df::per_position_acceptance(0, 0) == 0.0 && df::bonus_shortfall(0, 0) == 0,
          "accept: zero rounds, zero offers and zero bonus are answers rather than division by zero");

    if (failures == 0) std::printf("dflash accept: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
