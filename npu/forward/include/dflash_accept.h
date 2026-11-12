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

// engine#1780: what a DFlash draft block has to clear before it is worth running.
//
// A speculative round offers `k` drafted positions behind one verify pass and yields the positions the target accepted
// plus the one token the verify pass commits past them: `tokens = accepted + bonus`. With `rounds` rounds that is
//
//     tokens_per_round = (accepted + bonus) / rounds
//
// and the round costs `cost_steps` ordinary decode steps, so the draft pays only while `tokens_per_round > cost_steps`.
// A one-pass round (verify only) costs ~1.05 steps and a two-pass round (`mtp::run`: verify, then the step that consumes
// the committed token) costs ~2.05, both measured on the OGA route (docs/oga-drafter-path.md). That is why the same
// acceptance can be a pass in one round shape and a fail in the other, and why this is a function rather than a number.
//
// The `bonus` is not always one per round: when the sequence ends, the final round(s) commit no extra token. Measured on
// the OGA route at k=4 and k=7, the shortfall is 2, 2 and 4 rounds of the 54/46/47 -- which is exactly the difference
// between the ideal `1 + accepted/round` and the published 2.963 / 3.478 / 3.404. Carrying `bonus` explicitly keeps that
// visible instead of hiding it in a rounding difference.
#ifndef ONEBIT_NPU_DFLASH_ACCEPT_H
#define ONEBIT_NPU_DFLASH_ACCEPT_H

#include <cstdint>

namespace onebit {
namespace dflash {

/* Accepted positions per offered draft, the draft's own quality: 108/216 = 0.5 at k=4 on the OGA route. */
inline double per_position_acceptance(std::int64_t accepted, std::int64_t offered) {
    if (offered <= 0) return 0.0;
    return static_cast<double>(accepted) / static_cast<double>(offered);
}

/* Tokens a round yields: the accepted positions plus the tokens the verify pass commits past them. */
inline double tokens_per_round(std::int64_t accepted, std::int64_t bonus, std::int64_t rounds) {
    if (rounds <= 0) return 0.0;
    return static_cast<double>(accepted + bonus) / static_cast<double>(rounds);
}

/* The ideal form, `1 + accepted/round`, which assumes every round earns its bonus. Only for comparison: a measured
 * round whose sequence ended early is lower, and the difference is the shortfall below, not noise. */
inline double ideal_tokens_per_round(std::int64_t accepted, std::int64_t rounds) {
    if (rounds <= 0) return 0.0;
    return 1.0 + static_cast<double>(accepted) / static_cast<double>(rounds);
}

/* Rounds that committed no extra token -- the sequence-ending rounds. 0 for an ideal run. */
inline std::int64_t bonus_shortfall(std::int64_t rounds, std::int64_t bonus) {
    return rounds > bonus ? rounds - bonus : 0;
}

/* The break-even rule. `cost_steps` is the round's cost in ordinary decode steps (1.05 one pass, 2.05 two). */
inline bool pays(double tokens_per_round_value, double cost_steps) {
    return tokens_per_round_value > cost_steps;
}

}  // namespace dflash
}  // namespace onebit

#endif  // ONEBIT_NPU_DFLASH_ACCEPT_H
