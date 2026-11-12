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

// spec_greedy.h -- the two decisions of a greedy speculative round, pure (engine#1781).
//
// A round drafts k tokens after the last emitted one, the target verifies [last, d1..dk] in one batched pass, and
// row i of that pass is the target's own choice for the token after position i. So:
//   - accepted = the longest prefix of the drafts that equals the rows' choices: d_{i+1} is kept while row i chose it;
//   - next     = row `accepted`'s choice, the target's own token after the last kept one (always defined, because the
//                pass has k + 1 rows).
// The round emits accepted + 1 tokens, and the caller commits accepted + 1 rows of the verify pass.
//
// The fallback: a request whose last `window` rounds emitted fewer than `below` tokens on average stops speculating
// and finishes with plain greedy steps. On the served 27B a round costs about 3.7 plain steps, so `below` = 3.7 is
// break-even there (engine#1781). It looks only at the last rounds, so early luck does not keep a request speculating.

#pragma once

#include <cstddef>
#include <vector>

namespace onebit::npu {

struct GreedyRound {
    int accepted = 0;   // drafts kept, 0..k
    int next = -1;      // the target's own token after them
};

// drafts[0..k), choices[0..k] (k + 1 rows: the target's argmax per verify row). Returns accepted = -1 when the pass
// has fewer than k + 1 rows or k < 0.
inline GreedyRound greedy_round(const int* drafts, int k, const int* choices, int rows) {
    GreedyRound r;
    if (k < 0 || rows < k + 1 || (k > 0 && !drafts) || !choices) {
        r.accepted = -1;
        return r;
    }
    int a = 0;
    while (a < k && choices[a] == drafts[a]) a++;
    r.accepted = a;
    r.next = choices[a];
    return r;
}

class SpecFallback {
public:
    // below <= 0 or window < 1: never falls back.
    SpecFallback(double below, int window) : below_(below), window_(window) {}
    // Record one round's emitted tokens (accepted + 1). Returns true once the request should stop speculating; stays
    // true from then on.
    bool record(int emitted) {
        if (off_) return true;
        recent_.push_back(emitted);
        if (below_ <= 0.0 || window_ < 1 || int(recent_.size()) < window_) return false;
        double sum = 0.0;
        for (size_t i = recent_.size() - size_t(window_); i < recent_.size(); i++) sum += recent_[i];
        off_ = sum / double(window_) < below_;
        return off_;
    }
    bool off() const { return off_; }
    int rounds() const { return int(recent_.size()); }

private:
    double below_;
    int window_;
    std::vector<int> recent_;
    bool off_ = false;
};

}  // namespace onebit::npu
