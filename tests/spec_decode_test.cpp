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
// spec_decode_test -- the speculative round's bookkeeping (engine#1781). No model, no device, no RNG: `decide()` is
// pure and takes its uniforms from the caller, so every arm below is a hand-computed expected value rather than a
// tolerance. Five arms, and (v) is the one that matters most: with no provider the loop must behave exactly as it did
// before the wiring existed.
//
//   (i)   a partially-accepted round: emitted = accepted ++ resample, counts exact
//   (ii)  the chain stops at the FIRST rejection, so later drafts are never emitted
//   (iii) a fully-accepted round emits all k and resamples nothing
//   (iv)  q(x) == 0 rejects and resamples (the premise `x was drawn from q` is false) instead of dividing
//   (v)   the no-drafter path is the caller's untouched branch -- asserted here as "decide() refuses with no drafts",
//         which is the condition the loop keys on
#include "spec_decode.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

// Two-row fixtures at V = 4. p is the target, q the drafter. Both rows are distributions that sum to 1.
static const float kP[2][4] = {{0.70f, 0.10f, 0.10f, 0.10f}, {0.10f, 0.70f, 0.10f, 0.10f}};
static const float kQ[2][4] = {{0.10f, 0.70f, 0.10f, 0.10f}, {0.70f, 0.10f, 0.10f, 0.10f}};

int main() {
    const int V = 4;

    // (i) two drafts, both accepted: p[x]=0.70 against q[x]=0.10, so min(1,p/q) is 1 and u<1 always accepts.
    {
        const std::vector<int> drafts{0, 1};
        const double ua[2] = {0.99, 0.99}, ur[2] = {0.5, 0.5};
        const onebit::spec::SpecRound r = onebit::spec::decide(&kP[0][0], &kQ[0][0], V, drafts, ua, ur);
        check(r.proposed == 2 && r.accepted == 2 && !r.stopped_on_rejection && r.emitted.size() == 2 &&
                  r.emitted[0] == 0 && r.emitted[1] == 1,
              "(i) both accepted -> emitted = the two drafts, nothing resampled");
    }
    // (ii) first draft accepted, second rejected. Both rows propose token 0: row 0 has p=0.70,q=0.10 so min(1,p/q)=1
    // (accepts for any u), row 1 has p=0.10,q=0.70 so min(1,p/q)=1/7 and u=0.99 > 1/7 rejects. Proposing token 1 on row 1
    // would have been accepted (p/q = 7) -- which is what the first version of this fixture did, and the arm said so.
    {
        const std::vector<int> drafts{0, 0};
        const double ua[2] = {0.01, 0.99}, ur[2] = {0.5, 0.5};
        const onebit::spec::SpecRound r = onebit::spec::decide(&kP[0][0], &kQ[0][0], V, drafts, ua, ur);
        check(r.proposed == 2 && r.accepted == 1 && r.stopped_on_rejection && r.emitted.size() == 2,
              "(ii) one accepted then a rejection -> emitted = accepted ++ the resample");
        check(r.emitted[0] == 0 && r.emitted[1] >= 0 && r.emitted[1] < V,
              "(ii) emitted[0] is the accepted draft and emitted[1] is a valid resampled token");
    }
    // (iii) three drafts at a row where p == q: min(1, p/q) == 1, so every u accepts and nothing is resampled.
    {
        const float same[3][4] = {{0.25f, 0.25f, 0.25f, 0.25f}, {0.25f, 0.25f, 0.25f, 0.25f},
                                  {0.25f, 0.25f, 0.25f, 0.25f}};
        const std::vector<int> drafts{0, 1, 2};
        const double ua[3] = {0.9, 0.9, 0.9}, ur[3] = {0.5, 0.5, 0.5};
        const onebit::spec::SpecRound r = onebit::spec::decide(&same[0][0], &same[0][0], V, drafts, ua, ur);
        check(r.proposed == 3 && r.accepted == 3 && r.emitted.size() == 3 && !r.stopped_on_rejection,
              "(iii) p == q -> all k accepted, no resample, emitted.size() == k");
    }
    // (iv) q(x) == 0: the draft was not drawn from q, so the step must reject and resample rather than divide by zero.
    {
        const float q0[1][4] = {{0.0f, 0.5f, 0.5f, 0.0f}};
        const float p0[1][4] = {{0.9f, 0.05f, 0.05f, 0.0f}};
        const std::vector<int> drafts{0};
        const double ua[1] = {0.0}, ur[1] = {0.0};
        const onebit::spec::SpecRound r = onebit::spec::decide(&p0[0][0], &q0[0][0], V, drafts, ua, ur);
        check(r.proposed == 1 && r.accepted == 0 && r.stopped_on_rejection && r.emitted.size() == 1,
              "(iv) q(x) == 0 rejects instead of dividing, and still emits one resampled token");
    }
    // (v) the no-drafter path: no drafts is exactly the condition the loop keys the speculative branch on, so decide()
    //     must refuse for it. The caller's argmax branch is untouched by this header -- that is the guarantee.
    {
        const double ua[1] = {0.5}, ur[1] = {0.5};
        const onebit::spec::SpecRound r = onebit::spec::decide(&kP[0][0], &kQ[0][0], V, std::vector<int>{}, ua, ur);
        check(r.proposed == 0 && r.accepted == 0 && r.emitted.empty(),
              "(v) no drafts -> refuses, so the loop keeps its own argmax path");
    }
    // and the seam's other refusal: uniforms outside [0,1) are refused, not clamped -- a caller that forgot to seed its
    // RNG must not get a silent pass.
    {
        const std::vector<int> drafts{0};
        const double bad[1] = {1.0}, ok[1] = {0.5};
        const onebit::spec::SpecRound r1 = onebit::spec::decide(&kP[0][0], &kQ[0][0], V, drafts, bad, ok);
        const onebit::spec::SpecRound r2 = onebit::spec::decide(&kP[0][0], &kQ[0][0], V, drafts, ok, bad);
        check(r1.proposed == 0 && r2.proposed == 0, "a uniform of exactly 1.0 (or outside [0,1)) is refused, not clamped");
    }

    if (failures == 0) std::printf("spec_decode: ok\n");
    return failures ? 1 : 0;
}
