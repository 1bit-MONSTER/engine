#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate the non-finite-masked-V extension of tests/test-hrx-fa-masked-v.cpp.

Run inside the fork tree (default ~/wt/llama-315copy). Writes
/tmp/315-fa-masked-v-nonfinite.patch. Every anchor is asserted to occur exactly once, so a silent
no-op is impossible.

Why the extension: variants 0-4 rewrite the masked K/V rows with finite values only, so a kernel that
still consumes masked V rows can betray itself by as little as one F16 ulp. Variants 5-7 put NaN,
+inf and -inf into the masked V rows, where a correct kernel is provably insensitive (the design
zeroes those rows, so the product never sees the value) and a leaking kernel produces NaN/inf in the
output. Masked K stays finite on purpose: the masked score is `raw + mask` with mask = -inf, and
-inf + NaN is NaN, so a NaN masked K row poisons even a correct kernel.
"""
import pathlib
import subprocess
import sys

REL = "tests/test-hrx-fa-masked-v.cpp"
src = pathlib.Path(REL)
orig = src.read_text()
new = orig

A_OLD = """        // Rewrite only the rows past the sequence end, five ways.
        int mismatches = 0;
        for (int variant = 0; variant < 5; ++variant) {
"""

A_NEW = """        // Rewrite only the rows past the sequence end. Variants 0-4 are finite, as in the original suite
        // (random, +x, -x, +0, -0), and can only betray a leak by a fraction of an ulp. Variants 5-7 are
        // non-finite and go into V only: V reaches the output solely as P*V, and this kernel's design is to
        // zero the V rows of masked keys, so a correct kernel is insensitive to whatever sits there - even
        // NaN, because the zeroed row is exactly +0. A kernel that still consumes masked V rows turns those
        // variants into NaN/inf in the output.
        //
        // K is deliberately kept finite: the masked score is `raw + mask` with mask = -inf, and -inf + NaN
        // is NaN, so a NaN in a masked K row poisons even a correct kernel and would be a false positive.
        int mismatches = 0;
        int nonfinite  = 0;
        for (int variant = 0; variant < 8; ++variant) {
"""

B_OLD = """                switch (variant) {
                    case 0: kv = 3.0f * normal(stale_rng); vv = 3.0f * normal(stale_rng); break;
                    case 1: kv = 30.0f; vv = 1.0f; break;
                    case 2: kv = -30.0f; vv = -1.0f; break;
                    case 3: kv = 0.0f; vv = 0.0f; break;
                    default: kv = -0.0f; vv = -0.0f; break;
                }
"""

B_NEW = """                switch (variant) {
                    case 0: kv = 3.0f * normal(stale_rng); vv = 3.0f * normal(stale_rng); break;
                    case 1: kv = 30.0f; vv = 1.0f; break;
                    case 2: kv = -30.0f; vv = -1.0f; break;
                    case 3: kv = 0.0f; vv = 0.0f; break;
                    case 4: kv = -0.0f; vv = -0.0f; break;
                    case 5: kv = 1.0f; vv = std::nanf(""); break;
                    case 6: kv = 1.0f; vv = INFINITY; break;
                    default: kv = 1.0f; vv = -INFINITY; break;
                }
"""

C_OLD = """            const std::vector<float> got = compute(hrx, g, q, k2, v2, mask);
            if (std::memcmp(got.data(), reference.data(), got.size() * sizeof(float)) != 0) {
                ++mismatches;
            }
"""

C_NEW = """            const std::vector<float> got = compute(hrx, g, q, k2, v2, mask);
            if (std::memcmp(got.data(), reference.data(), got.size() * sizeof(float)) != 0) {
                ++mismatches;
            }
            for (const float x : got) {
                if (!std::isfinite(x)) {
                    ++nonfinite;
                }
            }
"""

D_OLD = """        const bool   ok    = mismatches == 0 && error <= kMaxNmse;
"""

D_NEW = """        const bool   ok    = mismatches == 0 && nonfinite == 0 && error <= kMaxNmse;
"""

E_OLD = """        std::printf("%s case %zu q=%lld visible=%lld cells=%lld (%s): masked-row variants differing %d/5, nmse %.3g\\n",
                    ok ? "ok  " : "FAIL", ci, static_cast<long long>(c.queries), static_cast<long long>(c.visible),
                    static_cast<long long>(c.cells), plan.c_str(), mismatches, error);
"""

E_NEW = """        std::printf("%s case %zu q=%lld visible=%lld cells=%lld (%s): masked-row variants differing %d/8, "
                    "non-finite outputs %d, nmse %.3g\\n",
                    ok ? "ok  " : "FAIL", ci, static_cast<long long>(c.queries), static_cast<long long>(c.visible),
                    static_cast<long long>(c.cells), plan.c_str(), mismatches, nonfinite, error);
        if (nonfinite != 0) {
            std::printf("     ^ masked V rows set to NaN/+inf/-inf reached the output: the kernel consumed a "
                        "row the mask hides\\n");
        }
"""

for tag, old, rep in (("A", A_OLD, A_NEW), ("B", B_OLD, B_NEW), ("C", C_OLD, C_NEW), ("D", D_OLD, D_NEW), ("E", E_OLD, E_NEW)):
    count = new.count(old)
    if count != 1:
        sys.exit(f"ABORT: anchor {tag} matched {count} times, expected 1")
    new = new.replace(old, rep)

tmp = pathlib.Path("/tmp/test-hrx-fa-masked-v.cpp.new")
tmp.write_text(new)
res = subprocess.run(
    ["diff", "-u", f"--label=a/{REL}", f"--label=b/{REL}", str(src), str(tmp)],
    capture_output=True, text=True,
)
if res.returncode != 1:
    sys.exit(f"ABORT: diff returned {res.returncode}: {res.stderr}")
patch = pathlib.Path("/tmp/315-fa-masked-v-nonfinite.patch")
patch.write_text(res.stdout)
print(f"wrote {patch} ({len(res.stdout)} bytes, {res.stdout.count(chr(10))} lines)")
