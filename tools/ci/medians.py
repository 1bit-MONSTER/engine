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
"""medians.py -- report a median AND the spread, and refuse to call a difference real when it is not.

Engine#679 measured four identical decode runs on an idle box at 18,598 tokens and got
7.94 / 11.78 / 10.39 / 12.57 tok/s: a 1.58x spread on the SAME command. Every figure this lane
published before that was quoted to two or three decimals as if the spread did not exist, and
item 3a's "2.0x slower decode" turned out to be that spread showing up in the first rep.

So: report median, min, max and the ratio, and answer the only question a reader actually has --
is arm A's median outside arm B's observed range, or is the difference inside the noise?

    medians.py 7.94 11.78 10.39 12.57                  # one arm
    medians.py --a 7.94 11.78 10.39 12.57 --b 9.1 9.3 # two arms, verdict
    medians.py --self-test

A verdict of NOISE is not a failure of the measurement; it is the measurement telling you to
take more samples or to stop quoting the difference.
"""
import statistics
import sys


def spread(values):
    """(median, min, max, max/min) for a non-empty list of positive floats."""
    if not values:
        raise ValueError("no values")
    lo, hi = min(values), max(values)
    if lo <= 0:
        raise ValueError("values must be positive to express a spread as a ratio")
    return statistics.median(values), lo, hi, hi / lo


def verdict(a, b):
    """'A' / 'B' when one median is outside the other's observed range, else 'NOISE'.

    Disjoint ranges are the only evidence a handful of samples can give: if A's median sits
    inside B's min..max, then B produced that number by itself and the difference needs more
    runs before it means anything.
    """
    _, alo, ahi, _ = spread(a)
    _, blo, bhi, _ = spread(b)
    # Disjoint ranges, not disjoint medians: if the loser's MAX reaches into the winner's
    # range, the loser produced that value on its own and the difference is not yet evidence.
    # (The first version compared medians and passed [5.0] against [5.0, 6.0] as "B" -- B had
    #  produced 5.0 itself. The self-test below is the arm that caught it.)
    if alo > bhi:
        return "A"
    if blo > ahi:
        return "B"
    return "NOISE"


def self_test():
    fails = 0

    def check(ok, what):
        nonlocal fails
        if not ok:
            fails += 1
            print(f"FAIL {what}")

    # the real data that produced this file: four identical runs, 1.58x apart
    med, lo, hi, ratio = spread([7.94, 11.78, 10.39, 12.57])
    check(abs(med - (10.39 + 11.78) / 2) < 1e-9, "median of four is the mean of the middle two")
    check(lo == 7.94 and hi == 12.57, "min and max are the observed ones")
    check(abs(ratio - 1.583) < 0.01, "the 1.58x spread of identical runs is reported as such")

    # two arms whose ranges overlap cannot be separated by these samples
    check(verdict([10.0, 11.0, 12.0], [10.5, 11.5]) == "NOISE", "overlapping ranges are noise")
    # an arm whose whole range sits above the other's is a real difference
    check(verdict([20.0, 21.0], [10.0, 11.0]) == "A", "disjoint above is A")
    check(verdict([10.0, 11.0], [20.0, 21.0]) == "B", "disjoint below is B")
    # touching ranges are not disjoint: B's max equals A's min means B can reach it
    check(verdict([20.0, 20.0], [19.0, 20.0]) == "NOISE", "a shared endpoint is not evidence")

    # a single sample has no range, so it can never be separated from anything
    check(verdict([5.0], [5.0, 6.0]) == "NOISE", "one sample cannot beat two")

    check(ratio != 1.0, "sanity: the real data is not a flat line")

    # spread() refuses what it cannot express
    for bad in ([], [0, 1]):
        try:
            spread(bad)
        except ValueError:
            pass
        else:
            fails += 1
            print(f"FAIL spread({bad}) should raise")

    print(f"self-test: {'PASS' if fails == 0 else str(fails) + ' FAILED'}")
    return 0 if fails == 0 else 1


def main(argv):
    if "--self-test" in argv:
        return self_test()
    if "--a" in argv and "--b" in argv:
        i, j = argv.index("--a"), argv.index("--b")
        a = [float(x) for x in argv[i + 1 : j]]
        b = [float(x) for x in argv[j + 1 :]]
    else:
        a, b = [float(x) for x in argv], None
    if not a:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        return 2
    med, lo, hi, ratio = spread(a)
    label = "A" if b is not None else "arm"
    print(f"{label}: median {med:.3f}  min {lo:.3f}  max {hi:.3f}  spread {ratio:.2f}x  (n={len(a)})")
    if b is None:
        return 0
    if not b:
        print("B: no samples", file=sys.stderr)
        return 2
    mb, blo, bhi, bratio = spread(b)
    print(f"B: median {mb:.3f}  min {blo:.3f}  max {bhi:.3f}  spread {bratio:.2f}x  (n={len(b)})")
    result = verdict(a, b)
    # a difference inside the observed spread is not a result
    print(f"verdict: {result}" + ("" if result != "NOISE" else " -- the ranges overlap, take more samples"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
