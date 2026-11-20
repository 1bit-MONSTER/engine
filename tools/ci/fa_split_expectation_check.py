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
"""fa_split_expectation_check.py -- a test cannot expect the decode-split FA kernel without the fixture that makes it legal.

engine#2870, measured on the pinned fork:

    a1f525117 (before #418)   tests expect ggml_flash_attention_decode_split_f32_f16_wmma_next_q8, the test file has
                              ZERO `key_value_capacity`, and the matcher refuses the split unless
                              `key->ne[1] >= ceil_div(kv, 64) * 64`
    fe820a0bf (now)           the fixture computes and allocates at that padded capacity, so the guard passes

The 13 rows in that issue were not a matcher bug: the fixture handed the matcher 65 slots for a 128-slot walk, the
guard declined (correctly, #418 exists because the unguarded claim faulted the device), and the general kernel was
scheduled. So the invariant this checks is a JOIN of two files, and each side has its own failure:

  fixture not padded   the expectation can never be satisfied -- the matcher will decline for storage it does not have
  guard missing        the matcher may CLAIM storage it was never proven to have, which is the fault #418 removed

A check that only looked at one side would pass in both of those states, which is why it looks at both.

Exit: 0 the expectation is satisfiable, 1 one side is missing (named), 2 cannot tell (a file this needs is absent).
"""
import argparse
import sys
from pathlib import Path

SPLIT = "ggml_flash_attention_decode_split_f32_f16_wmma_next_q8"
# The padding the matcher compares storage against: ceil_div(key_value_token_count, 64) * 64, which the fixture has to
# reproduce. Matched loosely on purpose (whitespace, constant spelling) -- this asserts the SHAPE is there, not its style.
# Whitespace-normalised before matching: the real expression is LINE-WRAPPED in the helper
# (`const int64_t key_value_capacity =` then `((key_value_token_count + 63) / 64) * 64;`), and a single-line pattern
# silently missed it -- the first version of this check failed on the very tree it was written to pass.
# Two spellings, because the fixture writes the tile size as a NAMED CONSTANT and pads only for the split kernel:
#     const int64_t key_value_capacity =
#         expected_kernel == kDecodeSplitKernelName
#             ? ((key_value_token_count + kDecodeKvTileSize - 1) / kDecodeKvTileSize) * kDecodeKvTileSize
#             : key_value_token_count;
# A literal-64-only pattern missed it (the first version of this check failed on the tree it was written to pass).
PADDING = (
    "+ kDecodeKvTileSize - 1) / kDecodeKvTileSize) * kDecodeKvTileSize",
    "(key_value_token_count + 63) / 64) * 64",
)


def _flat(text):
    return " ".join(text.split())
GUARD = ("key->ne[1] < match.key_value_capacity", "value->ne[1] < match.key_value_capacity")

TEST_PATH = Path("tests/hrx-ops-test.cpp")
DISPATCH_PATH = Path("ggml/src/ggml-hrx/dispatch_registration/common/dispatch-flash-attention.cpp")


def check(root: Path):
    """(failures, cannot_tell). A failure is (side, what)."""
    test = root / TEST_PATH
    dispatch = root / DISPATCH_PATH
    try:
        test_text = test.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], f"no {test}"
    if SPLIT not in test_text:
        return [], None  # nothing expects the split kernel: the invariant is vacuous, and that is not a failure
    try:
        dispatch_text = dispatch.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], f"no {dispatch} (the test expects the split kernel, so the matcher has to be readable)"

    flat_test = _flat(test_text)
    failures = []
    if not any(p in flat_test for p in PADDING):
        failures.append(("fixture", "the test expects the split kernel and never computes the padded capacity, so the "
                                    "fixture allocates at the token count and the matcher will decline for storage it "
                                    "does not have (engine#2870: 65 slots for a 128-slot walk)"))
    # both, not either: `any()` passed a matcher that kept the value guard and dropped the key one, which is half of
    # the fault engine#418 removed -- measured on 11-20 by deleting each guard in turn from a copy of the fork's file
    if not all(g in dispatch_text for g in GUARD):
        failures.append(("matcher", "the test expects the split kernel and the matcher no longer compares storage "
                                    "against the padded capacity, so it may claim a walk past the binding (the fault "
                                    "#418 removed)"))
    return failures, None


def self_test() -> int:
    import tempfile

    fails = 0

    def arm(name, ok, detail=""):
        nonlocal fails
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail and not ok else ""))
        if not ok:
            fails += 1

    def build(tmp, test_text, guard: bool):
        root = tmp / f"r{len(list(tmp.iterdir()))}"
        (root / TEST_PATH.parent).mkdir(parents=True)
        (root / DISPATCH_PATH.parent).mkdir(parents=True)
        (root / TEST_PATH).write_text(test_text, encoding="utf-8")
        # both comparisons, because the invariant is both: `any()` used to pass a matcher that kept the value guard and
        # dropped the key one (measured 11-20 by deleting each in turn from a copy of the fork's dispatch file)
        g = ("if (key->ne[1] < match.key_value_capacity) { return {}; }\n"
             "if (value->ne[1] < match.key_value_capacity) { return {}; }\n") if guard else "// no storage guard\n"
        (root / DISPATCH_PATH).write_text("namespace { void f() {\n" + g + "} }\n", encoding="utf-8")
        return root

    expecting = f'require_kernel_subsequence(s, {{ "{SPLIT}" }});\n'
    # Line-wrapped exactly as the real helper writes it, so the arm exercises the normalisation above.
    padded = expecting + (
        "const int64_t key_value_capacity =\n"
        "    expected_kernel == kDecodeSplitKernelName\n"
        "        ? ((key_value_token_count + kDecodeKvTileSize - 1) / kDecodeKvTileSize) * kDecodeKvTileSize\n"
        "        : key_value_token_count;\n")

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ok_root = build(tmp, padded, guard=True)
        got, why = check(ok_root)
        arm("a padded fixture beside the guard passes", got == [] and why is None, str(why or got))

        unpadded = build(tmp, expecting, guard=True)  # exactly a1f525117's state
        got, why = check(unpadded)
        arm("a fixture that pads nothing is a FAILURE naming the fixture",
            [s for s, _ in got] == ["fixture"], str(got))

        # half the invariant is a failure: this is the arm the old `any()` could not express
        half = tmp / "half"
        (half / TEST_PATH.parent).mkdir(parents=True)
        (half / TEST_PATH).write_text(padded, encoding="utf-8")
        (half / DISPATCH_PATH.parent).mkdir(parents=True)
        (half / DISPATCH_PATH).write_text(
            "namespace { void f() {\nif (key->ne[1] < match.key_value_capacity) { return {}; }\n} }\n", encoding="utf-8")
        got, _ = check(half)
        arm("a matcher that kept the key guard and dropped the value one is a FAILURE",
            [s for s, _ in got] == ["matcher"], str(got))

        w = build(tmp, padded, guard=False)
        got, why = check(w)
        arm("a matcher without the storage guard is a FAILURE naming the matcher",
            [s for s, _ in got] == ["matcher"], str(got))

        none = build(tmp, "// no split expectation here\n", guard=False)
        got, why = check(none)
        arm("no expectation at all is vacuous, not a failure", got == [] and why is None, str(got))

        empty = tmp / "empty"
        (empty / TEST_PATH.parent).mkdir(parents=True)
        got, why = check(empty)
        arm("a missing test file is cannot-tell", got == [] and why is not None, str(why))

        (empty / TEST_PATH).write_text(expecting, encoding="utf-8")
        got, why = check(empty)
        arm("an expectation with no readable matcher is cannot-tell, not a pass", got == [] and why is not None, str(why))

    print("self-test ok" if not fails else f"self-test FAILED ({fails})")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".", help="a llama.cpp checkout (default: .)")
    ap.add_argument("--self-test", action="store_true", help="hermetic fixtures, no checkout needed")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.check:
        ap.print_usage()
        return 2
    failures, why = check(Path(a.root))
    if why:
        print(f"fa-split-expectation: cannot tell: {why}", file=sys.stderr)
        return 2
    for side, what in failures:
        print(f"  {side}: {what}")
    if failures:
        return 1
    print("fa-split-expectation: every decode-split expectation has the padded fixture and the storage guard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
