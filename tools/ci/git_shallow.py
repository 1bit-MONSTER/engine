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
"""One shallow-clone check for the tools whose whole answer is "nothing was lost" (engine#3106).

WHY
---
`git rev-list --count A..B` returns **0** in a shallow clone for a branch that is genuinely behind, because history is
truncated at the graft. Every distance-derived verdict therefore reads "level with the base", and for a
loss detector that is the one answer it must never give: measured on `pr_branch_stale_check` in a `--depth 1` clone
(engine#3076/#3105), it answered *"merging it removes nothing main has"* — rc 0 — for a branch that was behind 1 **and
conflicted** when merged. A default `actions/checkout` is exactly that environment.

So a tool that reads a distance or a merge-base asks this module first and answers **cannot tell (rc 2)** when the clone
cannot support the measurement. A guard turns a false green into a skip; the CI side of the same fix is `fetch-depth: 0`
on the checkout, because a skip is honest but it is not enforcement.

WHAT IS CHECKED
---------------
`git rev-parse --is-shallow-repository` is the only reliable tell (a grafted history looks complete to everything else).
The decision is a pure function so its table can be asserted without building a shallow clone; `probe()` is the thin
wrapper that runs git, and it answers cannot-tell when git itself cannot be asked — a probe that failed must not read as
"full clone", which is the same rule one level out.
"""
import subprocess
import sys
from pathlib import Path

CANNOT_TELL = 2

# The reason text is the instruction, so it says what to run rather than what went wrong.
SHALLOW_REASON = ("ran in a SHALLOW clone (git rev-parse --is-shallow-repository: true): distances and merge-bases are "
                  "truncated at the graft, so a behind branch reads as level and this answer would be a false green -- "
                  "run `git fetch --unshallow`, or give the CI checkout `fetch-depth: 0`")


def decide(rc, out):
    """None when the clone can support a distance measurement, else the reason it cannot.

    `rc`/`out` are `git rev-parse --is-shallow-repository`'s status and stdout. A failed probe is cannot-tell, never an
    assumed full clone (engine#2932's rule, applied to the probe itself).
    """
    if rc != 0:
        return "cannot tell whether this is a shallow clone (git rev-parse --is-shallow-repository failed)"
    if str(out).strip() == "true":
        return SHALLOW_REASON
    return None


def refusal(root=".", allow_shallow=False):
    """`probe()` plus the escape hatch: an explicit --allow-shallow buys the answer anyway.

    `merged_tail_check` has carried `--allow-shallow` since engine#2659, and it is the right shape: the guard exists to
    stop an ACCIDENTAL shallow run from answering green, not to make a deliberate one impossible. The reason text names
    the flag, so a caller who wants the number through a shallow clone is told how to say so.
    """
    if allow_shallow:
        return None
    return probe(root)


def probe(root="."):
    """`decide()` over the real repository: None, or the reason this tool cannot measure in it."""
    try:
        p = subprocess.run(("git", "-C", str(root), "rev-parse", "--is-shallow-repository"),
                           capture_output=True, text=True)
    except OSError as e:   # git missing: a probe that did not run is not a full clone
        return f"cannot tell whether this is a shallow clone (git rev-parse could not run: {e})"
    return decide(p.returncode, p.stdout)


def self_test():
    ok = True

    def chk(cond, what):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + what)
        ok = ok and cond

    chk(decide(0, "false\n") is None, "a full clone is not a problem")
    chk(decide(0, "true\n") is not None, "a shallow clone is a problem")
    chk(decide(0, "true\n").find("--unshallow") > 0, "the reason names the command that fixes it")
    chk(decide(128, "") is not None and "cannot tell" in decide(128, ""),
        "a failed probe is cannot-tell, not an assumed full clone (engine#2932)")
    chk(decide(0, "\n") is None, "empty output is not 'true' (only the literal true means shallow)")
    chk(decide(0, "True") is None, "the check is on git's own spelling, not a case-insensitive guess")
    chk(refusal("/nonexistent-would-fail", allow_shallow=True) is None,
        "--allow-shallow buys the answer anyway, even when the probe would refuse (merged_tail_check's engine#2659 shape)")

    # ... and the probe against a real depth-1 clone, so the trigger is not only a table.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        src = Path(d, "src")
        src.mkdir()

        def git(*a, cwd):
            return subprocess.run(("git", "-C", str(cwd)) + a, capture_output=True, text=True)

        git("init", "-q", cwd=src)
        Path(src, "a.txt").write_text("one\n")
        git("add", "a.txt", cwd=src)
        git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "one", cwd=src)
        Path(src, "a.txt").write_text("one\ntwo\n")
        git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "two", cwd=src)
        deep = Path(d, "deep")
        subprocess.run(("git", "clone", "-q", "--depth", "1", f"file://{src}", str(deep)), capture_output=True)
        chk(probe(deep) is not None, "a real --depth 1 clone is reported as shallow")
        chk(probe(src) is None, "... and a full clone beside it is not")

    print("git_shallow: " + ("all arms pass" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        sys.exit(self_test())
    why = probe(sys.argv[1] if len(sys.argv) > 1 else ".")
    if why:
        print(f"git_shallow: {why}", file=sys.stderr)
        sys.exit(CANNOT_TELL)
    print("git_shallow: not a shallow clone")
