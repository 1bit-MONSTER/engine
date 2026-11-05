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
#
# vm-check's row verdict, in ONE place so the live driver and its self-test cannot disagree.
#
# Rows are (name, shell-command, wanted-substring-or-None). The shell command runs in the guest and its body is compared
# against the wanted substring. That comparison is a plain `want in body`, and engine #1111 found the hole in it: a row
# that cannot run says why with a marker like `console-heals-skip-no-tui`, and such a marker CONTAINS the row's own want
# string (`console-heals`), so a skipped row printed **PASS** - the one verdict that must never be wrong, on a row that
# never ran.
#
# The rule: a body carrying a `*-skip-*` token is a SKIP, checked BEFORE the want-in-body test. `skipping()` is the single
# implementation; `verdict()` is the single place a level is chosen; `selftest()` is what vm-check.sh runs before it boots
# anything, so a future refactor that reorders these tests fails the gate instead of quietly turning skips into passes.
import re

# A marker token: the shell prints one bare word, so word-ish characters only, and `-skip-` must be an infix of a token
# (`console-heals-skip-no-tui`, `u7-ssh-skip-no-key`, ...). Anchored on token boundaries so prose like "skips the check"
# in an evidence line cannot trigger it.
_SKIP = re.compile(r"(?<![A-Za-z0-9_.:/-])[A-Za-z0-9_.:-]+-skip-[A-Za-z0-9_.:-]+")


def skipping(body):
    """The skip marker in `body`, or None. The whole token, so the reason survives into the table."""
    m = _SKIP.search(body or "")
    return m.group(0) if m else None


def verdict(name, want, body):
    """(level, line) for one row, the line already flattened for the table. Level is INFO, PASS, FAIL or SKIP.

    Order matters and is the point of this module: INFO (nothing asserted) -> SKIP (the row could not run) -> PASS/FAIL.
    A skip is neither a pass nor a failure: it is a row this medium could not answer, and the table says so with the
    marker as its reason.
    """
    want = (want or "").strip() or None
    flat = (body or "").replace("\n", " | ").strip()
    if want is None:
        return "INFO", flat
    mark = skipping(flat)
    if mark:
        # the marker is the whole body in the common case (the row printed one token and stopped): say it once
        if flat == mark:
            return "SKIP", mark
        return "SKIP", ("%s (%s)" % (flat, mark)) if flat else mark
    return ("PASS" if want in flat else "FAIL"), flat


def selftest():
    """Every arm that must hold, as (label, want, body, expected level). Returns a list of failures (empty = ok)."""
    arms = [
        # the defect #1111 found: the skip marker contains the wanted string
        ("skip-no-tui", "console-heals", "console-heals-skip-no-tui", "SKIP"),
        ("skip-no-tty1", "console-heals", "console-heals-skip-no-tty1", "SKIP"),
        # and a skip marker for a row whose want is something else entirely is still a skip
        ("skip-other-want", "tty1-menu", "u7-ssh-skip-no-key | some evidence", "SKIP"),
        # the real verdicts still work
        ("pass", "console-heals", "actions=9 stray=0 | console-heals", "PASS"),
        ("fail", "console-heals", "actions=9 stray=2 | console-garbled", "FAIL"),
        ("info", None, "anything at all", "INFO"),
        # a legitimate row whose evidence merely mentions skipping must NOT be skipped: the marker is a token, and
        # "skips the check" has no `-skip-` infix
        ("prose-not-a-marker", "row-ok", "row-ok | this skips the check", "PASS"),
    ]
    bad = []
    for label, want, body, expect in arms:
        got, line = verdict(label, want, body)
        if got != expect:
            bad.append("%s: want %s, got %s (%s)" % (label, expect, got, line))
    return bad


if __name__ == "__main__":
    fails = selftest()
    for f in fails:
        print("FAIL %s" % f)
    print("vm-check-verdict: %d failure(s)" % len(fails))
    raise SystemExit(1 if fails else 0)
