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
"""The census issue excerpt keeps the lines that say something (engine#2677, engine#2696 review).

`.github/workflows/census-watch.yml` builds the issue body from the watcher's log with a `grep | head -n 40`: the
excerpt is capped, so WHICH lines fill it is the whole question. Two failures are measured:

  * `#2677` — 21 `? UNVERIFIABLE` and 19 answered `. NPU FLM` rows filled all 40 lines, so the run's only two
    alerts (`!! SUPPORT MOVED`) never reached the issue, which was then filed looking empty. Fixed by putting the
    `!!` rows first.

  * the review of `#2696` — with two tiers only, the watcher's `-> …` conclusions still landed in the last bucket.
    On the `#2677` log shape the `-> N Q4NX (NPU) pack(s) …` line sits at position 41 and is dropped, and that
    line is the ONLY output a Q4NX gap produces (`census_watch.py:889` returns 1 for it), so the issue for that
    one condition is still empty. Fixed by a third tier: `!!`, then `  -> `, then the per-row stream.

This runs the workflow's own pipeline rather than a copy of it: the block is extracted from the YAML, the log path
is pointed at a fixture, and the base64 the workflow would put in `lines=` is decoded and read. If the block moves
or changes shape the test fails loudly instead of passing on a stale copy.

Exit: 0 the excerpt holds every alert and conclusion line, 1 a line was dropped, 77 the block could not be found.
"""
import base64
import pathlib
import re
import subprocess
import sys
import tempfile

WORKFLOW = pathlib.Path(".github/workflows/census-watch.yml")
LOG = "/tmp/census-watch.log"


def excerpt_block(workflow_text):
    """The `pat=… lines=$(…)` shell the workflow runs, with the log path left as a placeholder."""
    lines = workflow_text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip().startswith("pat=")), None)
    if start is None:
        return None
    end = next((i for i in range(start, len(lines)) if "base64 -w0)" in lines[i]), None)
    if end is None:
        return None
    return "\n".join(ln.strip() for ln in lines[start:end + 1])


def excerpt(block, log_text):
    """Run the extracted block against `log_text`; the returned bytes are what the issue would show."""
    with tempfile.TemporaryDirectory() as d:
        log = pathlib.Path(d) / "watch.log"
        log.write_text(log_text)
        script = block.replace(LOG, str(log)).replace("lines=$((", "lines=$((\n") + '\nprintf %s "$lines"\n'
        p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit(f"the excerpt pipeline failed (rc {p.returncode}): {p.stderr.strip()}")
        raw = p.stdout.strip()
    return base64.b64decode(raw).decode() if raw else ""


def shape_1116(alerts=True, conclusion=True, n_unverifiable=21, n_answered=19, failures=0, glued=False):
    """The 11-16 13:00Z log: the filler first, the alerts and the summary after it, as the watcher prints them."""
    out = [f"  ? UNVERIFIABLE org/model{i:02d} (no config / no architecture)" for i in range(n_unverifiable)]
    out += [f"  . NPU FLM FastFlowLM/Family{i}-NPU2  … answered: engine#1786" for i in range(n_answered)]
    if failures:
        # engine#2812: the tool writes its own failures to stderr as `census-watch: !! …`, which matched none of
        # the row words and so never reached the excerpt except by accident (engine#2780's glue).
        out += ["census-watch: !! SUPPORT WATCH FAILED: https://api.github.com/repos/a/b/commits?per_page=1: "
                "HTTP Error 403: rate limit exceeded"] * failures
    if glued:
        # The shape that started this: a stderr line spliced onto the end of the last stdout row, before the
        # watcher was run unbuffered. It CONTAINS `!!` and must still not be treated as an alert row.
        out.append("  . NPU FLM FastFlowLM/Translategemma-4B-Instruct-NPU2  … answered: engine#1786"
                   "census-watch: !! SUPPORT WATCH FAILED: HTTP Error 403: rate limit exceeded")
    if alerts:
        out += ["  !! SUPPORT MOVED ggml-org/llama.cpp  head abc1234  reviewed 2026-11-01",
                "  !! SUPPORT MOVED lemonade-sdk/lemonade  head def5678  reviewed 2026-11-02"]
    if conclusion:
        out.append("  -> 3 Q4NX (NPU) pack(s) for a family the NPU route does not serve: gemma4 (3 pack(s))")
    return "\n".join(out) + "\n"


def main():
    if not WORKFLOW.is_file():
        print(f"census_watch_excerpt: skip: no {WORKFLOW}", file=sys.stderr)
        return 77
    block = excerpt_block(WORKFLOW.read_text())
    if block is None:
        print("census_watch_excerpt: cannot find the `pat=… lines=$(…)` block in the workflow: the excerpt moved, "
              "so this arm is looking at nothing", file=sys.stderr)
        return 77
    fails = 0

    def check(ok, what, detail=""):
        nonlocal fails
        print(f"{'PASS' if ok else 'FAIL'}  {what}" + (f"  [{detail}]" if detail and not ok else ""))
        if not ok:
            fails += 1

    got = excerpt(block, shape_1116())
    rows = got.splitlines()
    check(len(rows) <= 40, "the excerpt is still capped at 40 lines", f"{len(rows)} rows")
    check(sum("!!" in r for r in rows) == 2, "both `!!` alert rows survive the cap", got[:200])
    check(any("Q4NX (NPU)" in r for r in rows),
          "the `-> … Q4NX (NPU)` conclusion survives the cap (the line a Q4NX-only gap prints)",
          "; ".join(rows[:3]))
    check(rows and "!!" in rows[0], "alerts come first", rows[0] if rows else "")

    # The same 40 filler rows with NO alert: the conclusion alone is the whole finding.
    got2 = excerpt(block, shape_1116(alerts=False))
    check("Q4NX (NPU)" in got2, "a conclusion survives when it is the only finding", got2[:200])

    # Filler big enough to overrun the cap on its own: the per-row stream is what gets truncated, not the conclusions.
    got3 = excerpt(block, shape_1116(n_unverifiable=60))
    check("Q4NX (NPU)" in got3 and got3.count("!!") >= 2,
          "the per-row stream is truncated before the alerts or the conclusion", got3[:200])

    # engine#2812: a watch that could not read its own sources says so, and says it FIRST.
    got4 = excerpt(block, shape_1116(alerts=False, conclusion=False, failures=2))
    rows4 = got4.splitlines()
    check(sum("WATCH FAILED" in r for r in rows4) == 2,
          "both `census-watch: !! … FAILED` lines reach the excerpt", got4[:240])
    check(rows4 and rows4[0].startswith("census-watch: !!"),
          "a failure line leads the excerpt, ahead of the per-row stream", rows4[0] if rows4 else "")
    check(not any(r.startswith("  ? UNVERIFIABLE") for r in rows4[:2]),
          "…and the informational rows are still behind it", "; ".join(rows4[:3]))

    # engine#2780: the glued row starts with `  . NPU FLM`, so it is NOT an alert however many `!!` it carries.
    # A small log, because 40 filler rows would push the glued row past the cap and the arm would then be
    # reading the cap rather than the tiering.
    got5 = excerpt(block, shape_1116(alerts=False, conclusion=False, n_unverifiable=5, n_answered=5, glued=True))
    rows5 = got5.splitlines()
    check(any("FAILED" in r for r in rows5), "the glued row is still shown (the text is not lost)", got5[:200])
    check(rows5 and not rows5[0].strip().startswith(". NPU FLM"),
          "a glued `. NPU FLM … !!` row does not lead the excerpt as an alert", rows5[0] if rows5 else "")

    print(f"census_watch_excerpt: {'ok' if not fails else f'{fails} failure(s)'}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
