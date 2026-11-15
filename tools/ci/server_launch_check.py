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
"""server_launch_check.py: a server launched under `setsid` loses its kill switch.

`setsid CMD &` puts CMD in a **new session**, so the shell's `$!` is the `setsid` pid, which exits
immediately. Every `kill "$pid"` afterwards targets a dead process, and the server survives in that
session with PPID 1 — no ticket, no cleanup path, still holding whatever device it claimed.

That is not hypothetical. On 2026-11-15 an `npu-model-chart.sh` run left `flm-real serve lfm3:1.7b`
alive for **22 h 44 min**, holding every NPU hardware context; two lanes got -19 on every hwctx create
for most of a day. The chart had seven `kill "$pid"` sites, all of them correct-looking and all of them
pointing at `setsid`.

    server_launch_check.py                 # check the repo's scripts/ and tools/
    server_launch_check.py --include-bin    # also $HOME/bin, where the orphan lived (warn only)
    server_launch_check.py --self-test

Exit: 0 clean, 1 a repo script launches a server under setsid (or kills `$!` after doing so), 77 no
scripts matched (a skip, not a pass).
"""
import argparse
import os
import re
import sys

SKIP = 77

# A launch line starts a long-lived server; `setsid` is only a defect in that combination.
_LAUNCH = re.compile(r"\b(serve|forward-serve|llama-server|llama-cli)\b")
_SETSID = re.compile(r"\bsetsid\b")
_AMP = re.compile(r"&\s*$")
# A kill that targets a whole PROCESS GROUP is the correct remedy, not the defect: `setsid cmd &` makes the
# child a session leader, so its pgid IS the leader's pid -- which is what `$!` holds -- and a group kill
# reaches the server even though the leader has exited. Without this the checker reds the fix (2026-11-15).
_GROUP_KILL = re.compile(r"\bkill\s+(-[A-Za-z]+\s+)*--?\s*-")
# An explicit waiver, for the residue a text scan cannot decide: `# server-launch-check: ok <reason>` on or
# just above the kill line.
_SUPPRESS = re.compile(r"server-launch-check:\s*ok\b")


def check_text(text, path="<text>"):
    """-> list of (lineno, line, kind) findings. kind is 'setsid-launch' or 'kill-after-setsid'."""
    lines = text.splitlines()
    launches = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue                                  # a comment naming the pattern is not the pattern
        if _SETSID.search(stripped) and _LAUNCH.search(stripped) and _AMP.search(stripped):
            launches.append((i, stripped))
    if not launches:
        return []
    # A group kill anywhere in the file is the remedy, and it decides the whole file: `kill -- -$PID` signals
    # the session the server lives in, so the cleanup cannot miss it even though the leader has exited.
    body = [(i, l.strip()) for i, l in enumerate(lines, 1)
            if not l.strip().startswith("#") and re.match(r"kill\b", l.strip())]
    if any(_GROUP_KILL.search(t) for _, t in body):
        return []
    # The second half of the defect: `$!` captured from a setsid launch is the wrong process. Report it
    # once, because that is what turns "setsid is unusual" into "the cleanup provably misses the server".
    targets = [(i, t) for i, t in body if re.search(r"\bkill\b.*\$(PID|\!|pid)\b", t)]
    if not targets:                                   # no kill: fall back to the capture line itself
        targets = [(i, l.strip()) for i, l in enumerate(lines, 1)
                   if not l.strip().startswith("#") and l.strip().endswith("$!")]
    for i, stripped in targets:
        if any(_SUPPRESS.search(c) for c in lines[max(0, i - 4):i - 1]):
            return []                             # explicitly waived, with a stated reason
        return [(n, t, "setsid-launch") for n, t in launches] + [(i, stripped, "kill-after-setsid")]
    return [(n, t, "setsid-launch") for n, t in launches]


def self_test():
    """Three arms: the defect, the fix, and the comment that must not count as the defect."""
    arms = [
        ("a repo script launching serve under setsid is reported",
         'setsid "$BIN" forward-serve -m "$MODEL" -p "$PORT" &\nPID=$!\nkill $PID\n',
         ["setsid-launch", "kill-after-setsid"]),
        ("the same script with setsid dropped is clean",
         '"$BIN" forward-serve -m "$MODEL" -p "$PORT" &\nPID=$!\nkill $PID\n', []),
        ("a COMMENT naming setsid is not the defect",
         '# do not use setsid here: it makes $! the wrong pid\nnohup "$BIN" serve -p $PORT &\n', []),
            # 2026-11-15: the correct remedy must be GREEN, or the only way to satisfy the gate is to avoid
            # setsid rather than to fix the cleanup.
            ("kill -- -$PID (group kill by the captured setsid pid) is the fix, not the defect",
             'setsid "$BIN" forward-serve -p "$PORT" &\nPID=$!\nsleep 3\nkill -- -"$PID"\n', []),
            ("kill -TERM -- -$PGID is the same fix, named group",
             'setsid "$BIN" serve -p $PORT &\nsleep 3\nkill -TERM -- -$PGID\n', []),
            ("an explicit waiver just above the kill line is honoured",
             'setsid "$BIN" serve -p $PORT &\nPID=$!\n# server-launch-check: ok pid re-resolved below\nkill $PID\n',
             []),
            ("the un-waived defect itself is still the defect",
             'setsid "$BIN" serve -p $PORT &\nPID=$!\nkill $PID\n',
             ["setsid-launch", "kill-after-setsid"]),
    ]
    bad = 0
    for name, text, want in arms:
        got = [kind for _, _, kind in check_text(text)]
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'ok ' if ok else 'FAIL'} {name}" + ("" if ok else f"  (got {got}, want {want})"))
    print(f"{len(arms)} arms, {bad} failed")
    return 1 if bad else 0


def scan(root, include_bin):
    """Repo paths fail the build; $HOME/bin warns, because those scripts are not this repo's to own."""
    targets = []
    for sub in ("scripts", "tools"):
        base = os.path.join(root, sub)
        for dirpath, _, names in os.walk(base):
            for n in names:
                if n.endswith(".sh"):
                    targets.append((os.path.join(dirpath, n), "repo"))
    if include_bin:
        bindir = os.path.join(os.path.expanduser("~"), "bin")
        if os.path.isdir(bindir):
            for n in sorted(os.listdir(bindir)):
                p = os.path.join(bindir, n)
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    targets.append((p, "bin"))
    total = 0
    for path, kind in targets:
        try:
            with open(path, "r", errors="replace") as fh:
                findings = check_text(fh.read(), path)
        except OSError:
            continue
        for lineno, line, why in findings:
            rel = os.path.relpath(path, root)
            tag = "FAIL" if kind == "repo" else "WARN"
            print(f"  {tag} {rel}:{lineno} [{why}] {line[:96]}")
            if kind == "repo":
                total += 1
    if not targets:
        print("SKIPPED: server_launch_check did NOT run -- no .sh scripts under scripts/ or tools/", file=sys.stderr)
        return SKIP
    return 1 if total else 0


def main():
    ap = argparse.ArgumentParser(description="Reject a server launched under setsid: its kill switch cannot reach it.")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--include-bin", action="store_true", help="also scan ~/bin (warn only; not repo-owned)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    return scan(args.root, args.include_bin)


if __name__ == "__main__":
    sys.exit(main())
