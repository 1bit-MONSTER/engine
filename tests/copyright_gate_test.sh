#!/usr/bin/env bash
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
# Which files `tools/copyright.py` looks at, pinned (engine#1564).
#
# The gate reads `git ls-files` plus the files git would add (`--others --exclude-standard`), so its verdict depends on
# three states a developer actually meets, and only two of them are about the repository:
#   - a staged file without a notice        -> RED (it is about to be committed),
#   - an untracked, non-ignored file        -> RED (this is the state a new file is in before `git add`; it is why the
#                                              check is useful in a working clone at all),
#   - an ignored file                       -> not seen at all,
# which is the case #1564 hit from the other direction: run logs under a directory nothing ignored were untracked and
# therefore RED in a developer clone, while a clean checkout of main was green. The fix was to ignore the output
# directory; this test keeps the three states apart so a future `--check` complaint can be attributed in one look.
#
# usage: tests/copyright_gate_test.sh   (no build, no device: a throwaway git repo in a temp dir)
set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; [ -n "${3:-}" ] && echo "     $3"; fail=1; fi; }

git -C "$tmp" init -q
git -C "$tmp" config user.email t@example.com
git -C "$tmp" config user.name t
cp "$root/tools/copyright.py" "$tmp/copyright.py"
# a committed file with the notice, and an ignore rule for the output directory
python3 - "$tmp" <<'PY'
import sys
d = sys.argv[1]
notice = open(d + "/copyright.py", encoding="utf-8").read().split('NOTICE = """')[1].split('""".split')[0]
open(d + "/noticed.py", "w", encoding="utf-8").write("# " + notice.replace("\n", "\n# ") + "\n")
# .gitignore is checked too (it is in LINE_STYLES), so the fixture's own ignore file carries the notice
open(d + "/.gitignore", "w", encoding="utf-8").write("# " + notice.replace("\n", "\n# ") + "\nrun-output/\n")
PY
git -C "$tmp" add noticed.py .gitignore copyright.py
git -C "$tmp" commit -qm base

# captured into a variable, never piped: `gate | grep -q` would make grep close the pipe early and, with pipefail,
# report the gate's SIGPIPE instead of its verdict
gate() { ( cd "$tmp" && python3 copyright.py --check 2>&1 ); }
gate_has() { case "$(gate)" in *"$1"*) return 0;; *) return 1;; esac; }

# 1. a committed tree with the notice is green
check "a tree whose files carry the notice passes" "[ $(gate >/dev/null; echo $?) -eq 0 ]" "$(gate)"

# 2. a staged file without the notice is RED -- the case that makes the gate worth running in a clone
printf 'x = 1\n' > "$tmp/staged.py"
git -C "$tmp" add staged.py
check "a STAGED file without the notice is reported" "gate_has 'missing copyright notice: staged.py'" "$(gate)"
git -C "$tmp" reset -q staged.py && rm -f "$tmp/staged.py"

# 3. an untracked, non-ignored file is RED too -- a new file before `git add`, which `git ls-files` alone cannot see
printf 'y = 2\n' > "$tmp/fresh.py"
check "an UNTRACKED, non-ignored file without the notice is reported" "gate_has 'missing copyright notice: fresh.py'" "$(gate)"
rm -f "$tmp/fresh.py"

# 4. an untracked file inside an ignored directory is NOT seen -- engine#1564: run output is not source
mkdir -p "$tmp/run-output"
printf '{"log": 1}\n' > "$tmp/run-output/probe.jsonl"
check "an ignored output directory is not reported" "! gate_has 'run-output'" "$(gate)"
check "and it does not make the gate red" "[ $(gate >/dev/null; echo $?) -eq 0 ]" "$(gate)"

# 5. once such an output file is TRACKED it is reported, whatever the ignore rule says -- and this is the arm that
#    survives #1580, which makes an *untracked* file of unknown comment style a skip (a file nobody has committed
#    should not fail the gate) while a tracked one stays an error. The untracked case is therefore deliberately NOT
#    asserted here: #1580 owns that verdict, and asserting it would make whichever PR lands second turn CI red.
rm -f "$tmp/.gitignore" && git -C "$tmp" rm -q --cached .gitignore 2>/dev/null; git -C "$tmp" commit -qm noignore --allow-empty
git -C "$tmp" add -f run-output/probe.jsonl
check "a TRACKED output file is reported even though nothing ignores it" "gate_has 'run-output/probe.jsonl'" "$(gate)"

echo "$([ $fail -eq 0 ] && echo PASS || echo FAIL)"
exit $((fail))
