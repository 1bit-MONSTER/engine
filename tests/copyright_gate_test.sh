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

# 6. A NEAR-MISS of this repo's holder is replaced by --fix, not stacked beside it (engine#1606). The build step
#    "Copyright notice on every file" failed there because a hand-written `Copyright 2026 bong-water-bong` sat
#    beside the canonical block: --fix inserted the canonical notice and left the typo in place, and --check was
#    green because it only looked for the canonical pair. Both halves are asserted, and so is the shape that must
#    NOT red: dual attribution with a different holder (`The 1bit-MONSTER Authors`) is legitimate in this tree.
near=$tmp/near.py
printf '%s\n' '# Copyright 2026 bong-water-bong' '# SPDX-License-Identifier: Apache-2.0' 'z = 3' > "$near"
git -C "$tmp" add near.py
fix() { ( cd "$tmp" && python3 copyright.py --fix >/dev/null 2>&1 ); }
fix
check "--fix replaces a near-miss holder instead of stacking a second notice" \
      "[ $(grep -c 'bong-water-water-bong' "$near") -eq 1 ]" ok
check "and the earlier near-miss line is gone" "[ $(grep -c 'bong-water-bong$' "$near") -eq 0 ]" ok
check "--check is then clean for that file" "! gate_has 'near.py'" "$(gate)"
rm -f "$near"; git -C "$tmp" update-index --force-remove near.py 2>/dev/null

# The shape the tree really carries (tests/installed_rpath_test.sh, tools/check-installed-rpath.py): the repo's
# own full notice, then a second attribution naming a different holder. Copy the notice out of this file so the
# fixture cannot drift from what the gate accepts.
dual=$tmp/dual.py
{ sed -n '2,16p' "$root/tests/copyright_gate_test.sh" | sed 's/^# //; s/^#$//; s/^/# /' ; \
  echo '# Copyright 2026 The 1bit-MONSTER Authors.'; echo 'z = 4'; } > "$dual"
git -C "$tmp" add dual.py
#    Asserted on THIS file, not on the gate's exit code: earlier cases in this fixture deliberately leave a
#    tracked file with no notice (case 5), so the whole gate is red by design at this point.
check "dual attribution with a DIFFERENT holder is not a duplicate" "! gate_has 'dual.py'" "$(gate)"
rm -f "$dual"; git -C "$tmp" reset -q dual.py 2>/dev/null

# 7. a path that was staged and then deleted must not crash the gate (found by case 6 above): `git ls-files`
#    still lists it, and opening it took the whole check down with a FileNotFoundError traceback.
gone=$tmp/gone.py
printf '%s\n' 'z = 5' > "$gone"; git -C "$tmp" add gone.py; rm -f "$gone"
#    Asserted as "no traceback", not as an exit code: `-le 1` would pass on a crash too, which is how the first
#    version of this very assertion stayed green while the tool raised NameError.
check "a staged-then-deleted path does not crash the gate" "! gate 2>&1 | grep -q Traceback" "$(gate 2>&1 | tail -2)"
git -C "$tmp" update-index --force-remove gone.py 2>/dev/null

# 8. The state that actually shipped on #1606, one step past case 6: a near-miss holder AND a legitimate second
#    attribution in the same file. `--fix` used to insert the canonical block and remove neither, so `main` turned
#    this into THREE notices with a green `--check` (measured: main 3, this tree 2). The right answer is two: the
#    canonical notice plus the attribution that is not a misspelling of it.
both=$tmp/both.py
{ sed -n '2,16p' "$root/tests/copyright_gate_test.sh"
  echo '# Copyright 2026 The 1bit-MONSTER Authors.'
  echo '# Copyright 2026 bong-water-bong'
  echo 'z = 6'; } > "$both"
git -C "$tmp" add both.py
fix
check "near-miss beside an existing notice leaves exactly two copyright lines" \
      "[ $(grep -c '^# Copyright' "$both") -eq 2 ]" "$(grep -c '^# Copyright' "$both") line(s)"
check "and the near-miss spelling is the one that went" "[ $(grep -c 'bong-water-bong$' "$both") -eq 0 ]" "$(grep 'Copyright' "$both")"
rm -f "$both"; git -C "$tmp" update-index --force-remove both.py 2>/dev/null

# 2. the two-liner without the licence block: --fix must leave ONE notice, not a copy of the pair beside the block
# (engine#2129 -- --fix used to prepend the block, keep the two-liner, and then fail its own --check with "duplicated
# inside the header window", so the file it had just written was invalid and neither mode said so).
printf '# Copyright 2026 bong-water-bong\n# SPDX-License-Identifier: Apache-2.0\n\nx = 1\n' > "$tmp/twoliner.py"
git -C "$tmp" add twoliner.py
( cd "$tmp" && python3 copyright.py --fix >/dev/null 2>&1 )
check "--fix on a two-liner leaves a file that its own --check accepts" \
      "! ( cd \"$tmp\" && python3 copyright.py --check 2>&1 | grep -q duplicated )" \
      "$( cd "$tmp" && python3 copyright.py --check 2>&1 )"
check "...and exactly one SPDX line is left in the window" \
      "[ \$(head -20 \"$tmp/twoliner.py\" | grep -c 'SPDX-License-Identifier') -eq 1 ]" \
      "window: $(head -20 "$tmp/twoliner.py" | grep -c 'SPDX-License-Identifier') SPDX line(s)"

echo "$([ $fail -eq 0 ] && echo PASS || echo FAIL)"
exit $((fail))