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
# gpu_hold_script_id_test.sh: the take line names the copy that ran it (engine#2494, guard-drift class).
#
# The guards in scripts/gpu-hold protect the box only once they are installed, and what runs is
# ~/.local/bin/gpu-hold, a copy that nothing re-installs on merge. That has been measured twice: after #2486 the
# installed copy was "identical to engine main" (11-15 05:52Z), and 16 h later it was the pre-#2538 copy with no
# take-time refusal at all (11-16 03:2xZ). Telling the two apart meant comparing sizes, mtimes and pattern counts
# by hand, and the question asked each time was the same: WHICH copy ran?
#
# Arms:
#   1. the take line carries `script <12 hex>`, and that value is the sha256 of the file that ran;
#   2. a copy with one line appended reports a DIFFERENT id, so a stale install cannot pass for the repo's copy
#      (this arm fails on the pre-change script, where the field does not exist at all);
#   3. the id is computed, not embedded: no 12-hex literal in the script equals either id.
#
# No device: each claim holds the token for a fraction of a second and runs `true` on a private queue.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
script="$here/../scripts/gpu-hold"
command -v sha256sum > /dev/null 2>&1 || { echo "skip: no sha256sum"; exit 77; }
[ -f "$script" ] || { echo "skip: no scripts/gpu-hold"; exit 77; }
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
fail=0

claim() {
    local s=$1 tag=$2
    mkdir -p "$work/$tag/queue"
    HOME=$work/$tag ONEBIT_HOLD_DIR=$work/$tag \
    GPU_HOLD_TCTL_START=999 GPU_HOLD_RELEASE_WAIT=0 \
        bash "$s" -l "script-id-$tag" -t 0.05 -w 10 true 2>&1
}

id_of() {
    sha256sum -- "$1" | cut -c1-12
}

# the id gpu-hold logged, or the empty string when the line has no such field
logged_id() {
    printf '%s\n' "$1" | sed -nE 's/.*; script ([0-9a-f]{12})\).*/\1/p' | head -n 1
}

out=$(claim "$script" repo)
want=$(id_of "$script")
got=$(logged_id "$out")
if [ -n "$got" ] && [ "$got" = "$want" ]; then
    echo "PASS  the take line carries script '$got', the sha256 of the file that ran"
else
    echo "FAIL  want the take line to carry script '$want'; it carried '${got}'. gpu-hold said:"
    printf '%s\n' "$out" | tail -3 | sed 's/^/      /'
    fail=$((fail + 1))
fi

mkdir -p "$work/stale"
cp "$script" "$work/stale/gpu-hold"
printf '# one line a stale install has and a merged repo copy does not\n' >> "$work/stale/gpu-hold"
chmod +x "$work/stale/gpu-hold"
out2=$(claim "$work/stale/gpu-hold" stale)
want2=$(id_of "$work/stale/gpu-hold")
got2=$(logged_id "$out2")
if [ -n "$got2" ] && [ "$got2" = "$want2" ] && [ "$got2" != "$want" ]; then
    echo "PASS  a modified copy logs its own id '$got2', so a stale install cannot pass for the repo copy '$want'"
else
    echo "FAIL  the modified copy logged '${got2}'; want its own '$want2' and not the repo's '$want'. gpu-hold said:"
    printf '%s\n' "$out2" | tail -3 | sed 's/^/      /'
    fail=$((fail + 1))
fi

text=$(cat "$script")
if [ -z "$got" ] || [ -z "$got2" ]; then
    echo "FAIL  no id to look for: the field is missing, so this arm cannot pass"
    fail=$((fail + 1))
else
    case "$text" in
        *"$got"* | *"$got2"*) echo "FAIL  the script embeds one of the ids as a literal, so the field cannot tell copies apart"; fail=$((fail + 1)) ;;
        *) echo "PASS  the id is computed from the file, not embedded" ;;
    esac
fi

exit "$fail"
