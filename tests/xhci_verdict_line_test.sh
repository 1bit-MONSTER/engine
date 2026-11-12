#!/bin/bash
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
#
#
#
# The verdict line of tools/xhci_boot_capture.sh, driven on a fixture.
#
# That function says of itself: "A function, not an inline block, so the three classifications can be driven on a
# fixture without a device (the same reason every other rule in this tree is header-only/function-shaped: the decision
# is what can be wrong)." It was written to be testable and then was not tested, which is why the four verdicts below
# only entered the record through a manual drive during engine#823's review. This is that drive, as arms.
#
# The capture script runs its own capture at the bottom, so the function is extracted rather than sourced: the arms
# assert the DECISION, which is what the comment says can be wrong, and never the device.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
src="$here/tools/xhci_boot_capture.sh"
fn=$(mktemp)
trap 'rm -f "$fn"' EXIT
sed -n '/^xhci_verdict_line()/,/^}/p' "$src" > "$fn"
if [ ! -s "$fn" ]; then
    echo "FAIL the verdict function is not extractable from $src"
    exit 1
fi
fails=0
arm() {  # <bdf> <control> <power_state> <runtime_status> <want-rc> <want-text> <what>
    local out rc
    out=$(bash -c ". '$fn'; xhci_verdict_line '$1' '$2' '$3' '$4'" 2>&1); rc=$?
    if [ "$rc" != "$5" ]; then
        echo "FAIL $7: want rc $5, got $rc"
        fails=$((fails + 1))
    elif [ -z "$6" ]; then
        if [ -n "$out" ]; then
            echo "FAIL $7: want no line, got: $(printf '%s' "$out" | head -1)"
            fails=$((fails + 1))
        else
            echo "ok   $7"
        fi
    elif ! printf '%s' "$out" | grep -qF -- "$6"; then
        echo "FAIL $7: the line does not say '$6' (got: $(printf '%s' "$out" | head -1))"
        fails=$((fails + 1))
    else
        echo "ok   $7"
    fi
}

# The fault my #823 review named: suspended under a control of `auto` means the keep-awake rule did not hold the
# controller, and that is the one verdict a boot must not report as healthy.
arm 0000:0a:00.0 auto D3hot suspended 1 "the keep-awake rule did NOT hold it" \
    "control=auto + suspended is the fault (engine#823)"
arm 0000:0a:00.0 0    D3hot suspended 0 "held out of D0 deliberately" \
    "control=0 + suspended is deliberate, not a fault"
# An awake controller prints NOTHING and returns 0: the line exists to explain a state, and "healthy" is not one of
# the things it explains. Asserting the silence matters as much as the text of the other four -- a verdict file that
# says nothing about a fault is the failure this whole capture exists to avoid.
arm 0000:0a:00.0 0    D0    active    0 "" \
    "an awake controller is silent (rc 0, no line)"
arm 0000:0a:00.0 0    D0    resuming  2 "a transition that did not settle" \
    "a transition is neither awake nor asleep"
arm 0000:0a:00.0 0    D0    current   3 "is not one of" \
    "an unnameable state is its own verdict, not a pass"

if [ "$fails" -ne 0 ]; then
    echo "xhci_verdict_line: $fails arm(s) failed"
    exit 1
fi
echo "xhci_verdict_line: 5 arms pass"
