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
# gpu_hold_escape_test.sh: a hold that ends must end work that LEFT the process tree (engine#2741).
#
# Measured 2026-11-16: a `1bit serve` took 44.4 GiB of GTT, its ticket ended, and the serve kept the memory for 64
# minutes while the box went to 351 MiB available with two OOM kills in the hour. `ps -o ppid=` said 1, and the serve's
# own environment still carried `GPU_HOLD_TICKET=... pid 3216578` -- a pid that no longer existed. engine#1621 fixed
# grandchildren and engine#1649 the process group; a `setsid` payload escapes both, because both are anchored on the
# tree. The environment is not: every process the payload started inherited the ticket line.
#
# Arms:
#   1. a payload that `setsid`s a sleeper and returns: after the hold, that sleeper is GONE;
#   2. it was really started (the pid file exists and the pid was alive while the hold ran), so arm 1 cannot pass
#      because the payload never got that far;
#   3. a process started OUTSIDE the ticket with the same command is untouched -- the detector matches this ticket's
#      own line, not the command. This arm is also the LIVENESS CONTROL: on a box that is OOM-killing small
#      processes (2026-11-16 18:4xZ, 351 MiB available, the kernel had already killed `sleep` and `bash`), a sleeper
#      can vanish for a reason that is not this ticket, which would make arm 1 pass for the wrong reason. If the
#      outside control died too, the run SKIPS (77) instead of claiming a pass.
#   4. the hold SAYS it did it: the engine#2741 warning names the escaped pid. This is the arm a script without the
#      detector fails, whatever the box does to sleepers.
#
# No device: every claim runs on a private queue and a sandboxed HOME.
set -u
HOLD=${1:?usage: gpu_hold_escape_test.sh <path-to-gpu-hold>}
[ -f "$HOLD" ] || { echo "  SKIP no gpu-hold at $HOLD"; exit 77; }
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ghescape.XXXXXX") || exit 77
trap 'rm -rf "$tmp"; kill -KILL ${outside:-} 2> /dev/null' EXIT
fails=0

# the take-time refusal reads the kernel log; a fixture keeps this test about the code (engine#2712's lesson)
printf 'kernel: gpu-hold test fixture: no fault lines in this log\n' > "$tmp/klog"
env_common=(HOME="$tmp" ONEBIT_HOLD_DIR="$tmp/hold" GPU_HOLD_KLOG_FILE="$tmp/klog"
            GPU_HOLD_COOLDOWN=0 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_THERMAL_FLAG="$tmp/none"
            GPU_HOLD_TCTL_START=999 GPU_HOLD_KILL_GRACE=2)

printf '#!/usr/bin/env bash\nsetsid bash -c %s &\nsleep 1\nexit 0\n' \
    "'echo \$\$ > \"$tmp/escaped.pid\"; exec sleep 300'" > "$tmp/payload.sh"

# 3 (control, set up first): an identical command started OUTSIDE any ticket must survive the hold
setsid bash -c "echo \$\$ > \"$tmp/outside.pid\"; exec sleep 300" &
for _ in 1 2 3 4 5 6; do [ -s "$tmp/outside.pid" ] && break; sleep 0.5; done
outside=$(cat "$tmp/outside.pid" 2>/dev/null || printf '')

out=$(env "${env_common[@]}" timeout 90 bash "$HOLD" -l escape-arm -t 0.5 -- bash "$tmp/payload.sh" 2>&1)
rc=$?
if [ "$rc" != 0 ]; then
    echo "  FAIL arm1/2: the hold itself failed (rc=$rc); gpu-hold said:"
    printf '%s\n' "$out" | tail -3 | sed 's/^/      /'
    fails=$((fails + 1))
fi

escaped=$(cat "$tmp/escaped.pid" 2>/dev/null || printf '')
printf '  arm2 escaped pid while the hold ran: %s\n' "${escaped:-<never written>}"
if [ -z "$escaped" ]; then
    echo "  FAIL arm2: the payload never started its setsid child, so arm1 would pass for the wrong reason"
    fails=$((fails + 1))
fi

# arm 4 is the DECISIVE arm and needs no long-lived process: a hold that ends an escaped process must SAY so.
# It is the arm a script without the detector fails, whatever the box does to sleepers.
if printf '%s\n' "$out" | grep -q "escaped the ticket's process tree"; then
    echo "  PASS arm4: the hold reports ending an escaped process ($(printf '%s\n' "$out" | grep -c "escaped the ticket") line(s))"
    said=1
else
    echo "  FAIL arm4: nothing in the hold's output says it ended an escaped process -- a hold that kills one silently is the state this test exists to prevent"
    said=0
    fails=$((fails + 1))
fi

# arm 3 is the LIVENESS CONTROL for arm 1. On a box that is OOM-killing small processes (measured 2026-11-16 18:4xZ:
# 351 MiB available, the kernel had already killed `sleep` and `bash`), a sleeper can vanish for a reason that is not
# this ticket, which would make arm 1 pass for the wrong reason. If the outside control died, arm 1 is INCONCLUSIVE
# rather than green -- arm 4 still decides the run.
if [ -n "${outside:-}" ] && kill -0 "$outside" 2> /dev/null; then
    echo "  PASS arm3: an identical command started outside the ticket was not touched"
    if [ -n "$escaped" ] && kill -0 "$escaped" 2> /dev/null; then
        echo "  FAIL arm1: pid $escaped escaped the process tree (setsid) and SURVIVED the hold -- $(ps -o args= -p "$escaped" 2>/dev/null | cut -c1-60)"
        kill -KILL "$escaped" 2> /dev/null
        fails=$((fails + 1))
    else
        echo "  PASS arm1: the setsid child that escaped the tree was ended with the hold"
    fi
else
    echo "  INCONCLUSIVE arm1/arm3: the outside control (pid ${outside:-?}) died too, so this box is ending sleepers on"
    echo "        its own (MemAvailable $(awk '/MemAvailable/{printf "%.0f MiB", $2/1024}' /proc/meminfo)); arm 4 above is"
    echo "        the decisive one. Re-run on a calm box for the outcome arm."
    kill -KILL ${escaped:-} 2> /dev/null
fi

if [ "$fails" = 0 ]; then echo "gpu_hold_escape: pass (arm4 decisive; see arm1/arm3 for the outcome)"; exit 0; fi
echo "gpu_hold_escape: $fails failure(s)"
exit 1
