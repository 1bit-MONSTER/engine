#!/bin/sh
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
# engine#3421: `1bit-logs thermal <dir>` writes a thermal line while ANY process holds the GPU or NPU open (/dev/kfd,
# /dev/accel/*, /dev/dri/renderD*), naming it in busy=, and once a minute (every 6th 10 s call) when none does. Runs on
# a fake /proc (ONEBIT_PROC) whose fd links name the device nodes, so it needs no root and no device.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
logs=$here/os/v3/puppy/embed/usr/local/bin/1bit-logs
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
proc() {   # proc <pid> <comm> <link target>...: a fake /proc/<pid> holding those fds
    mkdir -p "$t/proc/$1/fd"; printf '%s\n' "$2" > "$t/proc/$1/comm"; p=$1; shift 2; n=3
    for l in "$@"; do ln -s "$l" "$t/proc/$p/fd/$n"; n=$((n + 1)); done
}
mkdir -p "$t/boot" "$t/bin"
# a pgrep that never matches: the legacy "/opt/1bit-package/1bit/1bit is running" test must not decide anything here
printf '#!/bin/sh\nexit 1\n' > "$t/bin/pgrep"; chmod +x "$t/bin/pgrep"
# the functions without the command dispatch, so the cadence can be driven in one shell (it lives in the daemon process)
sed '/^case \${1:-} in/,$d' "$logs" > "$t/lib.sh"
run() { PATH="$t/bin:$PATH" ONEBIT_PROC=$t/proc sh "$logs" thermal "$t/boot"; }
calls() { PATH="$t/bin:$PATH" ONEBIT_PROC=$t/proc sh -c '. "$1"; i=0; th_idle=${3:-0}
          while [ $i -lt "$2" ]; do thermal_snap "$4"; i=$((i + 1)); done' _ "$t/lib.sh" "$1" "${2:-0}" "$t/boot"; }

# 1. a lane build on the GPU (not /opt/1bit-package): busy, named
proc 4100 gufo /dev/kfd /dev/null; proc 4200 bash /dev/pts/0
run; check $? "exit 0"
[ "$(wc -l < "$t/boot/thermal.log")" = 1 ] && grep -qE ' tctl_c=[^ ]+ fan_rpm=[^ ]+ pkg_w=[^ ]+ busy=gufo$' "$t/boot/thermal.log"
check $? "a process holding /dev/kfd (here a /data/dev build) gets a line, busy=gufo"

# 2. the NPU and a DRI render node count too; a name is listed once
proc 4300 loom-serve /dev/accel/accel0 /dev/kfd; proc 4301 loom-serve /dev/kfd; proc 4400 llama-server /dev/dri/renderD128
run; tail -1 "$t/boot/thermal.log" | sed 's/.*busy=//' > "$t/b"
ok=0; for c in gufo loom-serve llama-server; do tr ',' '\n' < "$t/b" | grep -qx "$c" || ok=1; done
[ $ok = 0 ] && [ "$(tr ',' '\n' < "$t/b" | grep -cx loom-serve)" = 1 ]
check $? "accel and renderD holders are busy too, each name listed once"

# 3. idle: a line on the first call, then only every 6th (60 s at the daemon's 10 s)
rm -rf "${t:?}/proc"; proc 4200 bash /dev/pts/0; rm -f "$t/boot/thermal.log"
calls 13
[ "$(wc -l < "$t/boot/thermal.log")" = 3 ] && [ "$(grep -c 'busy=-$' "$t/boot/thermal.log")" = 3 ]
check $? "with nothing on the devices, 13 calls give 3 lines (calls 1, 7, 13), each busy=-"

# 4. a device opening ends the idle wait: the very next call writes
proc 4500 gufo /dev/kfd
calls 1 3
[ "$(wc -l < "$t/boot/thermal.log")" = 4 ] && tail -1 "$t/boot/thermal.log" | grep -q 'busy=gufo$'
check $? "the first call after a device opens writes a line at once"

echo "thermal_log_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
