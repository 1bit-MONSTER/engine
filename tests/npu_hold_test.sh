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
# tests/npu_hold_test.sh -- scripts/npu-hold puts its queue/token/log where ONEBIT_HOLD_DIR says, and hands the
# queue settings to gpu-hold only (engine#1701).
#
# Hermetic: a stub gpu-hold records the environment it was given and runs the payload, so nothing here touches a
# real queue, a token or a device.
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
fail=0
ck() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2', wanted '$3'"; fail=1; fi; }

t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
# a stub that proves what npu-hold handed it, then runs the payload the same way the real one does
cat > "$t/gpu-hold" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$GPU_QUEUE_DIR|$GPU_TOKEN_FILE|$GPU_HOLD_LOG|$GPU_HOLD_MINUTES|$GPU_HOLD_TCTL_START" > "$STUB_RECORD"
while [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done
[ "${1:-}" = -- ] && shift
exec "$@"
STUB
chmod +x "$t/gpu-hold"

# 1. ONEBIT_HOLD_DIR set and existing: the fleet layout wins
mkdir -p "$t/fleet"
env ONEBIT_HOLD_DIR="$t/fleet" ONEBIT_GPU_HOLD="$t/gpu-hold" STUB_RECORD="$t/r1" HOME="$t/home" \
    "$root/scripts/npu-hold" -t 1 -l probe -- sh -c 'echo payload-ran' > "$t/out1" 2>&1
ck "the payload ran" "$(cat "$t/out1")" "payload-ran"
IFS='|' read -r q tk lg mins start < "$t/r1"
ck "queue follows ONEBIT_HOLD_DIR" "$q" "$t/fleet/npu-queue"
ck "token follows ONEBIT_HOLD_DIR" "$tk" "$t/fleet/npu-token"
ck "log follows ONEBIT_HOLD_DIR" "$lg" "$t/fleet/npu-queue/.log"
ck "the NPU hold still relaxes the thermal gate" "$start" "90"

# 2. no ONEBIT_HOLD_DIR: unchanged, $HOME/NPU-* as before
mkdir -p "$t/home"
env -u ONEBIT_HOLD_DIR ONEBIT_GPU_HOLD="$t/gpu-hold" STUB_RECORD="$t/r2" HOME="$t/home" \
    "$root/scripts/npu-hold" -- sh -c ':' > /dev/null 2>&1
IFS='|' read -r q tk lg _ _ < "$t/r2"
ck "without ONEBIT_HOLD_DIR the queue is \$HOME/NPU-QUEUE" "$q" "$t/home/NPU-QUEUE"
ck "without ONEBIT_HOLD_DIR the token is \$HOME/NPU-TOKEN" "$tk" "$t/home/NPU-TOKEN"

# 3. the queue settings must NOT reach the payload: a gpu-hold inside it has to queue on the GPU
cat > "$t/payload" <<'P'
#!/usr/bin/env bash
echo "${GPU_QUEUE_DIR:-unset}|${GPU_TOKEN_FILE:-unset}|${GPU_HOLD_LOG:-unset}" > "$PAYLOAD_RECORD"
P
chmod +x "$t/payload"
env ONEBIT_HOLD_DIR="$t/fleet" ONEBIT_GPU_HOLD="$t/gpu-hold" STUB_RECORD="$t/r3" PAYLOAD_RECORD="$t/p3" \
    "$root/scripts/npu-hold" -- "$t/payload" > /dev/null 2>&1
ck "the payload sees no NPU queue settings" "$(cat "$t/p3")" "unset|unset|unset"

# 4. integration: the tree's own gpu-hold takes the NPU queue when it is the only one present, and releases it.
#    No device is touched - gpu-hold only creates a ticket/token and runs the payload - so this is hermetic too.
mkdir -p "$t/fleet2"
env ONEBIT_HOLD_DIR="$t/fleet2" ONEBIT_GPU_HOLD="$root/scripts/gpu-hold" HOME="$t/home" \
    "$root/scripts/npu-hold" -t 1 -l itest -- sh -c 'echo integrated' > "$t/out4" 2>&1
ck "the tree gpu-hold ran the payload" "$(grep -c integrated "$t/out4")" "1"
ck "and released the NPU token" "$([ -f "$t/fleet2/npu-token" ] && echo present || echo gone)" "gone"
ck "with no ticket left behind" "$(ls -A "$t/fleet2/npu-queue" 2>/dev/null | grep -v '^\.' | wc -l)" "0"

# 5. a missing command is a usage error, not a silent success
env ONEBIT_GPU_HOLD="$t/gpu-hold" "$root/scripts/npu-hold" -- > /dev/null 2>&1
ck "no command is a usage error" "$?" "2"

[ "$fail" = 0 ] && echo "npu_hold: ok" || echo "npu_hold: FAILED"
exit "$fail"
