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
# tests/hw_gate_mem_test.sh — the hardware gate's build does not start into a box that is short of memory, and
# a memory-guard stop really ends the job (engine#1504).
#
# What it pins, each one a thing that happened on 2026-11-08 when both self-hosted runners were OOM-killed:
#   1. preflight: below the build floor the compile is never started; the gate exits 2 ("the box could not run
#      the gate") and its result row carries the MemAvailable reading;
#   2. guard: when memory falls during the compile the gate stops it once, does NOT resume it the way it resumes a
#      thermal stop, exits 2 and names the guard;
#   3. the stop is real: the compile process is gone afterwards. mem-guard's default SIGKILL ends thermal-run but
#      not what thermal-run started (a SIGKILL cannot run its trap), so the job kept running and kept its memory;
#   4. the same for a GPU step, whose guard is in hw-gate.sh's gpu(): it must be --term.
#
# Usage: hw_gate_mem_test.sh [repo root]
#
# No GPU, no compiler: the gate runs in a fake root with a stub cmake and a stub thermal-run that has the real
# one's property (the job is in its own session; TERM/INT/HUP kill it, SIGKILL orphans it). The meminfo is fake
# (MEM_GUARD_MEMINFO): "plenty, then starved" cannot be produced on demand on a shared box.
set -u
src=$(cd "${1:-$(dirname "$0")/..}" && pwd)
scratch=$(mktemp -d)
trap 'for f in "$scratch"/*.pid; do [ -f "$f" ] && kill -9 "$(cat "$f")" 2> /dev/null; done; rm -rf "$scratch"' EXIT
fails=0
fail() { echo "FAIL $*"; fails=$((fails + 1)); }

root=$scratch/root
mkdir -p "$root/scripts" "$root/third_party/llama.cpp" "$root/third_party/hrx-system" "$scratch/bin"
cp "$src"/scripts/hw-gate.sh "$src"/scripts/mem-guard.sh "$src"/scripts/mem-wait.sh "$src"/scripts/hrx-env.sh "$root/scripts/"
touch "$root/third_party/llama.cpp/CMakeLists.txt" "$root/third_party/hrx-system/CMakeLists.txt"

sed 's/^MemAvailable:.*/MemAvailable:   900000000 kB/' /proc/meminfo > "$scratch/mem-high"
sed 's/^MemAvailable:.*/MemAvailable:     1048576 kB/' /proc/meminfo > "$scratch/mem-low"

# stub thermal-run: the job in its own session, killed on TERM/INT/HUP only (as ~/lb/thermal-run.sh)
cat > "$scratch/thermal.sh" <<'STUB'
#!/usr/bin/env bash
setsid "$@" &
p=$!
trap 'kill -KILL -- -$p 2> /dev/null; exit 143' TERM INT HUP
wait $p
STUB
# stub cmake: configure writes the cache; --build counts itself, then finishes (FAKE_BUILD=ok) or hangs
cat > "$scratch/bin/cmake" <<'STUB'
#!/usr/bin/env bash
if [ "$1" = -S ]; then mkdir -p "$4" && touch "$4/CMakeCache.txt"; exit 0; fi
echo x >> "$FAKE_STATE/builds"
if [ "$FAKE_BUILD" = ok ]; then
  mkdir -p "$2/bin"
  for b in llama-server test-backend-ops llama-perplexity; do printf '#!/bin/sh\n' > "$2/bin/$b"; chmod +x "$2/bin/$b"; done
  exit 0
fi
echo $$ > "$FAKE_STATE/build.pid"
exec sleep 600
STUB
chmod +x "$scratch/thermal.sh" "$scratch/bin/cmake"

gate() {  # gate <state dir> <FAKE_BUILD> [VAR=value...]: run the build phase; rc in $rc, seconds in $secs
  local state=$1 mode=$2; shift 2
  mkdir -p "$state"
  local t0; t0=$(date +%s)
  env PATH="$scratch/bin:$PATH" FAKE_STATE="$state" FAKE_BUILD="$mode" \
      MEM_GUARD_MEMINFO="$scratch/meminfo" MEM_WAIT_POLL=1 \
      HW_GATE_GPU_HOLD=0 HW_GATE_TOKEN=none HW_GATE_PRIVATE=0 HW_GATE_COOL_C=1000 \
      HW_GATE_STORE="$state/store" HW_GATE_OUT="$state/out" HW_GATE_THERMAL="$scratch/thermal.sh" \
      ONEBIT_HRX_TOOLCHAIN="$state/no-toolchain" HRX_SVM=1 GITHUB_STEP_SUMMARY= "$@" \
      timeout 120 "$root/scripts/hw-gate.sh" --phases build > "$state/stdout" 2>&1
  rc=$?; secs=$(( $(date +%s) - t0 ))
}
builds() { if [ -f "$1/builds" ]; then wc -l < "$1/builds"; else echo 0; fi; }

# 0. the harness: with memory to spare the stub build passes, so a refusal below is the memory check and not the fake root
cp "$scratch/mem-high" "$scratch/meminfo"
gate "$scratch/s0" ok
if [ "$rc" = 0 ] && [ "$(builds "$scratch/s0")" = 1 ]; then echo "ok   with memory to spare the build runs and passes"
else fail "the harness: build phase rc=$rc, $(builds "$scratch/s0") compile(s), with memory to spare"; tail -n 5 "$scratch/s0/stdout"; fi

# 1. preflight
cp "$scratch/mem-low" "$scratch/meminfo"
gate "$scratch/s1" ok HW_GATE_BUILD_MEM_WAIT=1
if [ "$(builds "$scratch/s1")" != 0 ]; then fail "preflight: the compile started at 1 GB MemAvailable (rc=$rc)"
elif [ "$rc" != 2 ]; then fail "preflight: rc=$rc, want 2 (the box could not run the gate)"
elif ! grep -q "MemAvailable 1 GB (floor 24 GB)" "$scratch/s1/out/results.tsv"; then
  fail "preflight: the result row does not carry the reading: $(cat "$scratch/s1/out/results.tsv")"
else echo "ok   below the floor the compile is not started: rc=2 after ${secs}s, the row carries the reading"; fi

# 2 + 3. the guard, during the compile
cp "$scratch/mem-high" "$scratch/meminfo"
( sleep 4; cp "$scratch/mem-low" "$scratch/meminfo" ) &
gate "$scratch/s2" hang
wait
pid=$(cat "$scratch/s2/build.pid" 2> /dev/null); cp "$scratch/s2/build.pid" "$scratch/s2.pid" 2> /dev/null
if [ "$rc" = 124 ]; then fail "guard: the compile ran to the 120 s bound: nothing stopped it when memory fell"
elif [ "$rc" != 2 ]; then fail "guard: rc=$rc, want 2"
elif [ "$(builds "$scratch/s2")" != 1 ]; then fail "guard: $(builds "$scratch/s2") compiles, want 1 (a guard stop was resumed like a thermal stop)"
elif ! grep -q "memory guard" "$scratch/s2/out/results.tsv"; then fail "guard: the result row does not name the guard: $(cat "$scratch/s2/out/results.tsv")"
else echo "ok   memory fell during the compile: stopped once, rc=2 after ${secs}s, the row names the guard"; fi
sleep 1
if [ -n "$pid" ] && kill -0 "$pid" 2> /dev/null; then fail "guard: the compile ($pid) is still running after the guard stopped it"
elif [ -n "$pid" ]; then echo "ok   the stopped compile is gone"; fi

# 4. a GPU step's guard: gpu() runs "<gate> --cool --mem - mem-guard <floor> thermal-run <step>"; take the guard
# invocation from the script itself, so the test follows what gpu() really runs
guard=$(sed -n 's/.*"\$ROOT\/scripts\/mem-guard.sh" \(.*\)"\$MEM_FLOOR" "\$THERMAL" "\$@".*/\1/p' "$root/scripts/hw-gate.sh")
if [ "$(grep -c '"$ROOT/scripts/mem-guard.sh" .*"$MEM_FLOOR" "$THERMAL" "$@"' "$root/scripts/hw-gate.sh")" != 1 ]; then
  fail "gpu step: cannot find gpu()'s one mem-guard call in hw-gate.sh"
else
  cp "$scratch/mem-high" "$scratch/meminfo"
  ( sleep 3; cp "$scratch/mem-low" "$scratch/meminfo" ) &
  # shellcheck disable=SC2086  # $guard is the option word, or nothing
  MEM_GUARD_MEMINFO="$scratch/meminfo" timeout 60 "$root/scripts/mem-guard.sh" $guard 12 "$scratch/thermal.sh" \
      bash -c 'echo $$ > "$0"; exec sleep 600' "$scratch/s4.pid" 2> /dev/null
  wait; sleep 1
  pid=$(cat "$scratch/s4.pid" 2> /dev/null)
  if [ -z "$pid" ]; then fail "gpu step: the step never started"
  elif kill -0 "$pid" 2> /dev/null; then fail "gpu step: the step ($pid) is still running after the guard stopped it (mem-guard ${guard:-<no option>})"
  else echo "ok   a GPU step stopped by the guard is gone (mem-guard $guard)"; fi
fi

# 5. the GPU token is not held while gpu() waits for memory. gpu(), take_token() and release_token() are taken from
# the script as they are and run with a real token file; the pre-check ("$0 --cool --mem") is this harness, which
# refuses twice (76) and records who holds the token each time it is asked and during each wait.
h=$scratch/s5; mkdir -p "$h/out"
for f in take_token release_token gpu; do
  awk -v f="$f" '$0 ~ "^" f "\\(\\) \\{" { on = 1 } on { print } on && /^}/ { exit }' "$root/scripts/hw-gate.sh"
done > "$h/funcs.sh"
cat > "$h/harness.sh" << 'EOS'
#!/usr/bin/env bash
h=$(dirname "$0")
held() { if [ -e "$h/token" ]; then echo held; else echo free; fi; }
if [ "${1:-}" = --cool ]; then
  n=$(cat "$h/asked" 2> /dev/null || echo 0); echo $((n + 1)) > "$h/asked"
  echo "precheck $(held)" >> "$h/log"
  [ "$n" -lt 2 ] && exit 76
  exit 0
fi
TOKEN=$h/token TOKEN_WAIT=30 TOKEN_HELD="" OUT=$h/out ROOT=/nonexistent MEM_FLOOR=1 THERMAL=true infra=0
note() { :; }; result() { echo "result $*" >> "$h/log"; }; wait_coder_idle() { :; }; ring_timeouts() { echo 0; }
sleep() { echo "wait $(held)" >> "$h/log"; }
. "$h/funcs.sh"
gpu true; echo "rc $? token-after $(held)" >> "$h/log"
EOS
if [ "$(grep -c '^gpu() {' "$h/funcs.sh")" != 1 ] || ! grep -q '^take_token() {' "$h/funcs.sh" || ! grep -q '^release_token() {' "$h/funcs.sh"; then
  fail "token: cannot find gpu(), take_token() and release_token() in hw-gate.sh"
else
  chmod +x "$h/harness.sh"      # gpu() runs "$0 --cool ..."
  HW_GATE_MEM_RETRY_SLEEP=0 timeout 60 "$h/harness.sh" > "$h/stdout" 2>&1
  want='precheck held
wait free
precheck held
wait free
precheck held
rc 0 token-after held'
  if [ "$(cat "$h/log" 2> /dev/null)" = "$want" ]; then
    echo "ok   the GPU token is given back for each memory wait and held again for each try"
  else
    fail "token: gpu() held the GPU token while waiting for memory (or lost it for a try); the sequence was:"
    sed 's/^/     /' "$h/log" 2> /dev/null; sed 's/^/     | /' "$h/stdout" | head -n 5
  fi
fi

[ "$fails" = 0 ] || { echo "$fails check(s) failed"; exit 1; }
echo "ok   hw-gate memory checks"
