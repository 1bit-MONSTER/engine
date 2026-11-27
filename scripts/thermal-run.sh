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
# thermal-run.sh <cmd...>: run cmd at nice 19 under the box's thermal governor.
#
#   freeze the whole job (SIGSTOP) at THERMAL_HI  (78 C)
#   thaw it (SIGCONT) below  THERMAL_LO  (68 C)
#   kill it  (SIGKILL)  at or above THERMAL_KILL (93 C), exit 137
#   on TERM/INT/HUP kill the whole job and exit 143
#   if the sensor stops reading mid-run, thaw a frozen job and report it (the job runs ungoverned until it reads)
#
# It pins no CPU cores: the caller's -j decides how many compilers run.
#
# The last rule is why scripts/mem-guard.sh --term names this script: a SIGKILLed thermal-run cannot run
# its trap, and whatever it started keeps running and keeps the memory (engine#1504). Run the job in its
# own session so a SIGTERM here reaches ninja and every compiler under it, not just this process.
#
# Why this file exists: docs/ci-runners.md:145 and docs/hw-gate.md:223 — hw-gate.yml and hrx-build.yml call
# ~/lb/thermal-run.sh through the runner's linked ~/lb, and the file was missing from this box, so every
# hrx-build job died `line 2: /home/bcloud/lb/thermal-run.sh: No such file or directory` (exit 127) before
# configuring anything (engine#3517).
#
# shortcut: the job is frozen as a process *group* (kill -STOP -$pgid), not as a systemd --user scope with
# cgroup.freeze; a child that calls setsid() escapes the governor. Upgrade to `systemd-run --user --scope`
# + `systemctl --user freeze` if a build ever daemonises.
#
# Env: THERMAL_TCTL (temp file; default the first k10temp hwmon), THERMAL_HI/LO/KILL, THERMAL_POLL (1 s),
#      THERMAL_NICE (19), THERMAL_DISABLE=1 to run without the governor (a box with no readable sensor).
set -u
hi=${THERMAL_HI:-78} lo=${THERMAL_LO:-68} kill_c=${THERMAL_KILL:-93} poll=${THERMAL_POLL:-1}
nice_n=${THERMAL_NICE:-19}
[ "$#" -gt 0 ] || { echo "usage: thermal-run.sh <cmd...>" >&2; exit 2; }

tctl_f=${THERMAL_TCTL:-}
if [ -z "$tctl_f" ]; then
  for hw in /sys/class/hwmon/*/name; do
    case $(cat "$hw" 2>/dev/null) in k10temp|zenpower) tctl_f=$(dirname "$hw")/temp1_input; break;; esac
  done
fi
read_temp() {  # C, integer; empty when unreadable
  local v; v=$(cat "$tctl_f" 2>/dev/null) || return 0
  echo $(( v / 1000 ))
}

setsid nice -n "$nice_n" "$@" &
job=$!
stopped=0
cleanup() { kill -KILL -- "-$job" 2>/dev/null; wait "$job" 2>/dev/null; exit 143; }
trap cleanup TERM INT HUP

if [ "${THERMAL_DISABLE:-0}" = 1 ] || [ -z "$tctl_f" ] || [ -z "$(read_temp)" ]; then
  [ "${THERMAL_DISABLE:-0}" = 1 ] || echo "thermal-run: no readable sensor (THERMAL_TCTL=${tctl_f:-unset}); running ungoverned" >&2
  wait "$job"; rc=$?
  trap - TERM INT HUP
  exit $rc
fi

lost=0
while kill -0 "$job" 2>/dev/null; do
  t=$(read_temp)
  if [ -z "$t" ]; then
    # The sensor stopped answering mid-run. A job frozen at that moment would wait for a reading that never comes,
    # so thaw it and say so: an ungoverned job is visible in the log, a frozen one just looks hung.
    if [ "$lost" = 0 ]; then
      echo "thermal-run: cannot read $tctl_f; running ungoverned until it reads again" >&2
      lost=1
    fi
    if [ "$stopped" = 1 ]; then
      kill -CONT -- "-$job" 2>/dev/null; stopped=0
      echo "thermal-run: job thawed because the sensor is unreadable" >&2
    fi
  else
    if [ "$lost" = 1 ]; then echo "thermal-run: $tctl_f reads again (${t} C), governing" >&2; lost=0; fi
    if [ "$t" -ge "$kill_c" ]; then
      echo "thermal-run: ${t} C >= ${kill_c} C, killing the job (engine#1504 holds the queue at 75 C)" >&2
      kill -KILL -- "-$job" 2>/dev/null; wait "$job" 2>/dev/null
      trap - TERM INT HUP; exit 137
    elif [ "$t" -ge "$hi" ] && [ "$stopped" = 0 ]; then
      kill -STOP -- "-$job" 2>/dev/null; stopped=1
      echo "thermal-run: ${t} C, frozen until ${lo} C" >&2
    elif [ "$t" -le "$lo" ] && [ "$stopped" = 1 ]; then
      kill -CONT -- "-$job" 2>/dev/null; stopped=0
      echo "thermal-run: ${t} C, resumed" >&2
    fi
  fi
  sleep "$poll"
done
wait "$job"; rc=$?
[ "$stopped" = 1 ] && kill -CONT -- "-$job" 2>/dev/null
trap - TERM INT HUP
exit $rc
