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
# mem-wait.sh <min_available_gb> <max_wait_min>: return 0 once MemAvailable is at least the floor; poll once a
# minute and give up with exit 75 after max_wait_min. A preflight for a heavy job on the shared box: starting a
# compile into an exhausted box is what turned one OOM into two on 2026-11-08 (engine#1504; both self-hosted
# runners were OOM-killed 76 s after a restart, with 0 swap free). Each poll prints what the box looks like, so a
# refusal says why. The floor is MemAvailable only: on this APU GTT is host RAM, and MemAvailable already counts it.
# MEM_GUARD_MEMINFO names the file read instead of /proc/meminfo, and MEM_WAIT_POLL the seconds between polls
# (default 60); both are for the tests.
floor=${1:?usage: mem-wait.sh <min_available_gb> <max_wait_min>}; max=${2:?usage: mem-wait.sh <min_available_gb> <max_wait_min>}
meminfo() { awk -v k="$1" '$1 == k":" {print int($2/1048576)}' "${MEM_GUARD_MEMINFO:-/proc/meminfo}"; }
for i in $(seq 0 "$max"); do
  avail=$(meminfo MemAvailable)
  line="MemAvailable ${avail} GB (floor ${floor} GB), swap free $(meminfo SwapFree) of $(meminfo SwapTotal) GB"
  if [ "$avail" -ge "$floor" ]; then echo "mem-wait: ok: $line"; exit 0; fi
  [ "$i" = "$max" ] && break
  echo "mem-wait: waiting ($((i + 1)) of $max min): $line"
  sleep "${MEM_WAIT_POLL:-60}"
done
echo "mem-wait: REFUSED after ${max} min: $line. The box is short of memory, so this job did not start its build;" \
     "re-run it once memory frees (engine#1504)." >&2
exit 75
