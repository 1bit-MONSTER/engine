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
# mem-guard.sh [--term] <min_available_gb> <cmd...>: run cmd and kill it if MemAvailable drops below
# the floor, before the kernel's OOM killer picks other processes (GPU allocations on Strix
# Halo's unified memory escape cgroup limits, so MemoryMax alone does not protect the box).
# --term sends SIGTERM first and SIGKILL 10 s later. Use it when cmd is ~/lb/thermal-run.sh: a
# SIGKILL'd thermal-run cannot run its trap, so its scope (ninja and every compiler under it) keeps
# running and keeps the memory; on SIGTERM it kills the whole scope (engine#1504).
# MEM_GUARD_MEMINFO names the file read instead of /proc/meminfo (the tests: "plenty, then starved" cannot be
# produced on demand on a shared box).
meminfo=${MEM_GUARD_MEMINFO:-/proc/meminfo}
term=0; [ "$1" = --term ] && { term=1; shift; }
floor=$1; shift
"$@" & pid=$!
while kill -0 $pid 2>/dev/null; do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' "$meminfo")
  if [ "$avail" -lt "$floor" ]; then
    echo "GUARD: MemAvailable ${avail} GB < ${floor} GB, killing $pid" >&2
    if [ $term = 1 ]; then
      kill -TERM $pid
      for _ in $(seq 20); do kill -0 $pid 2>/dev/null || break; sleep 0.5; done
    fi
    kill -9 $pid 2>/dev/null
  fi
  sleep 0.5
done
wait $pid
