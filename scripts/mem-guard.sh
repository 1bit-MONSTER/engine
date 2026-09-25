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
# mem-guard.sh <min_available_gb> <cmd...>: run cmd and kill it if MemAvailable drops below
# the floor, before the kernel's OOM killer picks other processes (GPU allocations on Strix
# Halo's unified memory escape cgroup limits, so MemoryMax alone does not protect the box).
floor=$1; shift
"$@" & pid=$!
while kill -0 $pid 2>/dev/null; do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$avail" -lt "$floor" ]; then echo "GUARD: MemAvailable ${avail} GB < ${floor} GB, killing $pid" >&2; kill -9 $pid; fi
  sleep 0.5
done
wait $pid
