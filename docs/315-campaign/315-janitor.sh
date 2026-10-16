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
# engine#315 janitor: stop arm scopes whose arm is ALREADY SCORED, so a slow teardown cannot inflate the next arm's
# baseline. Seen on 2026-10-10: `315row-arm-1l.scope` stayed resident 15 minutes after its arm was scored OK, and
# card GTT read 113.5 GiB instead of ~100 (evidence/315-queue-profile.md, appendix).
#
# Safety rule, and the reason this cannot disturb a measurement: a scope is stopped only when the run's own
# console already contains that arm's verdict line (`pair <n> arm <c>: <verdict>`), i.e. only after the detector
# has finished scoring it. An arm still being scored has no verdict line and is left alone, however long it runs.
# It stops declared work only - the scopes belong to this run's own tag - and never touches neighbours.
#
# Usage: 315-janitor.sh <tag> [minutes]
set -uo pipefail
TAG=${1:?usage: 315-janitor.sh <tag> [minutes]}
MINUTES=${2:-240}
CONSOLE=$HOME/wt/$TAG-console.log
DEADLINE=$((SECONDS + MINUTES * 60))

echo "janitor: watching $TAG arm scopes for ${MINUTES} min $(date -Is)"
while [ $SECONDS -lt $DEADLINE ]; do
  # nothing to clean once the runner is gone, unless scopes are still up - then clean once more and exit
  runner=0
  pgrep -f "315-(attrib|paired)\.sh .* $TAG" >/dev/null 2>&1 && runner=1
  for u in $(systemctl --user list-units --state=running "$TAG-arm-*" --no-legend 2>/dev/null | awk '{print $1}'); do
    n=${u#$TAG-arm-}; n=${n%.scope}
    pair=${n%?}; cell=${n: -1}
    case "$pair$cell" in
      [0-9]*[cl]) ;;
      *) continue ;;
    esac
    if [ -f "$CONSOLE" ] && grep -q "pair $pair arm $cell:" "$CONSOLE"; then
      echo "janitor: $u scored already, stopping it $(date +%T); gtt=$(( $(cat /sys/class/drm/card*/device/mem_info_gtt_used) / 1073741824 ))G"
      systemctl --user stop "$u" 2>/dev/null
    fi
  done
  [ "$runner" = 0 ] && break
  sleep 60
done
echo "janitor: done $(date -Is); gtt=$(( $(cat /sys/class/drm/card*/device/mem_info_gtt_used) / 1073741824 ))G"
