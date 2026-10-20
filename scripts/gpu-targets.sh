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
# gpu-targets.sh: the gfx targets of this machine's AMD GPUs (gfx1151, gfx1201, ...), one per
# line, from the KFD topology in sysfs (/sys/class/kfd/kfd/topology/nodes/*/properties). It reads
# sysfs only: no GPU is opened, so it works with ROCR_VISIBLE_DEVICES=-1 and on a busy GPU.
# Prints nothing when there is no AMD GPU (or no amdgpu/KFD). Used by tests/serve_e2e.sh (HRX
# skips without its target) and scripts/weekly.sh (names the host's GPU in a held bump).
set -euo pipefail
for p in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    [ -r "$p" ] || continue
    v=$(awk '$1 == "gfx_target_version" {print $2}' "$p")
    [ -n "$v" ] && [ "$v" != 0 ] || continue          # CPU nodes report 0
    # gfx_target_version is major*10000 + minor*100 + stepping; gfx names minor and stepping in hex
    printf 'gfx%d%x%x\n' $((v / 10000)) $((v / 100 % 100)) $((v % 100))
done | sort -u
