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
# laya_e2e_guard.sh <script> <bin> [<checkpoint>] [args...]: the Laya e2e rows' precondition, in one place.
#
# engine#3161: those rows were registered only when ONEBIT_LAYA_ROUTE_MODEL was set, so they ran in no job at all. They
# are registered unconditionally now, and this decides what "no checkpoint" means:
#   * not configured (or an empty value) -> 77 with a reason naming the variable and the installer: a SKIP that ctest
#     reports as "did not run", never a pass;
#   * configured but not there -> 2, i.e. a FAILURE: a path that was passed and is wrong is a mistake, not an environment;
#   * configured and present -> the row runs.
set -uo pipefail
script=${1:?usage: laya_e2e_guard.sh <script> <bin> [<checkpoint>] [args...]}
shift
ckpt=${1:-}
if [ -z "$ckpt" ]; then
    echo "skip: ONEBIT_LAYA_ROUTE_MODEL is not set, so there is no Laya checkpoint on this box (scripts/fetch-laya.sh installs the pinned one; its typed-decisions/ subtree is 808 MiB)"
    exit 77
fi
if [ ! -e "$ckpt" ]; then
    echo "laya_e2e_guard: ONEBIT_LAYA_ROUTE_MODEL=$ckpt does not exist: pass a real checkpoint dir, or configure with it unset so these rows skip" >&2
    exit 2
fi
exec "$script" "$@"
