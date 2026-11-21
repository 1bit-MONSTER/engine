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
# laya_e2e_guard.sh -- the checkpoint half of a Laya e2e row's contract (engine#3161).
#
# The seven Laya e2e rows are registered ONLY when ONEBIT_LAYA_ROUTE_MODEL names a checkpoint, and nothing in this tree
# or in CI ever sets it -- so the rail they cover has never had a row that runs, and `ctest -N` did not list them at all
# (the engine#3000 shape). Registering them unconditionally with SKIP_RETURN_CODE 77 makes the gap VISIBLE: the rows
# appear, they say why they are not runnable here, and `tools/ci/ctest_skips.py --write` baselines the reason.
#
# Usage: laya_e2e_guard.sh <checkpoint-dir> <command> [args...]
#   0    the command ran (its own exit status otherwise)
#   77   no checkpoint configured, or the named directory is not there -- with the reason on stdout
set -uo pipefail
dir=${1:-}
[ $# -ge 2 ] || { echo "usage: laya_e2e_guard.sh <checkpoint-dir> <command> [args...]" >&2; exit 2; }
shift
if [ -z "$dir" ]; then
    echo "SKIPPED: no Laya checkpoint configured -- pass -DONEBIT_LAYA_ROUTE_MODEL=<checkpoint dir> to run this row"
    exit 77
fi
if [ ! -d "$dir" ]; then
    echo "SKIPPED: ONEBIT_LAYA_ROUTE_MODEL=$dir is not a directory"
    exit 77
fi
exec "$@"
