#!/bin/bash
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
# engine#3161: the Laya rail's e2e rows were registered only when ONEBIT_LAYA_ROUTE_MODEL was set, and no CI job or
# default configure sets it -- so `laya_route_e2e`/`laya_models_e2e`/... had never run anywhere, and the rail they
# cover changed with nothing testing it. The fix is the pattern every other un-runnable row on this box already uses:
# register the row ALWAYS and make a missing checkpoint a visible SKIP (77) with a reason, instead of an absent test.
#
# usage: laya_e2e_guard.sh <row-script> <bin> <checkpoint-dir>
set -uo pipefail
row=${1:-}; bin=${2:-}; model=${3:-}
if [ -z "$row" ] || [ ! -x "$row" ]; then
    echo "laya_e2e_guard: no row script at '${row}'" >&2
    exit 1
fi
if [ -z "$model" ] || [ ! -d "$model" ]; then
    echo "skip $(basename "$row"): ONEBIT_LAYA_ROUTE_MODEL is '${model:-<empty>}', which is not a checkpoint directory" \
         "(--laya-models needs a converted Laya checkpoint; docs/laya-models.md). engine#3161" >&2
    exit 77
fi
exec "$row" "$bin" "$model"
