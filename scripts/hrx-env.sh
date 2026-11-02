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
# engine#715: the one place that puts the engine#315 save-area variables into a *shell's* environment, for the HRX
# processes the engine does not launch itself -- test-backend-ops, llama-bench, llama-cli, a harness that starts
# llama-server on its own. serve sets them for its backends and for itself (app/hrx_cwsr.h, apply_cwsr_env_to_self);
# a script-launched binary gets them from here or not at all, and without them ROCr page-faults on the first HRX
# graph replay on a box whose libhsa is the TheRock nightly (docs/hrx.md, the two tables).
#
#   source scripts/hrx-env.sh          # set what is not already set; a caller's value wins
#   source scripts/hrx-env.sh --check  # ... and refuse to continue if the trio is not in place afterwards
#
# HRX_SVM=1 opts out (a box whose ROCr does not need the GTT placement), exactly as in serve. Opting out is allowed
# but it is not silent: the caller sees the line below, because that mode is the one that faults here.

_hrx_env_opt_out=0
if [ "${HRX_SVM:-}" = 1 ]; then
    _hrx_env_opt_out=1
    echo "hrx-env: HRX_SVM=1; leaving the save-area placement to ROCr (engine#315 opt-out, engine#715)" >&2
else
    export HSA_USE_SVM=${HSA_USE_SVM:-0}
    export HSA_USERPTR_FOR_PAGED_MEM=${HSA_USERPTR_FOR_PAGED_MEM:-0}
    export HRX_ALLOW_NO_SVM=${HRX_ALLOW_NO_SVM:-1}
fi

if [ "${1:-}" = --check ] && [ "$_hrx_env_opt_out" = 0 ]; then
    for _v in HSA_USE_SVM HSA_USERPTR_FOR_PAGED_MEM HRX_ALLOW_NO_SVM; do
        if [ -z "${!_v:-}" ]; then
            echo "hrx-env: $_v is not set, so an HRX process started from here would page-fault on its first" >&2
            echo "         graph replay (engine#715). Export the trio, or HRX_SVM=1 to opt out deliberately:" >&2
            echo "           HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1" >&2
            return 3 2>/dev/null || exit 3
        fi
    done
fi
unset _hrx_env_opt_out _v
