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
# engine#315/#674: the trio stops that page fault, but on this box it does not make the device appear. The HAL names the
# HSA runtime with IREE_HAL_AMDGPU_LIBHSA_PATH and otherwise resolves the *system* ROCr, which with HSA_USE_SVM=0 registers
# no HRX0 at all -- `--list-devices` then prints "(none)" and every arm of a harness silently measures nothing (measured
# 2026-11-03 on a clean pin build: 10 arms of `-dev HRX0` all died in 1 s, and engine#674's own reproducer needed the
# export to run). serve resolves it for itself (--hrx-libhsa, then the baked ONEBIT_HRX_LIBHSA, then a scan of
# /opt/rocm-therock) and pushes it to its children; a tool started from a shell gets it from here or not at all. A
# caller that exports the variable keeps its own value, which is also the opt-out.
#
#   source scripts/hrx-env.sh          # set what is not already set; a caller's value wins
#   source scripts/hrx-env.sh --check  # ... and refuse to continue if a runtime that cannot work is named
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

# The runtime, same fallback serve uses, and only when the caller has not named one.
if [ -z "${IREE_HAL_AMDGPU_LIBHSA_PATH:-}" ]; then
    for _cand in /opt/rocm-therock/lib/python*/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1 \
                 /opt/rocm-therock/lib/libhsa-runtime64.so.1; do
        if [ -e "$_cand" ]; then
            export IREE_HAL_AMDGPU_LIBHSA_PATH=$_cand
            break
        fi
    done
    unset _cand
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
    # A path that does not exist is worse than an unset one: the HAL fails to dlopen and reports no device, which reads
    # like a GPU problem. Unset-but-nothing-found stays a warning, because a box whose system ROCr works is fine.
    if [ -n "${IREE_HAL_AMDGPU_LIBHSA_PATH:-}" ] && [ ! -e "$IREE_HAL_AMDGPU_LIBHSA_PATH" ]; then
        echo "hrx-env: IREE_HAL_AMDGPU_LIBHSA_PATH=$IREE_HAL_AMDGPU_LIBHSA_PATH does not exist, so the HAL cannot load a" >&2
        echo "         runtime and registers no HRX0 (--list-devices prints '(none)'). Point it at libhsa-runtime64.so.1." >&2
        return 3 2>/dev/null || exit 3
    elif [ -z "${IREE_HAL_AMDGPU_LIBHSA_PATH:-}" ]; then
        echo "hrx-env: no TheRock libhsa found and IREE_HAL_AMDGPU_LIBHSA_PATH is unset; the HAL will resolve the system" >&2
        echo "         ROCr. That works on a box with a full ROCm install and registers no HRX0 on a box whose device only" >&2
        echo "         opens through TheRock's runtime -- check --list-devices before trusting a measurement." >&2
    fi
    # engine#715: three HSA runtimes live on this box and a run has to name the one it used, or a fault report is
    # ambiguous. Only --check prints, so sourcing this file stays silent for the callers that do not ask.
    if [ -n "${IREE_HAL_AMDGPU_LIBHSA_PATH:-}" ] && [ -e "$IREE_HAL_AMDGPU_LIBHSA_PATH" ]; then
        if command -v sha256sum >/dev/null 2>&1; then
            echo "hrx-env: runtime $IREE_HAL_AMDGPU_LIBHSA_PATH sha256 $(sha256sum "$IREE_HAL_AMDGPU_LIBHSA_PATH" | cut -c1-16)" >&2
        else
            echo "hrx-env: runtime $IREE_HAL_AMDGPU_LIBHSA_PATH (no sha256sum on this machine)" >&2
        fi
    fi
fi
unset _hrx_env_opt_out _v
