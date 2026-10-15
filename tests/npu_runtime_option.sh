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
# ONEBIT_NPU_RUNTIME (docs/npu.md, "The runtime"): configuring with a value other than xrt|own, or
# with own and no runtime directory, must stop with a message that says so. Needs neither XRT nor
# an NPU, so it runs in CI; reuses the parent build's fetched dependencies (offline).
#
# usage: tests/npu_runtime_option.sh <source dir> <FetchContent base dir>
set -uo pipefail
src=${1:?usage: npu_runtime_option.sh <source dir> <fetchcontent dir>}
deps=${2:?}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0
expect() {  # expect <name> <message fragment> <cmake args...>
    local name=$1 msg=$2
    shift 2
    local out
    out=$(cmake -S "$src" -B "$tmp/$name" -DFETCHCONTENT_BASE_DIR="$deps" -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
          -DONEBIT_NPU=ON "$@" 2>&1)
    local rc=$?
    if [ $rc -ne 0 ] && [[ "$out" == *"$msg"* ]]; then echo "ok   $name"; else
        echo "FAIL $name (rc $rc)"; echo "$out" | tail -15; fail=1; fi
}
expect bad_value "ONEBIT_NPU_RUNTIME must be xrt or own" -DONEBIT_NPU_RUNTIME=bogus
expect own_without_dir "ONEBIT_NPU_RUNTIME=own needs ONEBIT_NPU_RT_DIR" -DONEBIT_NPU_RUNTIME=own -DONEBIT_NPU_RT_DIR="$tmp/none"
[ $fail -eq 0 ] && echo PASS
exit $fail
