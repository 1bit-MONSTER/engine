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
# The NPU runtime options (docs/npu.md, "The runtime"): ONEBIT_NPU=ON must stop with a message that
# says why when ONEBIT_NPU_RUNTIME is anything but own (the engine has one NPU runtime, its own), when
# neither or both of ONEBIT_NPU_RT_LIB and ONEBIT_NPU_RT_DIR are set, when ONEBIT_NPU_RT_LIB names no
# file, or when ONEBIT_NPU_RT_DIR is missing, empty or lacks the runtime's files. Needs no NPU and not the private
# runtime, so it runs in CI; reuses the parent build's fetched dependencies (offline).
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
mkdir -p "$tmp/half" && touch "$tmp/half/npu_rt.h"   # a header but no npu_rt.cpp
mkdir -p "$tmp/noapi" && touch "$tmp/noapi/npu_rt.h" "$tmp/noapi/npu_rt.cpp"   # no npu_rt_api.cpp
touch "$tmp/libnpu_rt_api.a"
expect not_own "ONEBIT_NPU_RUNTIME must be own" -DONEBIT_NPU_RUNTIME=external
expect bad_value "ONEBIT_NPU_RUNTIME must be own" -DONEBIT_NPU_RUNTIME=bogus
expect no_dir "ONEBIT_NPU needs the NPU runtime" -DONEBIT_NPU_RT_DIR=
expect missing_dir "ONEBIT_NPU needs the NPU runtime" -DONEBIT_NPU_RT_DIR="$tmp/none"
expect half_dir "ONEBIT_NPU needs the NPU runtime" -DONEBIT_NPU_RT_DIR="$tmp/half"
expect no_api_dir "ONEBIT_NPU needs the NPU runtime" -DONEBIT_NPU_RT_DIR="$tmp/noapi"
expect missing_lib "is not a library file" -DONEBIT_NPU_RT_LIB="$tmp/none.a"
expect dir_as_lib "is not a library file" -DONEBIT_NPU_RT_LIB="$tmp"
expect lib_and_dir "not both" -DONEBIT_NPU_RT_LIB="$tmp/libnpu_rt_api.a" -DONEBIT_NPU_RT_DIR="$tmp/noapi"
[ $fail -eq 0 ] && echo PASS
exit $fail
