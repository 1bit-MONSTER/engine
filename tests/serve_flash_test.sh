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
# `1bit serve --flash` (engine #1785) is a name for two settings: --ctx-size 1024 and --prefix-pin on --device npu.
# What contradicts the profile must be refused by name, before any model is opened or any device is touched, instead
# of one side quietly winning.
#
# No model, no device, no port: every case here ends in the option check. The model path does not exist on purpose;
# a case that got past the check would fail on it with a different message, and the pattern would not match.
#
#   serve_flash_test.sh <1bit binary>
set -uo pipefail

bin=${1:?usage: serve_flash_test.sh <1bit binary>}
[ -x "$bin" ] || { echo "serve_flash_test: $bin is not executable"; exit 1; }

fail=0
missing=/nonexistent/serve-flash-test-model
# refused NAME PATTERN ARGS...: the command must exit non-zero within the limit and name the reason.
refused() {
    local name=$1 pattern=$2 out rc
    shift 2
    out=$(timeout 20 "$bin" serve -m "$missing" "$@" 2>&1)
    rc=$?
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && printf '%s' "$out" | grep -qF -- "$pattern"; then
        echo "ok   $name"
    else
        echo "FAIL $name (rc=$rc): $out"
        fail=$(( fail + 1 ))
    fi
}
# not_refused_by_flash NAME ARGS...: the command may fail (the model is missing), but not on a --flash rule and
# not in the option parser.
not_refused_by_flash() {
    local name=$1 out rc
    shift
    out=$(timeout 20 "$bin" serve -m "$missing" "$@" 2>&1)
    rc=$?
    if [ "$rc" -ne 124 ] && ! printf '%s' "$out" | grep -qF -e "--flash" -e "unknown option"; then
        echo "ok   $name"
    else
        echo "FAIL $name (rc=$rc): $out"
        fail=$(( fail + 1 ))
    fi
}

refused "another context size is refused"     "--flash fixes the context at 1024 tokens; drop --ctx-size 2048" \
        --device npu --flash --ctx-size 2048
refused "and so is one given before --flash"  "--flash fixes the context at 1024 tokens; drop --ctx-size 512" \
        --device npu --ctx-size 512 --flash
refused "a cache turned off is refused"       "--flash needs the prefix cache; drop --prefix-slots 0" \
        --device npu --flash --prefix-slots 0
refused "another device is refused"           "--flash is a profile of --device npu" \
        --device cpu --flash
refused "no device at all is refused too"     "--flash is a profile of --device npu" \
        --flash

not_refused_by_flash "its own context size, spelled out, is accepted" --device npu --flash --ctx-size 1024
not_refused_by_flash "--prefix-pin next to it is accepted"            --device npu --flash --prefix-pin
not_refused_by_flash "without --flash none of the rules apply"        --device cpu --ctx-size 2048 --prefix-slots 0

[ "$fail" -eq 0 ] && echo "serve_flash: ok" || echo "serve_flash: $fail failed"
exit "$fail"
