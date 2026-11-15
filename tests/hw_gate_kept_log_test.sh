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
# hw-gate keeps a failing row's server log (engine#2500, #2513) under $HW_GATE_STORE/failed and names it in the
# failure line. The name must be unique per failure: CI sets HW_GATE_OUT=$RUNNER_TEMP/hw-gate, so a name built from
# basename "$OUT" was the constant "hw-gate-<model>.log" and every failing run on every PR overwrote the last one's
# evidence. These arms run the CI shape (same model, same run, same second), and check that a failed copy prints no
# path. A recording fake gpu-hold is first on PATH, so the internal mode is also proven to take no GPU ticket.
# No GPU, no model, runs in CI.
set -u
src=$(cd "${1:-$(dirname "$0")/..}" && pwd)
scratch=$(mktemp -d "${TMPDIR:-$HOME/.cache}/hw-gate-kept-log.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
fails=0
fail() { echo "FAIL $*"; fails=$((fails + 1)); }

mkdir -p "$scratch/bin"
printf '#!/bin/sh\necho "$*" >> "%s/hold.calls"\nexit 99\n' "$scratch" > "$scratch/bin/gpu-hold"
chmod +x "$scratch/bin/gpu-hold"
printf 'srv  load_model: loading\nboom: null tensor\n' > "$scratch/server-zaya1-8b.log"
keep() { PATH="$scratch/bin:$PATH" HW_GATE_STORE="$scratch/store" GITHUB_RUN_ID=43700000001 GITHUB_RUN_ATTEMPT=1 \
         bash "$src/scripts/hw-gate.sh" --keep-failed-log "$@"; }

# 1. the CI shape: two failures of one model in one run, in the same second -> two files, both intact
a=$(keep "$scratch/server-zaya1-8b.log" zaya1-8b); b=$(keep "$scratch/server-zaya1-8b.log" zaya1-8b)
[ -n "$a" ] && [ -f "$a" ] || fail "the first failure was not kept (got '$a')"
[ -n "$b" ] && [ -f "$b" ] || fail "the second failure was not kept (got '$b')"
[ "$a" != "$b" ] || fail "two failures got one name: $a (the second overwrites the first)"
for f in "$a" "$b"; do cmp -s "$scratch/server-zaya1-8b.log" "$f" || fail "$f is not the server log"; done
# 2. the name carries the run and the model, so the CI page and the file can be matched
case "$a" in *run43700000001.1-zaya1-8b-*.log) ;; *) fail "the kept name does not carry the run id and model: $a" ;; esac
# 3. a copy that fails prints nothing, so the failure line falls back to the original path, and leaves no empty file
n0=$(ls "$scratch/store/failed" | wc -l)
c=$(keep "$scratch/missing.log" zaya1-8b)
[ -z "$c" ] || fail "a failed copy printed a path: $c"
[ "$(ls "$scratch/store/failed" | wc -l)" = "$n0" ] || fail "a failed copy left a file behind"
# 4. the internal mode answers before the GPU queue
[ ! -s "$scratch/hold.calls" ] || fail "--keep-failed-log called gpu-hold: $(head -n 1 "$scratch/hold.calls")"

[ "$fails" = 0 ] && echo "hw_gate_kept_log: ok (unique per failure in the CI shape; run id + model in the name; a failed copy names nothing; no ticket)"
exit "$fails"
