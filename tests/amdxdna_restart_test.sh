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
# tests/amdxdna_restart_test.sh — the carried amdxdna series applies, in order, to the pinned driver, and the
# context-restart code it produces fails a context's jobs when the restart fails (engine#1750, engine#1855).
#
# Usage: amdxdna_restart_test.sh [repo root]
#
# No NPU, no kernel headers, no module: the series is applied to a copy of the driver source and the four
# functions under test are compiled into a user-space harness (tests/amdxdna_restart_harness.c says what is
# stubbed). The driver source is the engine's submodule (third_party/xdna-driver) or the git checkout
# ONEBIT_XDNA_SRC names; with neither at the pin, the test skips (77).
set -u
root=$(cd "${1:-$(dirname "$0")/..}" && pwd)
series=$root/third_party/patches/amdxdna
pin=$(git -C "$root" ls-tree HEAD third_party/xdna-driver 2> /dev/null | awk '{print $3}')
[ -n "$pin" ] || { echo "skip: cannot read the third_party/xdna-driver pin (not a git checkout)"; exit 77; }
src=""
for cand in "${ONEBIT_XDNA_SRC:-}" "$root/third_party/xdna-driver"; do
  [ -n "$cand" ] && git -C "$cand" cat-file -e "$pin^{commit}" 2> /dev/null && { src=$cand; break; }
done
[ -n "$src" ] || { echo "skip: no driver source at the pin ${pin:0:12} (check out third_party/xdna-driver, or set ONEBIT_XDNA_SRC)"; exit 77; }
cc=${CC:-cc}
command -v "$cc" > /dev/null || { echo "skip: no C compiler"; exit 77; }

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
mkdir "$scratch/tree"
git -C "$src" archive "$pin" drivers/accel/amdxdna | tar -x -C "$scratch/tree" || { echo "FAIL cannot export the driver at ${pin:0:12}"; exit 1; }
git -C "$scratch/tree" init -q
n=0
for p in "$series"/[0-9][0-9][0-9][0-9]-*.patch; do
  if ! git -C "$scratch/tree" apply "$p" 2> "$scratch/apply.err"; then
    echo "FAIL $(basename "$p") does not apply on the pin ${pin:0:12} after the $n patch(es) before it:"
    sed 's/^/     /' "$scratch/apply.err" | head -n 5
    exit 1
  fi
  n=$((n + 1))
done
echo "ok   $n patches apply in order on ${pin:0:12}"

# The test-only fault injection (engine#1750) has to keep applying on top of the series, or the one run it exists
# for cannot be built. It is checked, not applied: the harness below is the carried code. And it must stay out of
# the numbered glob above, which is what a package build takes.
inject=$series/test-only/restart-fault-injection.patch
if ! git -C "$scratch/tree" apply --check "$inject" 2> "$scratch/apply.err"; then
  echo "FAIL test-only/restart-fault-injection.patch does not apply on top of the $n carried patches:"
  sed 's/^/     /' "$scratch/apply.err" | head -n 5
  exit 1
fi
echo "ok   test-only/restart-fault-injection.patch applies on top of them (checked, not applied)"
if ls "$series"/[0-9][0-9][0-9][0-9]-*.patch | xargs grep -l "TEST ONLY" > "$scratch/leak" 2> /dev/null; then
  echo "FAIL a numbered patch carries TEST ONLY code: $(tr '\n' ' ' < "$scratch/leak")"
  exit 1
fi
echo "ok   no numbered patch carries TEST ONLY code"

# one function definition: from its header line (column 0, not a prototype) to the first closing brace in column 0
for f in aie2_hwctx_job_pause aie2_hwctx_stop aie2_hwctx_restart aie2_hwctx_job_unpause; do
  awk -v f="$f" '!on && $0 ~ "^[a-z].*[ *]" f "\\(" && $0 !~ /;[ \t]*$/ { on = 1 } on { print } on && /^}/ { exit }' \
      "$scratch/tree/drivers/accel/amdxdna/aie2_ctx.c" > "$scratch/$f.c"
  [ "$(tail -n 1 "$scratch/$f.c")" = "}" ] || { echo "FAIL cannot find $f() in the patched aie2_ctx.c"; exit 1; }
  cat "$scratch/$f.c"
done > "$scratch/funcs.inc"

if ! "$cc" -std=gnu11 -O1 -Wall -Wno-unused-function -DRESTART_FUNCS="\"$scratch/funcs.inc\"" \
     -o "$scratch/harness" "$root/tests/amdxdna_restart_harness.c" 2> "$scratch/cc.err"; then
  echo "FAIL the restart code no longer compiles into the harness (a new callee needs a stub?):"
  sed 's/^/     /' "$scratch/cc.err" | head -n 20
  exit 1
fi
"$scratch/harness"
