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
# empty_selector_test.sh: the private-name lists of scripts/weekly.sh and scripts/make-binary-drop.sh (engine#2469).
#
# Under `set -euo pipefail` a `for ... done | grep ...` pipeline fails when the selector matches nothing, so an empty
# (and perfectly valid) answer either killed the script or -- inside weekly.sh's package_checks, which is called with
# `||` so set -e is off -- left the list empty and turned the check reading it into an `ok` that examined nothing.
# This cuts the two list helpers out of the real scripts and runs them on fixture pins, under pipefail:
#
#   1. a pin that tracks only README/LICENSE, a pin whose checkout is not a git repo, a missing pin, and a config
#      with no require= line all give an EMPTY answer with status 0;
#   2. a pin with real sources gives exactly their basenames, so the lists are not trivially empty;
#   3. both call sites in package_checks print `skip` for an empty list instead of `ok`.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
fails=0
ok()  { printf '  ok   %s\n' "$1"; }
bad() { printf '  FAIL %s\n' "$1"; fails=$((fails+1)); }
cut_fn() { sed -n "/^$1() {\$/,/^}\$/p" "$2"; }

cut_fn private_tracked_names "$root/scripts/weekly.sh" > "$tmp/fn.sh"
cut_fn require_names "$root/scripts/make-binary-drop.sh" >> "$tmp/fn.sh"
grep -q '^private_tracked_names() {' "$tmp/fn.sh" || { bad "could not cut private_tracked_names() out of scripts/weekly.sh"; exit 1; }
grep -q '^require_names() {' "$tmp/fn.sh" || { bad "could not cut require_names() out of scripts/make-binary-drop.sh"; exit 1; }
# shellcheck source=/dev/null
. "$tmp/fn.sh"

# fixture: $W holds the private pin checkouts, $SRC/config their path= / require= lines
# shellcheck disable=SC2034  # src is read by require_names (make-binary-drop.sh names it src, weekly.sh SRC)
W=$tmp/w SRC=$tmp/src src=$tmp/src
mkdir -p "$W" "$SRC/config"
mkrepo() {  # mkrepo <pin> <file>...
    local d=$W/$1; shift
    git init -q "$d"
    for f in "$@"; do mkdir -p "$d/$(dirname "$f")"; echo x > "$d/$f"; done
    git -C "$d" add -A
}
mkrepo npu-runtime addons/README.md addons/LICENSE
mkrepo npu-kernels addons/k/attn.cpp addons/k/attn.h addons/NOTICE tools/outside.py
mkdir -p "$W/broken/addons"; echo x > "$W/broken/addons/x.c"          # a checkout that is not a git repo
for p in npu-runtime npu-kernels gpu-kernels broken; do echo "path=addons" > "$SRC/config/$p.config"; done
echo "require=lib/a/libnpu_rt.a include/npu_rt.h" >> "$SRC/config/npu-runtime.config"

run() {  # run <want-output> <label> <command...>: status 0 and exactly <want-output>, under pipefail
    local want=$1 label=$2 out rc; shift 2
    if out=$("$@"); then rc=0; else rc=$?; fi
    out=$(printf '%s' "$out" | tr '\n' ' ' | sed 's/ $//')
    if [ "$rc" = 0 ] && [ "$out" = "$want" ]; then ok "$label -> [${out}]"
    else bad "$label -> rc=$rc [${out}], want rc=0 [${want}]"; fi
}
privsrc() { private_tracked_names "$@" | { grep -E '\.(c|cc|cpp|cxx|h|hpp|py)$' || true; }; }

run ""                     "names: pin tracking only README/LICENSE"            private_tracked_names npu-runtime
run ""                     "names: missing pin, then a non-git checkout last"   private_tracked_names gpu-kernels broken
run "attn.cpp attn.h"      "names: a pin with sources, an empty pin last"       private_tracked_names npu-kernels npu-runtime
run ""                     "privsrc: no .c/.h/.py in the pins"                  privsrc npu-runtime broken
run "attn.cpp attn.h"      "privsrc: sources present"                           privsrc npu-runtime npu-kernels
run "libnpu_rt.a npu_rt.h" "require=: a config with no require= line last"      require_names npu-runtime gpu-kernels
run ""                     "require=: no config has a require= line"            require_names gpu-kernels

# the call sites read the helpers, and an empty list is a skip, not an ok
pc=$(sed -n '/^package_checks() {/,/^}$/p' "$root/scripts/weekly.sh")
grep -q 'names=$(private_tracked_names ' <<<"$pc" && grep -q 'privsrc=$(private_tracked_names ' <<<"$pc" &&
    ok "package_checks builds both lists with private_tracked_names" || bad "package_checks no longer uses private_tracked_names"
[ "$(grep -c '^ *echo "skip ' <<<"$pc")" -ge 2 ] && ok "package_checks prints skip for an empty list (2 sites)" ||
    bad "package_checks has no skip line for an empty private list"
grep -q '^names=$(require_names ' "$root/scripts/make-binary-drop.sh" && ok "make-binary-drop.sh builds names with require_names" ||
    bad "make-binary-drop.sh no longer uses require_names"

[ "$fails" = 0 ] && { echo "empty_selector: PASS"; exit 0; }
echo "empty_selector: FAIL ($fails)"; exit 1
