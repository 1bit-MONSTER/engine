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
# pin_names_test.sh: scripts/pin-names.sh under `set -euo pipefail`, the way the release scripts run it (engine#2469).
# An empty selection (no require= line, no source file in a pin, a pin not checked out) must be an empty list with
# status 0, never a pipeline failure; a non-empty one must list the names. The old inline form is kept as a negative
# control: it fails on the same input, which is the bug.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/../scripts/pin-names.sh"

fails=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fails=$((fails + 1)); fi; }

d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
mkdir -p "$d/src/config" "$d/work"
mkrepo() {   # mkrepo <pin> <file>...
    local r=$d/work/$1
    shift
    git init -q "$r"
    for f in "$@"; do mkdir -p "$r/$(dirname "$f")"; echo x > "$r/$f"; done
    git -C "$r" add -A
    git -C "$r" -c user.name=t -c user.email=t@t commit -q -m init
}

# require= names
printf 'path=x\n' > "$d/src/config/norequire.config"
printf 'require=a/b.c x/y.h\npath=addons\n' > "$d/src/config/withrequire.config"
out=$(pin_require_names "$d/src" norequire); check "no require= line: empty list, status 0" "$out" ""
out=$(pin_require_names "$d/src" withrequire norequire); check "require= names, last pin empty" "$out" "$(printf 'b.c\ny.h')"
out=$(pin_require_names "$d/src" absent); check "a missing config is an empty list" "$out" ""

# tracked names and private sources
mkrepo docsonly addons/README.md addons/notes.md addons/LICENSE
mkrepo withsrc addons/k.cc addons/k.h addons/run.py addons/data.bin other/skip.c
printf 'path=addons\n' > "$d/src/config/docsonly.config"
printf 'path=addons\n' > "$d/src/config/withsrc.config"
out=$(pin_tracked_names "$d/work" "$d/src" docsonly); check "tracked names drop README.md/NOTICE/LICENSE" "$out" "notes.md"
out=$(pin_private_sources "$d/work" "$d/src" docsonly); check "a pin with no source file: empty list, status 0" "$out" ""
out=$(pin_private_sources "$d/work" "$d/src" withsrc docsonly); check "source files under path= only, last pin has none" "$out" "$(printf 'k.cc\nk.h\nrun.py')"
out=$(pin_tracked_names "$d/work" "$d/src" notcheckedout); check "a pin that is not checked out is skipped" "$out" ""

# negative control: the inline form the scripts had, on the docs-only pin, fails under pipefail
old() {
    for pin in "$@"; do
        [ -d "$d/work/$pin" ] || continue
        git -C "$d/work/$pin" ls-files -- "$(sed -n 's/^path=//p' "$d/src/config/$pin.config")" 2>/dev/null
    done | grep -E '\.(c|cc|cpp|cxx|h|hpp|py)$' | xargs -r -n1 basename | LC_ALL=C sort -u
}
if (privsrc=$(old docsonly)); then rc=0; else rc=$?; fi
check "negative control: the old inline selector fails on an empty selection" "$([ $rc != 0 ] && echo fails || echo passes)" "fails"

[ $fails = 0 ] && echo "pin_names: ok" || { echo "pin_names: $fails failure(s)"; exit 1; }
