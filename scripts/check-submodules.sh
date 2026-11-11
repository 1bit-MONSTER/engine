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
# check-submodules.sh [--root DIR] [--self-test]
#
# engine#2129: every submodule's CHECKOUT must be the commit the repository pins. CI checks the pins themselves ("submodule
# pins only move forward", "hrx-system gitlink agrees with the hold file") and that is a check on the *repository*; nothing
# checks the tree on the box, which is where a build actually reads its sources.
#
# Measured on this box 2026-11-11 while triaging #2129 -- 7 of 7 disagreed with their pin:
#
#   third_party/hrx-system   pin 7f4236924  head 51b1739ae  dirty 0
#   third_party/llama.cpp    pin 2293f95ea  head c16424f56  dirty 301     <- the issue's "301 files short of its own HEAD"
#   third_party/xdna-driver  pin 526c60819  head 5d302c998  dirty 32
#   third_party/laya         pin b8a0f0991  head 593a1596f  dirty 4
#   third_party/tokenizers   pin 301277072  head 88a4498ad  dirty 20
#   third_party/lemonade     pin 248f818b4  head 593a1596f  dirty 4
#   third_party/ggmlc        pin df08d1b06  head 593a1596f  dirty 4
#
# A build from that tree does not build the pinned code, and nothing says so: the engine reads a handful of submodule
# files directly (CMakeLists.txt's jinja probe is one), so "the pin is healthy" and "the checkout is the pin" are two
# different claims. This is the second one, and it is cheap enough to run before every build.
#
# `dirty` matters even when HEAD matches: a checkout whose worktree differs from its HEAD builds sources the pin does not
# name, which is the same defect with a smaller diff. It is reported per submodule with its count so a reader can tell a
# stray file from a reset (`git -C <path> checkout -- . && git -C <path> clean -fd`).
#
# Exit: 0 when every checkout is its pin and clean; 1 on any disagreement; 2 when there is nothing to inspect (no
# .gitmodules, or git missing) -- an unchecked tree is the state this exists to end.
set -uo pipefail

root=.
self=0
while [ $# -gt 0 ]; do
    case $1 in
        --root) root=${2:?--root needs a directory}; shift 2 ;;
        --root=*) root=${1#*=}; shift ;;
        --self-test) self=1; shift ;;
        -h|--help) sed -n '16,36p' "$0"; exit 0 ;;
        *) echo "check-submodules.sh: unknown option $1" >&2; exit 2 ;;
    esac
done

if [ "$self" = 1 ]; then
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    fails=0
    ok() { printf '  ok   %s\n' "$1"; }
    bad() { printf '  FAIL %s\n' "$1"; fails=$((fails + 1)); }

    # A real submodule, built offline: two local repos, protocol.file.allow so `submodule add` accepts a path.
    git init -q "$tmp/sub"
    ( cd "$tmp/sub" && echo a > a.txt && git add a.txt \
        && git -c user.name=t -c user.email=t@t commit -qm first ) || { echo "fixture failed" >&2; exit 2; }
    git init -q "$tmp/repo"
    ( cd "$tmp/repo" && echo top > top.txt && git add top.txt \
        && git -c user.name=t -c user.email=t@t commit -qm top \
        && git -c protocol.file.allow=always -c user.name=t -c user.email=t@t \
             submodule add -q "$tmp/sub" third_party/sub >/dev/null 2>&1 \
        && git -c user.name=t -c user.email=t@t commit -qm pin ) || { echo "fixture failed" >&2; exit 2; }

    out=$(bash "$0" --root "$tmp/repo" 2>&1)
    [ $? = 0 ] && ok "a checkout at its pin and clean passes" || bad "a clean checkout at its pin should pass: $out"

    # HEAD moves away from the pin: the engine#2129 shape.
    ( cd "$tmp/repo/third_party/sub" && echo b > b.txt && git add b.txt \
        && git -c user.name=t -c user.email=t@t commit -qm second ) >/dev/null 2>&1
    out=$(bash "$0" --root "$tmp/repo" 2>&1)
    [ $? = 1 ] && ok "a checkout ahead of its pin fails" || bad "a checkout ahead of its pin should fail"
    printf '%s' "$out" | grep -q 'is not the pin' && ok "and the message says which submodule and both commits" \
        || bad "the message does not name the disagreement: $out"

    # HEAD back at the pin, worktree dirty: same defect, smaller diff.
    ( cd "$tmp/repo/third_party/sub" && git reset -q --hard HEAD~1 && echo stray > stray.txt ) >/dev/null 2>&1
    out=$(bash "$0" --root "$tmp/repo" 2>&1)
    [ $? = 1 ] && ok "a dirty checkout fails even when HEAD is the pin" || bad "a dirty checkout should fail"
    printf '%s' "$out" | grep -q 'clean' && ok "and it says the worktree differs from its HEAD" \
        || bad "the dirty message does not explain itself: $out"

    # No .gitmodules: cannot tell, not "clean".
    mkdir -p "$tmp/plain" && git init -q "$tmp/plain"
    bash "$0" --root "$tmp/plain" >/dev/null 2>&1
    [ $? = 2 ] && ok "no .gitmodules is 'cannot tell' (2), not a pass" || bad "no .gitmodules should exit 2"

    if [ "$fails" != 0 ]; then echo "check_submodules self-test: $fails failure(s)"; exit 1; fi
    echo "check_submodules self-test: ok"
    exit 0
fi

command -v git >/dev/null || { echo "check-submodules: no git on PATH" >&2; exit 2; }
[ -f "$root/.gitmodules" ] || { echo "check-submodules: no .gitmodules under $root -- nothing to inspect, which is not a pass" >&2; exit 2; }

bad=0
while read -r _key path; do
    [ -n "${path:-}" ] || continue
    pin=$(git -C "$root" ls-tree HEAD -- "$path" 2>/dev/null | awk '{print $3}')
    if [ -z "$pin" ]; then
        echo "::error::check-submodules: $path is in .gitmodules but the tree pins no commit for it"
        bad=$((bad + 1)); continue
    fi
    head=$(git -C "$root/$path" rev-parse HEAD 2>/dev/null)
    if [ -z "$head" ]; then
        echo "::error::check-submodules: $path is not checked out (the tree pins ${pin:0:9}); a build would not find its sources"
        bad=$((bad + 1)); continue
    fi
    if [ "${pin:0:9}" != "${head:0:9}" ]; then
        echo "::error::check-submodules: $path is not the pin: tree ${pin:0:9}, checkout ${head:0:9} -- the build would read sources the pin does not name"
        bad=$((bad + 1)); continue
    fi
    n=$(git -C "$root/$path" status --porcelain 2>/dev/null | wc -l)
    if [ "$n" != 0 ]; then
        echo "::error::check-submodules: $path is at the pin but its worktree is not clean ($n path(s) differ from HEAD) -- a build would read sources the pin does not name"
        bad=$((bad + 1)); continue
    fi
    printf 'ok   %-28s %s\n' "$path" "${pin:0:9}"
done < <(git -C "$root" config -f .gitmodules --get-regexp '^submodule\..*\.path$' 2>/dev/null)

[ "$bad" = 0 ] && { echo "check-submodules: every checkout is its pin and clean"; exit 0; }
echo "check-submodules: $bad submodule(s) are not what the tree pins"
exit 1
