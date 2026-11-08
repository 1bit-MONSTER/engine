#!/usr/bin/env bash
# Copyright 2026 The 1bit-MONSTER Authors.
# SPDX-License-Identifier: Apache-2.0
# The installed-RUNPATH audit (engine#1411) on fixtures it must pass and must fail.
#   $ORIGIN only                    -> PASS, rc 0
#   a build tree in the RUNPATH     -> FAIL, rc 1, the path named
#   $ORIGIN plus a toolchain dir    -> FAIL, rc 1   (the c22 shape: dead weight, still not $ORIGIN only)
#   RPATH instead of RUNPATH        -> FAIL, rc 1
#   a missing root                  -> rc 2, not a silent PASS
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
tool=$root/tools/check-installed-rpath.py
command -v patchelf >/dev/null || { echo "SKIP check-installed-rpath: no patchelf"; exit 0; }
command -v readelf  >/dev/null || { echo "SKIP check-installed-rpath: no readelf"; exit 0; }
[ -x /bin/true ] || { echo "SKIP check-installed-rpath: no /bin/true fixture"; exit 0; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf 'ok   %s\n' "$1"
    else printf 'FAIL %s (expected %s, got %s)\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
}

mk() { # dir runpath [--force-rpath]
    mkdir -p "$work/$1/bin"
    cp /bin/true "$work/$1/bin/onebit"
    patchelf --set-rpath "$2" ${3:-} "$work/$1/bin/onebit"
}

mk good '$ORIGIN'
out=$("$tool" "$work/good" 2>&1) && rc=0 || rc=$?
check "origin-only passes" 0 "$rc"
case $out in *PASS*) ;; *) echo "FAIL origin-only says: $out"; fails=$((fails + 1)) ;; esac

mk build '/home/x/wt/release-rc/work/build/hrx/llama/bin:/home/x/wt/build/lib'
out=$("$tool" "$work/build" 2>&1) && rc=0 || rc=$?
check "a build tree fails" 1 "$rc"
case $out in *release-rc/work/build*) ;; *) echo "FAIL build-tree run did not name the path: $out"; fails=$((fails + 1)) ;; esac

mk mixed '$ORIGIN:/work-c22/hrx-toolchain/lib/llvm/lib/clang/24/lib/x86_64-unknown-linux-gnu'
out=$("$tool" "$work/mixed" 2>&1) && rc=0 || rc=$?
check "origin plus a toolchain dir fails" 1 "$rc"

mk rpath '$ORIGIN' --force-rpath
out=$("$tool" "$work/rpath" 2>&1) && rc=0 || rc=$?
check "RPATH rather than RUNPATH fails" 1 "$rc"

out=$("$tool" "$work/does-not-exist" 2>&1) && rc=0 || rc=$?
check "a missing root is rc 2, not a pass" 2 "$rc"

mk subdir '$ORIGIN/xrt/lib:$ORIGIN/../lib'
out=$("$tool" "$work/subdir" 2>&1) && rc=0 || rc=$?
check "a directory under \$ORIGIN passes" 0 "$rc"

# no RUNPATH at all is not a failure: plenty of installed programs need nothing beside them.
mkdir -p "$work/plain/bin" && cp /bin/true "$work/plain/bin/onebit"
out=$("$tool" "$work/plain" 2>&1) && rc=0 || rc=$?
check "a binary with no RUNPATH passes" 0 "$rc"

[ "$fails" = 0 ] && { echo "PASS check-installed-rpath: all cases"; exit 0; }
echo "FAIL check-installed-rpath: $fails case(s)"; exit 1
