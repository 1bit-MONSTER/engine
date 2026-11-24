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
# scripts/lemond-onebit.sh: what the release package's launcher hands the lemond it starts. No GPU, no
# engine and no lemond run: `bin/lemond` is a stand-in that records its environment and argv, and the
# payload layout is the one the launcher is written for (bin/, 1bit/, lemond-onebit).
#
# engine#3399: Lemonade's stock llamacpp-hrx recipe (the *-HRX catalogue rows) must get the package's own
# HRX llama-server through LEMONADE_HRX_HRX_BIN. Left unset, lemond downloads upstream's hrx-b99 extra
# bundle and runs that, and on 1bit OS it registers no HRX0 (--list-devices -> "(none)"), so every -HRX
# model fails to load. The launcher is the one path every appliance start goes through (10lemonade runs
# it, and so does every package install), which is why the wiring is here and not in a config file.
#
# usage: tests/lemond_onebit.sh
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

# The payload layout: lemond-onebit beside bin/lemond and 1bit/{1bit,lemonade-hrx,hrx/llama/bin/llama-server}.
mkpkg() {  # <dir> <launcher>
    rm -rf "$1"
    mkdir -p "$1/bin" "$1/1bit/hrx/llama/bin"
    cp "$2" "$1/lemond-onebit"
    chmod +x "$1/lemond-onebit"
    for f in 1bit/1bit 1bit/lemonade-hrx 1bit/hrx/llama/bin/llama-server; do
        printf '#!/bin/sh\nexit 0\n' > "$1/$f"
        chmod +x "$1/$f"
    done
    cat > "$1/bin/lemond" <<'EOF'
#!/bin/sh
{ printf 'ARGV'; printf ' %s' "$@"; printf '\n'; env; } > "$LEMOND_RECORD"
EOF
    chmod +x "$1/bin/lemond"
}

run() {  # <payload launcher> [args...]
    rm -f "$scratch/rec"
    LEMOND_RECORD=$scratch/rec "$@" 2> "$scratch/err"
    rc=$?
}
val() { grep "^$1=" "$scratch/rec" 2>/dev/null | head -1 | cut -d= -f2-; }

pkg=$scratch/pkg
mkpkg "$pkg" "$here/../scripts/lemond-onebit.sh"
args=("$scratch/cache" "$scratch/config" --port 13305)
run "$pkg/lemond-onebit" "${args[@]}"
check "the launcher starts bin/lemond (exit $rc)" '[ "$rc" = 0 ] && [ -s "$scratch/rec" ]'
check "argv reaches lemond unchanged" '[ "$(head -1 "$scratch/rec")" = "ARGV ${args[*]}" ]'
check "LEMONADE_ONEBIT_BIN still defaults to the payload's engine" '[ "$(val LEMONADE_ONEBIT_BIN)" = "$pkg/1bit/1bit" ]'
check "the launcher still requires sealed NPU kernels" '[ "$(val ONEBIT_KCRYPT_REQUIRE_SEALED)" = 1 ]'

# engine#3399, the row this file exists for.
check "LEMONADE_HRX_HRX_BIN names the payload's lemonade-hrx" '[ "$(val LEMONADE_HRX_HRX_BIN)" = "$pkg/1bit/lemonade-hrx" ]'
check "the named path is the executable the payload ships" '[ -x "$(val LEMONADE_HRX_HRX_BIN)" ]'

# A caller's value wins, both the useful override and the deliberate opt-out; the launcher only sets a default.
LEMONADE_HRX_HRX_BIN=/custom/lemonade-hrx run "$pkg/lemond-onebit" "${args[@]}"
check "a caller's LEMONADE_HRX_HRX_BIN wins" '[ "$(val LEMONADE_HRX_HRX_BIN)" = /custom/lemonade-hrx ]'
LEMONADE_HRX_HRX_BIN=builtin run "$pkg/lemond-onebit" "${args[@]}"
check "LEMONADE_HRX_HRX_BIN=builtin keeps Lemonade's own download" '[ "$(val LEMONADE_HRX_HRX_BIN)" = builtin ]'

# Control: the launcher as it was before engine#3399 (this export removed) leaves the variable unset, which is
# the state every -HRX row failed in. The assertion above is the one that turns this into a FAIL.
old=$scratch/lemond-onebit-prefix
sed '/^export LEMONADE_HRX_HRX_BIN=/d' "$here/../scripts/lemond-onebit.sh" > "$old"
chmod +x "$old"
mkpkg "$scratch/pkgold" "$old"
run "$scratch/pkgold/lemond-onebit" "${args[@]}"
check "control: the pre-fix launcher hands lemond no LEMONADE_HRX_HRX_BIN" '[ -z "$(val LEMONADE_HRX_HRX_BIN)" ]'

[ "$fail" = 0 ] || { echo "FAILED: tests/lemond_onebit.sh"; exit 1; }
echo "PASS: tests/lemond_onebit.sh"
