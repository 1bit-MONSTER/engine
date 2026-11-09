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
# build_static_provenance_test.sh - `build-static.sh` must never print a quotable artifact hash for a tree that is
# not a commit (engine#1326, from engine#1323's mis-quoted hash: the number came from the tree *before* `cargo fmt`,
# and a 24-byte difference moved it - rustc embeds file:line in panic messages, so formatting alone moves the bytes).
#
# The guard is three lines of the script and it is what every cycle hand-off relies on, so it gets a test: the whole
# toolchain is stubbed (cargo, rustup, readelf, ldd and a fake binary), because the property under test is the
# provenance line, not the compile. Two directions, because either alone can pass for the wrong reason:
#
#   clean tree -> `ok <hash> <bytes> bytes (HEAD <sha>, tree clean)` and exit 0
#   dirty tree -> `DIRTY <hash>-dirty <bytes> bytes - NOT <sha>'s artifact: N uncommitted path(s)` and exit 1
#
# A fix that made the dirty case green by weakening the check would pass the first arm; one that refused
# unconditionally would pass the second. Both are asserted here.
set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
script=$root/os/v3/puppy/console-tui/build-static.sh
[ -f "$script" ] || { echo "build_static_provenance_test: no $script"; exit 1; }
command -v git > /dev/null || { echo "build_static_provenance_test: skip (no git)"; exit 77; }

work=$(mktemp -d /tmp/bsp.XXXXXX) || exit 1
trap 'rm -rf "$work"' EXIT
fails=0
check() {   # <what> <got> <want>
    if [ "$2" = "$3" ]; then echo "ok   $1 = $3"; else echo "FAIL $1 = $2 (wanted $3)"; fails=$((fails + 1)); fi
}

# ---- the stubbed toolchain -----------------------------------------------------------------------------------------
mkdir -p "$work/bin"
cat > "$work/bin/cargo" <<'STUB'
#!/bin/sh
# `cargo test` / `cargo build --release --target T` both succeed and leave a fake artifact where build-static looks.
for a in "$@"; do :; done
mkdir -p target/x86_64-unknown-linux-musl/release
printf '#!/bin/sh\nexit 0\n' > target/x86_64-unknown-linux-musl/release/1bit-console-tui
chmod +x target/x86_64-unknown-linux-musl/release/1bit-console-tui
exit 0
STUB
printf '#!/bin/sh\necho x86_64-unknown-linux-musl\n' > "$work/bin/rustup"
printf '#!/bin/sh\nexit 0\n' > "$work/bin/readelf"          # no INTERP -> "static"
printf '#!/bin/sh\necho "statically linked"\n' > "$work/bin/ldd"   # no "=>" -> no shared deps
chmod +x "$work/bin"/*

# ---- a throwaway repo holding a copy of the script, so `git status -- .` means this crate ---------------------------
crate="$work/repo/os/v3/puppy/console-tui"
mkdir -p "$crate"
cp "$script" "$crate/build-static.sh"
chmod +x "$crate/build-static.sh"
# The crate's own .gitignore is part of the property: `target/` must not read as an uncommitted path, or the guard
# would mark every build dirty (and a real build always writes there).
cp "$root/os/v3/puppy/console-tui/.gitignore" "$crate/.gitignore"
git -C "$work/repo" init -q
git -C "$work/repo" -c user.email=t@t -c user.name=t add -A
git -C "$work/repo" -c user.email=t@t -c user.name=t commit -qm "fixture"
sha=$(git -C "$work/repo" rev-parse --short HEAD)

run() { PATH="$work/bin:$PATH" "$crate/build-static.sh" 2> "$work/err" > "$work/out"; echo $?; }

# ---- arm 1: clean tree -> a hash that names the commit ------------------------------------------------------------
rc=$(run)
check "clean tree: exit" "$rc" "0"
grep -qE "^build-static: ok [0-9a-f]{16} [0-9]+ bytes \(HEAD $sha, tree clean\)$" "$work/out" &&
    echo "ok   clean tree: the line names HEAD $sha and says the tree is clean" ||
    { echo "FAIL clean tree: '$(tail -1 "$work/out")'"; fails=$((fails + 1)); }

# ---- arm 2: one uncommitted path -> the hash is marked and the exit is non-zero -------------------------------------
printf 'dirty\n' > "$crate/scratch.txt"
rc=$(run)
check "dirty tree: exit" "$rc" "1"
grep -qE "^build-static: DIRTY [0-9a-f]{16}-dirty [0-9]+ bytes - NOT $sha's artifact: 1 uncommitted path\(s\)" "$work/err" &&
    echo "ok   dirty tree: the hash is marked -dirty, names the commit it is NOT, and counts the path" ||
    { echo "FAIL dirty tree: '$(cat "$work/err")'"; fails=$((fails + 1)); }
grep -q "build-static: ok" "$work/err" "$work/out" &&
    { echo "FAIL dirty tree: a plain 'ok' line was still printed"; fails=$((fails + 1)); } ||
    echo "ok   dirty tree: no plain 'ok' hash is offered"

# ---- arm 3: two paths -> the count is the number of paths, not the number of lines of one ---------------------------
printf 'dirty\n' > "$crate/scratch2.txt"
rc=$(run)
grep -qE "NOT $sha's artifact: 2 uncommitted path\(s\)" "$work/err" &&
    echo "ok   two uncommitted paths are counted as 2" ||
    { echo "FAIL two paths: '$(cat "$work/err")'"; fails=$((fails + 1)); }

# ---- arm 4: the tree goes clean again -> the quotable line comes back ----------------------------------------------
rm -f "$crate/scratch.txt" "$crate/scratch2.txt"
rc=$(run)
check "clean again: exit" "$rc" "0"
grep -qE "^build-static: ok [0-9a-f]{16} [0-9]+ bytes \(HEAD $sha, tree clean\)$" "$work/out" &&
    echo "ok   clean again: the quotable line returns" ||
    { echo "FAIL clean again: '$(tail -1 "$work/out")'"; fails=$((fails + 1)); }

echo "build_static_provenance: $fails failure(s)"
[ "$fails" -eq 0 ] || exit 1
exit 0
