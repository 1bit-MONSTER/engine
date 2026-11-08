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
# scripts/code-objects.sh counts the code objects bundled into a fat library (engine#1443: the packers used to
# grep for a string that real libraries do not contain, so every real drop was refused). The counter has to be
# exact and binary-safe, which is what this pins: a bundle header on a line of its own, several on one line,
# one split by a NUL, and the word "ELF" in plain text that must not count.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
count=$here/../scripts/code-objects.sh
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
ELF=$(printf '\177ELF')

: > "$scratch/empty"
check "an empty file has no code objects" '[ "$("$count" "$scratch/empty")" = 0 ]'

printf 'not a bundle here\n' > "$scratch/text"
check "text with no bundle header counts 0" '[ "$("$count" "$scratch/text")" = 0 ]'

printf 'ELF is only a word here, three times: ELF ELF\n' > "$scratch/word"
check "the word ELF in text does not count" '[ "$("$count" "$scratch/word")" = 0 ]'

printf '%sdata\n' "$ELF" > "$scratch/one"
check "one on its own line counts 1" '[ "$("$count" "$scratch/one")" = 1 ]'

printf '%sA%sB%sC' "$ELF" "$ELF" "$ELF" > "$scratch/three-one-line"
check "three on one line count 3 (matches, not lines)" '[ "$("$count" "$scratch/three-one-line")" = 3 ]'

printf 'junk\0%s\0junk\0%s\0' "$ELF" "$ELF" > "$scratch/nul"
check "two split by NUL bytes count 2" '[ "$("$count" "$scratch/nul")" = 2 ]'

# a real bundle header is 0x7f 'E' 'L' 'F' immediately followed by the class/version bytes
printf '%s\2\1\1\0' "$ELF" > "$scratch/real"
check "a header with its class bytes counts 1" '[ "$("$count" "$scratch/real")" = 1 ]'

missing=$("$count" "$scratch/nope" 2>/dev/null); missing_rc=$?
check "a missing file counts 0 and exits non-zero" '[ "$missing" = 0 ] && [ "$missing_rc" = 1 ]'

if [ $fail -ne 0 ]; then echo FAIL; exit 1; fi
echo PASS
