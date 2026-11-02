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
# spec-insert-rows.sh <DISTRO_PKGS_SPECS file> <rows file>
#
# Insert the package rows of <rows file> as the LAST rows of woof-CE PKGS_SPECS_TABLE - i.e. immediately
# before the table's closing quote - because 0setup reads the table as one shell string and never sees
# anything appended after it. A row whose generic name (field 2) is already in the table is skipped, so
# running this twice, or over a woof-CE that grows the row later, is a no-op rather than a duplicate.
#
# It lives in its own file because build.sh runs inside `bash -c '...'`: an awk program in there would put
# single quotes inside a single-quoted string, which cost an edit round (measured 2026-11-02, twice).
#
# Exit 0 on success (also when nothing needed adding), 1 on a missing file or a spec with no table.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: spec-insert-rows.sh <DISTRO_PKGS_SPECS file> <rows file>" >&2; exit 1; }
spec=$1 rows=$2
[ -f "$spec" ] || { echo "spec-insert-rows.sh: no such spec file: $spec" >&2; exit 1; }
[ -f "$rows" ] || { echo "spec-insert-rows.sh: no such rows file: $rows" >&2; exit 1; }
tmp=$(mktemp) || exit 1
cnt=$(mktemp) || exit 1
trap 'rm -f "$tmp" "$cnt"' EXIT

awk -v rows="$rows" -v cnt="$cnt" '
    BEGIN { q = sprintf("%c", 39); FS = "|" }
    { lines[NR] = $0; if ($1 == "yes") have[$2] = 1; if ($0 == q) qline = NR }
    END {
        if (!qline) { print "no PKGS_SPECS_TABLE closing quote in this file" > "/dev/stderr"; exit 3 }
        n = 0
        while ((getline line < rows) > 0) {
            split(line, f, "|")
            if (f[2] in have) continue
            new[++n] = line; have[f[2]] = 1
        }
        for (i = 1; i <= NR; i++) {
            if (i == qline) for (j = 1; j <= n; j++) print new[j]
            print lines[i]
        }
        print n > cnt
    }
' "$spec" > "$tmp"
[ -s "$tmp" ] || { echo "spec-insert-rows.sh: the rewrite came out empty, not touching $spec" >&2; exit 1; }
cat "$tmp" > "$spec"
echo "   $(cat "$cnt") row(s) into $(basename "$spec")"
