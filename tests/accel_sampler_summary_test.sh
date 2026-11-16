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
# A reader of the sampler's summary must never see a truncated one (engine#2778).
#
# `summary()` used to end `} > "$out"`, which truncates in place. Since #2613 it is rewritten every ~15 minutes, so a
# 24 h capture exposes 96 windows in which a reader -- accel_attribution.py, or a lane grepping the summary at the
# moment of a fault -- sees a partial file whose header line is already written: a zero that reads as a datum.
#
# This drives the real `summary()` in a subshell with a synthetic sample file and a background reader, and asserts
# every observation is one of the two complete sizes. Hermetic: temp dir, no device, no sampler.
#
# usage: bash tests/accel_sampler_summary_test.sh [tools/diag/accel_sampler.sh]
#   rc 0 all arms pass, 1 an arm failed
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
sampler=${1:-$here/../tools/diag/accel_sampler.sh}
[ -f "$sampler" ] || { echo "accel_sampler_summary_test: no sampler at $sampler" >&2; exit 2; }
bash -n "$sampler" || { echo "accel_sampler_summary_test: $sampler does not parse" >&2; exit 2; }

fails=0
ok()   { echo "ok   $1"; }
bad()  { echo "FAIL $1"; fails=$((fails + 1)); }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# The two sizes a reader may legitimately see: "before any summary" (absent) and one of the complete writes. Drive the
# real function by sourcing only its definition, not the sampler's argument parsing.
fn=$(sed -n '/^summary() {/,/^}/p' "$sampler")
[ -n "$fn" ] || { echo "accel_sampler_summary_test: summary() not found in $sampler" >&2; exit 2; }

# two sample files that produce visibly different summary sizes
printf '1\tpidA\tcmd\t-\tscope\tticket\n' > "$tmp/a.samp"
printf '1\tpidA\tcmd\t-\tscope\tticket\n2\tpidB\tcmd\t-\tscope\tticket\n3\tpidC\tcmd\t-\tscope\tticket\n' > "$tmp/b.samp"

run_summary() {  # $1 = sample file
  DIR=$tmp SAMP=$1 ATTR=$tmp/attr.txt bash -c "$(declare -f); $fn; summary" >/dev/null 2>&1
}

run_summary "$tmp/a.samp"
small=$(wc -c < "$tmp/summary.txt")
run_summary "$tmp/b.samp"
large=$(wc -c < "$tmp/summary.txt")
[ "$small" -gt 0 ] && [ "$large" -gt 0 ] && [ "$small" != "$large" ] \
  && ok "the two synthetic summaries differ in size ($small vs $large bytes) -- the reader can tell them apart" \
  || bad "the two synthetic summaries should differ in size ($small vs $large)"

# The arm: a reader racing the writer must only ever see 0 (absent/empty before the first write), small, or large.
sizes=$tmp/sizes.txt
: > "$sizes"
( for _ in $(seq 1 4000); do wc -c < "$tmp/summary.txt" 2>/dev/null || echo 0; done >> "$sizes" ) &
reader=$!
for i in $(seq 1 40); do
  run_summary "$( [ $((i % 2)) -eq 0 ] && echo "$tmp/a.samp" || echo "$tmp/b.samp")"
done
wait "$reader" 2>/dev/null
bad_obs=$(awk -v s="$small" -v l="$large" '$1 != s && $1 != l' "$sizes" | wc -l)
[ "$bad_obs" -eq 0 ] \
  && ok "every one of $(wc -l < "$sizes") reader observations was a complete summary (${small} or ${large} bytes)" \
  || bad "$bad_obs observation(s) saw a size that is neither complete summary: $(awk -v s="$small" -v l="$large" '$1 != s && $1 != l' "$sizes" | sort -u | head -3 | tr '\n' ' ')"

# and the property the fix is: no in-place truncation left in summary()
if sed -n '/^summary() {/,/^}/p' "$sampler" | grep -qE '^\s*\} > "\$\{?out\}?"\s*$'; then
  bad "summary() still writes with an in-place truncation (} > \"\$out\")"
else
  ok "summary() writes through a temp file and renames"
fi

if [ "$fails" -eq 0 ]; then echo "accel_sampler_summary_test: pass"; exit 0; fi
echo "accel_sampler_summary_test: $fails failure(s)"; exit 1
