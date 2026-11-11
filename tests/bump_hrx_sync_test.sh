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
#
# scripts/bump-hrx-sync.sh, the part that says WHY a commit of our AMD mirror is on no AMD branch
# (engine#1945). Two answers, and they need opposite actions:
#   rebased  AMD rebased its line and dropped the old one; the commit is AMD's and its twin is in the pin
#   foreign  the commit is not AMD's and has to leave the mirror
# Local repositories only: no network, no GPU, no token.
set -euo pipefail

script=${1:?usage: bump_hrx_sync_test.sh <scripts/bump-hrx-sync.sh>}
script=$(cd "$(dirname "$script")" && pwd)/$(basename "$script")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
fails=0
ok()   { echo "ok   $*"; }
fail() { echo "FAIL $*"; fails=$((fails + 1)); }
expect() {  # expect <what> <actual> <wanted>
  if [ "$2" = "$3" ]; then ok "$1"; else fail "$1"; printf '     got:    %s\n     wanted: %s\n' "$2" "$3"; fi
}

export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null
repo=$work/r
git init -q "$repo"
g() { git -C "$repo" "$@"; }
# commit <author name> <author email> <author date> <committer date> <subject>
commit() {
  echo "$5 $RANDOM$RANDOM" >> "$repo/f"
  g add f
  GIT_AUTHOR_NAME=$1 GIT_AUTHOR_EMAIL=$2 GIT_AUTHOR_DATE=$3 \
    GIT_COMMITTER_NAME=c GIT_COMMITTER_EMAIL=c@example.invalid GIT_COMMITTER_DATE=$4 \
    git -C "$repo" commit -q -m "$5"
}
amd='amd dev'; amd_mail=dev@amd.example.invalid

# The old line: upstream base, then two AMD commits. The mirror is its tip.
commit up up@example.invalid 2026-01-01T00:00:00Z 2026-01-01T00:00:00Z "upstream base"
base=$(g rev-parse HEAD)
commit "$amd" "$amd_mail" 2026-08-17T03:02:45Z 2026-08-17T03:09:08Z "hrx: add the backend"
commit "$amd" "$amd_mail" 2026-10-28T03:17:36Z 2026-10-28T03:17:36Z "hrx: fix concat bounds (#155)"
mirror=$(g rev-parse HEAD)
old1=$(g rev-parse --short HEAD~1); old2=$(g rev-parse --short HEAD)

# The new line: a newer upstream base, the same two commits re-committed (same author, same author date,
# same subject, a later committer date -- what a rebase leaves), then one new commit. The pin is its tip.
g checkout -q -b newline "$base"
commit up up@example.invalid 2026-10-27T00:00:00Z 2026-10-27T00:00:00Z "upstream moved on"
commit "$amd" "$amd_mail" 2026-08-17T03:02:45Z 2026-10-28T09:00:00Z "hrx: add the backend"
new1=$(g rev-parse --short HEAD)
commit "$amd" "$amd_mail" 2026-10-28T03:17:36Z 2026-10-28T09:00:01Z "hrx: fix concat bounds (#155)"
new2=$(g rev-parse --short HEAD)
commit "$amd" "$amd_mail" 2026-11-09T15:29:43Z 2026-11-09T15:29:43Z "hrx: something new"
pin=$(g rev-parse HEAD)
g update-ref refs/amd/main "$pin"   # AMD's only branch: the old line is on none

run() { bash "$script" off-branch "$repo" "$1" "$2"; }

got=$(run "$mirror" "$pin")
expect "a rebased line: both old commits are named with their twins in the pin, newest first" "$got" \
"rebased $old2 $new2 hrx: fix concat bounds (#155)
rebased $old1 $new1 hrx: add the backend"
if grep -q '^foreign ' <<< "$got"; then fail "a rebased AMD commit is not called foreign"; else ok "a rebased AMD commit is not called foreign"; fi

# While AMD still has the old line on a branch, nothing is off-branch at all.
g update-ref refs/amd/backup "$mirror"
expect "an old line AMD still keeps on a branch reports nothing" "$(run "$mirror" "$pin")" ""
g update-ref -d refs/amd/backup

# A commit of ours on top of the mirror: no twin in the pin, so it is foreign, and the AMD ones stay rebased.
g checkout -q --detach "$mirror"
commit "a lane" lane@example.invalid 2026-11-01T00:00:00Z 2026-11-01T00:00:00Z "hrx: a fix of ours"
ours=$(g rev-parse HEAD); ours_s=$(g rev-parse --short HEAD)
got=$(run "$ours" "$pin")
expect "a commit that is not AMD's is foreign, beside the rebased ones" "$got" \
"foreign $ours_s hrx: a fix of ours
rebased $old2 $new2 hrx: fix concat bounds (#155)
rebased $old1 $new1 hrx: add the backend"

# The subject alone is not identity: the same subject from another author, or at another author date, is foreign.
g checkout -q --detach "$mirror"
commit "a lane" lane@example.invalid 2026-11-09T15:29:43Z 2026-11-09T15:29:43Z "hrx: something new"
other=$(g rev-parse HEAD)
expect "the same subject from another author is foreign" "$(run "$other" "$pin" | head -n1)" \
  "foreign $(g rev-parse --short "$other") hrx: something new"
g checkout -q --detach "$mirror"
commit "$amd" "$amd_mail" 2026-11-09T15:29:44Z 2026-11-09T15:29:44Z "hrx: something new"
other=$(g rev-parse HEAD)
expect "the same subject and author at another author date is foreign" "$(run "$other" "$pin" | head -n1)" \
  "foreign $(g rev-parse --short "$other") hrx: something new"

# ---- the mirror hold (config/llama-mirror-hold.config): what a named commit means for a run -------------------
holdf=$work/hold.config
hold() { LLAMA_HOLD_FILE=$holdf bash "$script" llama-hold "$repo" "$1" "$2"; }
# a pin that descends from the mirror: build one
g checkout -q --detach "$mirror"
commit "$amd" "$amd_mail" 2026-11-10T00:00:00Z 2026-11-10T00:00:00Z "hrx: one more on the old line"
desc=$(g rev-parse HEAD)

rm -f "$holdf"
expect "no hold file: the sync runs as usual" "$(hold "$mirror" "$pin")" "none"
printf '# a comment\n#   %s\n' "$mirror" > "$holdf"
expect "a commit that is only mentioned in a comment holds nothing" "$(hold "$mirror" "$pin")" "none"
printf '# held\n%s\n' "$mirror" > "$holdf"
expect "the mirror is the held commit and AMD's pin is on a rewritten line: held" "$(hold "$mirror" "$pin")" "held"
expect "the mirror has moved away from the held commit: the file is stale" "$(hold "$ours" "$pin")" "stale"
expect "AMD's pin contains the held commit again: the hold is not needed" "$(hold "$mirror" "$desc")" "unneeded"
expect "the pin IS the held commit: not needed either" "$(hold "$mirror" "$mirror")" "unneeded"
printf 'not-a-sha\n%s extra words\n' "$mirror" > "$holdf"
expect "the commit line may carry a trailing note; a line that is not a commit is ignored" "$(hold "$mirror" "$pin")" "held"
printf '%s\n' "${mirror:0:12}" > "$holdf"
expect "a short hash is not a hold: forty hex digits or nothing" "$(hold "$mirror" "$pin")" "none"

# The file shipped in the tree: whatever it names must be one full commit hash or nothing, and what it means for a
# run is decided by the same function as above (a hold on a commit the mirror has left is stale, and so on).
shipped=$(cd "$(dirname "$script")/.." && pwd)/config/llama-mirror-hold.config
if [ -f "$shipped" ]; then
  n=$(grep -v '^[[:space:]]*#' "$shipped" | grep -cE '^[0-9a-f]{40}' || true)
  if [ "$n" -le 1 ]; then ok "the shipped hold file names at most one commit"; else fail "the shipped hold file names at most one commit (found $n)"; fi
  junk=$(grep -v '^[[:space:]]*#' "$shipped" | grep -v '^[[:space:]]*$' | grep -vcE '^[0-9a-f]{40}([[:space:]].*)?$' || true)
  expect "every line of the shipped hold file is a comment, blank, or a full commit hash" "$junk" "0"
else
  fail "config/llama-mirror-hold.config is in the tree"
fi

# ---- which branch is the mirror (config/llama-mirror-branch.config) -----------------------------------------
mb=$work/mirror.config
branch() { LLAMA_MIRROR_FILE=$mb bash "$script" mirror-branch 2>&1; }
rm -f "$mb"
expect "no file: the mirror is 1bit/hrx-vulkan, as it always was" "$(branch)" "1bit/hrx-vulkan"
printf '# only comments\n\n#   1bit/hrx-vulkan-20261110\n' > "$mb"
expect "a branch that is only mentioned in a comment changes nothing" "$(branch)" "1bit/hrx-vulkan"
printf '# the dated mirror\n\n  1bit/hrx-vulkan-20261110  \n1bit/ignored-second-line\n' > "$mb"
expect "the first line that is not a comment names the mirror, spaces around it ignored" "$(branch)" "1bit/hrx-vulkan-20261110"
for bad in '1bit/hrx-vulkan-patched' 'refs/heads/1bit/x' '../x' '-x' 'a b' 'x;rm' 'x:y'; do
  printf '%s\n' "$bad" > "$mb"
  if out=$(branch); then fail "a mirror named '$bad' is refused (got: $out)"; else
    case "$out" in *"$mb"*|*llama-mirror*|*mirror*) ok "a mirror named '$bad' is refused, naming the file" ;; *) fail "the refusal of '$bad' names the file (got: $out)" ;; esac
  fi
done
shipped_mb=$(cd "$(dirname "$script")/.." && pwd)/config/llama-mirror-branch.config
if [ -f "$shipped_mb" ]; then
  expect "the shipped file names no branch, so the mirror is unchanged" "$(LLAMA_MIRROR_FILE=$shipped_mb bash "$script" mirror-branch)" "1bit/hrx-vulkan"
else
  fail "config/llama-mirror-branch.config is in the tree"
fi

if [ "$fails" -ne 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "bump_hrx_sync: ok"
