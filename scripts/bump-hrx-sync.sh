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
# The fork-side half of .github/workflows/bump-hrx.yml, kept in a script so it can be run
# locally without pushing anything:
#
#   DRY_RUN=1 AMD_LLAMA=<sha> OURS_LLAMA=<sha> OURS_HRX=<sha> scripts/bump-hrx-sync.sh llama
#   DRY_RUN=1 AMD_HRX=<sha> OURS_HRX=<sha> scripts/bump-hrx-sync.sh hrx
#
# Run it from the root of an engine checkout; it works in ./patched and ./hrxsystem.
# Results go to $GITHUB_OUTPUT and $GITHUB_STEP_SUMMARY when set, and are printed either way.
#
# llama: moves 1bit-MONSTER/llama.cpp 1bit/hrx-vulkan (our mirror of AMD's pin) to AMD's pin
#        and merges AMD's pin into 1bit/hrx-vulkan-patched (our line). Two cases the old
#        inline step could not handle:
#        - AMD rewrote its history (2026-10-10: AMD-Ecosystem/llama.cpp squashed 141 commits
#          into 53d443cd0, whose tree equals their old tip 8108322b7). The mirror then is not
#          an ancestor of the new pin. If every commit on the mirror is still on an AMD branch,
#          the old tip is tagged and the mirror moves; the merge treats the squash commit as a
#          child of its pre-squash equivalent, so only AMD's genuinely new changes are merged.
#          A mirror commit that is on no AMD branch fails the run and is named.
#        - hrx-system is held (config/hrx-system-hold.config). AMD's ggml-hrx is written against
#          AMD's hrx-system (e.g. #139 moved the kernel corpus to the `multiple_of` predicate that
#          our held hrx-system cannot parse), so only AMD's changes OUTSIDE the HRX backend paths
#          are taken, as an ordinary commit on our line. AMD is not recorded as a merge parent,
#          so its HRX changes are still merged in full once the hold is lifted.
# hrx:   merges AMD's hrx-system pin into 1bit-MONSTER/hrx-system 1bit/main, unless the hold
#        file names a commit, in which case it reports the hold and moves nothing.
#
# Conflicts that cannot be resolved by the rules above fail the run and name the files.
set -euo pipefail

cmd=${1:-}
DRY_RUN=${DRY_RUN:-0}
HOLD_FILE=${HOLD_FILE:-config/hrx-system-hold.config}
ENGINE_DIR=$(pwd)
# The HRX backend paths of the llama.cpp tree (what AMD's hrx-graph line adds on top of ggml).
# tests/CMakeLists.txt belongs here: it is where the HRX test targets are registered, so it moves
# with them. AMD renamed the four HRX tests (test-hrx-buffer -> hrx-buffer-test, ...) in its
# 2026-10-28 pin c1e1bcbf6839; the .cpp files are already covered by 'tests/*hrx*', but the
# registration block is in this one shared file, so leaving it out merges AMD's rename into our
# line and conflicts there (engine#628). While hrx-system is held we keep our registration with
# our tests; a full merge after the hold lifts brings AMD's block back in.
HRX_PATHS=(ggml/src/ggml-hrx ggml/include/ggml-hrx.h 'tests/*hrx*' tests/CMakeLists.txt)

note()    { echo "::notice::$*"; }
warn()    { echo "::warning::$*"; }
die()     { echo "::error::$*"; exit 1; }
output()  { echo "$1" >> "${GITHUB_OUTPUT:-/dev/null}"; echo "output: ${1%%$'\n'*}"; }
summary() { echo "$*" >> "${GITHUB_STEP_SUMMARY:-/dev/null}"; echo "summary: $*"; }
push() {
  if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: git push $*"; else git push -q "$@"; fi
}
remote_url() {
  if [ -n "${GH_TOKEN:-}" ]; then echo "https://x-access-token:${GH_TOKEN}@github.com/$1.git"
  else echo "https://github.com/$1.git"; fi
}
ident() { git -c user.name=hrx-bump -c user.email=hrx-bump@users.noreply.github.com "$@"; }
# Every command that writes a commit object must carry an identity: GitHub runners (and
# `env HOME=<empty>` locally) have no user.name/user.email, so a bare `git commit-tree`,
# `git commit`, `git merge` or `git replace --graft` dies with "empty ident name"/"Author
# identity unknown" (engine#406). ident() wraps the call sites; these exports cover any
# command added later.
export GIT_AUTHOR_NAME=hrx-bump GIT_AUTHOR_EMAIL=hrx-bump@users.noreply.github.com
export GIT_COMMITTER_NAME=hrx-bump GIT_COMMITTER_EMAIL=hrx-bump@users.noreply.github.com

# The commit hrx-system is held at, or empty. Lines starting with # are comments.
hold_sha() {
  [ -f "$ENGINE_DIR/$HOLD_FILE" ] || return 0
  grep -v '^[[:space:]]*#' "$ENGINE_DIR/$HOLD_FILE" | grep -oE '^[0-9a-f]{40}' | head -n1 || true
}

sync_llama() {
  : "${AMD_LLAMA:?}" "${OURS_HRX:?}"
  local hold refresh; hold=$(hold_sha)
  # Name the reason, not just the fact: every run since the token was scoped down logs
  # "refusing to allow a Personal Access Token to create or update workflow ... without
  # `workflow` scope (HTTP 422)" (engine#628), and "could not refresh" alone hid that the
  # fix is a token permission and not a fork problem. The message carries no secret.
  if [ "$DRY_RUN" != 1 ] && ! refresh=$(gh api -X POST repos/1bit-MONSTER/llama.cpp/merge-upstream -f branch=master 2>&1); then
    warn "could not refresh 1bit-MONSTER/llama.cpp master: $(printf '%s' "$refresh" | head -n1); continuing with the HRX pin sync"
  fi

  rm -rf patched && git init -q patched && cd patched
  git remote add fork "$(remote_url 1bit-MONSTER/llama.cpp)"
  git remote add amd https://github.com/AMD-Ecosystem/llama.cpp.git
  git fetch -q --filter=blob:none amd "$AMD_LLAMA"
  git fetch -q --filter=blob:none fork \
    "1bit/hrx-vulkan:refs/remotes/fork/mirror" "1bit/hrx-vulkan-patched:refs/remotes/fork/patched"
  local mirror patched pin=$AMD_LLAMA
  mirror=$(git rev-parse refs/remotes/fork/mirror)
  patched=$(git rev-parse refs/remotes/fork/patched)
  echo "AMD pin $pin, mirror 1bit/hrx-vulkan $mirror, our line 1bit/hrx-vulkan-patched $patched"

  # 1. The mirror. It holds AMD's commits only; ours live on 1bit/hrx-vulkan-patched.
  local rewritten=0
  if [ "$mirror" != "$pin" ] && ! git merge-base --is-ancestor "$mirror" "$pin"; then
    rewritten=1
    # Commits only: enough to compare histories and trees without downloading them.
    git init -q ../amdrefs
    git -C ../amdrefs fetch -q --filter=tree:0 https://github.com/AMD-Ecosystem/llama.cpp.git \
      '+refs/heads/*:refs/amd/*' "$pin"
    git -C ../amdrefs fetch -q --filter=tree:0 "$(remote_url 1bit-MONSTER/llama.cpp)" "+$mirror:refs/old-mirror"
    local foreign
    foreign=$(git -C ../amdrefs log --format='%h %s' "$mirror" --not "$pin" --glob='refs/amd/*')
    if [ -n "$foreign" ]; then
      die "1bit/hrx-vulkan has commits that are on no branch of AMD-Ecosystem/llama.cpp; the mirror must hold AMD's commits only. Move these to 1bit/hrx-vulkan-patched by hand: $(printf '%s; ' "$foreign")"
    fi
    # AMD rewrote history. Find the newest commit of the new pin whose tree equals a commit that
    # descends from the old mirror on some AMD branch (its pre-rewrite equivalent).
    local trees squash="" equiv=""
    trees=$( { git -C ../amdrefs log --format='%T %H' --ancestry-path --glob='refs/amd/*' "^$mirror"
               git -C ../amdrefs log -1 --format='%T %H' "$mirror"; } | sort -u -k1,1)
    while read -r c t; do
      equiv=$(awk -v t="$t" '$1 == t { print $2; exit }' <<< "$trees")
      if [ -n "$equiv" ]; then squash=$c; break; fi
    done < <(git -C ../amdrefs log --format='%H %T' "$pin" --not "$mirror")
    if [ -z "$squash" ]; then
      die "AMD's pin ${pin:0:12} does not contain 1bit/hrx-vulkan ${mirror:0:12} and no AMD commit after it has the same tree as any commit of the new pin; AMD rewrote its history in a way this job cannot map. Merge AMD's pin into 1bit/hrx-vulkan-patched by hand."
    fi
    git fetch -q --filter=blob:none amd "$equiv" 2>/dev/null || git fetch -q --filter=blob:none fork "$equiv"
    ident replace --graft "$squash" "$equiv"
    note "AMD rewrote its history: ${squash:0:12} (in the new pin) has the same tree as ${equiv:0:12}, which follows our mirror ${mirror:0:12}; merging as if ${squash:0:12} followed ${equiv:0:12}"
    summary "- AMD rewrote its llama.cpp history: \`${squash:0:12}\` = tree of \`${equiv:0:12}\`; old mirror tip \`${mirror:0:12}\` tagged \`hrx-vulkan-${mirror:0:12}\`."
  fi

  # 2. Our line.
  local rebased=$patched mode=merge
  [ -n "$hold" ] && mode=held
  if git merge-base --is-ancestor "$pin" "$patched"; then
    echo "1bit/hrx-vulkan-patched already contains AMD's ${pin:0:12}"
  elif [ "$mode" = merge ]; then
    git checkout -q --detach "$patched"
    if ! ident merge -q --no-edit --no-ff "$pin"; then
      local conflicts outside
      conflicts=$(git diff --name-only --diff-filter=U || true)
      outside=$(printf '%s\n' "$conflicts" | grep -v '^ggml/src/ggml-hrx/' | grep -v '^$' || true)
      if [ -n "$outside" ]; then
        git merge --abort || true
        die "merging AMD's ${pin:0:12} into 1bit/hrx-vulkan-patched conflicts outside ggml/src/ggml-hrx: $(printf '%s ' $outside); resolve those files by hand"
      fi
      # Every conflict is inside our HRX backend: keep ours (measured 2026-10-06: AMD core plus
      # our ggml-hrx passes the HRX0 gate 1073/1073 with 0 NaN; AMD's own ggml-hrx failed 54 cases
      # with NaN, engine#315). A file we deleted and AMD changed stays deleted.
      printf '%s\n' "$conflicts" | while read -r path; do
        [ -n "$path" ] || continue
        if git cat-file -e ":2:$path" 2>/dev/null; then
          git checkout --ours -- "$path" && git add -- "$path"
        else
          note "kept our deletion of $path (modify/delete conflict)"
          git rm -q -f -- "$path"
        fi
      done
      ident commit -q --no-edit
      note "merged AMD's ${pin:0:12} keeping our ggml/src/ggml-hrx for $(printf '%s\n' "$conflicts" | grep -c .) files"
    fi
    rebased=$(git rev-parse HEAD)
  else
    # hrx-system is held: take AMD's changes outside the HRX backend paths only.
    local base hrx_changed theirs_tree t res
    base=$(git merge-base "$patched" "$pin")
    hrx_changed=$(git diff --name-only "$base" "$pin" -- "${HRX_PATHS[@]}" | grep -c . || true)
    export GIT_INDEX_FILE="$PWD/.git/held-index"
    git read-tree "$pin"
    git rm -q -r --cached --ignore-unmatch -- "${HRX_PATHS[@]}"
    GIT_INDEX_FILE="$PWD/.git/base-index" git read-tree "$base"
    GIT_INDEX_FILE="$PWD/.git/base-index" git ls-files -s -- "${HRX_PATHS[@]}" | git update-index --index-info
    theirs_tree=$(git write-tree)
    unset GIT_INDEX_FILE
    if [ "$theirs_tree" = "$(git rev-parse "$base^{tree}")" ]; then
      note "hrx-system is held ($HOLD_FILE): AMD's ${pin:0:12} changes only HRX backend files ($hrx_changed since ${base:0:12}), which are written against AMD's hrx-system; 1bit/hrx-vulkan-patched stays at ${patched:0:12}"
      summary "- llama.cpp: AMD's \`${pin:0:12}\` changes only HRX backend files ($hrx_changed); our line stays at \`${patched:0:12}\` while hrx-system is held."
    else
      t=$(ident commit-tree "$theirs_tree" -p "$base" -m "AMD ${pin:0:12} outside the HRX backend")
      if ! res=$(git merge-tree --write-tree --name-only --merge-base="$base" "$patched" "$t"); then
        die "taking AMD's ${pin:0:12} changes outside the HRX backend conflicts in: $(printf '%s\n' "$res" | sed -n '2,/^$/p' | tr '\n' ' '); resolve those files by hand on 1bit/hrx-vulkan-patched"
      fi
      res=$(head -n1 <<< "$res")
      if [ "$res" = "$(git rev-parse "$patched^{tree}")" ]; then
        echo "AMD's changes outside the HRX backend are already on our line"
      else
        rebased=$(ident commit-tree "$res" -p "$patched" -m "Take AMD ${pin:0:12} outside the HRX backend (hrx-system held)

AMD's ggml-hrx changes ($hrx_changed files since ${base:0:12}) are left out because they are
written against AMD's hrx-system and ours is held ($HOLD_FILE). AMD is not a merge
parent here, so a later full merge still brings them in.")
        note "took AMD's ${pin:0:12} changes outside the HRX backend; left out $hrx_changed HRX backend files (hrx-system held)"
        summary "- llama.cpp: took AMD's \`${pin:0:12}\` changes outside the HRX backend; $hrx_changed HRX backend files left out (hrx-system held)."
      fi
    fi
  fi

  # 3. Push: our line first (tagging the commit the engine may pin), then the mirror.
  if [ "$rebased" != "$patched" ]; then
    git tag -f "patched-${patched:0:12}" "$patched"
    push fork "refs/tags/patched-${patched:0:12}"
    push --force-with-lease="refs/heads/1bit/hrx-vulkan-patched:$patched" fork "$rebased:refs/heads/1bit/hrx-vulkan-patched"
  fi
  if [ "$mirror" != "$pin" ]; then
    if [ "$rewritten" = 1 ]; then
      git tag -f "hrx-vulkan-${mirror:0:12}" "$mirror"
      push fork "refs/tags/hrx-vulkan-${mirror:0:12}"
      push --force-with-lease="refs/heads/1bit/hrx-vulkan:$mirror" fork "$pin:refs/heads/1bit/hrx-vulkan"
    else
      push fork "$pin:refs/heads/1bit/hrx-vulkan"
    fi
  fi
  output "rebased=$rebased"
  output "patches<<EOF
$(git log --format='- `%h` %s' "${OURS_LLAMA:-$pin}..$rebased")
EOF"
}

sync_hrx() {
  : "${AMD_HRX:?}" "${OURS_HRX:?}"
  local hold; hold=$(hold_sha)
  if [ -n "$hold" ]; then
    [ "$hold" = "$OURS_HRX" ] || warn "$HOLD_FILE holds ${hold:0:12} but the engine pins hrx-system ${OURS_HRX:0:12}; keeping the engine's pin"
    note "hrx-system is held at ${OURS_HRX:0:12} by $HOLD_FILE; AMD's tested pin is ${AMD_HRX:0:12}. Not merged."
    summary "- hrx-system: held at \`${OURS_HRX:0:12}\` ($HOLD_FILE); AMD's tested pin is \`${AMD_HRX:0:12}\`."
    output "hrx_rebased=$OURS_HRX"
    output "held=hrx-system HELD at \`${OURS_HRX:0:12}\` by \`$HOLD_FILE\` (AMD's tested pin is \`${AMD_HRX:0:12}\`)."
    return 0
  fi
  rm -rf hrxsystem && git init -q hrxsystem && cd hrxsystem
  git remote add fork "$(remote_url 1bit-MONSTER/hrx-system)"
  git remote add amd https://github.com/ROCm/hrx-system.git
  git fetch -q --filter=blob:none amd "$AMD_HRX" || git fetch -q --filter=blob:none fork "$AMD_HRX"
  git fetch -q --filter=blob:none fork 1bit/main:refs/remotes/fork/1bit-main
  local ours hrx_rebased
  ours=$(git rev-parse refs/remotes/fork/1bit-main)
  hrx_rebased=$ours
  if ! git merge-base --is-ancestor "$AMD_HRX" "$ours"; then
    git checkout -q --detach "$ours"
    if ! ident merge -q --no-edit --no-ff "$AMD_HRX"; then
      local conflicts
      conflicts=$(git diff --name-only --diff-filter=U | tr '\n' ' ' || true)
      git merge --abort || true
      die "merging AMD's hrx-system ${AMD_HRX:0:12} into 1bit-MONSTER/hrx-system 1bit/main conflicts in: ${conflicts}; resolve those files by hand"
    fi
    hrx_rebased=$(git rev-parse HEAD)
    git tag -f "main-${ours:0:12}" "$ours"
    push fork "refs/tags/main-${ours:0:12}"
    push --force-with-lease="refs/heads/1bit/main:$ours" fork "$hrx_rebased:refs/heads/1bit/main"
  fi
  output "hrx_rebased=$hrx_rebased"
  output "held="
}

case "$cmd" in
  llama) sync_llama ;;
  hrx) sync_hrx ;;
  *) echo "usage: $0 llama|hrx" >&2; exit 2 ;;
esac
