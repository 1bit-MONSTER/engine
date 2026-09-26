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
# weekly.sh <workdir> [stage ...]
#
# The Sunday release (docs/releases.md). Every upstream the engine pins moves during the week
# through the bump-* PRs, and Lemonade ships on Fridays. On Sunday, on Strix Halo:
#   bumps    verifies each open bump-* PR here, merged onto main, and merges the ones that
#            pass; the rest stay open and the release lists them
#   build    main at the merged pins, with every backend this machine builds
#   test     ctest, including the serve end-to-end tests on a small GGUF
#   package  the Linux tarball, Lemonade with the onebit recipe, the Windows zip, 1bit OS
#   release  what moved upstream since the last release (tools/weekly_changes.py), published
#            as the GitHub release v<ISO year>.<ISO week> with the packages
# No stage: all of them, in that order. WEEKLY_DRY_RUN=1 merges and publishes nothing.
# WEEKLY_REF (default main) is what build checks out; bumps always works on main.
# WEEKLY_JOBS (default 6) caps build parallelism; heavy steps wait for WEEKLY_MIN_FREE_GB (16).
# Needs gh (logged in, allowed to merge on the engine), TheRock in /opt/rocm-therock, and
# what the build scripts each list.
set -euo pipefail

[ $# -ge 1 ] || { sed -n '16,32p' "$0"; exit 2; }
W=$(mkdir -p "$1" && cd "$1" && pwd); shift
STAGES=${*:-bumps build test package release}
REPO=1bit-MONSTER/engine
SRC=$W/src BUILD=$W/build OUT=$W/out LOGS=$W/logs
GGUF=${ONEBIT_SERVE_TEST_GGUF:-$HOME/models/Qwen3-0.6B-Q4_K_M.gguf}
DRY=${WEEKLY_DRY_RUN:-0}
# the box is shared: a capped build leaves room for whatever else runs on Sunday
JOBS=${WEEKLY_JOBS:-6}
export CMAKE_BUILD_PARALLEL_LEVEL=$JOBS
# the submodules the packages are built from (linux, laya and comfyui.cpp only on their bumps)
SUBMODULES=(hrx-system llama.cpp llama.cpp-vulkan llama.cpp-rocmfpx zinc xdna-driver lemonade ryzenai-server ds4 tokenizers)
mkdir -p "$LOGS"
say() { echo "[$(date +%H:%M:%S)] $*"; }
trap 'say "FAILED at line $LINENO: $BASH_COMMAND (logs: $LOGS)"' ERR
# Heavy steps wait for room, and stop before the box runs out of memory (unified memory: GPU
# allocations count too); one the guard stopped is retried, and the builds pick up where they were.
MIN_GB=${WEEKLY_MIN_FREE_GB:-16}
room() {
    local i
    for i in $(seq 360); do
        [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge "$MIN_GB" ] && return 0
        [ "$i" = 1 ] && say "waiting for ${MIN_GB} GB of free memory"
        sleep 60
    done
    say "no ${MIN_GB} GB free in 6 hours"; return 1
}
guard() {
    local try
    for try in $(seq 10); do
        room || return 1
        "$SRC/scripts/mem-guard.sh" 4 "$@" && return 0
        say "memory guard stopped it (try $try): $*"
    done
    return 1
}

# --- the tree -----------------------------------------------------------------------------
checkout() {  # checkout <ref> [pr number to merge on top]
    [ -d "$SRC/.git" ] || git clone -q "https://github.com/$REPO.git" "$SRC"
    git -C "$SRC" fetch -q origin "$1"
    git -C "$SRC" checkout -q --detach FETCH_HEAD
    if [ -n "${2:-}" ]; then
        git -C "$SRC" fetch -q origin "pull/$2/head"
        git -C "$SRC" -c user.name=weekly -c user.email=weekly@1bit.gg merge -q --no-edit FETCH_HEAD
    fi
    local mods=("${SUBMODULES[@]/#/third_party/}") m
    for m in linux laya comfyui.cpp; do   # fetched only when that pin is what moves
        if [ -n "${2:-}" ] && ! git -C "$SRC" diff --quiet HEAD^1 HEAD -- "third_party/$m"; then
            mods+=("third_party/$m")
        fi
    done
    git -C "$SRC" submodule sync -q
    git -C "$SRC" submodule update -q --init --recursive --depth 1 -- "${mods[@]}" 2>/dev/null ||
        git -C "$SRC" submodule update -q --init --recursive -- "${mods[@]}"
}

pin() { git -C "$SRC" ls-tree HEAD "third_party/$1" | awk '{print $3}'; }

# --- build / test -------------------------------------------------------------------------
build() {
    # XRT + the XDNA plugin from the pinned xdna-driver, rebuilt when that pin moves
    local xdna=$W/xdna
    if [ "$(cat "$xdna/.pin" 2>/dev/null)" != "$(pin xdna-driver)" ]; then
        say "XDNA stack $(pin xdna-driver | cut -c1-12)"
        rm -rf "$xdna"
        guard "$SRC/scripts/build-xdna.sh" "$xdna" "$JOBS" > "$LOGS/xdna.log" 2>&1
        pin xdna-driver > "$xdna/.pin"
    fi
    say "engine $(git -C "$SRC" rev-parse --short HEAD)"
    local extra=()
    [ -d "$HOME/models/laya-pinned" ] && extra+=(-DONEBIT_LAYA_ROOT="$HOME/models/laya-pinned")
    # shellcheck disable=SC2206
    extra+=(${WEEKLY_CMAKE_ARGS:-})
    cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DONEBIT_VULKAN=ON -DONEBIT_HRX=ON -DONEBIT_LEAN=ON -DONEBIT_ZINC=ON -DONEBIT_ONNX=ON \
        -DONEBIT_DS4=ON -DONEBIT_HF_TOKENIZERS=ON \
        -DONEBIT_NPU=ON -DONEBIT_XRT_ROOT="$xdna/root/opt/xilinx/xrt" \
        -DONEBIT_SERVE_TEST_GGUF="$GGUF" "${extra[@]}" > "$LOGS/configure.log" 2>&1
    guard cmake --build "$BUILD" > "$LOGS/build.log" 2>&1
}

run_tests() {
    say "ctest"
    ctest --test-dir "$BUILD" --output-on-failure > "$LOGS/ctest.log" 2>&1
}

# the extra check a bump needs beyond build + ctest (docs/releases.md, "Bumps")
bump_check() {  # bump_check <branch>
    case "$1" in
        bump-lemonade/*) lemonade_suite ;;
        bump-laya/*)
            "$SRC/scripts/fetch-laya.sh" "$W/laya" > "$LOGS/laya.log" 2>&1
            "$SRC/tests/laya_route_e2e.sh" "$BUILD/1bit" "$W/laya" >> "$LOGS/laya.log" 2>&1 ;;
        bump-ds4/*) DS4_TEST=1 guard "$SRC/scripts/build-ds4.sh" "$W/ds4-test" rocm > "$LOGS/ds4.log" 2>&1 ;;
        bump-linux/*) guard "$SRC/scripts/build-kernel.sh" "$W/kernel" > "$LOGS/kernel.log" 2>&1 ;;
        bump-comfyui/*) guard "$SRC/scripts/build-comfyui.sh" "$W/comfyui" > "$LOGS/comfyui.log" 2>&1 ;;
        *) : ;;  # hrx, llama-vulkan, rocmfpx, zinc, xdna, tokenizers: covered by build + ctest
    esac
}

# Lemonade's own LLM suite through the onebit recipe, on Vulkan (docs/lemonade.md)
lemonade_suite() {
    guard "$SRC/scripts/build-lemonade.sh" "$W/lemonade" > "$LOGS/lemonade-build.log" 2>&1
    local venv=$W/lemonade-venv
    [ -x "$venv/bin/python" ] || { python3 -m venv "$venv" && "$venv/bin/pip" -q install openai requests huggingface_hub; }
    ( cd "$SRC/third_party/lemonade" &&
      PATH="$W/lemonade/bin:$BUILD:$PATH" LEMONADE_ONEBIT_BIN="$BUILD/1bit" \
      "$venv/bin/python" test/server_llm.py --wrapped-server onebit --backend vulkan ) > "$LOGS/lemonade-suite.log" 2>&1
}

# --- stages -------------------------------------------------------------------------------
stage_bumps() {
    local list=() n branch title log
    mapfile -t list < <(gh pr list -R "$REPO" --state open --limit 50 --json number,headRefName \
        --jq '.[] | select(.headRefName | startswith("bump-")) | "\(.number) \(.headRefName)"' | sort -n)
    echo "[]" > "$W/bumps.json"
    for entry in "${list[@]}"; do
        n=${entry%% *} branch=${entry#* }
        title=$(gh pr view "$n" -R "$REPO" --json title --jq .title)
        log=$LOGS/bump-$n.log
        say "bump #$n $branch"
        local result=merged why="" rc
        # a subshell outside any condition, so that set -e holds inside it
        set +e; ( set -e; checkout main "$n"; build; run_tests; bump_check "$branch" ) > "$log" 2>&1; rc=$?; set -e
        if [ $rc -ne 0 ]; then
            result=held why="failed on Strix Halo: $(tail -n 3 "$log" "$LOGS"/ctest.log 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
        elif [ "$DRY" = 1 ]; then
            result=passed why="dry run: not merged"
        else
            gh pr update-branch "$n" -R "$REPO" > /dev/null 2>&1 || true
            sleep 20
            if ! gh pr checks "$n" -R "$REPO" --watch --interval 30 > "$LOGS/bump-$n-checks.log" 2>&1; then
                result=held why="CI failed"
            elif ! gh pr merge "$n" -R "$REPO" --squash --delete-branch > /dev/null; then
                result=held why="merge refused"
            else
                gh pr comment "$n" -R "$REPO" --body "Verified on Strix Halo by the Sunday release (scripts/weekly.sh): build, ctest, and the check for this bump passed." > /dev/null || true
            fi
        fi
        say "  $result ${why}"
        python3 - "$W/bumps.json" "$n" "$branch" "$title" "$result" "$why" <<'PY'
import json, sys
p, n, branch, title, result, why = sys.argv[1:]
d = json.load(open(p)); d.append({"number": int(n), "branch": branch, "title": title, "result": result, "why": why})
json.dump(d, open(p, "w"), indent=1)
PY
    done
}

stage_build() { checkout "${WEEKLY_REF:-main}"; build; }
stage_test() { run_tests; }

TAG_FILE=$W/tag
stage_package() {
    local tag=${WEEKLY_TAG:-v$(date +%G.%V)} n=0
    while gh release view "$tag" -R "$REPO" > /dev/null 2>&1; do n=$((n + 1)); tag=v$(date +%G.%V).$n; done
    echo "$tag" > "$TAG_FILE"
    rm -rf "$OUT" && mkdir -p "$OUT"
    say "package $tag: Linux"
    "$SRC/scripts/package-linux.sh" "$BUILD" "$OUT" "$tag" "$W/xdna/root/opt/xilinx/xrt" > "$LOGS/package-linux.log" 2>&1
    say "package $tag: Lemonade + onebit"
    guard "$SRC/scripts/build-lemonade.sh" "$W/lemonade" > "$LOGS/lemonade-build.log" 2>&1
    tar -C "$W/lemonade" --transform "s,^,lemonade-onebit-$tag/," -cf - bin | zstd -q -T0 -19 \
        -o "$OUT/lemonade-onebit-$tag-linux-x86_64.tar.zst"
    say "package $tag: Windows"
    rm -rf "$W/windows"
    guard "$SRC/scripts/build-windows.sh" "$W/windows" > "$LOGS/windows.log" 2>&1
    cp "$SRC/LICENSE" "$SRC/NOTICE" "$W/windows/"
    (cd "$W/windows" && zip -q "$OUT/1bit-$tag-windows-x64.zip" ./*.exe ./*.dll LICENSE NOTICE)  # not work/
    say "package $tag: 1bit OS"
    rm -rf "$W/os"
    guard "$SRC/os/mini/mkimage.sh" "$W/os" "$BUILD" > "$LOGS/os.log" 2>&1
    zstd -q -T0 -19 "$W/os/1bit-os.img" -o "$OUT/1bit-os-$tag.img.zst"
    cp "$W/os/1bit-os.efi" "$OUT/1bit-os-$tag.efi"
    (cd "$OUT" && sha256sum -- * > SHA256SUMS)
    ls -la "$OUT"
}

stage_release() {
    local tag; tag=$(cat "$TAG_FILE")
    local prev from
    prev=$(gh release list -R "$REPO" --exclude-drafts --limit 1 --json tagName --jq '.[0].tagName // ""')
    git -C "$SRC" fetch -q origin main --tags
    if [ -n "$prev" ]; then from=$(git -C "$SRC" rev-list -1 "$prev");
    else from=$(git -C "$SRC" rev-list -1 --before="7 days ago" origin/main); fi
    [ -n "$from" ] || from=$(git -C "$SRC" rev-list --max-parents=0 origin/main | tail -1)
    say "release $tag: changes ${from:0:12}..$(git -C "$SRC" rev-parse --short HEAD) (previous: ${prev:-none})"
    python3 "$SRC/tools/weekly_changes.py" --src "$SRC" --from "$from" --to HEAD --tag "$tag" \
        --previous "$prev" --bumps "$W/bumps.json" --json "$OUT/changes.json" --md "$W/notes.md"
    if [ "$DRY" = 1 ]; then say "dry run: not publishing"; return; fi
    gh release create "$tag" -R "$REPO" --target "$(git -C "$SRC" rev-parse HEAD)" \
        --title "1bit engine $tag" -F "$W/notes.md" "$OUT"/*
    say "released $tag"
}

for s in $STAGES; do "stage_$s"; done
