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
#   package  the engine, folded into the Lemonade + onebit package, and 1bit OS
#   release  what moved upstream since the last release (tools/weekly_changes.py), published
#            as the GitHub release v<ISO year>.<ISO week> with the packages
# No stage: all of them, in that order. WEEKLY_DRY_RUN=1 merges and publishes nothing.
# WEEKLY_REF (default main) is what build checks out; bumps always works on main.
# WEEKLY_JOBS (default 4, what ~/lb/thermal-run.sh allows on Strix Halo) caps build parallelism; heavy steps wait for WEEKLY_MIN_FREE_GB (16).
# Needs gh (logged in, allowed to merge on the engine, and able to read the private repositories
# config/npu-runtime.config and config/gpu-kernels.config pin), TheRock in /opt/rocm-therock, and
# what the build scripts each list; preflight (at the end) checks the toolchain, hipcc, patchelf and the kernel image.
set -euo pipefail

[ $# -ge 1 ] || { sed -n '16,33p' "$0"; exit 2; }
W=$(mkdir -p "$1" && cd "$1" && pwd); shift
STAGES=${*:-bumps build test package release}
REPO=1bit-MONSTER/engine
SRC=$W/src BUILD=$W/build OUT=$W/out LOGS=$W/logs
GGUF=${ONEBIT_SERVE_TEST_GGUF:-$HOME/models/Qwen3-0.6B-Q4_K_M.gguf}
DRY=${WEEKLY_DRY_RUN:-0}
# the box is shared: a capped build leaves room for whatever else runs on Sunday
JOBS=${WEEKLY_JOBS:-4}
export CMAKE_BUILD_PARALLEL_LEVEL=$JOBS
# the submodules the packages are built from (laya only on its bump)
# (not xdna-driver: it pins the amdxdna kernel driver, which the release does not build)
SUBMODULES=(hrx-system llama.cpp lemonade ds4 tokenizers)
mkdir -p "$LOGS"
# pack_missing_libs (package_checks); this script's own copy, so it works before build checks out $SRC
# shellcheck source=scripts/pack-elf.sh
. "$(cd "$(dirname "$0")" && pwd)/pack-elf.sh"
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
    local rc
    for try in $(seq 10); do
        room || return 1
        rc=0; "$SRC/scripts/mem-guard.sh" 4 "$@" || rc=$?
        [ "$rc" = 0 ] && return 0
        # mem-guard kills with SIGKILL (exit 137); any other status is the command's own failure,
        # which a retry cannot fix
        if [ "$rc" != 137 ]; then say "FAILED (exit $rc): $*"; return "$rc"; fi
        say "memory guard stopped it (try $try): $*"
    done
    return 1
}

# --- the tree -----------------------------------------------------------------------------
checkout() {  # checkout <ref> [pr number to merge on top]
    [ -d "$SRC/.git" ] || git clone -q "https://github.com/$REPO.git" "$SRC"
    git -C "$SRC" fetch -q origin "$1"
    git -C "$SRC" checkout -q -f --detach FETCH_HEAD   # the workspace is ours: a stray edit never blocks a run
    if [ -n "${2:-}" ]; then
        git -C "$SRC" fetch -q origin "pull/$2/head"
        git -C "$SRC" -c user.name=weekly -c user.email=weekly@1bit.gg merge -q --no-edit FETCH_HEAD
    fi
    local mods=("${SUBMODULES[@]/#/third_party/}") m
    for m in laya; do   # fetched only when that pin is what moves
        if [ -n "${2:-}" ] && ! git -C "$SRC" diff --quiet HEAD^1 HEAD -- "third_party/$m"; then
            mods+=("third_party/$m")
        fi
    done
    git -C "$SRC" submodule sync -q
    git -C "$SRC" submodule update -q --init --recursive --depth 1 -- "${mods[@]}" 2>/dev/null ||
        git -C "$SRC" submodule update -q --init --recursive -- "${mods[@]}"
}

pin() { git -C "$SRC" ls-tree HEAD "third_party/$1" | awk '{print $3}'; }

# the machine a held bump failed on, for the release notes: its name and its AMD GPU(s), from
# sysfs (scripts/gpu-targets.sh), e.g. "strixhalo (gfx1151)" or "ryzen (gfx1036, gfx1201)"
host_desc() {
    local g; g=$("$(cd "$(dirname "$0")" && pwd)/gpu-targets.sh" 2>/dev/null | paste -sd, - | sed 's/,/, /g')
    echo "$(hostname -s) (${g:-no AMD GPU})"
}

# --- build / test -------------------------------------------------------------------------
build() {
    # the private components compiled in (docs/releases.md, "Private components"): the NPU runtime
    # (into 1bit) and the HIP kernels (into the HRX llama.cpp), each a checkout at the commit its
    # config/*.config pins; the build stops here when one cannot be fetched at that commit
    local npurt
    npurt=$("$SRC/scripts/fetch-private.sh" "$SRC/config/npu-runtime.config" "$W/npu-runtime" 2> "$LOGS/private.log") ||
        { cat "$LOGS/private.log"; return 1; }
    "$SRC/scripts/fetch-private.sh" "$SRC/config/gpu-kernels.config" "$W/gpu-kernels" > /dev/null 2>> "$LOGS/private.log" ||
        { cat "$LOGS/private.log"; return 1; }
    say "engine $(git -C "$SRC" rev-parse --short HEAD), NPU runtime $(git -C "$W/npu-runtime" rev-parse --short HEAD), HIP kernels $(git -C "$W/gpu-kernels" rev-parse --short HEAD)"
    local extra=()
    [ -d "$HOME/models/laya-pinned" ] && extra+=(-DONEBIT_LAYA_ROOT="$HOME/models/laya-pinned")
    # shellcheck disable=SC2206
    extra+=(${WEEKLY_CMAKE_ARGS:-})
    # The NPU kernels (docs/releases.md, "The NPU engine"): sealed, packed beside 1bit in
    # npu/kernels. On by default (the npu gate phase passed on Strix Halo; WEEKLY_NPU_KERNELS=0
    # builds a package without them). The private checkout does the work through its one entry
    # point (key install, sealing, the per-commit kernel cache); this script names nothing in it.
    # ONEBIT_NPU_RELEASE makes the built 1bit load sealed kernels only.
    if [ "${WEEKLY_NPU_KERNELS:-1}" = 1 ]; then
        "$SRC/scripts/fetch-private.sh" "$SRC/config/npu-kernels.config" "$W/npu-kernels" > /dev/null 2>> "$LOGS/private.log" ||
            { cat "$LOGS/private.log"; return 1; }
        guard "$W/npu-kernels/addons/release-kernels.sh" "$BUILD/npu/kernels" "${ONEBIT_NPU_KEY_DIR:-$HOME/.config/1bit-npu-key}" \
            > "$LOGS/npu-kernels.log" 2>&1 || { tail -n 5 "$LOGS/npu-kernels.log"; return 1; }
        extra+=(-DONEBIT_NPU_PRIVATE="$W/npu-kernels" -DONEBIT_NPU_RELEASE=ON)
        say "NPU kernels $(git -C "$W/npu-kernels" rev-parse --short HEAD): $(ls "$BUILD/npu/kernels" | wc -l) sealed sets"
    fi
    cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DONEBIT_HRX=ON -DONEBIT_GPU_PRIVATE="$W/gpu-kernels" \
        -DONEBIT_DS4=ON -DONEBIT_HF_TOKENIZERS=ON \
        -DONEBIT_NPU=ON -DONEBIT_NPU_RUNTIME=own -DONEBIT_NPU_RT_DIR="$npurt" \
        -DONEBIT_SERVE_TEST_GGUF="$GGUF" "${extra[@]}" > "$LOGS/configure.log" 2>&1
    guard cmake --build "$BUILD" > "$LOGS/build.log" 2>&1
}

run_tests() {
    say "ctest"
    ctest --test-dir "$BUILD" --output-on-failure > "$LOGS/ctest.log" 2>&1
}

# every pin a PR moves must be on the branch .gitmodules says that submodule tracks: a shallow
# clone of that branch cannot fetch a commit off it, and GitHub may drop an unreferenced one
pins_on_branch() {
    local path url branch sha repo status bad=0
    while read -r path; do
        branch=$(git config -f "$SRC/.gitmodules" --get "submodule.$path.branch" || true)
        [ -n "$branch" ] || continue
        url=$(git config -f "$SRC/.gitmodules" --get "submodule.$path.url")
        sha=$(git -C "$SRC" ls-tree HEAD "$path" | awk '{print $3}')
        repo=${url#https://github.com/}; repo=${repo%.git}
        status=$(gh api "repos/$repo/compare/$sha...$branch" --jq .status 2>/dev/null || echo unknown)
        case "$status" in
            identical|ahead) echo "ok   $path ${sha:0:12} is on $repo $branch" ;;
            *) echo "FAIL $path ${sha:0:12} is not on $repo $branch ($status)"; bad=1 ;;
        esac
    done < <(git -C "$SRC" diff --name-only HEAD^1 HEAD -- third_party)
    return $bad
}

# a llama.cpp bump re-runs every model row registry/check_models.tsv lists for that backend
# (the architectures only our fork has, such as zaya, and the MoE and Qwen rows among them)
model_rows() {  # model_rows <hrx|npu>
    guard python3 "$SRC/tools/registry_check.py" "$BUILD/1bit" --models "$HOME/models" --only "$1" \
        > "$LOGS/models-$1.log" 2>&1 || return 1
    ! grep -q '^FAIL' "$LOGS/models-$1.log"
}

# the extra check a bump needs beyond build + ctest (docs/releases.md, "Bumps")
bump_check() {  # bump_check <branch>
    case "$1" in
        bump-hrx/*) model_rows hrx ;;
        bump-lemonade/*) lemonade_suite ;;
        bump-laya/*)
            "$SRC/scripts/fetch-laya.sh" "$W/laya" > "$LOGS/laya.log" 2>&1
            "$SRC/tests/laya_route_e2e.sh" "$BUILD/1bit" "$W/laya" >> "$LOGS/laya.log" 2>&1 ;;
        bump-ds4/*) DS4_TEST=1 guard "$SRC/scripts/build-ds4.sh" "$W/ds4-test" > "$LOGS/ds4.log" 2>&1 ;;
        *) : ;;  # xdna (the kernel driver pin), tokenizers: covered by build + ctest
    esac
}

# Lemonade's own LLM suite through the onebit recipe, on HRX (docs/lemonade.md); lemond keeps its
# state in the workdir, not in the user's ~/.cache/lemonade and ~/.config/lemonade
lemonade_suite() {
    guard "$SRC/scripts/build-lemonade.sh" "$W/lemonade" > "$LOGS/lemonade-build.log" 2>&1
    local venv=$W/lemonade-venv
    [ -x "$venv/bin/python" ] || python3 -m venv "$venv"
    "$venv/bin/pip" -q install -r "$SRC/third_party/lemonade/test/requirements.txt"  # the suite's own list
    # the suite talks to a running lemond: ours, on a port of its own, with this build's 1bit
    local port=13399 pid rc=0
    mkdir -p "$W/lemonade-state/cache" "$W/lemonade-state/config"
    LEMONADE_ONEBIT_BIN="$BUILD/1bit" "$W/lemonade/bin/lemond" --port "$port" \
        "$W/lemonade-state/cache" "$W/lemonade-state/config" > "$LOGS/lemond.log" 2>&1 &
    pid=$!
    ( cd "$SRC/third_party/lemonade" &&
      PATH="$W/lemonade/bin:$BUILD:$PATH" LEMONADE_ONEBIT_BIN="$BUILD/1bit" LEMONADE_TEST_PORT="$port" \
      "$venv/bin/python" test/server_llm.py --wrapped-server onebit --backend hrx ) > "$LOGS/lemonade-suite.log" 2>&1 || rc=$?
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
    return $rc
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
        set +e
        ( set -e
          echo checkout > "$W/.step"; checkout main "$n"
          echo "the pins' branches" > "$W/.step"; pins_on_branch > "$LOGS/pins.log" 2>&1
          echo build > "$W/.step"; build
          echo ctest > "$W/.step"; run_tests
          echo "the $branch check" > "$W/.step"; bump_check "$branch" ) > "$log" 2>&1
        rc=$?; set -e
        if [ $rc -ne 0 ]; then
            # the step that failed, and the last lines of the log it wrote (the newest besides ours)
            local last; last=$(ls -t "$LOGS"/*.log | grep -v "/bump-" | head -1)
            result=held why="failed at $(cat "$W/.step") on $(host_desc) ($(basename "$last")): $(tail -n 2 "$last" | tr '\n' ' ' | cut -c1-240)"
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

# What the engine tarball must and must not hold (docs/releases.md, "Private components"):
# It is packed by scripts/package-linux.sh and folded into the Lemonade + onebit package as 1bit/;
#   XRT-free  no program or library needs or names AMD's XRT (libxrt*: readelf NEEDED, ldd,
#             strings), and no libxrt* file is packed; the NPU runtime is compiled into 1bit
#   kernels   the HRX ggml library carries the private HIP kernels (it exports the add-on's
#             ggml_hrx_hip_addon_register), compiled in as code objects
#   no source no source file (.hip .c .cc .cpp .cxx .h .hpp .inc .cu .py .cmake) and no file named
#             like one of the private checkouts' files
#   licensed  the closed binaries' LICENSE-BINARY and NOTICE-BINARY are packed beside LICENSE and NOTICE
#   npu       (with WEEKLY_NPU_KERNELS) every file under npu/kernels but layout.json is sealed, never a
#             plaintext ELF; every ELF in the tarball is for the host (x86-64); the packed 1bit has no
#             symbol table and its strings name no private source file and no build-machine home
# Checks the build's 1bit and every file in the tarball.
package_checks() {  # package_checks <linux tarball>
    # (no `cmd | grep -q` here: under pipefail grep's early exit fails the pipeline on a match)
    # (lists go to stderr through `tee >(cat >&2)`, not `tee /dev/stderr`: reopening the log by
    # that name truncates it, and the earlier lines of package-checks.log were lost, 2026-10-20)
    local t bad=0 f n names pin
    t=$(mktemp -d)
    tar -C "$t" -xf "$1" --use-compress-program=unzstd
    for f in "$BUILD/1bit" $(find "$t" -type f); do
        [ "$(head -c 4 "$f" | od -An -c | tr -d ' ')" = '177ELF' ] || continue
        n=$(readelf -d "$f" 2>/dev/null | grep -c 'NEEDED.*libxrt' || true)
        [ "$n" = 0 ] || { echo "FAIL $f needs libxrt"; bad=1; }
        n=$(ldd "$f" 2>/dev/null | grep -c libxrt || true)
        [ "$n" = 0 ] || { echo "FAIL $f: ldd shows libxrt"; bad=1; }
        n=$(strings -a "$f" | grep -c libxrt || true)
        [ "$n" = 0 ] || { echo "FAIL $f names libxrt"; bad=1; }
    done
    n=$(find "$t" -name 'libxrt*' | wc -l)
    [ "$n" = 0 ] || { echo "FAIL the tarball carries an XRT library"; bad=1; }
    [ $bad = 0 ] && echo "ok   XRT-free: $(basename "$1") and $BUILD/1bit"
    f=$(find "$t" -path '*/hrx/llama/bin/libggml-hrx.so*' -type f | head -1)
    n=0; [ -n "$f" ] && n=$(nm -D --defined-only "$f" | grep -c ggml_hrx_hip_addon_register || true)
    if [ "$n" != 0 ]; then
        echo "ok   private HIP kernels compiled into $(basename "$f") ($(strings -a "$f" | grep -c amdgcn-amd-amdhsa || true) code object targets)"
    else echo "FAIL no private HIP kernels in the HRX ggml library (${f:-missing})"; bad=1; fi
    n=$(find "$t" -type f \( -name '*.hip' -o -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \
            -o -name '*.h' -o -name '*.hpp' -o -name '*.inc' -o -name '*.cu' -o -name '*.py' -o -name '*.cmake' \
            -o -name '*.mlir' -o -name '*.mlirbc' \) | tee >(cat >&2) | wc -l)
    [ "$n" = 0 ] || { echo "FAIL source files in the tarball"; bad=1; }
    names=$(for pin in npu-runtime gpu-kernels npu-kernels; do
                [ -d "$W/$pin" ] || continue
                git -C "$W/$pin" ls-files -- "$(sed -n 's/^path=//p' "$SRC/config/$pin.config")" 2>/dev/null
            done | xargs -r -n1 basename | grep -v -x -E 'README.md|NOTICE|LICENSE' | LC_ALL=C sort -u)
    n=$(find "$t" -type f -printf '%f\n' | LC_ALL=C sort -u | LC_ALL=C comm -12 - <(echo "$names") | tee >(cat >&2) | wc -l)
    [ "$n" = 0 ] || { echo "FAIL files named like the private sources in the tarball"; bad=1; }
    [ $bad = 0 ] && echo "ok   no private source in $(basename "$1")"
    # The NPU kernels, when the package carries them (WEEKLY_NPU_KERNELS): every kernel file is
    # sealed (never a plaintext ELF), no ELF in the tarball is for anything but the host (an NPU
    # kernel would be), the shipped 1bit has no symbol table, and its strings name no private
    # source file and no build-machine home.
    local nk=0 m privsrc
    if [ "${WEEKLY_NPU_KERNELS:-1}" = 1 ]; then
        n=$(find "$t" -path '*/npu/kernels/*/layout.json' -type f | wc -l)
        [ "$n" -ge 3 ] || { echo "FAIL WEEKLY_NPU_KERNELS=1 but $(basename "$1") carries $n NPU kernel sets (want the 0.6B, 1.7B and 4B sets)"; bad=1; }
    fi
    # The shipped npu tree is an exact allowlist: <set>/layout.json and <set>/layer1_*/layer1.elf.
    # Anything else is the kernel method in the clear - layer1.work/ carries plaintext MLIR, object
    # files and intermediate ELFs - so a stray file is a leak, not just a wrong shape.
    while IFS= read -r -d '' f; do
        rel=${f#"$t"/}
        if [ "$(basename "$f")" = layout.json ]; then continue; fi
        if [ "$(head -c 8 "$f")" != 1BPKCRYP ]; then
            echo "FAIL leaked build artifact in the package (not a sealed kernel): $rel"; nk=$((nk + 1))
        elif [[ ! "$rel" =~ ^[^/]+/npu/kernels/[^/]+/layer1_[^/]+/layer1\.elf$ ]]; then
            echo "FAIL unexpected file under npu/kernels (a set ships layout.json + layer1_*/layer1.elf only): $rel"; nk=$((nk + 1))
        fi
    done < <(find "$t" -path '*/npu/kernels/*' -type f -print0)
    [ "$nk" = 0 ] || bad=1
    n=$(find "$t" -path '*/npu/kernels/*' -type f | wc -l)
    [ "$n" = 0 ] || [ "$nk" != 0 ] || echo "ok   NPU kernels: $n files under npu/kernels, all sealed"
    nk=0
    while IFS= read -r -d '' f; do
        pack_is_elf "$f" || continue
        m=$(od -An -t u2 -j 18 -N 2 "$f" | tr -d ' ')
        [ "$m" = 62 ] || { echo "FAIL $f is an ELF for machine $m, not the host (x86-64)"; nk=$((nk + 1)); }
    done < <(find "$t" -type f -print0)
    [ "$nk" = 0 ] && echo "ok   every ELF in $(basename "$1") is a host (x86-64) program or library" || bad=1
    f=$(find "$t" -maxdepth 2 -type f -name 1bit | head -1)
    if [ -n "$f" ]; then
        local b1=0
        n=$(nm --defined-only "$f" 2>/dev/null | wc -l)
        [ "$n" = 0 ] || { echo "FAIL the packed 1bit still has a symbol table ($n symbols)"; b1=1; }
        privsrc=$(for pin in npu-runtime npu-kernels; do
                      [ -d "$W/$pin" ] || continue
                      git -C "$W/$pin" ls-files -- "$(sed -n 's/^path=//p' "$SRC/config/$pin.config")" 2>/dev/null
                  done | grep -E '\.(c|cc|cpp|cxx|h|hpp|py)$' | xargs -r -n1 basename | LC_ALL=C sort -u)
        n=0; [ -z "$privsrc" ] || n=$(strings -a "$f" | grep -c -F -f <(printf '%s\n' "$privsrc") || true)
        [ "$n" = 0 ] || { echo "FAIL the packed 1bit names a private source file ($n strings)"; b1=1; }
        n=$(strings -a "$f" | grep -c '/home/' || true)
        [ "$n" = 0 ] || { echo "FAIL the packed 1bit names a build-machine home ($n strings)"; b1=1; }
        if [ $b1 = 0 ]; then echo "ok   the packed 1bit: no symbol table, no private source names, no build-machine paths"; else bad=1; fi
    fi
    for f in LICENSE NOTICE LICENSE-BINARY NOTICE-BINARY; do
        n=$(find "$t" -maxdepth 2 -type f -name "$f" | wc -l)
        [ "$n" != 0 ] || { echo "FAIL $f is not in $(basename "$1")"; bad=1; }
    done
    # every library each program loads is packed beside it (scripts/pack-elf.sh), except the TheRock
    # runtime the package documents (HIP, hipBLAS(Lt), rocBLAS, HSA); ldd runs with
    # ROCR_VISIBLE_DEVICES=-1, so nothing touches a GPU
    links_ok "$t" "$(basename "$1")" || bad=1
    local missing; missing=$(pack_missing_libs "$t")
    if [ -z "$missing" ]; then echo "ok   every library is packed (ldd: nothing missing beyond TheRock's runtime)"
    else echo "FAIL libraries ldd cannot find in $(basename "$1"):"; echo "$missing"; bad=1; fi
    rm -rf "$t"
    return $bad
}

# every symlink in a package stays inside it (scripts/pack-elf.sh pack_outside_links): no absolute
# link and none that resolves outside the tree or dangles; the .so -> .so.0 links in
# hrx/llama/bin are relative and stay. Lemonade's web app linked its KaTeX fonts to the build
# machine's /usr/share/fonts/truetype/katex (2026-10-21 rehearsal)
links_ok() {  # links_ok <unpacked dir> <name>
    local out; out=$(pack_outside_links "$1")
    if [ -z "$out" ]; then echo "ok   every symlink in $2 resolves inside it"; return 0; fi
    echo "FAIL symlinks in $2 that are absolute, point outside it or dangle:"; echo "$out"; return 1
}
# the Lemonade + onebit tarball: its links stay inside it, and it carries Lemonade's web app (the UI
# lemond serves at /, built with LEMONADE_WEB_APP=1), as 1bit OS does
lemonade_package_checks() {  # lemonade_package_checks <tarball>
    local t rc=0 n
    t=$(mktemp -d)
    tar -C "$t" -xf "$1" --use-compress-program=unzstd
    links_ok "$t" "$(basename "$1")" || rc=1
    n=$(find "$t" -path '*/bin/resources/web-app/index.html' -type f | wc -l)
    if [ "$n" != 0 ]; then echo "ok   Lemonade's web app: bin/resources/web-app/index.html and $(find "$t" -path '*/bin/resources/web-app/*' -type f | wc -l) files in $(basename "$1")"
    else echo "FAIL no Lemonade web app (bin/resources/web-app/index.html) in $(basename "$1")"; rc=1; fi
    n=$(find "$t" -maxdepth 2 -name lemond-onebit -type f -exec grep -c '^export ONEBIT_KCRYPT_REQUIRE_SEALED=1$' {} + 2>/dev/null | head -1)
    if [ "${n:-0}" != 0 ]; then echo "ok   lemond-onebit loads sealed NPU kernels only (ONEBIT_KCRYPT_REQUIRE_SEALED=1)"
    else echo "FAIL lemond-onebit does not export ONEBIT_KCRYPT_REQUIRE_SEALED=1"; rc=1; fi
    rm -rf "$t"
    return $rc
}

TAG_FILE=$W/tag
stage_package() {
    local tag=${WEEKLY_TAG:-v$(date +%G.%V)} n=0
    while gh release view "$tag" -R "$REPO" > /dev/null 2>&1; do n=$((n + 1)); tag=v$(date +%G.%V).$n; done
    echo "$tag" > "$TAG_FILE"
    # the image's licences step copies from the submodules of $SRC (checked out by build), not
    # from the tree this script runs from
    [ -f "$SRC/third_party/lemonade/LICENSE" ] ||
        { say "no $SRC/third_party/lemonade/LICENSE: run the build stage first (it checks out the submodules)"; return 1; }
    rm -rf "$OUT" && mkdir -p "$OUT"
    say "package $tag: the engine (into the Lemonade + onebit package)"
    # the engine tree (package-linux.sh) is not published on its own: it goes into the Lemonade
    # package below as 1bit/ (docs/releases.md), next to the lemond-onebit launcher
    rm -rf "$W/engine-pkg"
    "$SRC/scripts/package-linux.sh" "$BUILD" "$W/engine-pkg" "$tag" > "$LOGS/package-linux.log" 2>&1
    package_checks "$W/engine-pkg/1bit-$tag-linux-x86_64.tar.zst" > "$LOGS/package-checks.log" 2>&1 ||
        { cat "$LOGS/package-checks.log"; return 1; }
    say "package $tag: Lemonade + onebit, with the engine"
    # with Lemonade's web app (node and npm, preflight), the UI lemond serves at /
    LEMONADE_WEB_APP=1 guard "$SRC/scripts/build-lemonade.sh" "$W/lemonade" > "$LOGS/lemonade-build.log" 2>&1
    rm -rf "$W/lemonade/1bit" && mkdir -p "$W/lemonade/1bit"
    zstd -dc "$W/engine-pkg/1bit-$tag-linux-x86_64.tar.zst" | tar -C "$W/lemonade/1bit" --strip-components=1 -xf -
    install -m 755 "$SRC/scripts/lemond-onebit.sh" "$W/lemonade/lemond-onebit"
    # links out of the tree (the web app's KaTeX fonts -> the build machine's fonts-katex) become
    # copies of their targets; the links inside it stay links
    pack_localize_links "$W/lemonade/bin"
    # the S flag: member names only; GNU tar otherwise rewrites symlink targets too, and every
    # libggml*.so.0 -> libggml*.so.0.18.0 link in 1bit/hrx/llama/bin pointed into a directory that is not there
    tar -C "$W/lemonade" --transform "s,^,lemonade-onebit-$tag/,S" -cf - bin 1bit lemond-onebit | zstd -q -T4 -19 \
        -o "$OUT/lemonade-onebit-$tag-linux-x86_64.tar.zst"
    lemonade_package_checks "$OUT/lemonade-onebit-$tag-linux-x86_64.tar.zst" >> "$LOGS/package-checks.log" 2>&1 ||
        { cat "$LOGS/package-checks.log"; return 1; }
    say "package $tag: 1bit OS"
    rm -rf "$W/os"
    guard "$SRC/os/mini/mkimage.sh" "$W/os" "$BUILD" > "$LOGS/os.log" 2>&1
    zstd -q -T4 -19 "$W/os/1bit-os.img" -o "$OUT/1bit-os-$tag.img.zst"
    cp "$W/os/1bit-os.efi" "$OUT/1bit-os-$tag.efi"
    # SHA256SUMS must not list itself: it is written after the hashes are taken, so its own line can
    # never match and "sha256sum -c SHA256SUMS" would always report one failure.
    (cd "$OUT" && sha256sum $(ls | grep -vx SHA256SUMS) > SHA256SUMS)
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
    # stages run one at a time (docs/releases.md): without the bumps stage there is no bumps.json
    [ -f "$W/bumps.json" ] || echo "[]" > "$W/bumps.json"
    say "release $tag: changes ${from:0:12}..$(git -C "$SRC" rev-parse --short HEAD) (previous: ${prev:-none})"
    python3 "$SRC/tools/weekly_changes.py" --src "$SRC" --from "$from" --to HEAD --tag "$tag" \
        --previous "$prev" --bumps "$W/bumps.json" --json "$OUT/changes.json" --md "$W/notes.md"
    # the checksums cover changes.json too (package wrote them before it existed)
    (cd "$OUT" && rm -f SHA256SUMS && sha256sum -- * > SHA256SUMS)
    if [ "$DRY" = 1 ]; then say "dry run: not publishing"; return; fi
    gh release create "$tag" -R "$REPO" --target "$(git -C "$SRC" rev-parse HEAD)" \
        --title "1bit engine $tag" -F "$W/notes.md" "$OUT"/*
    say "released $tag"
}

# --- preflight ----------------------------------------------------------------------------
# What the stages need from the machine, checked before the first one runs, so that a missing
# tool stops the run at the start rather than after an hour of build (2026-10-19 rehearsal).
preflight() {
    local bad=0 s rocm d kver vmlinuz
    for s in $STAGES; do
        case "$s" in
            bumps|build)
                # the HRX build's toolchain (cmake/hrx.cmake), unless WEEKLY_CMAKE_ARGS names another
                case " ${WEEKLY_CMAKE_ARGS:-} " in
                    *ONEBIT_HRX_TOOLCHAIN=*) ;;
                    *) [ -d /opt/rocm-therock ] ||
                        { echo "preflight: no /opt/rocm-therock (the HRX toolchain): install TheRock there, or pass -DONEBIT_HRX_TOOLCHAIN=<dir> in WEEKLY_CMAKE_ARGS"; bad=1; } ;;
                esac
                # DwarfStar (-DONEBIT_DS4=ON) builds with hipcc, found the way scripts/build-ds4.sh finds it
                rocm=${ROCM_PATH:-}
                if [ -z "$rocm" ]; then
                    for d in /opt/rocm-therock/lib/python3*/site-packages/_rocm_sdk_devel /opt/rocm; do
                        [ -x "$d/bin/hipcc" ] && { rocm=$d; break; }
                    done
                fi
                [ -n "$rocm" ] && [ -x "$rocm/bin/hipcc" ] ||
                    { echo "preflight: no ROCm with bin/hipcc for the DwarfStar build (scripts/build-ds4.sh): set ROCM_PATH"; bad=1; } ;;
            package)
                # scripts/package-linux.sh rewrites RPATHs with patchelf
                command -v patchelf > /dev/null ||
                    { echo "preflight: patchelf is not installed (scripts/package-linux.sh needs it): apt install patchelf"; bad=1; }
                # Lemonade's web app (LEMONADE_WEB_APP=1 scripts/build-lemonade.sh): node and npm, and
                # the network for its `npm ci`
                for d in node npm; do
                    command -v "$d" > /dev/null ||
                        { echo "preflight: no $d on PATH (Lemonade's web app in the Lemonade + onebit package): install Node.js with npm"; bad=1; }
                done
                # os/mini/mkimage.sh reads the kernel image; on Ubuntu /boot/vmlinuz-* is root-only
                kver=${KVER:-$(uname -r)}
                vmlinuz=${VMLINUZ:-/boot/vmlinuz-$kver}
                [ -r "$vmlinuz" ] ||
                    { echo "preflight: cannot read the kernel image $vmlinuz (os/mini/mkimage.sh): set VMLINUZ to a readable copy"; bad=1; } ;;
        esac
    done
    return $bad
}
preflight || { say "preflight failed: nothing was run"; exit 1; }

for s in $STAGES; do "stage_$s"; done
