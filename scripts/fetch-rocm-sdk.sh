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
# fetch-rocm-sdk.sh [--check] [<pin file>] [<dest>]
#
# Bootstraps the HRX toolchain (docs/hrx.md, "The toolchain"): a venv holding TheRock's ROCm SDK
# at the version config/hrx-toolchain.config pins, installed from the nightly wheel index that
# version was published on, for the pinned GPU target. <dest> defaults to
# third_party/toolchain/rocm-sdk-<version>-<gfx> (gitignored: it is fetched, not committed). A
# <dest> already at the pin is reused, its .pin marker recording the version and target it holds,
# so the second run and every later build cost nothing. --check verifies an existing tree without
# touching the network, prints its path, and exits non-zero when it is missing or at another pin;
# scripts/weekly.sh's preflight and tests/hrx_toolchain_pin.sh use that. The path is printed on
# stdout, diagnostics on stderr, so callers can use it as a command substitution.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
me=fetch-rocm-sdk
check=0
while [ $# -gt 0 ]; do
    case "$1" in
        --check) check=1; shift ;;
        -h|--help) sed -n '16,26p' "$0"; exit 0 ;;
        --) shift; break ;;
        -*) echo "$me: unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
pinfile=${1:-$root/config/hrx-toolchain.config}
val() { sed -n "s/^$1=//p" "$pinfile" | tail -1; }
version=$(val version) gfx=$(val gfx) index=$(val index) require=$(val require)
[ -r "$pinfile" ] || { echo "$me: cannot read $pinfile" >&2; exit 1; }
[[ "$version" =~ ^[0-9][0-9A-Za-z.+]*$ ]] || { echo "$me: no valid version= in $pinfile" >&2; exit 1; }
[[ "$gfx" =~ ^gfx[0-9a-z]+$ ]] || { echo "$me: no valid gfx= in $pinfile" >&2; exit 1; }
[[ "$index" =~ ^https://[A-Za-z0-9._/-]+/$ ]] || { echo "$me: no valid index= (an https URL ending in /) in $pinfile" >&2; exit 1; }
[ -n "$require" ] || { echo "$me: no require= in $pinfile" >&2; exit 1; }
dest=${2:-$root/third_party/toolchain/rocm-sdk-$version-$gfx}
pin="$version $gfx"

# The device library (the amdgcn bitcode hipcc links for --offload-arch) is inside the SDK's pip
# tree, not under <root>/lib. It moved there in the 2026-10-31 rebuild and every HIP step died
# with "cannot find ROCm device library" until it was pointed at; a fetched toolchain carries it,
# so the build never needs HIP_DEVICE_LIB_PATH. One location, checked rather than globbed, because
# a tree that has it somewhere else is a pin this script does not know.
sdk_core() { local c; c=$(ls -d "$dest"/lib/python3*/site-packages/_rocm_sdk_core 2>/dev/null | head -1); echo "${c:-}"; }

# verify <dest>: is this tree the pinned toolchain, complete enough to build with?
verify() {    local f core
    [ "$(cat "$dest/.pin" 2>/dev/null || true)" = "$pin" ] ||
        { echo "$me: $dest is not the pinned toolchain (want '$pin', marker says '$(cat "$dest/.pin" 2>/dev/null || echo none)')" >&2; return 1; }
    for f in $require; do
        [ -e "$dest/$f" ] || { echo "$me: $dest has no $f" >&2; return 1; }
    done
    core=$(sdk_core)
    [ -n "$core" ] || { echo "$me: $dest has no _rocm_sdk_core in site-packages" >&2; return 1; }
    [ -d "$core/lib/llvm/amdgcn/bitcode" ] ||
        { echo "$me: $dest has no device library at _rocm_sdk_core/lib/llvm/amdgcn/bitcode" >&2; return 1; }
    return 0
}

if [ "$check" = 1 ]; then
    verify || { echo "$me: --check: the pinned toolchain is not in $dest (run without --check to fetch it)" >&2; exit 1; }
    echo "$dest"
    exit 0
fi
# The toolchain directory is this script's by contract, and a stale tree from an older pin is exactly what
# pi-prs found winning silently under the old glob order: whichever way we finish, the siblings go, so a bump
# leaves exactly one toolchain behind and nothing can pick the wrong one.
prune_stale() {
    local old
    for old in "$(dirname "$dest")"/rocm-sdk-*; do
        [ "$old" = "$dest" ] && continue
        [ -d "$old" ] || continue
        echo "$me: removing stale toolchain $old (the pin is '$pin')" >&2
        rm -rf "$old"
    done
}
if verify 2>/dev/null; then prune_stale; echo "$dest"; exit 0; fi

# A fresh fetch: the venv and the SDK at the pin. `rocm[...]` is the meta package, so the extras are
# the four parts the build uses (hipcc and the compilers from devel, the libraries from libraries,
# the amdgcn bitcode from device-<gfx>).
command -v python3 > /dev/null || { echo "$me: no python3 to create the venv with" >&2; exit 1; }
echo "$me: fetching rocm[libraries,devel,device-$gfx]==$version into $dest" >&2
rm -rf "$dest"
mkdir -p "$(dirname "$dest")"
python3 -m venv "$dest" || { echo "$me: python3 -m venv $dest failed (needs python3-venv)" >&2; exit 1; }
"$dest/bin/python" -m pip install --disable-pip-version-check --no-input -q \
    --index-url "$index" "rocm[libraries,devel,device-$gfx]==$version" ||
    { echo "$me: cannot install rocm[libraries,devel,device-$gfx]==$version from $index" >&2; exit 1; }
echo "$pin" > "$dest/.pin"
verify || exit 1
prune_stale
echo "$dest"
