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
# The pinned HRX toolchain (docs/hrx.md, "The toolchain"; engine#713): config/hrx-toolchain.config
# names it, scripts/fetch-rocm-sdk.sh fetches and verifies it, and cmake/hrx.cmake,
# scripts/build-ds4.sh and scripts/weekly.sh must all use that pin. No network and no SDK: the
# checks are the pin file's refusals, the completeness check against a fabricated tree, and that
# every consumer still names the pin file (a build that quietly went back to /opt/rocm-therock
# would pass every other test on the release box and fail on a fresh one).
set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
fetch=$root/scripts/fetch-rocm-sdk.sh
pin=$root/config/hrx-toolchain.config
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0
ok() { echo "ok: $*"; }
bad() { echo "FAIL: $*"; fail=1; }

# 1. the real pin file parses, and --check refuses a tree that is not there instead of guessing
out=$("$fetch" --check "$pin" "$tmp/absent" 2>&1)
[[ $? -ne 0 ]] && grep -q "not the pinned toolchain" <<< "$out" && ok "an absent toolchain is refused" ||
    bad "an absent toolchain was not refused: $out"

# 2. the pin file's own refusals: a malformed one must not reach pip
for field in version gfx index require; do
    grep -v "^$field=" "$pin" > "$tmp/pin-no-$field.config"
    if "$fetch" --check "$tmp/pin-no-$field.config" "$tmp/absent" > /dev/null 2>&1; then
        bad "a pin file without $field= was accepted"
    else
        ok "a pin file without $field= is refused"
    fi
done
sed 's/^gfx=.*/gfx=hda/' "$pin" > "$tmp/pin-bad-gfx.config"     # not a gfx target
sed 's|^index=.*|index=http://example.invalid/|' "$pin" > "$tmp/pin-bad-index.config"
for f in pin-bad-gfx pin-bad-index; do
    "$fetch" --check "$tmp/$f.config" "$tmp/absent" > /dev/null 2>&1 && bad "a bad $f was accepted" || ok "a bad $f is refused"
done

# 3. the completeness check: a tree with the marker but no device library is not the toolchain, and
#    one with the marker, the executables and the bitcode is (that is what cmake then reads)
fake=$tmp/fake-toolchain
mkdir -p "$fake/bin" "$fake/lib/python3.14/site-packages/_rocm_sdk_core/lib/llvm/amdgcn"
for b in amdclang hipcc hipconfig; do : > "$fake/bin/$b"; done
version=$(sed -n 's/^version=//p' "$pin") gfx=$(sed -n 's/^gfx=//p' "$pin")
echo "$version $gfx" > "$fake/.pin"
"$fetch" --check "$pin" "$fake" > /dev/null 2>&1 && bad "a toolchain without the device library was accepted" ||
    ok "a toolchain without the device library is refused"
mkdir -p "$fake/lib/python3.14/site-packages/_rocm_sdk_core/lib/llvm/amdgcn/bitcode"
got=$("$fetch" --check "$pin" "$fake" 2>/dev/null)
[ "$got" = "$fake" ] && ok "--check prints a complete tree at the pin" || bad "--check printed '$got', wanted $fake"
echo "$version ${gfx}x" > "$fake/.pin"                          # same tree, another pin
"$fetch" --check "$pin" "$fake" > /dev/null 2>&1 && bad "a tree at another pin was accepted" || ok "a tree at another pin is refused"

# 4. every consumer uses the pin, not the box's /opt/rocm-therock: the two build entry points default
#    to the tree copy, and the release run fetches the same pin into its own workdir (it passes the
#    path to cmake explicitly, so the checkout it builds does not have to hold it)
for f in cmake/hrx.cmake scripts/build-ds4.sh; do
    grep -q "third_party/toolchain" "$root/$f" && ok "$f defaults to the tree toolchain" ||
        bad "$f does not default to the tree toolchain (third_party/toolchain)"
done
grep -q "fetch-rocm-sdk.sh" "$root/scripts/weekly.sh" && ok "scripts/weekly.sh fetches the pinned toolchain" ||
    bad "scripts/weekly.sh does not fetch the pinned toolchain (scripts/fetch-rocm-sdk.sh)"
grep -q "config/hrx-toolchain.config" "$root/cmake/hrx.cmake" "$root/scripts/fetch-rocm-sdk.sh" "$root/scripts/weekly.sh" &&
    ok "the pin file is what cmake, the fetch and the release run name" ||
    bad "the pin file config/hrx-toolchain.config is not named by cmake, the fetch and the release run"

# 5. the release table records the pin beside the HRX pin (docs/releases.md), as the issue asks
grep -q "config/hrx-toolchain.config" "$root/docs/releases.md" && ok "docs/releases.md records the toolchain pin" ||
    bad "docs/releases.md does not record the toolchain pin"

[ "$fail" = 0 ] && echo "RESULT: PASS (pinned HRX toolchain)" || { echo "RESULT: FAIL"; exit 1; }
