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
# tools/npu_set_scheme.py: the two kernel-set schemes a package can carry, checked against fixtures built here so the
# row needs no device, no private tree and no 500 MB of kernels. The originals that made this necessary were
# "manifest.json does not read as a format 1 manifest ('utf-8' codec can't decode byte 0x80 …)" from the packager and
# package_checks demanding 1BPKCRYP of kernels that carry a signed manifest instead.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
tool="$root/tools/npu_set_scheme.py"
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=$((fail + 1)); fi; }
s=$(mktemp -d); trap 'rm -rf "$s"' EXIT

# a npumanifest 2 set: <id>/layer1.elf + a signed manifest, no layout.json (the scheme this release ships)
k="$s/k2/1024x3072x16x8"; mkdir -p "$k/38bf" "$k/5faf"
printf '\177ELFkernel-one' > "$k/38bf/layer1.elf"; printf '\177ELFkernel-two' > "$k/5faf/layer1.elf"
{ echo "npumanifest 2"; echo "set 1024x3072x16x8"; echo "key npu-v2"
  for d in 38bf 5faf; do f="$k/$d/layer1.elf"; echo "file $(sha256sum "$f" | cut -c1-64) $(wc -c <"$f" | tr -d ' ') $d/layer1.elf"; done
} > "$k/manifest"
head -c 64 /dev/zero | tr '\0' 'x' > "$k/manifest.sig"
check "detect: a signed-manifest set is kmanifest2" '[ "$(python3 "$tool" detect "$k")" = kmanifest2 ]'
check "check: the digests match and the manifest is signed" 'python3 "$tool" check "$k" >/dev/null 2>&1'
# negative: a flipped byte in one kernel is a digest mismatch, not a pass
cp -a "$k" "$s/k2bad/1024x3072x16x8" 2>/dev/null || { mkdir -p "$s/k2bad"; cp -a "$k" "$s/k2bad/1024x3072x16x8"; }
printf 'X' >> "$s/k2bad/1024x3072x16x8/5faf/layer1.elf"
check "check: a kernel that does not match its listing is refused" '! python3 "$tool" check "$s/k2bad/1024x3072x16x8" >/dev/null 2>&1'
# negative: no signature is not "signed"
cp -a "$k" "$s/k2nosig/1024x3072x16x8" 2>/dev/null || { mkdir -p "$s/k2nosig"; cp -a "$k" "$s/k2nosig/1024x3072x16x8"; }
rm -f "$s/k2nosig/1024x3072x16x8/manifest.sig"
check "check: a set without its signature is refused" '! python3 "$tool" check "$s/k2nosig/1024x3072x16x8" >/dev/null 2>&1'
# negative: a stray file is a leak
cp -a "$k" "$s/k2stray/1024x3072x16x8" 2>/dev/null || { mkdir -p "$s/k2stray"; cp -a "$k" "$s/k2stray/1024x3072x16x8"; }
printf 'mlir' > "$s/k2stray/1024x3072x16x8/38bf/layer1.work"
check "check: a file outside the scheme's allowlist is refused" '! python3 "$tool" check "$s/k2stray/1024x3072x16x8" >/dev/null 2>&1'
# stage: exactly the scheme's files, and the staged copy passes the check
python3 "$tool" stage "$k" "$s/staged" >/dev/null
check "stage: layout is optional, the manifest and kernels come along" '[ -f "$s/staged/manifest" ] && [ -f "$s/staged/manifest.sig" ] && [ -f "$s/staged/38bf/layer1.elf" ] && [ ! -e "$s/staged/layout.json" ]'
check "stage: the staged copy is itself a valid set" 'python3 "$tool" check "$s/staged" >/dev/null 2>&1'

# the older sealed scheme still reads: format-1 manifest.json listing 1BPKCRYP ciphertext
v="$s/v1/1024x3072x16x8"; mkdir -p "$v/layer1_65"
printf '1BPKCRYPciphertext' > "$v/layer1_65/layer1.elf"
h=$(sha256sum "$v/layer1_65/layer1.elf" | cut -c1-64); sz=$(wc -c <"$v/layer1_65/layer1.elf" | tr -d ' ')
printf '{"format": 1, "set": "1024x3072x16x8", "files": [{"path": "layer1_65/layer1.elf", "size": %s, "sha256": "%s"}]}\n' "$sz" "$h" > "$v/manifest.json"
printf '{}' > "$v/layout.json"; head -c 64 /dev/zero | tr '\0' 'y' > "$v/manifest.sig"
check "detect: a sealed set is sealed1 (the scheme both scripts already handled)" '[ "$(python3 "$tool" detect "$v")" = sealed1 ]'
check "check: the sealed set still verifies through the same tool" 'python3 "$tool" check "$v" >/dev/null 2>&1'
# and a directory that is not a set is cannot-tell (2), never a pass
mkdir -p "$s/nothing"; printf 'x' > "$s/nothing/file"
check "check: a directory that is not a set is cannot-tell, not a pass" '[ "$(python3 "$tool" check "$s/nothing" >/dev/null 2>&1; echo $?)" = 2 ]'

[ "$fail" = 0 ] && echo "npu_set_scheme: all checks passed" || echo "npu_set_scheme: $fail check(s) failed"
exit "$fail"
