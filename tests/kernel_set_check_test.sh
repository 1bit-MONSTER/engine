#!/usr/bin/env bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# tests/kernel_set_check_test.sh -- scripts/kernel-set-check.sh against fixture sets (engine#2106). No device, no private
# repository, no real kernel bytes: every fixture is a byte string with a readable head and a manifest that describes it.

set -uo pipefail

root=$(cd -- "$(dirname -- "$0")/.." && pwd)
script="$root/scripts/kernel-set-check.sh"
fails=0
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT

check() {  # check <ok|fail> <what>
    if [ "$1" = ok ]; then
        printf 'ok   %s\n' "$2"
    else
        printf 'FAIL %s\n' "$2"
        fails=$((fails + 1))
    fi
}

has() { grep -qF -- "$2" "$1"; }

# mk_set <dir> <set-name> <count> <kind: elf|sealed|mixed> [size-delta] [sha-broken]
mk_set() {
    local dir=$1 name=$2 count=$3 kind=$4 delta=${5:-0} broken=${6:-0}
    mkdir -p "$dir"
    python3 - "$dir" "$name" "$count" "$kind" "$delta" "$broken" <<'PY'
import hashlib, json, os, sys
root, name, count, kind, delta, broken = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
files = []
for i in range(count):
    rel = f"layer1_{100 + i}/layer1.elf"
    os.makedirs(os.path.join(root, os.path.dirname(rel)), exist_ok=True)
    body = (b"\x7fELF" if (kind == "elf" or (kind == "mixed" and i % 2 == 0)) else b"SEALED01") + bytes([i]) * 64
    with open(os.path.join(root, rel), "wb") as handle:
        handle.write(body)
    digest = hashlib.sha256(body).hexdigest()
    if broken == 1 and i == 0:
        digest = "0" * 64
    files.append({"path": rel, "size": len(body) + (delta if i == 0 else 0), "sha256": digest})
with open(os.path.join(root, "manifest.json"), "w") as handle:
    json.dump({"format": 1, "set": name, "files": files}, handle, indent=1)
PY
}

# 1. a set in one form, described correctly -> PASS in both forms
mk_set "$tmp/set-elf/1024x3072x16x8" 1024x3072x16x8 3 elf
"$script" "$tmp/set-elf/1024x3072x16x8" > "$tmp/out1" 2>&1
check $([ $? -eq 0 ] && echo ok || echo fail) "consistent ELF set: rc 0"
check $(has "$tmp/out1" "3 file(s) in plain ELF form" && echo ok || echo fail) "... and it names the form and the count"

mk_set "$tmp/set-sealed/1024x3072x16x8" 1024x3072x16x8 3 sealed
"$script" "$tmp/set-sealed/1024x3072x16x8" > "$tmp/out2" 2>&1
check $([ $? -eq 0 ] && echo ok || echo fail) "consistent sealed set: rc 0"
check $(has "$tmp/out2" "sealed/opaque" && echo ok || echo fail) "... and it says sealed, not ELF"

# 2. THE CASE #2106 WAS: one set, two forms -> the state where a build compares the other side's lengths
mk_set "$tmp/set-mixed/1024x3072x16x8" 1024x3072x16x8 4 mixed
"$script" "$tmp/set-mixed/1024x3072x16x8" > "$tmp/out3" 2>&1
check $([ $? -eq 1 ] && echo ok || echo fail) "MIXED forms: rc 1"
check $(has "$tmp/out3" "MIXED FORMS" && echo ok || echo fail) "... and the reason is named"
check $(has "$tmp/out3" "differ in size" && echo ok || echo fail) "... quoting the refusal this is meant to pre-empt"

# 3. the manifest's own claims
mk_set "$tmp/set-size/1024x3072x16x8" 1024x3072x16x8 2 elf 3
"$script" "$tmp/set-size/1024x3072x16x8" > "$tmp/out4" 2>&1
check $([ $? -eq 1 ] && echo ok || echo fail) "size disagrees with the manifest: rc 1"
check $(has "$tmp/out4" "manifest says" && echo ok || echo fail) "... naming both numbers"

mk_set "$tmp/set-sha/1024x3072x16x8" 1024x3072x16x8 2 elf 0 1
"$script" "$tmp/set-sha/1024x3072x16x8" > "$tmp/out5" 2>&1
check $([ $? -eq 1 ] && echo ok || echo fail) "sha256 disagrees with the manifest: rc 1"
check $(has "$tmp/out5" "sha256 differs" && echo ok || echo fail) "... named as a sha difference"

# 4. presence both ways: listed-but-absent, and on-disk-but-unlisted
mk_set "$tmp/set-absent/1024x3072x16x8" 1024x3072x16x8 2 elf
rm -rf "$tmp/set-absent/1024x3072x16x8/layer1_101"
"$script" "$tmp/set-absent/1024x3072x16x8" > "$tmp/out6" 2>&1
check $([ $? -eq 1 ] && echo ok || echo fail) "a listed file that is absent: rc 1"
check $(has "$tmp/out6" "listed but absent" && echo ok || echo fail) "... named"

mk_set "$tmp/set-extra/1024x3072x16x8" 1024x3072x16x8 2 elf
mkdir -p "$tmp/set-extra/1024x3072x16x8/layer1_999"
printf 'SEALED01extra' > "$tmp/set-extra/1024x3072x16x8/layer1_999/layer1.elf"
"$script" "$tmp/set-extra/1024x3072x16x8" > "$tmp/out7" 2>&1
check $([ $? -eq 1 ] && echo ok || echo fail) "an ELF on disk that the manifest does not list: rc 1"
check $(has "$tmp/out7" "not in the manifest" && echo ok || echo fail) "... named (a loader may still pick it up)"

# 5. the set's own name must agree with its directory
mk_set "$tmp/set-name/2048x3072x16x8" 1024x3072x16x8 1 elf
"$script" "$tmp/set-name/2048x3072x16x8" > "$tmp/out8" 2>&1
check $([ $? -eq 1 ] && echo ok || echo fail) "a set filed under the wrong geometry name: rc 1"
check $(has "$tmp/out8" "but the directory is" && echo ok || echo fail) "... naming both"

# 6. not a set at all is a different answer from a bad set
mkdir -p "$tmp/not-a-set"
"$script" "$tmp/not-a-set" > "$tmp/out9" 2>&1
check $([ $? -eq 2 ] && echo ok || echo fail) "a directory with no manifest: rc 2 (not the same as rc 1)"
check $(has "$tmp/out9" "no manifest.json" && echo ok || echo fail) "... and it says why"

# 7. several sets in one call: the worst one decides
"$script" "$tmp/set-elf/1024x3072x16x8" "$tmp/set-mixed/1024x3072x16x8" > "$tmp/out10" 2>&1
check $([ $? -eq 1 ] && echo ok || echo fail) "two sets, one bad: rc 1"
check $(has "$tmp/out10" "OK   " && has "$tmp/out10" "MIXED FORMS" && echo ok || echo fail) \
      "... both are reported, the good one included"

if [ "$fails" -ne 0 ]; then
    printf 'kernel_set_check: %d failure(s)\n' "$fails"
    exit 1
fi
printf 'kernel_set_check: all cases ok\n'
