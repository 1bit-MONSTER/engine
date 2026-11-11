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


# 8. the naming scheme, which is the other half of a binary/set mismatch (engine#2106): a current-main plan asks for
#    role-named full_i8_*.elf, a pre-change set carries layerN_M/layer1.elf, and a set with both is refused
mk_named() {  # mk_named <dir> <set> <role-count> <layer-count>
    mkdir -p "$1"; python3 - "$1" "$2" "$3" "$4" <<'PY2'
import hashlib, json, os, sys
root, name, nrole, nlayer = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
files = []
def put(rel):
    full = os.path.join(root, rel); os.makedirs(os.path.dirname(full) or root, exist_ok=True)
    body = b"\x7fELF" + rel.encode()
    open(full, "wb").write(body)
    files.append({"path": rel, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()})
for i in range(nrole):
    put(f"full_i8_QKV_K1024_N{4096 + i}.elf")
for i in range(nlayer):
    put(f"layer1_{100 + i}/layer1.elf")
json.dump({"format": 1, "set": name, "files": files}, open(os.path.join(root, "manifest.json"), "w"))
PY2
}
mk_named "$tmp/n-role/1024x3072x16x8" 1024x3072x16x8 2 0
"$script" "$tmp/n-role/1024x3072x16x8" > "$tmp/out11" 2>&1
check $([ $? -eq 0 ] && echo ok || echo fail) "role-named set: rc 0"
check $(has "$tmp/out11" "2 role-named full_i8_*.elf" && echo ok || echo fail) "... and the naming is named"

mk_named "$tmp/n-layer/1024x3072x16x8" 1024x3072x16x8 0 3
"$script" "$tmp/n-layer/1024x3072x16x8" > "$tmp/out12" 2>&1
check $([ $? -eq 0 ] && echo ok || echo fail) "layer-indexed set: rc 0 on its own"
check $(has "$tmp/out12" "3 layer-indexed" && echo ok || echo fail) "... and the naming is named"

mk_named "$tmp/n-mixed/1024x3072x16x8" 1024x3072x16x8 1 2
"$script" "$tmp/n-mixed/1024x3072x16x8" > "$tmp/out13" 2>&1
check $([ $? -eq 1 ] && echo ok || echo fail) "a set with BOTH naming schemes: rc 1"
check $(has "$tmp/out13" "MIXED NAMING" && echo ok || echo fail) "... and the reason is named"


# 9. engine#2128: a pack that names the wide/narrow sealed sets must name files that are there
mk_pack() {  # mk_pack <dir> <config-or-plan: config|plan> <key> <value> [<key2> <value2>]
    mkdir -p "$1"
    python3 - "$@" <<'PYP'
import json, os, sys
root, kind, *pairs = sys.argv[1:]
if kind == "config":
    body = {pairs[i]: pairs[i + 1] for i in range(0, len(pairs), 2)}
else:
    body = {"prefill": {}, "decode": {}}
    for i in range(0, len(pairs), 2):
        key, value = pairs[i], pairs[i + 1]
        body["prefill" if key == "sealed_elf" else "decode"][key] = value
json.dump(body, open(os.path.join(root, kind + ".json"), "w"))
PYP
}
mk_set "$tmp/named/1024x3072x16x8" 1024x3072x16x8 1 elf
ln -sf "$tmp/named/1024x3072x16x8/layer1_100/layer1.elf" "$tmp/named/1024x3072x16x8/wide32.elf" 2>/dev/null || true
mk_pack "$tmp/pack-named" config q4nx_t2s_sealed_wide_elf wide32.elf
"$script" --pack "$tmp/pack-named" "$tmp/named/1024x3072x16x8" > "$tmp/out14" 2>&1
check $(has "$tmp/out14" "wide set wide32.elf named in config.json is present" && echo ok || echo fail) \
      "a pack naming a wide set that exists: named and present"
mk_pack "$tmp/pack-missing" config q4nx_t2s_sealed_wide_elf absent32.elf
"$script" --pack "$tmp/pack-missing" "$tmp/named/1024x3072x16x8" > "$tmp/out15" 2>&1
check $([ $? -eq 1 ] && echo ok || echo fail) "a pack naming a set that is ABSENT: rc 1"
check $(has "$tmp/out15" "stay on the 16-row set" && echo ok || echo fail) "... and says what the serve would do instead"
mk_pack "$tmp/pack-plan" plan sealed_elf wide32.elf narrow_elf gone4.elf
"$script" --pack "$tmp/pack-plan" "$tmp/named/1024x3072x16x8" > "$tmp/out16" 2>&1
check $([ $? -eq 1 ] && echo ok || echo fail) "naming from plan.json is read too (the narrow file is absent)"
check $(has "$tmp/out16" "narrow set named in plan.json" && echo ok || echo fail) "... and the missing one is named"
mk_pack "$tmp/pack-none" config unused 1
"$script" --pack "$tmp/pack-none" "$tmp/named/1024x3072x16x8" > "$tmp/out17" 2>&1
check $(has "$tmp/out17" "names no wide/narrow set (16-row set only)" && echo ok || echo fail) \
      "a pack naming nothing: said explicitly, not silently"

# 10. a manifest lists more than kernels: every shipped one lists layout.json. Only .elf/.bin entries have a form, so a
#     correct plain-ELF set that lists its layout.json is one form, not two.
mk_set "$tmp/set-layout/1024x3072x16x8" 1024x3072x16x8 3 elf
python3 - "$tmp/set-layout/1024x3072x16x8" <<'PYL'
import hashlib, json, os, sys
root = sys.argv[1]
body = b'{"layers": 3}\n'
open(os.path.join(root, "layout.json"), "wb").write(body)
path = os.path.join(root, "manifest.json")
manifest = json.load(open(path))
manifest["files"].append({"path": "layout.json", "size": len(body), "sha256": hashlib.sha256(body).hexdigest()})
json.dump(manifest, open(path, "w"))
PYL
"$script" "$tmp/set-layout/1024x3072x16x8" > "$tmp/out18" 2>&1
check $([ $? -eq 0 ] && echo ok || echo fail) "a plain-ELF set that lists its layout.json: rc 0"
check $(has "$tmp/out18" "3 file(s) in plain ELF form" && ! has "$tmp/out18" "MIXED FORMS" && echo ok || echo fail) \
      "... three kernels in one form, the JSON not counted"
# ... and the listed JSON is still held to its size and sha256 like any other entry
printf 'x' >> "$tmp/set-layout/1024x3072x16x8/layout.json"
"$script" "$tmp/set-layout/1024x3072x16x8" > "$tmp/out19" 2>&1
check $([ $? -eq 1 ] && has "$tmp/out19" "layout.json: manifest says" && echo ok || echo fail) \
      "... but a listed layout.json that changed is still caught: rc 1"

# 11. a layer directory with a suffix after its number (layer1_100np/) is still layer-indexed, not "other"
mkdir -p "$tmp/n-suffix/1024x3072x16x8"
python3 - "$tmp/n-suffix/1024x3072x16x8" <<'PYS'
import hashlib, json, os, sys
root, files = sys.argv[1], []
for rel in ("layer1_100/layer1.elf", "layer1_101np/layer1.elf", "layer2_102np/layer2.elf"):
    full = os.path.join(root, rel); os.makedirs(os.path.dirname(full), exist_ok=True)
    body = b"\x7fELF" + rel.encode()
    open(full, "wb").write(body)
    files.append({"path": rel, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()})
json.dump({"format": 1, "set": "1024x3072x16x8", "files": files}, open(os.path.join(root, "manifest.json"), "w"))
PYS
"$script" "$tmp/n-suffix/1024x3072x16x8" > "$tmp/out20" 2>&1
check $([ $? -eq 0 ] && echo ok || echo fail) "layer directories with a suffix: rc 0"
check $(has "$tmp/out20" "3 layer-indexed" && ! has "$tmp/out20" "other" && echo ok || echo fail) \
      "... all three counted as layer-indexed, none as other"

if [ "$fails" -ne 0 ]; then
    printf 'kernel_set_check: %d failure(s)\n' "$fails"
    exit 1
fi
printf 'kernel_set_check: all cases ok\n'
