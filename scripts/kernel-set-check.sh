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
# kernel-set-check.sh: is a shipped kernel set self-consistent, and in which form? (engine#2106)
#
#   scripts/kernel-set-check.sh <set dir> [<set dir> ...]
#
# Why this exists: the release candidate serves small models at 87-101 tok/s with the set that ships beside it, and a
# build of current main refused that same set with "layer ELFs: the NB 2 and 3 builds differ in size". Neither the set
# nor the kernels were wrong — the two sides came from different revisions of the private repository, one on each side of
# its protection change, so the binary read sealed files as plain ELFs and compared their lengths. That diagnosis took a
# private-repo archaeology session; the *symptom* is checkable from the set alone, without a device and without the
# private repo, which is what this script does.
#
# It answers three questions a release run should ask before a device row:
#   1. does the manifest match the bytes? (every listed file present, with the listed size and sha256, and nothing
#      listed that is absent — a set that fails this cannot be trusted to describe itself);
#   2. is the set in ONE form? (every kernel entry either starts with \x7fELF or does not; a mixed set is the exact
#      state in which one side reads the other side's files and sees "differ in size");
#   3. does the set's own name agree with its directory? (a set copied to the wrong geometry name loads and mis-answers).
#
# Exit: 0 the set is consistent, 1 it is not, 2 the argument is not a set (no manifest.json).

set -uo pipefail

fail=0

check_set() {
    local dir=${1%/} name
    name=$(basename "$dir")
    if [ ! -f "$dir/manifest.json" ]; then
        printf 'FAIL %s: no manifest.json, so this is not a set this script can judge\n' "$dir"
        return 2
    fi

    # The manifest is the set's own description: format, set name, and one entry per file with size and sha256.
    local info
    if ! info=$(python3 - "$dir" <<'PY'
import json, os, re, sys

root = sys.argv[1]
path = os.path.join(root, "manifest.json")
try:
    with open(path) as handle:
        manifest = json.load(handle)
except Exception as exc:                                    # noqa: BLE001 - any parse failure is the finding
    print(f"FAIL {path}: not readable as JSON ({exc})")
    raise SystemExit(1)

problems = []
files = manifest.get("files")
if not isinstance(files, list) or not files:
    problems.append("the manifest lists no files")
    files = []
if manifest.get("set") and manifest["set"] != os.path.basename(root):
    problems.append(f"the manifest names set {manifest['set']!r} but the directory is {os.path.basename(root)!r}")

listed = set()
forms = {}
for entry in files:
    rel = entry.get("path")
    if not rel:
        problems.append("an entry has no path")
        continue
    listed.add(rel)
    full = os.path.join(root, rel)
    if not os.path.isfile(full):
        problems.append(f"{rel}: listed but absent")
        continue
    size = os.path.getsize(full)
    if "size" in entry and size != entry["size"]:
        problems.append(f"{rel}: manifest says {entry['size']} B, the file is {size} B")
    if "sha256" in entry:
        import hashlib
        digest = hashlib.sha256()
        with open(full, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            problems.append(f"{rel}: sha256 differs from the manifest")
    with open(full, "rb") as handle:
        head = handle.read(4)
    forms.setdefault("elf" if head == b"\x7fELF" else "sealed", []).append(rel)

# Kernels the manifest forgot: a layer1.elf on disk that nothing lists is one a loader may still pick up.
for current, _, names in os.walk(root):
    for name in names:
        full = os.path.join(current, name)
        rel = os.path.relpath(full, root)
        if rel in ("manifest.json", "manifest.sig", "layout.json") or rel in listed:
            continue
        if not rel.endswith((".elf", ".bin")):
            continue
        problems.append(f"{rel}: on disk but not in the manifest")

for rel, kind in (("elf", "plain ELF"), ("sealed", "sealed/opaque")):
    if forms.get(rel):
        print(f"OK   {os.path.basename(root)}: {len(forms[rel])} file(s) in {kind} form")

# How the files are NAMED, because that is the other half of a binary/set mismatch: a current-main plan asks for a
# role-named ELF (`full_i8_QKV_K1024_N4096.elf`) while a set built before the private repo's protection change carries
# per-layer files (`layer1_2081/layer1.elf`). engine#2106 measured exactly that refusal, and it is visible from the names
# alone -- no plan, no device, and no copy of the forward's naming rule (which is why this classifies rather than predicts).
role_named = sum(1 for rel in listed if re.search(r"(^|/)full_i8_[A-Za-z0-9_]+_K\d+", rel))
layer_indexed = sum(1 for rel in listed if re.search(r"(^|/)layer\d+_\d+/", rel))
other_named = len(listed) - role_named - layer_indexed
if listed:
    kinds = [n for n, c in (("role-named full_i8_*.elf", role_named), ("layer-indexed layerN_M/…", layer_indexed),
                             ("other", other_named)) if c]
    print(f"OK   {os.path.basename(root)}: names: " + ", ".join(
        f"{c} {n}" for n, c in (("role-named full_i8_*.elf", role_named), ("layer-indexed layerN_M/…", layer_indexed),
                                ("other", other_named)) if c))
    if role_named and layer_indexed:
        problems.append("MIXED NAMING: the set has both role-named and layer-indexed files, so a plan written for one "
                        "scheme will not find the other")

if len(forms) > 1:
    problems.append(
        "MIXED FORMS: "
        + ", ".join(f"{len(v)} {k}" for k, v in sorted(forms.items()))
        + " -- a build reads one form and compares the other's lengths, which is the 'differ in size' refusal")

for problem in problems:
    print(f"FAIL {os.path.basename(root)}: {problem}")
raise SystemExit(0 if not problems else 1)
PY
    ); then
        fail=1
    fi
    printf '%s\n' "$info"

}

if [ $# -eq 0 ]; then
    printf 'usage: %s <set dir> [<set dir> ...]\n' "$0" >&2
    exit 2
fi

rc=0
for dir in "$@"; do
    check_set "$dir" || { rc=$?; [ "$rc" -gt 1 ] && rc=2; }
done

if [ "$fail" != 0 ]; then
    [ "$rc" -eq 0 ] && rc=1
fi
exit "$rc"
