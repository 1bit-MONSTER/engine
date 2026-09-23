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
# fetch-laya.sh <dir>
#
# Downloads the Laya checkpoints pinned in config/laya.json (one Hugging Face
# revision holding all three: root, multilingual/, typed-decisions/) into
# <dir>, and checks every file against the Hub's list for that revision:
# sha256 for LFS files (the weights), the git blob id for the small ones.
# Images and the eval plots are skipped. Files already present and correct are
# not downloaded again.
set -euo pipefail
dir=${1:?usage: fetch-laya.sh <dir>}
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$dir"
python3 - "$root/config/laya.json" "$dir" <<'PYEOF'
import hashlib, json, os, sys, urllib.request

pin = json.load(open(sys.argv[1]))
repo, rev, out = pin["repo"], pin["revision"], sys.argv[2]
api = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
files = [f for f in json.load(urllib.request.urlopen(api))
         if f["type"] == "file" and not f["path"].startswith(("assets/", "eval/"))]

def digest(path, lfs):
    data_h = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:  # git blob id = sha1("blob <size>\0" + contents)
        data_h.update(f"blob {os.path.getsize(path)}\0".encode())
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            data_h.update(chunk)
    return data_h.hexdigest()

for f in files:
    lfs = f.get("lfs")
    want = lfs["oid"] if lfs else f["oid"]
    dst = os.path.join(out, f["path"])
    if os.path.exists(dst) and digest(dst, bool(lfs)) == want:
        continue
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{f['path']}"
    urllib.request.urlretrieve(url, dst + ".part")
    got = digest(dst + ".part", bool(lfs))
    if got != want:
        os.remove(dst + ".part")
        sys.exit(f"{f['path']}: hash {got} != pinned {want}")
    os.replace(dst + ".part", dst)
    print(f"fetched {f['path']} ({f['size']} B)")
print(f"Laya {repo}@{rev[:12]}: {len(files)} files verified in {out}")
PYEOF
