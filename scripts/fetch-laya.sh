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
# fetch-laya.sh [<dir>]
#
# Downloads the Laya checkpoints pinned in config/laya.json (one Hugging Face
# revision holding all three: root, multilingual/, typed-decisions/) into
# <dir> (default: ${XDG_DATA_HOME:-~/.local/share}/1bit/laya, where `1bit serve --laya`
# looks), and checks every file against the Hub's list for that revision:
# sha256 for LFS files (the weights), the git blob id for the small ones.
# Images and the eval plots are skipped. Files already present and correct are
# not downloaded again. The pinned ggmlc GGUF of the typed-decisions checkpoint
# (the router's fast scorer on HRX) goes to <dir>/gguf/, checked the same way.
set -euo pipefail
dir=${1:-${XDG_DATA_HOME:-$HOME/.local/share}/1bit/laya}
echo "fetch-laya: into $dir"
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$dir"
python3 - "$root/config/laya.json" "$dir" <<'PYEOF'
import hashlib, json, os, sys, urllib.request

pin = json.load(open(sys.argv[1]))

def digest(path, lfs):
    data_h = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:  # git blob id = sha1("blob <size>\0" + contents)
        data_h.update(f"blob {os.path.getsize(path)}\0".encode())
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            data_h.update(chunk)
    return data_h.hexdigest()

def fetch(repo, rev, out, keep):
    api = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
    files = [f for f in json.load(urllib.request.urlopen(api)) if f["type"] == "file" and keep(f["path"])]
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

fetch(pin["repo"], pin["revision"], sys.argv[2], lambda p: not p.startswith(("assets/", "eval/")))
if "gguf" in pin:  # the ggmlc GGUF of the typed-decisions checkpoint (docs/laya.md)
    g = pin["gguf"]
    fetch(g["repo"], g["revision"], os.path.join(sys.argv[2], "gguf"), lambda p: p in g["files"])
PYEOF
