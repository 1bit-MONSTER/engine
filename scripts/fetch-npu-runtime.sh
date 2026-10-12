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
# fetch-npu-runtime.sh <dest> [pin file]
#
# Checks out the private NPU runtime named by config/npu-runtime.config (repo, sha, path) into
# <dest> and prints the runtime directory to pass as -DONEBIT_NPU_RT_DIR (docs/releases.md). Only
# that path at that commit is fetched (a shallow, sparse, blob-filtered checkout of one commit).
# Credentials come from gh (`gh auth git-credential`), so whoever runs it needs read access to the
# private repository; no token is printed or written. Exits non-zero, and the release stops, when
# the commit cannot be fetched, the checkout is not at the pinned sha, or npu_rt.h / npu_rt.cpp
# are missing. A <dest> already at the pin is reused.
set -euo pipefail

dest=${1:?usage: fetch-npu-runtime.sh <dest> [pin file]}
src=$(cd "$(dirname "$0")/.." && pwd)
pinfile=${2:-$src/config/npu-runtime.config}
val() { sed -n "s/^$1=//p" "$pinfile" | tail -1; }
repo=$(val repo) sha=$(val sha) path=$(val path)
[[ "$repo" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || { echo "fetch-npu-runtime: no valid repo= in $pinfile" >&2; exit 1; }
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo "fetch-npu-runtime: sha= in $pinfile is not a full commit id" >&2; exit 1; }
[[ -n "$path" && "$path" != /* && "$path" != *..* ]] || { echo "fetch-npu-runtime: no valid path= in $pinfile" >&2; exit 1; }

git_() { git -C "$dest" -c credential.helper= -c 'credential.helper=!gh auth git-credential' "$@"; }
if [ "$(git -C "$dest" rev-parse HEAD 2>/dev/null || true)" != "$sha" ]; then
    rm -rf "$dest"
    mkdir -p "$dest"
    git -C "$dest" init -q
    git_ remote add origin "https://github.com/$repo.git"
    git_ sparse-checkout set --no-cone "/$path/" > /dev/null
    if ! git_ fetch -q --depth 1 --filter=blob:none origin "$sha"; then
        echo "fetch-npu-runtime: cannot fetch $repo $sha (gh logged in with access to it?)" >&2; exit 1
    fi
    git_ checkout -q --detach FETCH_HEAD
fi
head=$(git -C "$dest" rev-parse HEAD)
[ "$head" = "$sha" ] || { echo "fetch-npu-runtime: checkout is at $head, the pin is $sha" >&2; exit 1; }
rt=$dest/$path
for f in npu_rt.h npu_rt.cpp; do
    [ -f "$rt/$f" ] || { echo "fetch-npu-runtime: $repo $sha has no $path/$f" >&2; exit 1; }
done
echo "$rt"
