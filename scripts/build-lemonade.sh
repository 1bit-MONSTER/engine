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
#
# build-lemonade.sh <prefix>
#
# Builds lemond and the lemonade CLI from third_party/lemonade (our fork of
# lemonade-sdk/lemonade, with the onebit recipe; docs/lemonade.md) into
# <prefix>/bin. Needs cmake, ninja and a C++ compiler. The web app is left out unless
# LEMONADE_WEB_APP=1 (needs node and npm; 1bit OS ships it, os/README.md): lemond then serves
# Lemonade's own UI from resources/web-app.
# Run it with the engine on PATH, or point the recipe at it:
#   LEMONADE_ONEBIT_BIN=/path/to/1bit <prefix>/bin/lemond
set -euo pipefail
prefix=${1:?usage: build-lemonade.sh <prefix>}
root=$(cd "$(dirname "$0")/.." && pwd)
src=$root/third_party/lemonade
if [ ! -f "$src/CMakeLists.txt" ] && git -C "$root" rev-parse --git-dir > /dev/null 2>&1; then
    echo "fetching third_party/lemonade"
    git -C "$root" submodule update --init third_party/lemonade
fi
[ -f "$src/CMakeLists.txt" ] || { echo "third_party/lemonade is empty and could not be fetched: git submodule update --init third_party/lemonade"; exit 1; }
mkdir -p "$prefix/bin"
prefix=$(cd "$prefix" && pwd)
web=OFF; targets="lemond lemonade"
if [ "${LEMONADE_WEB_APP:-0}" = 1 ]; then web=ON; targets="$targets web-app"; fi
cmake -S "$src" -B "$prefix/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_WEB_APP=$web
cmake --build "$prefix/build" --target $targets
cp "$prefix/build/lemond" "$prefix/build/lemonade" "$prefix/bin/"
# lemond reads resources/ (defaults.json, the model list) beside itself
rm -rf "$prefix/bin/resources" && cp -r "$prefix/build/resources" "$prefix/bin/"
echo "built $prefix/bin/lemond (lemonade $(git -C "$src" rev-parse --short=12 HEAD 2>/dev/null || echo unknown))"
