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
# pin-names.sh: the names a release check compares a packed tree against, taken from the private pins.
# Sourced by scripts/make-binary-drop.sh and scripts/weekly.sh, and by tests/pin_names_test.sh.
#
#   pin_require_names   <src> <pin>...         basenames of each pin config's require= list
#   pin_tracked_names   <work> <src> <pin>...  basenames of the files each pin tracks under its path=,
#                                              without README.md / NOTICE / LICENSE
#   pin_private_sources <work> <src> <pin>...  the subset of those that are source files (.c .cc .cpp .cxx .h .hpp .py)
#
# Each prints one name per line, sorted and unique, and ALWAYS returns 0. engine#2469: these were inline
# `$(for pin ...; done | grep ... | ...)` assignments under `set -euo pipefail`, where a selector that matches
# nothing (a config with no require= line, a pin with no source files, a failing `git ls-files`) makes the
# pipeline's status non-zero. At top level that kills the script. Inside weekly.sh's package_checks, which
# runs as `package_checks ... ||`, set -e is off, so the assignment fails quietly, the name list is empty,
# and the check that follows reports "ok" having compared against nothing. "Nothing selected" is not an
# error here: it is an empty list, and the caller decides what an empty list means.

pin_require_names() {
    local src=$1 pin
    shift
    for pin in "$@"; do
        sed -n 's/^require=//p' "$src/config/$pin.config" 2>/dev/null | tr ' ' '\n' || true
    done | { grep -v '^$' || true; } | xargs -r -n1 basename | LC_ALL=C sort -u
}

pin_tracked_names() {
    local work=$1 src=$2 pin path
    shift 2
    for pin in "$@"; do
        [ -d "$work/$pin" ] || continue
        path=$(sed -n 's/^path=//p' "$src/config/$pin.config" 2>/dev/null || true)
        git -C "$work/$pin" ls-files -- "$path" 2>/dev/null || true
    done | xargs -r -n1 basename | { grep -v -x -E 'README.md|NOTICE|LICENSE' || true; } | LC_ALL=C sort -u
}

pin_private_sources() {
    pin_tracked_names "$@" | { grep -E '\.(c|cc|cpp|cxx|h|hpp|py)$' || true; }
}
