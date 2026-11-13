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
# config/kernel.config names the kernel by tag, commit, release and config hash. This checks the file is whole and
# consistent with the tree, offline:
#   - every key is present, and sha / tag_object are 40 lowercase hex;
#   - the config it names exists, and its sha256 is config_sha256;
#   - the config's own LOCALVERSION and CC_VERSION_TEXT agree with release= and compiler=.
# With --box it also compares the running kernel (uname -r, /boot/config-<release>) with the pin. That half is a
# box-side check for a release, not a CI row: a CI runner boots someone else's kernel.
#
# usage: tests/kernel_pin_test.sh [repo-root] [--box]
set -uo pipefail
root=${1:-$(cd "$(dirname "$0")/.." && pwd)}
box=0; [ "${2:-}" = --box ] && box=1
pin="$root/config/kernel.config"
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
val() { sed -n "s/^$1=//p" "$pin" | head -n 1; }
for k in source tag sha tag_object release config config_sha256 compiler; do
    check "config/kernel.config names $k" '[ -n "$(val $k)" ]'
done
check "sha is a 40-hex commit" '[[ "$(val sha)" =~ ^[0-9a-f]{40}$ ]]'
check "tag_object is 40-hex" '[[ "$(val tag_object)" =~ ^[0-9a-f]{40}$ ]]'
cfg="$root/$(val config)"
check "the config of record exists ($(val config))" '[ -f "$cfg" ]'
check "its sha256 is config_sha256" '[ "$(sha256sum "$cfg" 2>/dev/null | cut -d" " -f1)" = "$(val config_sha256)" ]'
rel=$(val release)
check "release ends with the config's LOCALVERSION" '[[ "$rel" == *"$(sed -n "s/^CONFIG_LOCALVERSION=\"\(.*\)\"$/\1/p" "$cfg")" ]]'
check "release starts with the tag's version (v7.4-rc1 -> 7.4.0-rc1)" '[[ "$rel" == "$(val tag | sed -E "s/^v([0-9]+)\.([0-9]+)(-rc[0-9]+)?$/\1.\2.0\3/")"* ]]'
check "compiler is the config's CC_VERSION_TEXT" '[ "$(sed -n "s/^CONFIG_CC_VERSION_TEXT=\"\(.*\)\"$/\1/p" "$cfg")" = "$(val compiler)" ]'
if [ "$box" = 1 ]; then
    check "the running kernel is the pinned release ($(uname -r))" '[ "$(uname -r)" = "$rel" ]'
    check "/boot/config-$rel is the config of record" '[ "$(sha256sum "/boot/config-$rel" 2>/dev/null | cut -d" " -f1)" = "$(val config_sha256)" ]'
fi
exit $fail
