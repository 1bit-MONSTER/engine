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
# base-modes.sh <tree> [--keep <keeplist>] [--report-only]
#
# The BASE Puppy layer's modes (engine#1474): 2,176 of its paths are group/other-writable, inherited from woof-CE's build
# with the builder's umask. Nothing needs them -- the appliance runs as root -- except the eight classes in
# os/v3/puppy/base-modes-keep (tmp, var/tmp, var/run, var/lock, var/spool, var/log, run, dev, proc, sys).
#
#   base-modes.sh <tree>                    normalise and report (dirs 0755, executables 0755, other files 0644)
#   base-modes.sh <tree> --report-only      print the group/other-writable paths outside the keep-list, change nothing
#   base-modes.sh <tree> --keep <file>      use another keep-list
#
# The normaliser and the reporter are the two functions this calls in scripts/pack-elf.sh, so the row that fails a cycle
# and the cycle itself cannot disagree. Run as root (it chmods a tree it did not create).
#
# Exit: 0 done (and, with --report-only, clean), 1 dirty (--report-only found paths), 2 usage/tree, 3 normalisation failed.
set -uo pipefail
prog=$(basename "$0")
tree=''
keep=$(cd "$(dirname "$0")/.." && pwd)/os/v3/puppy/base-modes-keep
mode=normalise
while [ $# -gt 0 ]; do
    case $1 in
        -h|--help) sed -n "/^# $prog /,/^# Exit:/p" "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --keep) keep=${2:?--keep needs a file}; shift ;;
        --report-only) mode=report ;;
        /*) tree=$1 ;;
        *) echo "$prog: unknown argument '$1'" >&2; exit 2 ;;
    esac
    shift
done
[ -n "$tree" ] || { sed -n "/^# $prog /,/^# Exit:/p" "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
[ -d "$tree" ] || { echo "$prog: no such tree: $tree" >&2; exit 2; }
[ -f "$keep" ] || { echo "$prog: no keep-list: $keep" >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091  # the repository's own library
. "$here/pack-elf.sh"

if [ "$mode" = report ]; then
    dirty=$(pack_writable_base_files "$tree" "$keep")
    if [ -z "$dirty" ]; then
        echo "base-modes: clean - no group/other-writable path outside $(basename "$keep")"
        exit 0
    fi
    printf '%s\n' "$dirty" | sed -n '1,40p'
    printf 'base-modes: %s group/other-writable path(s) outside %s\n' "$(printf '%s\n' "$dirty" | wc -l)" "$(basename "$keep")"
    exit 1
fi

pack_normalise_base_modes "$tree" "$keep" || exit 3
dirty=$(pack_writable_base_files "$tree" "$keep")
if [ -n "$dirty" ]; then
    printf '%s\n' "$dirty" | sed -n '1,20p' >&2
    echo "$prog: FAILED: $(printf '%s\n' "$dirty" | wc -l) path(s) are still writable after normalising" >&2
    exit 3
fi
exit 0
