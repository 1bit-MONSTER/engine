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
# One Laya checkpoint resolution for `serve --laya` and `1bit route`
# (app/laya_checkpoint.h). The chain is $ONEBIT_LAYA_MODEL, else
# ${XDG_DATA_HOME:-~/.local/share}/1bit/laya, preferring a typed-decisions/ checkpoint inside it;
# `--laya-model DIR` always wins. Two copies of that chain would drift, and the drift shows up as
# "the router routes but serve --laya cannot find a checkpoint", so this test pins it:
#   - `--laya-model DIR` resolves (the explicit form the other Laya tests use),
#   - `$ONEBIT_LAYA_MODEL` resolves without any flag,
#   - `${XDG_DATA_HOME}/1bit/laya` resolves without any flag or variable,
#   - a directory holding typed-decisions/ resolves to that subdirectory,
#   - nothing installed names the default path and says how to install it,
#   - `--laya-model` is optional in the usage text of both `route` forms.
#
# usage: tests/laya_checkpoint.sh path/to/1bit path/to/laya-model-dir
set -uo pipefail

bin=${1:?usage: laya_checkpoint.sh path/to/1bit path/to/laya-model-dir}
laya_model=${2:?usage: laya_checkpoint.sh path/to/1bit path/to/laya-model-dir}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

state="Write a Python function that reverses a linked list."
# A decision is the proof that the checkpoint was found and loaded: the scorer only loads a real one.
decides() {  # decides <env-assignments...> -- <route args...>  => exit 0 and prints a device
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    env "${envs[@]}" "$bin" route "$@" --state "$state" 2>"$scratch/err" | grep -qxE 'hrx|npu|cpu'
}

typo=$(mktemp -d); mkdir -p "$typo/1bit"
check "1bit route --laya-model DIR resolves it" \
    'decides HOME="$scratch/nohome" -- --laya-model "$laya_model"'
check "1bit route --serve --laya-model DIR resolves it" \
    'ONEBIT_LAYA_MODEL= XDG_DATA_HOME= HOME="$scratch/nohome" timeout 120 "$bin" route --serve --laya-model "$laya_model" </dev/null 2>"$scratch/err" | head -1 | grep -q "\"ready\":true"'
check "\$ONEBIT_LAYA_MODEL resolves it without a flag" \
    'decides ONEBIT_LAYA_MODEL="$laya_model" XDG_DATA_HOME= HOME="$scratch/nohome" --'
check "\${XDG_DATA_HOME}/1bit/laya resolves it without a flag" \
    'mkdir -p "$scratch/xdg/1bit" && ln -sfn "$laya_model" "$scratch/xdg/1bit/laya" && decides ONEBIT_LAYA_MODEL= XDG_DATA_HOME="$scratch/xdg" HOME="$scratch/nohome" --'
# A directory holding typed-decisions/ resolves INTO it (that checkpoint classifies best): a fake
# one makes the load fail on exactly that path, which is the proof the subdirectory was chosen.
mkdir -p "$scratch/fake/typed-decisions" && : > "$scratch/fake/typed-decisions/model.safetensors"
sub=$(ONEBIT_LAYA_MODEL= XDG_DATA_HOME= HOME="$scratch/nohome" "$bin" route --laya-model "$scratch/fake" \
        --state "$state" 2>&1 >/dev/null)
check "a directory holding typed-decisions/ resolves to that subdirectory" \
    '[[ "$sub" == *"$scratch/fake/typed-decisions/model.safetensors"* ]]'

# Nothing installed: the message must name the default path and how to install it, not fail vaguely.
missing=$(ONEBIT_LAYA_MODEL= XDG_DATA_HOME= HOME="$scratch/nohome" "$bin" route --state "$state" 2>&1 >/dev/null)
check "an absent checkpoint names the default path ($(echo "$missing" | head -c 60)...)" \
    '[[ "$missing" == *"/.local/share/1bit/laya"* && "$missing" == *fetch-laya.sh* ]]'
named=$(ONEBIT_LAYA_MODEL= XDG_DATA_HOME= HOME="$scratch/nohome" "$bin" route --laya-model "$scratch/nope" --state "$state" 2>&1 >/dev/null)
check "an absent --laya-model DIR is named back" '[[ "$named" == *"$scratch/nope"* ]]'

# The flag is optional now: the usage text must say so for both forms.
usage=$("$bin" route --help 2>&1)
check "usage shows --laya-model as optional in both forms" \
    '[[ "$usage" == *"1bit route [--laya-model <dir>]"* && "$usage" == *"1bit route --serve [--laya-model <dir>]"* ]]'

if [ "$fail" = 0 ]; then echo "PASS"; else echo "PASS/FAIL detail in the lines above"; fi
exit "$fail"
