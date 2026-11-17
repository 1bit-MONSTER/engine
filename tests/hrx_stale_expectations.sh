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
# engine#2836: the fork's recorded-stale table checked from the ENGINE's suite, so it runs on every
# PR (including every pin bump, which is when a recorded-stale entry can start hiding a real
# mis-selection: llama.cpp#419 added the check and the fork has no CI job that runs it).
#
# The tool lives in the pinned submodule, so this row can only run where the submodule is checked
# out, and only at a pin at or after llama.cpp#419. Both of those are facts about the CHECKOUT, not
# about the table, so they are a skip (77) rather than a pass: a row that returned 0 with no tool
# would be the "check that cannot run reports success" defect this suite keeps removing.
#
# usage: hrx_stale_expectations.sh <repo root>
set -u
root=${1:?usage: hrx_stale_expectations.sh <repo root>}
fork=$root/third_party/llama.cpp
tool=$fork/ggml/src/ggml-hrx/tools/check_stale_expectations.py
if [ ! -f "$tool" ]; then
    printf '%s\n' "hrx_stale_expectations: skip -- no $tool (submodule not checked out, or a pin before llama.cpp#419)" >&2
    exit 77
fi
# The tool's own codes are kept: 0 every recorded name is unselectable and still exported, 1 an
# entry is wrong, 2 it cannot tell (a file it needs is missing) -- 2 stays a failure here, because at
# this point the tree exists and a missing table is drift, not an absent corpus.
exec python3 "$tool" --check --root "$fork"
