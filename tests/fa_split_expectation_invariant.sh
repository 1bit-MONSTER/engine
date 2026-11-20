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
# Runs the FA decode-split invariant against the fork, mapping the tool's "cannot tell" to ctest's skip.
#
# engine#2925 asked for `add_test(… --check …)` + SKIP_RETURN_CODE 77, and those two together do NOT skip:
# SKIP_RETURN_CODE 77 means *77 is a skip*, and the tool returns 2 for "cannot tell" (its own documented contract,
# with 1 for a violation). So the row as proposed would go RED on any tree without the submodule -- including CI if
# the submodule fetch fails, which is exactly when a pass would be least trustworthy. This wrapper keeps the tool's
# contract and maps 2 -> 77, so "I could not look" can never read as "the invariant holds".

set -uo pipefail
tool=${1:?usage: fa_split_expectation_invariant.sh <tool.py> <llama.cpp checkout>}
root=${2:?usage: fa_split_expectation_invariant.sh <tool.py> <llama.cpp checkout>}

python3 "$tool" --check --root "$root"
rc=$?
if [ "$rc" -eq 2 ]; then
    echo "fa_split_expectation_invariant: SKIP -- the fork is not checked out here, so the invariant cannot be evaluated" >&2
    exit 77
fi
exit "$rc"
