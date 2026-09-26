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
# Regenerate registry/architectures.json for the pins in HEAD and fold it into HEAD.
# The bump workflows run this right after committing a new pin, because CI's registry_pins
# test fails a pin that moved without the registry. It fetches only the three trees the
# registry reads (llama.cpp, llama.cpp-vulkan, zinc), each at its pinned commit, depth 1.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
for s in third_party/llama.cpp third_party/llama.cpp-vulkan third_party/zinc; do
  git submodule update --init --depth 1 -- "$s"
done
python3 tools/registry_build.py
if ! git diff --quiet -- registry/architectures.json; then
  git add registry/architectures.json
  git commit -q --amend --no-edit
fi
python3 tools/registry_build.py --check-pins
