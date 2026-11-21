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
# tests/fake_llama_announce.sh: tests/fake_backend_route.py, announcing its port the way llama-server
# does first (engine#3171). A real llama-server prints "listening on http://host:port" as soon as the
# socket binds, and every `1bit serve` in a stack tees its child's stderr on unchanged -- so behind a
# nested `1bit serve` that line reaches the OUTER serve, which is what makes a nested launch look like
# the `--llama-server llama-server` mistake by line content alone.
#
# The fake itself is silent about its port, which is why the nest looked healthy: a guard that reads
# readiness lines cannot be tested against a child that never prints one.
#
# usage: fake_llama_announce.sh <the arguments of tests/fake_backend_route.py>
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
port=0
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    [ "${args[i]}" = "--port" ] && port=${args[i + 1]:-0}
done
# stderr, not stdout: serve tees the child's stderr and nothing else, and llama-server logs its bind line there
# (SRV_INF) -- a line on stdout reaches the log but never the guard, which would make this fixture measure nothing.
echo "srv  llama_server: listening on http://127.0.0.1:${port}" >&2
# The gap between the bind and the first answer is where a real llama-server loads its model (seconds to minutes), and
# it is the window a reader of readiness lines has to judge the child in. FAKE_LLAMA_LOAD_S models it, because a fake
# that answers within a millisecond measures this script's speed instead of the guard.
sleep "${FAKE_LLAMA_LOAD_S:-0}"
exec "$here/fake_backend_route.py" "$@"
