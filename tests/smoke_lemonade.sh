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
# Smoke test: `1bit lemonade` starts, reports healthy, and serves its model
# catalog. Uses scratch cache/config dirs so a real Lemonade setup is untouched.
#
# usage: tests/smoke_lemonade.sh path/to/1bit
set -euo pipefail

bin=${1:?usage: smoke_lemonade.sh path/to/1bit}
scratch=$(mktemp -d)
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$bin" lemonade --port "$port" --no-broadcast --log-file disabled "$scratch/cache" "$scratch/config" \
    >"$scratch/server.log" 2>&1 &
pid=$!
cleanup() { kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; rm -rf "$scratch"; }
trap cleanup EXIT

fail() { echo "FAIL: $*"; echo "--- server log"; cat "$scratch/server.log"; exit 1; }

for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$port/api/v1/health" >/dev/null && break
    kill -0 "$pid" 2>/dev/null || fail "server exited during startup"
    sleep 0.5
done

health=$(curl -sf "http://127.0.0.1:$port/api/v1/health") || fail "/api/v1/health did not answer"
python3 -c 'import json,sys; assert json.loads(sys.argv[1])["status"] == "ok"' "$health" || fail "health status is not ok"
echo "ok   /api/v1/health status ok"

catalog=$(curl -sf "http://127.0.0.1:$port/api/v1/models?show_all=true") || fail "/api/v1/models did not answer"
python3 - "$catalog" <<'PY' || fail "model catalog check"
import collections, json, sys
models = json.loads(sys.argv[1])["data"]
recipes = collections.Counter(m.get("recipe") for m in models)
print("ok   catalog: %d models, recipes: %s" % (len(models), dict(recipes)))
assert len(models) > 0, "empty catalog"
assert recipes["llamacpp-hrx"] > 0, "no llamacpp-hrx models in the catalog"
PY
curl -sf "http://127.0.0.1:$port/api/v1/system-info" >/dev/null || fail "/api/v1/system-info did not answer"
echo "ok   /api/v1/system-info"
echo PASS
