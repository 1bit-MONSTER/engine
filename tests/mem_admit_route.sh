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
# `1bit serve` refuses a model the box has no memory for before any backend starts (docs/serve.md,
# "Will it fit?"), without a GPU: a GGUF header extended to a sparse 4 TiB file stands in for a
# model no box holds, tests/fake_backend.py for llama-server.
#   - refused in under 2 s, one line naming what it needs and what is free, no backend started,
#   - --no-mem-check and ONEBIT_MEM_CHECK=0 skip the check,
#   - a small model starts as before, and serve logs its estimate,
#   - --mem-check prints the estimate and exits 0 (fits) or 3 (does not).
#
# usage: tests/mem_admit_route.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: mem_admit_route.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
pids=()
cleanup() {
    for p in "${pids[@]}"; do pkill -9 -P "$p" 2>/dev/null; kill -9 "$p" 2>/dev/null; done
    wait 2>/dev/null
    rm -rf "$scratch"
}
trap cleanup EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
newport() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
export FAKE_BACKEND_EVENTS="$scratch/events"

if [ ! -r /proc/meminfo ]; then echo "SKIP: no /proc/meminfo, serve does not check memory here"; exit 0; fi

python3 - "$scratch" <<'PY'
import os, struct, sys
def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
def u32(k, v): return s(k) + struct.pack("<I", 4) + struct.pack("<I", v)
kv = [s("general.architecture") + struct.pack("<I", 8) + s("qwen3"), u32("qwen3.block_count", 28),
      u32("qwen3.context_length", 40960), u32("qwen3.embedding_length", 1024), u32("qwen3.attention.head_count", 16),
      u32("qwen3.attention.head_count_kv", 8)]
for name, size in (("huge", 4 << 40), ("small", 0)):
    p = f"{sys.argv[1]}/{name}.gguf"
    with open(p, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv))
    if size:
        os.truncate(p, size)
PY

port=$(newport)
t0=$(date +%s%N)
"$bin" serve -m "$scratch/huge.gguf" --device cpu -c 4096 --port "$port" \
    --llama-server "$here/fake_backend.py" >"$scratch/huge.log" 2>&1
rc=$?
ms=$((($(date +%s%N) - t0) / 1000000))
check "a 4 TiB model is refused (rc $rc) in under 2 s (${ms} ms)" '[ "$rc" = 1 ] && [ "$ms" -lt 2000 ]'
check "  in one line with the numbers" '[ "$(wc -l < "$scratch/huge.log")" = 1 ] && grep -q "huge does not fit: need 4096.7 GiB (huge.gguf: weights 4096.0 + KV/state 0.2 + compute 0.5) at ctx 4096, free .* GiB (MemAvailable" "$scratch/huge.log"'
check "  before any backend started" '[ ! -e "$scratch/events" ] && ! grep -q "on cpu" "$scratch/huge.log"'

for how in flag env; do
    port=$(newport)
    if [ $how = flag ]; then
        "$bin" serve -m "$scratch/huge.gguf" --device cpu --port "$port" --no-mem-check \
            --llama-server "$here/fake_backend.py" >"$scratch/skip-$how.log" 2>&1 &
    else
        ONEBIT_MEM_CHECK=0 "$bin" serve -m "$scratch/huge.gguf" --device cpu --port "$port" \
            --llama-server "$here/fake_backend.py" >"$scratch/skip-$how.log" 2>&1 &
    fi
    pid=$!; pids+=("$pid")
    code=000
    for _ in $(seq 1 100); do code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health"); [ "$code" = 200 ] && break; sleep 0.1; done
    check "--no-mem-check ($how) skips the check: the backend starts ($code)" '[ "$code" = 200 ] && ! grep -q "does not fit" "$scratch/skip-$how.log"'
    kill "$pid"; wait "$pid" 2>/dev/null
done

port=$(newport)
"$bin" serve -m "$scratch/small.gguf" --device cpu -c 4096 --port "$port" \
    --llama-server "$here/fake_backend.py" >"$scratch/small.log" 2>&1 &
pid=$!; pids+=("$pid")
code=000
for _ in $(seq 1 100); do code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health"); [ "$code" = 200 ] && break; sleep 0.1; done
check "a small model starts ($code)" '[ "$code" = 200 ]'
check "  and serve logs its estimate" 'grep -q "1bit serve: memory: need 0.7 GiB (small.gguf: weights 0.0 + KV/state 0.2 + compute 0.5) at ctx 4096, free" "$scratch/small.log"'
kill "$pid"; wait "$pid" 2>/dev/null

out=$("$bin" serve -m "$scratch/small.gguf" --device cpu -c 4096 --mem-check); rc=$?
check "--mem-check on a small model: fits, rc 0" '[ "$rc" = 0 ] && [[ "$out" == "small: fits: need 0.7 GiB"* ]]'
out=$("$bin" serve -m "$scratch/huge.gguf" --device cpu -c 4096 --mem-check); rc=$?
check "--mem-check on a huge one: does not fit, rc 3" '[ "$rc" = 3 ] && [[ "$out" == "huge: does not fit: need"* ]]'

if [ $fail -ne 0 ]; then for f in "$scratch"/*.log; do echo "--- $f"; cat "$f"; done; echo FAIL; exit 1; fi
echo PASS
