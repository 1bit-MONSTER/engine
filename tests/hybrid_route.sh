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
# `1bit serve --device hybrid` (docs/hybrid.md) without a model or an NPU: tests/fake_backend.py
# stands in for llama-server (it answers /apply-template, /tokenize and the slot restore) and
# tests/fake_prefill_provider.py for the NPU engine (FAKE_PREFILL_COPY=1: the state file is the ids
# file). Checks the handoff order and what reaches the backend: the request pinned to slot 0 with
# cache_prompt on, the backend's --slot-save-path, the per-request stderr line, and the fallback
# (a provider that fails leaves the request to the backend, which still answers).
#
# usage: tests/hybrid_route.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: hybrid_route.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
touch "$scratch/tiny.gguf"
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
pid=
cleanup() { [ -n "$pid" ] && { kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; }; rm -rf "$scratch"; }
trap cleanup EXIT

start() {   # start <log> [env...]: serve on a free port with the fake backend, env applied to serve
    local log=$1; shift
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    env FAKE_SLOT_MARKER="$scratch/slot.json" "$@" "$bin" serve -m "$scratch/tiny.gguf" --device hybrid --port "$port" --alias hybrid-model \
        --llama-server "$here/fake_backend.py" --no-mem-check >"$log" 2>&1 &
    pid=$!
    api="http://127.0.0.1:$port"
    # A deadline in seconds, not a try count: the old 100-try loop took 6.7 s on a quiet box and 15.2 s on the
    # same commit on a loaded CI runner, so a slow runner exhausted it and four checks failed as if routing had
    # broken (engine #850).
    local code=000 deadline=$((SECONDS + ${HYBRID_ROUTE_START_S:-60}))
    while :; do
        code=$(curl -s -o /dev/null -w '%{http_code}' "$api/health")
        [ "$code" = 200 ] && break
        [ $SECONDS -lt $deadline ] || break
        sleep 0.1
    done
    if [ "$code" != 200 ]; then
        echo "FAIL serve did not answer $api/health within $((SECONDS - deadline + ${HYBRID_ROUTE_START_S:-60}))s: last code $code, pid $pid $(kill -0 "$pid" 2>/dev/null && echo alive || echo gone)"
        tail -5 "$log" 2>/dev/null
    fi
    [ "$code" = 200 ]
}
stop() { kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; pid=; }

# 1. the handoff: provider named, the backend sees id_slot 0 and cache_prompt true
check "serve starts with the fake provider" 'start "$scratch/a.log" ONEBIT_HYBRID_PREFILL="$here/fake_prefill_provider.py" FAKE_PREFILL_COPY=1'
reply=$(curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "hybrid-model", "messages": [{"role": "user", "content": "What is the capital of France?"}]}')
check "chat reply comes through" '[[ "$reply" == *Paris* ]]'
check "request pinned to slot 0 (fake_id_slot)" '[[ "$reply" == *"\"fake_id_slot\":0"* ]]'
check "cache_prompt on (fake_cache_prompt)" '[[ "$reply" == *"\"fake_cache_prompt\":true"* ]]'
sleep 0.3
check "stderr: the per-request line (prefill, restore, first byte)" 'grep -q "hybrid: 30 prompt tokens: prefill .* ms, restore .* ms, first byte .* ms" "$scratch/a.log"'
check "stderr: the startup line names the provider" 'grep -q "hybrid: prefill by .*fake_prefill_provider.py" "$scratch/a.log"'
# engine#941: the state file holds the prompt minus its last token, and restore says so. This pins the request
# shape only - what the backend does with `leave_one_token` is fork 1bit-MONSTER/llama.cpp#189, not in the pin yet
check "the provider was handed one token less than the prompt (29 of 30)" \
    'python3 -c "import json,sys; d=json.load(open(sys.argv[1])); b=d[\"body\"]; sys.exit(0 if d[\"action\"]==\"restore\" and b.get(\"leave_one_token\") is True and b.get(\"n_tokens_in_state_file\")==29 else 1)" "$scratch/slot.json" 2>/dev/null'
reply=$(curl -s "$api/v1/completions" -H 'Content-Type: application/json' -d '{"model": "hybrid-model", "prompt": "Hello hybrid"}')
check "/v1/completions goes through the handoff too" '[[ "$reply" == *"\"fake_id_slot\":0"* ]]'
chunks=$(curl -sN "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "hybrid-model", "stream": true, "messages": [{"role": "user", "content": "hi there"}]}' | grep -c '^data: {')
check "streamed reply relayed ($chunks chunks)" '[ "$chunks" = 7 ]'
check "no handoff file left in the slot directory" '[ -z "$(ls /tmp/1bit-hybrid-*/ 2>/dev/null)" ]'
slot_dirs_before=$(ls -d /tmp/1bit-hybrid-* 2>/dev/null | wc -l)
stop
slot_dirs_after=$(ls -d /tmp/1bit-hybrid-* 2>/dev/null | wc -l)
check "the slot directory is removed at exit ($slot_dirs_before -> $slot_dirs_after)" '[ "$slot_dirs_after" -lt "$slot_dirs_before" ]'

# 2. no provider: the device serves, the backend prefills (said once)
check "serve starts without a provider" 'start "$scratch/b.log"'
reply=$(curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "hybrid-model", "messages": [{"role": "user", "content": "hi"}]}')
check "answers without a provider" '[[ "$reply" == *Paris* ]]'
check "not pinned to a slot without a provider" '[[ "$reply" == *"\"fake_id_slot\":-1"* ]]'
check "stderr: no provider, said once" '[ "$(grep -c "hybrid: no prefill provider" "$scratch/b.log")" = 1 ]'
stop

# 3. a provider that fails: the request is forwarded as it came
printf '#!/bin/sh\nexit 3\n' > "$scratch/bad_provider.sh"; chmod +x "$scratch/bad_provider.sh"
check "serve starts with a failing provider" 'start "$scratch/c.log" ONEBIT_HYBRID_PREFILL="$scratch/bad_provider.sh"'
reply=$(curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "hybrid-model", "messages": [{"role": "user", "content": "hi"}]}')
check "answers when the provider fails" '[[ "$reply" == *Paris* ]]'
check "not pinned when the provider failed" '[[ "$reply" == *"\"fake_id_slot\":-1"* ]]'
check "stderr: the failure named" 'grep -q "hybrid: provider exited 3; the backend prefills this request itself" "$scratch/c.log"'
stop

# 3b. a provider that outlives its budget: SIGTERM first, SIGKILL only after the grace window
# An NPU provider tears down by releasing its BOs and destroying its hardware context, and ~/briefs/NPU-LANE-RULE-20261104.md
# section 4 asks for >= 20 s of grace before a kill. The route used to send SIGTERM and then SIGKILL one second later,
# so the two arms below are the check that separates them: the slow provider writes its marker only if it got to finish.
cat > "$scratch/slow_provider.py" <<'PY'
#!/usr/bin/env python3
import os, signal, sys, time

def term(*_a):
    time.sleep(8.0)                     # the teardown an NPU provider needs (>= the 3 s slack a loaded box can add)
    with open(os.environ["SLOW_MARKER"], "w") as f:
        f.write("clean\n")
    sys.exit(0)

signal.signal(signal.SIGTERM, term)
time.sleep(30)
PY
chmod +x "$scratch/slow_provider.py"
rm -f "$scratch/grace.marker"
check "serve starts with a provider that ignores its budget" \
    'start "$scratch/d.log" ONEBIT_HYBRID_PREFILL="$scratch/slow_provider.py" SLOW_MARKER="$scratch/grace.marker" ONEBIT_HYBRID_PREFILL_TIMEOUT_S=1'
reply=$(curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "hybrid-model", "messages": [{"role": "user", "content": "hi"}]}')
check "a timed-out provider still leaves the request to the backend" '[[ "$reply" == *Paris* ]]'
check "stderr: the timeout names the cooperative stop" 'grep -q "hybrid: provider timed out after 1 s and stopped on SIGTERM" "$scratch/d.log"'
check "the provider got its grace window and tore down cleanly" '[ -s "$scratch/grace.marker" ]'
stop

# ...and the failure mode of that check: with the window set to 0 the same provider is killed before it can finish.
rm -f "$scratch/kill.marker"
check "serve starts with the grace window closed" \
    'start "$scratch/e.log" ONEBIT_HYBRID_PREFILL="$scratch/slow_provider.py" SLOW_MARKER="$scratch/kill.marker" ONEBIT_HYBRID_PREFILL_TIMEOUT_S=1 ONEBIT_HYBRID_PREFILL_GRACE_S=0'
reply=$(curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "hybrid-model", "messages": [{"role": "user", "content": "hi"}]}')
check "answers with the window closed too" '[[ "$reply" == *Paris* ]]'
check "stderr: the kill is named as a kill" 'grep -q "hybrid: provider timed out after 1 s and had to be killed" "$scratch/e.log"'
check "a killed provider never reached its teardown" '[ ! -e "$scratch/kill.marker" ]'
stop

# 4. refused combinations and --print-backend
out=$("$bin" serve -m "$scratch/tiny.gguf" --device hybrid --parallel 2 --llama-server "$here/fake_backend.py" 2>&1)
check "--parallel 2 is refused" '[[ "$out" == *"one slot"* ]]'
"$bin" serve -m "$scratch/tiny.gguf" --device hybrid --print-backend --llama-server "$here/fake_backend.py" >/dev/null 2>&1
check "--print-backend says the route needs serve (exit 5)" '[ $? = 5 ]'

if [ $fail -ne 0 ]; then for l in a b c d e; do echo "--- $l.log"; tail -20 "$scratch/$l.log" 2>/dev/null; done; echo FAIL; exit 1; fi
echo PASS
