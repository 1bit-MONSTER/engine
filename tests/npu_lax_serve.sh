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
# `1bit serve --device npu` on Qwen3.6-35B-A3B (docs/npu-lax.md, docs/serve.md): the
# model directory routes to the lax decode on full ELFs, in process. Under strace, it
# checks /health, /v1/models, a chat that answers "The capital of France is Paris.", the
# same request again (it restores the DeltaNet state snapshot taken before the prompt's last
# token and must answer the same), the chat follow-up a client sends (the history
# re-rendered: it restores the snapshot taken before the first prompt's assistant header),
# a raw prompt that extends the last one (the live cache is reused and it answers Berlin), a
# streamed chat with its usage chunk, and that no .xclbin was opened while the lax kernels'
# insts.elf were.
#
# usage: tests/npu_lax_serve.sh <1bit> <model-dir> <kernel-dir> [extra serve args]
# The NPU is shared: run it under the box's lock, e.g. flock <lockfile> ctest -R npu_lax_serve.
set -uo pipefail

bin=${1:?usage: npu_lax_serve.sh <1bit> <model-dir> <kernel-dir> [args]}
model=${2:?model dir} kernels=${3:?kernel dir}
shift 3
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
log=$(mktemp) trace=$(mktemp)
tracer=()
if command -v strace >/dev/null; then tracer=(strace -f -qq --seccomp-bpf -e trace=open,openat -o "$trace"); fi
"${tracer[@]}" "$bin" serve -m "$model" --device npu --npu-kernels "$kernels" --port "$port" --alias lax-e2e "$@" \
    >"$log" 2>&1 &
pid=$!
stop() { pkill -TERM -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; }  # 1bit, then strace
cleanup() { stop; rm -f "$log" "$trace"; }
trap cleanup EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
api="http://127.0.0.1:$port"

code=000
t0=$(date +%s.%N)
for _ in $(seq 1 600); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "$api/health")
    [ "$code" = 200 ] && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.2
done
echo "     ready after $(python3 -c "import sys; print(f'{float(sys.argv[2]) - float(sys.argv[1]):.1f}')" "$t0" "$(date +%s.%N)") s"
check "/health 200" '[ "$code" = 200 ]'
models=$(curl -s "$api/v1/models")
check "/v1/models names lax-e2e" '[[ "$models" == *lax-e2e* ]]'

body='{"model": "lax-e2e", "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
       "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": false}}'
ask() {
    local reply
    reply=$(curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
    python3 -c '
import json, sys
r = json.loads(sys.argv[1])
t, u = r.get("timings", {}), r["usage"]
print(r["choices"][0]["message"]["content"])
print("%d prompt tokens in %.0f ms, %d generated in %.0f ms" % (u["prompt_tokens"], t.get("prompt_ms", 0),
      u["completion_tokens"], t.get("predicted_ms", 0)))
print(u["prompt_tokens"], u["prompt_tokens_details"]["cached_tokens"])
' "$reply" 2>/dev/null || echo "<no content: ${reply:0:300}>"
}
reply=$(ask)
first=$(head -1 <<< "$reply")
read -r n1 _ <<< "$(tail -1 <<< "$reply")"
echo "     said: $first ($(sed -n 2p <<< "$reply"))"
check "answers \"The capital of France is Paris.\"" '[ "$first" = "The capital of France is Paris." ]'
reply=$(ask)
read -r n2 c2 <<< "$(tail -1 <<< "$reply")"
echo "     again: $(head -1 <<< "$reply") ($(sed -n 2p <<< "$reply"), $c2 cached)"
check "the same request again answers the same" '[ "$(head -1 <<< "$reply")" = "$first" ]'
check "and restores the snapshot before its last token ($c2 of $n2 cached)" '[ "${c2:-x}" = "$((n2 - 1))" ]'
field() { python3 -c 'import json,sys; r=json.loads(sys.argv[1]); print(eval(sys.argv[2]))' "$1" "$2" 2>/dev/null; }
# The follow-up a chat client sends (the history re-rendered) after that turn: it cannot
# extend the device's tokens (the answer comes back without the think block it was
# generated after), but the first turn's user message is a snapshot's prefix.
r3=$(curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' -d '{"messages": [
      {"role": "user", "content": "What is the capital of France? Answer in one sentence."},
      {"role": "assistant", "content": "The capital of France is Paris."},
      {"role": "user", "content": "And of Germany?"}], "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": false}}')
c3=$(field "$r3" 'r["usage"]["prompt_tokens_details"]["cached_tokens"]')
echo "     chat follow-up: $(field "$r3" 'r["choices"][0]["message"]["content"]') ($c3 of $(field "$r3" 'r["usage"]["prompt_tokens"]') prompt tokens cached, from $(field "$r3" 'r["timings"].get("cache")'))"
# The first prompt's user message: all of it but "<|im_start|>assistant\n<think>\n\n</think>\n\n".
check "the chat follow-up reuses the first turn's user message ($c3 cached)" '[ "${c3:-0}" -ge $((n1 - 8)) ] && [ "${c3:-0}" -lt "$n1" ]'
check "and answers Berlin" '[[ "$r3" == *Berlin* ]]'
# A raw prompt that extends the last one's tokens continues the device's conversation.
p1=$'<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
complete() {
    curl -s "$api/v1/completions" -H 'Content-Type: application/json' \
        -d "$(python3 -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1], "max_tokens": 32}))' "$1")"
}
r1=$(complete "$p1")
a1=$(field "$r1" 'r["choices"][0]["text"]')
p2="$p1$a1"$'<|im_end|>\n<|im_start|>user\nAnd of Germany?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
r2=$(complete "$p2")
cached=$(field "$r2" 'r["usage"]["prompt_tokens_details"]["cached_tokens"]')
echo "     continued: $(field "$r2" 'r["choices"][0]["text"]') ($cached of $(field "$r2" 'r["usage"]["prompt_tokens"]') prompt tokens cached)"
want=$(field "$r1" 'r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"]')
check "a prompt extending the last one reuses the device's cache (all $want tokens of the last turn)" '[ "${cached:-x}" = "$want" ]'
check "and answers Berlin" '[[ "$r2" == *Berlin* ]]'
sse=$(curl -sN "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "lax-e2e", "stream": true, "stream_options": {"include_usage": true},
         "messages": [{"role": "user", "content": "Count to five."}],
         "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": false}}')
chunks=$(grep -c '^data: {' <<< "$sse")
check "streams ($chunks chunks)" '[ "$chunks" -gt 3 ]'
check "with a usage chunk last (stream_options.include_usage)" \
    '[[ "$(grep "^data: {" <<< "$sse" | tail -1)" == *'"'"'"prompt_tokens"'"'"'* ]]'

stop
if [ ${#tracer[@]} -gt 0 ]; then
    xclbins=$(grep -c '\.xclbin' "$trace")
    elfs=$(grep -c 'insts\.elf"' "$trace")
    check "no .xclbin opened ($xclbins), lax insts.elf opened ($elfs)" '[ "$xclbins" = 0 ] && [ "$elfs" -ge 4 ]'
else
    echo "skip strace not installed: the no-xclbin check did not run"
fi

grep -E "lax decode|ready in" "$log" | sed 's/^/     /'
if [ $fail -ne 0 ]; then echo "--- serve log (tail)"; tail -30 "$log"; echo FAIL; exit 1; fi
echo PASS
