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
# `1bit serve --print-backend` (docs/serve.md, "One hop from Lemonade"), without a GPU: the
# llama-server serve would start, as one JSON line, for Lemonade's onebit recipe to start itself.
#   - argv names the backend, Lemonade's port, --alias and --cache-reuse; nothing else is printed on stdout,
#   - an embedding model keeps its role and gets no --cache-reuse,
#   - routes that need serve in the request path exit 5 (NPU directory, --pm, --laya, --embed),
#   - a model that does not fit exits 3, as serve does,
#   - nothing is started: the fake backend never runs,
#   - --device hrx: strict placement by default (GGML_HRX_STRICT=1 in env); --no-strict-device,
#     ONEBIT_STRICT_DEVICE=0 and a GGML_HRX_STRICT the caller set leave it out,
#   - --print-backend-env prints the same env as NAME=VALUE lines (scripts/lemonade-hrx),
#   - a GGUF with its own MTP head (<arch>.nextn_predict_layers > 0) drafts with it on hrx
#     (--spec-type draft-mtp, no -md), as Lemonade's llamacpp recipe does; --no-mtp, ONEBIT_NO_MTP=1,
#     --device cpu, --parallel 2, --embedding and a file without the key leave it out,
#   - an hrx chat plan carries "warmup" (serve's warm-up request, for Lemonade to send before the
#     model counts as loaded); ONEBIT_NO_WARMUP=1, cpu and role servers have none.
#
# usage: tests/print_backend.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: print_backend.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
export FAKE_BACKEND_EVENTS="$scratch/events"

python3 - "$scratch" <<'PY'
import os, struct, sys
def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
def u32(k, v): return s(k) + struct.pack("<I", 4) + struct.pack("<I", v)
kv = [s("general.architecture") + struct.pack("<I", 8) + s("qwen3"), u32("qwen3.block_count", 28),
      u32("qwen3.context_length", 40960), u32("qwen3.embedding_length", 1024), u32("qwen3.attention.head_count", 16),
      u32("qwen3.attention.head_count_kv", 8)]
for name, size, extra in (("huge", 4 << 40, []), ("small", 0, []), ("mtp", 0, [u32("qwen3.nextn_predict_layers", 1)])):
    p = f"{sys.argv[1]}/{name}.gguf"
    with open(p, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv) + len(extra)) + b"".join(kv + extra))
    if size:
        os.truncate(p, size)
PY

plan() { "$bin" serve -m "$scratch/small.gguf" --device cpu --llama-server "$here/fake_backend.py" "$@" --print-backend 2>"$scratch/err"; }

out=$(plan --port 43117 --alias Qwen3-4B-1bit --ctx-size 4096); rc=$?
check "a .gguf prints its plan (exit $rc)" '[ "$rc" = 0 ]'
check "stdout is one JSON line" '[ "$(printf "%s\n" "$out" | wc -l)" = 1 ]'
get() { printf '%s' "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }
check "argv[0] is the backend" '[ "$(get "d[\"argv\"][0]")" = "$here/fake_backend.py" ]'
check "the port is Lemonade's" '[ "$(get "d[\"port\"]")" = 43117 ] && [ "$(get "d[\"argv\"][d[\"argv\"].index(\"--port\")+1]")" = 43117 ]'
check "--alias carries Lemonade's name" '[ "$(get "d[\"argv\"][d[\"argv\"].index(\"--alias\")+1]")" = Qwen3-4B-1bit ]'
check "--cache-reuse 256" '[ "$(get "d[\"argv\"][d[\"argv\"].index(\"--cache-reuse\")+1]")" = 256 ]'
check "--ctx-size reaches -c" '[ "$(get "d[\"argv\"][d[\"argv\"].index(\"-c\")+1]")" = 4096 ]'
check "env is an object" '[ "$(get "type(d[\"env\"]).__name__")" = dict ]'
check "the backend was not started" '[ ! -e "$scratch/events" ]'

out=$(plan --port 43118 --embedding); rc=$?
check "an embedding model keeps its role, no --cache-reuse" \
    '[ "$rc" = 0 ] && [ "$(get "\"--embedding\" in d[\"argv\"] and \"--cache-reuse\" not in d[\"argv\"]")" = True ]'

mkdir -p "$scratch/npu-model"
"$bin" serve -m "$scratch/npu-model" --print-backend >"$scratch/out" 2>"$scratch/err"; rc=$?
check "an NPU directory needs serve (exit $rc)" '[ "$rc" = 5 ] && [ ! -s "$scratch/out" ] && grep -q "run 1bit serve itself" "$scratch/err"'
plan --port 43119 --pm >"$scratch/out"; rc=$?
check "--pm needs serve (exit $rc)" '[ "$rc" = 5 ] && [ ! -s "$scratch/out" ]'
plan --port 43120 --embed "$scratch/small.gguf" >"$scratch/out"; rc=$?
check "--embed needs serve (exit $rc)" '[ "$rc" = 5 ] && [ ! -s "$scratch/out" ]'

# strict placement on HRX (docs/serve.md, "Off-HRX placement: strict by default")
env_of() { printf '%s' "$out" | python3 -c "import json,sys; print(json.load(sys.stdin)['env'].get('$1', '-'))"; }
out=$(env -u GGML_HRX_STRICT -u ONEBIT_STRICT_DEVICE "$bin" serve -m "$scratch/small.gguf" --device hrx \
      --llama-server "$here/fake_backend.py" --no-mem-check --port 43122 --print-backend 2>"$scratch/err"); rc=$?
check "hrx plan: strict by default (exit $rc, GGML_HRX_STRICT=$(env_of GGML_HRX_STRICT))" \
    '[ "$rc" = 0 ] && [ "$(env_of GGML_HRX_STRICT)" = 1 ] && [ "$(env_of GGML_HRX_REPORT)" = 1 ] && [ "$(get "\"-lv\" in d[\"argv\"]")" = False ]'
out=$(env -u GGML_HRX_STRICT -u ONEBIT_STRICT_DEVICE "$bin" serve -m "$scratch/small.gguf" --device hrx \
      --llama-server "$here/fake_backend.py" --no-mem-check --port 43123 --no-strict-device --print-backend 2>"$scratch/err"); rc=$?
check "hrx plan: --no-strict-device leaves GGML_HRX_STRICT out (exit $rc)" '[ "$rc" = 0 ] && [ "$(env_of GGML_HRX_STRICT)" = - ]'
out=$(env -u GGML_HRX_STRICT ONEBIT_STRICT_DEVICE=0 "$bin" serve -m "$scratch/small.gguf" --device hrx \
      --llama-server "$here/fake_backend.py" --no-mem-check --port 43124 --print-backend 2>"$scratch/err"); rc=$?
check "hrx plan: ONEBIT_STRICT_DEVICE=0 leaves it out (exit $rc)" '[ "$rc" = 0 ] && [ "$(env_of GGML_HRX_STRICT)" = - ]'
out=$(GGML_HRX_STRICT=0 "$bin" serve -m "$scratch/small.gguf" --device hrx \
      --llama-server "$here/fake_backend.py" --no-mem-check --port 43125 --print-backend 2>"$scratch/err"); rc=$?
check "hrx plan: a GGML_HRX_STRICT the caller set wins (exit $rc)" '[ "$rc" = 0 ] && [ "$(env_of GGML_HRX_STRICT)" = - ]'
out=$(env -u GGML_HRX_STRICT -u ONEBIT_STRICT_DEVICE "$bin" serve -m "$scratch/small.gguf" --device hrx \
      --llama-server "$here/fake_backend.py" --no-mem-check --port 43126 --strict-device --print-backend 2>"$scratch/err"); rc=$?
check "hrx plan: --strict-device adds the every-graph report and -lv 4 (exit $rc)" \
    '[ "$rc" = 0 ] && [ "$(env_of GGML_HRX_STRICT)" = 1 ] && [ "$(env_of GGML_HRX_REPORT)" = 2 ] && [ "$(get "d[\"argv\"][d[\"argv\"].index(\"-lv\")+1]")" = 4 ]'
out=$(env -u GGML_HRX_STRICT -u ONEBIT_STRICT_DEVICE "$bin" serve -m "$scratch/small.gguf" --device cpu \
      --llama-server "$here/fake_backend.py" --no-mem-check --port 43127 --print-backend 2>"$scratch/err"); rc=$?
check "cpu plan: no strict placement (exit $rc)" '[ "$rc" = 0 ] && [ "$(env_of GGML_HRX_STRICT)" = - ]'
"$bin" serve -m "$scratch/small.gguf" --device hrx --strict-device --no-strict-device --print-backend >"$scratch/out" 2>"$scratch/err"; rc=$?
check "--strict-device with --no-strict-device is refused (exit $rc)" '[ "$rc" != 0 ] && grep -q "pick one" "$scratch/err"'
out=$(env -u GGML_HRX_STRICT -u ONEBIT_STRICT_DEVICE "$bin" serve -m "$scratch/small.gguf" --device hrx \
      --llama-server "$here/fake_backend.py" --no-mem-check --port 43128 --print-backend-env 2>"$scratch/err"); rc=$?
check "--print-backend-env: NAME=VALUE lines, strict on HRX (exit $rc)" \
    '[ "$rc" = 0 ] && printf "%s\n" "$out" | grep -qx GGML_HRX_STRICT=1 && ! printf "%s" "$out" | grep -q "[{}]"'
out=$(GGML_HRX_STRICT=0 "$bin" serve -m "$scratch/small.gguf" --device hrx \
      --llama-server "$here/fake_backend.py" --no-mem-check --port 43129 --print-backend-env 2>"$scratch/err"); rc=$?
check "--print-backend-env: a value the caller set is left out (exit $rc)" '[ "$rc" = 0 ] && ! printf "%s" "$out" | grep -q GGML_HRX_STRICT'

# MTP by default from the model's own head, on hrx (docs/serve.md, "Multi-token prediction")
mplan() { env -u ONEBIT_NO_MTP -u ONEBIT_NO_WARMUP "$bin" serve -m "$scratch/$1.gguf" --device "$2" \
          --llama-server "$here/fake_backend.py" --no-mem-check --port 43130 "${@:3}" --print-backend 2>"$scratch/err"; }
spec() { get '" ".join(d["argv"][i+1] for i, a in enumerate(d["argv"]) if a == "--spec-type") or "-"'; }
out=$(mplan mtp hrx); rc=$?
check "hrx plan: own MTP head drafts (exit $rc, --spec-type $(spec))" \
    '[ "$rc" = 0 ] && [ "$(spec)" = draft-mtp ] && [ "$(get "\"-md\" in d[\"argv\"]")" = False ] && grep -q "own MTP head" "$scratch/err"'
out=$(mplan mtp hrx --mtp-max 6); rc=$?
check "hrx plan: --mtp-max applies to the own head" '[ "$rc" = 0 ] && [ "$(get "d[\"argv\"][d[\"argv\"].index(\"--spec-draft-n-max\")+1]")" = 6 ]'
out=$(mplan mtp hrx --no-mtp); rc=$?
check "hrx plan: --no-mtp leaves it out (exit $rc)" '[ "$rc" = 0 ] && [ "$(spec)" = - ]'
out=$(ONEBIT_NO_MTP=1 "$bin" serve -m "$scratch/mtp.gguf" --device hrx --llama-server "$here/fake_backend.py" \
      --no-mem-check --port 43131 --print-backend 2>"$scratch/err"); rc=$?
check "hrx plan: ONEBIT_NO_MTP=1 leaves it out (exit $rc)" '[ "$rc" = 0 ] && [ "$(spec)" = - ]'
out=$(mplan small hrx); rc=$?
check "hrx plan: no MTP head, no drafter (exit $rc)" '[ "$rc" = 0 ] && [ "$(spec)" = - ]'
out=$(mplan mtp cpu); rc=$?
check "cpu plan: no default drafter (exit $rc)" '[ "$rc" = 0 ] && [ "$(spec)" = - ]'
out=$(mplan mtp hrx --parallel 2); rc=$?
check "hrx plan: --parallel 2, no default drafter (exit $rc)" '[ "$rc" = 0 ] && [ "$(spec)" = - ]'
out=$(mplan mtp hrx --embedding); rc=$?
check "hrx plan: an embedding server, no drafter and no warmup (exit $rc)" \
    '[ "$rc" = 0 ] && [ "$(spec)" = - ] && [ "$(get "\"warmup\" in d")" = False ]'
out=$(mplan small hrx); rc=$?
check "hrx plan: warmup is serve's request" \
    '[ "$(get "d[\"warmup\"][\"path\"]")" = /v1/chat/completions ] && [ "$(get "(d[\"warmup\"][\"body\"][\"max_tokens\"], d[\"warmup\"][\"body\"][\"temperature\"], d[\"warmup\"][\"body\"][\"cache_prompt\"])")" = "(8, 0, False)" ]'
out=$(ONEBIT_NO_WARMUP=1 "$bin" serve -m "$scratch/small.gguf" --device hrx --llama-server "$here/fake_backend.py" \
      --no-mem-check --port 43132 --print-backend 2>"$scratch/err"); rc=$?
check "hrx plan: ONEBIT_NO_WARMUP=1, no warmup (exit $rc)" '[ "$rc" = 0 ] && [ "$(get "\"warmup\" in d")" = False ]'
out=$(mplan small cpu); rc=$?
check "cpu plan: no warmup (exit $rc)" '[ "$rc" = 0 ] && [ "$(get "\"warmup\" in d")" = False ]'

if [ -r /proc/meminfo ]; then
    "$bin" serve -m "$scratch/huge.gguf" --device cpu --port 43121 --llama-server "$here/fake_backend.py" \
        --print-backend >"$scratch/out" 2>"$scratch/err"; rc=$?
    check "a model that does not fit exits 3 (exit $rc)" '[ "$rc" = 3 ] && [ ! -s "$scratch/out" ] && grep -q "does not fit" "$scratch/err"'
fi

exit $fail
