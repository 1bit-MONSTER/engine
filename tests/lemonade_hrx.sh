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
# scripts/lemonade-hrx (docs/lemonade.md, "Ubuntu 26.04: the PPA's lemond"), without a GPU: started
# with the argv the Lemonade PPA's HRX recipe (llamacpp-hrx) gives llama-server, it
#   - execs the llama-server beside it with that argv unchanged, plus -ngl 99 and --cache-reuse 256,
#   - in the environment `1bit serve --print-backend-env` gives an HRX backend (GGML_HRX_STRICT=1),
#     where a value already set wins,
#   - adds -kvu with --parallel above 1, and leaves an -ngl from hrx_args alone,
#   - starts nothing when the model does not fit (serve's exit 3).
#
# usage: tests/lemonade_hrx.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: lemonade_hrx.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

# the package layout: 1bit, lemonade-hrx, hrx/llama/bin/llama-server (a stand-in that records itself)
pkg=$scratch/pkg
mkdir -p "$pkg/hrx/llama/bin"
ln -s "$(readlink -f "$bin")" "$pkg/1bit"
cp "$here/../scripts/lemonade-hrx" "$pkg/"
cat >"$pkg/hrx/llama/bin/llama-server" <<'EOF'
#!/bin/sh
printf '%s\n' "$@" >"$LEMONADE_HRX_TEST_OUT.argv"
env >"$LEMONADE_HRX_TEST_OUT.env"
EOF
chmod +x "$pkg/hrx/llama/bin/llama-server"
export LEMONADE_HRX_TEST_OUT=$scratch/run

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

# lemond's argv (hrx::build_server_argv in Lemonade v2026.42.1): -m, --ctx-size, --device HRX0,
# --port, --jinja, --metrics, then hrx_args with --parallel 1 appended when it has none
lemond=(-m "$scratch/small.gguf" --ctx-size 4096 --device HRX0 --port 43200 --jinja --metrics --parallel 1)
run() { rm -f "$scratch"/run.*; env -u GGML_HRX_STRICT -u ONEBIT_STRICT_DEVICE "$@" 2>"$scratch/err"; }
argv() { cat "$scratch/run.argv" 2>/dev/null; }
envv() { grep "^$1=" "$scratch/run.env" 2>/dev/null | cut -d= -f2-; }

run "$pkg/lemonade-hrx" "${lemond[@]}"; rc=$?
check "starts the llama-server beside it (exit $rc)" '[ "$rc" = 0 ] && [ -s "$scratch/run.argv" ]'
check "Lemonade's argv comes first, unchanged" '[ "$(argv | head -n ${#lemond[@]})" = "$(printf "%s\n" "${lemond[@]}")" ]'
check "-ngl 99 and --cache-reuse 256 added" 'argv | grep -qx -- -ngl && argv | grep -qx 99 && argv | grep -qx -- --cache-reuse'
check "no -kvu with one slot" '! argv | grep -qx -- -kvu'
check "GGML_HRX_STRICT=1 from 1bit serve" '[ "$(envv GGML_HRX_STRICT)" = 1 ] && [ "$(envv GGML_HRX_REPORT)" = 1 ]'

run env GGML_HRX_STRICT=0 "$pkg/lemonade-hrx" "${lemond[@]}"; rc=$?
check "a GGML_HRX_STRICT already set wins (exit $rc)" '[ "$rc" = 0 ] && [ "$(envv GGML_HRX_STRICT)" = 0 ]'

run "$pkg/lemonade-hrx" -m "$scratch/small.gguf" --ctx-size 4096 --device HRX0 --port 43201 --jinja --metrics -ngl 20 --parallel 4; rc=$?
check "--parallel 4 adds -kvu; -ngl from hrx_args is kept (exit $rc)" \
    '[ "$rc" = 0 ] && argv | grep -qx -- -kvu && [ "$(argv | grep -cx -- -ngl)" = 1 ] && argv | grep -qx 20'

if [ -r /proc/meminfo ]; then
    run "$pkg/lemonade-hrx" -m "$scratch/huge.gguf" --ctx-size 4096 --device HRX0 --port 43202 --jinja --metrics --parallel 1; rc=$?
    check "a model that does not fit: exit 3, nothing started" '[ "$rc" = 3 ] && [ ! -e "$scratch/run.argv" ] && grep -q "exited 3" "$scratch/err"'
fi

run "$pkg/lemonade-hrx" --version; rc=$?
check "no model (--version): straight to llama-server (exit $rc)" '[ "$rc" = 0 ] && argv | grep -qx -- --version'

exit $fail
