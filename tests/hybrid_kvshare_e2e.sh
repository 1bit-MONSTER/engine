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
# Target 1 end to end with real dma-bufs: an HRX llama-server exports its KV region, a *separate provider
# process* receives the descriptors over SCM_RIGHTS, maps them as its own KV and prefills the prompt into
# them, and the server adopts the cells with no data movement (docs/zero-copy.md "Target 1",
# docs/hybrid.md "The shared route"). The driver is kvshare_fd_handoff.py, which plays the role
# app/serve_hybrid.h plays in production, so no engine binary is involved -- what is under test is the
# contract between two processes over a shared region.
#
# Skips (exit 77) when this machine cannot run it: no gfx1151 for HRX, no model, or no reference provider.
# The provider is the fork's tools/kvshare-prefill; this build does not produce it, so it is passed in:
#
#   usage: hybrid_kvshare_e2e.sh <llama-server> <model.gguf> [provider]
#
#   ONEBIT_KVSHARE_PREFILL=/path/to/llama-kvshare-prefill   (same as the third argument)
#   PROMPT_TOKENS / CTX   prompt length to hand over (default 256) and the server's context (default 4096).
#                         The driver grows a sentence until it reaches that length, so a value below the
#                         sentence's own 145 tokens hands over the single sentence instead.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
server=${1:?usage: hybrid_kvshare_e2e.sh <llama-server> <model.gguf> [provider]}
model=${2:-}
provider=${3:-${ONEBIT_KVSHARE_PREFILL:-}}

have=$("$root/scripts/gpu-targets.sh" 2>/dev/null | tr '\n' ' ')
want=${ONEBIT_HRX_E2E_TARGETS:-gfx1151}
found=0
for t in $want; do case " $have " in *" $t "*) found=1 ;; esac; done
[ $found = 1 ] || { echo "skip: no $want GPU for HRX on this machine (GPUs: ${have:-none})"; exit 77; }
[ -x "$server" ] || { echo "skip: no llama-server at $server"; exit 77; }
[ -n "$model" ] && [ -f "$model" ] || { echo "skip: no model"; exit 77; }
[ -n "$provider" ] && [ -x "$provider" ] || {
    echo "skip: no reference provider (build the fork's tools/kvshare-prefill, then set ONEBIT_KVSHARE_PREFILL)"; exit 77; }

scratch=$(mktemp -d "${TMPDIR:-/tmp}/1bit-kvshare-XXXXXX")
trap 'rm -rf "$scratch"' EXIT

PROMPT_TOKENS=${PROMPT_TOKENS:-256} CTX=${CTX:-4096} \
    python3 "$here/kvshare_fd_handoff.py" "$server" "$provider" "$model" "$scratch" 8 2>&1 | tee "$scratch/test.log"
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || { echo "--- driver log"; tail -30 "$scratch/test.log"; echo FAIL; exit 1; }

fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
check "the region was exported and shared over SCM_RIGHTS" 'grep -q "share ok: share_id=" "$scratch/test.log"'
check "no prompt bytes were copied (n_read 0)" 'grep -q "\"n_read\": 0" "$scratch/test.log"'
check "the decoder decoded from the adopted cells (prompt_n 1)" 'grep -q "prompt_n=1 " "$scratch/test.log"'
check "the provider prefilled into the region" 'grep -q "provider result: .*\"prefill_ms\":" "$scratch/test.log"'
check "zero sanitizer reports" 'grep -q "ASan reports=0" "$scratch/test.log"'
if [ $fail -ne 0 ]; then echo "--- driver log (tail)"; tail -30 "$scratch/test.log"; echo FAIL; exit 1; fi
echo PASS
