#!/bin/bash
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
# Hyperloom custom workload: ZAYA1-8B Q4_K_M decode on HRX0 (Strix Halo, gfx1151) with the 1bit
# llama.cpp fork. Rebuilds the checkout, waits for the shared box lock and a cool idle GPU, gates on
# wikitext perplexity (prefill and decode-shaped) against a reference taken on the unmodified tree,
# then measures decode tok/s. Writes $RESULT_DIR/$RESULT_FILENAME.json for Hyperloom.
set -uo pipefail
REPO=${FRAMEWORK_REPO_PATH:-$HOME/hyperloom-zaya/llama.cpp}
GGUF=${ZAYA_GGUF:-$HOME/models/zaya1-8b-Q4_K_M.gguf}
OUT=${RESULT_DIR:-$PWD}; NAME=${RESULT_FILENAME:-inferencex_result}
WIKI=${WIKI_TEST:-$HOME/models/qwen38/wikitext-2-raw/wiki.test.raw}
REF=${ZAYA_REFERENCE:-$HOME/hyperloom-zaya/reference.json}
MEMGUARD=${MEM_GUARD:-$HOME/1bit-engine-176/scripts/mem-guard.sh}
B=$REPO/build/bin
export LD_LIBRARY_PATH=$B IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
mkdir -p "$OUT"; T0=$(date +%s)
tctl(){ for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name)" = k10temp ] && echo $(( $(cat $h/temp1_input)/1000 )); done; }
busy(){ cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | sort -n | tail -1; }

result(){ # passed tg pp prefill_ppl decode_ppl reason
  python3 - "$OUT/$NAME.json" "$@" "$REF" "$T0" "$OUT/lemonade-bench.json" <<'PY'
import json, sys, time, os
path, passed, tg, pp, pre, dec, reason, ref_path, t0, lem_path = sys.argv[1:11]
lem = {}
if os.path.exists(lem_path):  # Lemonade's standard (lemonade bench, its CI settings); ours is llama-bench above
    try:
        for sc in json.load(open(lem_path))["models"][0]["results"][0]["scenarios"]:
            if not sc.get("all_runs_failed"):
                lem[sc["name"]] = {"tps_mean": sc["tps"]["mean"], "ttft_ms_mean": sc["ttft_ms"]["mean"]}
    except Exception:
        pass
f = lambda v: float(v) if v not in ("", "nan") else None
ref = json.load(open(ref_path)) if os.path.exists(ref_path) else None
tgv = f(tg) or 0.0
json.dump({
    "framework": "custom", "workload_kind": "scriptable", "model_id": "zaya1-8b-Q4_K_M (HRX0, gfx1151)",
    "throughput_unit": "tok/s", "output_throughput": tgv, "total_token_throughput": tgv,
    "request_throughput": tgv / 128.0 if tgv else 0.0, "completed": 1 if passed == "1" else 0,
    "total_output_tokens": 128, "duration": time.time() - float(t0),
    "quality_gate": {"passed": passed == "1", "reason": reason, "prefill_ppl": f(pre), "decode_ppl": f(dec),
                     "reference": ref, "tolerance": {"prefill_rel": 0.005, "decode_rel": 0.02}},
    "bench_summary": {"decode_tg128_tok_s": f(tg), "prefill_pp512_tok_s": f(pp),
                      "lemonade_bench": lem or None},
}, open(path, "w"), indent=1)
PY
}
fail(){ echo "FAIL: $1" >&2; result 0 "" "" "" "" "$1"; exit 1; }

# 1. build what the benchmark runs
$MEMGUARD 8 cmake --build "$REPO/build" -j 12 --target llama-bench llama-perplexity llama-server > "$OUT/build.log" 2>&1 || fail "build failed (see build.log)"

# 2. the GPU is shared: take the box lock, then wait (max 30 min) for a cool idle GPU
exec 9> "$HOME/.cache/lax-decode/box.lock"; flock 9
for i in $(seq 360); do [ "$(tctl)" -lt 60 ] && [ "$(busy)" -lt 10 ] && break; sleep 5; done

guarded(){ # guarded NAME cmd...: run after cooling below 60 C, with a 93 C thermal stop; log to $OUT/NAME.log
  local log="$OUT/$1.log"; shift
  while [ "$(tctl)" -ge 60 ]; do sleep 5; done
  "$@" > "$log" 2>&1 & local p=$!
  while kill -0 $p 2>/dev/null; do [ "$(tctl)" -ge 93 ] && { kill $p; echo "THERMAL_ABORT at $(tctl) C" >> "$log"; }; sleep 1; done
  wait $p; local rc=$?; cat "$log"; return $rc; }
ppl(){ # ppl NAME cmd...: one retry (a first run on a new build pays for JIT compiles)
  local name=$1; shift; local v; for attempt in 1 2; do
    v=$(guarded "$name-$attempt" timeout 900 "$@" | grep -oE "Final estimate: PPL = [0-9.]+" | awk "{print \$5}")
    [ -n "$v" ] && { echo "$v"; return 0; }
  done; return 1; }

# 3. quality gate
PRE=$(ppl prefill-ppl $B/llama-perplexity -m "$GGUF" -dev HRX0 -f "$WIKI" -c 512 -b 512 --chunks 4)
DEC=$(ppl decode-ppl $B/llama-perplexity -m "$GGUF" -dev HRX0 -f "$WIKI" -c 128 -b 128 -ub 1 --chunks 4)
[ -z "$PRE" ] || [ -z "$DEC" ] && fail "perplexity run failed (prefill '$PRE', decode '$DEC')"
if [ ! -s "$REF" ]; then
  python3 -c "import json; json.dump({'prefill_ppl': $PRE, 'decode_ppl': $DEC}, open('$REF', 'w'))"
  echo "reference written: prefill $PRE decode $DEC" >&2
fi
GATE=$(python3 - "$REF" "$PRE" "$DEC" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); pre, dec = float(sys.argv[2]), float(sys.argv[3])
ok_pre = abs(pre / r["prefill_ppl"] - 1) <= 0.005
ok_dec = abs(dec / r["decode_ppl"] - 1) <= 0.02
print("1" if ok_pre and ok_dec else "0", f"prefill {pre} vs {r['prefill_ppl']}, decode {dec} vs {r['decode_ppl']}")
PY
)
PASSED=${GATE%% *}; WHY=${GATE#* }

# 4. speed
TG=$(guarded bench-tg timeout 900 $B/llama-bench -m "$GGUF" -dev HRX0 -p 0 -n 128 -r 3 -o csv | tail -n 1 | awk -F, '{print $(NF-1)}' | tr -d '"')
PP=$(guarded bench-pp timeout 900 $B/llama-bench -m "$GGUF" -dev HRX0 -p 512 -n 0 -r 2 -o csv | tail -n 1 | awk -F, '{print $(NF-1)}' | tr -d '"')
[ -z "$TG" ] && fail "llama-bench failed"
# Lemonade's standard too (reported, not the optimisation metric): scripts/lemonade_bench.sh
guarded lemonade-bench timeout 900 bash "$(dirname "$0")/lemonade_bench.sh" $B/llama-server "$GGUF" "$OUT/lemonade-bench.json" > /dev/null || echo "lemonade bench failed (see lemonade-bench.log)" >&2
result "$PASSED" "$TG" "$PP" "$PRE" "$DEC" "$WHY"
echo "decode $TG tok/s, prefill $PP tok/s, gate $PASSED ($WHY)"
