#!/usr/bin/env bash
# W3b device half — runs INSIDE an npu-hold ticket. Serve the imported OLMo container on MY 2048x6144 kernel set,
# ask one known-answer question three times (greedy), and report the three texts plus a measured decode tok/s.
# Everything here touches the device; anything that can be done host-side is already done (import, ELF build).
set -uo pipefail
B=$HOME/wt/zha-main/build-npu/1bit
M=${M:-$HOME/scratch-packs/olmo1b-b8-container}   # overridable: the family under test
K=${K:-$HOME/ogaload/oga13/kernels-2048-6144-b8}   # overridable for the QKV-isolation run
PORT=18099
LOG=/home/bcloud/.cache/scratch-pi-rlk/w3b-serve-$$.log
PROMPT=${PROMPT:-"The first person to walk on the Moon was"}

export ONEBIT_NPU_KERNELS=$K
echo "engine: $($B --version 2>&1 | head -1)  (built from 51b6b547c)"
echo "model:  $M"
echo "kernels: $K ($(ls "$K"/*.elf | wc -l) ELFs)"
echo "log:    $LOG"
echo

"$B" serve -m "$M" --device npu --port "$PORT" --host 127.0.0.1 --ctx-size 2048 >"$LOG" 2>&1 &
SP=$!
cleanup() { kill "$SP" 2>/dev/null; wait "$SP" 2>/dev/null; }
trap cleanup EXIT

ready=0
for _ in $(seq 1 90); do
    kill -0 "$SP" 2>/dev/null || { echo "SERVE DIED during startup; log:"; tail -25 "$LOG"; exit 1; }
    if curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 1
done
[ "$ready" = 1 ] || { echo "SERVE NOT READY in 90 s; log:"; tail -25 "$LOG"; exit 1; }
echo "serve ready on :$PORT"
grep -iE 'kernel|elf|npu|self-check' "$LOG" | head -8
echo

# /v1/completions, not /v1/chat/completions: this build has no Jinja renderer (ONEBIT_JINJA=OFF) and the OLMo
# tokenizer has no ChatML tokens, so the chat route is disabled by design (#963) and the completion route is the
# one that answers. A plain string prompt is tokenized by the model's own tokenizer (docs/npu.md line ~1450).
req() {
    curl -s --max-time 180 "http://127.0.0.1:$PORT/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "$(printf '{"prompt":%s,"max_tokens":32,"temperature":0,"stream":false}' \
              "$(printf '%s' "$PROMPT" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')")"
}

# 3 repeats of one prompt (determinism), then one different prompt (the plan's edge case; text equality is NOT a
# bit-exactness proof against a numeric host reference, and the report must say so).
for r in ${RUNS:-1 2 3 edge}; do
    [ "$r" = edge ] && PROMPT=${EDGE_PROMPT:-"Count from 1 to 5 and then stop."}
    t0=$(date +%s.%N)
    resp=$(req)
    t1=$(date +%s.%N)
    el=$(python3 -c "print(f'{$t1-$t0:.2f}')")
    echo "--- run $r (${el}s wall)"
    python3 - "$resp" "$el" <<'PY'
import json,sys
raw,el=sys.argv[1],float(sys.argv[2])
try: d=json.loads(raw)
except Exception: print("  NOT JSON:",raw[:400]); raise SystemExit
ch=d.get('choices',[{}])[0]
msg=ch.get('text') if ch.get('text') is not None else ch.get('message',{}).get('content')
u=d.get('usage',{}) or {}
print("  text:",repr(msg))
print("  usage:",{k:u[k] for k in ('prompt_tokens','completion_tokens') if k in u})
ct=u.get('completion_tokens') or 0
if ct and el>0: print(f"  decode: {ct/el:.2f} tok/s (end-to-end incl. prefill)")
PY
done
echo
echo "=== serve log tail ==="
tail -12 "$LOG"
