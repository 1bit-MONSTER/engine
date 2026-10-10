#!/usr/bin/env bash
# engine#315 cheap-probe.sh <tag> <n_requests> <compact:0|1>
#
# A small-model detector for the preemption-sensitive fault class, modelled on the study's
# hrx123-probe.sh: one llama-server with Qwen3-0.6B on HRX0, N identical greedy requests in the same
# process. Healthy = every response identical. Faulty = responses diverge (the stale-bytes read
# depends on request history) and/or the server logs "HRX returned NaN logits".
#
# Runs the instrumented build (315fp-build, the pin's kernel plus the nan-dump fingerprint): if a
# fault happens, the log then says which tail node first went NaN.
#
# compact=1 forces KFD queue evictions the way the study did (sudo -n tee /proc/sys/vm/compact_memory),
# which is what opens the preemption window for this class.
set -u
TAG=${1:?tag}; N=${2:-120}; COMPACT=${3:-0}; CTX=${CTX:-4096}
B=$HOME/wt/315fp-build/bin
D=$HOME/wt/315-cheap-probe; mkdir -p "$D"
SRV=$D/$TAG.server.log; OUT=$D/$TAG.jsonl
: > "$SRV"; : > "$OUT"
PORT=$((21000 + RANDOM % 2000))
HSA=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1

echo "=== cheap probe tag=$TAG n=$N compact=$COMPACT $(date -Is) port=$PORT gtt=$(awk '{printf "%.1f", $1/1073741824}' /sys/class/drm/card*/device/mem_info_gtt_used | head -1)GiB"

env LD_LIBRARY_PATH="$B" IREE_HAL_AMDGPU_LIBHSA_PATH="$HSA" \
    GGML_HRX_COPY_PROBE=1 GGML_HRX_LOGITS_FP=100000 \
    GGML_HRX_TENSOR_FP="${FP-tail:16}" ${FPONLY:+GGML_HRX_TENSOR_FP_NANONLY=1} \
    "$B/llama-server" -m "$HOME/models/Qwen3-0.6B-Q4_K_M.gguf" -dev HRX0 -ngl 99 -fa on \
    --host 127.0.0.1 --port "$PORT" -c "$CTX" -np 1 -t 8 >> "$SRV" 2>&1 &
SP=$!
CP=
cleanup() { [ -n "$SP" ] && kill "$SP" 2>/dev/null; [ -n "$CP" ] && kill "$CP" 2>/dev/null; }
trap cleanup EXIT

up=0
for _ in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }; sleep 1; done
[ "$up" = 1 ] || { echo "ABORT: server never became healthy"; tail -5 "$SRV"; exit 2; }

if [ "$COMPACT" = 1 ]; then
    ( while true; do echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null 2>&1; sleep 0.2; done ) &
    CP=$!
fi

# SHAPE: the arms' prompts ran up to ~868 tokens of KV; the default probe prompt is 6 tokens, so a
# longer prompt is a different point in the shape space. LONG=<words> pads the prompt with a repeated
# sentence (the same padding for every request, so greedy output stays comparable).
if [ "${LONG:-0}" != "0" ]; then
    PROMPT="$(python3 -c "print(' '.join(['The quick brown fox jumps over a lazy dog while engineers measure kernel launch gaps.'] * $(( ${LONG} / 12 + 1 ))))" | cut -c1-$(( ${LONG} * 5 ))) The capital of France is"
else
    PROMPT="The capital of France is"
fi
for i in $(seq 1 "$N"); do
    r=$(curl -s --max-time 120 "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" \
        -d "{\"prompt\":\"$PROMPT\",\"n_predict\":24,\"temperature\":0,\"seed\":1,\"cache_prompt\":false,\"stream\":false}")
    echo "$r" >> "$OUT"
done
echo "@@END" >> "$SRV"

python3 - "$OUT" "$SRV" <<'PY'
import hashlib, json, sys
from collections import Counter
bodies = []
for line in open(sys.argv[1], errors="replace"):
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        bodies.append("UNPARSEABLE:" + line[:40])
        continue
    bodies.append(d.get("content", "NO-CONTENT"))
c = Counter(bodies)
nan = sum(1 for l in open(sys.argv[2], errors="replace") if "NaN logits" in l)
print(f"requests={len(bodies)} distinct_outputs={len(c)} nan_lines={nan}")
for body, n in c.most_common(5):
    print(f"   {n:5d} x {body[:70]!r}")
PY
