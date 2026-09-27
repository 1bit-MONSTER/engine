#!/usr/bin/env bash
# probe_divergence.sh <model> <port> <n> <tag> [extra llama-server args...]
# Documented #140 probe: N identical greedy /completion requests, count results whose
# first-token top-1 logprob differs from the most common value.
set -u
source "$(dirname "$0")/env.sh"
D="$(dirname "$0")"
MODEL="$1"; PORT="$2"; N="${3:-300}"; TAG="$4"; shift 4
LOGDIR="$D/logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/probe-$TAG.log"; OUT="$LOGDIR/probe-$TAG.jsonl"; SRV="$LOGDIR/probe-server-$TAG.log"
: > "$LOG"; : > "$OUT"; : > "$SRV"
say() { echo "$@" | tee -a "$LOG"; }
say "=== divergence probe $TAG :: model=$(basename "$MODEL") n=$N $(date -Is) ==="
say "llama.cpp: $(git -C "$HOME/1bit-engine/third_party/llama.cpp" rev-parse --short HEAD)"
say "quiet: KFD=[$(fuser /dev/kfd 2>/dev/null || echo none)] render=[$(fuser /dev/dri/renderD128 2>/dev/null || echo none)]"
say "page movers: ksm=$(cat /sys/kernel/mm/ksm/run 2>/dev/null) proact=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null) thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
say "decode-split: ${GGML_HRX_DISABLE_DISPATCH:-<default: enabled>}"
env ${GGML_HRX_DISABLE_DISPATCH:+GGML_HRX_DISABLE_DISPATCH=$GGML_HRX_DISABLE_DISPATCH} \
    "$HRX123_BIN/llama-server" -m "$MODEL" --host 127.0.0.1 --port "$PORT" -c 4096 -np 1 -t 8 "$@" \
    > "$SRV" 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null' EXIT
for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
PROMPT='Question: What is the capital of France? Answer in one word.\nAnswer:'
for i in $(seq 1 "$N"); do
  curl -s --max-time 120 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"$PROMPT\",\"n_predict\":1,\"temperature\":0,\"seed\":1,\"cache_prompt\":false,\"n_probs\":5,\"stream\":false}" \
    >> "$OUT"
  echo >> "$OUT"
done
kill $SP 2>/dev/null; wait $SP 2>/dev/null
python3 - "$OUT" "$LOG" << 'PY'
import json,sys,collections
out,log=sys.argv[1],sys.argv[2]
vals=[]; errs=0
for line in open(out):
    line=line.strip()
    if not line: continue
    try: d=json.loads(line)
    except Exception: errs+=1; continue
    cp=d.get("completion_probabilities") or []
    tl=(cp[0].get("top_logprobs") if cp else None) or []
    if tl:
        vals.append(tuple(round(t["logprob"],6) for t in tl))
    else: errs+=1
lines=[]
lines.append("parsed=%d errors=%d" % (len(vals), errs))
if vals:
    c=collections.Counter(vals); mode,cnt=c.most_common(1)[0]
    div=len(vals)-cnt
    lines.append("n=%d  modal_count=%d  divergent=%d  rate=%.1f%%  distinct=%d  match_reference_probe=top5"
                 % (len(vals), cnt, div, 100.0*div/len(vals), len(c)))
    lines.append("modal top5: %s" % (list(mode),))
    for v,k in c.most_common(6):
        lines.append("   x%-4d %s" % (k, list(v)))
open(log,"a").write("\n".join(lines)+"\n")
print("\n".join(lines))
PY
