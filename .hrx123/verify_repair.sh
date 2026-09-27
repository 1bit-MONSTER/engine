#!/usr/bin/env bash
# verify_repair.sh <tag> <port>...  -- one fresh llama-server per port, concurrently.
# Classifies each sample as MATCH / LOUD-ABORT / SILENT-WRONG / LAUNCH-FAILED and archives
# the raw JSON + server logs under logs/verify-<tag>/ for inspection.
set -u
source "$(dirname "$0")/env.sh"
D="$(dirname "$0")"; TAG="$1"; shift
ART="$D/logs/verify-$TAG"; mkdir -p "$ART"; rm -f "$ART"/*
{
  echo "=== verify $TAG :: $(date -Is) n=$# ==="
  echo "llama.cpp: $(git -C "$HOME/1bit-engine/third_party/llama.cpp" rev-parse --short HEAD) (branch $(git -C "$HOME/1bit-engine/third_party/llama.cpp" rev-parse --abbrev-ref HEAD))"
  echo "engine:    $(git -C "$HOME/1bit-engine" rev-parse --short HEAD)"
  echo "quiet:     KFD=[$(fuser /dev/kfd 2>/dev/null || echo none)] render=[$(fuser /dev/dri/renderD128 2>/dev/null || echo none)]"
  echo "movers:    ksm=$(cat /sys/kernel/mm/ksm/run) proact=$(cat /proc/sys/vm/compaction_proactiveness) thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled) unevictable=$(cat /proc/sys/vm/compact_unevictable_allowed)"
  echo "load:      $(cut -d' ' -f1-3 /proc/loadavg)"
} | tee "$ART/summary.txt"
run() { local port=$1
  "$HRX123_BIN/llama-server" -m "$HRX123_MODEL" --host 127.0.0.1 --port "$port" -c 8192 -np 1 -t 8 --device HRX0 -ngl 99 > "$ART/server-$port.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 1; done
  curl -s --max-time 900 "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' --data-binary @/tmp/longprompt.json > "$ART/out-$port.json" 2>&1
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}
for p in "$@"; do run "$p" & done; wait
python3 - "$ART" << 'PY'
import json,glob,sys,os
ART=sys.argv[1]
REF=" Paris. \n\nThe quick brown fox jumps over the lazy dog near the riverbank"  # real newlines
match=[]; loud=[]; silent=[]; failed=[]
for f in sorted(glob.glob(os.path.join(ART,'out-*.json'))):
    port=os.path.basename(f)[4:-5]
    logf=os.path.join(ART,'server-%s.log'%port)
    log=open(logf,errors='ignore').read() if os.path.exists(logf) else ''
    fired='NaN logits (engine#123)' in log
    try: c=json.load(open(f)).get('content')
    except Exception: c=None
    if c is None:
        (loud if fired else failed).append(port)
    elif c==REF: match.append(port)
    else: silent.append(port)
lines=[]
lines.append("MATCH          n=%d  %s" % (len(match), match))
lines.append("LOUD-ABORT     n=%d  %s" % (len(loud), loud))
lines.append("SILENT-WRONG   n=%d  %s" % (len(silent), silent))
lines.append("LAUNCH-FAILED  n=%d  %s" % (len(failed), failed))
lines.append("TOTAL samples=%d" % (len(match)+len(loud)+len(silent)+len(failed)))
out="\n".join(lines)
print(out); open(os.path.join(ART,'summary.txt'),'a').write(out+"\n")
PY
echo "artifacts: $ART"
