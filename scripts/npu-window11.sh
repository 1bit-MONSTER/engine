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
# npu-window11.sh -- the identical-text + prof-table gate (engine #799 P1).
#
#   scripts/npu-window11.sh <model.q4nx> <kernels dir> --label NAME --out FILE [--port N] [--max-tokens N]
#   scripts/npu-window11.sh --compare A.log B.log [--tol PCT]
#
# Three fixed greedy prompts, run against one served model, recorded as
#   text <i> sha256=<hash> chars=<n> text=<prefix>
#   prof <phase> <role> M=<m> ms=<ms/launch>
# so two runs can be compared without trusting a human to eyeball two terminals.  The gate a change has to
# pass is: 3/3 transcripts byte-identical, the prof table's rows (phase, role, M) identical, and no row's
# ms/launch outside --tol percent.
#
# It starts the serve itself, on the device (no NPU_INFER_HOST_GEMM here): the point is the real kernel
# path.  Call it under npu-hold; it does not take a ticket of its own.
set -u

compare() {
    local a="$1" b="$2" tol="${3:-10}"
    python3 - "$a" "$b" "$tol" <<'PY'
import re, sys
a, b, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])

def load(path):
    texts, prof = {}, {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"text (\d+) sha256=(\S+) chars=(\d+) text=(.*)$", line)
        if m:
            texts[int(m.group(1))] = (m.group(2), m.group(3), m.group(4))
            continue
        m = re.match(r"prof (dec|pre) (\S+) M=(\d+) ms=([-\d.]+) launches=(\d+)$", line)
        if m:
            prof[(m.group(1), m.group(2), m.group(3))] = float(m.group(4))
    return texts, prof

ta, pa = load(a)
tb, pb = load(b)
fail = 0
if len(ta) != 3 or len(tb) != 3:
    print(f"FAIL only {len(ta)}/{len(tb)} transcripts recorded (3 expected)"); fail = 1
same = sum(1 for i in ta if i in tb and ta[i][0] == tb[i][0])
print(f"texts: {same}/3 identical")
for i in sorted(ta):
    if i in tb and ta[i][0] != tb[i][0]:
        print(f"  MISMATCH prompt {i}: {ta[i][2][:60]!r} vs {tb[i][2][:60]!r}")
if same != 3: fail = 1

rows_a, rows_b = set(pa), set(pb)
if rows_a != rows_b:
    print(f"FAIL prof rows differ: only in A {sorted(rows_a - rows_b)}, only in B {sorted(rows_b - rows_a)}")
    fail = 1
else:
    print(f"prof rows: {len(rows_a)} identical (phase, role, M)")
for k in sorted(rows_a & rows_b):
    va, vb = pa[k], pb[k]
    if va <= 0 and vb <= 0: continue
    delta = abs(vb - va) / max(va, 1e-9) * 100.0
    flag = "" if delta <= tol else "  <-- outside tolerance"
    print(f"  {k[0]:3} {k[1]:5} M={k[2]:3} {va:8.3f} -> {vb:8.3f} ms  ({delta:+.1f}%){flag}")
    if delta > tol: fail = 1
print("window11: PASS" if not fail else "window11: FAIL")
sys.exit(1 if fail else 0)
PY
}

if [ "${1:-}" = "--compare" ]; then
    [ $# -ge 3 ] || { echo "usage: $0 --compare A.log B.log [--tol PCT]" >&2; exit 2; }
    tol=10
    [ "${4:-}" = "--tol" ] && tol="${5:-10}"
    compare "$2" "$3" "$tol"
    exit $?
fi

MODEL="${1:-}"; KERNELS="${2:-}"; LABEL="run"; OUT=""; PORT=8110; MAXTOK=8
shift 2 2>/dev/null || true
while [ $# -gt 0 ]; do
    case "$1" in
        --label) LABEL="${2:-}"; shift 2;;
        --out)   OUT="${2:-}";   shift 2;;
        --port)  PORT="${2:-}";  shift 2;;
        --max-tokens) MAXTOK="${2:-}"; shift 2;;
        *) echo "unknown option $1" >&2; exit 2;;
    esac
done
[ -n "$MODEL" ] && [ -n "$KERNELS" ] || { echo "usage: $0 <model.q4nx> <kernels> --label NAME --out FILE" >&2; exit 2; }
[ -n "$OUT" ] || OUT="/tmp/window11-$LABEL.log"
BIN="${ONEBIT_BIN:-$(dirname "$0")/../1bit}"
[ -x "$BIN" ] || { echo "no 1bit at $BIN (set ONEBIT_BIN)" >&2; exit 2; }

: > "$OUT"
NPU_PROFILE=1 NPU_NO_DEVICE_LOCK=1 "$BIN" forward-serve -m "$MODEL" --kernels "$KERNELS" -p "$PORT" \
    > "$OUT.serve" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT INT TERM

for _ in $(seq 1 600); do
    curl -sf -m 2 "localhost:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $SPID 2>/dev/null || { echo "serve died; see $OUT.serve" >&2; tail -5 "$OUT.serve" >&2; exit 1; }
    sleep 1
done
curl -sf -m 2 "localhost:$PORT/health" >/dev/null 2>&1 || { echo "serve never became ready" >&2; exit 1; }

# Three fixed greedy prompts (no temperature field = greedy), fixed token budget: the same three every run.
PROMPTS=(
  "List the three primary colours."
  "What is the capital of France? Answer in one word."
  "Write one sentence about the sea."
)
i=0
for p in "${PROMPTS[@]}"; do
    i=$((i+1))
    body=$(python3 - "$p" "$MAXTOK" <<'PY'
import json, sys
print(json.dumps({"messages": [{"role": "user", "content": sys.argv[1]}],
                  "max_tokens": int(sys.argv[2])}))
PY
)
    # The transcript is the whole assistant message, not just `content`: a short greedy reply on a
    # thinking model is all `reasoning_content`, and two empty strings would compare "identical"
    # without having tested anything.
    text=$(curl -sS -m 600 -H 'Content-Type: application/json' "localhost:$PORT/v1/chat/completions" \
        --data-binary "$body" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    m = d["choices"][0]["message"]
    parts = [m.get("content") or "", m.get("reasoning_content") or ""]
    parts.append(json.dumps(m.get("tool_calls") or [], sort_keys=True))
    parts.append(str(d["choices"][0].get("finish_reason")))
    print("\u241f".join(parts))
except Exception:
    print("")')
    sha=$(printf '%s' "$text" | sha256sum | cut -d" " -f1)
    prefix=$(printf '%s' "$text" | tr '\n' ' ' | cut -c1-60)
    printf 'text %d sha256=%s chars=%d text=%s\n' "$i" "$sha" "${#text}" "$prefix" >> "$OUT"
done

kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; trap - EXIT INT TERM

# The prof table: one row per (phase, role, M) from the serve's own [prof] dump.
python3 - "$OUT.serve" "$OUT" <<'PY'
import re, statistics, sys
serve, out = sys.argv[1], sys.argv[2]
acc = {}
for line in open(serve, encoding="utf-8", errors="replace"):
    m = re.match(r"\[prof\] (dec|pre)\s+(\S+)\s+(\d+)\s+(\d+)\s+\S+\s+\S+\s+\S+\s+\S+\s+([\d.]+)\s*$",
                 line)
    if m:
        key = (m.group(1), m.group(2), int(m.group(3)))
        acc.setdefault(key, []).append((float(m.group(5)), int(m.group(4))))
# The dump reports launches since the previous dump, so a run has several windows per row: the row's
# cost is the median of the windows, and its launch count the sum.
with open(out, "a", encoding="utf-8") as f:
    for key in sorted(acc):
        ms = statistics.median(v[0] for v in acc[key])
        launches = sum(v[1] for v in acc[key])
        f.write(f"prof {key[0]} {key[1]} M={key[2]} ms={ms:.3f} launches={launches}\n")
print(f"{len(acc)} prof rows")
PY

echo "wrote $OUT"
grep -c '^text ' "$OUT" | sed 's/^/transcripts: /'
