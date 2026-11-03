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
# npu-window11.sh -- the identical-text + prof-table gate (engine #799 P1), and the folded-container row
#                     that adds the numeric taps (engine#876).
#
#   scripts/npu-window11.sh <model.q4nx> <kernels dir> --label NAME --out FILE [--port N] [--max-tokens N]
#                           [--folded [--tap-layer N]]
#   scripts/npu-window11.sh --compare A.log B.log (--tol PCT | --control C1.log C2.log)
#   scripts/npu-window11.sh --self-test
#
# Three fixed greedy prompts, run against one served model, recorded as
#   text <i> sha256=<hash> chars=<n> text=<prefix>
#   prof <phase> <role> M=<m> ms=<ms/launch>
# and, with --folded,
#   folded layer=<n> taps=<count>
#   tap <the container's own [pb16i-trace]/[qkv-dump] line, whitespace-normalised>
# so two runs can be compared without trusting a human to eyeball two terminals.  The gate a change has to
# pass is: 3/3 transcripts byte-identical, the prof table's rows (phase, role, M) identical, and no row's
# ms/launch outside the tolerance.
#
# The tolerance is NOT a constant: engine#834 measured the same binary against itself on this box and the rows
# moved up to 7.7 %, so a 2 % assertion cannot be evaluated here at all.  `--compare` therefore refuses to run
# without either an explicit `--tol` (a number you can defend) or `--control C1.log C2.log` (the same binary
# run twice, whose own spread becomes the tolerance, rounded up to the next whole percent).  Both statistics
# are printed for every row and the one the tolerance applies to is named in the header:
#
#   ratio-of-medians  median(ms_B) / median(ms_A)          -- what this script used to print
#   median-of-ratios  median over rounds of ms_B / ms_A    -- paired; needs both logs to carry the same rounds
#
# The paired statistic is the honest one for a "no regression" claim, because the box's noise is a level shift
# over a round (someone else's traffic, a BO/ELF warm-up) that a paired ratio cancels and a ratio of medians
# turns into a fake movement — in engine#834's control those two had opposite signs on 5 of 9 rows.  A log with
# several rounds per row carries the same `prof`/`text` lines once per round, in round order: append the rounds
# of one arm into one file, and alternate the order inside a round (A B, then B A) so the second serve does not
# pay for the warm-up in every round.
#
# **--folded is the row for a folded (pb16i/t2s) container.**  Such a container needs a folded basis (its
# activations are rotated before every projection), a ternary LM head and a split full-attention QKV, none of
# which the `rm` Qwen3-0.6B the row above was written for has -- so defects living in exactly those parts were
# invisible to it (engine#876: three of them, engine#808).  With --folded the serve runs with the container's own
# debug taps on (NPU_PB16I_TRACE / NPU_QKV_DUMP, both already in the forward, so no arm is instrumented
# differently) and the taps of layer N are recorded per launch: the rotated activations, the int8 codes the
# kernel was given, the int32 accumulator, the dequantised output and the side-table `eref`.  Two runs then have
# to agree on those lines too -- the tap equality is the assertion that bites: a wrong rotation still produces
# well-formed text, and it cannot produce the same intermediate numbers.  With no numbers to be had from the
# text, ms/launch on a folded row is only scored against a measured `--control` pair (engine#834); a hand-typed
# `--tol` is reported there and not asserted.
#
# It starts the serve itself, on the device (no NPU_INFER_HOST_GEMM here): the point is the real kernel
# path.  Call it under npu-hold; it does not take a ticket of its own.
set -u

compare() {
    python3 - "$@" <<'PY'
# engine#834: the tolerance has to come from a measurement, and the statistic has to be named.
import math, re, sys, statistics

def die(msg, code=2):
    """Argument and policy refusals exit 2; a failing comparison exits 1."""
    print(msg, file=sys.stderr)
    sys.exit(code)


args = sys.argv[1:]
a = b = None
tol = None
control = None
i = 0
while i < len(args):
    if args[i] == "--tol":
        tol = float(args[i + 1]); i += 2
    elif args[i] == "--control":
        control = (args[i + 1], args[i + 2]); i += 3
    elif a is None:
        a = args[i]; i += 1
    elif b is None:
        b = args[i]; i += 1
    else:
        die(f"compare: unexpected argument {args[i]}")
if a is None or b is None:
    die("compare: need two logs")
if tol is None and control is None:
    die(
        "compare: no tolerance given.\n"
        "  Two runs differ on this box by more than any fixed constant is worth: engine#834 measured the\n"
        "  SAME binary against itself moving up to 7.7 % per row. Pass --tol PCT with a number you can\n"
        "  defend, or --control C1.log C2.log (that same-binary pair) and the spread it shows becomes the\n"
        "  tolerance, rounded up to the next whole percent.")


def load(path):
    texts, prof, order, taps, head = {}, {}, [], [], {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"text (\d+) sha256=(\S+) chars=(\d+) text=(.*)$", line)
        if m:
            texts.setdefault(int(m.group(1)), []).append(m.group(2))
            continue
        m = re.match(r"prof (dec|pre) (\S+) M=(\d+) ms=([-\d.]+) launches=(\d+)$", line)
        if m:
            key = (m.group(1), m.group(2), m.group(3))
            if key not in prof:
                prof[key] = []
                order.append(key)
            prof[key].append(float(m.group(4)))
            continue
        m = re.match(r"folded layer=(\d+) taps=(\d+)$", line)
        if m:
            head = {"layer": int(m.group(1)), "taps": int(m.group(2))}
            continue
        if line.startswith("tap "):
            taps.append(line[4:].rstrip("\n"))
    return texts, prof, order, taps, head


def stat_row(pa, pb, paired_ok):
    """(ratio-of-medians %, median-of-ratios %); the second is None without pairs."""
    ra = statistics.median(pa)
    rb = statistics.median(pb)
    med = (rb - ra) / max(ra, 1e-9) * 100.0 if (ra > 0 or rb > 0) else 0.0
    if not paired_ok:
        return med, None
    ratios = [y / x for x, y in zip(pa, pb) if x > 0]
    if not ratios:
        return med, None
    return med, (statistics.median(ratios) - 1.0) * 100.0


def spread(c1, c2):
    """One binary run twice: its own spread is the only tolerance this box can justify."""
    _, p1, o1, _, _ = load(c1)
    _, p2, _, _, _ = load(c2)
    rows = [k for k in o1 if k in p2]
    if not rows:
        die("compare: the control logs share no prof rows")
    paired_ok = min(min(len(p1[k]), len(p2[k])) for k in rows) >= 2
    worst_m = worst_p = 0.0
    print(f"control: {len(rows)} rows from {c1} vs {c2} (same binary)")
    for k in rows:
        m, p = stat_row(p1[k], p2[k], paired_ok)
        worst_m = max(worst_m, abs(m))
        if p is not None:
            worst_p = max(worst_p, abs(p))
        print(f"  {k[0]:3} {k[1]:5} M={k[2]:3} ratio-of-medians {m:+6.1f}%"
              + (f"  median-of-ratios {p:+6.1f}%" if p is not None else ""))
    worst = worst_p if paired_ok else worst_m
    # A control whose rows all move the same way by a similar amount is not noise: it is the second
    # position in the round paying for the BO/ELF warm-up (engine#834's fixed "A B" order). Saying so
    # is worth more than the number, because no number of rounds averages a systematic bias out.
    vals = [(stat_row(p1[k], p2[k], paired_ok)[1] if paired_ok else stat_row(p1[k], p2[k], False)[0])
            for k in rows]
    vals = [v for v in vals if v is not None]
    if len(vals) >= 3:
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        same_way = min(vals) > 1.0 or max(vals) < -1.0
        if same_way and mean != 0 and sd / abs(mean) < 0.5:
            print(f"  note: every control row moves the same way ({min(vals):+.1f}% .. {max(vals):+.1f}%). That is a")
            print("        POSITION BIAS (the second serve of a round pays for the warm-up), not noise, and more")
            print("        rounds will not average it out: alternate the order inside a round (A B, then B A)")
            print("        before trusting the tolerance this control suggests.")
    return math.ceil(worst), ("median-of-ratios" if paired_ok else "ratio-of-medians"), worst, paired_ok


tol_stat = None
if control is not None:
    der_tol, der_stat, worst, ctrl_paired = spread(*control)
    tol = float(max(der_tol, 1))
    tol_from = f"control (worst row {worst:.1f}%, rounded up)"
    if ctrl_paired:
        tol_stat = der_stat
else:
    tol_from = "--tol"

ta, pa, oa, ka, ha = load(a)
tb, pb, ob, kb, hb = load(b)

rounds = {k: min(len(pa[k]), len(pb[k])) for k in set(pa) & set(pb)}
paired_ok = bool(rounds) and min(rounds.values()) >= 2
stat_name = "median-of-ratios" if paired_ok else "ratio-of-medians"
# A folded row's assertion is its taps; with no control pair its ms/launch is reported, not asserted (engine#834).
tol_scored = not (ka or kb) or control is not None
fail = 0

if len(ta) != 3 or len(tb) != 3:
    print(f"FAIL only {len(ta)}/{len(tb)} transcript prompts recorded (3 expected)"); fail = 1
same = sum(1 for i in ta if i in tb and ta[i] == tb[i])
print(f"texts: {same}/3 identical")
for i in sorted(ta):
    if i in tb and ta[i] != tb[i]:
        print(f"  MISMATCH prompt {i}: {ta[i][:1]} vs {tb[i][:1]}")
if same != 3: fail = 1

# The folded row: a run that carries taps may only be compared with a run that carries the same ones.
# This is deliberately not "compare the taps if they happen to be there": a folded log against a non-folded
# one is a comparison of two different questions, which is the failure this row exists to prevent.
if ha or hb:
    if not (ha and hb):
        print(f"FAIL one side is a folded run and the other is not "
              f"(layers {ha.get('layer')} / {hb.get('layer')})"); fail = 1
    elif ha["layer"] != hb["layer"]:
        print(f"FAIL tap layers differ: {ha['layer']} vs {hb['layer']}"); fail = 1
    elif not ka or len(ka) != len(kb):
        print(f"FAIL tap lines: A {len(ka)}, B {len(kb)} (must be equal and non-empty)"); fail = 1
    elif ka != kb:
        for i, (x, y) in enumerate(zip(ka, kb)):
            if x != y:
                print(f"FAIL tap {i} differs:")
                print(f"   A {x[:200]}")
                print(f"   B {y[:200]}")
                break
        fail = 1
    else:
        print(f"taps: {len(ka)} lines identical (layer {ha['layer']}, "
              f"{sum(1 for t in ka if t.startswith('[pb16i-trace]'))} launches + "
              f"{sum(1 for t in ka if t.startswith('[qkv-dump]'))} qkv dumps)")

rows_a, rows_b = set(pa), set(pb)
if rows_a != rows_b:
    print(f"FAIL prof rows differ: only in A {sorted(rows_a - rows_b)}, only in B {sorted(rows_b - rows_a)}")
    fail = 1
else:
    print(f"prof rows: {len(rows_a)} identical (phase, role, M)")
if not paired_ok:
    print("note: one round per row in these logs, so median-of-ratios == ratio-of-medians; concatenate each"
          " arm's rounds (alternating A B / B A inside a round) to get the paired statistic")
if tol_stat is not None and tol_stat != stat_name:
    print(f"warning: the control was paired ({tol_stat}) but these logs are not; using {stat_name}")

print(f"statistic in force: {stat_name}   tolerance {tol:g}% from {tol_from}")
for k in sorted(rows_a & rows_b):
    va, vb = statistics.median(pa[k]), statistics.median(pb[k])
    med, paired = stat_row(pa[k], pb[k], paired_ok)
    used = abs(paired) if (paired_ok and paired is not None) else abs(med)
    flag = "" if used <= tol else "  <-- outside tolerance"
    if not tol_scored and used > tol:
        flag = "  (reported, not scored on the folded row: engine#834)"
    shown = f"ratio-of-medians {med:+6.1f}%"
    if paired_ok and paired is not None:
        shown += f"  median-of-ratios {paired:+6.1f}%"
    print(f"  {k[0]:3} {k[1]:5} M={k[2]:3} {va:8.3f} -> {vb:8.3f} ms  {shown}{flag}")
    if used > tol and tol_scored: fail = 1
print("window11: PASS" if not fail else "window11: FAIL")
sys.exit(1 if fail else 0)
PY
}

if [ "${1:-}" = "--compare" ]; then
    shift
    compare "$@"
    exit $?
fi

if [ "${1:-}" = "--self-test" ]; then
    exec python3 - "$0" <<'PY'
# The policy stays falsifiable without a device: write small logs, then check that a comparison with no
# tolerance is refused, that an explicit tolerance is honoured and named, that a control pair sets the
# tolerance from its own worst row, and that the paired statistic is used when the logs carry rounds.
import os, subprocess, sys, tempfile

script = sys.argv[1]
d = tempfile.mkdtemp(prefix="window11-selftest-")
failures = 0


def write(name, rows, texts=("a", "b", "c"), extra_rounds=0):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        for _ in range(1 + extra_rounds):
            for i, t in enumerate(texts, 1):
                f.write(f"text {i} sha256={t} chars=1 text=x\n")
            for key, v in rows.items():
                f.write(f"prof {key[0]} {key[1]} M={key[2]} ms={v:.3f} launches=100\n")
    return p


keys = [("dec", "QKV", "16"), ("pre", "GU", "128")]
base = {keys[0]: 2.000, keys[1]: 5.000}


def run(*args):
    r = subprocess.run(["bash", script, "--compare", *args], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def check(ok, what, out=""):
    global failures
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        failures += 1
        print("     " + out.replace("\n", "\n     ")[:400])


a = write("a.log", base)
b = write("b.log", {keys[0]: 2.100, keys[1]: 5.150})       # +5 % and +3 %
c1 = write("c1.log", base)
c2 = write("c2.log", {keys[0]: 2.154, keys[1]: 5.050})      # same binary: worst row 7.7 %
bad = write("bad.log", {keys[0]: 2.600, keys[1]: 5.100})    # +30 % and +2 %

rc, out = run(a, b)
check(rc == 2 and "no tolerance given" in out, "no tolerance and no control is refused", out)
rc, out = run(a, b, "--tol", "10")
check(rc == 0 and "statistic in force: ratio-of-medians" in out and "tolerance 10% from --tol" in out,
      "--tol 10 passes a 5 % row and names the statistic", out)
rc, out = run(a, bad, "--tol", "10")
check(rc == 1 and "outside tolerance" in out, "--tol 10 still fails a 30 % row", out)
rc, out = run(a, b, "--control", c1, c2)
check(rc == 0 and "tolerance 8% from control" in out and "worst row 7.7%" in out,
      "the control pair sets the tolerance from its own worst row (7.7 -> 8 %)", out)
rc, out = run(a, bad, "--control", c1, c2)
check(rc == 1, "a 30 % row still fails at the control's 8 %", out)

# two rounds per arm whose per-round ratios (0.90, 1.20) are not the ratio of medians
pa = os.path.join(d, "pa.log")
with open(pa, "w") as f:
    for r in range(2):
        for i, t in enumerate(("a", "b", "c"), 1):
            f.write(f"text {i} sha256={t} chars=1 text=x\n")
        for key in keys:
            f.write(f"prof {key[0]} {key[1]} M={key[2]} ms={base[key]:.3f} launches=100\n")
pb = os.path.join(d, "pb.log")
with open(pb, "w") as f:
    for r, fac in enumerate((0.90, 1.20)):
        for i, t in enumerate(("a", "b", "c"), 1):
            f.write(f"text {i} sha256={t} chars=1 text=x\n")
        for key in keys:
            f.write(f"prof {key[0]} {key[1]} M={key[2]} ms={base[key] * fac:.3f} launches=100\n")
rc, out = run(pa, pb, "--tol", "10")
check("statistic in force: median-of-ratios" in out and "median-of-ratios" in out,
      "two rounds per row: the paired statistic is named and printed", out)
print()
print("self-test: " + ("PASS" if not failures else "FAIL"))
sys.exit(1 if failures else 0)
PY
fi

MODEL="${1:-}"; KERNELS="${2:-}"; LABEL="run"; OUT=""; PORT=8110; MAXTOK=8; TOKSET=0; FOLDED=0; TAPLAYER=0
shift 2 2>/dev/null || true
while [ $# -gt 0 ]; do
    case "$1" in
        --label) LABEL="${2:-}"; shift 2;;
        --out)   OUT="${2:-}";   shift 2;;
        --port)  PORT="${2:-}";  shift 2;;
        --max-tokens) MAXTOK="${2:-}"; TOKSET=1; shift 2;;
        --folded) FOLDED=1; shift;;
        --tap-layer) TAPLAYER="${2:-0}"; shift 2;;
        *) echo "unknown option $1" >&2; exit 2;;
    esac
done
[ -n "$MODEL" ] && [ -n "$KERNELS" ] || { echo "usage: $0 <model.q4nx> <kernels> --label NAME --out FILE" >&2; exit 2; }
# A folded container is a 27B thinking model: at the 0.6B row's 8 tokens every reply is the same preamble stub, so the
# text signal would compare nothing (measured 2026-11-03: prompts 1 and 2 came back with the same sha).  32 tokens
# gets past the preamble for ~40 s of decode per arm; the caller can still ask for less.
if [ "$FOLDED" = 1 ] && [ "$TOKSET" = 0 ]; then MAXTOK=32; fi
[ -n "$OUT" ] || OUT="/tmp/window11-$LABEL.log"
BIN="${ONEBIT_BIN:-$(dirname "$0")/../1bit}"
[ -x "$BIN" ] || { echo "no 1bit at $BIN (set ONEBIT_BIN)" >&2; exit 2; }

: > "$OUT"
# --folded: the container's own debug taps, so the row compares numbers and not just plausible text.  Both
# taps already exist in the forward; setting them here means neither arm of a comparison is instrumented
# differently (engine#876).
TAPENV=()
if [ "$FOLDED" = 1 ]; then
    TAPENV=(NPU_PB16I_TRACE="$TAPLAYER" NPU_QKV_DUMP="$TAPLAYER")
    echo "folded layer=$TAPLAYER taps=0" > "$OUT"     # rewritten below with the real count
fi
# `env` and not a bare array expansion: in command position bash reads `NPU_PB16I_TRACE=0` as the command name
# (measured: every arm "served" for 0 s and the log said `NPU_PB16I_TRACE=0: command not found`).
NPU_PROFILE=1 NPU_NO_DEVICE_LOCK=1 env "${TAPENV[@]}" "$BIN" forward-serve -m "$MODEL" --kernels "$KERNELS" \
    -p "$PORT" > "$OUT.serve" 2>&1 &
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

# The folded row's taps: every [pb16i-trace] / [qkv-dump] line the serve emitted, in order, whitespace
# normalised (the forward pads its fields for a human; two runs that compute the same numbers differ only in
# that padding).  Written after the prof rows so the header line can carry the real count.
if [ "$FOLDED" = 1 ]; then
    python3 - "$OUT.serve" "$OUT" "$TAPLAYER" <<'PY'
import re, sys
serve, out, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
taps, launches, dumps = [], 0, 0
for line in open(serve, encoding="utf-8", errors="replace"):
    if line.startswith("[pb16i-trace]") or line.startswith("[qkv-dump]"):
        taps.append(re.sub(r"\s+", " ", line).strip())
        launches += line.startswith("[pb16i-trace]")
        dumps += line.startswith("[qkv-dump]")
lines = open(out, encoding="utf-8").read().splitlines()
lines[0] = f"folded layer={layer} taps={len(taps)}"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
    for t in taps:
        f.write(f"tap {t}\n")
print(f"{len(taps)} tap lines ({launches} launches, {dumps} qkv dumps) at layer {layer}")
if not taps:
    print("wrote NO taps: --folded on a container with no fold would compare nothing", file=sys.stderr)
PY
fi

grep -c '^text ' "$OUT" | sed 's/^/transcripts: /'
