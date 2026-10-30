#!/bin/bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# ceiling.sh -- phase 1 of the 2026-10-30 14:58 addendum: measure the attainable
# weight-stream bandwidth for EACH phase's tap/tile geometry with the matrix unit
# replaced by a pass-through kernel (probe/mm_dmabw.cc), 3 reps each, and write
# CEILING.md.
#
# The probe keeps the phase's objectfifos, taps, blocked B layout, tile geometry
# and column-group loop, so its GB/s is the ceiling that phase's bytes can reach
# on this box -- the number every real kernel entry is judged against.
#
# Variants: nm16 = M=16 / 1 core row (8 cores, the shipped twin's shape)
#           r4   = M=64 / 4 core rows (32 cores, same B taps, B broadcast down rows)
#
# Device work only when the box is mine: run with ARENA_INSIDE_HOLD=1 from inside
# a gpu-hold, or let the script take it itself.
set -u
ARENA=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$ARENA/lib/env.sh"
REPS=${REPS:-3}
ITERS=${ITERS:-5}
PHASES=${PHASES:-"GU D QKV O Z"}
OUT=$ARENA/CEILING.md

if [ -n "$(fuser /dev/accel/accel0 2>/dev/null)" ]; then
    echo "ceiling: /dev/accel/accel0 is held by $(fuser /dev/accel/accel0 2>/dev/null)"; exit 9
fi
if [ "${ARENA_INSIDE_HOLD:-0}" = 1 ]; then RUN=(); else RUN=("$GPU_HOLD" -l "npu arena" -t 25 --); fi

tmp=$(mktemp -d /tmp/ceiling.XXXXXX)
for T in $PHASES; do
    for V in nm16 r4; do
        D=$ARENA/probe/$T-$V
        [ -s "$D/out/$T.elf" ] || { echo "ceiling: no $D/out/$T.elf (run make_probe.sh + build)"; continue; }
        wargs=$(bash "$D/layout.sh" "$ARENA/tasks/$T/harness/seed1_B.bin" "$tmp/B.bin" \
                    "$(sed -n 's/^K //p' "$ARENA/tasks/$T/harness/geometry.txt")" \
                    "$(sed -n 's/^N //p' "$ARENA/tasks/$T/harness/geometry.txt")" >/dev/null 2>&1 \
                && echo "--w $tmp/B.bin")
        for r in $(seq 1 "$REPS"); do
            "${RUN[@]}" "$ARENA/runner/runner" --elf "$D/out/$T.elf" --task "$ARENA/tasks/$T" \
                --seed 1 --iters "$ITERS" --no-check $wargs \
                | grep '^RESULT' | sed "s/^/  /" >> "$tmp/$T-$V.log"
        done
    done
done

python3 - "$tmp" "$OUT" <<'PY'
import pathlib, re, statistics, sys
tmp, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rows = {}
for f in sorted(tmp.glob("*.log")):
    phase, variant = f.stem.split("-")
    gbs, ms = [], []
    for line in f.read_text().splitlines():
        m = re.search(r"gbsB=([0-9.]+).*?med_ms=([0-9.]+)", line)
        if not m:
            m2 = re.search(r"med_ms=([0-9.]+).*?gbsB=([0-9.]+)", line)
            if not m2:
                continue
            ms.append(float(m2.group(1))); gbs.append(float(m2.group(2)))
        else:
            gbs.append(float(m.group(1))); ms.append(float(m.group(2)))
    if gbs:
        rows[(phase, variant)] = (statistics.median(gbs), statistics.median(ms), len(gbs))
lines = ["# DMA ceiling per phase (phase 1 of the 14:58 addendum)", "",
         "Weight-stream bandwidth with the matrix unit replaced by a pass-through kernel",
         "(`~/wt/npu-arena/probe/mm_dmabw.cc`): the phase's own objectfifos, blocked-B taps,",
         "tile geometry and column-group loop, no arithmetic. Median of "
         f"{rows and next(iter(rows.values()))[2] or 0} device reps per cell, gpu-hold, box otherwise idle.",
         "",
         "`nm16` = M=16 / 1 core row (8 cores, the shipped decode twin's shape);",
         "`r4` = M=64 / 4 core rows (32 cores, the same B taps with B broadcast down the",
         "four rows -- this is what the generator can place today; the real 32-core decode",
         "needs N partitioned across rows, phase 2).",
         "",
         "| phase | B streamed | nm16 GB/s | nm16 ms | r4 GB/s (32 cores) | r4 ms | r4 / nm16 |",
         "|---|---|---|---|---|---|---|"]
for (phase, variant) in sorted(rows):
    pass
for phase in ["GU", "D", "QKV", "O", "Z"]:
    a = rows.get((phase, "nm16")); b = rows.get((phase, "r4"))
    if not a and not b:
        continue
    gb = {"GU": 89.13, "D": 89.13, "QKV": 52.43, "O": 31.46, "Z": 31.46}[phase]
    ratio = f"{b[0]/a[0]:.2f}x" if a and b else "-"
    lines.append(f"| {phase} | {gb:.2f} MB | {a[0]:.2f} | {a[1]:.2f} | {b[0] if b else 0:.2f} | "
                 f"{b[1] if b else 0:.2f} | {ratio} |")
lines += ["", "Reference points: the dx route's own in-loop weight stream on Qwen3-1.7B/4B measured",
          "45 GB/s (docs/npu-dx.md) and 55-59 GB/s effective (addendum 14:58); the Bonsai blocked-B",
          "kernel runs at 8.1-8.4 GB/s in the arena, i.e. the same class as this ceiling divided by",
          "the number below.", ""]
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
rm -rf "$tmp"
