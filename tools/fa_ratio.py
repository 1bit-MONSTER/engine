#!/usr/bin/env python3
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
"""Is the HRX flash-attention path worth anything for prefill, measured so the answer survives a busy box?

engine#1517 needs a number for the FA path inside prefill before anyone changes the kernel. Two ways to get one are
closed on this hardware and both cost a ticket to discover: `rocprofv3` records nothing for HRX (the runtime dlopens its
own ROCr and writes its own AQL packets), and the runtime's own HAL profiler (`HRX_PROFILE_MODE=dispatch`) produces no
file on a shipped payload even with `HRX_GPU_DEBUG=1`, because the sink-owning path is not executed there. What is left is
an end-to-end comparison — the same model and shape with `-fa 1` and with `-fa 0` — and that comparison is a trap on a
shared box: levels move by 10x inside one ticket, so a ratio computed from arm *levels* taken minutes apart is noise.

So this tool compares **pairs**. Each pair is one `-fa 1` run and one `-fa 0` run taken back to back, with the pair order
alternating so monotonic drift cancels instead of accumulating, and the estimator is the median of the per-pair ratios.
A run is then only quoted when its own pairs agree: `--tolerance` bounds `max/min` of the ratios, and a run that fails
prints VOID and exits 3. On this box the measured prefill ratio was 1.273x across four pairs (spread 1.08x, ACCEPTED) —
and a run taken at load 73 came back with a 9.29x spread, which this tool refuses to report as a number.

`-fa 0` is only usable below roughly 5.4k context on our route (it segfaults above that), so the default shape is
pp2048 with a 0.6B model: small enough to fit a quiet window, long enough for attention to matter.

Usage:
    tools/fa_ratio.py --llama-bench <bin> --model <gguf> [--pairs 4] [--reps 2] [--tolerance 1.25]
    tools/fa_ratio.py --self-test          # device-free: exercises the estimator and its acceptance verdict
"""
import argparse
import math
import re
import shutil
import statistics
import subprocess
import sys

EXIT_ACCEPTED = 0
EXIT_ERROR = 2
EXIT_VOID = 3

# A row of llama-bench output looks like:
#   | qwen3 0.6B Q4_K - Medium | 372.65 MiB | 596.05 M | HRX | 99 | 1 | pp2048 | 1718.51 ± 0.00 |
_TOKENS_PER_SECOND = re.compile(r"\|\s*([0-9]+(?:\.[0-9]+)?)\s*±\s*[0-9.]+\s*\|\s*$")


def parse_rate(line):
    """-> the t/s of one llama-bench row, or None when the line is not a row."""
    match = _TOKENS_PER_SECOND.search(line)
    return float(match.group(1)) if match else None


def pair_ratios(runs):
    """-> [(pair_index, fa_on, fa_off, ratio)] from [(arm, rate)] in run order.

    Each pair is two adjacent runs. The arms are paired by their *labels*, not by their position, because a run whose row
    failed to parse drops out and would otherwise shift every later pair's arm assignment by one.
    """
    ratios = []
    pending = None
    for arm, rate in runs:
        if rate is None or rate <= 0.0:
            pending = None
            continue
        if pending is not None and pending[0] != arm:
            first_arm, first_rate = pending
            on, off = (first_rate, rate) if first_arm == "fa1" else (rate, first_rate)
            ratios.append((len(ratios) + 1, on, off, on / off))
            pending = None
        else:
            pending = (arm, rate)
    return ratios


def summarise(ratios, tolerance):
    """-> (verdict, text lines). Verdict is 'accepted', 'void' or 'insufficient'.

    A single pair is never accepted: with one pair there is no way to tell a ratio from a fluctuation, which is the
    mistake that produced a table of three mutually contradictory numbers before this tool existed.
    """
    lines = [f"pair {i}: fa1/fa0 = {r:.3f}x   (fa1 {on:.2f}, fa0 {off:.2f})" for i, on, off, r in ratios]
    if not ratios:
        return "insufficient", lines + ["no usable pairs: both arms must print a positive t/s"]
    values = [r for _, _, _, r in ratios]
    median = statistics.median(values)
    spread = max(values) / min(values)
    geometric = math.exp(statistics.median([math.log(v) for v in values]))
    lines.append("")
    lines.append(
        f"paired estimator: median ratio {median:.3f}x, median-of-log-ratios {geometric:.3f}x, "
        f"range {min(values):.3f}-{max(values):.3f}x (n={len(values)})"
    )
    lines.append(f"implied saving at the median: {(1.0 - 1.0 / median) * 100.0:.1f}% of the -fa 0 time")
    if len(values) < 3:
        lines.append(f"CONSISTENCY: only {len(values)} pair(s); three are the minimum for a stable median -> VOID")
        return "void", lines
    if spread > tolerance:
        lines.append(
            f"CONSISTENCY: spread {spread:.2f}x over {len(values)} pairs exceeds {tolerance:.2f}x -> VOID (do not quote)"
        )
        return "void", lines
    lines.append(f"CONSISTENCY: spread {spread:.2f}x over {len(values)} pairs -> ACCEPTED")
    return "accepted", lines


def bench_arms(llama_bench, model, n_prompt, reps, pairs, extra):
    """Run the alternating arms and return the rates in run order (None where a row failed to parse)."""
    runs = []
    for index in range(pairs):
        on_first = index % 2 == 0
        for arm in ("-fa 1", "-fa 0") if on_first else ("-fa 0", "-fa 1"):
            command = [
                llama_bench, "-m", model, "-p", str(n_prompt), "-n", "0",
                "-r", str(reps), "-ngl", "99",
            ] + arm.split() + extra
            print(f"$ {' '.join(command)}", file=sys.stderr)
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                print(completed.stderr[-2000:], file=sys.stderr)
                raise SystemExit(f"llama-bench failed with exit {completed.returncode}")
            rate = None
            for line in completed.stdout.splitlines():
                parsed = parse_rate(line)
                if parsed is not None:
                    rate = parsed
            runs.append(("fa1" if arm == "-fa 1" else "fa0", rate))
            print(f"  -> {rate if rate is not None else 'no rate row'}", file=sys.stderr)
    return runs


def self_test():
    """Device-free: the estimator must accept agreement and refuse disagreement."""
    failures = []

    def check(name, condition, detail=""):
        if not condition:
            failures.append(f"{name} {detail}".strip())

    # Adjacent pairs, order alternating: (1 700, 1 350) then (1 350, 1 700) is the same ratio twice.
    ratios = pair_ratios([("fa1", 1700.0), ("fa0", 1350.0), ("fa0", 1350.0), ("fa1", 1700.0)])
    check("alternating order", len(ratios) == 2 and abs(ratios[0][3] - ratios[1][3]) < 1e-9, str(ratios))
    verdict, lines = summarise(ratios, 1.25)
    check("two pairs are never enough", verdict == "void", verdict)

    agreeing = pair_ratios([("fa1", 1774.0), ("fa0", 1388.0), ("fa0", 1388.0), ("fa1", 1774.0),
                            ("fa1", 1790.0), ("fa0", 1406.0), ("fa0", 1406.0), ("fa1", 1790.0)])
    verdict, lines = summarise(agreeing, 1.25)
    check("agreement is accepted", verdict == "accepted", "\n".join(lines))
    median = statistics.median([r for _, _, _, r in agreeing])
    check("median ratio", abs(median - 1774.0 / 1388.0) < 0.01, f"{median}")

    # The shape a loaded box produces: one arm spanning 161-1773 t/s, ratios 0.16x-1.53x.
    wild = pair_ratios([("fa1", 200.36), ("fa0", 1218.58), ("fa0", 1338.99), ("fa1", 833.85),
                        ("fa1", 246.14), ("fa0", 161.19)])
    verdict, lines = summarise(wild, 1.25)
    check("disagreement is void", verdict == "void", "\n".join(lines))

    # A dropped run must not shift the arms of the following pair: without labels, (None, off, off, on) would assign
    # "on" and "off" from position and report 1350/1700 instead of 1700/1350.
    shifted = pair_ratios([("fa1", None), ("fa0", 1350.0), ("fa0", 1350.0), ("fa1", 1700.0)])
    check("dropped run does not shift arms", len(shifted) == 1 and abs(shifted[0][3] - 1700.0 / 1350.0) < 1e-9, str(shifted))
    check("zero rate skipped", pair_ratios([("fa1", 0.0), ("fa0", 1350.0)]) == [])
    verdict, _ = summarise([], 1.25)
    check("no pairs reported", verdict == "insufficient", verdict)

    row = "| qwen3 0.6B Q4_K - Medium       | 372.65 MiB |   596.05 M | HRX        |  99 |   1 |          pp2048 |       1718.51 ± 0.00 |"
    check("row parse", parse_rate(row) == 1718.51, str(parse_rate(row)))
    check("non-row parse", parse_rate("build: d8bd0ff (1)") is None)

    if failures:
        print("fa_ratio self-test FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("fa_ratio self-test: ok (estimator, alternating pairs, acceptance verdict, row parsing)")
    return 0


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="exercise the estimator without a GPU")
    parser.add_argument("--llama-bench", help="path to llama-bench from an HRX build")
    parser.add_argument("--model", help="GGUF to measure")
    parser.add_argument("--n-prompt", type=int, default=2048, help="prefill length; -fa 0 segfaults above ~5.4k")
    parser.add_argument("--pairs", type=int, default=4, help="pairs of (fa 1, fa 0) runs; 3 is the minimum accepted")
    parser.add_argument("--reps", type=int, default=2, help="llama-bench repetitions inside one arm")
    parser.add_argument("--tolerance", type=float, default=1.25, help="max/min of the pair ratios that still counts")
    parser.add_argument("--max-load", type=float, default=60.0, help="refuse to start above this 1-min load average")
    parser.add_argument("--extra", default="", help="extra llama-bench flags, quoted")
    arguments = parser.parse_args(argv)

    if arguments.self_test:
        return self_test()
    if not arguments.llama_bench or not arguments.model:
        parser.error("--llama-bench and --model are required (or --self-test)")

    load = 0.0
    try:
        with open("/proc/loadavg", encoding="utf-8") as handle:
            load = float(handle.read().split()[0])
    except OSError:
        pass
    if load > arguments.max_load:
        print(f"refusing to measure at load {load:.2f} > {arguments.max_load:.2f}: a saturated box is what makes"
              " the ratio unquotable", file=sys.stderr)
        return EXIT_VOID
    for tool in (arguments.llama_bench, arguments.model):
        if not shutil.os.path.exists(tool):
            parser.error(f"not found: {tool}")

    runs = bench_arms(arguments.llama_bench, arguments.model, arguments.n_prompt, arguments.reps,
                      arguments.pairs, arguments.extra.split())
    verdict, lines = summarise(pair_ratios(runs), arguments.tolerance)
    print("\n".join(lines))
    return {"accepted": EXIT_ACCEPTED, "void": EXIT_VOID}.get(verdict, EXIT_VOID)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
