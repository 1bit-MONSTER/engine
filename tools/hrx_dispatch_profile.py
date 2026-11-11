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
"""hrx_dispatch_profile: time per HRX dispatch, from the backend's own serial trace (engine#679).

  hrx_dispatch_profile.py run [--log FILE] [--top N] [--json FILE] -- llama-bench -m M.gguf -ngl 99 ...
  hrx_dispatch_profile.py parse LOG [--top N] [--json FILE]
  hrx_dispatch_profile.py --self-test

The HRX backend has a serial mode, GGML_HRX_DEBUG_SERIAL_EXECUTION=1: it waits for the device after every dispatch
and prints one line when the dispatch is launched and one when the device has finished it. The lines carry the
kernel's id but no time and no name. `run` sets that switch, stamps every trace line with a monotonic time as it
arrives, and adds the two up per kernel; the names come from the kernel manifests of the pinned llama.cpp fork.

What the numbers are: the wall time from "dispatch launched" to "device idle", one dispatch at a time. That is the
kernel's device time plus one wake-up of the waiting thread, so a sum over a graph is larger than the same graph in
a normal run, where dispatches are queued back to back. Use it to see WHICH kernel a graph's time is in and how
that changes with the context length, not as a tok/s figure. The serial mode slows decode much more than prefill,
so compare shares inside one program, never between programs. Run the command under gpu-hold like any device run.

Times are milliseconds from time.monotonic_ns() (CLOCK_MONOTONIC, nanosecond steps), read when the trace line arrives.

A kept trace (`run --log`, the input of `parse`) is one line per trace line, the arrival time in ns and the line:
  <ns> hrx debug serial: begin phase=command-dispatch list=main index=0 list_commands=612 ordinal=0 kind=Kernel kernel_id=<id> bindings=3
  <ns> hrx debug serial: end phase=command-dispatch ...                 the launch returned
  <ns> hrx debug serial: sync-end phase=command-post-dispatch ...       the device finished it
A dispatch is the three lines with the same list, list_commands and index. Lines of other phases are ignored.

Kernels of a build with extra kernel directories are not in the fork's manifests and show as a numeric id; give
their directory with a second --manifests.
"""
import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys
import time

TRACE_PREFIX = "hrx debug serial: "
DEFAULT_MANIFESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "llama.cpp", "ggml",
                                 "src", "ggml-hrx", "kernel-corpus", "kernels")
# manifests that do not name a family per export use the directory's catalog family
DIRECTORY_FAMILY = {"qwen_moe": "qwen3_moe", "qwen": "qwen", "hrx": "hrx", "loom-libs": "loom_libs"}


def kernel_id(family, name):
    """kernel_catalog_id() of the fork's kernel-corpus-catalog.h: FNV-1a over family, a zero byte, name."""
    mask, prime = (1 << 64) - 1, 1099511628211
    value = 1469598103934665603
    for byte in family.encode():
        value = ((value ^ byte) * prime) & mask
    value = (value * prime) & mask
    for byte in name.encode():
        value = ((value ^ byte) * prime) & mask
    return value


def load_kernel_names(manifest_dir):
    names = {}
    for path in sorted(glob.glob(os.path.join(manifest_dir, "*", "manifest.json"))):
        directory = os.path.basename(os.path.dirname(path))
        with open(path) as handle:
            manifest = json.load(handle)
        for export in manifest.get("exports", []):
            family = export.get("family") or DIRECTORY_FAMILY.get(directory, directory)
            names[kernel_id(family, export["name"])] = family + ":" + export["name"]
    return names


def parse_fields(text):
    return dict(re.findall(r"(\w+)=(\S+)", text))


def profile(lines):
    """lines: iterable of (time_ns, trace text without the prefix).
    Returns (per-program per-kernel totals, kernel ids of the dispatches that were launched and never finished)."""
    programs = collections.OrderedDict()
    open_dispatch = {}
    for time_ns, text in lines:
        event, _, rest = text.partition(" ")
        fields = parse_fields(rest)
        phase = fields.get("phase")
        if "kernel_id" not in fields or "index" not in fields:
            continue
        key = (fields.get("list"), fields.get("list_commands"), fields["index"])
        if event == "begin" and phase == "command-dispatch":
            open_dispatch[key] = [time_ns, None, int(fields["kernel_id"])]
        elif event == "end" and phase == "command-dispatch" and key in open_dispatch:
            open_dispatch[key][1] = time_ns
        elif event == "sync-end" and phase == "command-post-dispatch" and key in open_dispatch:
            begin_ns, launched_ns, _ = open_dispatch.pop(key)
            program = programs.setdefault((fields.get("list"), int(fields.get("list_commands", 0))), {})
            row = program.setdefault(int(fields["kernel_id"]), {"dispatches": 0, "total_ns": 0, "launch_ns": 0})
            row["dispatches"] += 1
            row["total_ns"] += time_ns - begin_ns
            row["launch_ns"] += (launched_ns if launched_ns is not None else begin_ns) - begin_ns
    return programs, [entry[2] for entry in open_dispatch.values()]


def report(programs, names, top, out=sys.stdout, unfinished=(), lines_read=None):
    result = []
    if programs:
        print("times: ms, time.monotonic_ns() at the arrival of each trace line; launch to device-idle, serial mode",
              file=out)
    for (list_kind, command_count), kernels in programs.items():
        total_ns = sum(row["total_ns"] for row in kernels.values())
        dispatches = sum(row["dispatches"] for row in kernels.values())
        runs = dispatches / command_count if command_count else 0.0
        print(f"\nprogram list={list_kind} commands={command_count}: {dispatches} dispatches "
              f"({runs:.1f} runs), {total_ns / 1e6:.1f} ms in total", file=out)
        print(f"  {'ms':>10} {'share':>6} {'count':>7} {'ms each':>9} {'launch ms':>10}  kernel", file=out)
        rows = sorted(kernels.items(), key=lambda item: -item[1]["total_ns"])
        entry = {"list": list_kind, "commands": command_count, "dispatches": dispatches, "total_ms": total_ns / 1e6,
                 "kernels": []}
        for index, (kid, row) in enumerate(rows):
            name = names.get(kid, str(kid))
            share = row["total_ns"] / total_ns if total_ns else 0.0
            entry["kernels"].append({"kernel": name, "dispatches": row["dispatches"], "ms": row["total_ns"] / 1e6,
                                     "launch_ms": row["launch_ns"] / 1e6, "share": share})
            if index < top:
                print(f"  {row['total_ns'] / 1e6:10.2f} {share:6.1%} {row['dispatches']:7d} "
                      f"{row['total_ns'] / 1e6 / row['dispatches']:9.3f} {row['launch_ns'] / 1e6:10.2f}  {name}",
                      file=out)
        if len(rows) > top:
            print(f"  ... {len(rows) - top} more kernels (--top)", file=out)
        result.append(entry)
    if unfinished:
        kernels = sorted({names.get(kid, str(kid)) for kid in unfinished})
        print(f"\nnote: {len(unfinished)} dispatch(es) were launched and never finished (the run was killed or the "
              f"trace is cut). Their time is NOT in the tables: {', '.join(kernels)}", file=out)
    if not result:
        if lines_read is not None and lines_read[1] == 0:
            print(f"no dispatch was traced: {lines_read[0]} line(s) read, none is a time-stamped "
                  f"'{TRACE_PREFIX.strip()}' line. Is this a trace kept by `run --log`?", file=out)
        else:
            print("no dispatch was traced: is the binary an HRX build, and did the run reach the device?", file=out)
    return result


def read_log(path):
    """Returns (trace lines, lines read)."""
    lines, read = [], 0
    with open(path, errors="replace") as handle:
        for line in handle:
            read += 1
            stamp, _, text = line.rstrip("\n").partition(" ")
            if stamp.isdigit() and text.startswith(TRACE_PREFIX):
                lines.append((int(stamp), text[len(TRACE_PREFIX):]))
    return lines, read


def run_command(command, log_path):
    env = dict(os.environ, GGML_HRX_DEBUG_SERIAL_EXECUTION="1")
    lines = []
    log = open(log_path, "w") if log_path else None
    process = subprocess.Popen(command, env=env, stderr=subprocess.PIPE, text=True, errors="replace", bufsize=1)
    try:
        for line in process.stderr:
            now = time.monotonic_ns()
            if line.startswith(TRACE_PREFIX):
                lines.append((now, line[len(TRACE_PREFIX):].rstrip("\n")))
                if log:
                    log.write(f"{now} {line}")
            else:
                sys.stderr.write(line)
    finally:
        process.wait()
        if log:
            log.close()
    return lines, process.returncode


def self_test():
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append("%s: got %r, want %r" % (name, got, want))

    check("catalog id hash", kernel_id("loom_libs", "ggml_mul_mat_f32_f32_wmma"), 5916100870017593149)
    detail = "list=main index={i} list_commands=2 ordinal={i} kind=Kernel kernel_id={k} bindings=3"
    trace = []
    clock = 1000
    for run in range(2):
        for index, (kid, launch, device) in enumerate([(7, 10, 90), (9, 5, 20)]):
            text = detail.format(i=index, k=kid)
            trace.append((clock, "sync-begin phase=command-pre-dispatch " + text))
            trace.append((clock, "sync-end phase=command-pre-dispatch " + text))
            trace.append((clock, "begin phase=command-dispatch " + text))
            trace.append((clock + launch, "end phase=command-dispatch " + text))
            trace.append((clock + launch, "sync-begin phase=command-post-dispatch " + text))
            trace.append((clock + launch + device, "sync-end phase=command-post-dispatch " + text))
            clock += launch + device + 1000  # host time between dispatches is not charged to a kernel
    trace.append((clock, "begin phase=command-list list=main list_commands=2 main_commands=2"))
    programs, unfinished = profile(trace)
    check("programs", list(programs), [("main", 2)])
    check("slow kernel", programs.get(("main", 2), {}).get(7), {"dispatches": 2, "total_ns": 200, "launch_ns": 20})
    check("fast kernel", programs.get(("main", 2), {}).get(9), {"dispatches": 2, "total_ns": 50, "launch_ns": 10})
    check("all finished", unfinished, [])

    class Sink:
        def __init__(self):
            self.text = ""

        def write(self, text):
            self.text += text

    sink = Sink()
    result = report(programs, {7: "f:slow", 9: "f:fast"}, 1, out=sink)
    check("order", [k["kernel"] for k in result[0]["kernels"]] if result else None, ["f:slow", "f:fast"])
    check("share", round(result[0]["kernels"][0]["share"], 9) if result else None, 0.8)
    check("units in the header", "time.monotonic_ns()" in sink.text, True)

    # a dispatch that is launched and never finished is named, not dropped
    cut = trace[:6] + [(9000, "begin phase=command-dispatch " + detail.format(i=1, k=9)),
                       (9005, "end phase=command-dispatch " + detail.format(i=1, k=9))]
    programs, unfinished = profile(cut)
    check("unfinished dispatch", unfinished, [9])
    check("finished one still counted", programs.get(("main", 2), {}).get(7, {}).get("dispatches"), 1)
    sink = Sink()
    report(programs, {9: "f:fast"}, 5, out=sink, unfinished=unfinished)
    check("unfinished is reported", "1 dispatch(es) were launched and never finished" in sink.text
          and "f:fast" in sink.text, True)

    # nothing traced: a file that is not a trace is told apart from a run that reached no device
    sink = Sink()
    check("empty result", report({}, {}, 5, out=sink, lines_read=(3, 0)), [])
    check("wrong file message", "3 line(s) read, none is a time-stamped" in sink.text, True)
    sink = Sink()
    report({}, {}, 5, out=sink, lines_read=(3, 2))
    check("no device message", "is the binary an HRX build" in sink.text, True)

    for fail in fails:
        print("FAIL " + fail)
    print("hrx_dispatch_profile self-test: %s" % ("FAILED" if fails else "OK"))
    return 1 if fails else 0


def main():
    if "--self-test" in sys.argv[1:]:
        return self_test()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("run", "parse"):
        p = sub.add_parser(mode)
        p.add_argument("--top", type=int, default=12, help="kernels printed per program (default 12)")
        p.add_argument("--json", help="write the full table here")
        p.add_argument("--manifests", action="append",
                       help="a kernel-corpus/kernels directory; repeat it for a build with extra kernel directories "
                            "(default: the pinned fork's)")
        if mode == "run":
            p.add_argument("--log", help="keep the time-stamped trace here (on disk, not /tmp)")
            p.add_argument("command", nargs=argparse.REMAINDER, help="-- CMD ARGS...")
        else:
            p.add_argument("log")
    args = parser.parse_args()
    names = {}
    for directory in args.manifests or [DEFAULT_MANIFESTS]:
        names.update(load_kernel_names(directory))
    if not names:
        print(f"note: no kernel manifest under {args.manifests or DEFAULT_MANIFESTS}; kernels are shown by id",
              file=sys.stderr)
    code = 0
    lines_read = None
    if args.mode == "run":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("run needs a command after --")
        lines, code = run_command(command, args.log)
    else:
        lines, read = read_log(args.log)
        lines_read = (read, len(lines))
    programs, unfinished = profile(lines)
    result = report(programs, names, args.top, unfinished=unfinished, lines_read=lines_read)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=1)
    return code if code else (0 if result else 1)


if __name__ == "__main__":
    sys.exit(main())
