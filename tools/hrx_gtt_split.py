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
"""hrx_gtt_split: where one HRX process's GTT goes, from its own log (engine#856).

  hrx_gtt_split.py run [--out DIR] [--hz N] -- llama-cli -m M.gguf -ngl 99 ...   run it, sample its GTT, print the split
  hrx_gtt_split.py parse LOG [--peak-mib N | --samples FILE]                     the split of a log you already have
  hrx_gtt_split.py --self-test

engine#856 measured the same weights resident twice in GTT. Its reproduction named `tools/bo_trace.py`, which needs
root for tracefs and was never committed, so nobody could re-measure the issue from a clean checkout. Every number in
the split below comes from lines llama.cpp and HRX already print, plus the process's own fdinfo:

  loader's model buffer   `load_tensors: HRX0 model buffer size = X MiB` (one per HRX buffer type, summed)
  HRX's packed copies     the largest `hrx host weights: ... resident=X MiB allocs=N` (GGML_HRX_HOST_WEIGHT_TRACE=1):
                          the weights HRX re-laid-out for a kernel, held next to the loader's copy
  KV / recurrent state    `KV buffer size =`, `RS buffer size =` on an HRX buffer type
  compute + output        `compute buffer size =`, `output buffer size =` on an HRX buffer type
  process peak            the largest sum of `drm-total-gtt` over the process's amdgpu DRM clients (each client once:
                          one client can be open on several fds), sampled while it runs
  not attributed          peak minus the rows above (runtime module memory, staging, transient arenas)

`duplicate` is packed / model buffer: the share of the loader's copy that is resident a second time. Buffers on a
non-HRX type (`CPU`, `CPU_Mapped`) are host memory, are listed apart and are not in the GTT sum.

`run` needs `-v` for the buffer lines and adds it, sets GGML_HRX_HOST_WEIGHT_TRACE=1, and writes `<out>/split.log`
(the child's stderr) and `<out>/gtt.txt` (`ms MiB` per sample). It touches the device only through the command you
give it, so run it inside your gpu-hold ticket: `gpu-hold -l L -t 5 -- tools/hrx_gtt_split.py run -- llama-cli ...`.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

BUF = re.compile(r"\b(\S+)\s+(model|KV|RS|compute|output) buffer size =\s*([0-9.]+) MiB")
HOSTW = re.compile(r"hrx host weights: .*?resident=([0-9.]+) MiB allocs=(\d+)")


def is_gtt_buft(name):
    """HRX0, HRX0_HOST, HRX1 ... are GTT on this APU; CPU / CPU_Mapped / CPU_REPACK are not."""
    return name.startswith("HRX")


def parse(text):
    """Return the split as a dict; every value is MiB."""
    model = {}      # buft -> summed MiB (a buffer type may be split into several buffers)
    other = {}      # (kind, buft) -> last MiB (a context re-reserves; the last one is what is held)
    packed, packed_n, traced = 0.0, 0, False
    for line in text.splitlines():
        m = BUF.search(line)
        if m:
            buft, kind, mib = m.group(1), m.group(2), float(m.group(3))
            if kind == "model":
                model[buft] = model.get(buft, 0.0) + mib
            else:
                other[(kind, buft)] = mib
            continue
        m = HOSTW.search(line)
        if m:
            traced = True
            if float(m.group(1)) >= packed:
                packed, packed_n = float(m.group(1)), int(m.group(2))
    s = {
        "model_gtt": sum(v for k, v in model.items() if is_gtt_buft(k)),
        "model_host": sum(v for k, v in model.items() if not is_gtt_buft(k)),
        "packed": packed if traced else None,
        "packed_weights": packed_n,
        "state": sum(v for (k, b), v in other.items() if k in ("KV", "RS") and is_gtt_buft(b)),
        "compute": sum(v for (k, b), v in other.items() if k in ("compute", "output") and is_gtt_buft(b)),
        "model_bufts": model,
    }
    return s


def client_gtt_kib(fdinfo_dir):
    """Sum drm-total-gtt over the amdgpu DRM clients behind one process's fds, each client once."""
    seen, total = set(), 0
    try:
        fds = os.listdir(fdinfo_dir)
    except OSError:
        return 0
    for fd in fds:
        try:
            with open(os.path.join(fdinfo_dir, fd)) as f:
                txt = f.read()
        except OSError:
            continue
        if not re.search(r"^drm-driver:\s*amdgpu", txt, re.M):
            continue
        cid = re.search(r"^drm-client-id:\s*(\d+)", txt, re.M)
        if cid:
            if cid.group(1) in seen:
                continue
            seen.add(cid.group(1))
        g = re.search(r"^drm-total-gtt:\s*(\d+) KiB", txt, re.M) or re.search(r"^drm-memory-gtt:\s*(\d+) KiB", txt, re.M)
        total += int(g.group(1)) if g else 0
    return total


NO_TRACE = "unknown: no `hrx host weights` line (GGML_HRX_HOST_WEIGHT_TRACE off, or no weight was repacked)"


def report(s, peak):
    rows = [("loader's model buffer (HRX)", s["model_gtt"])]
    if s["packed"] is None:
        rows.append(("HRX's packed copies", None))
    else:
        rows.append(("HRX's packed copies (%d weights)" % s["packed_weights"], s["packed"]))
    rows += [("KV + recurrent state", s["state"]), ("compute + output buffers", s["compute"])]
    known = sum(v for _, v in rows if v is not None)
    out = []
    if not s["model_bufts"]:
        out += ["warning: no `model buffer size` line, so this is not a -v log of a model load", ""]
    out += ["| part | MiB |", "|---|---:|"]
    for name, v in rows:
        out.append("| %s | %s |" % (name, NO_TRACE if v is None else "%.1f" % v))
    if peak is not None:
        out.append("| not attributed | %.1f |" % (peak - known))
        out.append("| **process peak GTT** | **%.1f** |" % peak)
    else:
        out.append("| attributed total (no peak given) | %.1f |" % known)
    if s["model_host"]:
        out.append("")
        out.append("host memory, not in the GTT sum: %.1f MiB of model buffers on %s" % (
            s["model_host"], ", ".join(sorted(k for k in s["model_bufts"] if not is_gtt_buft(k)))))
    if s["packed"] is not None and s["model_gtt"]:
        out.append("duplicate: %.1f of %.1f MiB of the loader's copy is resident a second time (%.1f %%)" % (
            s["packed"], s["model_gtt"], 100.0 * s["packed"] / s["model_gtt"]))
    return "\n".join(out)


def peak_of_samples(path):
    peak = None
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                v = float(p[1])
                peak = v if peak is None or v > peak else peak
    return peak


def cmd_run(a):
    argv = list(a.command)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        sys.exit("hrx_gtt_split: run needs a command after --")
    if "-v" not in argv and "--verbose" not in argv:
        argv.append("-v")
    os.makedirs(a.out, exist_ok=True)
    log, samples = os.path.join(a.out, "split.log"), os.path.join(a.out, "gtt.txt")
    env = dict(os.environ, GGML_HRX_HOST_WEIGHT_TRACE="1")
    t0 = time.monotonic()
    with open(log, "w") as err, open(samples, "w") as gt:
        p = subprocess.Popen(argv, stderr=err, env=env)
        fdinfo = "/proc/%d/fdinfo" % p.pid
        peak = 0
        try:
            while p.poll() is None:
                kib = client_gtt_kib(fdinfo)
                peak = max(peak, kib)
                gt.write("%d %.1f\n" % (int((time.monotonic() - t0) * 1000), kib / 1024.0))
                time.sleep(1.0 / a.hz)
        except KeyboardInterrupt:
            p.kill()
            p.wait()
            raise
    with open(log) as f:
        s = parse(f.read())
    print("# %s  rc=%d  log=%s  samples=%s" % (" ".join(argv), p.returncode, log, samples))
    if not peak:
        print("warning: the process never had an amdgpu DRM client with GTT, so there is no peak to split (not an HRX run?)")
    print(report(s, peak / 1024.0 if peak else None))
    return p.returncode


def cmd_parse(a):
    with open(a.log, errors="replace") as f:
        s = parse(f.read())
    peak = a.peak_mib if a.peak_mib is not None else (peak_of_samples(a.samples) if a.samples else None)
    print(report(s, peak))
    return 0


# Lines from a real run: Qwen3-4B-Q4_K_M, -ngl 99 -fit off -c 256 -n 16 -v, GGML_HRX_HOST_WEIGHT_TRACE=1, llama.cpp
# 40f58efe1 (the #1351 pin), whose process peaked at 3925 MiB of drm-total-gtt.
FIXTURE_4B = """\
0.00.378.782 I load_tensors:          CPU model buffer size =   304.28 MiB
0.00.378.783 I load_tensors:         HRX0 model buffer size =  2375.91 MiB
0.00.402.110 I llama_context:  HRX0_HOST  output buffer size =     0.58 MiB
0.00.402.300 I llama_kv_cache:       HRX0 KV buffer size =    36.00 MiB
0.00.410.001 I sched_reserve:       HRX0 compute buffer size =    37.50 MiB
0.00.410.002 I sched_reserve:  HRX0_HOST compute buffer size =     5.13 MiB
0.02.369.640 I hrx host weight working set: 304.3 MiB accumulated, 304.3 MiB in this program
0.02.520.878 I hrx host weights: refilled=0 of 1 | hit=0 miss=1 evict=0 resident=304.3 MiB allocs=1 limit=0.0 MiB
0.03.101.877 I hrx host weights: refilled=0 of 36 | hit=1 miss=36 evict=0 resident=784.6 MiB allocs=36 limit=0.0 MiB
0.03.511.020 I hrx host weights: refilled=0 of 3 | hit=2 miss=38 evict=0 resident=795.8 MiB allocs=38 limit=0.0 MiB
0.04.900.100 D ~llama_context:       HRX0 compute buffer size is  37.5000 MiB, matches expectation of  37.5000 MiB
"""


def self_test():
    fails = []

    def check(name, got, want):
        if (abs(got - want) > 0.01) if isinstance(want, float) else (got != want):
            fails.append("%s: got %r, want %r" % (name, got, want))

    s = parse(FIXTURE_4B)
    check("model buffer on HRX", s["model_gtt"], 2375.91)
    check("CPU model buffer kept out of GTT", s["model_host"], 304.28)
    check("packed = the largest resident, not the first or the last-but-one", s["packed"], 795.8)
    check("packed weight count", s["packed_weights"], 38)
    check("KV", s["state"], 36.0)
    check("compute + output, and the '~llama_context ... size is' echo is not counted twice", s["compute"], 43.21)
    r = report(s, 3925.0)
    check("not attributed = peak - rows", "| not attributed | 674.1 |" in r, True)
    check("duplicate line", "(33.5 %)" in r, True)

    # Synthetic, not from the run: a later program reports less resident (an eviction under GGML_HRX_HOST_WEIGHT_LIMIT_MB).
    # The split wants the most that was held at once, so a smaller later line must not replace it.
    s = parse(FIXTURE_4B + "0.05.000.000 I hrx host weights: refilled=2 of 2 | hit=0 miss=2 evict=5 resident=512.0 MiB allocs=40 limit=600.0 MiB\n")
    check("packed after an eviction keeps the largest", s["packed"], 795.8)

    s = parse(FIXTURE_4B.replace("hrx host weights", "something else"))
    check("trace off: packed is unknown, not zero", s["packed"], None)
    check("trace off: the report says why", NO_TRACE in report(s, 3925.0), True)
    check("not a -v log: the report says so", "warning: no `model buffer size` line" in report(parse("hello\n"), 1.0), True)

    s = parse(FIXTURE_4B + "0.00.378.790 I load_tensors:         HRX0 model buffer size =   100.00 MiB\n")
    check("a split buffer type is summed", s["model_gtt"], 2475.91)

    with tempfile.TemporaryDirectory() as d:
        def fd(n, body):
            with open(os.path.join(d, str(n)), "w") as f:
                f.write(body)
        a = "drm-driver:\tamdgpu\ndrm-client-id:\t7\ndrm-total-gtt:\t1048576 KiB\n"
        fd(3, a)
        fd(4, a)                                                                  # same client on a second fd
        fd(5, "drm-driver:\tamdgpu\ndrm-client-id:\t9\ndrm-memory-gtt:\t2048 KiB\n")  # older kernel key
        fd(6, "pos:\t0\nflags:\t02\n")                                            # not a DRM fd
        fd(7, "drm-driver:\ti915\ndrm-client-id:\t1\ndrm-total-gtt:\t999 KiB\n")   # another driver
        check("fdinfo: each amdgpu client once", client_gtt_kib(d), 1048576 + 2048)
        p = os.path.join(d, "gtt.txt")
        with open(p, "w") as f:
            f.write("0 0.0\n200 2674.0\n400 3925.0\n600 3100.0\n")
        check("peak of samples", peak_of_samples(p), 3925.0)

    if fails:
        for f in fails:
            print("hrx_gtt_split self-test FAIL: " + f)
        return 1
    print("hrx_gtt_split self-test: ok")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        return self_test()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a command with the trace on and print its split")
    r.add_argument("--out", default="hrx-gtt-split")
    r.add_argument("--hz", type=float, default=5.0)
    r.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("parse", help="print the split of an existing -v log")
    p.add_argument("log")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--peak-mib", type=float)
    g.add_argument("--samples", help="a `ms MiB` file, as run writes")
    a = ap.parse_args()
    return cmd_run(a) if a.cmd == "run" else cmd_parse(a)


if __name__ == "__main__":
    sys.exit(main())
