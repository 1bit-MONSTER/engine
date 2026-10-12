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
"""mem-budget: will this GGUF fit on Strix Halo next to what is already loaded?

  mem_budget.py box                                 GTT / RAM ledger + every GPU-holding process
  mem_budget.py prom                                the same ledger + Tctl as Prometheus text (textfile collector)
  mem_budget.py model M.gguf [-c N] [-np N] [...]   estimate one HRX llama-server, and the verdict

Why: HRX reports free memory == total memory (ggml-hrx device_memory()), so llama.cpp's --fit and
anything else asking the device sees 120 GiB free no matter what is loaded. On this box the real
limits are the GTT pool (amdgpu mem_info_gtt_*), MemAvailable (the same DRAM), and engine#315
(faults seen only when GTT use is >= ~95 GiB).

Estimate = weights (sum of the GGUF tensors, all shards) + KV/recurrent state (from llama.cpp's
own llama-fit-params --fit-print when a binary is available: arch-aware, no device init, no weight
load; else a plain-attention formula) + compute buffer (HRX measured 20-505 MiB; default 512 MiB)
+ HRX graph-program cache (cap 64; measured peaks 2.2-8.7 GiB; default 4 GiB) + mmproj/drafts.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

GIB = 1 << 30
MIB = 1 << 20
FAULT_ZONE_GIB = 95.0  # engine#315: GLM-4.7-Flash faults observed only at GTT >= 95 GiB


def read_int(path, default=None):
    try:
        with open(path) as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return default


def meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            out[k] = int(v.split()[0]) * 1024
    return out


def amdgpu_card():
    for d in sorted(glob.glob("/sys/class/drm/card*/device")):
        if os.path.exists(os.path.join(d, "mem_info_gtt_total")):
            return d
    return None


def box_ledger():
    m = meminfo()
    card = amdgpu_card()
    led = {"mem_total": m.get("MemTotal", 0), "mem_available": m.get("MemAvailable", 0)}
    if card:
        led["gtt_total"] = read_int(os.path.join(card, "mem_info_gtt_total"), 0)
        led["gtt_used"] = read_int(os.path.join(card, "mem_info_gtt_used"), 0)
        led["vram_total"] = read_int(os.path.join(card, "mem_info_vram_total"), 0)
        led["vram_used"] = read_int(os.path.join(card, "mem_info_vram_used"), 0)
    procs = []
    for pid_dir in glob.glob("/proc/[0-9]*"):
        pid = int(os.path.basename(pid_dir))
        gtt = vram = 0
        seen = set()
        try:
            for fd in os.listdir(os.path.join(pid_dir, "fdinfo")):
                try:
                    txt = open(os.path.join(pid_dir, "fdinfo", fd)).read()
                except OSError:
                    continue
                if "drm-driver:\tamdgpu" not in txt:
                    continue
                cid = re.search(r"drm-client-id:\s*(\d+)", txt)
                if cid and cid.group(1) in seen:
                    continue  # one DRM client can be open on several fds
                if cid:
                    seen.add(cid.group(1))
                g = re.search(r"drm-memory-gtt:\s*(\d+) KiB", txt)
                v = re.search(r"drm-memory-vram:\s*(\d+) KiB", txt)
                gtt += int(g.group(1)) * 1024 if g else 0
                vram += int(v.group(1)) * 1024 if v else 0
        except OSError:
            continue
        kfd = 0
        for f in glob.glob(f"/sys/class/kfd/kfd/proc/{pid}/vram_*"):
            kfd += read_int(f, 0) or 0
        if gtt or vram or kfd:
            try:
                cmd = open(os.path.join(pid_dir, "cmdline")).read().replace("\0", " ").strip()
            except OSError:
                cmd = "?"
            procs.append({"pid": pid, "drm_gtt": gtt, "drm_vram": vram, "kfd": kfd, "cmd": cmd[:160]})
    procs.sort(key=lambda p: -(p["drm_gtt"] + p["drm_vram"] + p["kfd"]))
    led["processes"] = procs
    return led


def tctl_c():
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if open(os.path.join(h, "name")).read().strip() != "k10temp":
                continue
        except OSError:
            continue
        for lab in glob.glob(os.path.join(h, "temp*_label")):
            if open(lab).read().strip() == "Tctl":
                v = read_int(lab.replace("_label", "_input"))
                return v / 1000.0 if v is not None else None
    return None


def cmd_prom(a):
    led = box_ledger()
    out = ["# HELP onebit_box_bytes Strix Halo memory ledger (bytes)", "# TYPE onebit_box_bytes gauge"]
    for k in ("mem_total", "mem_available", "gtt_total", "gtt_used", "vram_total", "vram_used"):
        if k in led:
            out.append(f'onebit_box_bytes{{kind="{k}"}} {led[k]}')
    out += ["# HELP onebit_process_gpu_bytes GPU memory per process (DRM fdinfo, KFD sysfs)",
            "# TYPE onebit_process_gpu_bytes gauge"]
    for p in led["processes"]:
        exe = os.path.basename(p["cmd"].split(" ")[0]).replace('"', "")
        for k in ("drm_gtt", "drm_vram", "kfd"):
            out.append(f'onebit_process_gpu_bytes{{pid="{p["pid"]}",exe="{exe}",kind="{k}"}} {p[k]}')
    t = tctl_c()
    if t is not None:
        out += ["# TYPE onebit_tctl_celsius gauge", f"onebit_tctl_celsius {t}"]
    if "gtt_used" in led:
        out += ["# TYPE onebit_gtt_fault_zone gauge",
                f"onebit_gtt_fault_zone {int(led['gtt_used'] >= FAULT_ZONE_GIB * GIB)}"]
    text = "\n".join(out) + "\n"
    if a.out:
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, a.out)
    else:
        sys.stdout.write(text)


def gguf_reader_mod(llama_dir):
    for cand in (llama_dir, os.environ.get("LLAMA_CPP_DIR"), os.path.expanduser("~/src/llama.cpp-1bit"),
                 os.path.expanduser("~/engine/third_party/llama.cpp")):
        if cand and os.path.isdir(os.path.join(cand, "gguf-py", "gguf")):
            sys.path.insert(0, os.path.join(cand, "gguf-py"))
            break
    import gguf  # noqa: E402
    return gguf


def shards(path):
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
    if not m:
        return [path]
    return [f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf" for i in range(1, int(m.group(3)) + 1)]


def gguf_facts(gguf, path):
    total = 0
    arch = None
    kv = {}
    for i, p in enumerate(shards(path)):
        r = gguf.GGUFReader(p)
        total += sum(int(t.n_bytes) for t in r.tensors)
        if i == 0:
            for f in r.fields.values():
                if f.name == "general.architecture":
                    arch = bytes(f.parts[f.data[0]]).decode()
            for f in r.fields.values():
                if arch and f.name.startswith(arch + ".") and len(f.data) == 1:
                    try:
                        kv[f.name[len(arch) + 1:]] = f.parts[f.data[0]].tolist()[0]
                    except Exception:
                        pass
    return arch, total, kv


def kv_formula(kv, n_ctx, kv_bytes):
    """Plain attention only (every layer, no SWA/MLA/recurrent): an upper bound for hybrids."""
    n_layer = kv.get("block_count", 0)
    n_head = kv.get("attention.head_count", 0) or 1
    n_kv = kv.get("attention.head_count_kv", n_head) or n_head
    n_embd = kv.get("embedding_length", 0)
    kd = kv.get("attention.key_length", n_embd // n_head if n_head else 0)
    vd = kv.get("attention.value_length", kd)
    return int(n_ctx * n_layer * n_kv * (kd + vd) * kv_bytes)


def kv_from_fit_params(binary, model, n_ctx, n_par, ctk, ctv):
    cmd = [binary, "-m", model, "-c", str(n_ctx), "--device", "none", "--fit-print", "on", "-np", str(n_par)]
    # llama-fit-params takes no -kvu; with one shared cache (what serve passes on HRX) the KV size is
    # the same as one sequence over the whole context, so the default estimate already matches it.
    if ctk:
        cmd += ["-ctk", ctk]
    if ctv:
        cmd += ["-ctv", ctv]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"^Host\s+(\d+)\s+(\d+)\s+(\d+)\s*$", p.stdout + p.stderr, re.M)
    return int(m.group(2)) * MIB if m else None


def find_fit_params(explicit):
    cands = [explicit, os.environ.get("LLAMA_FIT_PARAMS")]
    cands += sorted(glob.glob(os.path.expanduser("~/engine/build*/**/bin/llama-fit-params"), recursive=True))
    cands += sorted(glob.glob(os.path.expanduser("~/wt/*/bin/llama-fit-params")))
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def fmt(b):
    return f"{b / GIB:7.2f} GiB"


def cmd_box(a):
    led = box_ledger()
    if a.json:
        json.dump(led, sys.stdout, indent=1)
        print()
        return
    print(f"RAM   total {fmt(led['mem_total'])}   MemAvailable {fmt(led['mem_available'])}")
    if "gtt_total" in led:
        print(f"GTT   total {fmt(led['gtt_total'])}   used {fmt(led['gtt_used'])}   "
              f"free {fmt(led['gtt_total'] - led['gtt_used'])}")
        if led["gtt_used"] >= FAULT_ZONE_GIB * GIB:
            print(f"WARNING: GTT use is in the engine#315 fault zone (>= {FAULT_ZONE_GIB:.0f} GiB)")
    print("GPU memory by process (DRM fdinfo GTT+VRAM; KFD = HRX/ROCr allocations, sysfs kfd/proc):")
    for p in led["processes"]:
        print(f"  {p['pid']:>8}  drm {fmt(p['drm_gtt'] + p['drm_vram'])}  kfd {fmt(p['kfd'])}  {p['cmd'][:90]}")
    print("(processes of other users are not readable without root)")


def cmd_model(a):
    gguf = gguf_reader_mod(a.llama_cpp)
    arch, weights, kv = gguf_facts(gguf, a.model)
    n_ctx = a.ctx_size or kv.get("context_length", 4096)
    kvb = {"f16": 2, "bf16": 2, "f32": 4, "q8_0": 34 / 32, "q4_0": 18 / 32}.get(a.ctk or "f16", 2)
    fitbin = None if a.no_fit_params else find_fit_params(a.fit_params)
    kv_bytes = kv_from_fit_params(fitbin, a.model, n_ctx, a.parallel, a.ctk, a.ctv) if fitbin else None
    kv_src = f"llama-fit-params ({fitbin})" if kv_bytes is not None else "plain-attention formula (upper bound)"
    if kv_bytes is None:
        kv_bytes = kv_formula(kv, n_ctx, kvb)
    extras = sum(os.path.getsize(p) for p in a.extra)
    compute = int(a.compute_mib * MIB)
    progcache = int(a.program_cache_gib * GIB)
    need = weights + kv_bytes + compute + progcache + extras
    led = box_ledger()
    gtt_free = led.get("gtt_total", 0) - led.get("gtt_used", 0) if "gtt_total" in led else None
    avail = led["mem_available"]
    limit = min(x for x in (gtt_free, avail) if x is not None)
    margin = max(int(a.margin_gib * GIB), int(0.10 * limit))
    after_gtt = led.get("gtt_used", 0) + need
    res = {"model": a.model, "arch": arch, "n_ctx": n_ctx, "parallel": a.parallel,
           "weights": weights, "kv_state": kv_bytes, "kv_source": kv_src, "compute": compute,
           "program_cache": progcache, "extras": extras, "need": need, "gtt_free": gtt_free,
           "mem_available": avail, "margin": margin, "gtt_used_after": after_gtt,
           "fits": need + margin <= limit, "fault_zone_after": after_gtt >= FAULT_ZONE_GIB * GIB}
    if a.json:
        json.dump(res, sys.stdout, indent=1)
        print()
    else:
        print(f"{os.path.basename(a.model)}  arch={arch}  ctx={n_ctx}  slots={a.parallel}")
        print(f"  weights        {fmt(weights)}")
        print(f"  KV + state     {fmt(kv_bytes)}   [{kv_src}]")
        print(f"  compute        {fmt(compute)}   [assumed; HRX measured 20-505 MiB]")
        print(f"  program cache  {fmt(progcache)}   [assumed; cap 64, measured peaks 2.2-8.7 GiB]")
        if extras:
            print(f"  extras         {fmt(extras)}   [mmproj / draft / mtp files]")
        print(f"  NEED           {fmt(need)}")
        print(f"box: GTT free {fmt(gtt_free) if gtt_free is not None else '   n/a'}, MemAvailable {fmt(avail)}, "
              f"margin {fmt(margin)}")
        print(f"verdict: {'FITS' if res['fits'] else 'DOES NOT FIT'}"
              + (f"; GTT after load ~{after_gtt / GIB:.1f} GiB is in the engine#315 fault zone"
                 if res["fault_zone_after"] else ""))
    sys.exit(0 if res["fits"] else 3)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("box")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_box)
    s = sp.add_parser("prom")
    s.add_argument("--out", help="write atomically (node_exporter --collector.textfile.directory)")
    s.set_defaults(fn=cmd_prom)
    s = sp.add_parser("model")
    s.add_argument("model")
    s.add_argument("-c", "--ctx-size", type=int, default=0, help="as passed to 1bit serve (default: the model's)")
    s.add_argument("-np", "--parallel", type=int, default=1)
    s.add_argument("-ctk", default=None)
    s.add_argument("-ctv", default=None)
    s.add_argument("--extra", action="append", default=[], help="mmproj / draft / mtp file (repeatable)")
    s.add_argument("--compute-mib", type=float, default=512)
    s.add_argument("--program-cache-gib", type=float, default=4)
    s.add_argument("--margin-gib", type=float, default=8)
    s.add_argument("--fit-params", help="llama-fit-params binary (else $LLAMA_FIT_PARAMS or a build on this box)")
    s.add_argument("--no-fit-params", action="store_true")
    s.add_argument("--llama-cpp", help="llama.cpp checkout for gguf-py")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_model)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
