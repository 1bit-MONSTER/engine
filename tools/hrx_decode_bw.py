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
"""hrx_decode_bw: effective decode bandwidth of one model, as one table row (CL-220GBS, owner goal 220 GB/s).

  hrx_decode_bw.py bytes MODEL.gguf [--depth N] [--mtp]          the bytes one decode step reads, term by term
  hrx_decode_bw.py run MODEL.gguf --bench BIN [--depth N] [--reps 3] [--n 128] [--label L] [--json FILE]
                       [-- extra llama-bench args]                 bytes, then llama-bench tg, then the row
  hrx_decode_bw.py table ROW.json [ROW.json ...]                  rows written by run --json, as one markdown table
  hrx_decode_bw.py top EVENTS.jsonl --steps N [--top 10] [--map KERNEL-IDS.map]
                                                                  top kernels by device time per decode step
  hrx_decode_bw.py --self-test

effective GB/s = bytes one decode step must read / step time. Bytes come from the GGUF itself (every split file):

  weights   every tensor of the decode graph at its stored size, except
            - token embeddings (any tensor whose name contains "token_embd" and that is not also the output head):
              one row per token, nbytes / rows;
            - routed experts (names containing "_exps"): expert_used_count / expert_count of the tensor;
            - the MTP block(s) (the last nextn_predict_layers blocks): left out unless --mtp.
  kv        full-attention layers (blocks with attn_k / attn_qkv-with-attention or attn_kv) read K and V for every
            cached position: layers x depth x n_head_kv x (key_length + value_length) x 2 bytes (f16 cache);
            `run` counts it at depth + n/2, the mean position its n generated tokens read.
  ssm       linear-attention layers read and write their recurrent state each token:
            layers x 2 x time_step_rank x state_size x (inner_size / time_step_rank) x 4 bytes (f32 state).

Activations, norms' f32 reads, the logits write and the extra rows of a per-layer n-gram embedding (a few KB) are
not counted (each under 1 MB per token on the models measured).
ms/step is llama-bench `-p 0 -n N -r REPS` (with `-d DEPTH`), the median of its per-repetition t/s.

`top` reads HRX's own dispatch profile (rocprofv3 does not see HRX dispatches):
  HRX_PROFILE_MODE=dispatch HRX_PROFILE_FILE=x.prof llama-bench -m M -ngl 99 -p 0 -n 32 -r 1
  iree-profile dispatch --format=jsonl --dispatch_events x.prof > x.events.jsonl   (the iree-profile of the same build)
and sums duration_ns per kernel key over --steps decode steps (llama-bench's warm-up token counts as a step: give
n + 1). Profiling slows a step down, so read the shares and the device-time total, not the step time it implies.
Add-on kernels have opaque ids (k_<12 hex>); --map names them from the build's KERNEL-IDS.map.
"""
import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GGUF_PY = os.path.join(HERE, "..", "third_party", "llama.cpp", "gguf-py")


def split_files(path):
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
    if not m:
        return [path]
    return sorted(glob.glob(f"{m.group(1)}-*-of-{m.group(3)}.gguf"))


def read_model(path):
    sys.path.insert(0, GGUF_PY)
    import gguf  # noqa: E402

    fields, tensors = {}, []
    for f in split_files(path):
        r = gguf.GGUFReader(f)
        for k, v in r.fields.items():
            if len(v.data) == 1 and k not in fields:
                try:
                    val = v.parts[v.data[0]]
                    fields[k] = bytes(val).decode() if v.types and v.types[0] == gguf.GGUFValueType.STRING else val[0]
                except Exception:
                    pass
        for t in r.tensors:
            tensors.append((t.name, int(t.n_bytes), [int(x) for x in t.shape]))
    return fields, tensors


def step_bytes(fields, tensors, depth=0, mtp=False):
    """{term: bytes} plus the notes that explain each term."""
    arch = fields.get("general.architecture", "")
    g = lambda k, d=0: int(fields.get(f"{arch}.{k}", d))  # noqa: E731
    blocks, nextn = g("block_count"), g("nextn_predict_layers")
    n_exp, n_used = g("expert_count"), g("expert_used_count")
    mtp_from = blocks - nextn if nextn else blocks
    has_output = any(n == "output.weight" for n, _, _ in tensors)
    terms = {"weights_dense": 0, "weights_experts": 0, "embedding_rows": 0, "kv": 0, "ssm_state": 0}
    attn_layers, ssm_layers = set(), set()
    for name, nbytes, shape in tensors:
        m = re.match(r"blk\.(\d+)\.", name)
        blk = int(m.group(1)) if m else None
        if blk is not None and blk >= mtp_from and not mtp:
            continue
        if "token_embd" in name and not (name == "token_embd.weight" and not has_output):
            rows = shape[-1] if len(shape) > 1 else 1
            terms["embedding_rows"] += nbytes // max(rows, 1)
            continue
        if "_exps" in name and n_exp:
            terms["weights_experts"] += nbytes * n_used // n_exp
        else:
            terms["weights_dense"] += nbytes
        if name == "token_embd.weight" and not has_output:
            terms["embedding_rows"] += 0  # tied: read in full as the output head, counted in weights_dense
        if blk is not None and re.search(r"\.attn_(k|kv|k_b)\.weight$|\.attn_q\.weight$", name):
            attn_layers.add(blk)
        if blk is not None and re.search(r"\.ssm_(out|a|conv1d)", name):
            ssm_layers.add(blk)
    attn_layers -= ssm_layers
    hkv, kl, vl = g("attention.head_count_kv"), g("attention.key_length"), g("attention.value_length")
    if not kl:
        kl = vl = g("embedding_length") // max(g("attention.head_count", 1), 1)
    terms["kv"] = len(attn_layers) * depth * hkv * (kl + (vl or kl)) * 2
    rank, state, inner = g("ssm.time_step_rank"), g("ssm.state_size"), g("ssm.inner_size")
    if ssm_layers and rank and state and inner:
        terms["ssm_state"] = len(ssm_layers) * 2 * rank * state * (inner // rank) * 4
    notes = (f"arch {arch}; {blocks} blocks{f' ({nextn} MTP, ' + ('counted' if mtp else 'left out') + ')' if nextn else ''}; "
             f"experts {n_used}/{n_exp}; {len(attn_layers)} full-attention layers x depth {depth}; "
             f"{len(ssm_layers)} linear-attention layers")
    return terms, notes


def bench(binary, model, depth, reps, n, extra):
    cmd = [binary, "-m", model, "-p", "0", "-n", str(n), "-r", str(reps), "-o", "json"]
    if depth:
        cmd += ["-d", str(depth)]
    cmd += extra
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    rows = json.loads(out[out.index("["):])
    row = [r for r in rows if r.get("n_gen", 0) > 0][-1]
    ts = row.get("samples_ts") or [row["avg_ts"]]
    return 1000.0 / statistics.median(ts), ts, row.get("backends", "")


def fmt_gb(b):
    return f"{b / 1e9:.3f}"


def cmd_bytes(a):
    fields, tensors = read_model(a.model)
    terms, notes = step_bytes(fields, tensors, a.depth, a.mtp)
    print(notes)
    for k, v in terms.items():
        print(f"  {k:16s} {fmt_gb(v)} GB")
    print(f"  {'total':16s} {fmt_gb(sum(terms.values()))} GB")


def cmd_run(a):
    fields, tensors = read_model(a.model)
    # the n generated tokens read the cache at depth .. depth + n: count KV at the mean, depth + n / 2
    terms, notes = step_bytes(fields, tensors, a.depth + a.n // 2, a.mtp)
    total = sum(terms.values())
    ms, ts, backend = bench(a.bench, a.model, a.depth, a.reps, a.n, a.extra)
    row = {"label": a.label or os.path.basename(a.model), "depth": a.depth, "bytes": total, "terms": terms,
           "notes": notes, "ms_per_step": ms, "gb_s": total / 1e6 / ms, "samples_ts": ts, "backend": backend,
           "bench": a.bench, "extra": a.extra}
    print(f"{row['label']}: {fmt_gb(total)} GB/step, {ms:.2f} ms/step (median of {len(ts)}: "
          f"{', '.join(f'{t:.2f}' for t in ts)} t/s), {row['gb_s']:.1f} GB/s  [{notes}]")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(row, f, indent=1)


def cmd_table(a):
    print("| model | depth | GB/step | ms/step | GB/s | of 220 | backend |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for p in a.rows:
        r = json.load(open(p))
        print(f"| {r['label']} | {r['depth']} | {fmt_gb(r['bytes'])} | {r['ms_per_step']:.2f} | {r['gb_s']:.1f} | "
              f"{100 * r['gb_s'] / 220:.0f} % | {r['backend']} |")


def top_kernels(events_path, steps, n_top=10, id_map=None):
    """([(name, us_per_step, share, dispatches_per_step)], device_us_per_step)."""
    import collections
    names = {}
    if id_map:
        for line in open(id_map):
            parts = line.split()
            if len(parts) >= 2:
                names[parts[1]] = parts[0]
    total, count = collections.Counter(), collections.Counter()
    for line in open(events_path):
        d = json.loads(line)
        if d.get("type") != "dispatch_event" or not d.get("valid", True):
            continue
        k = names.get(d["key"], d["key"])
        total[k] += d.get("duration_ns", 0)
        count[k] += 1
    all_ns = sum(total.values()) or 1
    rows = [(k, ns / 1e3 / steps, ns / all_ns, count[k] / steps) for k, ns in total.most_common(n_top)]
    return rows, all_ns / 1e3 / steps


def cmd_top(a):
    rows, dev_us = top_kernels(a.events, a.steps, a.top, a.map)
    print(f"device time {dev_us / 1e3:.2f} ms per step over {a.steps} steps (profiled; shares are what to read)")
    print("| kernel | us/step | share | dispatches/step |")
    print("|---|---:|---:|---:|")
    for k, us, share, n in rows:
        print(f"| `{k}` | {us:.0f} | {100 * share:.1f} % | {n:.0f} |")


def self_test():
    f = {"general.architecture": "t", "t.block_count": 3, "t.nextn_predict_layers": 1, "t.expert_count": 8,
         "t.expert_used_count": 2, "t.attention.head_count_kv": 2, "t.attention.key_length": 4,
         "t.attention.value_length": 4, "t.ssm.time_step_rank": 2, "t.ssm.state_size": 4, "t.ssm.inner_size": 8}
    t = [("token_embd.weight", 1000, [10, 100]), ("output.weight", 1000, [10, 100]),
         ("blk.0.attn_q.weight", 40, [4, 4]), ("blk.0.attn_k.weight", 40, [4, 4]),
         ("blk.0.ffn_up_exps.weight", 800, [4, 4, 8]), ("blk.1.ssm_out.weight", 64, [8, 8]),
         ("blk.1.attn_qkv.weight", 64, [8, 8]), ("blk.2.attn_q.weight", 999, [4, 4])]
    terms, _ = step_bytes(f, t, depth=10)
    want = {"weights_dense": 1000 + 40 + 40 + 64 + 64, "weights_experts": 200, "embedding_rows": 10,
            "kv": 1 * 10 * 2 * 8 * 2, "ssm_state": 1 * 2 * 2 * 4 * 4 * 4}
    assert terms == want, (terms, want)
    terms_mtp, _ = step_bytes(f, t, depth=10, mtp=True)
    assert terms_mtp["weights_dense"] == want["weights_dense"] + 999 and terms_mtp["kv"] == 2 * 10 * 2 * 8 * 2
    # tied output: the embedding is read in full as the head, no row term
    terms_tied, _ = step_bytes(f, [x for x in t if x[0] != "output.weight"])
    assert terms_tied["embedding_rows"] == 0 and terms_tied["weights_dense"] == 1000 + 40 + 40 + 64 + 64
    assert split_files("/x/m-00001-of-00003.gguf") == [] and split_files("/x/m.gguf") == ["/x/m.gguf"]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ev, mp = os.path.join(d, "e.jsonl"), os.path.join(d, "m.map")
        with open(ev, "w") as f:
            for k, ns in (("k_aaaaaaaaaaaa", 3000), ("k_aaaaaaaaaaaa", 3000), ("loom_x", 4000)):
                f.write(json.dumps({"type": "dispatch_event", "key": k, "duration_ns": ns, "valid": True}) + "\n")
            f.write(json.dumps({"type": "dispatch_summary"}) + "\n")
        with open(mp, "w") as f:
            f.write("gemv_small\tk_aaaaaaaaaaaa\tgemv\n")
        rows, dev = top_kernels(ev, 2, 10, mp)
        assert rows[0][0] == "gemv_small" and abs(rows[0][1] - 3.0) < 1e-9 and abs(rows[0][2] - 0.6) < 1e-9, rows
        assert rows[0][3] == 1 and abs(dev - 5.0) < 1e-9 and rows[1][0] == "loom_x"
    print("self-test passed")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("bytes", "run"):
        p = sub.add_parser(name)
        p.add_argument("model")
        p.add_argument("--depth", type=int, default=0)
        p.add_argument("--mtp", action="store_true")
        if name == "run":
            p.add_argument("--bench", required=True)
            p.add_argument("--reps", type=int, default=3)
            p.add_argument("--n", type=int, default=128)
            p.add_argument("--label")
            p.add_argument("--json")
            p.add_argument("extra", nargs="*")
    p = sub.add_parser("table")
    p.add_argument("rows", nargs="+")
    p = sub.add_parser("top")
    p.add_argument("events")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--map")
    a = ap.parse_args(argv[1:])
    {"bytes": cmd_bytes, "run": cmd_run, "table": cmd_table, "top": cmd_top}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
