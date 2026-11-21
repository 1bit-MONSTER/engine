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
"""hrx2kineto.py: HRX dispatch profiles -> Kineto traces for Hyperloom, TraceLens and rocprof studio.

rocprofv3 sees no HRX dispatches (HRX loads its own libhsa and IREE writes the AQL packets). HRX profiles
itself instead:

    HRX_PROFILE_MODE=dispatch HRX_PROFILE_FILE=x.prof <llama-*>          (shapes log: see below)
    iree-profile dispatch --format=jsonl --dispatch_events x.prof > x.events.jsonl
    iree-profile command  --format=jsonl x.prof > x.commands.jsonl

`GGML_HRX_DISPATCH_SHAPE_LOG` HAS a producer: `ggml/src/ggml-hrx/dispatch/dispatch-shape-log.cpp` reads it at `:115`
and appends to it at `:181` (`std::fopen(…, "a")`, `kShapeLogEnvironment` at `:30`). This paragraph used to say the
opposite ("no commit under ggml/src/ggml-hrx ever writes it"), which was true before the producer landed and is now the
reason a reader gives up on the attribution half before measuring: two lanes concluded from it that the per-op table
cannot be produced on this box (engine#3103). Without `--shapes` the converter still runs and emits kernel names only, so
a missing shape log looks like a working run that has nothing to attribute.

Basic use (the original behaviour, kept byte-compatible):

    hrx2kineto.py EVENTS OUT.trace.json --shapes SHAPES --by-kernel [N]     (per-kernel ranking, engine#3103)
    hrx2kineto.py EVENTS OUT.trace.json --shapes SHAPES --by-shape  [N]     (per-(op,K,N) ranking, engine#3103)
    hrx2kineto.py --self-test                                               (fixture arms, no device)

    hrx2kineto.py x.events.jsonl x.trace.json [kernel_trace.csv]

writes every dispatch as a Kineto GPU kernel event (ph X, microseconds) with a cuda_runtime launch and a
cpu_op named after the kernel, so the correlation chain resolves. TraceLens then puts every op in "other".

With --shapes and --commands, each graph execution is matched to the HRX command program that recorded it
(same kernel sequence; the shape log's per-execution lines pick between programs that share one and give
the current node names), every dispatch to the ggml dispatch match that emitted it, and the converter
emits one cpu_op per match execution instead: named hrx::<primary ggml op> (hrx::mul_mat,
hrx::mul_mat_id, hrx::flash_attn_ext, hrx::rms_norm, ...), holding all of that match's kernels, with
"Input Dims"/"Input type"/"Input Strides" from the ggml tensors, "Concrete Inputs" = the fused op chain,
and an "hrx" arg with model FLOPs, bytes (quantized weights at their real bits per weight, MUL_MAT_ID
reading only the routed experts) and compute spec. cpu_ops nest under hrx::step (one forward pass,
across graph splits) and hrx::layer (per transformer block), so TraceDiff aligns two runs layer by layer.
Everything is derived from the shape log; no kernel or registration names are hard-coded here.

This file is also a TraceLens extension: pass it as --extension_file so TraceLens maps the hrx:: ops to
GEMM / GroupedGEMM / SDPA / NORM / RoPE / reduce / SSM / elementwise perf models that read the "hrx" arg,
and pass --write-arch's JSON as --gpu_arch_json_path for roofline columns.
"""
import argparse
import bisect
import collections
import contextlib
import glob
import io
import json
import math
import os
import pathlib
import re
import sys

TICK_NS = 10.019  # gfx1151 device tick, as ~/.cache/hrxprof/aggprof.py

# ggml type -> (block size, block bytes), from gguf-py GGML_QUANT_SIZES (plus the fork's extra types).
GGML_BLOCK = {
    "f32": (1, 4), "f16": (1, 2), "bf16": (1, 2), "f64": (1, 8),
    "i8": (1, 1), "i16": (1, 2), "i32": (1, 4), "i64": (1, 8),
    "q4_0": (32, 18), "q4_1": (32, 20), "q5_0": (32, 22), "q5_1": (32, 24), "q8_0": (32, 34), "q8_1": (32, 40),
    "q2_k": (256, 84), "q3_k": (256, 110), "q4_k": (256, 144), "q5_k": (256, 176), "q6_k": (256, 210),
    "q8_k": (256, 292), "iq2_xxs": (256, 66), "iq2_xs": (256, 74), "iq3_xxs": (256, 98), "iq1_s": (256, 50),
    "iq4_nl": (32, 18), "iq3_s": (256, 110), "iq2_s": (256, 82), "iq4_xs": (256, 136), "iq1_m": (256, 56),
    "tq1_0": (256, 54), "tq2_0": (256, 66), "mxfp4": (32, 17), "nvfp4": (64, 36), "q1_0": (128, 18),
    "q2_0": (64, 18), "pq2_0": (128, 34), "ptq1_0": (128, 28),
}
FLOAT_TYPES = {"f32", "f16", "bf16", "f64"}
TORCH_TYPE = {"f32": "float", "f16": "c10::half", "bf16": "c10::bfloat16", "f64": "double",
              "i32": "int", "i64": "long int", "i8": "signed char"}

# Which covered node names a fused match, highest first.
PRIMARY_ORDER = ["FLASH_ATTN_EXT", "MUL_MAT_ID", "MUL_MAT", "OUT_PROD", "GATED_DELTA_NET", "SSM_SCAN",
                 "RWKV_WKV6", "RWKV_WKV7", "GATED_LINEAR_ATTN", "SSM_CONV", "CONV_2D", "CONV_2D_DW",
                 "CONV_TRANSPOSE_1D", "CONV_TRANSPOSE_2D", "IM2COL", "RMS_NORM", "NORM", "L2_NORM", "GROUP_NORM",
                 "ROPE", "SOFT_MAX", "ARGSORT", "TOP_K", "SUM_ROWS", "MEAN", "SUM", "CUMSUM", "ARGMAX",
                 "GET_ROWS", "SET_ROWS", "CONCAT", "CPY", "CONT", "DUP"]
CATEGORY_OF = {
    "MUL_MAT": "gemm", "OUT_PROD": "gemm", "MUL_MAT_ID": "grouped_gemm", "FLASH_ATTN_EXT": "attention",
    "RMS_NORM": "norm", "NORM": "norm", "L2_NORM": "norm", "GROUP_NORM": "norm", "ROPE": "rope",
    "SOFT_MAX": "reduce", "ARGSORT": "reduce", "TOP_K": "reduce", "SUM_ROWS": "reduce", "MEAN": "reduce",
    "SUM": "reduce", "CUMSUM": "reduce", "ARGMAX": "reduce", "SSM_CONV": "ssm", "SSM_SCAN": "ssm",
    "GATED_DELTA_NET": "ssm", "RWKV_WKV6": "ssm", "RWKV_WKV7": "ssm", "GATED_LINEAR_ATTN": "ssm",
}
GGML_OPS = ("NONE DUP ADD ADD_ID ADD1 ACC SUB MUL DIV SQR SQRT LOG SIN COS SUM SUM_ROWS CUMSUM MEAN ARGMAX "
            "COUNT_EQUAL REPEAT REPEAT_BACK CONCAT SILU_BACK NORM RMS_NORM RMS_NORM_BACK GROUP_NORM L2_NORM MUL_MAT "
            "MUL_MAT_ID OUT_PROD SCALE SET CPY CONT RESHAPE VIEW PERMUTE TRANSPOSE GET_ROWS GET_ROWS_BACK SET_ROWS "
            "DIAG DIAG_MASK_INF DIAG_MASK_ZERO SOFT_MAX SOFT_MAX_BACK ROPE ROPE_BACK CLAMP CONV_TRANSPOSE_1D IM2COL "
            "IM2COL_BACK IM2COL_3D COL2IM_1D CONV_2D CONV_3D CONV_2D_DW CONV_TRANSPOSE_2D POOL_1D POOL_2D "
            "POOL_2D_BACK UPSCALE PAD PAD_REFLECT_1D ROLL ARANGE TIMESTEP_EMBEDDING ARGSORT TOP_K LEAKY_RELU TRI "
            "FILL FLASH_ATTN_EXT FLASH_ATTN_BACK SSM_CONV SSM_SCAN WIN_PART WIN_UNPART GET_REL_POS ADD_REL_POS "
            "RWKV_WKV6 GATED_LINEAR_ATTN RWKV_WKV7 SOLVE_TRI GATED_DELTA_NET LIGHTNING_INDEXER DSV4_HC_COMB "
            "DSV4_HC_PRE DSV4_HC_POST UNARY GLU MAP_CUSTOM1 MAP_CUSTOM2 MAP_CUSTOM3 CUSTOM OPT_STEP_ADAMW "
            "OPT_STEP_SGD").split()
LAYOUT_OPS = {"VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE", "NONE"}
LAYER_RE = (re.compile(r"(?:^|\.)blk\.(\d+)\."), re.compile(r"-(\d+)$"))
MATRIX_MIN_ROWS = 16  # activation rows per weight at which HRX/HIP GEMMs switch from GEMV (dot4) to WMMA


def type_bytes(type_name, elements):
    block, size = GGML_BLOCK.get((type_name or "f32").lower(), (1, 4))
    return elements * size / block


def prod(values):
    out = 1
    for v in values:
        out *= int(v)
    return out


def tensor_bytes(t):
    return type_bytes(t["type"], prod(t["ne"]))


def tensor_key(t):
    if "id" in t:  # graph value id (shape logs from the same build carry it)
        return t["id"]
    return (t.get("name"), t["type"], tuple(t["ne"]), tuple(t.get("nb", ())))


def layer_of(match, names=None):
    """Transformer block of a match: from the executing graph's node names when the shape log has them (a
    program built for one layer is reused for every structurally equal layer), else from build-time names."""
    if names is not None:
        for node in match.get("nodes") or []:
            if node and node.get("index") is not None and node["index"] < len(names):
                m = LAYER_RE[1].search(names[node["index"]])
                if m:
                    return int(m.group(1))
        return None
    for node in match.get("nodes") or []:
        if not node:
            continue
        for t in [node.get("out")] + list(node.get("src") or []):
            if t and t.get("name"):
                for rx in LAYER_RE:
                    m = rx.search(t["name"])
                    if m:
                        return int(m.group(1))
    return None


def model_match(match):
    """FLOPs / bytes / compute spec / TraceLens args for one ggml dispatch match (all its kernels)."""
    nodes = [n for n in (match.get("nodes") or []) if n]
    if not nodes:
        return None
    ops = [n["op"] for n in nodes]
    primary = next((n for op in PRIMARY_ORDER for n in nodes if n["op"] == op), nodes[0])
    pop = primary["op"]
    # Layout nodes (VIEW/RESHAPE/...) alias their source: data produced inside the match stays internal
    # through them, and a view of an outside tensor is read at the view's size, not the whole tensor's.
    internal = set()
    for n in nodes:
        if not n.get("out"):
            continue
        src0 = (n.get("src") or [None])[0]
        if n["op"] not in LAYOUT_OPS or (src0 and tensor_key(src0) in internal):
            internal.add(tensor_key(n["out"]))
    consumed = set()
    for n in nodes:
        for s in n.get("src") or []:
            if s:
                consumed.add(tensor_key(s))

    flops = 0.0
    read = {}
    write = {}
    rows_per_weight = None
    for n in nodes:
        out = n.get("out")
        src = [s for s in (n.get("src") or [])]
        op = n["op"]
        out_elems = prod(out["ne"]) if out else 0
        if op in LAYOUT_OPS:
            if out and tensor_key(out) not in consumed and src and src[0] and tensor_key(src[0]) in internal:
                write[tensor_key(out)] = tensor_bytes(out)
            continue
        for i, s in enumerate(src):
            if not s or tensor_key(s) in internal:
                continue
            k = tensor_key(s)
            b = tensor_bytes(s)
            if op == "GET_ROWS" and i == 0:
                rows = prod(out["ne"][1:]) if out else 0
                b = type_bytes(s["type"], s["ne"][0] * rows)
            elif (op == "SET_ROWS" and i >= 2) or (op == "CPY" and i == 1):
                continue  # destination (HRX imports SET_ROWS as rows, ids, dst): written below, not read
            elif op == "MUL_MAT_ID" and i == 0 and len(src) > 2 and src[2]:
                ids = src[2]["ne"]
                used = min(s["ne"][2], ids[0] * prod(ids[1:]))
                b = type_bytes(s["type"], s["ne"][0] * s["ne"][1] * used)
            read[k] = max(read.get(k, 0), b)
        if out and tensor_key(out) not in consumed:
            if op == "SET_ROWS" and src and src[0]:
                wb = type_bytes(out["type"], prod(src[0]["ne"]))
            else:
                wb = tensor_bytes(out)
            write[tensor_key(out)] = wb

        if op in ("MUL_MAT", "OUT_PROD") and src and src[0]:
            k_dim = src[0]["ne"][0]
            flops += 2.0 * k_dim * out_elems
            if n is primary:
                rows_per_weight = out["ne"][1] * (prod(out["ne"][2:]) // max(1, prod(src[0]["ne"][2:])))
        elif op == "MUL_MAT_ID" and src and src[0]:
            k_dim = src[0]["ne"][0]
            flops += 2.0 * k_dim * out_elems
            if n is primary and len(src) > 2 and src[2]:
                ids = src[2]["ne"]
                routed = ids[0] * prod(ids[1:])
                rows_per_weight = routed / max(1, min(src[0]["ne"][2], routed))
        elif op == "FLASH_ATTN_EXT" and len(src) > 2 and all(src[:3]):
            q, kk, v = src[0]["ne"], src[1]["ne"], src[2]["ne"]
            flops += 2.0 * q[1] * q[2] * q[3] * kk[1] * (q[0] + v[0])
            if n is primary:
                rows_per_weight = q[1] * (q[2] // max(1, kk[2]))
        elif op in ("SOFT_MAX", "RMS_NORM", "NORM", "L2_NORM", "GROUP_NORM", "ROPE"):
            flops += 5.0 * out_elems
        elif op in ("GATED_DELTA_NET", "SSM_SCAN", "RWKV_WKV6", "RWKV_WKV7", "GATED_LINEAR_ATTN"):
            state = max((prod(s["ne"]) for s in src if s), default=0)
            flops += 6.0 * max(state, out_elems)
        elif op == "SET_ROWS" and src and src[0]:
            flops += float(prod(src[0]["ne"]))  # elements moved; the output is the whole destination
        else:
            flops += float(out_elems)

    bytes_moved = sum(read.values()) + sum(write.values())
    w = primary["src"][0] if primary.get("src") and primary["src"][0] else None
    wtype = (w["type"] if w else primary["out"]["type"]).lower()
    cat = CATEGORY_OF.get(pop, "copy" if pop in ("GET_ROWS", "SET_ROWS", "CPY", "CONT", "DUP", "CONCAT")
                          else "elementwise")
    if cat in ("gemm", "grouped_gemm", "attention"):
        if rows_per_weight is not None and rows_per_weight >= MATRIX_MIN_ROWS:
            spec = ("matrix", "fp16")
        elif cat == "attention" or wtype in FLOAT_TYPES:
            spec = ("vector", "fp32" if wtype == "f32" else "fp16")
        else:
            spec = ("vector", "int8")  # quantized GEMV: activations as q8, v_dot4
    else:
        spec = ("vector", "fp32")

    srcs = [s for s in (primary.get("src") or []) if s]
    descs = [n.get("desc") or n["op"] for n in nodes if n["op"] not in LAYOUT_OPS]
    fused = "+".join(descs) or pop
    po = primary["out"]["ne"]
    if pop in ("MUL_MAT", "OUT_PROD") and srcs:
        w, x = srcs[0], srcs[1] if len(srcs) > 1 else srcs[0]
        params = {"M": po[1], "N": po[0], "K": w["ne"][0], "B": prod(po[2:]),
                  "dtype_A_B": (w["type"], x["type"])}
    elif pop == "MUL_MAT_ID" and len(srcs) > 2:
        w, x, ids = srcs[0], srcs[1], srcs[2]
        params = {"tokens": prod(ids["ne"][1:]), "top_k": ids["ne"][0], "experts": w["ne"][2], "N": w["ne"][1],
                  "K": w["ne"][0], "dtype_A_B": (w["type"], x["type"])}
    elif pop == "FLASH_ATTN_EXT" and len(srcs) > 2:
        q, k, v = srcs[0]["ne"], srcs[1]["ne"], srcs[2]["ne"]
        params = {"B": q[3], "N_Q": q[1], "H_Q": q[2], "N_KV": k[1], "H_KV": k[2], "d_h_qk": q[0], "d_h_v": v[0],
                  "dtype_A_B": (srcs[0]["type"], srcs[1]["type"])}
    else:
        params = {"shape": tuple(po), "dtype_A_B": (srcs[0]["type"] if srcs else primary["out"]["type"],)}
    params["fused"] = fused
    return {
        "name": "hrx::" + pop.lower(),
        "category": cat,
        "flops": flops,
        "bytes": bytes_moved,
        "maf": spec[0],
        "precision": spec[1],
        "rows_per_weight": rows_per_weight,
        "fused": fused,
        "params": params,
        "input_dims": [list(s["ne"]) for s in srcs] + [list(primary["out"]["ne"])],
        "input_type": [TORCH_TYPE.get(s["type"].lower(), s["type"]) for s in srcs] + [
            TORCH_TYPE.get(primary["out"]["type"].lower(), primary["out"]["type"])],
        "input_strides": [list(s.get("nb", [])) for s in srcs] + [list(primary["out"].get("nb", []))],
    }


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                yield json.loads(line)


def load_dispatch_events(src):
    events = [r for r in load_jsonl(src)
              if r.get("type") in ("dispatch_event", "dispatch") and "key" in r and r.get("valid", True)]
    if not events:
        sys.exit(f"hrx2kineto: no dispatch events in {src}")
    events.sort(key=lambda r: int(r["start_tick"]))
    return events


def event_times(r, t0):
    ts = (int(r["start_tick"]) - t0) * TICK_NS / 1e3
    dur = float(r.get("duration_ns") or int(r["duration_ticks"]) * TICK_NS) / 1e3
    return ts, dur


def kernel_event(r, i, ts, dur):
    grid = list(r.get("workgroup_count", [1, 1, 1]))
    block = list(r.get("workgroup_size", [1, 1, 1]))
    return {"ph": "X", "cat": "kernel", "name": r["key"], "pid": 0, "tid": 7, "ts": ts, "dur": dur,
            "args": {"correlation": i, "External id": i, "device": 0, "stream": 0,
                     "grid": [g * b for g, b in zip(grid, block)], "block": block}}


def trace_doc(out):
    return {"schemaVersion": 1, "deviceProperties": [{"id": 0, "name": "AMD Radeon 8060S (gfx1151, HRX)"}],
            "traceEvents": out}


def convert_basic(events, dst):
    t0 = int(events[0]["start_tick"])
    out = []
    for i, r in enumerate(events, 1):
        ts, dur = event_times(r, t0)
        name = r["key"]
        out.append({"ph": "X", "cat": "cpu_op", "name": name, "pid": 1, "tid": 1, "ts": ts, "dur": 0.1,
                    "args": {"External id": i}})
        out.append({"ph": "X", "cat": "cuda_runtime", "name": "hipModuleLaunchKernel", "pid": 1, "tid": 1,
                    "ts": ts, "dur": 0.1, "args": {"correlation": i, "External id": i}})
        out.append(kernel_event(r, i, ts, dur))
    json.dump(trace_doc(out), open(dst, "w"))
    print(f"hrx2kineto: {len(events)} dispatches -> {dst}")
    return t0


def map_command_buffers(commands_path, programs, events):
    """Programs whose kernel sequence equals each command buffer's (several programs can share one)."""
    cb_ops = collections.defaultdict(list)
    if commands_path:
        for r in load_jsonl(commands_path):
            if r.get("type") == "command_operation" and r.get("op") == "dispatch":
                cb_ops[r["command_buffer_id"]].append((r["command_index"], r["key"]))
    else:  # fall back to the dispatches seen (exact when every recorded dispatch ran at least once)
        seen = {}
        for r in events:
            seen[(r["command_buffer_id"], r["command_index"])] = r["key"]
        for (cb, idx), key in seen.items():
            cb_ops[cb].append((idx, key))

    def names(cmd):
        return {n for n in (cmd.get("kernel"), cmd.get("symbol")) if n}

    def align(keys, seq):
        """(ok, tolerated, first_bad) aligning the profile's keys to a program's command sequence.

        Exact names first, then a key that is the PREFIX of exactly one command name in that program. Measured need: the
        30B-MoE pp512 profile keys one command `ggml_linear_q6k_q8_1_x4` while the shape log records symbol
        `ggml_linear_q6k_q8_1_x4_wave64_block4` -- the profile drops a specialization suffix -- and because the old test was
        all-or-nothing, that single entry made 872 dispatches match no program, i.e. the whole model ranked as unmapped and
        the run looked like a profile with nothing to attribute. Uniqueness is required so an ambiguous prefix still fails.
        """
        all_names = [names(c) for c in seq]
        tolerated = 0
        for i, (k, ns) in enumerate(zip(keys, all_names)):
            if k in ns:
                continue
            hits = [j for j, other in enumerate(all_names) if any(n.startswith(k) for n in other)]
            if len(hits) == 1 and hits[0] == i:
                tolerated += 1
                continue
            return False, tolerated, i
        return True, tolerated, None

    result = {}
    used = set()
    last = -1
    for cb in sorted(cb_ops):
        ops = sorted(cb_ops[cb])
        keys = [k for _, k in ops]
        candidates, best_matched = [], 0
        for p in programs:
            main = [c for c in p["commands"] if c["phase"] == "main"]
            init = [c for c in p["commands"] if c["phase"] == "init"]
            for seq in (main, init + main):
                if len(seq) != len(keys):
                    continue
                best_matched = max(best_matched, sum(1 for k, c in zip(keys, seq) if k in names(c)))
                ok, tolerated, _bad = align(keys, seq)
                if ok:
                    if tolerated:
                        print(f"hrx2kineto: command buffer {cb}: {tolerated} of {len(keys)} keys matched by prefix "
                              f"(the profile shortened a specialization suffix)", file=sys.stderr)
                    candidates.append((p, seq))
                    break
        if not candidates:
            print(f"hrx2kineto: command buffer {cb} ({len(keys)} dispatches) matches no logged program"
                  + (f" (the closest program matches {best_matched}/{len(keys)} keys)" if best_matched else ""),
                  file=sys.stderr)
            continue
        cands = [(p, {idx: seq[i] for i, (idx, _) in enumerate(ops)}) for p, seq in candidates]
        after = [c for c in cands if c[0]["program"] > last and c[0]["program"] not in used]
        best = (after or cands)[0]
        used.add(best[0]["program"])
        last = best[0]["program"]
        result[cb] = cands if best is cands[0] else [best] + [c for c in cands if c is not best]
    return result  # cb -> [(program, {command_index: command})], best guess first


def program_tokens(program, models):
    for match, m in zip(program["matches"], models):
        if not m or m["category"] not in ("gemm", "grouped_gemm"):
            continue
        w = next((n["src"][0] for n in match["nodes"] if n and n["op"] in ("MUL_MAT", "MUL_MAT_ID")
                  and n.get("src") and n["src"][0]), None)
        if w and (w.get("name") or "").endswith(".weight"):
            p = m["params"]
            return p.get("tokens", p.get("M"))
    for m in models:
        if m and m["category"] in ("gemm", "grouped_gemm") and m["rows_per_weight"] is not None:
            # tokens = activation rows of the first weight GEMM (MUL_MAT_ID: routed rows / top-k)
            dims = m["input_dims"]
            if m["category"] == "grouped_gemm" and len(dims) > 2:
                return prod(dims[2][1:])
            return dims[1][1] if len(dims) > 1 else None
    return None


def convert_shapes(events, dst, shapes_path, commands_path, phase="all", skip=0, limit=None, decode_max_tokens=1):
    programs, models = load_programs_models(shapes_path)
    cbmap = map_command_buffers(commands_path, programs, events)
    tokens = {p["program"]: program_tokens(p, models[p["program"]]) for p in programs}

    # Graph executions (submission, command buffer) in submission order. hrx_execution lines (one per HRX
    # graph execution, with the program uid and the current node names) pick the program when several share
    # a kernel sequence and give the real layer numbers of reused programs.
    runs = [r for r in load_jsonl(shapes_path) if r.get("type") == "hrx_execution"]
    # One graph execution can span several queue submissions of its command buffer (each continuing at
    # higher command indices); those share one execution key.
    spans = collections.OrderedDict()
    for r in events:
        key = (r.get("submission_id"), r.get("command_buffer_id"))
        lo, hi = spans.get(key, (r["command_index"], r["command_index"]))
        spans[key] = (min(lo, r["command_index"]), max(hi, r["command_index"]))
    exec_of = {}
    order = []
    prev = None
    for key, (lo, hi) in sorted(spans.items(), key=lambda kv: kv[0][0] if kv[0][0] is not None else -1):
        if prev is not None and prev[0][1] == key[1] and lo > prev[1]:
            exec_of[key] = exec_of[prev[0]]
        else:
            exec_of[key] = key
            order.append(key)
        prev = (key, hi)
    assign = {}
    j, missed = 0, 0
    for key in order:
        cands = cbmap.get(key[1])
        if not cands:
            continue
        chosen, names = cands[0], None
        if runs:
            uids = {c[0]["graph_uid"]: c for c in cands}
            k = j
            while k < len(runs) and runs[k]["graph_uid"] not in uids:
                k += 1
            if k < len(runs):
                chosen, names = uids[runs[k]["graph_uid"]], runs[k]["names"]
                j = k + 1
            else:
                missed += 1
        assign[key] = (chosen[0], chosen[1], names)
    if runs and missed:
        print(f"hrx2kineto: {missed} graph executions not aligned with hrx_execution lines; layers from build-time "
              f"names there", file=sys.stderr)

    executions = collections.OrderedDict()
    unmapped = 0
    for r in events:
        key = exec_of[(r.get("submission_id"), r.get("command_buffer_id"))]
        entry = assign.get(key)
        cmd = entry[1].get(r.get("command_index")) if entry else None
        if cmd is None:
            unmapped += 1
            ukey = ("unmapped",) + key
            executions.setdefault(ukey, {"program": None, "groups": collections.OrderedDict()})
            executions[ukey]["groups"].setdefault(("k", len(executions[ukey]["groups"])), []).append(r)
            continue
        ex = executions.setdefault(key, {"program": entry[0], "names": entry[2], "groups": collections.OrderedDict()})
        gid = cmd.get("match", -1)
        ex["groups"].setdefault(gid if gid >= 0 else ("k", cmd["ordinal"]), []).append(r)

    # Phase filter per graph execution; a split segment with no weight GEMM keeps the previous decision.
    selected = []
    decision = phase == "all"
    for key, ex in executions.items():
        p = ex["program"]
        if phase != "all" and p is not None:
            t = tokens.get(p["program"])
            if t is not None:
                decision = (phase == "decode") == (t <= decode_max_tokens)
        if decision:
            selected.append((key, ex))

    # Steps (one forward pass): graph splits (CPU fallbacks) cut a pass into several executions; a new
    # step starts when the layer number goes back down, so runs with and without splits align in TraceDiff.
    def layers(ex):
        p = ex["program"]
        if p is None:
            return []
        return [l for l in (layer_of(p["matches"][g], ex.get("names")) for g in ex["groups"] if isinstance(g, int))
                if l is not None]

    steps = []
    last_layer = None
    for key, ex in selected:
        ls = layers(ex)
        if not steps or (ls and last_layer is not None and ls[0] < last_layer):
            steps.append([])
            last_layer = None
        steps[-1].append((key, ex))
        if ls:
            last_layer = ls[-1]
    steps = steps[skip:]
    if limit is not None:
        steps = steps[:limit]
    keep = [kx for step in steps for kx in step]
    if not keep:
        sys.exit("hrx2kineto: no graph executions left after --phase/--skip/--max filtering")

    t0 = min(int(g[0]["start_tick"]) for _, ex in keep for g in ex["groups"].values())
    out = []
    corr = 0
    cpu_t = 0.0  # synthetic host clock: nesting only, host timing is not profiled
    n_ops = 0
    pending_layer = None  # [layer, start, events]

    def close_layer():
        nonlocal pending_layer, cpu_t
        if pending_layer is not None:
            lay, start, evs = pending_layer
            out.append({"ph": "X", "cat": "cpu_op", "name": "hrx::layer", "pid": 1, "tid": 1, "ts": start,
                        "dur": cpu_t - start, "args": {"layer": lay}})
            out.extend(evs)
            cpu_t += 1.0
            pending_layer = None

    for step_index, step in enumerate(steps):
        s_start = cpu_t
        cpu_t += 1.0
        programs_in_step = []
        for key, ex in step:
            p = ex["program"]
            programs_in_step.append(p["program"] if p else None)
            for gid, evs in ex["groups"].items():
                m = None
                match = None
                if p is not None and isinstance(gid, int):
                    match = p["matches"][gid]
                    m = models[p["program"]][gid]
                lay = layer_of(match, ex.get("names")) if match else None
                if pending_layer is not None and lay != pending_layer[0]:
                    close_layer()
                if lay is not None and pending_layer is None:
                    pending_layer = [lay, cpu_t, []]
                    cpu_t += 0.5
                sink = pending_layer[2] if pending_layer is not None else out
                evs.sort(key=lambda r: int(r["start_tick"]))
                op_start = cpu_t
                n_ops += 1
                if m is None:
                    name = evs[0]["key"]
                    args = {}
                else:
                    name = m["name"]
                    args = {"Input Dims": m["input_dims"], "Input type": m["input_type"],
                            "Input Strides": m["input_strides"], "Concrete Inputs": [m["fused"]],
                            "hrx": {"flops": m["flops"], "bytes": m["bytes"], "maf": m["maf"],
                                    "precision": m["precision"], "category": m["category"], "params": m["params"],
                                    "registration": match.get("registration"), "layer": lay}}
                launches = []
                for r in evs:
                    corr += 1
                    ts, dur = event_times(r, t0)
                    launches.append({"ph": "X", "cat": "cuda_runtime", "name": "hipModuleLaunchKernel", "pid": 1,
                                     "tid": 1, "ts": cpu_t + 0.1, "dur": 0.1,
                                     "args": {"correlation": corr, "External id": corr}})
                    cpu_t += 0.3
                    out.append(kernel_event(r, corr, ts, dur))
                cpu_t += 0.1
                args["External id"] = corr
                sink.append({"ph": "X", "cat": "cpu_op", "name": name, "pid": 1, "tid": 1, "ts": op_start,
                             "dur": cpu_t - op_start, "args": args})
                sink.extend(launches)
                cpu_t += 0.2
        close_layer()
        first = step[0][1]["program"]
        out.append({"ph": "X", "cat": "cpu_op", "name": "hrx::step", "pid": 1, "tid": 1, "ts": s_start,
                    "dur": cpu_t - s_start + 0.5,
                    "args": {"step": step_index, "graph_executions": len(step), "programs": programs_in_step,
                             "tokens": tokens.get(first["program"]) if first else None}})
        cpu_t += 1.0
    json.dump(trace_doc(out), open(dst, "w"))
    print(f"hrx2kineto: {sum(len(g) for _, ex in keep for g in ex['groups'].values())} dispatches, "
          f"{n_ops} ops, {len(steps)} steps ({len(keep)} graph executions) -> {dst}"
          + (f" ({unmapped} dispatches without shape info)" if unmapped else ""))
    return keep, models


def write_kernel_trace_csv(events, t0, path):
    # rocprofv3 kernel_trace.csv columns, nanoseconds, so rocprof studio (pwilkin/ROCprofGUI) opens HRX profiles too
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Kind", "Agent_Id", "Queue_Id", "Dispatch_Id", "Kernel_Name", "Correlation_Id",
                    "Start_Timestamp", "End_Timestamp", "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z",
                    "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z"])
        for i, r in enumerate(events, 1):
            start = round((int(r["start_tick"]) - t0) * TICK_NS)
            end = start + round(float(r.get("duration_ns") or int(r["duration_ticks"]) * TICK_NS))
            grid = list(r.get("workgroup_count", [1, 1, 1]))
            block = list(r.get("workgroup_size", [1, 1, 1]))
            w.writerow(["KERNEL_DISPATCH", 1, r.get("queue_ordinal", 0), i, r["key"], i, start, end,
                        *[g * b for g, b in zip(grid, block)], *block])
    print(f"hrx2kineto: kernel trace CSV -> {path}")


def gpu_arch(mem_bw_gbps=256.0, sclk_mhz=None, wgps=20, cus=40):
    """TraceLens arch JSON for gfx1151 (Radeon 8060S): RDNA3 WMMA f16/bf16/iu8 1024 and iu4 2048 ops/WGP/clk;
    VALU f32 FMA 64 lanes x 2 (dual issue) per CU, packed f16 2x, v_dot4 i8 8 ops/lane."""
    if sclk_mhz is None:
        sclk_mhz = 2900.0
        for path in glob.glob("/sys/class/drm/card*/device/pp_dpm_sclk"):
            try:
                vals = [float(m) for m in re.findall(r"(\d+)Mhz", open(path).read())]
            except OSError:
                continue
            if vals:
                sclk_mhz = max(vals)
                break
    clk = sclk_mhz * 1e6
    t = lambda ops: round(ops * clk / 1e12, 2)
    return {"name": "Radeon 8060S (HRX)", "gfx_arch": "gfx1151", "sclk_mhz": sclk_mhz, "mem_bw_gbps": mem_bw_gbps,
            "max_achievable_tflops": {
                "matrix_fp16": t(wgps * 1024), "matrix_bf16": t(wgps * 1024), "matrix_int8": t(wgps * 1024),
                "matrix_int4": t(wgps * 2048), "matrix_fp32": t(cus * 64 * 2 * 2),
                "vector_fp32": t(cus * 64 * 2 * 2), "vector_fp16": t(cus * 64 * 2 * 2 * 2),
                "vector_bf16": t(cus * 64 * 2 * 2), "vector_int8": t(cus * 64 * 8)}}


def overlap_ns(keep):
    """Per dispatch event: ns of its run that other dispatches also ran (HRX issues independent dispatches
    without a barrier, so concurrent kernels share the GPU and each one's duration includes the other)."""
    evs = sorted((r for _, ex in keep for g in ex["groups"].values() for r in g), key=lambda r: int(r["start_tick"]))
    spans = []
    for r in evs:
        start = int(r["start_tick"]) * TICK_NS
        spans.append((start, start + float(r.get("duration_ns") or int(r["duration_ticks"]) * TICK_NS), r))
    out = {}
    for i, (s0, e0, r) in enumerate(spans):
        cuts = []
        for j in range(max(0, i - 16), len(spans)):
            s1, e1, _ = spans[j]
            if j == i:
                continue
            if s1 >= e0:
                break
            lo, hi = max(s0, s1), min(e0, e1)
            if hi > lo:
                cuts.append((lo, hi))
        covered, end = 0.0, s0
        for lo, hi in sorted(cuts):
            lo = max(lo, end)
            if hi > lo:
                covered += hi - lo
                end = hi
        out[id(r)] = covered
    return out


def summary(keep, arch, top=25):
    """Per (op, shape) roofline table straight from the converter, same formulas TraceLens uses."""
    rows = collections.OrderedDict()
    total = 0.0
    overlap = overlap_ns(keep)
    for _, ex in keep:
        p = ex["program"]
        for gid, evs in ex["groups"].items():
            busy = sum(float(r.get("duration_ns") or int(r["duration_ticks"]) * TICK_NS) for r in evs) / 1e3
            total += busy
            if p is None or not isinstance(gid, int):
                key = (evs[0]["key"], "", "")
                m = None
            else:
                match = p["matches"][gid]
                m = model_match(match)
                if m is None:
                    continue
                key = (m["name"], m["fused"][:60], str(m["input_dims"][:2]))
            row = rows.setdefault(key, {"n": 0, "us": 0.0, "flops": 0.0, "bytes": 0.0, "m": m, "ovl": 0.0})
            row["n"] += 1
            row["us"] += busy
            row["ovl"] += sum(overlap.get(id(r), 0.0) for r in evs) / 1e3
            if m:
                row["flops"] += m["flops"]
                row["bytes"] += m["bytes"]
    peaks = arch["max_achievable_tflops"]
    bw = arch["mem_bw_gbps"]
    print(f"{'op':22} {'fused':40} {'n':>5} {'us/call':>8} {'%time':>6} {'GB/s':>7} {'TFLOP/s':>8} "
          f"{'bound':>7} {'%roof':>6} {'%ovl':>5}  dims")
    for key, row in sorted(rows.items(), key=lambda kv: -kv[1]["us"])[:top]:
        m = row["m"]
        us = row["us"] / row["n"]
        gbs = row["bytes"] / (row["us"] * 1e3) if row["us"] else 0.0
        tfs = row["flops"] / (row["us"] * 1e6) if row["us"] else 0.0
        bound, roof = "", float("nan")
        if m:
            peak = peaks.get(f"{m['maf']}_{m['precision']}")
            if peak:
                ct = row["flops"] / (peak * 1e12)
                mt = row["bytes"] / (bw * 1e9)
                bound = "compute" if ct >= mt else "memory"
                roof = 100.0 * max(ct, mt) / (row["us"] * 1e-6)
        print(f"{key[0][:22]:22} {key[1][:40]:40} {row['n']:5d} {us:8.1f} {100 * row['us'] / total:6.1f} "
              f"{gbs:7.1f} {tfs:8.2f} {bound:>7} {roof:6.1f} {100 * row['ovl'] / max(row['us'], 1e-9):5.0f}  "
              f"{key[2][:70]}")
    print(f"total device busy {total / 1e3:.2f} ms over {len(keep)} graph executions (sum of kernel durations)")


def tracediff_top(diff_csv, steps=1.0, top=15):
    """TraceLens TraceDiff diff_stats.csv -> per-step kernel time of each aligned subtree (lowest common
    ancestor) in trace1 (the report's own trace) and trace2 (--comparison_json_path), largest change first."""
    import csv
    by = collections.defaultdict(lambda: {"trace1": [], "trace2": []})
    for r in csv.DictReader(open(diff_csv)):
        by[r["lowest_common_ancestor_id"]][r["source"]].append(r)

    def desc(rows):
        ops = []
        for r in sorted(rows, key=lambda r: int(r["gpu_op_uid"] or 0)):
            k = f'{r["cpu_op_name"]}[{r["Concrete Inputs"].strip(chr(40) + chr(41) + ",' ")}]'
            if not ops or ops[-1] != k:
                ops.append(k)
        return " ; ".join(ops) or "-"

    pairs = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for d in by.values():
        p = pairs[(desc(d["trace1"]), desc(d["trace2"]))]
        p[0] += sum(float(r["kernel_time"]) for r in d["trace1"])
        p[1] += sum(float(r["kernel_time"]) for r in d["trace2"])
        p[2] += 1
    t1 = sum(p[0] for p in pairs.values()) / steps
    t2 = sum(p[1] for p in pairs.values()) / steps
    print(f"kernel time per step: trace1 {t1 / 1e3:.2f} ms, trace2 {t2 / 1e3:.2f} ms ({steps:g} steps)")
    print(f"{'trace1 us':>10} {'trace2 us':>10} {'t2-t1 us':>9} {'n/step':>6}  trace1 ops  =>  trace2 ops")
    for (a, b), (x1, x2, n) in sorted(pairs.items(), key=lambda kv: -abs(kv[1][0] - kv[1][1]))[:top]:
        print(f"{x1 / steps:10.1f} {x2 / steps:10.1f} {(x2 - x1) / steps:9.1f} {n / steps:6.1f}  {a[:80]}  =>  {b[:80]}")


# ---------------------------------------------------------------- TraceLens extension (--extension_file) ----

class _HRXOp:
    """Duck-typed TraceLens perf model: reads the "hrx" arg the converter wrote."""
    category = "other"
    bwd_category = None

    def __init__(self, event, arch=None, python_path=None, **kwargs):
        self.event = event
        self.arch = arch
        self.hrx = (event.get("args") or {}).get("hrx") or {}
        self.param_details = {k: tuple(v) if isinstance(v, list) else v
                              for k, v in (self.hrx.get("params") or {}).items()}

    def flops(self):
        return float(self.hrx.get("flops") or 0.0)

    def bytes(self, *args, **kwargs):
        b = self.hrx.get("bytes")
        return None if b is None else float(b)

    def get_compute_precision(self):
        return self.hrx.get("precision")

    def get_maf_type(self):
        return self.hrx.get("maf")

    def flops_bwd(self):
        raise NotImplementedError

    def bytes_bwd(self, *args, **kwargs):
        raise NotImplementedError


class HRXGemm(_HRXOp):
    category = "GEMM"


class HRXGroupedGemm(_HRXOp):
    category = "GroupedGEMM_fwd"


class HRXAttention(_HRXOp):
    category = "SDPA_fwd"


class HRXNorm(_HRXOp):
    category = "NORM_fwd"
    sheet_category = "Normalization"


class HRXRoPE(_HRXOp):
    category = "RoPE_fwd"


class HRXReduce(_HRXOp):
    category = "reduce"
    sheet_category = "Reduce"


class HRXSSM(_HRXOp):
    category = "SSM_fwd"


class HRXElementwise(_HRXOp):
    category = "elementwise"
    sheet_category = "BinaryElementwise"


class HRXCopy(_HRXOp):
    category = "elementwise"
    sheet_category = "UnaryElementwise"


_CLASS_OF = {"gemm": HRXGemm, "grouped_gemm": HRXGroupedGemm, "attention": HRXAttention, "norm": HRXNorm,
             "rope": HRXRoPE, "reduce": HRXReduce, "ssm": HRXSSM, "copy": HRXCopy}
perf_model_extension = {}
for _op in GGML_OPS + PRIMARY_ORDER:
    _cat = CATEGORY_OF.get(_op, "copy" if _op in ("GET_ROWS", "SET_ROWS", "CPY", "CONT", "DUP", "CONCAT")
                           else "elementwise")
    perf_model_extension["hrx::" + _op.lower()] = _CLASS_OF.get(_cat, HRXElementwise)
op_category_extension = {"hrx::step": "other", "hrx::layer": "other"}


def load_programs_models(shapes_path):
    """([program line], {program id: [model_match]}) -- the FLOP/bytes model is built in ONE place, so `--summary` and
    `--by-kernel` cannot drift apart."""
    programs = [r for r in load_jsonl(shapes_path) if r.get("type") == "hrx_command_program"]
    if not programs:
        sys.exit(f"hrx2kineto: no hrx_command_program lines in {shapes_path}")
    return programs, {p["program"]: [model_match(m) for m in p["matches"]] for p in programs}


def rank_dispatches(events, shapes_path, commands_path, key_of):
    """([(key, row)], total_ns, unmapped) -- the aggregation `--by-kernel` and `--by-shape` share.

    `key_of(name, match, event)` decides the row a dispatch lands in, so the two rankings differ only in that function and
    cannot drift in how they credit FLOPs. rocprofv3 sees no HRX dispatches (HRX writes its own AQL packets), so this is
    built from HRX's own profile plus its shape log.

    A match's FLOPs are credited to the kernel with the MOST time inside that match -- the GEMM, not a quantize/scale
    companion that happens to run first -- once per execution: the instance key is (submission_id, program, match), so two
    executions of one reused program are counted twice and a companion is never credited a GEMM that does not exist.
    Dispatches the shape log does not name stay visible under the event's own key instead of being dropped.
    """
    programs, models = load_programs_models(shapes_path)
    cbmap = map_command_buffers(commands_path, programs, events)
    rows = collections.defaultdict(lambda: {"calls": 0, "ns": 0.0, "flops": 0.0, "shared": 0, "cats": set(),
                                            "params": None})
    recs, total, unmapped = [], 0.0, 0
    for e in events:
        ns = e.get("duration_ns")
        if ns is None:
            continue
        ns = float(ns)
        total += ns
        key = e.get("key") or "<unnamed>"
        name, match = key, None
        for prog, cmdmap in cbmap.get(e.get("command_buffer_id")) or []:   # best guess first, like the converter
            cmd = cmdmap.get(e.get("command_index"))
            if cmd is None:
                continue
            name = cmd.get("kernel") or cmd.get("symbol") or key
            mi = cmd.get("match")
            mm = models.get(prog["program"], [])
            if mi is not None and mi < len(mm) and mm[mi]:
                match = (mi, prog, mm[mi])
            break
        else:
            unmapped += 1
        rk = key_of(name, match[2] if match else None, e)
        rows[rk]["calls"] += 1
        rows[rk]["ns"] += ns
        if rows[rk]["params"] is None and match:
            rows[rk]["params"] = match[2]["params"]
        if match is None:
            continue
        mi, prog, m = match
        rows[rk]["cats"].add(m["category"])
        recs.append((rk, ns, (e.get("submission_id"), prog["program"], mi), m))
    if not total:
        sys.exit("hrx2kineto: no dispatch durations in the profile -- nothing measured")

    by_instance = collections.defaultdict(list)
    for rec in recs:
        by_instance[rec[2]].append(rec)
    for group in by_instance.values():
        primary = max(group, key=lambda r: r[1])
        rows[primary[0]]["flops"] += float(primary[3]["flops"] or 0.0)
        for rec in group:
            if rec is not primary:
                rows[rec[0]]["shared"] += 1
    return sorted(rows.items(), key=lambda kv: -kv[1]["ns"]), total, unmapped


def _roof_line(arch, events, total, unmapped):
    print(f"roof {float(arch['max_achievable_tflops']['matrix_fp16']):.1f} TFLOP/s fp16 WMMA "
          f"({arch['sclk_mhz']:.0f} MHz, {arch['mem_bw_gbps']:g} GB/s)  | "
          f"{len(events)} dispatches, {total / 1e6:.1f} ms GPU-busy, {unmapped} unmapped\n", file=sys.stderr)


def by_kernel(events, shapes_path, commands_path, arch, top=25):
    """Rank dispatches BY KERNEL: calls, ms, GPU-busy share, TFLOP/s and % of the fp16 WMMA roof (engine#3103).

    `--summary` ranks by ggml OP, which is the right unit for "which layer is expensive" and the wrong one for "which
    kernel is slow": a fused match is several kernels and the question is which of them takes the time.
    """
    roof = float(arch["max_achievable_tflops"]["matrix_fp16"])
    order, total, unmapped = rank_dispatches(events, shapes_path, commands_path, lambda name, m, e: name)
    _roof_line(arch, events, total, unmapped)
    print(f"{'kernel':54s} {'cat':10s} {'calls':>6s} {'ms':>8s} {'%busy':>6s} {'TFLOP/s':>8s} {'%roof':>6s}")
    for name, d in order[:top]:
        ms = d["ns"] / 1e6
        tf = d["flops"] / (d["ns"] / 1e9) / 1e12 if d["ns"] else 0.0
        cat = ",".join(sorted(c for c in d["cats"] if c)) or "-"
        more = "" if not d["shared"] else f" +{d['shared']}"
        print(f"{name:54.54s} {cat:10.10s} {d['calls']:6d} {ms:8.2f} {100 * d['ns'] / total:6.1f} "
              f"{tf:8.1f} {100 * tf / roof:5.1f}%{more}")
    print(f"{'TOTAL':54s} {'':10s} {sum(d['calls'] for _, d in order):6d} {total / 1e6:8.2f} {100.0:6.1f} "
          f"{sum(d['flops'] for _, d in order) / (total / 1e9) / 1e12:8.1f}")
    return 0


def by_shape(events, shapes_path, commands_path, arch, top=25):
    """Rank a profile by ggml op AND SHAPE: (op, K, N), with the launch width the shape implies (engine#3103).

    A per-kernel average can hide a bimodal population. On Qwen3-4B pp512 the same `q4_k_f16_wmma_prefill` kernel runs at
    54 % of roof where N is wide and at 18 % where N is 1024, and the average (40 %) belongs to neither -- which shape you
    are looking at is the difference between "the tile is fine" and "this launch uses a fifth of the machine".

    `workgroups` is the packed prefill schedule's only parallel axis: `kernel-corpus/kernels/loom-libs/ops/
    mul_mat_f32_f32_wmma.loom` launches `workgroups(token_tiles = 1, output_size/64, 1)`, so a narrow N means few
    workgroups on a 40-CU part. It is printed for the reader to check against %roof -- not asserted as a law: the same
    table at M=2048 has 64 workgroups below 40, so N is one of two terms (the other is the K-work per wave).
    """
    def key_of(name, m, e):
        if not m:
            return (name, None, None)
        pr = m["params"]
        return (m["name"], pr.get("K"), pr.get("N"))

    roof = float(arch["max_achievable_tflops"]["matrix_fp16"])
    order, total, unmapped = rank_dispatches(events, shapes_path, commands_path, key_of)
    _roof_line(arch, events, total, unmapped)
    print(f"{'op':16s} {'K':>7s} {'N':>8s} {'workgroups':>10s} {'calls':>6s} {'ms':>8s} {'%busy':>6s} "
          f"{'TFLOP/s':>8s} {'%roof':>6s}")
    for (name, k_dim, n_dim), d in order[:top]:
        ms = d["ns"] / 1e6
        tf = d["flops"] / (d["ns"] / 1e9) / 1e12 if d["ns"] else 0.0
        wg = str(n_dim // 64) if isinstance(n_dim, int) and n_dim else "-"
        print(f"{name.replace('hrx::', ''):16.16s} {str(k_dim or '-'):>7s} {str(n_dim or '-'):>8s} {wg:>10s} "
              f"{d['calls']:6d} {ms:8.2f} {100 * d['ns'] / total:6.1f} {tf:8.1f} {100 * tf / roof:5.1f}%")
    print(f"{'TOTAL':16s} {'':7s} {'':8s} {'':10s} {sum(d['calls'] for _, d in order):6d} {total / 1e6:8.2f} "
          f"{100.0:6.1f} {sum(d['flops'] for _, d in order) / (total / 1e9) / 1e12:8.1f}")
    return 0


def self_test():
    """Arms for --by-kernel, built from a synthetic profile pair so they need no device, no GPU ticket and no shape log
    from a real run. The two properties that are easy to get wrong and were both wrong in the first version:

      * a match's FLOPs belong to the kernel that takes the MOST time in it (crediting the first command gave the biggest
        kernel 0.0 TFLOP/s, because a quantize companion runs first);
      * an instance is (submission, program, match) -- two executions of one reused program are two GEMMs, and merging
        them halves the count while keeping the time.
    """
    import tempfile
    ok = [0, 0]

    def chk(cond, what):
        ok[0 if cond else 1] += 1
        print(("ok   " if cond else "FAIL ") + what)

    def tensor(name, ne, typ):
        nb = []; stride = 4
        for n in ne:
            nb.append(stride); stride *= n
        return {"id": hash(name) & 0xffff, "name": name, "type": typ, "ne": ne, "nb": nb}

    # one ggml MUL_MAT: weight [K=2048, N=4096] q4_K, activation [2048, T=256], out [4096, 256]
    node = {"op": "MUL_MAT", "index": 1, "desc": "MUL_MAT",
            "out": tensor("out-0", [4096, 256, 1, 1], "f32"),
            "src": [tensor("blk.0.w.weight", [2048, 4096, 1, 1], "q4_K"),
                    tensor("act-0", [2048, 256, 1, 1], "f32")]}
    program = {"type": "hrx_command_program", "program": 0, "graph_uid": 1,
               "commands": [{"phase": "main", "ordinal": 0, "kernel": "hip:pack", "symbol": "pack", "match": 0},
                            {"phase": "main", "ordinal": 1, "kernel": "hip:gemm", "symbol": "gemm", "match": 0}],
               "matches": [{"nodes": [node]}]}

    def ev(i, cb, idx, key, ns, sub):
        return {"type": "dispatch_event", "key": key, "valid": True, "command_buffer_id": cb,
                "command_index": idx, "start_tick": 1000 * i, "duration_ns": ns, "submission_id": sub}

    # two executions of the same program: in each, the pack companion runs FIRST and the gemm takes longer
    # durations shaped like a real prefill tile: the companion is short, the GEMM is milliseconds
    gemm_ns, pack_ns = 4.0e6, 5.0e5
    events = [ev(1, 7, 0, "hip:pack", pack_ns, 1), ev(2, 7, 1, "hip:gemm", gemm_ns, 1),
              ev(3, 7, 0, "hip:pack", pack_ns, 2), ev(4, 7, 1, "hip:gemm", gemm_ns, 2)]
    with tempfile.TemporaryDirectory() as td:
        shapes = pathlib.Path(td, "shapes.jsonl")
        shapes.write_text(json.dumps(program) + "\n")
        events_path = pathlib.Path(td, "ev.jsonl")
        events_path.write_text("".join(json.dumps(e) + "\n" for e in events))
        arch = gpu_arch(256.0, 2900.0)
        programs, models = load_programs_models(str(shapes))
        m = models[0][0]
        chk(m is not None and m["category"] == "gemm", "the fixture reads as a gemm match (so the roof column means something)")
        want = m["flops"]
        evs = load_dispatch_events(str(events_path))
        cbmap = map_command_buffers(None, programs, evs)
        chk(cbmap.get(7) is not None, "the synthetic command buffer matches a logged program (the join the tool depends on)")
        # run the ranking exactly as main() does, capturing rows through a tiny re-implementation of its crediting rule
        rows = collections.defaultdict(float)
        inst = collections.defaultdict(list)
        for e in evs:
            cmd = cbmap[7][0][1][e["command_index"]]
            inst[(e["submission_id"], 0, cmd["match"])].append((cmd["kernel"], float(e["duration_ns"])))
        for key, group in inst.items():
            primary = max(group, key=lambda r: r[1])
            rows[primary[0]] += want
        chk(rows.get("hip:gemm", 0.0) == 2 * want,
            "a match's FLOPs go to the kernel with the most time in it, once per execution (the first-command bug)")
        chk("hip:pack" not in rows, "the quantize companion is not credited the GEMM's FLOPs")
        chk(len(inst) == 2, "two executions (submissions) are two instances, not one merged instance")
        total = sum(e["duration_ns"] for e in evs)
        roof = float(arch["max_achievable_tflops"]["matrix_fp16"])
        tf = rows["hip:gemm"] / (2 * gemm_ns / 1e9) / 1e12           # FLOPs over the seconds those FLOPs took
        chk(58.0 <= roof <= 60.0, f"the roof is the 8060S fp16 WMMA number ({roof:.2f} TFLOP/s, not a plug value)")
        chk(abs(rows["hip:gemm"] - 2 * 2.0 * 2048 * 4096 * 256) < 1.0,
            "the fixture's FLOPs are 2*K*N*T per matched execution (the arithmetic the table divides by)")
        chk(1.5 < 100 * tf / roof < 2.2,
            f"%roof is FLOP/s over that roof ({tf:.2f} TFLOP/s -> {100 * tf / roof:.1f} % of {roof:.1f})")
        chk(total == 2 * gemm_ns + 2 * pack_ns,
            "time is summed per dispatch, not per match (a fused match is several kernels' time)")

    # engine#3103: a key the profile SHORTENED must not cost the whole command buffer. Both halves are pinned: the real
    # pair (871 exact + 1 prefix) joins, and a program where two commands share the prefix must still refuse to guess.
    def cb_fixture(second_symbol):
        # Both event keys are the shapes they claim to be: key 0 is a PREFIX of command 0's symbol (the measured defect),
        # key 1 is command 1's symbol EXACTLY. My first version keyed the second dispatch "other", which matches no
        # command name at all -- so align() refused at position 1 whatever the uniqueness rule did, and the ambiguity
        # arm passed while proving nothing (review, 03:36Z: the mutant `len(hits) >= 1` stayed green).
        prog = {"type": "hrx_command_program", "program": 1, "graph_uid": 1,
                "commands": [{"phase": "main", "ordinal": 0, "kernel": "qwen3_moe:ggml_linear_q6k_q8_1_x4",
                              "symbol": "ggml_linear_q6k_q8_1_x4_wave64_block4", "match": 0},
                             {"phase": "main", "ordinal": 1, "kernel": "qwen3_moe:other",
                              "symbol": second_symbol, "match": 0}],
                "matches": [{"nodes": [node]}]}
        evs = [{"type": "dispatch_event", "key": "ggml_linear_q6k_q8_1_x4", "valid": True, "command_buffer_id": 9,
                "command_index": 1, "start_tick": 1000, "duration_ns": 100.0, "submission_id": 1},
               {"type": "dispatch_event", "key": second_symbol, "valid": True, "command_buffer_id": 9,
                "command_index": 2, "start_tick": 2000, "duration_ns": 100.0, "submission_id": 1}]
        return prog, evs

    prog, evs = cb_fixture("other_symbol")                                  # unique prefix -> resolves
    joined = map_command_buffers(None, [prog], evs)
    chk(9 in joined, "a key the profile shortened (prefix of exactly one command) still joins the program (engine#3103)")

    prog, evs = cb_fixture("ggml_linear_q6k_q8_1_x4_wave32_block2")         # both share the prefix -> refuse
    chk(9 not in map_command_buffers(None, [prog], evs),
        "... but an AMBIGUOUS prefix joins nothing: the tool refuses to guess which command a dispatch was")

    # engine#3103: the by-shape ranking must split one op into its shapes (the average hides a bimodal population), and the
    # workgroup column must be the packed schedule's own `output_size/64`, not a guess.
    n_wide = {"op": "MUL_MAT", "index": 1, "desc": "MUL_MAT",
              "out": tensor("out-0", [4096, 256, 1, 1], "f32"),
              "src": [tensor("blk.0.q.weight", [2048, 4096, 1, 1], "q4_K"), tensor("act-0", [2048, 256, 1, 1], "f32")]}
    n_narrow = {"op": "MUL_MAT", "index": 2, "desc": "MUL_MAT",
                "out": tensor("out-1", [512, 256, 1, 1], "f32"),
                "src": [tensor("blk.0.k.weight", [2048, 512, 1, 1], "q4_K"), tensor("act-1", [2048, 256, 1, 1], "f32")]}
    prog2 = {"type": "hrx_command_program", "program": 2, "graph_uid": 2,
             "commands": [{"phase": "main", "ordinal": 0, "kernel": "hip:wide", "symbol": "wide", "match": 0},
                          {"phase": "main", "ordinal": 1, "kernel": "hip:narrow", "symbol": "narrow", "match": 1}],
             "matches": [{"nodes": [n_wide]}, {"nodes": [n_narrow]}]}
    evs2 = [{"type": "dispatch_event", "key": "wide", "valid": True, "command_buffer_id": 11,
             "command_index": 1, "start_tick": 1000, "duration_ns": 2000000.0, "submission_id": 1},
            {"type": "dispatch_event", "key": "narrow", "valid": True, "command_buffer_id": 11,
             "command_index": 2, "start_tick": 3000, "duration_ns": 4000000.0, "submission_id": 1}]
    with tempfile.TemporaryDirectory() as td:
        sp = pathlib.Path(td, "shapes.jsonl"); sp.write_text(json.dumps(prog2) + "\n")
        ep = pathlib.Path(td, "ev.jsonl"); ep.write_text("".join(json.dumps(e) + "\n" for e in evs2))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            by_shape(load_dispatch_events(str(ep)), str(sp), None, arch)
        out = buf.getvalue()
        chk(" 512" in out and "4096" in out, "by-shape splits one op into its shapes (K2048/N4096 and K2048/N512 are two rows)")
        rows = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 9 and parts[0] == "mul_mat":
                rows[parts[2]] = {"workgroups": parts[3], "pct": float(parts[-1].rstrip("%"))}
        chk(rows.get("512", {}).get("workgroups") == "8",
            "by-shape prints the packed schedule's workgroup count for N=512 (output_size/64 = 8), not a guess")
        chk("512" in rows and "4096" in rows and rows["512"]["pct"] < rows["4096"]["pct"] / 5,
            "by-shape reports %roof per SHAPE: the narrow launch is far below the wide one at the same K "
            f"({rows.get('512', {}).get('pct')}% vs {rows.get('4096', {}).get('pct')}%)")

    print(f"hrx2kineto --by-kernel self-test: {ok[0]} ok, {ok[1]} failed")
    return 0 if ok[1] == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("events", nargs="?", help="iree-profile dispatch --format=jsonl --dispatch_events output")
    ap.add_argument("out", nargs="?", help="Kineto trace JSON to write")
    ap.add_argument("csv", nargs="?", help="optional rocprofv3-style kernel_trace.csv")
    ap.add_argument("--shapes", help="GGML_HRX_DISPATCH_SHAPE_LOG file (enables hrx:: ops with shapes)")
    ap.add_argument("--commands", help="iree-profile command --format=jsonl output (command_index map)")
    ap.add_argument("--phase", choices=["all", "decode", "prefill"], default="all",
                    help="keep graph executions of <= --decode-max-tokens tokens (decode) or more (prefill)")
    ap.add_argument("--decode-max-tokens", type=int, default=1, help="token count still called decode (MTP: 2-4)")
    ap.add_argument("--skip", type=int, default=0, help="drop the first N kept steps (forward passes; warmup)")
    ap.add_argument("--max", type=int, default=None, help="keep at most N steps")
    ap.add_argument("--write-arch", help="write a TraceLens gpu arch JSON for this gfx1151 box")
    ap.add_argument("--mem-bw-gbps", type=float, default=256.0, help="roofline bandwidth (256 = theoretical)")
    ap.add_argument("--sclk-mhz", type=float, default=None, help="clock for peak TFLOPS (default: sysfs max)")
    ap.add_argument("--summary", action="store_true", help="print a per-op roofline table")
    ap.add_argument("--by-kernel", type=int, nargs="?", const=25, default=0, metavar="N",
                    help="print the per-KERNEL ranking instead (calls, ms, TFLOP/s, %% of the fp16 WMMA roof); needs "
                         "--shapes. The unit #3103 asks about: --summary ranks a fused match as one op")
    ap.add_argument("--by-shape", type=int, nargs="?", const=25, default=0, metavar="N",
                    help="rank by ggml op AND shape, (op, K, N), with the launch width that shape implies; needs --shapes. "
                         "The unit a per-kernel average hides: one kernel, one roof, two populations")
    ap.add_argument("--self-test", action="store_true", help="run the fixture arms for --by-kernel and exit")
    ap.add_argument("--tracediff-top", metavar="DIFF_STATS_CSV", help="summarise TraceDiff diff_stats.csv")
    ap.add_argument("--steps-from", metavar="TRACE", help="with --tracediff-top: per-step numbers (hrx::step count)")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.tracediff_top:
        steps = 1.0
        if a.steps_from:
            steps = float(sum(1 for e in json.load(open(a.steps_from))["traceEvents"] if e["name"] == "hrx::step"))
        tracediff_top(a.tracediff_top, max(steps, 1.0))
        return
    arch = gpu_arch(a.mem_bw_gbps, a.sclk_mhz)
    if a.write_arch:
        json.dump(arch, open(a.write_arch, "w"), indent=2)
        print(f"hrx2kineto: gpu arch ({arch['sclk_mhz']:.0f} MHz, {a.mem_bw_gbps:g} GB/s) -> {a.write_arch}")
    if not a.events:
        if not a.write_arch:
            ap.error("need EVENTS.jsonl OUT.trace.json")
        return
    if not a.out:
        ap.error("need OUT.trace.json")
    events = load_dispatch_events(a.events)
    if a.shapes:
        if a.by_kernel:
            by_kernel(events, a.shapes, a.commands, arch, a.by_kernel)
        elif a.by_shape:
            by_shape(events, a.shapes, a.commands, arch, a.by_shape)
        else:
            keep, _ = convert_shapes(events, a.out, a.shapes, a.commands, a.phase, a.skip, a.max,
                                     a.decode_max_tokens)
            if a.summary:
                summary(keep, arch)
        t0 = int(events[0]["start_tick"])
    else:
        t0 = convert_basic(events, a.out)
    if a.csv:
        write_kernel_trace_csv(events, t0, a.csv)


if __name__ == "__main__":
    sys.exit(main())
