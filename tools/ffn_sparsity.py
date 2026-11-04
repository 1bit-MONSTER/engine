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
"""Can a dense model skip most of its FFN rows? Two host-side halves of that question.

usage: tools/ffn_sparsity.py budget <model.gguf> [--per-layer] [--json out.json]
       tools/ffn_sparsity.py stats  <capture.csv> [--every N] [--json out.json]
       tools/ffn_sparsity.py --self-test

A token's cost is dominated by the bytes it must stream, and for a SwiGLU FFN the gate,
up and down matrices are most of them.  Skipping a fraction f of the `intermediate`
rows would save f of that stream -- but only if the rows can be skipped in the blocks a
kernel can actually address.  These are the two device-free measurements:

budget  reads a GGUF's tensor table (stdlib only, no dequantizing) and prints what share
        of one token's stream each part of the model is: ffn_gate_up, ffn_down, attn,
        ssm_gdn, embd_head, norms.  This is the ceiling: everything else is a fraction
        of the ffn_gate_up + ffn_down row.

stats   reads the per-(layer,token) capture a harness writes for the FFN intermediate
        h = silu(gate(x)) * up(x) -- the input of ffn_down -- and summarizes it.  The
        capture CSV needs one row per (layer, token) with at least `layer`, `tok`, and
        the two families of columns this tool reports:

          r90 r95 r99 / m10 m20 m30 m50
              plain row selection: the row fraction needed for 90/95/99 % of |h|_1, and
              the |h|_1 mass carried by the top 10/20/30/50 % of rows.  Suspiciously
              good numbers here are the trap: a kernel streams rows in fixed-size
              blocks, so scattered rows buy nothing.

          b<B>_e90 b<B>_e95 b<B>_e99 / b<B>_em10 .. b<B>_em50
              block granularity, blocks of B rows kept by their own energy -- the rule a
              streaming kernel can obey.  Compare with the plain columns: the gap between
              them is the part of the row sparsity that is unreachable.

        A capture also carries block-rule columns the kernel *cannot* obey
        (b<B>_r*/m*/k*/km*, a block kept because some selected row is inside it); this
        tool prints them only in the per-layer table, because they bound the answer from
        the wrong side.

Exit status is 0 on success and 1 on a malformed input, so a pipeline sees a bad capture
rather than a plausible table.
"""
import argparse
import json
import os
import struct
import sys

# ggml types this tool understands: bits per value.  Quantized types are exact as
# (block bytes * 8) / block values, so no rounding enters the byte budget.
BITS = {
    0: 32, 1: 16, 2: 18 * 8 / 32, 3: 20 * 8 / 32, 6: 22 * 8 / 32, 7: 24 * 8 / 32,
    8: 34 * 8 / 32, 9: 36 * 8 / 32, 10: 84 * 8 / 256, 11: 110 * 8 / 256, 12: 144 * 8 / 256,
    13: 176 * 8 / 256, 14: 210 * 8 / 256, 15: 292 * 8 / 256,
    16: 66 * 8 / 256, 17: 74 * 8 / 256, 18: 98 * 8 / 256, 19: 56 * 8 / 256, 20: 18 * 8 / 32,
    21: 110 * 8 / 256, 22: 82 * 8 / 256, 23: 136 * 8 / 256,
    24: 8, 25: 16, 26: 32, 27: 64, 28: 64, 30: 16,
    34: 54 * 8 / 256, 35: 66 * 8 / 256, 142: 34 * 8 / 128, 143: 28 * 8 / 128,
}

# The categories a token's stream splits into, in report order.  Names follow the GGUF
# tensor naming the engine's converter already reads (docs/npu.md, "Container").
CATEGORIES = ["ffn_gate_up", "ffn_down", "attn", "ssm_gdn", "embd_head", "norms", "other"]

# The capture columns this tool reports, in the order a reader wants them: the block rule
# first (what a kernel can do), then plain rows (what it cannot).
REPORT_KEYS = [
    "b64_e90", "b64_e95", "b64_e99", "b128_e90", "b128_e95", "b128_e99",
    "r90", "r95", "r99", "m10", "m20", "m30", "m50",
]


def category(name):
    """Which part of the stream a tensor belongs to."""
    if "ffn_gate" in name or "ffn_up" in name:
        return "ffn_gate_up"
    if "ffn_down" in name:
        return "ffn_down"
    if any(k in name for k in ("attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate")):
        return "attn"
    if "ssm" in name:
        return "ssm_gdn"
    if "token_embd" in name or name.startswith("output."):
        return "embd_head"
    if "norm" in name:
        return "norms"
    return "other"


def read_gguf_table(path):
    """(tensors, n_kv) from a GGUF header, without touching the data section."""
    f = open(path, "rb")
    try:
        def take(fmt):
            n = struct.calcsize(fmt)
            b = f.read(n)
            if len(b) != n:
                raise ValueError("truncated GGUF header")
            return struct.unpack(fmt, b)

        def string():
            (n,) = take("<Q")
            b = f.read(n)
            if len(b) != n:
                raise ValueError("truncated GGUF string")
            return b.decode("utf-8", "replace")

        def value(t):
            fixed = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                     10: "<Q", 11: "<q", 12: "<d"}
            if t in fixed:
                return take(fixed[t])
            if t == 8:
                return (string(),)
            if t == 9:
                (et,) = take("<I")
                (n,) = take("<Q")
                return tuple(value(et)[0] for _ in range(n))
            raise ValueError(f"unknown GGUF metadata type {t}")

        magic, _version, n_tensors, n_kv = take("<4sIQQ")
        if magic != b"GGUF":
            raise ValueError(f"{path} is not a GGUF (magic {magic!r})")
        for _ in range(n_kv):
            string()
            (t,) = take("<I")
            value(t)
        tensors = []
        for _ in range(n_tensors):
            name = string()
            (n_dims,) = take("<I")
            dims = take("<" + "Q" * n_dims)
            (ttype,) = take("<I")
            take("<Q")  # offset: the budget is dims * type size, so the layout is not needed
            if ttype not in BITS:
                raise ValueError(f"{name}: ggml type {ttype} is not in this tool's table")
            tensors.append((name, dims, ttype))
        return tensors
    finally:
        f.close()


def budget(tensors):
    """{category: bytes}, {(layer, category): bytes}, and the total."""
    total = dict.fromkeys(CATEGORIES, 0.0)
    per_layer = {}
    for name, dims, ttype in tensors:
        n = 1
        for d in dims:
            n *= d
        nbytes = n * BITS[ttype] / 8.0
        c = category(name)
        total[c] += nbytes
        # blk.<N>.… is one transformer block; anything else (embedding, output) has no layer
        parts = name.split(".")
        if len(parts) > 1 and parts[0] == "blk" and parts[1].isdigit():
            per_layer.setdefault(int(parts[1]), dict.fromkeys(CATEGORIES, 0.0))[c] += nbytes
    return total, per_layer


def load_capture(path):
    """[(layer, tok, {column: float})] from a capture CSV; drops a half-written last line."""
    import csv

    rows = []
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames or "layer" not in rdr.fieldnames:
            raise ValueError(f"{path}: no `layer` column, is this a capture CSV?")
        for r in rdr:
            if r.get("layer") in (None, "") or None in r.values():
                continue
            try:
                rec = {"layer": int(r["layer"]), "tok": int(r.get("tok") or 0),
                       "cols": {k: float(v) for k, v in r.items() if k not in ("layer", "tok")}}
            except (TypeError, ValueError) as e:
                raise ValueError(f"{path}: bad row {r!r} ({e})")
            rows.append(rec)
    if not rows:
        raise ValueError(f"{path}: no complete capture rows")
    return rows


def summarize(rows):
    """Per-layer means and the mean over layers for the columns the capture carries."""
    by_layer = {}
    for r in rows:
        by_layer.setdefault(r["layer"], []).append(r["cols"])
    keys = [k for k in REPORT_KEYS if all(k in c for cols in by_layer.values() for c in cols)]
    per_layer = {}
    for L, cols in by_layer.items():
        per_layer[L] = {k: sum(c[k] for c in cols) / len(cols) for k in keys}
    mean = {k: sum(per_layer[L][k] for L in per_layer) / len(per_layer) for k in keys}
    tokens = len({(r["layer"], r["tok"]) for r in rows}) // max(1, len(by_layer))
    return {"layers": len(by_layer), "tokens_per_layer": tokens, "keys": keys,
            "mean": mean, "per_layer": per_layer}


def cmd_budget(args):
    total, per_layer = budget(read_gguf_table(args.path))
    grand = sum(total.values())
    if not grand:
        raise ValueError(f"{args.path}: no tensors")
    print(f"{'category':<14} {'bytes':>16} {'share':>8}")
    for c in CATEGORIES:
        if total[c]:
            print(f"{c:<14} {int(total[c]):>16,} {100 * total[c] / grand:7.2f}%")
    print(f"{'total':<14} {int(grand):>16,} {100.0:7.2f}%")
    ffn = total["ffn_gate_up"] + total["ffn_down"]
    print(f"\nffn_gate_up + ffn_down = {int(ffn):,} B = {100 * ffn / grand:.2f}% of one token's stream;"
          f" skipping a fraction f of the intermediate rows is worth f of it, in the blocks the"
          f" kernel can address (see `stats`).")
    if args.per_layer and per_layer:
        print(f"\n{'layer':>5} " + " ".join(f"{c:>12}" for c in CATEGORIES if any(per_layer[L][c] for L in per_layer)))
        for L in sorted(per_layer):
            print(f"{L:5d} " + " ".join(f"{int(per_layer[L][c]):>12,}" for c in CATEGORIES
                                        if any(per_layer[X][c] for X in per_layer)))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"total": total, "ffn": ffn,
                       "per_layer": {str(L): per_layer[L] for L in per_layer}}, f, indent=1)
    return 0


def cmd_stats(args):
    s = summarize(load_capture(args.path))
    keys = s["keys"]
    if not keys:
        raise ValueError(f"{args.path}: no column from {', '.join(REPORT_KEYS)} in the capture")
    print(f"capture: {s['layers']} layers, ~{s['tokens_per_layer']} tokens per layer")
    print(f"\nmean over {s['layers']} layers and their tokens (%, of rows or of |h|_1);"
          f" b<B>_e* is the rule a streaming kernel can obey, r*/m* cannot:")
    for k in keys:
        print(f"  {k:>9} {s['mean'][k]:7.2f}")
    if args.every:
        print(f"\n{'layer':>5} " + " ".join(f"{k:>9}" for k in keys))
        for L in sorted(s["per_layer"]):
            if L % args.every and L != max(s["per_layer"]):
                continue
            print(f"{L:5d} " + " ".join(f"{s['per_layer'][L][k]:9.2f}" for k in keys))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(s, f, indent=1)
    return 0


def self_test():
    """Synthetic GGUF + synthetic capture, each with a hand-computed expected answer."""
    import tempfile

    bad = []

    def gguf(path, tensors):
        """A minimal GGUF: no metadata, F32 tensors, offsets laid out as the format wants."""
        with open(path, "wb") as f:
            f.write(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), 0))
            for name, dims, ttype in tensors:
                nb = name.encode()
                f.write(struct.pack("<Q", len(nb)) + nb)
                f.write(struct.pack("<I", len(dims)))
                f.write(struct.pack("<" + "Q" * len(dims), *dims))
                f.write(struct.pack("<I", ttype))
                f.write(struct.pack("<Q", 0))

    with tempfile.TemporaryDirectory() as td:
        # 2 blocks x (gate 4x2, up 4x2, down 2x4) + an embedding 8x2, all F32 = 4 bytes a value
        tensors = [
            ("token_embd.weight", (8, 2), 0),
            ("blk.0.ffn_gate.weight", (2, 4), 0),
            ("blk.0.ffn_up.weight", (2, 4), 0),
            ("blk.0.ffn_down.weight", (4, 2), 0),
            ("blk.1.ffn_gate.weight", (2, 4), 0),
            ("blk.1.ffn_up.weight", (2, 4), 0),
            ("blk.1.ffn_down.weight", (4, 2), 0),
            ("blk.1.attn_q.weight", (2, 2), 0),
            ("blk.1.attn_norm.weight", (2,), 0),
        ]
        p = os.path.join(td, "m.gguf")
        gguf(p, tensors)
        total, per_layer = budget(read_gguf_table(p))
        want = {"ffn_gate_up": 4 * 8 * 4.0, "ffn_down": 2 * 8 * 4.0, "attn": 4 * 4.0,
                "embd_head": 16 * 4.0, "norms": 2 * 4.0, "ssm_gdn": 0.0, "other": 0.0}
        for c in CATEGORIES:
            if abs(total[c] - want[c]) > 1e-9:
                bad.append(f"budget {c}: {total[c]} != {want[c]}")
        if sum(per_layer) != 1:
            bad.append(f"per-layer keys {sorted(per_layer)} (an embedding is not a block)")
        if abs(per_layer[1]["attn"] - 16.0) > 1e-9:
            bad.append(f"per-layer attn {per_layer[1]['attn']} != 16")
        if category("blk.3.ffn_gate_inp.weight") != "ffn_gate_up":
            bad.append("a router named ffn_gate_inp was not counted with the gate")
        if category("blk.3.attn_norm.weight") != "norms":
            bad.append("a norm was not counted with the norms")
        if category("blk.3.attn_gate.weight") != "attn":
            bad.append("a gated attention output was not counted with attention")

        # an unknown ggml type must be an error, not a silently short budget
        p2 = os.path.join(td, "bad.gguf")
        gguf(p2, [("blk.0.ffn_up.weight", (2, 2), 199)])
        try:
            read_gguf_table(p2)
            bad.append("ggml type 199 was accepted")
        except ValueError:
            pass

        # capture: two tokens, one layer.  Token A needs 50% of the rows for 90 % of the mass,
        # token B 30 %, so r90 = 40.  Blocks of 64 cover the same story: 4 blocks, A needs 2, B 1.
        csv_path = os.path.join(td, "c.csv")
        with open(csv_path, "w") as f:
            f.write("layer,tok,r90,r95,r99,m10,m20,m30,m50,b64_e90,b64_e95,b64_e99\n")
            f.write("0,0,50,60,80,20,35,50,70,50,66.67,80\n")
            f.write("0,1,30,40,60,40,60,70,85,25,50,66.67\n")
            f.write("0,2,30,40\n")  # half-written last line, as a killed capture leaves it
        s = summarize(load_capture(csv_path))
        if s["layers"] != 1 or s["tokens_per_layer"] != 2:
            bad.append(f"capture shape {s['layers']} layers {s['tokens_per_layer']} tokens")
        for k, v in (("r90", 40.0), ("m50", 77.5), ("b64_e90", 37.5)):
            if abs(s["mean"][k] - v) > 1e-9:
                bad.append(f"mean {k}: {s['mean'][k]} != {v}")
        if s["keys"] != ["b64_e90", "b64_e95", "b64_e99", "r90", "r95", "r99", "m10", "m20", "m30", "m50"]:
            bad.append(f"reported columns {s['keys']}")

        # a CSV without the block columns still summarizes, and one with no capture columns is refused
        with open(csv_path, "w") as f:
            f.write("layer,tok,r90\n0,0,50\n")
        if summarize(load_capture(csv_path))["keys"] != ["r90"]:
            bad.append("a rows-only capture did not report r90 alone")
        with open(csv_path, "w") as f:
            f.write("layer,tok\n0,0\n")
        try:
            s = summarize(load_capture(csv_path))
            if s["keys"]:
                bad.append("an empty capture reported columns")
        except ValueError:
            pass

    if bad:
        print("self-test FAILED:")
        for b in bad:
            print("  " + b)
        return 1
    print("self-test passed: GGUF byte budget per category and per block, unknown ggml type refused,"
          " capture means from two tokens, half-written line dropped, rows-only capture reported,"
          " capture with no measurable column refused")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true", help="synthetic GGUF and capture: pass/fail cases")
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("budget", help="byte budget of one token's stream from a GGUF")
    b.add_argument("path")
    b.add_argument("--per-layer", action="store_true")
    b.add_argument("--json")
    s = sub.add_parser("stats", help="row and block sparsity from a capture CSV")
    s.add_argument("path")
    s.add_argument("--every", type=int, default=0, help="print every Nth layer in the per-layer table")
    s.add_argument("--json")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    try:
        if args.cmd == "budget":
            return cmd_budget(args)
        if args.cmd == "stats":
            return cmd_stats(args)
    except (ValueError, OSError) as e:
        print(f"ffn_sparsity: {e}", file=sys.stderr)
        return 1
    ap.error("a subcommand is required (budget, stats) or --self-test")


if __name__ == "__main__":
    sys.exit(main())
