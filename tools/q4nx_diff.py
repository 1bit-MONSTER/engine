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
"""Compare two Q4NX containers, and decode either tile layout against a GGUF.

A Q4NX `model.q4nx` is a safetensors-style container: u64 header length, a JSON
table of {tensor: {dtype, shape, data_offsets}} (+ an optional `__metadata__`),
then the data.  Two converters can produce the same container *format* and the
same tensor geometry and still disagree in the payload, so this tool separates
three questions that a byte comparison alone cannot:

  compare  which tensors are byte-identical, and for the quantized ones whether it
           is the scale plane, the min plane or the codes that differ.  A
           converter that keeps Q4_0 as "scale d, min -8d, unsigned codes" and one
           that keeps it as "scale d, min 0, signed codes" are both exact against
           the GGUF and share not one min byte -- the planes say which.
  decode   decode a container under a named layout and compare against the GGUF
           ground truth, so "the files differ" becomes "each is exact under its
           own decoder, and neither reads the other's".

Layouts (both are 32x256 tiles of 5120 B: 512 scale, 512 min, 4096 codes):

  ours  codes are [row-half(2)][group(8)][col-in-group(32)][row-pair(8)], the
        16-row "parallel" split the NPU kernel walks (npu/q4nx.cpp); nibbles are
        unsigned and the Q4_0 zero-point lives in the min plane as -8d.
  flm   codes are plain column major [col(256)][row-pair(16)]; the official
        gguf->Q4NX converter keeps Q4_0 as min 0 with *signed* nibbles (q-8).

usage:
  tools/q4nx_diff.py compare A/model.q4nx B/model.q4nx [--summary]
  tools/q4nx_diff.py decode  --gguf model.gguf --q4nx DIR --layout ours|flm [-v]

Needs numpy and gguf (gguf-py).
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

ROWS, COLS, GROUP, TILE = 32, 256, 32, 5120
PLANE_SCALE, PLANE_MIN, PLANE_CODES = 512, 512, 4096


def load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    return hdr, 8 + n, np.memmap(path, dtype=np.uint8, mode="r")


def tensor_bytes(hdr, base, mm, name):
    lo, hi = hdr[name]["data_offsets"]
    return np.asarray(mm[base + lo:base + hi])


def bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def tiles_of(raw, rows, cols):
    """Split a tile tensor into (ntiles, 5120); `rows`/`cols` are the weight's."""
    if rows % ROWS or cols % COLS:
        raise ValueError("not a whole tile grid: %dx%d" % (rows, cols))
    return np.asarray(raw).reshape(-1, TILE)


def decode_ours(t, rows, cols):
    RT, KT = rows // ROWS, cols // COLS
    t = t.reshape(RT, KT, TILE)
    sc = bf16(t[..., :PLANE_SCALE].copy().view(np.uint16)).reshape(RT, KT, 8, ROWS)
    zr = bf16(t[..., PLANE_SCALE:PLANE_SCALE + PLANE_MIN].copy().view(np.uint16)
              ).reshape(RT, KT, 8, ROWS)
    p = t[..., PLANE_SCALE + PLANE_MIN:].reshape(RT, KT, 2, 8, COLS // 8, 8)
    c = np.stack([p & 0x0F, p >> 4], axis=-1)                  # [..., r2, parity]
    c = c.transpose(0, 2, 5, 6, 1, 3, 4).reshape(RT, ROWS, KT, 8, COLS // 8)
    s = sc.transpose(0, 3, 1, 2)[..., None]
    z = zr.transpose(0, 3, 1, 2)[..., None]
    return (c * s + z).reshape(rows, cols)


def decode_flm(t, rows, cols, signed=True):
    RT, KT = rows // ROWS, cols // COLS
    t = t.reshape(RT, KT, TILE)
    sc = bf16(t[..., :PLANE_SCALE].copy().view(np.uint16)).reshape(RT, KT, 8, ROWS)
    mn = bf16(t[..., PLANE_SCALE:PLANE_SCALE + PLANE_MIN].copy().view(np.uint16)
              ).reshape(RT, KT, 8, ROWS)
    p = t[..., PLANE_SCALE + PLANE_MIN:].reshape(RT, KT, COLS, ROWS // 2)
    lo = (p & 0x0F).astype(np.int8)
    hi = ((p >> 4) & 0x0F).astype(np.int8)
    if signed:                      # the official Q4_0 path stores q-8 as a signed nibble
        lo = np.where(lo > 7, lo - 16, lo)
        hi = np.where(hi > 7, hi - 16, hi)
    c = np.stack([lo, hi], axis=-1).reshape(RT, KT, COLS, ROWS)
    g = np.arange(COLS) // GROUP
    return (sc[:, :, g, :] * c + mn[:, :, g, :]).transpose(0, 3, 1, 2).reshape(rows, cols)


DECODERS = {"ours": decode_ours, "flm": decode_flm}


def compare_cmd(args):
    ha, basea, ma = load(args.a)
    hb, baseb, mb = load(args.b)
    print("A header %d B, data at %d | B header %d B, data at %d"
          % (basea - 8, basea, baseb - 8, baseb))
    meta = [(ha.pop("__metadata__", None), "A"), (hb.pop("__metadata__", None), "B")]
    for m, tag in meta:
        print("  __metadata__ %s: %s" % (tag, "absent" if m is None else json.dumps(m)[:120]))
    ka, kb = set(ha), set(hb)
    print("tensors: A=%d B=%d; only A=%d only B=%d" % (len(ha), len(hb), len(ka - kb), len(kb - ka)))
    for k in sorted(ka - kb):
        print("   only A:", k)
    for k in sorted(kb - ka):
        print("   only B:", k)
    common = sorted(ka & kb)
    dt = [(k, ha[k]["dtype"], hb[k]["dtype"]) for k in common if ha[k]["dtype"] != hb[k]["dtype"]]
    sh = [(k, ha[k]["shape"], hb[k]["shape"]) for k in common if ha[k]["shape"] != hb[k]["shape"]]
    print("dtype diffs: %d | shape diffs: %d" % (len(dt), len(sh)))
    for x in dt[:8]:
        print("   ", x)
    for x in sh[:8]:
        print("   ", x)

    same, diff, planes = [], [], {}
    for k in common:
        da, db = ha[k], hb[k]
        if da.get("shape") != db.get("shape"):
            diff.append(k)
            continue
        a = tensor_bytes(ha, basea, ma, k)
        b = tensor_bytes(hb, baseb, mb, k)
        if a.tobytes() == b.tobytes():
            same.append(k)
            continue
        diff.append(k)
        if da["dtype"] == "I8" and a.size and a.size % TILE == 0 and b.size == a.size:
            ta = a.reshape(-1, TILE)
            tb = b.reshape(-1, TILE)
            for lo, hi, label in ((0, PLANE_SCALE, "scale"),
                                  (PLANE_SCALE, PLANE_SCALE + PLANE_MIN, "min"),
                                  (PLANE_SCALE + PLANE_MIN, TILE, "codes")):
                eq = int((ta[:, lo:hi] == tb[:, lo:hi]).all(axis=1).sum())
                tot = ta.shape[0]
                acc = planes.setdefault(label, [0, 0])
                acc[0] += eq
                acc[1] += tot
    print("\nper-tensor bytes: identical=%d differing=%d" % (len(same), len(diff)))
    for label in ("scale", "min", "codes"):
        if label in planes:
            eq, tot = planes[label]
            print("  plane %-5s identical tiles: %d/%d" % (label, eq, tot))
    if not args.summary:
        for k in diff[:20]:
            print("   differs:", k)
        if len(diff) > 20:
            print("   ... %d more" % (len(diff) - 20))
    return 0


def decode_cmd(args):
    import gguf
    from gguf.quants import dequantize

    hdr, base, mm = load(os.path.join(args.q4nx, "model.q4nx"))
    dec = DECODERS[args.layout]
    r = gguf.GGUFReader(args.gguf)
    want = [t for t in r.tensors if t.name.endswith(".weight") or t.name in ("output.weight",)]
    if args.tensor:
        want = [t for t in want if args.tensor in t.name]
    if not want:
        print("no tensors matched", file=sys.stderr)
        return 2
    exact = inexact = skipped = 0
    worst = (0.0, "")
    # our converter's names, so a GGUF tensor can be looked up in the container
    lay = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
           "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm",
           "attn_k_norm": "self_attn.k_norm", "ffn_gate": "mlp.gate_proj",
           "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj", "attn_norm": "input_layernorm",
           "ffn_norm": "post_attention_layernorm", "token_embd": "model.embed_tokens",
           "output_norm": "model.norm", "output": "lm_head"}

    def q4nx_name(n):
        head, _, field = n.rpartition(".")
        if head.startswith("blk."):
            idx = head.split(".")[1]
            return "model.layers.%s.%s.weight" % (idx, lay[head.split(".", 2)[2]])
        return lay[head] + "." + field

    for t in want:
        if len(t.shape) < 2:            # 1-D norms: nothing tiled to decode
            skipped += 1
            continue
        name = q4nx_name(t.name)
        if name not in hdr:
            skipped += 1
            continue
        cols, rows = int(t.shape[0]), int(t.shape[1])
        d = hdr[name]
        if d["dtype"] != "I8":
            skipped += 1
            continue
        raw = tensor_bytes(hdr, base, mm, name)
        try:
            got = dec(raw, rows, cols).astype(np.float64)
        except ValueError as e:
            print("skip %-46s %s" % (name, e))
            skipped += 1
            continue
        ref = dequantize(t.data, t.tensor_type).astype(np.float64).reshape(rows, cols)
        err = np.abs(got - ref)
        ok = not err.any()
        exact += ok
        inexact += (not ok)
        if err.max() > worst[0]:
            worst = (float(err.max()), t.name)
        if args.verbose or not ok:
            print("%s %-46s %-5s max|err| %.3e  rms %.3e"
                  % ("ok  " if ok else "BAD ", t.name, t.tensor_type.name, err.max(),
                     float(np.sqrt((err ** 2).mean()))))
    print("\nlayout=%s: exact=%d inexact=%d skipped=%d   worst max|err| %.3e (%s)"
          % (args.layout, exact, inexact, skipped, worst[0], worst[1] or "-"))
    return 0 if inexact == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="compare two model.q4nx containers")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--summary", action="store_true", help="counts only")
    c.set_defaults(func=compare_cmd)
    d = sub.add_parser("decode", help="decode a container and check it against its GGUF")
    d.add_argument("--gguf", required=True)
    d.add_argument("--q4nx", required=True, help="the model directory")
    d.add_argument("--layout", required=True, choices=sorted(DECODERS))
    d.add_argument("--tensor", help="only tensors whose GGUF name contains this")
    d.add_argument("-v", "--verbose", action="store_true")
    d.set_defaults(func=decode_cmd)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
