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
"""Write the GGUF that carries exactly the weights of a Q4NX model (the inverse of gguf_to_q4nx.py).

usage: tools/q4nx_to_gguf.py write  <model.q4nx> <source.gguf> <out.gguf>
       tools/q4nx_to_gguf.py verify <model.q4nx> <out.gguf> [--roundtrip <source.gguf>]

WHY. gguf_to_q4nx.py keeps the 4-bit codes of Q4_0 / Q4_1 / Q4_K tensors but rounds their scale and
zero to bf16, and it requantizes every other type (Q5_K, Q6_K, Q8_0, F16, ...) to 4-bit q4_1, so a
model.q4nx does not hold the weights of the GGUF it came from. A GPU that decodes on the source GGUF
and an NPU that runs model.q4nx then compute two different models. This writes the GGUF of the Q4NX
model itself, so both sides decode the same weights:

  q4_1 tiles  (every projection and lm_head)  -> GGML Q4_1 blocks: the tile's 4-bit codes, d = its
              bf16 scale and m = its bf16 zero, both as f16. bf16 -> f16 is exact for every value
              that is a normal or subnormal f16 at full precision; `write` checks every scale and zero
              and stops if one is not (no such case is handled, because none was found).
  bf16        (norms, the embedding)          -> F32 for 1-D tensors (exact), F16 for the embedding
              when every value survives bf16 -> f16 exactly, else F32 (always exact).

lm_head becomes output.weight even when the source ties it to token_embd: gguf_to_q4nx.py builds a
tied lm_head from token_embd and quantizes it, so the Q4NX head is not the embedding any more.

The metadata (architecture, hyperparameters, tokenizer) is copied from the source GGUF; only
general.file_type changes (MOSTLY_Q4_1). Tensors keep the source's names and order; output.weight
goes last, where gguf_to_q4nx.py puts a tied lm_head, so a round trip packs the same file.

The Q4NX file must be one gguf_to_q4nx.py wrote (its __metadata__ says so): tiles row-major over the
32 x 256 grid, gate and up as separate tensors, no row reordering. Other Q4NX producers are refused.

verify: dequant(out.gguf) == dequant(model.q4nx) on every tensor, max |diff| 0 (gguf-py's
dequantize against the decoder of npu/q4nx.cpp). --roundtrip: gguf_to_q4nx.py pack of out.gguf,
under the source's file name (the name is recorded in the header), is byte-identical to model.q4nx.

Needs numpy and gguf (gguf-py).
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import gguf
from gguf.quants import dequantize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_to_q4nx as g2q  # noqa: E402

ROWS, COLS, TILE = g2q.ROWS, g2q.COLS, g2q.TILE


def read_q4nx(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    base = 8 + n

    def raw(name):
        lo, hi = hdr[name]["data_offsets"]
        return np.asarray(mm[base + lo:base + hi])
    return hdr, raw


def tile_parts(raw, N, K):
    """[tiles, 5120] -> codes u8 [N, K], scale and zero bf16 bits u16 [N, K/32]."""
    RT, KT = N // ROWS, K // COLS
    t = raw.reshape(RT, KT, TILE)
    sc = t[..., :512].copy().view(np.uint16).reshape(RT, KT, 8, 32).transpose(0, 3, 1, 2).reshape(N, K // 32)
    zr = t[..., 512:1024].copy().view(np.uint16).reshape(RT, KT, 8, 32).transpose(0, 3, 1, 2).reshape(N, K // 32)
    p = t[..., 1024:].reshape(RT, KT, 2, 8, 32, 8)  # rt, kt, h, g, j, r2
    c = np.stack([p & 0xF, p >> 4], axis=-1)  # ..., r2, parity
    codes = c.transpose(0, 2, 5, 6, 1, 3, 4).reshape(N, K)  # rt, r, kt, g, j
    return codes.astype(np.uint8), sc, zr


def bf16_to_f16_exact(bits, what):
    f = g2q.from_bf16(bits)
    h = f.astype(np.float16)
    bad = h.astype(np.float32) != f
    return h, int(bad.sum()), (float((np.abs(h.astype(np.float32) - f)[bad] / np.abs(f[bad])).max()) if bad.any() else 0.0)


def q4_1_blocks(codes, d16, m16):
    """GGML block_q4_1 (f16 d, f16 m, qs[16]: element j low nibble, j + 16 high) -> u8 [N, K/32*20]."""
    N, K = codes.shape
    c = codes.reshape(N, K // 32, 32)
    qs = (c[..., :16] | (c[..., 16:] << 4)).astype(np.uint8)
    out = np.empty((N, K // 32, 20), np.uint8)
    out[..., 0:2] = d16.reshape(N, K // 32, 1).view(np.uint8)
    out[..., 2:4] = m16.reshape(N, K // 32, 1).view(np.uint8)
    out[..., 4:] = qs
    return out.reshape(N, K // 32 * 20)


def copy_metadata(r, w):
    for f in r.fields.values():
        if f.name.startswith("GGUF.") or f.name in ("general.architecture", "general.file_type", "general.alignment"):
            continue
        vt = f.types[0]
        if vt == gguf.GGUFValueType.ARRAY:
            w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1])
        else:
            w.add_key_value(f.name, f.contents(), vt)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_1)


def write(args):
    hdr, raw = read_q4nx(args.q4nx)
    meta = hdr.get("__metadata__", {})
    if meta.get("converter") != "tools/gguf_to_q4nx.py":
        raise SystemExit(f"{args.q4nx}: not written by tools/gguf_to_q4nx.py ({meta}); its tile order is unknown")
    r = gguf.GGUFReader(args.source)
    arch = g2q.field(r, "general.architecture")
    align = g2q.field(r, "general.alignment", 32)
    w = gguf.GGUFWriter(args.out, arch)
    if align != 32:
        w.add_custom_alignment(align)
    copy_metadata(r, w)
    planned = g2q.planned(r)
    total = changed = 0
    worst = 0.0
    for name, t, kind in planned:
        gname = t.name if name != "lm_head.weight" or t.name != "token_embd.weight" else "output.weight"
        if name not in hdr:
            raise SystemExit(f"{name} is not in {args.q4nx}")
        h = hdr[name]
        if kind == "tiles":
            K, N = int(t.shape[0]), int(t.shape[1])
            codes, sc, zr = tile_parts(raw(name), N, K)
            d16, nd, wd = bf16_to_f16_exact(sc, name + " scale")
            m16, nm, wm = bf16_to_f16_exact(zr, name + " zero")
            total += sc.size + zr.size
            changed += nd + nm
            worst = max(worst, wd, wm)
            if nd or nm:
                raise SystemExit(f"{name}: {nd} scales and {nm} zeros are not exact in f16 (max rel {max(wd, wm):.3g})")
            w.add_tensor(gname, q4_1_blocks(codes, d16, m16), raw_shape=(N, K // 32 * 20),
                         raw_dtype=gguf.GGMLQuantizationType.Q4_1)
            how = "Q4_1"
        else:
            bits = raw(name).view(np.uint16)
            f = g2q.from_bf16(bits).reshape(h["shape"])
            if f.ndim == 2:
                f16v, nb, wb = bf16_to_f16_exact(bits, name)
                total += bits.size
                if nb == 0:
                    w.add_tensor(gname, f16v.reshape(f.shape), raw_dtype=gguf.GGMLQuantizationType.F16)
                    how = "F16 (exact)"
                else:
                    w.add_tensor(gname, f.astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
                    how = f"F32 ({nb} values not exact in f16)"
            else:
                w.add_tensor(gname, f.astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
                how = "F32"
        print(f"{gname:28s} <- {name:52s} {how}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"bf16 -> f16: {changed} of {total} scale/zero/embedding values change (max rel {worst:.3g})")
    print(f"wrote {args.out}")


def verify(args):
    hdr, raw = read_q4nx(args.q4nx)
    r = gguf.GGUFReader(args.gguf)
    bad, n = 0, 0
    for t in r.tensors:
        name = g2q.hf_name(t.name) if t.name != "output.weight" else "lm_head.weight"
        got = dequantize(t.data, t.tensor_type).astype(np.float32)
        h = hdr[name]
        if h["dtype"] == "I8":
            K, N = int(t.shape[0]), int(t.shape[1])
            ref = g2q.untile(raw(name), N, K).astype(np.float32)
            got = got.reshape(N, K)
        else:
            ref = g2q.from_bf16(raw(name).view(np.uint16)).reshape(got.shape)
        d = float(np.abs(got - ref).max())
        n += 1
        bad += d != 0.0
        if d != 0.0 or args.all or t.name.startswith(("blk.0.", "token_embd", "output")):
            print(f"{'ok ' if d == 0.0 else 'BAD'} {t.name:28s} {t.tensor_type.name:4s} max |gguf - q4nx| {d:.3g}")
    print(f"{n - bad}/{n} tensors dequantize identically")
    rc = 1 if bad else 0
    if args.roundtrip:
        src = hdr.get("__metadata__", {}).get("source", os.path.basename(args.roundtrip))
        with tempfile.TemporaryDirectory() as td:
            os.symlink(os.path.abspath(args.gguf), os.path.join(td, src))
            out = os.path.join(td, "q")
            subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gguf_to_q4nx.py"),
                            "pack", os.path.join(td, src), out], check=True, stdout=subprocess.DEVNULL)
            same = subprocess.run(["cmp", os.path.join(out, "model.q4nx"), args.q4nx]).returncode == 0
        print(f"round trip gguf_to_q4nx.py pack(out.gguf) {'byte-identical to' if same else 'DIFFERS from'} {args.q4nx}")
        rc |= 0 if same else 1
    return rc


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("write")
    a.add_argument("q4nx")
    a.add_argument("source")
    a.add_argument("out")
    b = sub.add_parser("verify")
    b.add_argument("q4nx")
    b.add_argument("gguf")
    b.add_argument("--roundtrip", metavar="SOURCE_GGUF", help="also pack out.gguf again and compare bytes")
    b.add_argument("--all", action="store_true", help="print every tensor")
    args = p.parse_args()
    return write(args) if args.cmd == "write" else verify(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
