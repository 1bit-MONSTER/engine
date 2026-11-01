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
"""Repack a Hugging Face safetensors model into a Q4NX directory (docs/npu.md).

usage: tools/safetensors_to_q4nx.py pack   <hf dir> <out dir> [--tokenizer <dir>]
       tools/safetensors_to_q4nx.py verify <hf dir> <out dir>
       tools/safetensors_to_q4nx.py decision <hf dir> <out dir>

`pack` is the safetensors counterpart of tools/gguf_to_q4nx.py and shares its packers
(tools/q4nx_pack.py), so both write the same container: 8-byte header length, JSON tensor
table, data.  The HF tensor names ARE the Q4NX names -- gguf_to_q4nx maps GGUF names onto
these -- so nothing has to be translated here; every 2-D tensor except the embedding becomes
q4_1 tiles (I8 [tiles, 5120]) and norms, biases and the embedding stay bf16.  config.json is
the model's own HF config with the two Q4NX fields added, and the tokenizer files are copied.

`decision` converts a Decision-3.0 pack's readout head -- `readout.safetensors` plus
`decision_config.json` -- into `decision.q4nx` in the same container, so one loader reads both.
The head is one F32 matrix [codes, hidden]: the per-option logits for a pooled backbone state.

This exists because the d3 packs ship HF safetensors and the engine only had a GGUF converter
(engine #653); nothing here needs a device, a model download or gguf-py.
"""
import argparse
import json
import os
import shutil
import struct
import sys

import numpy as np

from q4nx_pack import COLS, ROWS, bf16_bits, quantize_q4_1, tiles, untile

# bf16 storage: 1-D tensors (norms, biases, rope tables) and the token embedding, as in the
# GGUF converter, plus any tensor whose dimensions are not a whole number of 32x256 tiles.
KEEP_BF16 = ("model.embed_tokens.weight",)


def read_header(path):
    """(header dict, data offset) of a safetensors file."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise SystemExit(f"{path}: not a safetensors file (no 8-byte header length)")
        n = struct.unpack("<Q", raw)[0]
        if n > 100 * 1024 * 1024:
            raise SystemExit(f"{path}: header length {n} is not plausible")
        header = json.loads(f.read(n))
    return header, 8 + n


def read_tensor(path, header, off, name):
    """One tensor as float32 (safetensors has no strides: the data is dense and row-major)."""
    t = header[name]
    dtype = t["dtype"]
    lo, hi = t["data_offsets"]
    with open(path, "rb") as f:
        f.seek(off + lo)
        buf = f.read(hi - lo)
    if len(buf) != hi - lo:
        raise SystemExit(f"{path}: {name} is truncated ({len(buf)} of {hi - lo} bytes)")
    if dtype == "F32":
        a = np.frombuffer(buf, dtype=np.float32)
    elif dtype == "F16":
        a = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
    elif dtype == "BF16":
        a = (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        raise SystemExit(f"{path}: {name} has dtype {dtype}, which this converter does not read "
                         "(F32, F16 and BF16 only)")
    return a.reshape(t["shape"]).astype(np.float32)


def tensors(hf_dir):
    """(name, path, header, data offset) for every tensor, sharded files included."""
    index = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        names = list(weight_map)
        shards = sorted(set(weight_map.values()))
    else:
        single = os.path.join(hf_dir, "model.safetensors")
        if not os.path.exists(single):
            raise SystemExit(f"{hf_dir}: no model.safetensors and no model.safetensors.index.json")
        header, _ = read_header(single)
        names = [n for n in header if n != "__metadata__"]
        shards = [os.path.basename(single)]
        weight_map = {n: shards[0] for n in names}
    loaded = {}
    for shard in shards:
        header, off = read_header(os.path.join(hf_dir, shard))
        loaded[shard] = (header, off)
    return [(n, os.path.join(hf_dir, weight_map[n]), *loaded[weight_map[n]]) for n in names]


def pack(args):
    os.makedirs(args.out, exist_ok=True)
    table, blobs, off = {}, [], 0
    count = 0
    for name, path, header, data_off in tensors(args.hf):
        w = read_tensor(path, header, data_off, name)
        kind = (len(w.shape) == 1 or name in KEEP_BF16 or w.shape[0] % ROWS or
                (len(w.shape) == 2 and w.shape[1] % COLS))
        if kind:
            flat = w.reshape(-1)
            table[name] = {"dtype": "BF16", "shape": [int(flat.size)]}
            data = bf16_bits(flat)
            how = "bf16"
        else:
            codes, scale, zero = quantize_q4_1(w)
            data = tiles(codes, scale, zero)
            table[name] = {"dtype": "I8", "shape": list(data.shape)}
            how = "q4_1"
        b = np.ascontiguousarray(data).tobytes()
        pad = (-off) % 64
        if pad:
            blobs.append(b"\0" * pad)
            off += pad
        table[name]["data_offsets"] = [off, off + len(b)]
        blobs.append(b)
        off += len(b)
        count += 1
        print(f"{name:52s} {str(list(w.shape)):14s} -> {table[name]['dtype']} {table[name]['shape']} {how}")
    table["__metadata__"] = {"format": "q4nx", "source": os.path.basename(args.hf.rstrip("/")),
                             "converter": "tools/safetensors_to_q4nx.py"}
    header = json.dumps(table, separators=(",", ":")).encode()
    header += b" " * ((-len(header)) % 64)
    with open(os.path.join(args.out, "model.q4nx"), "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for b in blobs:
            f.write(b)
    cfg_path = os.path.join(args.hf, "config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"{args.hf}: no config.json; the Q4NX directory needs one")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.setdefault("q4nx_source", {})
    cfg["q4nx_source"].update({"converter": "tools/safetensors_to_q4nx.py",
                               "safetensors": os.path.basename(args.hf.rstrip("/"))})
    cfg["q4nx_weight_layout"] = args.weight_layout
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    tok = args.tokenizer or args.hf
    copied = []
    for n in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(tok, n)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(args.out, n))
            copied.append(n)
    if "tokenizer.json" not in copied:
        print(f"WARNING: no tokenizer.json in {tok}: {args.out} has model.q4nx and config.json "
              "only, and 1bit forward-serve will refuse it. Pass --tokenizer <dir>.")
    print(f"wrote {args.out}/model.q4nx ({8 + len(header) + off} bytes, {count} tensors)")


def verify(args):
    """Every Q4NX tensor against the safetensors source, dequantized."""
    with open(os.path.join(args.out, "model.q4nx"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        table = json.loads(f.read(n))
        data = f.read()
    bad = 0
    for name, path, header, data_off in tensors(args.hf):
        if name not in table:
            print(f"MISSING {name}")
            bad += 1
            continue
        ent = table[name]
        lo, hi = ent["data_offsets"]
        w = read_tensor(path, header, data_off, name)
        if ent["dtype"] == "BF16":
            got = (np.frombuffer(data[lo:hi], dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            ref = w.reshape(-1)
            err = float(np.abs(got - ref).max()) if ref.size else 0.0
            bound = float(np.abs(ref).max()) / 256.0 if ref.size else 0.0  # bf16 has 8 mantissa bits
            ok = err <= max(bound, 1e-6)
            print(f"{'ok  ' if ok else 'BAD '} {name:52s} bf16 max err {err:.3g} (bound {bound:.3g})")
        else:
            N, K = w.shape
            got = untile(np.frombuffer(data[lo:hi], dtype=np.uint8).reshape(-1, 5120), N, K)
            err = float(np.abs(got - w).max())
            rms = float(np.sqrt(((got - w) ** 2).mean()) / (np.sqrt((w ** 2).mean()) + 1e-12))
            ok = rms < 0.15  # q4_1's own loss, the same bar gguf_to_q4nx's verify uses
            print(f"{'ok  ' if ok else 'BAD '} {name:52s} q4_1 max err {err:.4g}, rel rms {rms:.4f}")
        bad += 0 if ok else 1
    print(f"{len(table) - 1 - bad}/{len(table) - 1} tensors ok")
    return 1 if bad else 0


def decision(args):
    """A Decision-3.0 readout head: readout.safetensors + decision_config.json -> decision.q4nx."""
    cfg_path = os.path.join(args.hf, "decision_config.json")
    head_path = os.path.join(args.hf, "readout.safetensors")
    for p in (cfg_path, head_path):
        if not os.path.exists(p):
            raise SystemExit(f"{args.hf}: no {os.path.basename(p)}; a d3 pack has one")
    with open(cfg_path) as f:
        cfg = json.load(f)
    codes = cfg.get("codes")
    if not codes:
        raise SystemExit(f"{cfg_path}: no codes; the readout is scored per option code")
    header, off = read_header(head_path)
    names = [n for n in header if n != "__metadata__"]
    if len(names) != 1:
        raise SystemExit(f"{head_path}: expected one tensor, found {names}")
    w = read_tensor(head_path, header, off, names[0])
    if w.ndim != 2:
        raise SystemExit(f"{head_path}: the head must be a matrix, got shape {list(w.shape)}")
    if w.shape[0] != len(codes) and w.shape[1] != len(codes):
        raise SystemExit(f"{head_path}: shape {list(w.shape)} does not match {len(codes)} codes")
    # Keep the head in f32: it is a few hundred KB and it is the model's decision, not a weight
    # to be quantized.  Stored in the same container so one loader reads both.
    os.makedirs(args.out, exist_ok=True)
    table = {names[0]: {"dtype": "F32", "shape": list(w.shape), "data_offsets": [0, w.nbytes]},
             "__metadata__": {"format": "q4nx", "kind": "d3-readout",
                              "converter": "tools/safetensors_to_q4nx.py decision",
                              "format_id": cfg.get("format_id", "")}}
    hdr = json.dumps(table, separators=(",", ":")).encode()
    hdr += b" " * ((-len(hdr)) % 64)
    with open(os.path.join(args.out, "decision.q4nx"), "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        f.write(w.astype(np.float32).tobytes())
    shutil.copyfile(cfg_path, os.path.join(args.out, "decision_config.json"))
    print(f"wrote {args.out}/decision.q4nx ({8 + len(hdr) + w.nbytes} bytes, "
          f"{list(w.shape)} f32, {len(codes)} codes) and decision_config.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, extra in (("pack", pack, True), ("verify", verify, False), ("decision", decision, False)):
        p = sub.add_parser(name)
        p.add_argument("hf")
        p.add_argument("out")
        if extra:
            p.add_argument("--tokenizer")
            p.add_argument("--weight-layout", default="rm", choices=("rm", "b8"))
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    rc = args.fn(args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
