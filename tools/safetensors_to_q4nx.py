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

usage: tools/safetensors_to_q4nx.py pack   <hf dir> <out dir> [--tokenizer <dir>] [--dtype auto|bf16]
       tools/safetensors_to_q4nx.py verify <hf dir> <out dir>
       tools/safetensors_to_q4nx.py decision <hf dir> <out dir>

`pack` is the safetensors counterpart of tools/gguf_to_q4nx.py and shares its packers
(tools/q4nx_pack.py), so both write the same container: 8-byte header length, JSON tensor
table, data.  Every 2-D tensor except the embedding becomes q4_1 tiles (I8 [tiles, 5120]) and
norms, biases and the embedding stay bf16.  config.json is the model's own HF config with the
two Q4NX fields added, and the tokenizer files are copied.

`--dtype bf16` writes no tiles at all: every tensor keeps the source dtype (bf16, or the F32
the transforms force).  That container is for measuring, not for serving -- quantization is
what `--dtype auto` is for -- because it lets a run hold the weights fixed and ask whether the
engine's forward and a reference agree (engine #796's parity row), and it makes the cost of
q4_1 a separate, measurable delta instead of an unexplained one.

Two name conventions exist and `--arch` picks one:

* `hf` (default for a `model.layers.*` pack, e.g. Qwen3-0.6B): the HF tensor names ARE the Q4NX
  names -- gguf_to_q4nx maps GGUF names onto these -- so nothing is translated.
* `qwen3_5` (the d3 packs, engine #796): the language model is nested under `language_model.*`
  and the GDN tensors carry qwen3_5's own names, so they are renamed to the names
  Q4nxNpuForward loads (configs/q4nx/qwen3.5_0.8b.json's `q4nx_name` column is the authority)
  and three transforms are applied.  Each of the three loads and computes garbage if it is
  wrong, so each is named here and covered by a known-answer test in
  tests/safetensors_to_q4nx_test.py:

  1. `linear_attn.A_log` -> `linear_attn.ssm_a`, written **F32 as -exp(A_log)**.  The engine
     computes `expf(ssm_a[h] * softplus(dt))` (q4nx_forward.cpp:1844), so `ssm_a` is the decay
     log-rate and must be negative; the raw A_log is not.
  2. `linear_attn.conv1d.weight` `[CD, 1, CK]` -> `linear_attn.ssm_conv1d.weight`, **transposed**
     to `[CK, CD]` flattened: the engine indexes `conv[j*CD + c]` (q4nx_forward.cpp:1802), i.e.
     tap-major, while HF is channel-major.
  3. `linear_attn.in_proj_a/b.weight` `[NVH, H]` -> `ssm_alpha_proj`/`ssm_beta_proj` with the
     rows **zero-padded to 32** so they tile as q4_1; the engine dequantises at the stored
     height and keeps the first NVH rows (`deq_pad`, q4nx_forward.cpp:1640).  The official FLM
     config asks for Q8_0 here; q4_1 is what this packer can write, and the loss is measured in
     `verify` rather than assumed.

  `visual.*` is skipped (the forward runs the text path) and the count is printed.  A
  `language_model.*` tensor with no mapping is a refusal, never a silent drop.

`decision` converts a Decision-3.0 pack's readout head -- `readout.safetensors` plus
`decision_config.json` -- into `decision.q4nx` in the same container, so one loader reads both.
The head is one F32 matrix [codes, hidden]: the per-option logits for a pooled backbone state.

This exists because the d3 packs ship HF safetensors and the engine only had a GGUF converter
(engine #653); nothing here needs a device, a model download or gguf-py.
"""
import argparse
import json
import os
import re
import shutil
import struct
import sys

import numpy as np

from q4nx_pack import COLS, ROWS, bf16_bits, quantize_q4_1, tiles, untile

# bf16 storage: 1-D tensors (norms, biases, rope tables) and the token embedding, as in the
# GGUF converter, plus any tensor whose dimensions are not a whole number of 32x256 tiles.
KEEP_BF16 = ("model.embed_tokens.weight",)

# qwen3_5 (d3) tensor suffixes that only need the `language_model.` -> `model.` rename.
QWEN35_PLAIN = (
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
)

# qwen3_5 GDN names, from configs/q4nx/qwen3.5_0.8b.json's name_map (q4nx_name column), which
# is what Q4nxNpuForward loads.  in_proj_z is the GDN output gate and goes to the SAME name a
# full-attention layer's output gate uses, because that is the tensor the forward reads.
QWEN35_GDN = {
    "linear_attn.in_proj_qkv.weight": "linear_attn.qkv_proj.weight",
    "linear_attn.in_proj_z.weight": "self_attn.gate_proj.weight",
    "linear_attn.in_proj_a.weight": "linear_attn.ssm_alpha_proj.weight",
    "linear_attn.in_proj_b.weight": "linear_attn.ssm_beta_proj.weight",
    "linear_attn.out_proj.weight": "linear_attn.ssm_out_proj.weight",
    "linear_attn.conv1d.weight": "linear_attn.ssm_conv1d.weight",
    "linear_attn.norm.weight": "linear_attn.ssm_norm.weight",
    "linear_attn.A_log": "linear_attn.ssm_a",
    "linear_attn.dt_bias": "linear_attn.ssm_dt.bias",
}


def qwen35_rename(name):
    """A d3 pack's HF tensor name -> the name Q4nxNpuForward loads (engine #796)."""
    if name.startswith("visual."):
        return None                      # the text path: the forward never reads these
    if not name.startswith("language_model."):
        raise SystemExit(f"{name}: not a qwen3_5 tensor; expected language_model.* or visual.*")
    rest = name[len("language_model."):]
    if rest in ("embed_tokens.weight", "norm.weight"):
        return "model." + rest
    m = re.match(r"layers\.(\d+)\.(.+)$", rest)
    if not m:
        raise SystemExit(f"{name}: no mapping for this tensor (a d3 pack has layers.N.* only)")
    bid, suffix = m.group(1), m.group(2)
    if suffix in QWEN35_PLAIN:
        return f"model.layers.{bid}.{suffix}"
    if suffix not in QWEN35_GDN:
        raise SystemExit(f"{name}: no qwen3_5 mapping for "
                         f"{suffix!r}; refusing rather than writing a tensor nothing loads")
    return f"model.layers.{bid}.{QWEN35_GDN[suffix]}"


def qwen35_transform(name, w):
    """The three transforms, keyed on the renamed tensor.  Returns (array, forced dtype)."""
    if name.endswith(".ssm_a"):
        return -np.exp(np.asarray(w, dtype=np.float32)), "F32"
    if name.endswith(".ssm_dt.bias"):
        return np.asarray(w, dtype=np.float32), "F32"
    if name.endswith(".ssm_conv1d.weight"):
        if w.ndim != 3 or w.shape[1] != 1:
            raise SystemExit(f"{name}: conv1d is expected as [CD, 1, CK], got {list(w.shape)}")
        return np.ascontiguousarray(w.reshape(w.shape[0], w.shape[2]).T).reshape(-1), None
    if name.endswith("ssm_alpha_proj.weight") or name.endswith("ssm_beta_proj.weight"):
        prows = -(-w.shape[0] // ROWS) * ROWS
        if prows == w.shape[0]:
            return w, None
        out = np.zeros((prows, w.shape[1]), dtype=np.float32)
        out[:w.shape[0]] = w
        return out, None
    return w, None


def transcode(name, w, arch):
    """(q4nx name, transformed array, forced dtype), or None when the arch skips the tensor."""
    if arch != "qwen3_5":
        return name, w, None
    q = qwen35_rename(name)
    if q is None:
        return None
    out, dtype = qwen35_transform(q, w)
    return q, out, dtype


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


def detect_arch(names):
    """`qwen3_5` for a pack whose language model is nested under language_model.*, else `hf`."""
    return "qwen3_5" if any(n.startswith("language_model.") for n in names) else "hf"


def pack(args):
    os.makedirs(args.out, exist_ok=True)
    table, blobs, off = {}, [], 0
    count = 0
    entries = tensors(args.hf)
    arch = detect_arch([n for n, *_ in entries]) if args.arch == "auto" else args.arch
    print(f"arch {arch}")
    skipped = 0
    for name, path, header, data_off in entries:
        if arch == "qwen3_5" and name.startswith("visual."):
            skipped += 1                    # decided from the name, before reading its bytes
            continue
        w = read_tensor(path, header, data_off, name)
        got = transcode(name, w, arch)
        if got is None:
            skipped += 1
            continue
        name, w, forced = got
        kind = (forced != "F32" and
                (args.dtype == "bf16" or len(w.shape) == 1 or name in KEEP_BF16 or
                 w.shape[0] % ROWS or
                 (len(w.shape) == 2 and w.shape[1] % COLS)))
        if forced == "F32":
            flat = w.reshape(-1)
            table[name] = {"dtype": "F32", "shape": [int(flat.size)]}
            data = np.ascontiguousarray(flat, dtype=np.float32).tobytes()
            how = "f32"
        elif kind:
            flat = w.reshape(-1)
            # The bf16 tensors keep the shape the source gave them: model_load reads the
            # embedding's vocab and hidden off shape[0]/shape[1] (model.c:890), so a flattened
            # [vocab*hidden] makes it derive vocab=vocab*hidden, hidden=0.
            table[name] = {"dtype": "BF16", "shape": [int(d) for d in w.shape]}
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
    if skipped:
        print(f"skipped {skipped} tensor(s) ({'visual.*, the text path' if arch == 'qwen3_5' else 'unpaired'})")
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
    if arch == "qwen3_5":
        cfg["q4nx_source"]["arch"] = arch
        # The engine reads config.json at depth 1 only (q4nx_semantics.c's value_for_top), so a pack
        # that nests its language model under text_config loads with hidden_size=0 and a default
        # layer count.  Lift those keys to the top; text_config stays for anyone reading the pack.
        for k, v in cfg.get("text_config", {}).items():
            cfg.setdefault(k, v)
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
    entries = tensors(args.hf)
    arch = detect_arch([n for n, *_ in entries]) if args.arch == "auto" else args.arch
    for name, path, header, data_off in entries:
        w = read_tensor(path, header, data_off, name)
        got = transcode(name, w, arch)
        if got is None:
            continue
        name, w, _forced = got
        if name not in table:
            print(f"MISSING {name}")
            bad += 1
            continue
        ent = table[name]
        lo, hi = ent["data_offsets"]
        if ent["dtype"] == "F32":
            f32 = np.frombuffer(data[lo:hi], dtype=np.float32)
            err = float(np.abs(f32 - w.reshape(-1)).max())
            ok = err == 0.0                # exact by construction: the transform is f32 maths
            print(f"{'ok  ' if ok else 'BAD '} {name:52s} f32 max err {err:.3g}")
        elif ent["dtype"] == "BF16":
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
        if name != "decision":
            p.add_argument("--arch", default="auto", choices=("auto", "hf", "qwen3_5"),
                           help="hf: HF names are Q4NX names; qwen3_5: the d3 pack's renames and "
                                "transforms; auto detects a language_model.* pack")
        if extra:
            p.add_argument("--tokenizer")
            p.add_argument("--dtype", default="auto", choices=("auto", "bf16"),
                           help="auto: q4_1 tiles where the shape allows (serving); bf16: no "
                                "quantization, every tensor keeps its source dtype (measuring)")
            p.add_argument("--weight-layout", default="rm", choices=("rm", "b8"))
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    rc = args.fn(args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
