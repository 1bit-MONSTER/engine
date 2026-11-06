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
"""Import a 1BP package (`.1bp`) as a model-generic NPU container (docs/npu.md).

usage: tools/onebp_import.py <model.1bp> <out dir> --config <snapshot dir> [--embedding-from <gguf>]
       tools/onebp_import.py --verify <model.1bp> <out dir>

A 1BP package is one file: a 256-byte header, a tensor index, then the weights.  Its 4-bit payload is
**already this engine's tile layout** -- 32 x 256 tiles of 5120 B, `bf16 scale[8*32+r] @0`,
`bf16 zero[8*32+r] @512`, 4-bit codes @1024 (`npu/q4nx.h`) -- so the import copies tile bytes and
neither dequantizes nor re-quantizes them; `--verify` proves it by comparing every byte the container
holds against the package's own slice.  F16/F32 tensors (embeddings, norms, biases) become bf16, which
is the container's storage for everything the host reads.

Two things the format does not give you, both refused by name rather than papered over:

* **v1 packages carry no embedding and no output/lm_head.**  Measured: `bong-water-water-bong/ZR1-1.5B-1BP`
  (v1, 337 tensors) holds exactly 28 layers x 7 projections + 141 f32 norms/biases -- the 466 MB
  `token_embd` and the tied-or-not `lm_head` are simply absent, and `token_embd` first appears in **v4**
  (`ZAYA1-8B-1BP`, as f16).  A v1 package is therefore not a complete model: pass
  `--embedding-from <bf16|f16 gguf>` and the missing tensors come from there (dequantized to bf16, never
  requantized).  Without it the import stops instead of writing a container that cannot answer.
* **A tensor name with no Q4NX mapping is refused** (`no Q4NX name for blk.0.cca_conv_grp.weight`).
  Zaya1's grouped conv, EDA router and fused `ffn_gate_up_exps` are exactly that case: the container
  format and the per-op forward both lack those ops, so importing them would produce a directory that
  loads and then refuses (or worse).  The op list and its cost is `~/briefs/ZAYA-OPFAMILY-NOTE.md`;
  this tool does not guess a mapping for them.

Quant types: only `Q4NX` (0), `F16` (4) and `F32` (5) are read; I8/TQ1/TQ2/TQ2NZ/... are refused by
number, because this decoder implements q4_1/Q8/t2 and would otherwise serve noise as a good model.
Needs numpy; gguf only for --embedding-from.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import sys

import numpy as np

from gguf_to_q4nx import hf_name
from q4nx_pack import COLS, ROWS, TILE, bf16_bits

MAGIC = 0x00504231
Q4NX, I8, TQ1, TQ2, F16, F32 = 0, 1, 2, 3, 4, 5
QUANT_NAMES = {Q4NX: "Q4NX", I8: "I8", TQ1: "TQ1", TQ2: "TQ2", F16: "F16", F32: "F32"}
HEADER_MAX_VERSION = 4

# The header as the format defines it (include/onebp_format.h in the archived 1bit-MONSTER repo).
HEADER = [
    ("magic", "I"), ("version", "I"), ("arch", "I"), ("quant", "I"), ("scale_type", "I"),
    ("hidden_size", "i"), ("num_layers", "i"), ("num_attention_heads", "i"), ("num_kv_heads", "i"),
    ("head_dim", "i"), ("intermediate_size", "i"), ("vocab_size", "i"), ("max_seq_len", "i"),
    ("tile_rows", "I"), ("tile_cols", "I"), ("group_size", "I"),
    ("has_q_norm", "I"), ("has_k_norm", "I"), ("has_bias", "I"), ("rope_theta_f", "I"),
    ("bos_token_id", "I"), ("eos_token_id", "I"), ("tensor_count", "I"),
    ("num_experts", "I"), ("n_expert_used", "I"), ("n_ff_exp", "I"), ("n_ff_shexp", "I"),
    ("n_layer_dense_lead", "I"), ("sliding_window", "I"), ("swa_period", "I"),
    ("expert_gating_func", "I"), ("expert_weights_norm", "I"), ("expert_weights_scale_f", "I"),
    ("attn_gate_type", "I"), ("rope_freq_base_swa_f", "I"), ("n_rot_swa", "I"), ("n_rot_full", "I"),
]


def read_header(f):
    raw = f.read(256)
    if len(raw) < 160:
        raise SystemExit(f"not a 1BP package: {len(raw)} bytes in the header (256 expected)")
    h, off = {}, 0
    for name, fmt in HEADER:
        h[name] = struct.unpack_from("<" + fmt, raw, off)[0]
        off += struct.calcsize("<" + fmt)
    h["model_tag"] = raw[192:256].split(b"\0")[0].decode("ascii", "replace")
    if h["magic"] != MAGIC:
        raise SystemExit(f"not a 1BP package: magic {h['magic']:#010x}, expected {MAGIC:#010x} (\"1BP\\0\")")
    if not 1 <= h["version"] <= HEADER_MAX_VERSION:
        raise SystemExit(f"1BP version {h['version']}: this importer reads 1..{HEADER_MAX_VERSION}")
    if h["tile_rows"] != ROWS or h["tile_cols"] != COLS or h["group_size"] != 32:
        raise SystemExit(f"1BP tiles are {h['tile_rows']}x{h['tile_cols']}/group {h['group_size']}: this "
                         f"engine's chunks are {ROWS}x{COLS}/group 32, so the payload is not its layout")
    if h["quant"] != Q4NX:
        raise SystemExit(f"container quant {QUANT_NAMES.get(h['quant'], h['quant'])}: only Q4NX (0) tiles are "
                         f"this engine's q4_1 layout -- importing another quant would decode as noise")
    return h


def index(f, h):
    """[(name, ndim, dims, offset, nbytes, quant)], data offsets relative to the data section.

    v2+ appends a per-entry quant (mixed-quant files) *after* the offset/size pair; v1 has none and its
    quant is the header's.  Order matters: [name][ndim][dims][offset][bytes][quant], so the quant is the
    tail -- reading it as a prefix shifts every field after it (it looks plausible and is wrong)."""
    entries = []
    has_quant = h["version"] >= 2
    for i in range(h["tensor_count"]):
        head = f.read(4)
        if len(head) < 4:
            raise SystemExit(f"index entry {i}: truncated at offset {f.tell()}")
        (nl,) = struct.unpack("<I", head)
        if not 0 < nl <= 512:
            raise SystemExit(f"index entry {i}: name length {nl} at offset {f.tell() - 4} -- not a 1BP index")
        name = f.read(nl).decode()
        if f.read(1) != b"\0":
            raise SystemExit(f"index entry {i}: {name!r} is not NUL-terminated")
        (ndim,) = struct.unpack("<I", f.read(4))
        if ndim not in (1, 2, 3):
            raise SystemExit(f"index entry {i} ({name}): rank {ndim} is not 1, 2 or 3")
        dims = struct.unpack("<%dI" % ndim, f.read(4 * ndim))
        off, nbytes = struct.unpack("<QQ", f.read(16))
        quant = struct.unpack("<I", f.read(4))[0] if has_quant else h["quant"]
        entries.append((name, ndim, dims, off, nbytes, quant))
    return entries, f.tell()


def tiled(rows, cols):
    """(rows, cols) -> [tile rows, tile cols, 5120 B], rounded up as the writer's tiling does."""
    return [-(-rows // ROWS), -(-cols // COLS), TILE]


def classify(name, ndim, dims, nbytes, quant, version):
    """(container dtype, recorded shape, elements to convert) for one index entry.

    A v1 package has ONE quant for the whole file (the header's, Q4NX), yet stores norms and biases as
    raw f32 -- so v1's entry quant says nothing about the entry and the kind has to come from rank and
    size.  v2+ writes the per-entry quant, which is then trusted and size-checked."""
    elems = int(np.prod(dims))
    if version < 2 and quant == Q4NX:
        if ndim == 1 and nbytes == elems * 4:
            quant = F32
        elif ndim == 1 and nbytes == elems * 2:
            quant = F16
        elif ndim in (2, 3) and nbytes == int(np.prod(tiled(*dims[-2:]) if ndim == 2 else
                                                         [dims[0]] + tiled(dims[1], dims[2]))):
            quant = Q4NX
    if quant == Q4NX:
        if ndim == 2:
            shape, want = tiled(dims[0], dims[1]), nbytes
        elif ndim == 3:
            rt, kt, chunk = tiled(dims[1], dims[2])
            shape, want = [dims[0] * rt, kt, chunk], nbytes
        else:
            raise SystemExit(f"{name}: a Q4NX tile tensor has rank 2 (rows, cols) or 3 (experts, rows, cols), "
                             f"this one is rank {ndim}")
        if int(np.prod(shape)) != want:
            raise SystemExit(f"{name}: {want} bytes is not {shape} of {dims} -- the package's own tiling says "
                             f"{int(np.prod(shape))}")
        return "tiles", shape, None
    if quant in (F16, F32):
        if nbytes == 0:
            raise SystemExit(f"{name}: alias entry (bytes 0) of a raw tensor -- the format says aliases point "
                             f"at tiles; this package is not one this importer understands")
        elems = int(np.prod(dims))
        width = 2 if quant == F16 else 4
        if elems * width != nbytes:
            raise SystemExit(f"{name}: {nbytes} bytes for {dims} of {QUANT_NAMES[quant]} ({elems * width} "
                             f"expected)")
        return "bf16", [elems], (quant, elems)
    raise SystemExit(f"{name}: quant {QUANT_NAMES.get(quant, quant)} ({quant}) has no reader in this engine "
                     f"(q4_1 tiles, Q8, t2 only)")


def copy_stream(src, dst, n):
    """Copy n bytes without materializing them: a 1 GB f16 embedding is not a reason to OOM."""
    while n:
        b = src.read(min(n, 1 << 24))
        if not b:
            raise SystemExit("short read: the package is truncated")
        dst.write(b)
        n -= len(b)


def bf16_stream(src, dst, quant, elems):
    """F16/F32 -> bf16, chunked, nearest-even."""
    step = 1 << 23
    width = 2 if quant == F16 else 4
    done = 0
    while done < elems:
        k = min(step, elems - done)
        raw = src.read(k * width)
        if len(raw) < k * width:
            raise SystemExit("short read: the package is truncated")
        v = np.frombuffer(raw, dtype=np.float16 if quant == F16 else np.float32).astype(np.float32)
        dst.write(bf16_bits(v).tobytes())
        done += k


def from_embedding_gguf(path, wanted):
    """token_embd/output from a GGUF, as bf16 -- exactly, when the GGUF stores them F16/BF16/F32."""
    import gguf
    from gguf.quants import dequantize
    out = {}
    r = gguf.GGUFReader(path)
    for t in r.tensors:
        if t.name not in wanted:
            continue
        w = dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)
        out[hf_name(t.name)] = (t.tensor_type.name, bf16_bits(w).tobytes(), [int(w.size)])
    return out


def config(snapshot, h, container_bytes):
    """`--config` is a snapshot DIRECTORY (its tokenizer files are copied too), so read <dir>/config.json."""
    cfg = json.load(open(os.path.join(snapshot, "config.json")))
    for key, mine in (("hidden_size", h["hidden_size"]), ("num_hidden_layers", h["num_layers"]),
                      ("num_attention_heads", h["num_attention_heads"]),
                      ("num_key_value_heads", h["num_kv_heads"]), ("intermediate_size", h["intermediate_size"]),
                      ("vocab_size", h["vocab_size"])):
        if cfg.get(key) != mine:
            raise SystemExit(f"config.json says {key}={cfg.get(key)}, the package's header says {mine}: "
                             f"these are not the same model")
    if not cfg.get("head_dim"):
        cfg["head_dim"] = h["head_dim"] or cfg["hidden_size"] // cfg["num_attention_heads"]
    cfg["torch_dtype"] = "bfloat16"
    cfg["q4nx_weight_layout"] = "rm"
    cfg["q4nx_source"] = {"onebp": container_bytes["file"], "onebp_version": h["version"],
                          "onebp_sha256": container_bytes["sha256"],
                          "converter": "tools/onebp_import.py"}
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("onebp")
    p.add_argument("out_dir", nargs="?")
    p.add_argument("--config", help="a Hugging Face snapshot directory (config.json, tokenizer files) for "
                                    "this same model -- the package carries dims, not a full config")
    p.add_argument("--embedding-from", help="GGUF to take a missing token_embd.weight/lm_head from "
                                            "(v1 packages carry neither); bf16/f16 source recommended")
    p.add_argument("--verify", action="store_true",
                   help="check the container on disk against the package, byte for byte, and exit")
    args = p.parse_args(argv)
    if not args.verify and (not args.out_dir or not args.config):
        p.error("<out dir> and --config are required (or --verify)")

    sha = hashlib.sha256()
    with open(args.onebp, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            sha.update(b)
    fsize = os.path.getsize(args.onebp)

    with open(args.onebp, "rb") as f:
        h = read_header(f)
        entries, data_start = index(f, h)
        data_span = fsize - data_start
        last = max(e[3] + e[4] for e in entries)
        if last != data_span:
            raise SystemExit(f"index ends {last} bytes into a {data_span}-byte data section: the package's "
                             f"index and payload disagree")
        print(f"1BP v{h['version']}: {len(entries)} tensors, H={h['hidden_size']} L={h['num_layers']} "
              f"NH={h['num_attention_heads']} NKV={h['num_kv_heads']} HD={h['head_dim']} "
              f"IM={h['intermediate_size']} V={h['vocab_size']}")

        plan = []          # (container name, dtype, shape, source offset, nbytes, quant, elems)
        have = set()
        for name, ndim, dims, off, nbytes, quant in entries:
            if nbytes == 0:
                raise SystemExit(f"{name}: v4 alias entries are not read by this importer yet (the container "
                                 f"would need the aliased tensor's bytes duplicated); re-export without aliases")
            cname = hf_name(name)
            dtype, shape, conv = classify(name, ndim, dims, nbytes, quant, h["version"])
            plan.append((cname, dtype, shape, off, nbytes, quant, conv))
            have.add(cname)

        missing = [n for n in ("model.embed_tokens.weight", "lm_head.weight") if n not in have]
        extra = {}
        if missing:
            if not args.embedding_from:
                raise SystemExit(
                    f"this package has no {', '.join(missing)} (v1 packages carry neither the embedding nor "
                    f"the head -- measured on ZR1-1.5B-1BP; they first appear as f16 in v4). Pass "
                    f"--embedding-from <a bf16/f16 GGUF of the same model> to supply them, or import a v4 "
                    f"package")
            gguf_names = {"model.embed_tokens.weight": "token_embd.weight",
                          "lm_head.weight": "output.weight"}
            want = {gguf_names[m]: m for m in missing}
            got = from_embedding_gguf(args.embedding_from, set(want))
            for gname, cname in want.items():
                if cname not in got:
                    raise SystemExit(f"--embedding-from {args.embedding_from}: no {gname} tensor, so "
                                     f"{cname} would still be missing")
                kind, blob, shape = got[cname]
                print(f"  {cname:44s} {kind:5s} from {os.path.basename(args.embedding_from)} -> bf16 "
                      f"{'x'.join(str(d) for d in shape)}")
                extra[cname] = (blob, shape)

        if not args.verify:
            os.makedirs(args.out_dir, exist_ok=True)

        # Container layout: 8-byte header length, JSON tensor table, data at 64-byte-aligned offsets
        # (same shape gguf_to_q4nx writes, so every reader of a q4nx directory is unchanged).
        table, off = {}, 0
        for cname, dtype, shape, _o, nbytes, _q, conv in plan:
            off += (-off) % 64
            size = int(np.prod(shape)) * (2 if dtype == "bf16" else 1)
            table[cname] = {"dtype": "BF16" if dtype == "bf16" else "I8", "shape": shape,
                            "data_offsets": [off, off + size]}
            off += size
        for cname, (blob, shape) in extra.items():
            off += (-off) % 64
            table[cname] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + len(blob)]}
            off += len(blob)
        cfg = config(args.config, h, {"file": os.path.basename(args.onebp), "sha256": sha.hexdigest()}) \
            if not args.verify else None
        table["__metadata__"] = {"format": "q4nx", "source": os.path.basename(args.onebp),
                                 "converter": "tools/onebp_import.py",
                                 "onebp_sha256": sha.hexdigest()}
        header = json.dumps(table, separators=(",", ":")).encode()
        header += b" " * ((-len(header)) % 64)

        if args.verify:
            return verify(args, h, plan, extra, table, header, data_start)

        path = os.path.join(args.out_dir, "model.q4nx")
        print(f"writing {path}")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            f.truncate(8 + len(header) + off)
            base = 8 + len(header)
            with open(args.onebp, "rb") as src:
                for cname, dtype, shape, src_off, nbytes, quant, conv in plan:
                    lo, hi = table[cname]["data_offsets"]
                    f.seek(base + lo)
                    src.seek(data_start + src_off)
                    if conv is None:
                        copy_stream(src, f, nbytes)
                    else:
                        bf16_stream(src, f, conv[0], conv[1])
                    if f.tell() != base + hi:
                        raise SystemExit(f"{cname}: wrote to {f.tell()}, the table says {base + hi}")
                    print(f"  {cname:44s} {'verbatim q4_1 tiles' if conv is None else 'f16/f32 -> bf16':24s} "
                          f"{dtype} {'x'.join(str(d) for d in shape)}")
            for cname, (blob, shape) in extra.items():
                f.seek(base + table[cname]["data_offsets"][0])
                f.write(blob)
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        copied = []
        for n in ("tokenizer.json", "tokenizer_config.json"):
            src = os.path.join(args.config, n)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(args.out_dir, n))
                copied.append(n)
        if "tokenizer.json" not in copied:
            print(f"WARNING: no tokenizer.json in {args.config}: {args.out_dir} cannot serve until one is "
                  f"there (npu/tokenizer.cpp reads it)", file=sys.stderr)
        print(f"wrote {args.out_dir}/model.q4nx ({os.path.getsize(path)} bytes), config.json, "
              f"{len(copied)} tokenizer file(s); next: tools/npu_plan.py {args.out_dir}")
        return 0


def verify(args, h, plan, extra, table, header, data_start):
    """Every byte the container holds must be the bytes this import would write again.

    It walks the *classified* plan, not the package index: in a v1 package every entry claims the
    header's Q4NX quant while the norms and biases are raw f32 (that is what `classify()` resolves), so
    a verifier reading the index quant compares f32 source bytes against bf16 container bytes and
    reports 141 of 337 tensors "differing" on a container it just wrote itself.
    """
    path = os.path.join(args.out_dir, "model.q4nx")
    bad, checked = 0, 0
    with open(path, "rb") as cf, open(args.onebp, "rb") as sf:
        (hlen,) = struct.unpack("<Q", cf.read(8))
        if hlen != len(header):
            raise SystemExit(f"{path}: header is {hlen} bytes on disk, {len(header)} from this package")
        disk = json.loads(cf.read(hlen))
        if disk != table:
            raise SystemExit(f"{path}: the tensor table is not what this package generates (re-import)")
        base = 8 + hlen
        for cname, dtype, _shape, src_off, nbytes, _quant, conv in plan:
            lo, _hi = disk[cname]["data_offsets"]
            cf.seek(base + lo)
            sf.seek(data_start + src_off)
            if conv is None:                       # q4_1 tiles: copied verbatim, compare them verbatim
                left, same = nbytes, True
                while left:
                    k = min(left, 1 << 24)
                    if cf.read(k) != sf.read(k):
                        same = False
                        break
                    left -= k
            else:                                  # f16/f32 -> bf16: compare against the same conversion
                quant, elems = conv
                width = 2 if quant == F16 else 4
                v = np.frombuffer(sf.read(elems * width), dtype=np.float16 if quant == F16 else np.float32)
                same = cf.read(len(bf16_bits(v.astype(np.float32)).tobytes())) == \
                    bf16_bits(v.astype(np.float32)).tobytes()
            checked += 1
            if not same:
                bad += 1
                print(f"BAD  {cname}: container bytes are not this package's")
        for cname, (blob, _shape) in extra.items():   # supplied from --embedding-from, so checked here
            cf.seek(base + disk[cname]["data_offsets"][0])
            checked += 1
            if cf.read(len(blob)) != blob:
                bad += 1
                print(f"BAD  {cname}: does not match the GGUF it was taken from")
    print(f"{checked} tensors compared byte for byte: {'all identical' if not bad else f'{bad} differ'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
