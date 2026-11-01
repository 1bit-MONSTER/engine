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
"""Repack a GGUF into a Q4NX model directory the NPU routes read (docs/npu.md).

usage: tools/gguf_to_q4nx.py pack   <model.gguf> <out dir> [--tokenizer <dir>]
       tools/gguf_to_q4nx.py verify <model.gguf> <out dir>

pack writes <out dir>/model.q4nx (8-byte header length, JSON tensor table, data, as in
safetensors), config.json from the GGUF metadata, and copies tokenizer.json and
tokenizer_config.json from --tokenizer (the GGUF's own vocabulary is not converted).

Every projection and lm_head becomes q4_1 tiles (I8 [tiles, 5120], npu/q4nx.h): 32 rows x
256 columns, a bf16 scale and zero per row and 32-column group, 4-bit codes. How each GGUF
type gets there:

  Q4_0   w = d*(q-8)             scale d, zero -8d; codes unchanged
  Q4_1   w = d*q + m             scale d, zero m;   codes unchanged
  Q4_K   w = d*sc*q - dmin*mn    scale d*sc, zero -dmin*mn per 32-column sub-block; codes
                                 unchanged (a Q4_K super-block is one tile row's 256 columns)
  other  (Q5_K, Q6_K, Q8_0, F16, BF16, F32) dequantized, then quantized to q4_1 per group
         (min/max): lossy

The first three are exact apart from rounding scale and zero to bf16 (nearest even);
verify checks every tensor against that bound and reports the error of the lossy ones.
Norms, the embedding and the router are stored as bf16. A tied lm_head (no output.weight)
is made from token_embd.

MoE (qwen3moe): the routed expert stacks (ffn_{gate,up,down}_exps, three-dimensional in the
GGUF as [cols, rows, experts]) become ONE tensor per layer and projection, the experts
stacked along the tile grid: [experts x rows/32, cols/256, 5120] tiles. Expert e is tile
rows [e*rows/32, (e+1)*rows/32), so its chunks are contiguous — the span a decode dispatch
points a kernel at — and __metadata__.moe records that decomposition with every expert's
byte offset, i.e. the list a per-dispatch chunk-address kernel takes. The router
(ffn_gate_inp) stays bf16 and untiled: it runs on the host and its top-k must be the
GGUF's own. A shared expert (ffn_*_shexp, ffn_gate_inp_shexp) is packed like any dense
projection; Qwen3-30B-A3B has none.

Supported architectures: qwen3, qwen3moe and qwen2. Needs numpy and gguf (gguf-py).
"""
import argparse
import json
import os
import shutil
import struct
import sys

import numpy as np
import gguf
from gguf.quants import dequantize

from q4nx_pack import (COLS, GROUP, ROWS, TILE, bf16_bits, f16, from_bf16,  # noqa: F401
                       quantize_q4_1, tiles, untile)


# --- GGUF blocks -> (codes [N, K] u8, scale [N, K/32] f32, zero [N, K/32] f32), exact ---

def split_q4_0(raw, N, K):
    b = raw.reshape(N, K // 32, 18)
    d = f16(b[..., 0:2])[..., 0]
    qs = b[..., 2:18]
    codes = np.concatenate([qs & 0xF, qs >> 4], axis=-1).reshape(N, K)
    return codes, d, -8.0 * d


def split_q4_1(raw, N, K):
    b = raw.reshape(N, K // 32, 20)
    d = f16(b[..., 0:2])[..., 0]
    m = f16(b[..., 2:4])[..., 0]
    qs = b[..., 4:20]
    codes = np.concatenate([qs & 0xF, qs >> 4], axis=-1).reshape(N, K)
    return codes, d, m


def split_q4_k(raw, N, K):
    b = raw.reshape(N, K // 256, 144)
    d = f16(b[..., 0:2])[..., 0][..., None]
    dmin = f16(b[..., 2:4])[..., 0][..., None]
    s = b[..., 4:16].astype(np.uint16)
    sc = np.empty(s.shape[:-1] + (8,), np.uint16)
    mn = np.empty_like(sc)
    sc[..., 0:4] = s[..., 0:4] & 63
    mn[..., 0:4] = s[..., 4:8] & 63
    sc[..., 4:8] = (s[..., 8:12] & 0xF) | ((s[..., 0:4] >> 6) << 4)
    mn[..., 4:8] = (s[..., 8:12] >> 4) | ((s[..., 4:8] >> 6) << 4)
    qs = b[..., 16:144].reshape(N, K // 256, 4, 32)  # 4 chunks of 64 values
    codes = np.stack([qs & 0xF, qs >> 4], axis=3).reshape(N, K)  # low nibbles, then high
    scale = (d * sc.astype(np.float32)).reshape(N, K // 32)  # exact in f32 (11 x 6 bits)
    zero = (-dmin * mn.astype(np.float32)).reshape(N, K // 32)
    return codes, scale, zero


EXACT = {
    gguf.GGMLQuantizationType.Q4_0: split_q4_0,
    gguf.GGMLQuantizationType.Q4_1: split_q4_1,
    gguf.GGMLQuantizationType.Q4_K: split_q4_k,
}


def to_q4_1(t):
    """A GGUF tensor as (codes, scale, zero, exact); an expert stack as expert-major rows."""
    K, N = int(t.shape[0]), int(t.shape[1])  # ggml order: ne0 = columns
    if len(t.shape) == 3:
        N *= int(t.shape[2])  # expert e is rows [e*N, (e+1)*N) of the ggml stream
    if N % ROWS or K % COLS:
        raise SystemExit(f"{t.name}: {N} x {K} is not a whole number of 32 x 256 tiles")
    split = EXACT.get(t.tensor_type)
    if split:
        return (*split(np.asarray(t.data).reshape(-1), N, K), True)
    return (*quantize_q4_1(dequantize(t.data, t.tensor_type).reshape(N, K).astype(np.float32)), False)


# --- names and config ---

LAYER = {
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
    "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
}
TOP = {"token_embd": "model.embed_tokens", "output_norm": "model.norm", "output": "lm_head"}

# MoE block parts (Qwen3MoE). The router is a plain linear on the hidden state; _exps are the
# routed expert stacks; _shexp is a shared expert (some qwen3moe models have one, Qwen3-30B-A3B
# does not).
MOE = {
    "ffn_gate_inp": "mlp.gate", "ffn_gate_inp_shexp": "mlp.shared_expert_gate",
    "ffn_gate_exps": "mlp.experts.gate_proj", "ffn_up_exps": "mlp.experts.up_proj",
    "ffn_down_exps": "mlp.experts.down_proj",
    "ffn_gate_shexp": "mlp.shared_expert.gate_proj",
    "ffn_up_shexp": "mlp.shared_expert.up_proj",
    "ffn_down_shexp": "mlp.shared_expert.down_proj",
}
LAYER.update(MOE)
ROUTER = {"ffn_gate_inp", "ffn_gate_inp_shexp"}


def hf_name(name):
    base, _, suffix = name.rpartition(".")
    if base.startswith("blk."):
        _, L, part = base.split(".", 2)
        if part not in LAYER:
            raise SystemExit(f"no Q4NX name for {name}")
        return f"model.layers.{L}.{LAYER[part]}.{suffix}"
    if base not in TOP:
        raise SystemExit(f"no Q4NX name for {name}")
    return f"{TOP[base]}.{suffix}"


def field(r, key, default=None):
    f = r.fields.get(key)
    return default if f is None else f.contents()


def config(r, vocab):
    arch = field(r, "general.architecture")
    if arch not in ("qwen3", "qwen2", "qwen3moe"):
        raise SystemExit(f"architecture {arch!r}: only qwen3, qwen2 and qwen3moe are supported")
    a = lambda k, d=None: field(r, f"{arch}.{k}", d)  # noqa: E731
    heads = a("attention.head_count")
    tied = not any(t.name == "output.weight" for t in r.tensors)
    moe = arch == "qwen3moe"
    # A MoE layer's FFN is the routed expert; the dense feed_forward_length is unused.
    ff = a("expert_feed_forward_length") if moe else a("feed_forward_length")
    c = {
        "architectures": [{"qwen3": "Qwen3ForCausalLM", "qwen2": "Qwen2ForCausalLM",
                           "qwen3moe": "Qwen3MoeForCausalLM"}[arch]],
        "model_type": arch,
        "hidden_size": a("embedding_length"),
        "intermediate_size": ff,
        "num_hidden_layers": a("block_count"),
        "num_attention_heads": heads,
        "num_key_value_heads": a("attention.head_count_kv", heads),
        "head_dim": a("attention.key_length", a("embedding_length") // heads),
        "max_position_embeddings": a("context_length"),
        "rms_norm_eps": float(f'{a("attention.layer_norm_rms_epsilon"):.7g}'),
        "rope_theta": a("rope.freq_base"),
        "vocab_size": vocab,
        "bos_token_id": field(r, "tokenizer.ggml.bos_token_id"),
        "eos_token_id": field(r, "tokenizer.ggml.eos_token_id"),
        "tie_word_embeddings": tied,
        "torch_dtype": "bfloat16",
        "hidden_act": "silu",
        "q4nx_source": {"converter": "tools/gguf_to_q4nx.py"},
    }
    if arch == "qwen2":
        # qwen2 differs from qwen3 in exactly two attention ways: no per-head q/k
        # RMSNorm, and q/k/v projection bias.  Both are declared here so a consumer
        # does not have to sniff the tensor table.
        c["attention_bias"] = any(t.name.endswith(("attn_q.bias", "attn_k.bias", "attn_v.bias"))
                                  for t in r.tensors)
        c["qk_norm"] = False
    if moe:
        shared = a("expert_shared_feed_forward_length", 0)
        c.update({
            "moe_intermediate_size": ff,
            "num_experts": a("expert_count"),
            "num_experts_per_tok": a("expert_used_count"),
            "norm_topk_prob": True,  # Qwen3MoE renormalises the top-k weights (llama.cpp too)
        })
        if shared:
            c["shared_expert_intermediate_size"] = shared
    return {k: v for k, v in c.items() if v is not None}


def planned(r):
    """(q4nx name, gguf tensor, kind), kind 'tiles', 'expert-tiles' or 'bf16'.

    The router stays bf16 (it runs on the host and its top-k must be the GGUF's own), as do the
    norms and the embedding. A 3-D blk.N.ffn_*_exps tensor is a routed expert stack.
    """
    out = []
    names = {t.name for t in r.tensors}
    for t in r.tensors:
        n = hf_name(t.name)
        if len(t.shape) == 1 or t.name == "token_embd.weight" or part_of(t.name) in ROUTER:
            kind = "bf16"
        else:
            kind = "expert-tiles" if len(t.shape) == 3 else "tiles"
        out.append((n, t, kind))
    if "output.weight" not in names:
        emb = next(t for t in r.tensors if t.name == "token_embd.weight")
        out.append(("lm_head.weight", emb, "tiles"))
    return out


def part_of(name):
    """The ggml part of a blk.N.<part>.<suffix> name, else ''."""
    base = name.rpartition(".")[0]
    return base.split(".", 2)[2] if base.startswith("blk.") else ""


def grid(rows, cols):
    """A rows x cols tensor's tile grid, after the 32 x 256 divisibility check."""
    if rows % ROWS or cols % COLS:
        raise SystemExit(f"{rows} x {cols} is not a whole number of 32 x 256 tiles")
    return rows // ROWS, cols // COLS


def blob_shape(t, kind):
    """The shape the container records for this tensor's data."""
    if kind == "bf16":
        return [int(t.shape[1]), int(t.shape[0])] if len(t.shape) == 2 else [int(np.prod(t.shape))]
    K, N = int(t.shape[0]), int(t.shape[1])
    rows, cols = (N * int(t.shape[2]), K) if kind == "expert-tiles" else (N, K)
    rt, kt = grid(rows, cols)
    return [rt, kt, TILE]


def blob_bytes(t, kind):
    return int(np.prod(blob_shape(t, kind))) * (2 if kind == "bf16" else 1)


def encode(t, kind):
    """One tensor's container bytes, and whether the conversion was exact."""
    if kind == "bf16":
        w = dequantize(t.data, t.tensor_type).astype(np.float32)
        return np.ascontiguousarray(bf16_bits(w if w.ndim == 2 else w.reshape(-1))).tobytes(), True
    codes, scale, zero, exact = to_q4_1(t)
    data = tiles(codes, scale, zero)
    if kind == "expert-tiles":
        rt, kt = grid(int(t.shape[1]), int(t.shape[0]))
        data = data.reshape(int(t.shape[2]) * rt, kt, TILE)
    return np.ascontiguousarray(data).tobytes(), exact


def moe_meta(r, items, table):
    """__metadata__.moe: the expert decomposition and every expert's byte offset."""
    arch = field(r, "general.architecture")
    tensors = {}
    for name, t, kind in items:
        if kind != "expert-tiles":
            continue
        K, N, E = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])
        rt, kt = grid(N, K)
        chunk = rt * kt * TILE
        lo, hi = table[name]["data_offsets"]
        if hi - lo != E * chunk:
            raise SystemExit(f"{name}: expert stack does not fill its blob")
        tensors[name] = {
            "experts": E, "rows": N, "cols": K, "tile_rows": rt, "tile_cols": kt,
            "tiles": rt * kt, "bytes": chunk,
            "offsets": [lo + e * chunk for e in range(E)],
        }
    return {
        "arch": arch,
        "experts": int(field(r, f"{arch}.expert_count")),
        "used": int(field(r, f"{arch}.expert_used_count")),
        "norm_topk_prob": True,
        "tensors": tensors,
    }


def pack(args):
    r = gguf.GGUFReader(args.gguf)
    os.makedirs(args.out, exist_ok=True)
    items = planned(r)
    # Every blob's size follows from its shape, so the table (and so the header) is written
    # before any weight is converted and the data is streamed to its offset in one pass: a 30B
    # expert stack does not fit in memory as a list of blobs.
    table, off = {}, 0
    for name, t, kind in items:
        off += (-off) % 64
        n = blob_bytes(t, kind)
        table[name] = {"dtype": "BF16" if kind == "bf16" else "I8",
                       "shape": blob_shape(t, kind), "data_offsets": [off, off + n]}
        off += n
    meta = {"format": "q4nx", "source": os.path.basename(args.gguf),
            "converter": "tools/gguf_to_q4nx.py"}
    if field(r, "general.architecture") == "qwen3moe":
        meta["moe"] = moe_meta(r, items, table)
    table["__metadata__"] = meta
    header = json.dumps(table, separators=(",", ":")).encode()
    header += b" " * ((-len(header)) % 64)
    path = os.path.join(args.out, "model.q4nx")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.truncate(8 + len(header) + off)  # sparse: the 64-byte pads stay zero
        for name, t, kind in items:
            data, exact = encode(t, kind)
            lo, hi = table[name]["data_offsets"]
            if len(data) != hi - lo:
                raise SystemExit(f"{name}: wrote {len(data)} B, the table says {hi - lo} B")
            f.seek(8 + len(header) + lo)
            f.write(data)
            how = "bf16" if kind == "bf16" else ("exact" if exact else "requantized")
            if kind == "expert-tiles":
                how += f" {int(t.shape[2])} experts"
            print(f"{name:52s} {t.tensor_type.name:5s} -> {table[name]['dtype']} "
                  f"{table[name]['shape']} {how}", flush=True)
    vocab = table["model.embed_tokens.weight"]["shape"][0]
    cfg = config(r, vocab)
    cfg["q4nx_source"]["gguf"] = os.path.basename(args.gguf)
    # Which device-side weight layout the kernels for this container must be
    # built for.  It is a property of the artifacts, not of the payload: the
    # engine packs the same int8 codes either way, so a container is "rm" unless
    # the person building it also emitted the _b8 kernel family (the blocked
    # 8x8-microtile layout is 2.4-2.6x faster to stream and produces the
    # bit-identical int32 result; see docs/npu.md).
    cfg["q4nx_weight_layout"] = args.weight_layout
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    # A model directory is not servable without tokenizer.json (npu/tokenizer.cpp reads it), so
    # fall back to the GGUF's own directory -- an HF snapshot or export dir usually has one -- and
    # say so loudly when it is still missing, rather than reporting success on an unusable directory.
    tokdir = args.tokenizer or os.path.dirname(os.path.abspath(args.gguf))
    copied = []
    for n in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(tokdir, n)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(args.out, n))
            copied.append(n)
    if "tokenizer.json" not in copied:
        print(f"WARNING: no tokenizer.json in {tokdir}: {args.out} has model.q4nx and config.json "
              "only, and 1bit forward-serve will refuse it. Pass --tokenizer <dir>.")
    print(f"wrote {args.out}/model.q4nx ({8 + len(header) + off} bytes)")


def verify(args):
    """Every Q4NX tensor against the GGUF, dequantized."""
    r = gguf.GGUFReader(args.gguf)
    with open(os.path.join(args.out, "model.q4nx"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
    mm = np.memmap(os.path.join(args.out, "model.q4nx"), dtype=np.uint8, mode="r")
    bad = 0
    for name, t, kind in planned(r):
        lo, hi = hdr[name]["data_offsets"]
        raw = np.asarray(mm[base + lo:base + hi])
        ref = dequantize(t.data, t.tensor_type).astype(np.float64)
        if kind == "bf16":
            got = from_bf16(raw.view(np.uint16)).astype(np.float64).reshape(-1)
            ref = ref.reshape(-1)
            err = np.abs(got - ref)
            bound = np.abs(ref) * 2.0 ** -8 + 1e-30  # half a bf16 ulp is <= 2^-8 relative
            ok = bool((err <= bound).all())
            note = f"bf16: max rel err {(err / (np.abs(ref) + 1e-30)).max():.3g}"
        else:
            # One tile grid per expert for a routed stack, one for a dense projection.
            K, N = int(t.shape[0]), int(t.shape[1])
            E = int(t.shape[2]) if kind == "expert-tiles" else 1
            rt, kt = grid(N, K)
            tiles_e = raw.reshape(E, rt * kt, TILE)
            ref = ref.reshape(E, N, K)
            exact = t.tensor_type in EXACT
            codes, scale, zero = (None, None, None)
            if exact:
                codes, scale, zero, _ = to_q4_1(t)
                codes = codes.reshape(E, N, K)
                scale, zero = scale.reshape(E, N, K // 32), zero.reshape(E, N, K // 32)
            over, worst, worstb = 0, 0.0, 0.0
            gse = rsse = dot = 0.0
            for e in range(E):
                got = untile(tiles_e[e], N, K).astype(np.float64)
                err = np.abs(got - ref[e])
                worst = max(worst, float(err.max()))
                if exact:
                    # rounding scale and zero to bf16 moves each weight by at most
                    # code * |ds| + |dz| (half a bf16 ulp each), plus one f32 ulp for the
                    # rounding inside gguf-py's own f32 dequantize (untile's is exact)
                    ds = np.abs(from_bf16(bf16_bits(scale[e])) - scale[e]).astype(np.float64)
                    dz = np.abs(from_bf16(bf16_bits(zero[e])) - zero[e]).astype(np.float64)
                    bound = codes[e].reshape(N, K // 32, 32) * ds[..., None] + dz[..., None]
                    bound = bound.reshape(N, K) + np.spacing(np.abs(ref[e]).astype(np.float32))
                    over += int((err > bound).sum())
                    worstb = max(worstb, float(bound.max()))
                else:
                    gse += float((got * got).sum())
                    rsse += float((ref[e] * ref[e]).sum())
                    dot += float((got * ref[e]).sum())
            if exact:
                ok = over == 0
                note = (f"exact to bf16 scales: max err {worst:.3g} (bound {worstb:.3g}), "
                        f"{over} over")
            else:
                rel = float(np.sqrt(max(gse + rsse - 2 * dot, 0.0) / max(rsse, 1e-30)))
                cos = float(dot / np.sqrt(max(gse * rsse, 1e-30)))
                ok = cos > 0.99
                note = f"requantized: rel rms err {rel:.4f}, cos {cos:.6f}"
            if E > 1:
                note += f" ({E} experts)"
        bad += not ok
        if not ok or args.all or "layers." not in name or ".layers.0." in name:
            print(f"{'ok ' if ok else 'BAD'} {name:52s} {t.tensor_type.name:5s} {note}")
    print(f"{len(planned(r)) - bad}/{len(planned(r))} tensors ok")
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pack")
    a.add_argument("gguf")
    a.add_argument("out")
    a.add_argument("--tokenizer", help="directory with tokenizer.json (and tokenizer_config.json)")
    a.add_argument("--weight-layout", dest="weight_layout", choices=("rm", "b8"), default="rm",
                   help="device weight layout the kernels will be built for: rm (default, "
                        "row-major, matches every container before this) or b8 (blocked "
                        "8x8 microtiles, needs the _b8 kernel family, streams 2.4-2.6x faster "
                        "with a bit-identical result)")
    b = sub.add_parser("verify")
    b.add_argument("gguf")
    b.add_argument("out")
    b.add_argument("--all", action="store_true", help="print every tensor, not only layer 0")
    args = p.parse_args()
    return pack(args) if args.cmd == "pack" else verify(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
