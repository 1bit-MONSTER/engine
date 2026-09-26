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
Norms and the embedding are stored as bf16. A tied lm_head (no output.weight) is made
from token_embd.

Supported architectures: qwen3 (the dx route's model). Needs numpy and gguf (gguf-py).
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

ROWS, COLS, GROUP, TILE = 32, 256, 32, 5120


def bf16_bits(x):
    """f32 -> bf16 bits, round to nearest even."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(b):
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f16(raw):
    return np.ascontiguousarray(raw).view(np.float16).astype(np.float32)


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


def quantize_q4_1(w):
    """f32 [N, K] -> q4_1 per 32-column group (min/max), scale and zero already bf16."""
    N, K = w.shape
    g = w.reshape(N, K // GROUP, GROUP)
    lo, hi = g.min(-1), g.max(-1)
    zero = from_bf16(bf16_bits(lo))
    scale = from_bf16(bf16_bits((hi - lo) / 15.0))
    safe = np.where(scale > 0, scale, 1.0)
    codes = np.clip(np.rint((g - zero[..., None]) / safe[..., None]), 0, 15).astype(np.uint8)
    return codes.reshape(N, K), scale, zero


def to_q4_1(t):
    """A GGUF 2-D tensor as (codes, scale, zero, exact)."""
    K, N = int(t.shape[0]), int(t.shape[1])  # ggml order: ne0 = columns
    if N % ROWS or K % COLS:
        raise SystemExit(f"{t.name}: {N} x {K} is not a whole number of 32 x 256 tiles")
    split = EXACT.get(t.tensor_type)
    if split:
        return (*split(np.asarray(t.data).reshape(-1), N, K), True)
    return (*quantize_q4_1(dequantize(t.data, t.tensor_type).reshape(N, K).astype(np.float32)), False)


def tiles(codes, scale, zero):
    """q4_1 parts -> [tiles, 5120] bytes, tiles row-major over the 32 x 256 grid."""
    N, K = codes.shape
    RT, KT = N // ROWS, K // COLS
    # scale/zero index g*32 + r within a tile
    def plane(v):
        return bf16_bits(v).reshape(RT, ROWS, KT, 8).transpose(0, 2, 3, 1).reshape(RT * KT, 256)
    # codes: region h*8+g (h = r/16), byte 8*j + (r%16)/2, even row in the low nibble
    c = codes.astype(np.uint8).reshape(RT, 2, 8, 2, KT, 8, 32)  # rt, h, r2, parity, kt, g, j
    packed = c[:, :, :, 0] | (c[:, :, :, 1] << 4)  # rt, h, r2, kt, g, j
    packed = packed.transpose(0, 3, 1, 4, 5, 2).reshape(RT * KT, 4096)  # rt, kt, h, g, j, r2
    return np.concatenate([plane(scale).view(np.uint8), plane(zero).view(np.uint8), packed], axis=1)


def untile(raw, N, K):
    """[tiles, 5120] q4_1 -> f32 [N, K] (the decoder of npu/q4nx.cpp)."""
    RT, KT = N // ROWS, K // COLS
    t = raw.reshape(RT, KT, TILE)
    sc = from_bf16(t[..., :512].copy().view(np.uint16)).reshape(RT, KT, 8, 32)
    zr = from_bf16(t[..., 512:1024].copy().view(np.uint16)).reshape(RT, KT, 8, 32)
    p = t[..., 1024:].reshape(RT, KT, 2, 8, 32, 8)  # rt, kt, h, g, j, r2
    c = np.stack([p & 0xF, p >> 4], axis=-1)  # ..., r2, parity
    c = c.transpose(0, 2, 5, 6, 1, 3, 4).reshape(RT, 32, KT, 8, 32)  # rt, r, kt, g, j
    s = sc.transpose(0, 3, 1, 2)[..., None]  # rt, r, kt, g, 1
    z = zr.transpose(0, 3, 1, 2)[..., None]
    return (c * s + z).reshape(N, K)


# --- names and config ---

LAYER = {
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
    "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
}
TOP = {"token_embd": "model.embed_tokens", "output_norm": "model.norm", "output": "lm_head"}


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
    if f is None:
        return default
    v = f.parts[f.data[0]]
    if f.types and f.types[0] == gguf.GGUFValueType.STRING:
        return bytes(v).decode()
    return v.tolist()[0] if len(v) == 1 else v.tolist()


def config(r, vocab):
    arch = field(r, "general.architecture")
    if arch != "qwen3":
        raise SystemExit(f"architecture {arch!r}: only qwen3 is supported")
    a = lambda k, d=None: field(r, f"{arch}.{k}", d)  # noqa: E731
    heads = a("attention.head_count")
    tied = not any(t.name == "output.weight" for t in r.tensors)
    c = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": arch,
        "hidden_size": a("embedding_length"),
        "intermediate_size": a("feed_forward_length"),
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
    return {k: v for k, v in c.items() if v is not None}


def planned(r):
    """(q4nx name, gguf tensor, kind) with kind 'tiles' or 'bf16'."""
    out = []
    names = {t.name for t in r.tensors}
    for t in r.tensors:
        n = hf_name(t.name)
        out.append((n, t, "bf16" if len(t.shape) == 1 or t.name == "token_embd.weight" else "tiles"))
    if "output.weight" not in names:
        emb = next(t for t in r.tensors if t.name == "token_embd.weight")
        out.append(("lm_head.weight", emb, "tiles"))
    return out


def pack(args):
    r = gguf.GGUFReader(args.gguf)
    os.makedirs(args.out, exist_ok=True)
    table, blobs, off = {}, [], 0
    for name, t, kind in planned(r):
        if kind == "tiles":
            codes, scale, zero, exact = to_q4_1(t)
            data = tiles(codes, scale, zero)
            table[name] = {"dtype": "I8", "shape": list(data.shape)}
            how = "exact" if exact else "requantized"
        else:
            w = dequantize(t.data, t.tensor_type).astype(np.float32)
            if kind == "bf16" and w.ndim == 2:
                table[name] = {"dtype": "BF16", "shape": list(w.shape)}
            else:
                w = w.reshape(-1)
                table[name] = {"dtype": "BF16", "shape": [int(w.size)]}
            data = bf16_bits(w)
            how = "bf16"
        b = np.ascontiguousarray(data).tobytes()
        pad = (-off) % 64
        if pad:
            blobs.append(b"\0" * pad)
            off += pad
        table[name]["data_offsets"] = [off, off + len(b)]
        blobs.append(b)
        off += len(b)
        print(f"{name:52s} {t.tensor_type.name:5s} -> {table[name]['dtype']} {table[name]['shape']} {how}")
    table["__metadata__"] = {"format": "q4nx", "source": os.path.basename(args.gguf),
                             "converter": "tools/gguf_to_q4nx.py"}
    header = json.dumps(table, separators=(",", ":")).encode()
    header += b" " * ((-len(header)) % 64)
    with open(os.path.join(args.out, "model.q4nx"), "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for b in blobs:
            f.write(b)
    vocab = table["model.embed_tokens.weight"]["shape"][0]
    cfg = config(r, vocab)
    cfg["q4nx_source"]["gguf"] = os.path.basename(args.gguf)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    if args.tokenizer:
        for n in ("tokenizer.json", "tokenizer_config.json"):
            src = os.path.join(args.tokenizer, n)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(args.out, n))
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
        if kind == "tiles":
            K, N = int(t.shape[0]), int(t.shape[1])
            got = untile(raw, N, K).astype(np.float64)
            ref = ref.reshape(N, K)
            err = np.abs(got - ref)
            if t.tensor_type in EXACT:
                codes, scale, zero, _ = to_q4_1(t)
                # rounding scale and zero to bf16 moves each weight by at most
                # code * |ds| + |dz| (half a bf16 ulp each), plus one f32 ulp for the
                # rounding inside gguf-py's own f32 dequantize (untile's is exact)
                ds = np.abs(from_bf16(bf16_bits(scale)) - scale).astype(np.float64)
                dz = np.abs(from_bf16(bf16_bits(zero)) - zero).astype(np.float64)
                bound = codes.reshape(N, K // 32, 32) * ds[..., None] + dz[..., None]
                bound = bound.reshape(N, K) + np.spacing(np.abs(ref).astype(np.float32))
                over = int((err > bound).sum())
                ok = over == 0
                note = f"exact to bf16 scales: max err {err.max():.3g} (bound {bound.max():.3g}), {over} over"
            else:
                rel = np.sqrt((err ** 2).mean() / max((ref ** 2).mean(), 1e-30))
                cos = float((got * ref).sum() / np.sqrt((got * got).sum() * (ref * ref).sum()))
                ok = cos > 0.99
                note = f"requantized: rel rms err {rel:.4f}, cos {cos:.6f}"
        else:
            got = from_bf16(raw.view(np.uint16)).astype(np.float64).reshape(-1)
            ref = ref.reshape(-1)
            err = np.abs(got - ref)
            bound = np.abs(ref) * 2.0 ** -8 + 1e-30  # half a bf16 ulp is <= 2^-8 relative
            ok = bool((err <= bound).all())
            note = f"bf16: max rel err {(err / (np.abs(ref) + 1e-30)).max():.3g}"
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
    b = sub.add_parser("verify")
    b.add_argument("gguf")
    b.add_argument("out")
    b.add_argument("--all", action="store_true", help="print every tensor, not only layer 0")
    args = p.parse_args()
    return pack(args) if args.cmd == "pack" else verify(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
