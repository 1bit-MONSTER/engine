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
         (min/max): lossy -- or to Q8 tiles (I8 [tiles, 8704]: bf16 scale per row and
         32-column group, int8 codes) when the tensor's Q4NX name matches --dtype8

The first three are exact apart from rounding scale and zero to bf16 (nearest even);
verify checks every tensor against that bound and reports the error of the lossy ones.
--dtype8 exists because q4_1 spends 4 bits of every 32-value group on a min and only has
16 levels: a tensor that was Q6_K or Q8_0 in the source (typically v_proj, down_proj and
an untied lm_head) is strictly worse *and* bigger after the crush, so naming it keeps it at
8 bits for 8.5 bits/weight instead of 5 (e.g. --dtype8 'v_proj|down_proj'). Default is
today's behaviour, so no existing container changes.
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
import re
import shutil
import struct
import sys

import numpy as np
import gguf
from gguf.quants import dequantize

from q4nx_pack import (COLS, GROUP, Q8, ROWS, TILE, bf16_bits, f16, from_bf16,  # noqa: F401
                       quantize_q4_1, quantize_q8, tiles, tiles_q8, untile, untile_q8)


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
    if is_expert_stack(t):
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


def refuse_unsupported_rope(r):
    """A scaling this route cannot apply is refused before anything is written (engine#1087).

    A declared block that is dropped is worse than a failed conversion: the container would carry
    rope_theta and nothing else, be served plain RoPE, stay fluent and be wrong past the model's
    original window, with no line anywhere saying so.  This route has no scaling rule at all (the packed
    lane computes theta^(-i/half) and nothing else), so every declared type is refused by name.  The
    forward implements llama3 and longrope short_factor, but only for a container whose config.json
    carries the block, which is a different route (1bit npu-pack import, or a checkpoint's own config).
    """
    arch = field(r, "general.architecture")
    scaling = field(r, f"{arch}.rope.scaling.type")
    if scaling is not None and str(scaling) not in ("none", "default", ""):
        raise SystemExit(
            f"{arch}.rope.scaling.type is {str(scaling)!r}: this converter has no RoPE scaling rule, so the "
            f"container would be served plain RoPE and reply wrongly past the model's original window. "
            f"Convert from the checkpoint with its config.json (the forward route reads rope_scaling), or "
            f"file the scaling type you need (engine#1087)")


def config(r, vocab):
    arch = field(r, "general.architecture")
    if arch not in ("qwen3", "qwen2", "qwen3moe"):
        raise SystemExit(f"architecture {arch!r}: only qwen3, qwen2 and qwen3moe are supported")
    a = lambda k, d=None: field(r, f"{arch}.{k}", d)  # noqa: E731
    refuse_unsupported_rope(r)
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
    # The tokenizer's add_bos/add_eos flags are part of the prompt, not decoration.
    # app/forward_serve.cpp reads config.json's add_bos_token ("the rendered chat prompt needs
    # BOS") and defaults to 0 when the key is absent, so a checkpoint whose tokenizer_config.json
    # sets add_bos_token: true is served chat prompts with NO BOS.  Measured on Zyphra/ZR1-1.5B
    # (a Qwen2 fine-tune): the GGUF's metadata says true, the container's config.json was silent,
    # and the forward built a 5-token prompt where llama.cpp built 6 -- its reply opened with the
    # end-of-turn token.  Carry both flags through while the GGUF still has them.
    for gguf_key, hf_key in (("tokenizer.ggml.add_bos_token", "add_bos_token"),
                             ("tokenizer.ggml.add_eos_token", "add_eos_token")):
        flag = field(r, gguf_key)
        if flag is not None:
            c[hf_key] = bool(flag)
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


# The routed expert stacks, by NAME. Rank is not the test: a conv or an SSM/CCA kernel is 3-D too,
# and taking `len(shape) == 3` as "expert stack" decomposed such a tensor into shape[2] "experts"
# of shape[1] rows -- silently, because __metadata__.moe is only written for qwen3moe and verify()
# runs planned() as well, so it checks the same wrong shape. A Zaya-style model (CCA conv weights)
# would have packed garbage and verified clean.
EXPERT_PARTS = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"}


def is_expert_stack(t):
    """A routed expert stack: 3-D AND one of the ggml ffn_*_exps parts, [K, N, E]."""
    return len(t.shape) == 3 and part_of(t.name) in EXPERT_PARTS


def planned(r):
    """(q4nx name, gguf tensor, kind), kind 'tiles', 'expert-tiles' or 'bf16'.

    The router stays bf16 (it runs on the host and its top-k must be the GGUF's own), as do the
    norms and the embedding. A 3-D blk.N.ffn_*_exps tensor is a routed expert stack; any other 3-D
    tensor is refused here, by name, because the container has no way to record a conv kernel.
    """
    out = []
    names = {t.name for t in r.tensors}
    for t in r.tensors:
        n = hf_name(t.name)
        if len(t.shape) == 1 or t.name == "token_embd.weight" or part_of(t.name) in ROUTER:
            kind = "bf16"
        elif is_expert_stack(t):
            kind = "expert-tiles"
        else:
            if len(t.shape) == 3:
                raise SystemExit(
                    f"{t.name}: 3-D tensor {'x'.join(str(d) for d in t.shape)} is not a routed expert "
                    f"stack (only {', '.join(sorted(EXPERT_PARTS))} are); a dense 3-D weight cannot be "
                    "tiled into 32 x 256 chunks")
            kind = "tiles"
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


def blob_shape(t, kind, chunk=TILE):
    """The shape the container records for this tensor's data."""
    if kind == "bf16":
        return [int(t.shape[1]), int(t.shape[0])] if len(t.shape) == 2 else [int(np.prod(t.shape))]
    K, N = int(t.shape[0]), int(t.shape[1])
    rows, cols = (N * int(t.shape[2]), K) if kind == "expert-tiles" else (N, K)
    rt, kt = grid(rows, cols)
    return [rt, kt, chunk]


def blob_bytes(t, kind, chunk=TILE):
    return int(np.prod(blob_shape(t, kind, chunk))) * (2 if kind == "bf16" else 1)


def encode(t, kind, chunk=TILE):
    """One tensor's container bytes, and whether the conversion was exact."""
    if kind == "bf16":
        w = dequantize(t.data, t.tensor_type).astype(np.float32)
        return np.ascontiguousarray(bf16_bits(w if w.ndim == 2 else w.reshape(-1))).tobytes(), True
    if chunk == Q8:
        K, n = int(t.shape[0]), int(t.shape[1])
        n *= int(t.shape[2]) if len(t.shape) == 3 else 1  # expert-major rows, as to_q4_1
        codes, scale = quantize_q8(dequantize(t.data, t.tensor_type).astype(np.float32).reshape(n, K))
        data, exact = tiles_q8(codes, scale), False
    else:
        codes, scale, zero, exact = to_q4_1(t)
        data = tiles(codes, scale, zero)
    if kind == "expert-tiles":
        rt, kt = grid(int(t.shape[1]), int(t.shape[0]))
        data = data.reshape(int(t.shape[2]) * rt, kt, chunk)
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
        # The chunk kind is in the table already (5120 for q4_1, 8704 for a --dtype8 stack).
        chunk = rt * kt * table[name]["shape"][-1]
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
    refuse_unsupported_rope(r)   # before the output directory exists: no half-written container
    os.makedirs(args.out, exist_ok=True)
    items = planned(r)
    # --dtype8 names tensors to store as Q8 chunks instead of q4_1. Selection is by Q4NX name,
    # so one pattern covers every layer; the router and the bf16 tensors are never affected, and
    # neither is a tensor the source already held at 4 bits (Q4_K_M mixes types layer by layer:
    # an exact Q4_K tensor has nothing to gain from 8 bits and only grows).
    pattern = re.compile(args.dtype8) if args.dtype8 else None
    chunk = {}
    for name, t, kind in items:
        q8 = bool(pattern and kind != "bf16" and pattern.search(name))
        if q8 and t.tensor_type in EXACT:
            print(f"{name:52s} {t.tensor_type.name:5s} -> q4_1 (named by --dtype8, already exact "
                  "at 4 bits)", flush=True)
            q8 = False
        chunk[name] = Q8 if q8 else TILE
    # Every blob's size follows from its shape, so the table (and so the header) is written
    # before any weight is converted and the data is streamed to its offset in one pass: a 30B
    # expert stack does not fit in memory as a list of blobs.
    table, off = {}, 0
    for name, t, kind in items:
        off += (-off) % 64
        n = blob_bytes(t, kind, chunk[name])
        table[name] = {"dtype": "BF16" if kind == "bf16" else "I8",
                       "shape": blob_shape(t, kind, chunk[name]), "data_offsets": [off, off + n]}
        off += n
    meta = {"format": "q4nx", "source": os.path.basename(args.gguf),
            "converter": "tools/gguf_to_q4nx.py"}
    if args.dtype8:
        # The chunk kinds are already in every tensor's shape; the pattern is recorded so a
        # round trip (tools/q4nx_to_gguf.py verify --roundtrip) re-packs the same container.
        meta["dtype8"] = args.dtype8
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
            data, exact = encode(t, kind, chunk[name])
            lo, hi = table[name]["data_offsets"]
            if len(data) != hi - lo:
                raise SystemExit(f"{name}: wrote {len(data)} B, the table says {hi - lo} B")
            f.seek(8 + len(header) + lo)
            f.write(data)
            how = "bf16" if kind == "bf16" else (
                "Q8 int8" if chunk[name] == Q8 else ("exact" if exact else "requantized"))
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
            # A Q8 tensor's chunk is 8704 B and its codes are symmetric int8, so the q4_1 bound
            # below does not apply: check against 0.5 * the group scale the codes were rounded
            # with, plus the bf16 rounding of the stored scale (<= 127 * |d_bf16 - d|).
            q8 = int(hdr[name]["shape"][-1]) == Q8
            tiles_e = raw.reshape(E, rt * kt, Q8 if q8 else TILE)
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
                if q8:
                    got = untile_q8(tiles_e[e], N, K).astype(np.float64)
                    err = np.abs(got - ref[e]).reshape(N, K // 32, 32)
                    se = np.abs(ref[e]).reshape(N, K // 32, 32).max(-1) / 127.0
                    d_bf = from_bf16(bf16_bits(se)).astype(np.float64)
                    eps = np.spacing(np.abs(ref[e]).astype(np.float32)).reshape(N, K // 32, 32)
                    bound = (0.5 * se + 127.0 * np.abs(d_bf - se))[..., None] + eps
                    over += int((err > bound).sum())
                    worst = max(worst, float(err.max()))
                    worstb = max(worstb, float(bound.max()))
                    gse += float((got * got).sum())
                    rsse += float((ref[e] * ref[e]).sum())
                    dot += float((got * ref[e]).sum())
                    continue
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
            if q8:
                rel = float(np.sqrt(max(gse + rsse - 2 * dot, 0.0) / max(rsse, 1e-30)))
                cos = float(dot / np.sqrt(max(gse * rsse, 1e-30)))
                ok = over == 0 and cos > 0.99
                note = (f"Q8 int8: max err {worst:.3g} (bound {worstb:.3g}), {over} over, "
                        f"rel rms err {rel:.4f}, cos {cos:.6f}")
            elif exact:
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
    a.add_argument("--dtype8", metavar="REGEX",
                   help="store the tensors whose Q4NX name matches REGEX as Q8 chunks (int8 "
                        "codes, bf16 scale) instead of requantizing them to q4_1, e.g. "
                        "'v_proj|down_proj'; meant for the tensors the source never had at "
                        "4 bits, so an exactly-4-bit source is left alone")
    b = sub.add_parser("verify")
    b.add_argument("gguf")
    b.add_argument("out")
    b.add_argument("--all", action="store_true", help="print every tensor, not only layer 0")
    args = p.parse_args()
    return pack(args) if args.cmd == "pack" else verify(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
