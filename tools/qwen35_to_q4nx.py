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
"""qwen35_to_q4nx.py -- GGUF (qwen35, un-rotated) -> Q4NX model dir for the NPU forward.

Takes a plain qwen35 GGUF stored as Q4_0 or Q4_1 and writes the Q4NX directory the NPU forward
reads (npu/q4nx.h). `tools/gguf_to_q4nx.py` covers qwen3, qwen3moe and qwen2 only: no GDN
(linear attention) families, no v-head reorder, no Q8/F32 kinds.  This adds exactly those for
arch qwen35, and gguf_to_q4nx.py refuses a qwen35 GGUF by pointing here (engine#2013).

Two inputs are refused before anything is written:
  * a GGUF that still carries prism.hadamard.* keys: its weights are stored in a
    Walsh-Hadamard-rotated basis, this tool writes weights as they are and no sign table, so
    the container would load and answer wrongly.  It has to be un-rotated first.
  * a tensor that is not Q4_0/Q4_1 where tiles are needed (PTQ1_0, ggml type 143, is the usual
    one): tools/ternary_to_q4_0.py rewrites PTQ1_0 as Q4_0 but keeps the rotation, so a
    PrismML ternary file needs both steps.
The 27B container on the dev box was made by this tool from an un-rotated Q4_0 GGUF; the
un-rotation step itself is not in this tree.

Writes, matching the forward's own readers (npu/forward/src/q4nx_forward.cpp):

  q4_1 tiles [tiles, 5120]   every 2-D projection, the embedding and lm_head.  Q4_0/Q4_1
                             convert exactly: codes kept, scale/zero to bf16.
  Q8 tiles   [rt, kt, 8704]  ssm_alpha/beta, row-padded 48 -> 64 because the forward's
                             deq_pad dequantises at the padded height (40 tiles x 8704 B).
  F32        [n]             ssm_a, ssm_dt.bias (the forward memcpy's these raw).
  BF16       [n]             every RMSNorm, ssm_norm, and ssm_conv1d transposed to [CK][CD].

The v-head reorder is the H2 container's, measured against the FLM reference container
(ssm_a cos 1.000000 interleaved vs 0.456755 identity): 48 v-heads are stored rep-major
(`i = rep*16 + kh`) and the forward wants HF grouping (`j = kh*3 + rep`; it repeats q/k per
v-head as `src = h / rep`), so HF head j takes stored head `perm[j] = (j%3)*16 + j//3`.
Applied to the 48-head axis of the GDN families (head-blocks of 128 for qkv v-rows,
gate_proj rows, conv1d v-channels, ssm_out columns; whole rows for ssm_alpha/beta; scalars
for ssm_a/ssm_dt) and, on full-attention layers, to the per-head `[q|gate]` interleave of
attn_q (`(h, p, d) -> (p, h, d)`, p = 2, d = head_dim).

usage: qwen35_to_q4nx.py pack   <model.gguf> <out dir> [--tokenizer <dir>] [--no-reorder]
       qwen35_to_q4nx.py verify <model.gguf> <out dir>
"""
import argparse
import json
import mmap
import os
import shutil
import struct
import sys

import numpy as np

ROWS, COLS, GROUP = 32, 256, 32
F32, F16, Q4_0, Q4_1, BF16, PTQ1 = 0, 1, 2, 3, 30, 143
BLK = {F32: (4, 1), F16: (2, 1), Q4_0: (18, 32), Q4_1: (20, 32), BF16: (2, 1), PTQ1: (28, 128)}
NKH, NVH = 16, 48                  # k-groups, v-heads (qwen35 27B: group_count 16, inner 6144 / 128)


# --- GGUF reader --------------------------------------------------------------------------
class Gguf:
    """Tensor table, an mmap, and per-type readers for the four types this model uses."""

    _SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: 0, 9: 0, 10: 8, 11: 8, 12: 8}
    _FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}

    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        magic, _ = struct.unpack("<II", self.f.read(8))
        if magic != 0x46554747:
            raise SystemExit(f"{path}: not a GGUF")
        n_t, n_kv = struct.unpack("<QQ", self.f.read(16))
        self.kv = {}
        for _ in range(n_kv):
            k = self._str()
            (t,) = struct.unpack("<I", self.f.read(4))
            self.kv[k] = self._val(t)
        self.tensors = {}
        for _ in range(n_t):
            name = self._str()
            (nd,) = struct.unpack("<I", self.f.read(4))
            dims = struct.unpack("<%dQ" % nd, self.f.read(8 * nd))
            (tt,) = struct.unpack("<I", self.f.read(4))
            (off,) = struct.unpack("<Q", self.f.read(8))
            self.tensors[name] = (dims, tt, off)
        pos = self.f.tell()
        self.base = pos + (-pos) % int(self.kv.get("general.alignment", 32))
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def _str(self):
        (n,) = struct.unpack("<Q", self.f.read(8))
        return self.f.read(n).decode("utf-8", "replace")

    def _val(self, t):
        if t == 8:
            return self._str()
        if t == 9:
            at, n = struct.unpack("<IQ", self.f.read(12))
            return [self._val(at) for _ in range(n)]
        return struct.unpack("<" + self._FMT[t], self.f.read(self._SZ[t]))[0]

    def raw(self, name):
        dims, tt, off = self.tensors[name]
        sz, per = BLK[tt]
        n = int(np.prod(dims))
        return np.frombuffer(self.mm, np.uint8, count=(n // per) * sz, offset=self.base + off), dims, tt

    def f32(self, name):
        a, _, tt = self.raw(name)
        if tt == F32:
            return np.frombuffer(a.tobytes(), "<f4")
        if tt == BF16:
            return from_bf16(np.frombuffer(a.tobytes(), "<u2"))
        raise SystemExit(f"{name}: type {tt}, expected f32/bf16")

    def q4(self, name):
        """-> codes [N, K] u8, scale [N, K/32] f32, zero [N, K/32] f32, exact for Q4_0/Q4_1."""
        a, dims, tt = self.raw(name)
        if tt not in (Q4_0, Q4_1):
            raise SystemExit(f"{name}: ggml type {tt}, expected Q4_0 (2) or Q4_1 (3)"
                             + ("; 143 is PTQ1_0: rewrite it as Q4_0 (tools/ternary_to_q4_0.py) and un-rotate it "
                                "first" if tt == PTQ1 else ""))
        K, N = int(dims[0]), int(dims[1])
        b = a.reshape(N, K // 32, 18 if tt == Q4_0 else 20)
        f16 = lambda x: np.frombuffer(x.tobytes(), "<f2").reshape(N, K // 32).astype(np.float32)  # noqa: E731
        d = f16(b[:, :, 0:2])
        if tt == Q4_0:
            m, q = -8.0 * d, b[:, :, 2:18]
        else:
            m, q = f16(b[:, :, 2:4]), b[:, :, 4:20]
        return np.ascontiguousarray(np.concatenate([q & 0xF, q >> 4], axis=-1).reshape(N, K)), d, m


# --- reorders ---------------------------------------------------------------------------------
def head_perm():
    """HF v-head j <- stored head (j%3)*16 + j//3."""
    j = np.arange(NVH)
    return (j % 3) * NKH + j // 3


def perm_rows(n, hd, off):
    """Row gather for n head-blocks of hd rows starting at row `off` (identity elsewhere)."""
    p = head_perm()[:n]
    idx = np.arange(off) if off else np.arange(0)
    return np.concatenate([idx, np.concatenate([off + b * hd + np.arange(hd) for b in p])])


def perm_cols(n, hd, off):
    """Column gather for n head-blocks of hd columns starting at column `off`."""
    return np.concatenate([off + b * hd + np.arange(hd) for b in head_perm()[:n]])


def deinterleave_qgate(n, hd):
    """Stored [head][q|gate][hd] -> [all q][all gate]."""
    nh = n // (2 * hd)
    out = np.empty(n, np.int64)
    for h in range(nh):
        for p in range(2):
            dst = p * nh * hd + h * hd
            out[dst:dst + hd] = h * 2 * hd + p * hd + np.arange(hd)
    return out


def row_gather(codes, scale, zero, idx):
    if idx is None:
        return codes, scale, zero
    return codes[idx], scale[idx], zero[idx]


def col_gather(codes, scale, zero, hd, off):
    """Reorder the 48 head-blocks of `hd` columns starting at column `off` (rows kept)."""
    ci = np.concatenate([off + b * hd + np.arange(hd) for b in head_perm()])
    gi = np.concatenate([(off + b * hd) // GROUP + np.arange(hd // GROUP) for b in head_perm()])
    return codes[:, ci], scale[:, gi], zero[:, gi]


# --- tiles ------------------------------------------------------------------------------------
def bf16_bits(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(b):
    return (np.asarray(b, np.uint16).astype(np.uint32) << 16).view(np.float32)


def tile_q4_1(codes, scale, zero):
    N, K = codes.shape
    RT, KT = N // ROWS, K // COLS

    def plane(v):
        return bf16_bits(v).reshape(RT, ROWS, KT, 8).transpose(0, 2, 3, 1).reshape(RT * KT, 256)

    c = codes.astype(np.uint8).reshape(RT, 2, 8, 2, KT, 8, 32)
    packed = (c[:, :, :, 0] | (c[:, :, :, 1] << 4)).transpose(0, 3, 1, 4, 5, 2).reshape(RT * KT, 4096)
    return np.concatenate([plane(scale).view(np.uint8), plane(zero).view(np.uint8), packed], axis=1)


def untile_q4_1(raw, N, K):
    """[tiles, 5120] -> f32 [N, K] (npu/q4nx.h: w = code*scale + zero)."""
    RT, KT = N // ROWS, K // COLS
    t = np.asarray(raw).reshape(RT, KT, 5120)
    sc = from_bf16(t[..., :512].copy().view("<u2")).reshape(RT, KT, 8, 32)
    zr = from_bf16(t[..., 512:1024].copy().view("<u2")).reshape(RT, KT, 8, 32)
    p = t[..., 1024:].reshape(RT, KT, 2, 8, 32, 8)
    c = np.stack([p & 0xF, p >> 4], axis=-1).transpose(0, 2, 5, 6, 1, 3, 4).reshape(RT, 32, KT, 8, 32)
    return (c * sc.transpose(0, 3, 1, 2)[..., None] + zr.transpose(0, 3, 1, 2)[..., None]).reshape(N, K)


def tile_q8(w):
    """f32 [N, K] (multiples of 32) -> [N/32, K/256, 8704]; d[g*32+r] @0, int8 codes @512 at k*32+r."""
    N, K = w.shape
    RT, KT = N // ROWS, K // COLS
    out = np.empty((RT, KT, 8704), np.uint8)
    for rt in range(RT):
        for kt in range(KT):
            g = w[rt * ROWS:(rt + 1) * ROWS, kt * COLS:(kt + 1) * COLS].reshape(ROWS, 8, 32)
            d = from_bf16(bf16_bits(np.abs(g).max(-1) / 127.0))
            d = np.where(d > 0, d, 1.0)
            q = np.clip(np.rint(g / d[..., None]), -127, 127).astype(np.int8)
            out[rt, kt, 0:512] = np.ascontiguousarray(bf16_bits(d).T).view(np.uint8).reshape(-1)
            out[rt, kt, 512:] = np.ascontiguousarray(q.reshape(ROWS, COLS).T).view(np.uint8).reshape(-1)
    return out


# --- names ------------------------------------------------------------------------------------
GDN = {"attn_qkv": "linear_attn.qkv_proj", "attn_gate": "self_attn.gate_proj",
       "ssm_out": "linear_attn.ssm_out_proj",
       "ssm_alpha": "linear_attn.ssm_alpha_proj", "ssm_beta": "linear_attn.ssm_beta_proj",
       "ssm_conv1d": "linear_attn.ssm_conv1d", "ssm_norm": "linear_attn.ssm_norm",
       "ssm_a": "linear_attn.ssm_a", "ssm_dt": "linear_attn.ssm_dt"}
FULL = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm",
        "attn_k_norm": "self_attn.k_norm"}
COMMON = {"attn_norm": "input_layernorm", "post_attention_norm": "post_attention_layernorm",
          "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
SUFFIX = {"ssm_a": "", "ssm_dt": ".bias"}
KIND = {"ssm_a": "f32", "ssm_dt": "f32", "ssm_conv1d": "conv1d", "ssm_alpha": "q8", "ssm_beta": "q8",
        "ssm_norm": "bf16", "attn_norm": "bf16", "post_attention_norm": "bf16",
        "attn_q_norm": "bf16", "attn_k_norm": "bf16"}


def plan(nl, fai):
    out = []
    for l in range(nl):
        full = (l % fai) == (fai - 1)
        fam = dict(COMMON)
        fam.update(FULL if full else GDN)
        for g, hf in fam.items():
            src = f"blk.{l}.{g}{SUFFIX.get(g, '.weight')}"
            # ssm_a / ssm_dt.bias are suffix-less / .bias in the container too (model.c looks
            # them up by those exact names), so they carry no generic ".weight".
            dst = f"model.layers.{l}.{hf}" + ("" if g == "ssm_a" else ".bias" if g == "ssm_dt" else ".weight")
            out.append((src, dst, KIND.get(g, "tiles"), full, g))
    out.append(("token_embd.weight", "model.embed_tokens.weight", "tiles", True, "emb"))
    out.append(("output_norm.weight", "model.norm.weight", "bf16", True, "norm"))
    if "output.weight" in GB.tensors:
        out.append(("output.weight", "lm_head.weight", "tiles", True, "head"))
    return [t for t in out if t[0] in GB.tensors]


def reorder_idx(g, full, N, hd_g, hd_a):
    """The row gather for this tensor, or None.  hd_g = the GDN head dim, hd_a = attention's."""
    if not REORDER:
        return None
    if g == "attn_qkv" and not full:                     # [q(16) | k(16) | v(48)]
        return perm_rows(NVH, hd_g, off=2 * NKH * hd_g)
    if g == "attn_q" and full:                           # [head][q|gate][hd_a] -> [q|gate]
        return deinterleave_qgate(N, hd_a)
    if g == "attn_gate":                                 # z, the GDN swish gate (48 heads)
        return perm_rows(NVH, hd_g, off=0)
    return None


def config(nl, fai):
    a = lambda k, d=None: GB.kv.get(f"qwen35.{k}", d)  # noqa: E731
    s = lambda k, d=0: int(GB.kv.get(f"qwen35.ssm.{k}", d))  # noqa: E731
    nvh = s("inner_size") // s("state_size")
    c = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "hidden_size": a("embedding_length"),
        "intermediate_size": a("feed_forward_length"),
        "num_hidden_layers": nl,
        "num_attention_heads": a("attention.head_count"),
        "num_key_value_heads": a("attention.head_count_kv"),
        "head_dim": a("attention.key_length"),
        "max_position_embeddings": a("context_length"),
        "rms_norm_eps": float(a("attention.layer_norm_rms_epsilon")),
        "rope_theta": float(a("rope.freq_base")),
        "vocab_size": int(np.prod(GB.tensors["token_embd.weight"][0][1:])),
        "bos_token_id": GB.kv.get("tokenizer.ggml.bos_token_id"),
        "eos_token_id": GB.kv.get("tokenizer.ggml.eos_token_id"),
        "tie_word_embeddings": "output.weight" not in GB.tensors,
        "torch_dtype": "bfloat16",
        "hidden_act": "silu",
        "linear_num_key_heads": s("group_count"),
        "linear_key_head_dim": s("state_size"),
        "linear_value_head_dim": s("state_size"),
        "linear_num_value_heads": nvh,
        "linear_conv_kernel_dim": s("conv_kernel", 4),
        "attn_output_gate": 1,
        "layer_types": ["full_attention" if (i % fai) == (fai - 1) else "linear_attention"
                        for i in range(nl)],
        "partial_rotary_factor": 0.25,
        "q4nx_weight_layout": "rm",
        "q4nx_source": {"converter": "qwen35_to_q4nx.py", "gguf": os.path.basename(GB.path),
                        "v_head_reorder": "hf" if REORDER else "stored"},
    }
    return {k: v for k, v in c.items() if v is not None}


def refuse_unsupported(gb):
    """The inputs this tool would turn into a wrong container, refused by name (engine#2013)."""
    arch = gb.kv.get("general.architecture")
    if arch != "qwen35":
        raise SystemExit(f"{gb.path}: general.architecture is {arch!r}; this tool packs qwen35 only "
                         "(qwen3, qwen3moe and qwen2 are tools/gguf_to_q4nx.py's)")
    rotated = sorted(k for k in gb.kv if k.startswith("prism.hadamard"))
    if rotated:
        raise SystemExit(f"{gb.path}: carries {rotated[0]} ({len(rotated)} prism.hadamard key(s)), so its weights "
                         "are stored in a Hadamard-rotated basis; this tool writes them as they are and no sign "
                         "table, so the container would answer wrongly. Un-rotate the GGUF first")
    tern = [name for name, (_dims, tt, _off) in gb.tensors.items() if tt == PTQ1]
    if tern:
        raise SystemExit(f"{gb.path}: {len(tern)} tensor(s) are ggml type 143, PTQ1_0 (first: {tern[0]}); this tool "
                         "packs Q4_0/Q4_1 codes. Rewrite the file as Q4_0 (tools/ternary_to_q4_0.py) and un-rotate "
                         "it first")
    odd = {}
    for name, (_dims, tt, _off) in gb.tensors.items():
        if tt not in BLK:
            odd.setdefault(tt, name)
    if odd:
        raise SystemExit(f"{gb.path}: " + "; ".join(f"ggml tensor type {tt} (first: {n})" for tt, n in sorted(odd.items()))
                         + " is not one this tool reads (F32, F16, Q4_0, Q4_1, BF16, PTQ1_0)")


def pack(args):
    global GB, REORDER
    GB, REORDER = Gguf(args.gguf), args.reorder
    refuse_unsupported(GB)
    nl = int(GB.kv["qwen35.block_count"])
    fai = int(GB.kv.get("qwen35.full_attention_interval", 4))
    hd_g = int(GB.kv["qwen35.ssm.state_size"])
    hd_a = int(GB.kv["qwen35.attention.key_length"])
    os.makedirs(args.out, exist_ok=True)
    table, blobs, off, done = {}, [], 0, 0

    def emit(name, data, dtype, shape):
        nonlocal off
        b = np.ascontiguousarray(data).tobytes()
        pad = (-off) % 64
        if pad:
            blobs.append(b"\0" * pad)
            off += pad
        table[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)

    for src, dst, kind, full, g in plan(nl, fai):
        dims = GB.tensors[src][0]
        N = int(dims[1]) if len(dims) > 1 else 1
        K = int(dims[0])
        if kind == "tiles":
            codes, scale, zero = GB.q4(src)
            idx = reorder_idx(g, full, N, hd_g, hd_a)
            if idx is not None:
                codes, scale, zero = row_gather(codes, scale, zero, idx)
            elif g == "ssm_out" and REORDER:
                codes, scale, zero = col_gather(codes, scale, zero, hd_g, 0)
            d = tile_q4_1(codes, scale, zero)
            emit(dst, d, "I8", d.shape)
        elif kind == "q8":
            w = GB.f32(src).reshape(N, K)
            if REORDER:
                w = w[head_perm()]
            pad = np.zeros(((N + ROWS - 1) // ROWS * ROWS, K), np.float32)   # the forward's deq_pad
            pad[:N] = w
            d = tile_q8(pad)
            emit(dst, d, "I8", d.shape)
        elif kind == "conv1d":
            ck, cd = int(dims[0]), int(dims[1])
            m = GB.f32(src).reshape(cd, ck).T.copy()                        # [cd][ck] -> [ck][cd]
            if REORDER:
                m = m[:, np.concatenate([np.arange(2 * NKH * hd_g),
                                         perm_cols(NVH, hd_g, 2 * NKH * hd_g)])]
            emit(dst, bf16_bits(m.reshape(-1)), "BF16", [m.size])
        elif kind == "f32":
            v = GB.f32(src)
            emit(dst, (v[head_perm()] if REORDER else v).astype("<f4"), "F32", [v.size])
        else:
            v = GB.f32(src)
            emit(dst, bf16_bits(v), "BF16", [v.size])
        done += 1
        if done % 100 == 0:
            print(f"  {done} tensors, {off / 1e9:.2f} GB", flush=True)

    table["__metadata__"] = {"format": "q4nx", "source": os.path.basename(GB.path),
                             "converter": "qwen35_to_q4nx.py"}
    header = json.dumps(table, separators=(",", ":")).encode()
    header += b" " * ((-len(header)) % 64)
    with open(os.path.join(args.out, "model.q4nx"), "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for b in blobs:
            f.write(b)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config(nl, fai), f, indent=2)
        f.write("\n")
    tok = args.tokenizer or os.path.dirname(os.path.abspath(args.gguf))
    copied = []
    for n in ("tokenizer.json", "tokenizer_config.json"):
        if os.path.exists(os.path.join(tok, n)):
            shutil.copyfile(os.path.join(tok, n), os.path.join(args.out, n))
            copied.append(n)
    if "tokenizer.json" not in copied:
        print(f"WARNING: no tokenizer.json in {tok}; 1bit forward-serve will refuse {args.out}")
    print(f"wrote {args.out}/model.q4nx ({8 + len(header) + off} bytes, {len(table) - 1} tensors)")


def verify(args):
    """Every tile tensor untiles back to code*scale+zero exactly, at the shape the plan asks for."""
    global GB, REORDER
    GB, REORDER = Gguf(args.gguf), True
    refuse_unsupported(GB)
    nl = int(GB.kv["qwen35.block_count"])
    fai = int(GB.kv.get("qwen35.full_attention_interval", 4))
    hd_g = int(GB.kv["qwen35.ssm.state_size"])
    hd_a = int(GB.kv["qwen35.attention.key_length"])
    with open(os.path.join(args.out, "model.q4nx"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
    mm = np.memmap(os.path.join(args.out, "model.q4nx"), np.uint8, mode="r")
    bad = 0
    for src, dst, kind, full, g in plan(nl, fai):
        lo, hi = hdr[dst]["data_offsets"]
        raw = np.asarray(mm[base + lo:base + hi])
        dims = GB.tensors[src][0]
        N = int(dims[1]) if len(dims) > 1 else 1
        K = int(dims[0])
        if kind == "tiles":
            # reference: the GGUF's own dequant, reordered the same way, with bf16-rounded scale/zero
            codes, scale, zero = GB.q4(src)
            idx = reorder_idx(g, full, N, hd_g, hd_a)
            if idx is not None:
                codes, scale, zero = row_gather(codes, scale, zero, idx)
            elif g == "ssm_out":
                codes, scale, zero = col_gather(codes, scale, zero, hd_g, 0)
            ref = (codes.reshape(N, K // GROUP, GROUP) * from_bf16(bf16_bits(scale))[..., None]
                   + from_bf16(bf16_bits(zero))[..., None]).reshape(N, K)
            err = np.abs(untile_q4_1(raw, N, K) - ref)
            ok, note = err.max() == 0, f"exact, {hdr[dst]['shape']}"
        elif kind == "f32":
            got = np.frombuffer(raw.tobytes(), "<f4")
            ref = GB.f32(src)
            exp = ref[head_perm()] if REORDER else ref
            ok, note = np.array_equal(got, exp), "f32 exact"
        elif kind == "conv1d":
            ck, cd = int(dims[0]), int(dims[1])
            got = from_bf16(raw.view("<u2")).reshape(ck, cd)
            m = GB.f32(src).reshape(cd, ck).T.copy()             # [cd][ck] -> [ck][cd]
            exp = m[:, np.concatenate([np.arange(2 * NKH * hd_g),
                                       perm_cols(NVH, hd_g, 2 * NKH * hd_g)])] if REORDER else m
            exact = np.array_equal(got, from_bf16(bf16_bits(exp)))
            # the v-band permutation's DIRECTION, checked independently of the packer's gather:
            # Both sides are summed from bf16-rounded values, in f64: a head's sum can cancel to near zero, and
            # a relative tolerance between a rounded and an unrounded sum then fails a correct container.
            gb = got[:, 2 * NKH * hd_g:].astype(np.float64).reshape(ck, NVH, hd_g).sum(-1)
            rb = from_bf16(bf16_bits(m))[:, 2 * NKH * hd_g:].astype(np.float64).reshape(ck, NVH, hd_g).sum(-1)
            hf = np.array_equal(gb, rb[:, head_perm()])
            ok = exact and hf
            where = "hf" if hf else "stored" if np.array_equal(gb, rb) else "neither"
            note = f"bf16 {'exact' if exact else 'MISMATCH'}, v-band direction {where}"
        elif kind == "bf16":
            got, ref = from_bf16(raw.view("<u2")), GB.f32(src)
            rel = np.max(np.abs(got - ref) / (np.abs(ref) + 1e-30))
            ok, note = rel <= 2 ** -8 + 1e-6, f"bf16 max rel {rel:.3g}"
        else:
            ok, note = got_q8(raw, src, N, K), "Q8 (checked separately)"
        bad += not ok
        if not ok or ".layers.0." in dst or ".layers.3." in dst or ".layers." not in dst:
            print(f"{'ok ' if ok else 'BAD'} {dst:52s} {note}")
    print(f"{bad} bad tensors")
    return 1 if bad else 0


def got_q8(raw, src, N, K):
    """Q8 tiles back to f32, compared against the padded, reordered BF16 source."""
    w = GB.f32(src).reshape(N, K)
    if REORDER:
        w = w[head_perm()]
    pad = np.zeros(((N + ROWS - 1) // ROWS * ROWS, K), np.float32)
    pad[:N] = w
    got = np.empty_like(pad)
    rt, ktn = pad.shape[0] // ROWS, pad.shape[1] // COLS
    t = raw.reshape(rt, ktn, 8704)
    for r in range(rt):
        for k in range(ktn):
            d = from_bf16(t[r, k, :512].copy().view("<u2")).reshape(8, ROWS)
            q = t[r, k, 512:].copy().view(np.int8).reshape(COLS, ROWS).T
            got[r * ROWS:(r + 1) * ROWS, k * COLS:(k + 1) * COLS] = \
                (q.reshape(ROWS, 8, 32) * d.T[..., None]).reshape(ROWS, COLS)
    rel = np.abs(got[:N] - w).max() / (np.abs(w).max() + 1e-30)
    print(f"     Q8 {src}: max abs err {np.abs(got[:N] - w).max():.3g} (rel to max {rel:.2e})")
    return rel < 0.01


def selftest(_args=None):
    """The tile round-trips, the head permutation and the q|gate de-interleave, on small data."""
    rng = np.random.default_rng(0)
    N, K = 64, 512
    codes = rng.integers(0, 16, (N, K), dtype=np.uint8)
    scale = rng.random((N, K // GROUP)).astype(np.float32)
    zero = (rng.random((N, K // GROUP)) - 0.5).astype(np.float32)
    ref = (codes.reshape(N, K // GROUP, GROUP) * from_bf16(bf16_bits(scale))[..., None]
           + from_bf16(bf16_bits(zero))[..., None]).reshape(N, K)
    got = untile_q4_1(tile_q4_1(codes, scale, zero), N, K)
    assert np.array_equal(got, ref), "q4_1 tile round-trip"
    p = head_perm()
    assert sorted(p.tolist()) == list(range(NVH)), "head_perm is not a bijection"
    assert all(p[j] == (j % 3) * NKH + j // 3 for j in range(NVH)), "head_perm formula"
    idx = perm_rows(NVH, 128, off=4096)
    assert idx[:4096].tolist() == list(range(4096)) and len(idx) == 4096 + NVH * 128
    assert sorted(idx[4096:].tolist()) == list(range(4096, 4096 + NVH * 128)), "perm_rows tail"
    q = deinterleave_qgate(2 * 24 * 256, 256)
    assert sorted(q.tolist()) == list(range(12288)), "deinterleave is not a bijection"
    for h in (0, 7):                                     # head h, part p, dim d  ->  p, h, d
        for part in (0, 1):
            src = h * 512 + part * 256 + 3
            assert q[part * 6144 + h * 256 + 3] == src, "deinterleave mapping"
    w = rng.standard_normal((64, 512)).astype(np.float32)
    back = np.zeros_like(w)
    t = tile_q8(w)
    for r in range(2):
        for k in range(2):
            d = from_bf16(t[r, k, :512].copy().view("<u2")).reshape(8, ROWS)
            qq = t[r, k, 512:].copy().view(np.int8).reshape(COLS, ROWS).T
            back[r * 32:(r + 1) * 32, k * 256:(k + 1) * 256] = \
                (qq.reshape(ROWS, 8, 32) * d.T[..., None]).reshape(ROWS, COLS)
    rel = np.abs(back - w).max() / np.abs(w).max()
    assert rel < 0.01, f"q8 round-trip {rel}"
    print(f"selftest ok (q4_1 exact, q8 max rel {rel:.4f}, head_perm/deinterleave bijective)")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pack")
    a.add_argument("gguf")
    a.add_argument("out")
    a.add_argument("--tokenizer")
    a.add_argument("--no-reorder", dest="reorder", action="store_false",
                   help="keep the stored (rep-major) v-head order")
    b = sub.add_parser("verify")
    b.add_argument("gguf")
    b.add_argument("out")
    sub.add_parser("selftest")
    args = p.parse_args()
    return {"pack": pack, "verify": verify, "selftest": selftest}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
