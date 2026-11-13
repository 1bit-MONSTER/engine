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
"""engine#1786: the bidirectional gemma3 encoder (FastFlowLM's Embedding-Gemma-300M-NPU2) through the per-op forward.

usage: tests/embed_forward_ci.py <embed_pooled_test binary> (needs numpy)

Writes a two-layer container in that pack's own layout -- every projection BF16 [out, in], the sentence-transformers
names with no `model.` prefix (`layers.N.*`, `embed_tokens`, `norm`, `dense_2`, `dense_3`), the embedding stored
pre-scaled by sqrt(H), norms as HF's raw w, `use_bidirectional_attention: true`, a sliding layer then a full one, and an
intermediate size off the 256 grid (384 -> 512) -- and holds Q4nxNpuForward::embed_pooled() to a numpy Gemma3 encoder
on the same weights: (1 + w) RMSNorm everywhere, q/k norms, rotate-half RoPE on the whole head (base 10000 on the
sliding layer, rope_theta on the full one), query_pre_attn_scalar, attention over EVERY position, the sandwich norms,
GELU(tanh) gated MLP, mean pooling, dense_2, dense_3, L2. The real pack against transformers' Gemma3TextModel scored
cos 0.99999 on the host; any one of those rules wrong moves this one far off (the norm rule alone: 0.09 on the real
pack, the sliding layers left causal: 0.70 to 0.90). A second request one token past half the window must be refused.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

H, NH, NKV, HD, IM, V, D2 = 256, 2, 1, 128, 384, 96, 512
WIN = 16                   # sliding window: inputs of up to WIN // 2 tokens
QPAS = 128.0               # query_pre_attn_scalar
THETA, LOCAL = 1000000.0, 10000.0
EPS = 1e-6
MIN_COS = 0.99999


def tmp_root():
    for d in (os.environ.get("ONEBIT_SCRATCH"), os.path.expanduser("~/.cache"), None):
        if d is None or os.path.isdir(d):
            return d
    return None


def bf16(np, a):
    """Round to bf16 (nearest even) and return (the raw uint16 words, the float32 values they stand for)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return r, (r.astype(np.uint32) << 16).view(np.float32)


def write_container(path, arrays):
    header, blobs, off = {}, [], 0
    for name, raw in arrays.items():
        b = raw.tobytes()
        header[name] = {"dtype": "BF16", "shape": list(raw.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((-len(h)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def make(d, np):
    rng = np.random.default_rng(1786)
    W, raw = {}, {}

    def put(name, a):
        r, v = bf16(np, a)
        raw[name], W[name] = r, v.reshape(a.shape)

    put("embed_tokens.weight", rng.standard_normal((V, H)).astype(np.float32) * 0.5 * H ** 0.5)   # stored pre-scaled
    put("norm.weight", rng.standard_normal(H).astype(np.float32) * 0.2)
    for l in range(2):
        p = f"layers.{l}."
        for n, shape, sc in (("self_attn.q_proj.weight", (NH * HD, H), 0.06), ("self_attn.k_proj.weight", (NKV * HD, H), 0.06),
                             ("self_attn.v_proj.weight", (NKV * HD, H), 0.06), ("self_attn.o_proj.weight", (H, NH * HD), 0.06),
                             ("mlp.gate_proj.weight", (IM, H), 0.06), ("mlp.up_proj.weight", (IM, H), 0.06),
                             ("mlp.down_proj.weight", (H, IM), 0.06)):
            put(p + n, rng.standard_normal(shape).astype(np.float32) * sc)
        for n, size in (("input_layernorm.weight", H), ("post_attention_layernorm.weight", H),
                        ("pre_feedforward_layernorm.weight", H), ("post_feedforward_layernorm.weight", H),
                        ("self_attn.q_norm.weight", HD), ("self_attn.k_norm.weight", HD)):
            put(p + n, rng.standard_normal(size).astype(np.float32) * 0.2)
    put("dense_2.weight", rng.standard_normal((D2, H)).astype(np.float32) * 0.05)
    put("dense_3.weight", rng.standard_normal((H, D2)).astype(np.float32) * 0.05)
    os.makedirs(d)
    write_container(os.path.join(d, "model.q4nx"), raw)
    cfg = {"architectures": ["Gemma3TextModel"], "model_type": "gemma3_text", "hidden_size": H,
           "num_hidden_layers": 2, "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD,
           "intermediate_size": IM, "vocab_size": V, "rms_norm_eps": EPS, "rope_theta": THETA,
           "rope_local_base_freq": LOCAL, "sliding_window": WIN, "query_pre_attn_scalar": QPAS,
           "hidden_activation": "gelu_pytorch_tanh", "layer_types": ["sliding_attention", "full_attention"],
           "use_bidirectional_attention": True, "max_position_embeddings": 64}
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    with open(os.path.join(d, "tokenizer.json"), "w") as f:
        f.write("{}")
    return W


def reference(np, W, ids):
    def rms(x, w):
        x = x.astype(np.float64)
        return (x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * (1.0 + w)).astype(np.float32)

    def rope(x, base):           # x [T, heads, HD], rotate-half on the whole head
        inv = 1.0 / base ** (np.arange(0, HD, 2) / HD)
        fr = np.arange(x.shape[0])[:, None] * inv[None, :]
        c, s = np.cos(np.concatenate([fr, fr], -1))[:, None], np.sin(np.concatenate([fr, fr], -1))[:, None]
        x = x.astype(np.float64)
        return (x * c + np.concatenate([-x[..., HD // 2:], x[..., :HD // 2]], -1) * s).astype(np.float32)

    gelu = lambda a: 0.5 * a * (1.0 + np.tanh(0.7978845608028654 * (a + 0.044715 * a ** 3)))
    T = len(ids)
    x = W["embed_tokens.weight"][ids].astype(np.float32)
    for l, base in ((0, LOCAL), (1, THETA)):
        p = f"layers.{l}."
        h = rms(x, W[p + "input_layernorm.weight"])
        q = rms((h @ W[p + "self_attn.q_proj.weight"].T).reshape(T, NH, HD), W[p + "self_attn.q_norm.weight"])
        k = rms((h @ W[p + "self_attn.k_proj.weight"].T).reshape(T, NKV, HD), W[p + "self_attn.k_norm.weight"])
        v = (h @ W[p + "self_attn.v_proj.weight"].T).reshape(T, NKV, HD)
        q, k = rope(q, base), rope(k, base)
        k, v = np.repeat(k, NH // NKV, 1), np.repeat(v, NH // NKV, 1)
        sc = np.einsum("thd,shd->hts", q, k) / np.sqrt(QPAS)          # no mask: every position, both ways
        sc = np.exp(sc - sc.max(-1, keepdims=True))
        sc /= sc.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", sc, v).reshape(T, NH * HD) @ W[p + "self_attn.o_proj.weight"].T
        x = x + rms(o, W[p + "post_attention_layernorm.weight"])
        h = rms(x, W[p + "pre_feedforward_layernorm.weight"])
        f = (gelu(h @ W[p + "mlp.gate_proj.weight"].T) * (h @ W[p + "mlp.up_proj.weight"].T)) @ W[p + "mlp.down_proj.weight"].T
        x = x + rms(f, W[p + "post_feedforward_layernorm.weight"])
    pooled = rms(x, W["norm.weight"]).mean(0)
    e = pooled @ W["dense_2.weight"].T @ W["dense_3.weight"].T
    return e / np.linalg.norm(e)


def run(binary, d, ids, out):
    env = dict(os.environ, ONEBIT_TEST_Q4NX=d, ONEBIT_TEST_IDS=" ".join(map(str, ids)), ONEBIT_TEST_EMBED_OUT=out)
    env.pop("NPU_INFER_HOST_GEMM", None)
    return subprocess.run([binary], env=env, capture_output=True, text=True, timeout=110)


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2])
        return 2
    try:
        import numpy as np
    except ImportError:
        print("SKIP: numpy is not installed")
        return 77
    with tempfile.TemporaryDirectory(prefix="embed_forward_ci.", dir=tmp_root()) as tmp:
        d, out = os.path.join(tmp, "enc"), os.path.join(tmp, "vec.f32")
        W = make(d, np)
        ids = [2, 17, 40, 5, 63, 1]
        r = run(sys.argv[1], d, ids, out)
        if r.returncode != 0:
            print(f"FAIL  embed_pooled_test rc={r.returncode}: {(r.stdout + r.stderr)[-800:]}")
            return 1
        ours = np.fromfile(out, dtype=np.float32)
        ref = reference(np, W, ids)
        cos = float(ours @ ref) if ours.shape == ref.shape else float("nan")
        ok = ours.shape == ref.shape and cos >= MIN_COS
        print(f"{'PASS' if ok else 'FAIL'}  embed_pooled vs the numpy bidirectional gemma3 encoder: cos {cos:.7f} "
              f"(needs {MIN_COS}), {ours.size} floats, |v| {float(np.linalg.norm(ours)):.6f}")
        too_long = list(range(2, 2 + WIN // 2 + 1))
        r2 = run(sys.argv[1], d, too_long, out)
        refused = r2.returncode == 3 and "half the" in r2.stdout
        print(f"{'PASS' if refused else 'FAIL'}  a {len(too_long)}-token input (window {WIN}) is refused by name: "
              f"{r2.stdout.strip()[:160]}")
        return 0 if ok and refused else 1


if __name__ == "__main__":
    sys.exit(main())
