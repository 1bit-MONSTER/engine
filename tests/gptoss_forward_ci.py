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
"""The per-op forward on a GPT-OSS container made here, against a numpy GPT-OSS forward of the same bytes.

usage: tests/gptoss_forward_ci.py <batch_check_state_test binary>        (needs numpy)

FastFlowLM's GPT-OSS-20B-NPU2 stores its attention and head as D2560 chunks and its experts as MX2560 (MXFP4) chunks
(npu/forward/include/q4nx_dequant.h), and the model adds attention sinks, an o_proj bias, YaRN RoPE, a sliding window
on alternate layers, and a top-k MoE with expert biases and a clamped SwiGLU (engine#1786). CI has no such pack, so
this writes a two-layer one (hidden 256, 4 heads of 64 on 2 kv heads, 4 experts top 2, ffn 256, vocab 256, sliding
window 2) in that container format, runs batch_check_state_test on it in host-GEMM mode with ONEBIT_TEST_LOGITS_OUT,
and compares the forward's logits after tokens 1..6 with gptoss_reference() below. The reference is transcribed from
our llama.cpp fork (src/models/openai-moe.cpp, build_moe_ffn's SOFTMAX_WEIGHT gating, ggml-cpu's swiglu_oai) and
transformers' _compute_yarn_parameters; its chunk readers follow ggml's MXFP4 (E2M1 * 2^(e-127)).
Exits 77 (skip) without numpy.
"""
import json
import math
import os
import struct
import subprocess
import sys
import tempfile

H, NH, NKV, HD, IM, E, TOPK, V, LAYERS, WIN = 256, 4, 2, 64, 256, 4, 2, 256, 2, 2
MIN_CORR, MAX_DIFF = 0.99999, 1e-3          # the forward against the reference; measured in the PR (engine#1786)
LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def bf16_bytes(np, a):
    u = np.asarray(a, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2").tobytes()


def bf16_round(np, a):
    u = np.asarray(a, np.float32).view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)


def chunk_codes(np, rng):
    """2048 bytes of 4-bit codes for one 32 x 128 chunk, and the [32][128] code grid they encode."""
    q = rng.integers(0, 16, size=(32, 128))
    b = np.zeros(2048, np.uint8)
    for r in range(32):
        for c in range(128):
            i = (r // 16) * 1024 + c * 8 + (r % 16) // 2
            b[i] |= (q[r, c] << 4) if (r & 1) else q[r, c]
    return b, q


def d2560_tensor(np, rng, rows, cols, S=4):
    """[RB, CB, S, 2560] bytes for a rows x cols tensor, and its dequantised values (rows padded to the grid)."""
    RB, CB = -(-rows // (S * 32)), -(-cols // 128)
    data = bytearray()
    w = np.zeros((RB * S * 32, CB * 128), np.float32)
    for rb in range(RB):
        for cb in range(CB):
            for s in range(S):
                sc = bf16_round(np, rng.uniform(0.004, 0.012, size=(4, 32)))       # [g, r]
                zp = bf16_round(np, rng.uniform(-0.09, -0.03, size=(4, 32)))
                codes, q = chunk_codes(np, rng)
                hdr = bf16_bytes(np, sc.reshape(-1)) + bf16_bytes(np, zp.reshape(-1))
                data += hdr + codes.tobytes()
                r0 = (rb * S + s) * 32
                for g in range(4):
                    w[r0:r0 + 32, cb * 128 + g * 32: cb * 128 + g * 32 + 32] = \
                        q[:, g * 32:(g + 1) * 32] * sc[g][:, None] + zp[g][:, None]
    return bytes(data), [RB, CB, S, 2560], w[:rows, :cols]


def mx2560_expert(np, rng, S=4):
    """One expert: 3*RB row-blocks (gate/up alternating by 128 rows, then down) x CB, as MX2560 bytes."""
    RB, CB = -(-max(IM, H) // 128), -(-max(IM, H) // 128)
    data = bytearray()
    w = np.zeros((3 * RB * 128, CB * 128), np.float32)
    lut = np.array(LUT, np.float32)
    for bi in range(3 * RB):
        for cb in range(CB):
            for s in range(S):
                ex = rng.integers(119, 123, size=(4, 32)).astype(np.uint8)       # [g, r], scales 2^-8 .. 2^-5
                codes, q = chunk_codes(np, rng)
                data += ex.reshape(-1).tobytes() + bytes(384) + codes.tobytes()
                r0 = bi * 128 + s * 32
                for g in range(4):
                    w[r0:r0 + 32, cb * 128 + g * 32: cb * 128 + g * 32 + 32] = \
                        lut[q[:, g * 32:(g + 1) * 32]] * (2.0 ** (ex[g].astype(np.float64) - 127))[:, None]
    gate = np.concatenate([w[i * 256:i * 256 + 128] for i in range(RB)])[:IM, :H]
    up = np.concatenate([w[i * 256 + 128:i * 256 + 256] for i in range(RB)])[:IM, :H]
    down = w[2 * RB * 128: 2 * RB * 128 + H, :IM]
    return bytes(data), [3 * RB, CB, S, 2560], gate, up, down


def make_model(d, np):
    rng = np.random.default_rng(1786)
    tensors, W = [], {}

    def add(name, dtype, shape, blob):
        tensors.append((name, dtype, shape, blob))

    def bf(name, a):
        a = bf16_round(np, a); W[name] = a
        add(name, "BF16", list(a.shape), bf16_bytes(np, a.reshape(-1)))

    bf("model.embed_tokens.weight", rng.standard_normal((V, H)) * 0.5)
    bf("model.norm.weight", 1.0 + 0.1 * rng.standard_normal(H))
    for l in range(LAYERS):
        p = f"model.layers.{l}."
        bf(p + "input_layernorm.weight", 1.0 + 0.1 * rng.standard_normal(H))
        bf(p + "post_attention_layernorm.weight", 1.0 + 0.1 * rng.standard_normal(H))
        for nm, rows, cols in (("q_proj", NH * HD, H), ("k_proj", NKV * HD, H), ("v_proj", NKV * HD, H),
                               ("o_proj", H, NH * HD)):
            blob, shape, w = d2560_tensor(np, rng, rows, cols)
            W[p + f"self_attn.{nm}.weight"] = w
            add(p + f"self_attn.{nm}.weight", "I8", shape, blob)
        bf(p + "self_attn.q_proj.bias", 0.3 * rng.standard_normal(NH * HD))
        bf(p + "self_attn.k_proj.bias", 0.3 * rng.standard_normal(NKV * HD))
        bf(p + "self_attn.v_proj.bias", 0.3 * rng.standard_normal(NKV * HD))
        bf(p + "self_attn.o_proj.bias", 0.3 * rng.standard_normal(H))
        bf(p + "self_attn.sinks.weight", rng.uniform(-1.0, 2.0, NH))
        # the router is K-padded in FastFlowLM's packs ([E][2944] for H 2880): pad it here too, by 128
        rw = np.zeros((E, H + 128), np.float32); rw[:, :H] = 0.3 * rng.standard_normal((E, H))
        bf(p + "mlp.router.weight_prefill", rw)
        bf(p + "mlp.router.bias", 0.5 * rng.standard_normal(E))
        # large enough that min(gate, 7) and clamp(up, -7, 7) bind on part of the ffn (the clamp is a rule too)
        bf(p + "mlp.experts.gate_proj_bias", 6.0 * rng.standard_normal((E, IM)))
        bf(p + "mlp.experts.up_proj_bias", 6.0 * rng.standard_normal((E, IM)))
        bf(p + "mlp.experts.down_proj_bias", 0.3 * rng.standard_normal((E, H)))
        blobs, ws = [], []
        for e in range(E):
            blob, shape, g, u, dn = mx2560_expert(np, rng)
            blobs.append(blob); ws.append((g, u, dn))
        W[p + "experts"] = ws
        add(p + "ffn_gate_up_down_exps.weight", "U8", [E] + shape, b"".join(blobs))
    blob, shape, w = d2560_tensor(np, rng, V, H)
    W["lm_head.weight"] = w
    add("lm_head.weight", "I8", shape, blob)

    header, off = {}, 0
    for name, dtype, shape, blob in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(blob)]}
        off += len(blob)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((-len(h)) % 8)
    os.makedirs(d)
    with open(os.path.join(d, "model.q4nx"), "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h)
        for _, _, _, blob in tensors:
            f.write(blob)
    cfg = {"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss", "hidden_size": H,
           "num_hidden_layers": LAYERS, "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD,
           "intermediate_size": IM, "vocab_size": V, "num_local_experts": E, "num_experts_per_tok": TOPK,
           "experts_per_token": TOPK, "rms_norm_eps": 1e-5, "rope_theta": 150000, "attention_bias": True,
           # GPT-OSS's own block but a short original window: with 6 positions, 4096 would leave every pair that
           # yarn changes unrotated and the rule untested; at 16 the ramp sits on pairs that turn within 6 steps
           "rope_scaling": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 32.0,
                            "original_max_position_embeddings": 16, "rope_type": "yarn", "truncate": False},
           "sliding_window": WIN, "layer_types": ["sliding_attention", "full_attention"],
           "tie_word_embeddings": False, "max_position_embeddings": 64, "hidden_act": "silu"}
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    with open(os.path.join(d, "tokenizer.json"), "w") as f:
        f.write("{}")
    with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
        json.dump({"add_bos_token": False}, f)
    return W, cfg


def gptoss_reference(np, W, cfg, ids):
    T, eps = len(ids), cfg["rms_norm_eps"]
    rs, base = cfg["rope_scaling"], cfg["rope_theta"]

    def corr(nrot):
        return (HD * math.log(rs["original_max_position_embeddings"] / (nrot * 2 * math.pi))) / (2 * math.log(base))
    low, high = max(corr(rs["beta_fast"]), 0), min(corr(rs["beta_slow"]), HD - 1)
    ramp = np.clip((np.arange(HD // 2) - low) / (high - low), 0, 1)
    extra = 1.0 / base ** (np.arange(0, HD, 2) / HD)
    inv = (extra / rs["factor"]) * ramp + extra * (1 - ramp)
    att = 0.1 * math.log(rs["factor"]) + 1.0
    fr = np.arange(T)[:, None] * inv[None, :]
    cos = (np.cos(np.concatenate([fr, fr], -1)) * att)[:, None]
    sin = (np.sin(np.concatenate([fr, fr], -1)) * att)[:, None]

    def rope(x):
        x = x.astype(np.float64)
        return (x * cos + np.concatenate([-x[..., HD // 2:], x[..., :HD // 2]], -1) * sin).astype(np.float32)

    def rms(x, w):
        return w * (x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)).astype(np.float32)

    x = W["model.embed_tokens.weight"][ids].astype(np.float32)
    for l, lt in enumerate(cfg["layer_types"]):
        p = f"model.layers.{l}."
        xn = rms(x, W[p + "input_layernorm.weight"])
        q = (xn @ W[p + "self_attn.q_proj.weight"].T + W[p + "self_attn.q_proj.bias"]).reshape(T, NH, HD)
        k = (xn @ W[p + "self_attn.k_proj.weight"].T + W[p + "self_attn.k_proj.bias"]).reshape(T, NKV, HD)
        v = (xn @ W[p + "self_attn.v_proj.weight"].T + W[p + "self_attn.v_proj.bias"]).reshape(T, NKV, HD)
        q, k = rope(q), rope(k)
        k, v = np.repeat(k, NH // NKV, 1), np.repeat(v, NH // NKV, 1)
        sc = np.einsum("thd,shd->hts", q, k) / math.sqrt(HD)
        mask = np.tril(np.ones((T, T), bool))
        if lt == "sliding_attention":
            mask &= (np.arange(T)[:, None] - np.arange(T)[None, :]) < WIN
        sc = np.where(mask[None], sc, -np.inf)
        sink = W[p + "self_attn.sinks.weight"][:, None, None]
        m = np.maximum(sc.max(-1, keepdims=True), sink)
        ex = np.exp(sc - m)
        ctx = np.einsum("hts,shd->thd", ex / (ex.sum(-1, keepdims=True) + np.exp(sink - m)), v)
        x = x + ctx.reshape(T, NH * HD) @ W[p + "self_attn.o_proj.weight"].T + W[p + "self_attn.o_proj.bias"]
        h = rms(x, W[p + "post_attention_layernorm.weight"])
        logits = h @ W[p + "mlp.router.weight_prefill"][:, :H].T + W[p + "mlp.router.bias"]
        out = np.zeros_like(x)
        for t in range(T):
            sel = sorted(range(E), key=lambda e: (-logits[t, e], e))[:TOPK]
            wv = np.exp(logits[t, sel] - logits[t, sel].max()); wv /= wv.sum()
            for e, wt in zip(sel, wv):
                g, u, dn = W[p + "experts"][e]
                gg = h[t] @ g.T + W[p + "mlp.experts.gate_proj_bias"][e]
                uu = h[t] @ u.T + W[p + "mlp.experts.up_proj_bias"][e]
                xg, yu = np.minimum(gg, 7.0), np.clip(uu, -7.0, 7.0)
                act = xg / (1.0 + np.exp(-1.702 * xg)) * (yu + 1.0)
                out[t] += wt * (act @ dn.T + W[p + "mlp.experts.down_proj_bias"][e])
        x = x + out
    return rms(x, W["model.norm.weight"])[-1] @ W["lm_head.weight"].T


def tmp_root():
    """tests/tmp_root.h's rule (engine#2029): scratch on disk, not tmpfs."""
    e = os.environ.get("ONEBIT_TEST_TMP")
    if e:
        return e
    h = os.environ.get("HOME")
    if h:
        p = os.path.join(h, ".cache", "scratch-tests", "tmp")
        try:
            os.makedirs(p, exist_ok=True)
            return p
        except OSError:
            pass
    return None


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2])
        return 2
    try:
        import numpy as np
    except ImportError:
        print("SKIP: numpy is not installed")
        return 77
    with tempfile.TemporaryDirectory(prefix="gptoss_forward_ci.", dir=tmp_root()) as tmp:
        d = os.path.join(tmp, "q4nx")
        W, cfg = make_model(d, np)
        logits_out = os.path.join(tmp, "sequential.f32")
        env = dict(os.environ, ONEBIT_TEST_Q4NX=d, ONEBIT_TEST_LOGITS_OUT=logits_out)
        env.pop("NPU_INFER_HOST_GEMM", None)
        r = subprocess.run([sys.argv[1]], env=env, capture_output=True, text=True, timeout=110)
        for line in (r.stdout + r.stderr).splitlines():
            if line.startswith(("PASS", "FAIL", "SKIP", "ALL PASS", "FAILED")) or "gpt_oss" in line:
                print(line)
        if r.returncode == 77:
            print("FAIL  batch_check_state_test skipped the fabricated container (init_host refused it)")
            return 1
        if r.returncode != 0:
            return r.returncode
        ours = np.fromfile(logits_out, dtype=np.float32)[:V]
        ref = gptoss_reference(np, W, cfg, [1, 2, 3, 4, 5, 6])
        c = float(np.corrcoef(ours, ref)[0, 1])
        dmax, rmax = float(np.abs(ours - ref).max()), float(np.abs(ref).max())
        ok = c >= MIN_CORR and dmax <= MAX_DIFF * rmax and int(np.argmax(ours)) == int(np.argmax(ref))
        print(f"{'PASS' if ok else 'FAIL'}  sequential logits vs the GPT-OSS reference on the container's own bytes: "
              f"corr {c:.7f} (needs {MIN_CORR}), max |d| {dmax:.3g} of {rmax:.3g}, argmax {int(np.argmax(ours))} vs "
              f"{int(np.argmax(ref))}")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
