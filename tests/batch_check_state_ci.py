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
"""The batch-check state pin (tests/batch_check_state_test.cpp, engine#1905/#1946) on a container made here.

usage: tests/batch_check_state_ci.py <batch_check_state_test binary> [lfm2|lfm3|qwen2|qwen3_5] (needs numpy)

batch_check_state_test needs a Q4NX container, and CI has none, so on its own it skips (77) there and a regression of
#1909 -- the NPU_BATCH_CHECK oracle resetting the prefix it replays from -- lands green. This writes a two-layer qwen3
(hidden 256, two 128-wide query heads on one kv head, ffn 512, vocab 256; the smallest geometry the I8 tiles take whole),
packs it with tools/safetensors_to_q4nx.py, and runs the test on it in host-GEMM mode: no device, no model on disk.

Measured when it was written (engine#1946): passes on main; with #1909 reverted it fails 3 of 4 arms (the forward
holds 2 and 1 positions instead of 5 and 6, last-chunk rel-L1 0.78) -- the same failure the real containers show.
Weights are seeded, so the run is the same every time. Exits 77 (skip) without numpy.

With `lfm2` the container is an LFM2-shaped hybrid instead (engine#1786): three layers, a gated short convolution, an
attention layer with q/k norms, another short convolution, `model_type: lfm2`, `norm_eps` and no `layer_types`. It is
the only place CI loads a container with conv layers through the forward. What it pins: the container loads at all
(on 2026-11-11 a rule added for Hunyuan, "q/k norm is on and this layer has no q_norm, refuse", refused every LFM2
pack at layer 0, a conv layer with no attention, and the suite was green with that in it), and a batched prefill of
the conv layers leaves the logits and the state a token-by-token run leaves, which is what the test binary compares.

With `lfm3` it is an LFM3-shaped hybrid (engine#1786): conv, full attention, conv, sliding attention (window 2),
written under the UPSTREAM Liquid names (conv.*, operator_norm, ffn_norm, feed_forward.w1/w2/w3, post_operator_norm,
post_ffn_norm, embedding_norm) as FastFlowLM's LFM3-1.7B-NPU2 pack keeps them, with `layer_types` naming "conv",
the zero-tail partial RoPE on the full layer, the sandwich norms, and a bf16 per-head output gate
(self_attn.gate_proj [heads, hidden]). What it pins: model_load's renaming, the forward accepting "conv" in
layer_types, and the batched prefill running the gate and the sandwich exactly as the token-by-token path does.
batched == sequential compares the forward with itself, so this arm also runs LFM3's math in numpy (lfm3_reference,
transcribed from modeling_lfm3.py) on the container's own weights, untiled, and compares the sequential logits:
measured at engine#2254, corr 1.0000000 and max |d| 1.2e-05 of 7.63. A dead head gate (in both paths) gives corr 0.90
and another argmax, the usual partial RoPE 0.98, no sandwich 0.38; every one of them passes batched == sequential.

With `qwen2` it is a two-layer Qwen2 (engine#1786): q/k/v projection biases, no q/k norms, and a config.json with no
`attention_bias` key, which is how every Qwen2 config is written. With the key absent the forward loaded the biases
and never added them, so FastFlowLM's Qwen2.5-3B-Instruct-NPU2 answered noise. batched == sequential cannot see that
(both paths skip the biases alike), so the arm also compares the sequential logits with a numpy Qwen2 forward on the
container's own weights (qwen2_reference). Measured at the fix: see QWEN2_* below.

With `qwen3_5` it is a qwen3_5 hybrid (engine#1889): a linear-attention (GDN) layer, then a full-attention layer with
q/k norms, written under `language_model.*` so the packer takes its qwen3_5 path (renames, the conv1d transpose,
F32 `ssm_a` / `ssm_dt.bias`). The same rule, scoped to "not a conv layer", still refused every qwen3_5 hybrid: a GDN
layer has no self_attn.q_norm either, and layer 0 of every such pack is one. The one test that loads a real
Qwen3.5 pack skips when the pack is not on disk, which in CI is always, so that too was green.
The fabricated geometry is deliberately OFF the kernel's N grid (the fused q|k|v is 768 wide and its design's N is
1024; the gate projection is 256 and 512), which no shipped qwen3_5 pack is. That is what found the second defect
this row pins: the batched GDN path read row m of a projection at m * (model width) while the projection returns its
rows at the design's N, so a batched prefill of two or more tokens left other logits and another state than the
token-by-token path (rel-L1 0.6 to 0.7 here, exact for one-row batches, exact on every real pack).
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools", "safetensors_to_q4nx.py")

H, HEADS, KV_HEADS, HEAD_DIM, FFN, VOCAB, LAYERS = 256, 2, 1, 128, 512, 256, 2
LFM3_MIN_CORR = 0.99999      # measured at #2254: corr 1.0000000, max |d| 1.2e-05 of 7.63
LFM3_MAX_DIFF = 1e-3         # of the reference's largest |logit|


def write_safetensors(path, arrays):
    """(name -> float32 numpy array) -> a safetensors file, dense row-major."""
    header, blobs, off = {}, [], 0
    for name, a in arrays.items():
        raw = a.astype("<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((-len(h)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def make_model(d, np, family="qwen3"):
    rng = np.random.default_rng(1946)
    lfm2 = family == "lfm2"
    lfm3 = family == "lfm3"
    qwen2 = family == "qwen2"
    q35 = family == "qwen3_5"
    layers = 3 if lfm2 else (4 if lfm3 else LAYERS)
    conv_layers = (0, 2) if (lfm2 or lfm3) else ()
    gdn_layers = (0,) if q35 else ()
    # a qwen3_5 pack nests its language model under language_model.* and ties the head to the embedding
    top = "language_model." if q35 else "model."
    NKH = NVH = 2
    KHD = VHD = 128            # key_dim = value_dim = 256, so the fused q|k|v is 768 x H: whole tiles
    CK = 4

    def w(*shape, sc=0.05):
        return (rng.standard_normal(shape) * sc).astype(np.float32)

    def norm(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    weights = {top + "embed_tokens.weight": w(VOCAB, H, sc=0.5),
               top + ("embedding_norm.weight" if lfm3 else "norm.weight"): norm(H)}
    if not q35:
        weights["lm_head.weight"] = w(VOCAB, H, sc=0.2)
    for layer in range(layers):
        p = f"{top}layers.{layer}."
        if layer in gdn_layers:
            # HF names; the packer renames them and transposes conv1d ([CD, 1, CK] -> [CK][CD]). No self_attn.q_norm:
            # a GDN layer normalises inside its own recurrence (linear_attn.norm), not per attention head.
            kd, vd = NKH * KHD, NVH * VHD
            weights.update({
                p + "input_layernorm.weight": norm(H), p + "post_attention_layernorm.weight": norm(H),
                p + "linear_attn.in_proj_qkv.weight": w(2 * kd + vd, H), p + "linear_attn.in_proj_z.weight": w(vd, H),
                p + "linear_attn.in_proj_a.weight": w(NVH, H), p + "linear_attn.in_proj_b.weight": w(NVH, H),
                p + "linear_attn.out_proj.weight": w(H, vd),
                p + "linear_attn.conv1d.weight": w(2 * kd + vd, 1, CK, sc=0.3),
                p + "linear_attn.norm.weight": norm(VHD),
                p + "linear_attn.A_log": w(NVH, sc=0.3), p + "linear_attn.dt_bias": w(NVH, sc=0.3),
                p + "mlp.gate_proj.weight": w(FFN, H), p + "mlp.up_proj.weight": w(FFN, H),
                p + "mlp.down_proj.weight": w(H, FFN)})
            continue
        if lfm3:
            # the upstream names, which model_load renames; the gate is 2 x 256, not a tile, so it stays bf16
            weights.update({p + "operator_norm.weight": norm(H), p + "post_operator_norm.weight": norm(H),
                            p + "ffn_norm.weight": norm(H), p + "post_ffn_norm.weight": norm(H),
                            p + "feed_forward.w1.weight": w(FFN, H), p + "feed_forward.w3.weight": w(FFN, H),
                            p + "feed_forward.w2.weight": w(H, FFN)})
            if layer in conv_layers:
                weights.update({p + "conv.in_proj.weight": w(3 * H, H), p + "conv.conv.weight": w(H, 3, sc=0.4),
                                p + "conv.out_proj.weight": w(H, H)})
            else:
                weights.update({
                    p + "self_attn.q_proj.weight": w(HEADS * HEAD_DIM, H),
                    p + "self_attn.k_proj.weight": w(KV_HEADS * HEAD_DIM, H),
                    p + "self_attn.v_proj.weight": w(KV_HEADS * HEAD_DIM, H),
                    p + "self_attn.o_proj.weight": w(H, HEADS * HEAD_DIM),
                    p + "self_attn.q_norm.weight": norm(HEAD_DIM), p + "self_attn.k_norm.weight": norm(HEAD_DIM),
                    p + "self_attn.gate_proj.weight": w(HEADS, H, sc=0.3)})
            continue
        if layer in conv_layers:
            # in_proj is H -> 3H ([B | C | X]), the taps are [H][L] and stay bf16 (3 columns are not a tile), out_proj
            # is H -> H. No self_attn.* at all: that absence is what makes it a conv layer to the loader.
            weights.update({
                p + "input_layernorm.weight": norm(H), p + "post_attention_layernorm.weight": norm(H),
                p + "shortconv.in_proj.weight": w(3 * H, H), p + "shortconv.conv.weight": w(H, 3, sc=0.4),
                p + "shortconv.out_proj.weight": w(H, H),
                p + "mlp.gate_proj.weight": w(FFN, H), p + "mlp.up_proj.weight": w(FFN, H),
                p + "mlp.down_proj.weight": w(H, FFN)})
            continue
        weights.update({
            p + "input_layernorm.weight": norm(H), p + "post_attention_layernorm.weight": norm(H),
            p + "self_attn.q_proj.weight": w(HEADS * HEAD_DIM, H),
            p + "self_attn.k_proj.weight": w(KV_HEADS * HEAD_DIM, H),
            p + "self_attn.v_proj.weight": w(KV_HEADS * HEAD_DIM, H),
            p + "self_attn.o_proj.weight": w(H, HEADS * HEAD_DIM),
            p + "self_attn.q_norm.weight": norm(HEAD_DIM), p + "self_attn.k_norm.weight": norm(HEAD_DIM),
            p + "mlp.gate_proj.weight": w(FFN, H), p + "mlp.up_proj.weight": w(FFN, H),
            p + "mlp.down_proj.weight": w(H, FFN)})
        if qwen2:
            # Qwen2: biases on q/k/v (large enough to matter), no per-head q/k norms
            del weights[p + "self_attn.q_norm.weight"], weights[p + "self_attn.k_norm.weight"]
            weights.update({p + "self_attn.q_proj.bias": w(HEADS * HEAD_DIM, sc=0.5),
                            p + "self_attn.k_proj.bias": w(KV_HEADS * HEAD_DIM, sc=0.5),
                            p + "self_attn.v_proj.bias": w(KV_HEADS * HEAD_DIM, sc=0.5)})
    os.makedirs(d)
    write_safetensors(os.path.join(d, "model.safetensors"), weights)
    cfg = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": H,
           "num_hidden_layers": LAYERS, "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
           "head_dim": HEAD_DIM, "intermediate_size": FFN, "vocab_size": VOCAB, "rms_norm_eps": 1e-6,
           "rope_theta": 1000000.0, "tie_word_embeddings": False, "max_position_embeddings": 64,
           "hidden_act": "silu", "torch_dtype": "float32"}
    if q35:
        cfg.update({"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text", "tie_word_embeddings": True,
                    "layer_types": ["linear_attention", "full_attention"], "linear_num_key_heads": NKH,
                    "linear_key_head_dim": KHD, "linear_num_value_heads": NVH, "linear_value_head_dim": VHD,
                    "linear_conv_kernel_dim": CK})
    if lfm2:
        del cfg["rms_norm_eps"]
        cfg.update({"architectures": ["Lfm2ForCausalLM"], "model_type": "lfm2", "num_hidden_layers": layers,
                    "norm_eps": 1e-5, "conv_L_cache": 3, "full_attn_idxs": [1]})
    if qwen2:
        cfg.update({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"})   # and no attention_bias key
    if lfm3:
        del cfg["rms_norm_eps"], cfg["rope_theta"]
        cfg.update({"architectures": ["Lfm3ForCausalLM"], "model_type": "lfm3", "num_hidden_layers": layers,
                    "norm_eps": 1e-5, "conv_L_cache": 3, "use_sandwich_norm": True, "sliding_window": 2,
                    "layer_types": ["conv", "full_attention", "conv", "sliding_attention"],
                    "rope_parameters": {
                        "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1000000.0,
                                           "rope_type": "default"},
                        "sliding_attention": {"rope_theta": 10000, "rope_type": "default"}}})
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    with open(os.path.join(d, "tokenizer.json"), "w") as f:
        f.write("{}")
    with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
        json.dump({"add_bos_token": False}, f)
    return {name: a.shape for name, a in weights.items()}


def lfm3_reference(q4nx_dir, np, ids, shapes):
    """LFM3's forward (LiquidAI's modeling_lfm3.py, transcribed) on the weights the forward itself reads: the packed
    container's tensors, the I8 ones untiled. Returns the last position's logits.

    batched == sequential compares our forward with itself, so a rule both paths get wrong passes it (#2254 review:
    with the head gate a no-op in step() and step_batch() alike, the arm still matched). This is the check that
    can see the gate, the zero-tail RoPE, the sandwich and the conv layers."""
    sys.path.insert(0, os.path.join(HERE, "..", "tools"))
    from q4nx_pack import untile
    with open(os.path.join(q4nx_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(q4nx_dir, "model.q4nx"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        table = json.loads(f.read(n))
        blob = f.read()

    def W(name):
        e = table[name]
        lo, hi = e["data_offsets"]
        if e["dtype"] == "BF16":
            v = (np.frombuffer(blob[lo:hi], dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            return v.reshape(e["shape"])
        if e["dtype"] == "F32":
            return np.frombuffer(blob[lo:hi], dtype=np.float32).reshape(e["shape"])
        rows, k = shapes[name]          # an I8 entry's shape is its tile grid; the logical one is the source's
        return untile(np.frombuffer(blob[lo:hi], dtype=np.uint8).reshape(-1, 5120), rows, k)

    nh, nkv, hd, eps = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["norm_eps"]
    T, win, hsz = len(ids), cfg["sliding_window"], cfg["hidden_size"]

    def rms(x, w):
        return w * (x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)).astype(np.float32)

    def rope(x, layer_type):
        rp = cfg["rope_parameters"][layer_type]
        angles = int(rp.get("partial_rotary_factor", 1.0) * hd) // 2
        inv = 1.0 / rp["rope_theta"] ** (np.arange(0, 2 * angles, 2) / hd)
        inv = np.concatenate([inv, np.zeros(hd // 2 - angles)])         # the zero tail
        fr = np.arange(T)[:, None] * inv[None, :]
        c, sn = np.cos(np.concatenate([fr, fr], -1))[:, None], np.sin(np.concatenate([fr, fr], -1))[:, None]
        x = x.astype(np.float64)
        return (x * c + np.concatenate([-x[..., hd // 2:], x[..., :hd // 2]], -1) * sn).astype(np.float32)

    x = W("model.embed_tokens.weight")[ids].astype(np.float32)
    for layer, lt in enumerate(cfg["layer_types"]):
        p = f"model.layers.{layer}."
        xn = rms(x, W(p + "operator_norm.weight"))
        if lt == "conv":
            bcx = xn @ W(p + "conv.in_proj.weight").T
            b, c, v = bcx[:, :hsz], bcx[:, hsz:2 * hsz], bcx[:, 2 * hsz:]
            taps = W(p + "conv.conv.weight")
            L = taps.shape[1]
            pad = np.concatenate([np.zeros((L - 1, hsz), np.float32), b * v], 0)
            y = np.stack([(pad[t:t + L].T * taps).sum(-1) for t in range(T)])
            y = (c * y) @ W(p + "conv.out_proj.weight").T
        else:
            q = rms((xn @ W(p + "self_attn.q_proj.weight").T).reshape(T, nh, hd), W(p + "self_attn.q_norm.weight"))
            k = rms((xn @ W(p + "self_attn.k_proj.weight").T).reshape(T, nkv, hd), W(p + "self_attn.k_norm.weight"))
            v = (xn @ W(p + "self_attn.v_proj.weight").T).reshape(T, nkv, hd)
            q, k = rope(q, lt), rope(k, lt)
            k, v = np.repeat(k, nh // nkv, 1), np.repeat(v, nh // nkv, 1)
            sc = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd)
            mask = np.tril(np.ones((T, T), bool))
            if lt == "sliding_attention":
                mask &= (np.arange(T)[:, None] - np.arange(T)[None, :]) < win
            sc = np.exp(np.where(mask[None], sc, -np.inf) - sc.max(-1, keepdims=True))
            ctx = np.einsum("hts,shd->thd", sc / sc.sum(-1, keepdims=True), v)
            gate = 1.0 / (1.0 + np.exp(-(xn @ W(p + "self_attn.gate_proj.weight").T)))
            y = (ctx * gate[:, :, None]).reshape(T, nh * hd) @ W(p + "self_attn.o_proj.weight").T
        x = x + rms(y, W(p + "post_operator_norm.weight"))
        h = rms(x, W(p + "ffn_norm.weight"))
        a = h @ W(p + "feed_forward.w1.weight").T
        ff = ((a / (1.0 + np.exp(-a))) * (h @ W(p + "feed_forward.w3.weight").T)) @ W(p + "feed_forward.w2.weight").T
        x = x + rms(ff, W(p + "post_ffn_norm.weight"))
    return rms(x, W("model.embedding_norm.weight"))[-1] @ W("lm_head.weight").T


def qwen2_reference(q4nx_dir, np, ids, shapes):
    """HF's Qwen2 forward (llama with q/k/v biases) on the container's own weights; the last position's logits."""
    sys.path.insert(0, os.path.join(HERE, "..", "tools"))
    from q4nx_pack import untile
    with open(os.path.join(q4nx_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(q4nx_dir, "model.q4nx"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        table = json.loads(f.read(n))
        blob = f.read()

    def W(name):
        e = table[name]
        lo, hi = e["data_offsets"]
        if e["dtype"] == "BF16":
            return (np.frombuffer(blob[lo:hi], dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(e["shape"])
        if e["dtype"] == "F32":
            return np.frombuffer(blob[lo:hi], dtype=np.float32).reshape(e["shape"])
        rows, k = shapes[name]
        return untile(np.frombuffer(blob[lo:hi], dtype=np.uint8).reshape(-1, 5120), rows, k)

    nh, nkv, hd, eps, T = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"], len(ids)

    def rms(x, w):
        return w * (x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)).astype(np.float32)

    def rope(x):
        inv = 1.0 / cfg["rope_theta"] ** (np.arange(0, hd, 2) / hd)
        fr = np.arange(T)[:, None] * inv[None, :]
        c, sn = np.cos(np.concatenate([fr, fr], -1))[:, None], np.sin(np.concatenate([fr, fr], -1))[:, None]
        x = x.astype(np.float64)
        return (x * c + np.concatenate([-x[..., hd // 2:], x[..., :hd // 2]], -1) * sn).astype(np.float32)

    x = W("model.embed_tokens.weight")[ids].astype(np.float32)
    for layer in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        xn = rms(x, W(p + "input_layernorm.weight"))
        q = (xn @ W(p + "self_attn.q_proj.weight").T + W(p + "self_attn.q_proj.bias")).reshape(T, nh, hd)
        k = (xn @ W(p + "self_attn.k_proj.weight").T + W(p + "self_attn.k_proj.bias")).reshape(T, nkv, hd)
        v = (xn @ W(p + "self_attn.v_proj.weight").T + W(p + "self_attn.v_proj.bias")).reshape(T, nkv, hd)
        q, k = rope(q), rope(k)
        k, v = np.repeat(k, nh // nkv, 1), np.repeat(v, nh // nkv, 1)
        sc = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd)
        sc = np.where(np.tril(np.ones((T, T), bool))[None], sc, -np.inf)
        sc = np.exp(sc - sc.max(-1, keepdims=True))
        ctx = np.einsum("hts,shd->thd", sc / sc.sum(-1, keepdims=True), v)
        x = x + ctx.reshape(T, nh * hd) @ W(p + "self_attn.o_proj.weight").T
        h = rms(x, W(p + "post_attention_layernorm.weight"))
        a = h @ W(p + "mlp.gate_proj.weight").T
        x = x + ((a / (1.0 + np.exp(-a))) * (h @ W(p + "mlp.up_proj.weight").T)) @ W(p + "mlp.down_proj.weight").T
    return rms(x, W("model.norm.weight"))[-1] @ W("lm_head.weight").T


def tmp_root():
    """tests/tmp_root.h's rule (engine#2029): scratch goes on disk, not in tmpfs. ONEBIT_TEST_TMP if set, else
    ~/.cache/scratch-tests/tmp, else the system temp directory."""
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
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] not in ("lfm2", "lfm3", "qwen2", "qwen3_5")):
        print(__doc__.strip().splitlines()[2])
        return 2
    family = sys.argv[2] if len(sys.argv) == 3 else "qwen3"
    try:
        import numpy as np
    except ImportError:
        print("SKIP: numpy is not installed")
        return 77
    with tempfile.TemporaryDirectory(prefix="batch_check_state_ci.", dir=tmp_root()) as tmp:
        src, out = os.path.join(tmp, "hf"), os.path.join(tmp, "q4nx")
        shapes = make_model(src, np, family)
        r = subprocess.run([sys.executable, TOOL, "pack", src, out], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL  packing the fabricated container: {r.stderr[-600:]}")
            return 1
        env = dict(os.environ, ONEBIT_TEST_Q4NX=out)
        logits_out = os.path.join(tmp, "sequential.f32")
        if family in ("lfm3", "qwen2"):
            env["ONEBIT_TEST_LOGITS_OUT"] = logits_out
        env.pop("NPU_INFER_HOST_GEMM", None)   # init_host is host-GEMM already; nothing else should steer it
        r = subprocess.run([sys.argv[1]], env=env, capture_output=True, text=True, timeout=110)
        for line in (r.stdout + r.stderr).splitlines():
            if (line.startswith(("PASS", "FAIL", "SKIP", "ALL PASS", "FAILED", "[batchcheck]")) or
                    "short-convolution" in line or "linear/GDN" in line or "linear_attention" in line):
                print(line)
        if r.returncode == 77:
            # The container is made here, so a skip would mean the test could not load it: that is a failure of this
            # arm, not a reason to skip it.
            print("FAIL  batch_check_state_test skipped on the fabricated container (init_host refused it)")
            return 1
        if family in ("lfm3", "qwen2") and r.returncode == 0:
            ours = np.fromfile(logits_out, dtype=np.float32)[:VOCAB]
            ref = (lfm3_reference if family == "lfm3" else qwen2_reference)(out, np, [1, 2, 3, 4, 5, 6], shapes)
            corr = float(np.corrcoef(ours, ref)[0, 1])
            ok = (corr >= LFM3_MIN_CORR and float(np.abs(ours - ref).max()) <= LFM3_MAX_DIFF * float(np.abs(ref).max())
                  and int(np.argmax(ours)) == int(np.argmax(ref)))
            print(f"{'PASS' if ok else 'FAIL'}  sequential logits vs the {family} reference on the container's "
                  f"weights: corr {corr:.7f} (needs {LFM3_MIN_CORR}), max |d| {float(np.abs(ours - ref).max()):.3g} of "
                  f"{float(np.abs(ref).max()):.3g}, argmax {int(np.argmax(ours))} vs "
                  f"{int(np.argmax(ref))}")
            return 0 if ok else 1
        return r.returncode


if __name__ == "__main__":
    sys.exit(main())
