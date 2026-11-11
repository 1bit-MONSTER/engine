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

usage: tests/batch_check_state_ci.py <batch_check_state_test binary> [lfm2]        (needs numpy)

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
    layers = 3 if lfm2 else LAYERS
    conv_layers = (0, 2) if lfm2 else ()

    def w(*shape, sc=0.05):
        return (rng.standard_normal(shape) * sc).astype(np.float32)

    def norm(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    weights = {"model.embed_tokens.weight": w(VOCAB, H, sc=0.5), "model.norm.weight": norm(H),
               "lm_head.weight": w(VOCAB, H, sc=0.2)}
    for layer in range(layers):
        p = f"model.layers.{layer}."
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
    os.makedirs(d)
    write_safetensors(os.path.join(d, "model.safetensors"), weights)
    cfg = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": H,
           "num_hidden_layers": LAYERS, "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
           "head_dim": HEAD_DIM, "intermediate_size": FFN, "vocab_size": VOCAB, "rms_norm_eps": 1e-6,
           "rope_theta": 1000000.0, "tie_word_embeddings": False, "max_position_embeddings": 64,
           "hidden_act": "silu", "torch_dtype": "float32"}
    if lfm2:
        del cfg["rms_norm_eps"]
        cfg.update({"architectures": ["Lfm2ForCausalLM"], "model_type": "lfm2", "num_hidden_layers": layers,
                    "norm_eps": 1e-5, "conv_L_cache": 3, "full_attn_idxs": [1]})
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    with open(os.path.join(d, "tokenizer.json"), "w") as f:
        f.write("{}")
    with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
        json.dump({"add_bos_token": False}, f)


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
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != "lfm2"):
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
        make_model(src, np, family)
        r = subprocess.run([sys.executable, TOOL, "pack", src, out], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL  packing the fabricated container: {r.stderr[-600:]}")
            return 1
        env = dict(os.environ, ONEBIT_TEST_Q4NX=out)
        env.pop("NPU_INFER_HOST_GEMM", None)   # init_host is host-GEMM already; nothing else should steer it
        r = subprocess.run([sys.argv[1]], env=env, capture_output=True, text=True, timeout=110)
        for line in (r.stdout + r.stderr).splitlines():
            if line.startswith(("PASS", "FAIL", "SKIP", "ALL PASS", "FAILED", "[batchcheck]")) or "short-convolution" in line:
                print(line)
        if r.returncode == 77:
            # The container is made here, so a skip would mean the test could not load it: that is a failure of this
            # arm, not a reason to skip it.
            print("FAIL  batch_check_state_test skipped on the fabricated container (init_host refused it)")
            return 1
        return r.returncode


if __name__ == "__main__":
    sys.exit(main())
