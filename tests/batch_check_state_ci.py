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

usage: tests/batch_check_state_ci.py <batch_check_state_test binary>        (needs numpy)

batch_check_state_test needs a Q4NX container, and CI has none, so on its own it skips (77) there and a regression of
#1909 -- the NPU_BATCH_CHECK oracle resetting the prefix it replays from -- lands green. This writes a two-layer qwen3
(hidden 256, two 128-wide query heads on one kv head, ffn 512, vocab 256; the smallest geometry the I8 tiles take whole),
packs it with tools/safetensors_to_q4nx.py, and runs the test on it in host-GEMM mode: no device, no model on disk.

Measured when it was written (engine#1946): passes on main; with #1909 reverted it fails 3 of 4 arms (the forward
holds 2 and 1 positions instead of 5 and 6, last-chunk rel-L1 0.78) -- the same failure the real containers show.
Weights are seeded, so the run is the same every time. Exits 77 (skip) without numpy.
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


def make_model(d, np):
    rng = np.random.default_rng(1946)

    def w(*shape, sc=0.05):
        return (rng.standard_normal(shape) * sc).astype(np.float32)

    def norm(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    weights = {"model.embed_tokens.weight": w(VOCAB, H, sc=0.5), "model.norm.weight": norm(H),
               "lm_head.weight": w(VOCAB, H, sc=0.2)}
    for layer in range(LAYERS):
        p = f"model.layers.{layer}."
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
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    with open(os.path.join(d, "tokenizer.json"), "w") as f:
        f.write("{}")
    with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
        json.dump({"add_bos_token": False}, f)


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2])
        return 2
    try:
        import numpy as np
    except ImportError:
        print("SKIP: numpy is not installed")
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        src, out = os.path.join(tmp, "hf"), os.path.join(tmp, "q4nx")
        make_model(src, np)
        r = subprocess.run([sys.executable, TOOL, "pack", src, out], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL  packing the fabricated container: {r.stderr[-600:]}")
            return 1
        env = dict(os.environ, ONEBIT_TEST_Q4NX=out)
        env.pop("NPU_INFER_HOST_GEMM", None)   # init_host is host-GEMM already; nothing else should steer it
        r = subprocess.run([sys.argv[1]], env=env, capture_output=True, text=True, timeout=110)
        for line in (r.stdout + r.stderr).splitlines():
            if line.startswith(("PASS", "FAIL", "SKIP", "ALL PASS", "FAILED", "[batchcheck]")):
                print(line)
        if r.returncode == 77:
            # The container is made here, so a skip would mean the test could not load it: that is a failure of this
            # arm, not a reason to skip it.
            print("FAIL  batch_check_state_test skipped on the fabricated container (init_host refused it)")
            return 1
        return r.returncode


if __name__ == "__main__":
    sys.exit(main())
