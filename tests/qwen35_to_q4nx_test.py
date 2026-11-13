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
"""tools/qwen35_to_q4nx.py on a small qwen35 GGUF written here (engine#2013).

usage: tests/qwen35_to_q4nx_test.py        (needs numpy; no gguf package, no model, no device)

One gated-linear-attention layer and one full-attention layer, at the 16 key groups and 48 value
heads the tool is written for and a hidden size of 256, so the file is a few megabytes.  Checks:

  * pack, then verify: every tensor of both layer families comes back, 0 bad tensors;
  * the container has the names the forward looks up, and config.json the layer types;
  * verify notices a container packed in the stored (not reordered) v-head order;
  * the three inputs that would give a wrong container are refused by name and leave no
    output directory: another architecture, a file that still carries prism.hadamard.* keys,
    and a projection stored as PTQ1_0 (ggml type 143).
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools", "qwen35_to_q4nx.py")

F32, Q4_0, PTQ1_0 = 0, 2, 143
U32, F32V, STR = 4, 6, 8
H, IM, V = 256, 512, 512            # hidden, feed-forward, vocabulary
HD_G, NKH, NVH = 128, 16, 48        # GDN head dim, key groups, value heads (the tool's constants)
HD_A, NH, NKV = 128, 2, 1           # attention head dim, heads, kv heads


def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, kv, tensors):
    """GGUF v3: kv is [(key, type, value)], tensors is [(name, ggml dims, type, bytes)]."""
    head = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kv))
    for k, t, v in kv:
        head += gstr(k) + struct.pack("<I", t)
        head += gstr(v) if t == STR else struct.pack("<I" if t == U32 else "<f", v)
    off, infos, blobs = 0, b"", []
    for name, dims, tt, data in tensors:
        infos += gstr(name) + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims)
        infos += struct.pack("<IQ", tt, off)
        pad = (-len(data)) % 32
        blobs.append(data + b"\0" * pad)
        off += len(data) + pad
    body = head + infos
    body += b"\0" * ((-len(body)) % 32)
    with open(path, "wb") as f:
        f.write(body)
        for b in blobs:
            f.write(b)


def model(rng, arch="qwen35", extra_kv=(), qkv_type=Q4_0):
    kv = [("general.architecture", STR, arch),
          ("qwen35.block_count", U32, 2), ("qwen35.full_attention_interval", U32, 2),
          ("qwen35.embedding_length", U32, H), ("qwen35.feed_forward_length", U32, IM),
          ("qwen35.attention.head_count", U32, NH), ("qwen35.attention.head_count_kv", U32, NKV),
          ("qwen35.attention.key_length", U32, HD_A), ("qwen35.context_length", U32, 4096),
          ("qwen35.attention.layer_norm_rms_epsilon", F32V, 1e-6), ("qwen35.rope.freq_base", F32V, 1e7),
          ("qwen35.ssm.state_size", U32, HD_G), ("qwen35.ssm.inner_size", U32, NVH * HD_G),
          ("qwen35.ssm.group_count", U32, NKH), ("qwen35.ssm.conv_kernel", U32, 4),
          ("tokenizer.ggml.bos_token_id", U32, 2), ("tokenizer.ggml.eos_token_id", U32, 1)]
    kv += list(extra_kv)
    out = []

    def q4(name, K, N, tt=Q4_0):
        # 32 weights per block: an fp16 scale, then 16 bytes of nibbles
        blocks = np.zeros((N * K // 32, 18), np.uint8)
        blocks[:, 0:2] = rng.uniform(0.01, 0.05, N * K // 32).astype("<f2").view(np.uint8).reshape(-1, 2)
        blocks[:, 2:] = rng.integers(0, 256, (N * K // 32, 16), dtype=np.uint8)
        data = blocks.tobytes()
        if tt == PTQ1_0:      # 28 bytes per 128 weights; the bytes are never decoded, only refused
            data = bytes(N * K // 128 * 28)
        out.append((name, (K, N), tt, data))

    def f32(name, *dims):
        out.append((name, dims, F32, rng.standard_normal(int(np.prod(dims))).astype("<f4").tobytes()))

    cd = (2 * NKH + NVH) * HD_G                      # q | k | v rows of the fused projection
    q4("token_embd.weight", H, V)
    f32("output_norm.weight", H)
    # layer 0: gated linear attention
    q4("blk.0.attn_qkv.weight", H, cd, qkv_type)
    q4("blk.0.attn_gate.weight", H, NVH * HD_G)
    q4("blk.0.ssm_out.weight", NVH * HD_G, H)
    f32("blk.0.ssm_alpha.weight", H, NVH)
    f32("blk.0.ssm_beta.weight", H, NVH)
    f32("blk.0.ssm_conv1d.weight", 4, cd)
    f32("blk.0.ssm_norm.weight", HD_G)
    f32("blk.0.ssm_a", NVH)
    f32("blk.0.ssm_dt.bias", NVH)
    # layer 1: full attention, q stored per head as [q | gate]
    q4("blk.1.attn_q.weight", H, 2 * NH * HD_A)
    q4("blk.1.attn_k.weight", H, NKV * HD_A)
    q4("blk.1.attn_v.weight", H, NKV * HD_A)
    q4("blk.1.attn_output.weight", NH * HD_A, H)
    f32("blk.1.attn_q_norm.weight", HD_A)
    f32("blk.1.attn_k_norm.weight", HD_A)
    for l in (0, 1):
        f32(f"blk.{l}.attn_norm.weight", H)
        f32(f"blk.{l}.post_attention_norm.weight", H)
        q4(f"blk.{l}.ffn_gate.weight", H, IM)
        q4(f"blk.{l}.ffn_up.weight", H, IM)
        q4(f"blk.{l}.ffn_down.weight", IM, H)
    return kv, out


def run(*args):
    r = subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    rng = np.random.default_rng(7)
    fails = 0

    def check(good, label, detail=""):
        nonlocal fails
        print(f"{'ok  ' if good else 'FAIL'} {label}")
        if not good:
            fails += 1
            if detail:
                print(detail[-1200:])

    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "tiny.gguf"), os.path.join(d, "q4nx")
        with open(os.path.join(d, "tokenizer.json"), "w") as f:
            f.write("{}")
        write_gguf(src, *model(rng))

        rc, log = run("pack", src, out)
        check(rc == 0, "pack exits 0", log)
        rc, log = run("verify", src, out)
        check(rc == 0 and "0 bad tensors" in log, "verify: 0 bad tensors", log)

        with open(os.path.join(out, "model.q4nx"), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            table = json.loads(f.read(n))
        want = {
            "model.layers.0.linear_attn.qkv_proj.weight": [(2 * NKH + NVH) * HD_G // 32 * (H // 256), 5120],
            "model.layers.0.self_attn.gate_proj.weight": [NVH * HD_G // 32 * (H // 256), 5120],
            "model.layers.0.linear_attn.ssm_out_proj.weight": [H // 32 * (NVH * HD_G // 256), 5120],
            "model.layers.0.linear_attn.ssm_alpha_proj.weight": [2, 1, 8704],     # 48 rows padded to 64
            "model.layers.0.linear_attn.ssm_beta_proj.weight": [2, 1, 8704],
            "model.layers.0.linear_attn.ssm_conv1d.weight": [4 * (2 * NKH + NVH) * HD_G],
            "model.layers.0.linear_attn.ssm_norm.weight": [HD_G],
            "model.layers.0.linear_attn.ssm_a": [NVH],
            "model.layers.0.linear_attn.ssm_dt.bias": [NVH],
            "model.layers.1.self_attn.q_proj.weight": [2 * NH * HD_A // 32 * (H // 256), 5120],
            "model.layers.1.self_attn.q_norm.weight": [HD_A],
            "model.layers.1.post_attention_layernorm.weight": [H],
            "model.embed_tokens.weight": [V // 32 * (H // 256), 5120],
            "model.norm.weight": [H],
        }
        for name, shape in want.items():
            got = table.get(name, {}).get("shape")
            check(got == shape, f"{name} is in the container with shape {shape}", f"got {got}")
        check("lm_head.weight" not in table, "no lm_head tensor for a GGUF without output.weight")
        check(table.get("__metadata__", {}).get("converter") == "qwen35_to_q4nx.py",
              "the container names its converter")
        with open(os.path.join(out, "config.json")) as f:
            cfg = json.load(f)
        check(cfg.get("layer_types") == ["linear_attention", "full_attention"], "config.json: layer_types")
        check(cfg.get("model_type") == "qwen3_5" and cfg.get("linear_num_value_heads") == NVH
              and cfg.get("tie_word_embeddings") is True, "config.json: model_type, value heads, tied head")
        check(os.path.exists(os.path.join(out, "tokenizer.json")), "tokenizer.json is copied from the GGUF's directory")

        # verify has to notice the one thing a wrong pack gets wrong silently: the v-head order
        stored = os.path.join(d, "stored")
        rc, log = run("pack", src, stored, "--no-reorder")
        check(rc == 0, "pack --no-reorder exits 0", log)
        rc, log = run("verify", src, stored)
        check(rc != 0 and "BAD" in log, "verify fails a container packed in the stored v-head order", log)

        def refused(label, needle, **kw):
            p, o = os.path.join(d, label + ".gguf"), os.path.join(d, label + "-out")
            write_gguf(p, *model(rng, **kw))
            rc, log = run("pack", p, o)
            check(rc != 0 and needle in log and "Traceback" not in log, f"{label}: refused, naming {needle!r}", log)
            check(not os.path.exists(o), f"{label}: no output directory is left")

        refused("other-arch", "general.architecture is 'qwen3'", arch="qwen3")
        refused("rotated", "prism.hadamard.axis", extra_kv=[("prism.hadamard.axis", U32, 1)])
        refused("ternary", "ggml type 143", qkv_type=PTQ1_0)

    print("FAILED" if fails else "qwen35_to_q4nx: ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
