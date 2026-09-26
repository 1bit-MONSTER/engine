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
"""tools/gguf_to_q4nx.py: the tile layout and a whole repack.

usage: tests/gguf_to_q4nx_test.py        (needs numpy and gguf)

1. The committed real q4_1 tile (tests/golden/q4nx/q4_1.bin, Qwen3-0.6B) decodes to the
   SHA-256 that tests/npu_q4nx_test.cpp checks (1bit-MONSTER's verified decoder), and its
   parts re-encode to the same 5120 bytes: the writer and reader use the real layout.
2. A small synthetic qwen3 GGUF with Q4_0, Q4_1, Q4_K, Q8_0 and F32 tensors packs, and
   verify passes (exact types within the bf16 bound, the rest requantized).
"""
import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np
import gguf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import gguf_to_q4nx as G  # noqa: E402

Q = gguf.GGMLQuantizationType
GOLDEN_Q4_1 = "9f58d4201c4844abc937204c2b7f8d364ff44bc574b0bb87292928c54447d2c8"


def golden_tile():
    tile = np.fromfile(os.path.join(HERE, "golden", "q4nx", "q4_1.bin"), dtype=np.uint8)
    w = G.untile(tile.reshape(1, G.TILE), 32, 256).astype(np.float32)
    ok = hashlib.sha256(w.tobytes()).hexdigest() == GOLDEN_Q4_1
    print(f"{'ok  ' if ok else 'FAIL'} golden q4_1 tile decodes to the verified SHA-256")
    # its parts, back through the writer
    sc = G.from_bf16(tile[:512].view(np.uint16)).reshape(8, 32).T  # [r, g]
    zr = G.from_bf16(tile[512:1024].view(np.uint16)).reshape(8, 32).T
    codes = np.clip(np.rint((w.reshape(32, 8, 32) - zr[..., None]) / np.where(sc == 0, 1, sc)[..., None]), 0, 15)
    again = G.tiles(codes.reshape(32, 256).astype(np.uint8), sc, zr)
    same = again.tobytes() == tile.tobytes()
    print(f"{'ok  ' if same else 'FAIL'} golden q4_1 tile re-encodes to the same bytes")
    return ok and same


def q4_k_blocks(rng, N, K):
    """Random but sane Q4_K rows: small positive d, dmin; random 6-bit scales and codes."""
    nb = K // 256
    b = rng.integers(0, 256, size=(N, nb, 144), dtype=np.uint8)
    b[..., 0:2] = rng.uniform(1e-4, 2e-3, (N, nb)).astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    b[..., 2:4] = rng.uniform(1e-4, 2e-3, (N, nb)).astype(np.float16).view(np.uint8).reshape(N, nb, 2)
    return b.reshape(N, nb * 144)


def synthetic(path, rng):
    H, IM, NH, NKV, HD, V = 256, 512, 2, 1, 128, 512
    w = gguf.GGUFWriter(path, "qwen3")
    w.add_block_count(1)
    w.add_context_length(4096)
    w.add_embedding_length(H)
    w.add_feed_forward_length(IM)
    w.add_head_count(NH)
    w.add_head_count_kv(NKV)
    w.add_key_length(HD)
    w.add_value_length(HD)
    w.add_rope_freq_base(1e6)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_eos_token_id(1)

    def quant(name, N, K, qt):
        x = rng.standard_normal((N, K)).astype(np.float32) * 0.05
        if qt == Q.Q4_K:
            data = q4_k_blocks(rng, N, K)
        elif qt == Q.F32:
            data = x
        else:
            data = gguf.quants.quantize(x, qt)
        w.add_tensor(name, data, raw_dtype=qt)

    quant("token_embd.weight", V, H, Q.Q8_0)
    w.add_tensor("output_norm.weight", rng.standard_normal(H).astype(np.float32))
    for n in ("attn_norm", "ffn_norm"):
        w.add_tensor(f"blk.0.{n}.weight", rng.standard_normal(H).astype(np.float32))
    for n in ("attn_q_norm", "attn_k_norm"):
        w.add_tensor(f"blk.0.{n}.weight", rng.standard_normal(HD).astype(np.float32))
    quant("blk.0.attn_q.weight", NH * HD, H, Q.Q4_0)
    quant("blk.0.attn_k.weight", NKV * HD, H, Q.Q4_1)
    quant("blk.0.attn_v.weight", NKV * HD, H, Q.Q4_K)
    quant("blk.0.attn_output.weight", H, NH * HD, Q.Q8_0)
    quant("blk.0.ffn_gate.weight", IM, H, Q.Q4_K)
    quant("blk.0.ffn_up.weight", IM, H, Q.F32)
    quant("blk.0.ffn_down.weight", H, IM, Q.Q4_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main():
    ok = golden_tile()
    rng = np.random.default_rng(1)
    tool = os.path.join(HERE, "..", "tools", "gguf_to_q4nx.py")
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "tiny.gguf"), os.path.join(d, "q4nx")
        synthetic(src, rng)
        for cmd in (["pack", src, out], ["verify", "--all", src, out]):
            r = subprocess.run([sys.executable, tool, *cmd], capture_output=True, text=True)
            print(r.stdout, end="")
            if r.returncode:
                print(r.stderr, end="")
                print(f"FAIL {cmd[0]} exited {r.returncode}")
                ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
