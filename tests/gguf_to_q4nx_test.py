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
# one real Q8 tile (Qwen3.5-4B lm_head tile 0), hashed by the same test as npu/q4nx.cpp: this is
# the layout check that a self-inverse round trip cannot make, because our encoder and decoder
# could agree with each other and still disagree with the container a device serves.
GOLDEN_Q8 = "09ad9e7a61cad612ffcb9cc2a1d7dfd27b6813a6d184329404045d4691c5fdf2"


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


def golden_tile8():
    """The committed real Q8 tile: decoded by our decoder, and re-encoded by our encoder."""
    tile = np.fromfile(os.path.join(HERE, "golden", "q4nx", "q8.bin"), dtype=np.uint8)
    assert tile.size == G.Q8, f"q8.bin is {tile.size} B"
    w = G.untile_q8(tile.reshape(1, G.Q8), 32, 256).astype(np.float32)
    ok = hashlib.sha256(w.tobytes()).hexdigest() == GOLDEN_Q8
    print(f"{'ok  ' if ok else 'FAIL'} golden q8 tile decodes to the verified SHA-256")
    # its parts, back through the writer (scale plane is [g, r]; tiles_q8 wants [r, g])
    sc = G.from_bf16(tile[:512].view(np.uint16)).reshape(8, 32).T
    codes = np.clip(np.rint(w.reshape(32, 8, 32) / np.where(sc == 0, 1, sc)[..., None]), -127, 127)
    again = G.tiles_q8(codes.reshape(32, 256).astype(np.int8), sc)
    same = again.tobytes() == tile.tobytes()
    print(f"{'ok  ' if same else 'FAIL'} golden q8 tile re-encodes to the same bytes")
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
    # The two flags app/forward_serve.cpp builds the prompt from; add_bos_token is the one a
    # qwen2 checkpoint can set to true (Zyphra/ZR1-1.5B does) and the packer has to carry.
    w.add_bos_token_id(2)
    w.add_add_bos_token(True)
    w.add_add_eos_token(False)

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


def moe_synthetic(path, rng):
    """A small qwen3moe GGUF: 4 experts, one shared-expert-free MoE block (Qwen3-30B-A3B's shape)."""
    H, IM, E, NH, NKV, HD, V = 256, 512, 4, 2, 1, 128, 512
    w = gguf.GGUFWriter(path, "qwen3moe")
    w.add_block_count(1)
    w.add_context_length(4096)
    w.add_embedding_length(H)
    w.add_feed_forward_length(1024)          # dense value, unused by the arch
    w.add_head_count(NH)
    w.add_head_count_kv(NKV)
    w.add_key_length(HD)
    w.add_value_length(HD)
    w.add_rope_freq_base(1e6)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_eos_token_id(1)
    w.add_expert_count(E)
    w.add_expert_used_count(2)
    w.add_expert_feed_forward_length(IM)
    w.add_expert_shared_feed_forward_length(0)

    def quant(name, N, K, qt):
        x = rng.standard_normal((N, K)).astype(np.float32) * 0.05
        if qt == Q.Q4_K:
            data = q4_k_blocks(rng, N, K)
        elif qt == Q.F32:
            data = x
        else:
            data = gguf.quants.quantize(x, qt)
        w.add_tensor(name, data, raw_dtype=qt)

    def quant_exps(name, N, K, qt):
        """A routed expert stack: ggml 3-D [K, N, E], expert e on rows [e*N, (e+1)*N)."""
        if qt == Q.Q4_K:
            data = q4_k_blocks(rng, E * N, K).reshape(E, N, (K // 256) * 144)
        else:
            x = (rng.standard_normal((E * N, K)).astype(np.float32) * 0.05)
            data = x.reshape(E, N, K)
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
    w.add_tensor("blk.0.ffn_gate_inp.weight", rng.standard_normal((H, E)).astype(np.float32))
    quant_exps("blk.0.ffn_gate_exps.weight", IM, H, Q.Q4_K)
    quant_exps("blk.0.ffn_up_exps.weight", IM, H, Q.F32)     # requantized expert stack
    quant_exps("blk.0.ffn_down_exps.weight", H, IM, Q.Q4_K)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def moe_check(gguf_path, out, ok):
    """The expert decomposition: __metadata__.moe's offsets must point at THAT expert."""
    import json
    with open(os.path.join(out, "model.q4nx"), "rb") as f:
        import struct
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
    mm = np.memmap(os.path.join(out, "model.q4nx"), dtype=np.uint8, mode="r")
    moe = hdr["__metadata__"].get("moe")
    checks = []
    checks.append(("__metadata__.moe present with 4 experts / top-2",
                   moe is not None and moe["experts"] == 4 and moe["used"] == 2))
    gate = "model.layers.0.mlp.experts.gate_proj.weight"
    checks.append((f"{gate} is an expert stack of 4",
                   gate in (moe or {}).get("tensors", {})
                   and len(moe["tensors"][gate]["offsets"]) == 4))
    checks.append(("the router is bf16, not tiled",
                   hdr["model.layers.0.mlp.gate.weight"]["dtype"] == "BF16"))
    with open(os.path.join(out, "config.json")) as f:
        cfg = json.load(f)
    checks.append(("config.json carries the MoE fields",
                   cfg["model_type"] == "qwen3moe" and cfg["num_experts"] == 4
                   and cfg["num_experts_per_tok"] == 2 and cfg["norm_topk_prob"] is True
                   and cfg["intermediate_size"] == 512))

    r = gguf.GGUFReader(gguf_path)
    for name, key in ((gate, "blk.0.ffn_gate_exps.weight"),
                      ("model.layers.0.mlp.experts.up_proj.weight", "blk.0.ffn_up_exps.weight"),
                      ("model.layers.0.mlp.experts.down_proj.weight", "blk.0.ffn_down_exps.weight")):
        t = next(t for t in r.tensors if t.name == key)
        K, N = int(t.shape[0]), int(t.shape[1])
        ref = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float64).reshape(4, N, K)
        info = moe["tensors"][name]
        chunk = hdr[name]["shape"][-1]  # 5120, or 8704 for a --dtype8 stack
        worst = 0.0
        for e in range(4):
            lo = base + info["offsets"][e]
            tiles = np.asarray(mm[lo:lo + info["bytes"]]).reshape(-1, chunk)
            got = (G.untile_q8(tiles, N, K) if chunk == G.Q8 else G.untile(tiles, N, K))
            # A wrong offset points at another expert (rel rms ~sqrt(2)); the right one is within
            # the chunk's own quantisation error (0.08 for q4_1, 0.005 for Q8), so compare
            # distributions rather than a max.
            rel = float(np.sqrt(((got - ref[e]) ** 2).sum() / (ref[e] ** 2).sum()))
            worst = max(worst, rel)
        checks.append((f"each of {name}'s 4 offsets decodes to THAT expert "
                       f"({'Q8' if chunk == G.Q8 else 'q4_1'} chunk, rel rms {worst:.4f})",
                       worst <= 0.15))
    for label, good in checks:
        print(f"{'ok  ' if good else 'FAIL'} {label}")
        ok = ok and good
    return ok


def tokenizer_flags_check(src, out, ok):
    """config.json must carry the tokenizer's add_bos/add_eos flags, or the forward builds chat
    prompts without BOS (it reads add_bos_token from config.json and defaults to 0)."""
    import json
    with open(os.path.join(out, "config.json")) as f:
        cfg = json.load(f)
    checks = [("config.json keeps add_bos_token from the GGUF", cfg.get("add_bos_token") is True),
              ("config.json keeps add_eos_token from the GGUF", cfg.get("add_eos_token") is False),
              ("bos_token_id still written", cfg.get("bos_token_id") == 2)]
    for label, good in checks:
        print(f"{'ok  ' if good else 'FAIL'} {label}")
        ok = ok and good
    return ok


def q8_check(gguf_path, out, ok):
    """--dtype8: the named tensors are 8704-B Q8 chunks, an exactly-4-bit one is left alone, and
    the Q8 tile re-encodes byte for byte from the source (so the encoder IS the decoder's inverse)."""
    import json
    import struct
    with open(os.path.join(out, "model.q4nx"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
    mm = np.memmap(os.path.join(out, "model.q4nx"), dtype=np.uint8, mode="r")
    o_proj = "model.layers.0.self_attn.o_proj.weight"       # Q8_0 in the source
    up_proj = "model.layers.0.mlp.up_proj.weight"           # F32 in the source
    v_proj = "model.layers.0.self_attn.v_proj.weight"       # Q4_K: exact already, must stay q4_1
    checks = [
        ("--dtype8 writes 8704-B Q8 chunks",
         hdr[o_proj]["shape"][-1] == G.Q8 and hdr[up_proj]["shape"][-1] == G.Q8),
        ("an exact Q4_K tensor named by --dtype8 stays q4_1",
         hdr[v_proj]["shape"][-1] == G.TILE),
        ("the pattern is recorded for a round trip",
         hdr["__metadata__"].get("dtype8") == "o_proj|up_proj|v_proj"),
    ]
    r = gguf.GGUFReader(gguf_path)
    for name, key in ((o_proj, "blk.0.attn_output.weight"), (up_proj, "blk.0.ffn_up.weight")):
        t = next(t for t in r.tensors if t.name == key)
        K, N = int(t.shape[0]), int(t.shape[1])
        w = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(N, K)
        lo, hi = hdr[name]["data_offsets"]
        raw = np.asarray(mm[base + lo:base + hi])
        got = G.untile_q8(raw, N, K)
        # the source, quantised by the packer's own rule, must give these bytes back
        codes, scale = G.quantize_q8(w)
        again = G.tiles_q8(codes, scale).tobytes() == raw.tobytes()
        rel = float(np.sqrt(((got - w) ** 2).sum() / (w ** 2).sum()))
        checks.append((f"{key}: Q8 tile re-encodes to the same bytes", again))
        checks.append((f"{key}: 8-bit error rel rms {rel:.4f} (<= 0.01)", rel <= 0.01))
    for label, good in checks:
        print(f"{'ok  ' if good else 'FAIL'} {label}")
        ok = ok and good
    return ok


def main():
    ok = golden_tile()
    ok = golden_tile8() and ok
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
        mout = os.path.join(d, "moe")
        moe_synthetic(os.path.join(d, "tiny-moe.gguf"), rng)
        for cmd in (["pack", os.path.join(d, "tiny-moe.gguf"), mout],
                    ["verify", "--all", os.path.join(d, "tiny-moe.gguf"), mout]):
            r = subprocess.run([sys.executable, tool, *cmd], capture_output=True, text=True)
            print(r.stdout, end="")
            if r.returncode:
                print(r.stderr, end="")
                print(f"FAIL moe {cmd[0]} exited {r.returncode}")
                ok = False
        ok = moe_check(os.path.join(d, "tiny-moe.gguf"), mout, ok)
        # the same MoE container with one expert stack in Q8: __metadata__.moe's offsets must
        # follow the 8704-B chunk, not stay on 5120
        mq8 = os.path.join(d, "moe-q8")
        for cmd in (["pack", os.path.join(d, "tiny-moe.gguf"), mq8, "--dtype8", "experts.up_proj"],
                    ["verify", "--all", os.path.join(d, "tiny-moe.gguf"), mq8]):
            r = subprocess.run([sys.executable, tool, *cmd], capture_output=True, text=True)
            print(r.stdout, end="")
            if r.returncode:
                print(r.stderr, end="")
                print(f"FAIL moe-q8 {cmd[0]} exited {r.returncode}")
                ok = False
        ok = moe_check(os.path.join(d, "tiny-moe.gguf"), mq8, ok)
        ok = tokenizer_flags_check(src, out, ok)
        q8out, q8gguf = os.path.join(d, "q8"), os.path.join(d, "q8.gguf")
        for cmd in (["pack", src, q8out, "--dtype8", "o_proj|up_proj|v_proj"],
                    ["verify", "--all", src, q8out]):
            r = subprocess.run([sys.executable, tool, *cmd], capture_output=True, text=True)
            print(r.stdout, end="")
            if r.returncode:
                print(r.stderr, end="")
                print(f"FAIL q8 {cmd[0]} exited {r.returncode}")
                ok = False
        ok = q8_check(src, q8out, ok)
        # and the inverse: the Q8 container written back as a GGUF, identical weights both ways
        inv = os.path.join(HERE, "..", "tools", "q4nx_to_gguf.py")
        for cmd in (["write", os.path.join(q8out, "model.q4nx"), src, q8gguf],
                    ["verify", os.path.join(q8out, "model.q4nx"), q8gguf, "--roundtrip", src]):
            r = subprocess.run([sys.executable, inv, *cmd], capture_output=True, text=True)
            if r.returncode:
                print(r.stdout[-2000:], r.stderr[-2000:])
                print(f"FAIL q4nx_to_gguf {cmd[0]} exited {r.returncode}")
                ok = False
        q8w = gguf.GGUFReader(q8gguf)
        q8_0 = sum(t.tensor_type == Q.Q8_0 for t in q8w.tensors)
        print(f"{'ok  ' if q8_0 == 2 else 'FAIL'} the Q8 container writes back as 2 Q8_0 tensors "
              f"(got {q8_0})")
        ok = ok and q8_0 == 2
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
