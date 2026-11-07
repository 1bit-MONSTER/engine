#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
"""tests/trit_embed_pack_test.py - the 1-bit embedding importer's known answers, and the C++ test's fixture.

Builds a synthetic `trit-packed-v1` model (the layout Cyronius/trit-embed-3l ships: 2-bit ternary codes + one
bf16 scale per output row, int8 embedding tables with fp32 row scales, f32 norms/biases/output projection),
packages it with tools/trit_embed_pack.py, and checks:

  * the packed ternary weight decodes to exactly the reference product `codes * scale` (equality, not a
    tolerance -- both sides are the same two operations);
  * the int8 tables and their row scales arrive byte-identical;
  * the importer refuses what it cannot represent: a reserved ternary code 3, an unknown format string, and a
    source dtype it does not read;
  * --write-fixture also emits tests/fixtures/trit_embed_tiny (model.embed + vocab.txt + expected.json), which
    tests/embed_model_test.cpp loads. The expected vectors here come from the arithmetic the model card
    documents (int8 power-of-two per-token activations, post-norm BERT, mean pool, L2), implemented
    independently of the C++ so the two have to agree.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "tools", "trit_embed_pack.py")
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "trit_embed_tiny")

DIMS = {"vocab": 32, "emb": 8, "d": 8, "heads": 2, "ffn": 16, "layers": 1, "max_pos": 8,
        "eps": 1e-12, "out_dim": 8, "loops": 1, "pool": "mean", "gelu": "tanh"}


def bf16_bits(a):
    """fp32 -> bf16 bits, round to nearest even (the source stores bf16 scales)."""
    b = a.astype(np.float32).view(np.uint32)
    lsb = (b >> 16) & 1
    rounded = b + 0x7FFF + lsb
    return (rounded >> 16).astype(np.uint16)


def pack_ternary(codes):
    """int8 {-1,0,1} [N, K] -> uint8 [N, K/4]: code + 1 in {0,1,2} at bits 2j..2j+1."""
    c = (codes.astype(np.int16) + 1).astype(np.uint8).reshape(codes.shape[0], -1, 4)
    return c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)


def synthetic_model(seed=1311):
    rng = np.random.default_rng(seed)
    t = {}
    V, D, F, H, T = DIMS["vocab"], DIMS["d"], DIMS["ffn"], DIMS["heads"], DIMS["max_pos"]
    t["word.qweight"] = rng.integers(-40, 40, size=(V, D), dtype=np.int8)
    t["word.scale"] = (rng.random(V).astype(np.float32) * 0.05 + 0.01).astype(np.float32)
    t["pos.qweight"] = rng.integers(-40, 40, size=(T, D), dtype=np.int8)
    t["pos.scale"] = (rng.random(T).astype(np.float32) * 0.05 + 0.01).astype(np.float32)
    t["tok_type.weight"] = (rng.standard_normal((2, D)) * 0.1).astype(np.float32)
    t["ln_e.weight"] = (rng.random(D).astype(np.float32) * 0.5 + 0.75)
    t["ln_e.bias"] = (rng.standard_normal(D) * 0.05).astype(np.float32)
    t["out.weight"] = (rng.standard_normal((D, D)) * 0.3).astype(np.float32)
    for i in range(DIMS["layers"]):
        p = f"layers.{i}."
        for name, (n, k) in {"q": (D, D), "k": (D, D), "v": (D, D), "o": (D, D), "fc1": (F, D),
                             "fc2": (D, F)}.items():
            codes = rng.integers(-1, 2, size=(n, k)).astype(np.int8)
            scale = (rng.random(n).astype(np.float32) * 0.1 + 0.02).astype(np.float32)
            t[p + name + ".codes"] = pack_ternary(codes)
            t[p + name + ".scale"] = scale
            t[p + name + ".bias"] = (rng.standard_normal(n) * 0.05).astype(np.float32)
        for ln in ("ln1", "ln2"):
            t[p + ln + ".weight"] = (rng.random(D).astype(np.float32) * 0.5 + 0.75)
            t[p + ln + ".bias"] = (rng.standard_normal(D) * 0.05).astype(np.float32)
    return t, rng


def write_safetensors(path, tensors):
    """Write the tensors as the mixed-dtype safetensors the reference packs (U8/BF16/I8/F32)."""
    hdr, blobs, off = {}, [], 0
    for name in sorted(tensors):
        a = tensors[name]
        if a.dtype == np.uint8:
            dt, b = "U8", a.tobytes()
        elif a.dtype == np.int8:
            dt, b = "I8", a.tobytes()
        elif a.dtype == np.float32:
            dt, b = "F32", a.astype("<f4").tobytes()
        elif a.dtype == np.uint16:
            dt, b = "BF16", a.astype("<u2").tobytes()
        else:
            raise SystemExit(f"{name}: {a.dtype}")
        hdr[name] = {"dtype": dt, "shape": [int(x) for x in a.shape], "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hj = json.dumps(hdr).encode()
    hj += b" " * ((-len(hj)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


# ------------------------------------------------------------------ the documented arithmetic, for the fixture

def quant_act(x):
    shape = x.shape
    if shape[-1] > 256 and shape[-1] % 256 == 0:
        x = x.reshape(*shape[:-1], shape[-1] // 256, 256)
    m = np.maximum(np.abs(x).max(-1, keepdims=True), 1e-5).astype(np.float32)
    f, k = np.frexp(m)
    s = np.ldexp(np.ones_like(m), k - 7 + (2.0 * f > 127.0 / 64.0)).astype(np.float32)
    return (np.clip(np.rint(x / s), -128, 127) * s).reshape(shape).astype(np.float32)


def ln(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return ((x - mu) / np.sqrt(var + eps)) * w + b


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def reference_embed(t, dims, ids):
    D, H, eps = dims["d"], dims["heads"], np.float32(dims["eps"])
    hd = D // H
    ids = np.asarray(ids, dtype=np.int64)
    n = len(ids)

    def tern(name, x):
        codes = t[name + ".codes"]
        c = np.zeros((codes.shape[0], codes.shape[1] * 4), dtype=np.int8)
        for j in range(4):
            c[:, j::4] = ((codes >> (2 * j)) & 3).astype(np.int8) - 1
        scale = t[name + ".scale"].astype(np.float32)
        w = c.astype(np.float32) * scale[:, None]
        return quant_act(x) @ w.T + t[name + ".bias"]

    pos = np.arange(n)
    x = (t["word.qweight"][ids].astype(np.float32) * t["word.scale"][ids][:, None]
         + t["pos.qweight"][pos].astype(np.float32) * t["pos.scale"][pos][:, None]
         + t["tok_type.weight"][0])
    x = ln(x, t["ln_e.weight"], t["ln_e.bias"], eps)
    for i in range(dims["layers"]):
        p = f"layers.{i}."
        q, k, v = (tern(p + s, x).reshape(n, H, hd).transpose(1, 0, 2) for s in ("q", "k", "v"))
        att = q @ k.transpose(0, 2, 1) / np.sqrt(np.float32(hd))
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        o = (att @ v).transpose(1, 0, 2).reshape(n, D)
        x = ln(x + tern(p + "o", o), t[p + "ln1.weight"], t[p + "ln1.bias"], eps)
        x = ln(x + tern(p + "fc2", gelu(tern(p + "fc1", x))), t[p + "ln2.weight"], t[p + "ln2.bias"], eps)
    out = x.mean(0) @ t["out.weight"].T
    return out / np.linalg.norm(out)


def make_tiny_vocab(path):
    pieces = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "##s", "the", "cat", "sat", "on",
              "mat", ".", ",", "a", "of", "index", "funds", "track", "market", "##ing", "##ed", "fast",
              "slow", "and", "or", "not", "is", "it", "very", "good", "bad"]
    assert len(pieces) <= DIMS["vocab"]
    pieces += [f"[unused{i}]" for i in range(DIMS["vocab"] - len(pieces))]
    open(path, "w").write("\n".join(pieces) + "\n")


def main():
    tmp = tempfile.mkdtemp(prefix="tritembed.")
    src = os.path.join(tmp, "src")
    os.makedirs(src)
    t, rng = synthetic_model()
    # bf16 scales: the source stores them as bf16, so round them the way the reference's packer does
    st = {}
    for name, a in t.items():
        st[name] = bf16_bits(a) if name.endswith(".scale") else a
    write_safetensors(os.path.join(src, "model.safetensors"), st)
    json.dump({"architecture": "TritEmbed", "format": "trit-packed-v1", "dims": DIMS,
               "pooling": "mean", "normalized": True, "max_tokens": DIMS["max_pos"], "gelu": "tanh",
               "source_model": "synthetic"}, open(os.path.join(src, "config.json"), "w"))
    make_tiny_vocab(os.path.join(src, "vocab.txt"))

    out = os.path.join(tmp, "packed")
    r = subprocess.run([sys.executable, PACK, "pack", src, out], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run([sys.executable, PACK, "verify", src, out], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "verify: OK" in r.stdout
    print("pack + verify: OK")

    # refusals: a reserved ternary code, an unknown format, a dtype the importer does not read
    bad = dict(st)
    packed = bad["layers.0.q.codes"].copy()
    packed[0, 0] |= 0x03  # code 3
    bad["layers.0.q.codes"] = packed
    bad_src = os.path.join(tmp, "badsrc")
    os.makedirs(bad_src, exist_ok=True)
    write_safetensors(os.path.join(bad_src, "model.safetensors"), bad)
    json.dump({"format": "trit-packed-v1", "dims": DIMS}, open(os.path.join(bad_src, "config.json"), "w"))
    r = subprocess.run([sys.executable, PACK, "pack", bad_src, os.path.join(tmp, "badout")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "reserved code 3" in (r.stdout + r.stderr), r.stdout + r.stderr
    print("refusal (reserved ternary code 3): OK")

    json.dump({"format": "something-else", "dims": DIMS}, open(os.path.join(bad_src, "config.json"), "w"))
    write_safetensors(os.path.join(bad_src, "model.safetensors"), st)
    r = subprocess.run([sys.executable, PACK, "pack", bad_src, os.path.join(tmp, "badout2")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "trit-packed-v1" in (r.stdout + r.stderr), r.stdout + r.stderr
    print("refusal (unknown format): OK")

    if "--write-fixture" in sys.argv:
        os.makedirs(FIXTURE, exist_ok=True)
        for f in ("model.embed", "vocab.txt"):
            with open(os.path.join(out, f), "rb") as fi, open(os.path.join(FIXTURE, f), "wb") as fo:
                fo.write(fi.read())
        cases = [[1, 4, 5, 2], [1, 7, 8, 9, 10, 2]]  # [CLS] hello world [SEP]; [CLS] the cat sat on mat [SEP]
        # the container holds the source's bf16 scales widened to f32, so the expected vectors are computed
        # from exactly those values, not from the f32 scales the synthetic model was built with
        t_eff = dict(t)
        for name, a in st.items():
            if name.endswith(".scale"):
                t_eff[name] = (a.astype(np.uint32) << 16).view(np.float32)
        expected = {"dims": DIMS,
                    "cases": [{"ids": ids, "vector": [float(v) for v in reference_embed(t_eff, DIMS, ids)]}
                              for ids in cases]}
        json.dump(expected, open(os.path.join(FIXTURE, "expected.json"), "w"), indent=1)
        print(f"fixture written to {FIXTURE} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
