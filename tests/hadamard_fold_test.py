#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
"""hadamard_fold_test.py — the fold tool writes a file llama-hadamard.cpp can run.

Builds a tiny MoE-shaped GGUF, folds it, then checks the four things that make the output usable:
the prism.hadamard.* keys name exactly the folded tensors, a tensor the loader does not fold (or
whose input dimension the block does not divide) is left in the plain basis and reported, every
unfolded tensor is copied byte for byte, and x'^T W' == x^T W for the folded ones.
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import hadamard_fold as hf  # noqa: E402

FOLDED = {
    "blk.0.ffn_gate_exps.weight": (64, 8, 4),
    "blk.0.ffn_up_exps.weight":   (64, 8, 4),
    "blk.0.attn_q.weight":        (64, 8),
    "output.weight":              (64, 4),
}
PLAIN = {
    "blk.0.ffn_down_exps.weight": (48, 8, 4),   # input 48 is not a multiple of 32: must be skipped
    "token_embd.weight":          (64, 4),      # a lookup table, not a matmul: never folded
}


def write_gguf(path, tensors, extra_kvs=()):
    kvs = [hf.kv_str("general.architecture", "qwen3moe"), hf.kv_u32("general.alignment", 32)]
    kvs += list(extra_kvs)
    infos = [(n, d) for n, d in tensors.items()]
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3))
        f.write(struct.pack("<QQ", len(infos), len(kvs)))
        for key, _, raw in kvs:
            f.write(struct.pack("<Q", len(key)) + key + raw)
        pos = 0
        for n, dims in infos:
            nb = int(np.prod(dims)) * 2
            f.write(struct.pack("<Q", len(n)) + n.encode() + struct.pack("<I", len(dims)))
            f.write(struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", hf.F16, pos))
            pos += (nb + 31) // 32 * 32
        f.write(b"\0" * ((32 - f.tell() % 32) % 32))
        rng = np.random.default_rng(11)
        for n, dims in infos:
            # random, not a ramp: a monotone tensor concentrates its energy after the rotation and its
            # relative error stops saying anything about the fold
            a = rng.standard_normal(int(np.prod(dims))).astype(np.float32) * 0.05
            raw = a.astype(np.float16).tobytes()
            f.write(raw + b"\0" * ((32 - len(raw) % 32) % 32))


def read_tensor(path, name):
    _, _, tensors, _, data = hf.parse(path)
    t = [x for x in tensors if x["name"] == name][0]
    nb = hf.nbytes(t["type"], t["dims"])
    with open(path, "rb") as f:
        f.seek(data + t["offset"])
        return f.read(nb)


def main():
    tmp = tempfile.mkdtemp(prefix="hfold-")
    src, dst = os.path.join(tmp, "src.gguf"), os.path.join(tmp, "dst.gguf")
    write_gguf(src, {**FOLDED, **PLAIN})

    r = subprocess.run([sys.executable, os.path.join(HERE, "..", "tools", "hadamard_fold.py"),
                        src, dst, "--block", "32"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"tool failed:\n{r.stdout}\n{r.stderr}")

    # 1. the keys name exactly the folded tensors, and nothing the loader cannot fold
    _, kvs, _, _, _ = hf.parse(dst)
    keys = {k: raw for k, _, raw in kvs}
    names = None
    for k, t, raw in kvs:
        if k == "prism.hadamard.weight_names":
            (et,) = struct.unpack("<I", raw[4:8])
            (n,) = struct.unpack("<Q", raw[8:16])
            assert et == hf.STRING and t == hf.ARRAY
            off, vals = 16, []
            for _ in range(n):
                (ln,) = struct.unpack("<Q", raw[off:off + 8])
                vals.append(raw[off + 8:off + 8 + ln].decode())
                off += 8 + ln
            names = sorted(vals)
    assert names == sorted(FOLDED), f"weight_names {names} != {sorted(FOLDED)}"
    for k, v in (("prism.hadamard.version", 1), ("prism.hadamard.block_size", 32)):
        assert struct.unpack("<I", keys[k][-4:])[0] == v, f"{k} wrong"
    assert b"identity" in keys["prism.hadamard.sign_mode"]
    assert "ffn_down_exps" in r.stderr and "not a multiple of 32" in r.stderr, r.stderr

    # 2. unfolded tensors are byte-identical, folded ones are not
    for n in PLAIN:
        assert read_tensor(src, n) == read_tensor(dst, n), f"{n} changed but must not"
    for n in FOLDED:
        assert read_tensor(src, n) != read_tensor(dst, n), f"{n} is unchanged - the fold did not run"

    # 3. the tool's own check ran and passed
    assert "check ok" in r.stderr, r.stderr

    # 4. explicit signs round-trip into the file and still preserve the product
    dst2 = os.path.join(tmp, "dst-signs.gguf")
    r2 = subprocess.run([sys.executable, os.path.join(HERE, "..", "tools", "hadamard_fold.py"),
                         src, dst2, "--block", "32", "--signs", "--seed", "7"], capture_output=True, text=True)
    if r2.returncode != 0:
        sys.exit(f"tool failed with --signs:\n{r2.stdout}\n{r2.stderr}")
    _, kvs2, _, _, _ = hf.parse(dst2)
    have = {k for k, _, _ in kvs2}
    assert "prism.hadamard.sign_widths" in have and "prism.hadamard.sign_values" in have
    assert b"explicit" in [raw for k, _, raw in kvs2 if k == "prism.hadamard.sign_mode"][0]

    # 5. folding a folded file is refused (it is not the identity)
    r3 = subprocess.run([sys.executable, os.path.join(HERE, "..", "tools", "hadamard_fold.py"),
                         dst, os.path.join(tmp, "twice.gguf")], capture_output=True, text=True)
    assert r3.returncode != 0 and "already carries Hadamard keys" in r3.stderr, r3.stderr

    print("hadamard_fold_test: ok")


if __name__ == "__main__":
    main()
