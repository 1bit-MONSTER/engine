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
"""tests/ternary_to_q4_0_test.py — tools/ternary_to_q4_0.py on a small made-up PTQ1_0 GGUF.

The PTQ1_0 blocks are written here with PrismML's own encoder (quantize_row_ptq1_0_ref in its
ggml-quants.c, ported line by line), not with the tool's decoder, and the Q4_0 output is read
back with Q4_0's definition (w = (q - 8) * d): every weight must come back exactly, the other
tensors byte for byte, the metadata unchanged apart from general.file_type and the stamp.
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools", "ternary_to_q4_0.py")
STAGES = (32, 16, 8)


def ptq1_0_block(x):
    """PrismML's quantize_row_ptq1_0_ref for one block of 128 floats"""
    d = np.float16(np.abs(x).max())
    idv = 1.0 / float(d) if d else 0.0
    qs, j, off = [0] * 24, 0, 0
    for c in STAGES:
        while j + c <= 24:
            for m in range(c):
                q = 0
                for n in range(5):
                    q = q * 3 + int(np.round(x[off + m + n * c] * idv)) + 1
                qs[j + m] = (q * 256 + 242) // 243
            off += 5 * c
            j += c
    qh = []
    for h in range(2):
        q = 0
        for m in range(4):
            q = q * 3 + int(np.round(x[off + h + m * 2] * idv)) + 1
        q *= 3
        qh.append((q * 256 + 242) // 243)
    return bytes(qs) + bytes(qh) + d.tobytes()


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def kv_u32(k, v):
    return s(k) + struct.pack("<II", 4, v)


def read(path):
    """-> (kvs {key: raw value bytes}, tensors {name: (type, dims, bytes)})"""
    b = open(path, "rb").read()
    assert b[:4] == b"GGUF"
    n_t, n_kv = struct.unpack_from("<QQ", b, 8)
    p, kvs = 24, {}
    for _ in range(n_kv):
        (n,) = struct.unpack_from("<Q", b, p); k = b[p + 8:p + 8 + n].decode(); p += 8 + n
        (t,) = struct.unpack_from("<I", b, p); p += 4
        start = p
        if t == 4:
            p += 4
        elif t == 8:
            (n,) = struct.unpack_from("<Q", b, p); p += 8 + n
        elif t == 9:
            et, cnt = struct.unpack_from("<IQ", b, p); p += 12
            for _ in range(cnt):
                (n,) = struct.unpack_from("<Q", b, p); p += 8 + n
        kvs[k] = (t, b[start:p])
    infos = []
    for _ in range(n_t):
        (n,) = struct.unpack_from("<Q", b, p); name = b[p + 8:p + 8 + n].decode(); p += 8 + n
        (nd,) = struct.unpack_from("<I", b, p); p += 4
        dims = struct.unpack_from(f"<{nd}Q", b, p); p += 8 * nd
        t, off = struct.unpack_from("<IQ", b, p); p += 12
        infos.append((name, t, dims, off))
    data = (p + 31) // 32 * 32
    size = {0: lambda n: 4 * n, 2: lambda n: n // 32 * 18, 143: lambda n: n // 128 * 28}
    return kvs, {nm: (t, dims, b[data + off:data + off + size[t](int(np.prod(dims)))]) for nm, t, dims, off in infos}


def main():
    rng = np.random.default_rng(7)
    rows, cols = 3, 256
    trits = rng.integers(-1, 2, size=(rows, cols)).astype(np.float32)
    scales = rng.uniform(0.001, 0.05, size=(rows, cols // 128)).astype(np.float16).astype(np.float32)
    trits[0, :128] = 0  # an all-zero group: scale 0
    w = trits * np.repeat(scales, 128, axis=1)
    w[0, :128] = 0
    ptq = b"".join(ptq1_0_block(w[r, g * 128:(g + 1) * 128]) for r in range(rows) for g in range(cols // 128))
    norm = rng.standard_normal(cols).astype(np.float32).tobytes()

    kvs = [s("general.architecture") + struct.pack("<I", 8) + s("qwen35"), kv_u32("general.file_type", 143),
           s("prism.hadamard.weight_names") + struct.pack("<IIQ", 9, 8, 1) + s("blk.0.ffn_up.weight"),
           kv_u32("prism.hadamard.version", 1)]
    tensors = [("blk.0.ffn_up.weight", 143, (cols, rows), ptq), ("blk.0.attn_norm.weight", 0, (cols,), norm)]
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs)
    off, blobs = 0, b""
    for name, t, dims, blob in tensors:
        head += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", t, off)
        pad = (-len(blob)) % 32
        blobs += blob + b"\0" * pad
        off += len(blob) + pad
    head += b"\0" * ((-len(head)) % 32)

    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "t.gguf"), os.path.join(tmp, "q.gguf")
        open(src, "wb").write(head + blobs)
        subprocess.run([sys.executable, TOOL, src, dst], check=True)
        okv, ot = read(dst)

    fail = []
    t, dims, blob = ot["blk.0.ffn_up.weight"]
    q = np.frombuffer(blob, np.uint8).reshape(-1, 18)
    d = q[:, :2].copy().view(np.float16)[:, 0].astype(np.float32)
    nib = np.concatenate([q[:, 2:] & 15, q[:, 2:] >> 4], axis=1).astype(np.float32) - 8
    back = (nib * d[:, None]).reshape(rows, cols)
    if t != 2 or dims != (cols, rows):
        fail.append(f"ffn_up is type {t} {dims}, want Q4_0 {(cols, rows)}")
    if not np.array_equal(back, w):
        fail.append(f"Q4_0 weights differ from the PTQ1_0 ones in {int((back != w).sum())} places")
    if ot["blk.0.attn_norm.weight"][2] != norm:
        fail.append("the F32 tensor changed")
    if okv["general.file_type"][1] != struct.pack("<I", 2):
        fail.append("general.file_type is not MOSTLY_Q4_0")
    if okv.get("onebit.ternary_q4_0", (0, b""))[1] != struct.pack("<I", 128):
        fail.append("no onebit.ternary_q4_0 = 128 stamp")
    if "prism.hadamard.weight_names" not in okv or "prism.hadamard.version" not in okv:
        fail.append("prism.hadamard keys were dropped")
    for f in fail:
        print("FAIL", f)
    print("FAIL" if fail else "PASS: PTQ1_0 -> Q4_0 exact, other tensors and metadata kept")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
