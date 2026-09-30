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
"""ternary_to_q4_0.py — a ternary GGUF (PrismML PTQ1_0) as Q4_0, bit for bit the same weights.

    tools/ternary_to_q4_0.py IN.gguf OUT.gguf [--check N]

PTQ1_0 (ggml type 143, PrismML's llama.cpp fork) stores 128 weights as trits t in {-1, 0, +1}
(base-3, 5 per byte) with one fp16 scale d: w = t * d. Q4_0 stores 32 weights as nibbles q with
one fp16 scale: w = (q - 8) * d. Writing q = t + 8 and the same fp16 d into each of the four
Q4_0 blocks of a group gives exactly the same weights, so every backend with a Q4_0 matmul
(HRX among them) runs the model without a ternary kernel. The file is bigger (4.5 instead of
1.75 bits per weight), which is what a native ternary kernel wins back.

Everything else is copied as it is, metadata included: PrismML's prism.hadamard.* keys say
which weights were stored in a Walsh-Hadamard-rotated basis, and the loader still has to rotate
the activations to match. general.file_type becomes MOSTLY_Q4_0 and onebit.ternary_q4_0 = 128
(the source group size) marks the file. --check N decodes N random groups per tensor back from
the written file and compares them with the source (all of them are compared by default: 0 = all).
"""
import argparse
import os
import struct
import sys

import numpy as np

GGUF_MAGIC = b"GGUF"
F32, F16, Q4_0, BF16, PTQ1_0 = 0, 1, 2, 30, 143
# bytes per block, elements per block
SIZES = {F32: (4, 1), F16: (2, 1), Q4_0: (18, 32), BF16: (2, 1), PTQ1_0: (28, 128)}
MOSTLY_Q4_0 = 2
# the order PTQ1_0 stores its 128 values in (ggml-quants.c, dequantize_row_ptq1_0): bytes
# 0..15 give 5 trits each (value n*16 + m from byte m, trit n), bytes 16..23 the next 40,
# the two qh bytes 4 trits each
POW3 = np.array([1, 3, 9, 27, 81, 243], dtype=np.uint16)


def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n)


def skip_value(f, t):
    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in fixed:
        return f.read(fixed[t])
    if t == 8:
        return read_str(f)
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if et in fixed:
            f.read(fixed[et] * n)
        else:
            for _ in range(n):
                skip_value(f, et)
        return None
    raise ValueError(f"unknown GGUF value type {t}")


def parse(path):
    """-> (version, [(key, type, raw bytes of type+value)], [tensor dicts], alignment, data offset)"""
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        kvs, alignment = [], 32
        for _ in range(n_kv):
            key = read_str(f).decode()
            start = f.tell()
            (t,) = struct.unpack("<I", f.read(4))
            v = skip_value(f, t)
            end = f.tell()
            f.seek(start)
            raw = f.read(end - start)
            if key == "general.alignment":
                alignment = struct.unpack("<I", v)[0]
            kvs.append((key, t, raw))
        tensors = []
        for _ in range(n_tensors):
            name = read_str(f).decode()
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            t, off = struct.unpack("<IQ", f.read(12))
            tensors.append({"name": name, "dims": dims, "type": t, "offset": off})
        data = (f.tell() + alignment - 1) // alignment * alignment
    return version, kvs, tensors, alignment, data


def nbytes(t, dims):
    if t not in SIZES:
        raise ValueError(f"ggml type {t} is not handled by this tool")
    bs, be = SIZES[t]
    n = int(np.prod(dims))
    assert n % be == 0
    return n // be * bs


def ptq1_0_trits(raw):
    """raw PTQ1_0 blocks (n, 28) uint8 -> trits (n, 128) int8 in -1..1, fp16 scales (n,) as uint16"""
    qs, qh = raw[:, :24].astype(np.uint16), raw[:, 24:26].astype(np.uint16)
    d = raw[:, 26:28].copy().view(np.uint16)[:, 0]

    def digits(b, count):  # (n, k) bytes -> (n, count, k) trits, trit n of each byte
        q = (b[:, None, :] * POW3[:count, None]) & 0xFF
        return ((q * 3) >> 8).astype(np.int8) - 1

    a = digits(qs[:, :16], 5).reshape(len(raw), 80)
    b = digits(qs[:, 16:24], 5).reshape(len(raw), 40)
    c = digits(qh, 4).reshape(len(raw), 8)
    return np.concatenate([a, b, c], axis=1), d


def q4_0_from_trits(t, d):
    """trits (n, 128), scales (n,) uint16 -> Q4_0 blocks (4n, 18) uint8 with the same weights"""
    n = len(t)
    q = (t.reshape(n * 4, 32) + 8).astype(np.uint8)
    out = np.empty((n * 4, 18), dtype=np.uint8)
    out[:, 0:2] = np.repeat(d, 4).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return out


def q4_0_decode(raw):
    """Q4_0 blocks (n, 18) -> (values (n, 32) int8 as q - 8, fp16 scales (n,) uint16)"""
    d = raw[:, 0:2].copy().view(np.uint16)[:, 0]
    qs = raw[:, 2:]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.int8) - 8
    return q, d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--check", type=int, default=0, help="groups per tensor to verify after writing (0 = all)")
    a = ap.parse_args()

    version, kvs, tensors, alignment, data = parse(a.src)
    if not any(t["type"] == PTQ1_0 for t in tensors):
        sys.exit(f"{a.src}: no PTQ1_0 tensors")
    for t in tensors:
        t["out_type"] = Q4_0 if t["type"] == PTQ1_0 else t["type"]
        t["out_bytes"] = nbytes(t["out_type"], t["dims"])
        t["in_bytes"] = nbytes(t["type"], t["dims"])

    # metadata: the source keys, general.file_type -> MOSTLY_Q4_0, plus our stamp
    out_kvs = []
    for key, typ, raw in kvs:
        if key == "general.file_type":
            raw = struct.pack("<I", 4) + struct.pack("<I", MOSTLY_Q4_0)
        if key == "onebit.ternary_q4_0":
            continue
        out_kvs.append((key, raw))
    out_kvs.append(("onebit.ternary_q4_0", struct.pack("<I", 4) + struct.pack("<I", 128)))

    header = bytearray()
    header += GGUF_MAGIC + struct.pack("<I", version) + struct.pack("<QQ", len(tensors), len(out_kvs))
    for key, raw in out_kvs:
        k = key.encode()
        header += struct.pack("<Q", len(k)) + k + raw
    off = 0
    for t in tensors:
        k = t["name"].encode()
        header += struct.pack("<Q", len(k)) + k + struct.pack("<I", len(t["dims"]))
        header += struct.pack(f"<{len(t['dims'])}Q", *t["dims"]) + struct.pack("<IQ", t["out_type"], off)
        t["out_offset"] = off
        off += (t["out_bytes"] + alignment - 1) // alignment * alignment
    pad = (-len(header)) % alignment
    header += b"\0" * pad
    out_data = len(header)

    tmp = a.dst + ".part"
    src = np.memmap(a.src, dtype=np.uint8, mode="r")
    with open(tmp, "wb") as f:
        f.write(header)
        for i, t in enumerate(tensors):
            f.seek(out_data + t["out_offset"])
            blob = src[data + t["offset"]: data + t["offset"] + t["in_bytes"]]
            if t["type"] == PTQ1_0:
                chunk = 1 << 16  # groups per step: bounded memory for the 248K-row output head
                raw = blob.reshape(-1, 28)
                for s in range(0, len(raw), chunk):
                    tr, d = ptq1_0_trits(np.asarray(raw[s:s + chunk]))
                    f.write(q4_0_from_trits(tr, d).tobytes())
            else:
                f.write(blob.tobytes())
            padded = (t["out_bytes"] + alignment - 1) // alignment * alignment
            f.write(b"\0" * (padded - t["out_bytes"]))
            print(f"\r[{i + 1}/{len(tensors)}] {t['name']:<40}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    # read the written file back: every Q4_0 group must decode to the source trits and scale
    dst = np.memmap(tmp, dtype=np.uint8, mode="r")
    rng = np.random.default_rng(0)
    checked = 0
    for t in tensors:
        s = src[data + t["offset"]: data + t["offset"] + t["in_bytes"]]
        o = dst[out_data + t["out_offset"]: out_data + t["out_offset"] + t["out_bytes"]]
        if t["type"] != PTQ1_0:
            if not np.array_equal(s, o):
                sys.exit(f"{t['name']}: copied bytes differ")
            continue
        s, o = s.reshape(-1, 28), o.reshape(-1, 4 * 18)
        idx = np.arange(len(s)) if a.check <= 0 else rng.choice(len(s), min(a.check, len(s)), replace=False)
        for c in range(0, len(idx), 1 << 16):
            sel = np.sort(idx[c:c + (1 << 16)])
            tr, d = ptq1_0_trits(np.asarray(s[sel]))
            q, qd = q4_0_decode(np.asarray(o[sel]).reshape(-1, 18))
            if not (np.array_equal(q.reshape(len(sel), 128), tr) and np.array_equal(qd.reshape(len(sel), 4), np.repeat(d, 4).reshape(-1, 4))):
                sys.exit(f"{t['name']}: Q4_0 does not reproduce the PTQ1_0 weights")
            checked += len(sel)
    os.replace(tmp, a.dst)
    n_t = sum(t["type"] == PTQ1_0 for t in tensors)
    print(f"{a.dst}: {n_t} PTQ1_0 tensors -> Q4_0, {checked} groups verified identical, "
          f"{os.path.getsize(a.dst) / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
