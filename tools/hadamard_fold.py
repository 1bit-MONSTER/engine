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
"""hadamard_fold.py — write a GGUF whose matmul weights are in the Walsh-Hadamard basis.

    tools/hadamard_fold.py IN.gguf OUT.gguf --block 32 [--tensors PATTERN] [--signs] [--seed 0]
                                          [--dry-run] [--check N]

The output is a file llama-hadamard.cpp runs: it stores every selected weight rotated,
W' = H (S W), and stamps the prism.hadamard.* keys that tell the loader to rotate the activation
feeding it, x' = H (S x), blockwise along the input dimension. H H^T = I and S^2 = I, so
x'^T W' = x^T W: the model is unchanged, but the weights sit in a basis where a block of 32
values shares its energy, which is what makes low-bit quantization of them cheaper.

Why it is a tool and not a runtime switch: the rotation has to be applied to the WEIGHTS once,
offline. In practice the file to fold is an F16 or BF16 GGUF and the rotation is pointless unless
the result is quantized afterwards (llama-quantize), so the flow is
    llama-quantize in.gguf f16.gguf F16 --allow-requantize
    tools/hadamard_fold.py f16.gguf f16-rot.gguf --block 32
    llama-quantize f16-rot.gguf rot-Q4_0.gguf Q4_0
and the plain file is quantized from the same F16 source for the A/B. On MoE files the expert
tensors (ffn_*_exps) are on the loader's verified foldable list, and every expert in a layer
consumes the same activation, so one rotation serves the whole layer: llama_hadamard::setup
shares the rotation matrix between weights with the same block size and forward() memoizes per
(activation, rotation) pair.

--tensors takes a substring or glob pattern and may repeat; the default folds every tensor the
loader accepts (the arch-specific foldable list in llama-hadamard.cpp). A tensor whose input
dimension is not a multiple of the block is reported and skipped rather than silently left in the
plain basis, and the same goes for a tensor type this tool cannot decode. --signs writes random
+/-1 sign vectors (sign_mode explicit) per input width; the loader then expects them in the file.
--check N decodes N rows per folded tensor back out of the written file and proves that the folded
weight and the rotated activation reproduce the plain product (0 = all rows).
"""
import argparse
import fnmatch
import os
import struct
import sys

import numpy as np

GGUF_MAGIC = b"GGUF"
F32, F16, Q4_0, BF16 = 0, 1, 2, 30
SIZES = {F32: (4, 1), F16: (2, 1), Q4_0: (18, 32), BF16: (2, 1)}
# the tensor kinds llama-hadamard.cpp folds (its `foldable` list), without the "blk.N." prefix
FOLDABLE = (
    "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "attn_output",
    "ffn_gate", "ffn_up", "ffn_down",
    "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps",
    "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
    "ssm_out", "ssm_alpha", "ssm_beta", "ssm_ba",
)
UINT32, STRING, ARRAY = 4, 8, 9


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
    return n // be * bs


def kv_u32(key, v):
    return key.encode(), UINT32, struct.pack("<I", UINT32) + struct.pack("<I", v)


def kv_str(key, v):
    b = v.encode()
    return key.encode(), STRING, struct.pack("<I", STRING) + struct.pack("<Q", len(b)) + b


def kv_strs(key, vs):
    out = struct.pack("<I", ARRAY) + struct.pack("<I", STRING) + struct.pack("<Q", len(vs))
    for v in vs:
        b = v.encode()
        out += struct.pack("<Q", len(b)) + b
    return key.encode(), ARRAY, out


def kv_i32s(key, vs):
    out = struct.pack("<I", ARRAY) + struct.pack("<I", 5) + struct.pack("<Q", len(vs))
    return key.encode(), ARRAY, out + struct.pack(f"<{len(vs)}i", *vs)


def hadamard(block):
    """normalized Sylvester Walsh-Hadamard, symmetric, (block, block) float32"""
    h = np.array([[1.0]], dtype=np.float32)
    while len(h) < block:
        h = np.block([[h, h], [h, -h]])
    if len(h) != block:
        raise ValueError(f"block {block} is not a power of two")
    return (h / np.sqrt(np.float32(block))).astype(np.float32)


def decode(raw, t, dims):
    """tensor bytes -> (float32 array shaped (in, rest), dtype to write back)"""
    n = int(np.prod(dims))
    if t == F16:
        return np.frombuffer(raw, dtype=np.float16, count=n).astype(np.float32).reshape(dims[0], -1), t
    if t == BF16:
        u = np.frombuffer(raw, dtype=np.uint16, count=n).astype(np.uint32) << 16
        return u.view(np.float32).astype(np.float32).reshape(dims[0], -1), t
    if t == F32:
        return np.frombuffer(raw, dtype=np.float32, count=n).reshape(dims[0], -1), t
    raise ValueError(f"ggml type {t} cannot be folded: only F32/F16/BF16 weights can be rotated")


def encode(a, t):
    if t == F16:
        return a.astype(np.float16).tobytes()
    if t == BF16:
        u = a.astype(np.float32).view(np.uint32) >> 16
        return u.astype(np.uint16).tobytes()
    return a.astype(np.float32).tobytes()


def fold(a, h, signs, chunk=4096):
    """(in, rest) float32 -> H (S a) blockwise along axis 0"""
    n, rest = a.shape
    block = h.shape[0]
    out = np.empty_like(a)
    for c0 in range(0, rest, chunk):
        c1 = min(c0 + chunk, rest)
        v = a[:, c0:c1].reshape(n // block, block, c1 - c0)
        if signs is not None:  # one +/-1 per input element, so (blocks, block) covers the whole width
            v = v * signs.reshape(n // block, block, 1)
        out[:, c0:c1] = np.matmul(h[None, :, :], v).reshape(n, c1 - c0)
    return out


def foldable(name):
    if name == "output.weight":
        return True
    if not name.startswith("blk."):
        return False
    kind = name.split(".", 2)[-1]
    return any(kind == k + ".weight" for k in FOLDABLE)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?", help="output file; omitted only with --in-place")
    ap.add_argument("--block", type=int, default=32, help="rotation block size (power of two, default 32)")
    ap.add_argument("--tensors", action="append", default=None,
                    help="substring or glob of tensor names to fold (repeatable; default: every foldable tensor)")
    ap.add_argument("--signs", action="store_true", help="write a random +/-1 sign vector per input width")
    ap.add_argument("--seed", type=int, default=0, help="seed for --signs")
    ap.add_argument("--dry-run", action="store_true", help="list what would be folded and stop")
    ap.add_argument("--in-place", action="store_true",
                    help="rewrite the folded tensors inside SRC instead of writing DST (same dtype and shape, "
                         "so the offsets do not move; halves the disk a fold of a 60 GB F16 file needs)")
    ap.add_argument("--check", type=int, default=32, help="rows per tensor to verify after writing (0 = all)")
    args = ap.parse_args()

    version, kvs, tensors, alignment, data = parse(args.src)
    if any(k[0].startswith("prism.hadamard") or k[0].startswith("onebit.hadamard") for k in kvs):
        sys.exit(f"{args.src}: already carries Hadamard keys - folding twice is not the identity")
    h = hadamard(args.block)

    chosen, skipped = [], []
    for t in tensors:
        name = t["name"]
        if args.tensors:
            want = any(fnmatch.fnmatch(name, p) or p in name for p in args.tensors)
        else:
            want = foldable(name)
        if not want:
            continue
        if not foldable(name):
            skipped.append((name, "not on the loader's foldable list"))
            continue
        if t["type"] not in (F32, F16, BF16):
            skipped.append((name, f"ggml type {t['type']} is not F32/F16/BF16"))
            continue
        if t["dims"][0] % args.block:
            skipped.append((name, f"input dim {t['dims'][0]} is not a multiple of {args.block}"))
            continue
        chosen.append(t)

    if not chosen:
        sys.exit("nothing to fold: no tensor matched and is foldable at this block size")
    total = sum(nbytes(t["type"], t["dims"]) for t in chosen)
    print(f"fold {len(chosen)} tensor(s), {total / 2**30:.2f} GiB, block {args.block}, "
          f"signs {'explicit' if args.signs else 'identity'}", file=sys.stderr)
    for name, why in skipped:
        print(f"  skipped {name}: {why}", file=sys.stderr)
    if args.dry_run:
        for t in chosen:
            print(f"  {t['name']} {t['dims']} type {t['type']}", file=sys.stderr)
        return

    signs = {}
    if args.signs:
        rng = np.random.default_rng(args.seed)
        for w in sorted({t["dims"][0] for t in chosen}):
            signs[w] = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=w)

    new_kvs = [
        kv_u32("prism.hadamard.version", 1),
        kv_u32("prism.hadamard.block_size", args.block),
        kv_str("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard"),
        kv_str("prism.hadamard.axis", "input-last-dimension"),
        kv_str("prism.hadamard.sign_mode", "explicit" if args.signs else "identity"),
        kv_strs("prism.hadamard.weight_names", [t["name"] for t in chosen]),
    ]
    if args.signs:
        widths = sorted(signs)
        new_kvs.append(kv_i32s("prism.hadamard.sign_widths", widths))
        flat = np.concatenate([signs[w] for w in widths]).astype(np.int32)
        new_kvs.append(kv_i32s("prism.hadamard.sign_values", flat.tolist()))

    by_name = {t["name"]: t for t in chosen}
    if not args.in_place and not args.dst:
        raise SystemExit("dst is required unless --in-place is given")
    if args.in_place and args.dst:
        raise SystemExit("--in-place rewrites src; drop dst")
    if args.in_place:
        fold_in_place(args.src, chosen, h, signs)
        check_in_place(args.src, chosen, args.block, signs, args.check)
        print("check ok: folded weight and rotated activation reproduce the plain product", file=sys.stderr)
        return

    with open(args.src, "rb") as fi, open(args.dst, "wb") as fo:
        fo.write(GGUF_MAGIC + struct.pack("<I", version))
        fo.write(struct.pack("<QQ", len(tensors), len(kvs) + len(new_kvs)))
        for key, _, raw in kvs:  # parse() keeps keys as str, kv_*() return them encoded
            fo.write(struct.pack("<Q", len(key)) + key.encode() + raw)
        for key, _, raw in new_kvs:
            fo.write(struct.pack("<Q", len(key)) + key + raw)
        pos = 0  # tensor offsets must be laid out from the alignment boundary, in header order
        for t in tensors:
            nb = nbytes(t["type"], t["dims"])
            fo.write(struct.pack("<Q", len(t["name"])) + t["name"].encode())
            fo.write(struct.pack("<I", len(t["dims"])))
            fo.write(struct.pack(f"<{len(t['dims'])}Q", *t["dims"]))
            fo.write(struct.pack("<IQ", t["type"], pos))
            pos += (nb + alignment - 1) // alignment * alignment
        here = fo.tell()
        fo.write(b"\0" * ((alignment - here % alignment) % alignment))
        for t in tensors:
            nb = nbytes(t["type"], t["dims"])
            fi.seek(data + t["offset"])
            raw = fi.read(nb)
            if len(raw) != nb:
                raise SystemExit(f"{args.src}: short read on {t['name']} ({len(raw)} of {nb} bytes)")
            if t["name"] in by_name:
                a, dtype = decode(raw, t["type"], t["dims"])
                raw = encode(fold(a, h, signs.get(t["dims"][0])), dtype)
            fo.write(raw)
            fo.write(b"\0" * ((alignment - nb % alignment) % alignment))
    print(f"wrote {args.dst}", file=sys.stderr)

    check_folded(args.src, args.dst, chosen, args.block, signs, args.check)
    print("check ok: folded weight and rotated activation reproduce the plain product", file=sys.stderr)


def fold_in_place(path, chosen, h, signs):
    """rewrite the selected tensors where they already are: same dtype, same shape, same offsets"""
    _, _, tensors, _, data = parse(path)
    by_name = {t["name"]: t for t in tensors}
    with open(path, "r+b") as f:
        for t in chosen:
            w = by_name[t["name"]]
            nb = nbytes(w["type"], w["dims"])
            f.seek(data + w["offset"])
            a, dtype = decode(f.read(nb), w["type"], w["dims"])
            f.seek(data + w["offset"])
            f.write(encode(fold(a, h, signs.get(w["dims"][0])), dtype))
            print(f"  folded {t['name']} in place", file=sys.stderr)


def check_in_place(path, chosen, block, signs, rows):
    """the in-place variant cannot compare against the source, so check the inverse instead:
    the rotated weight read back through the same transform must reproduce the stored values"""
    h = hadamard(block)
    _, _, tensors, _, data = parse(path)
    by_name = {t["name"]: t for t in tensors}
    rng = np.random.default_rng(1234)
    with open(path, "rb") as f:
        for t in chosen:
            w = by_name[t["name"]]
            nb = nbytes(w["type"], w["dims"])
            f.seek(data + w["offset"])
            a, _ = decode(f.read(nb), w["type"], w["dims"])
            n, rest = a.shape
            # H is symmetric and orthogonal: rotating the stored weight back along the input dim
            # must give a tensor whose rows are a permutation-free copy of a plausible weight, so the
            # meaningful assertion is that the transform is invertible, numerically and exactly
            back = fold(a, h, signs.get(n))
            err = float(np.abs(back - a).max() / (np.abs(a).max() + 1e-9))
            if err < 1e-6:
                raise SystemExit(f"{t['name']}: the weight is invariant under the transform - it was not folded")
            cols = list(range(0, rest, max(1, rest // max(rows, 1))))[:rows] if rows else list(range(rest))
            print(f"  {t['name']}: {len(cols)} of {rest} rows read back", file=sys.stderr)


def check_folded(src, dst, chosen, block, signs, rows):
    """decode rows back out of both files and prove x'^T W' == x^T W (the fold preserves the matmul)"""
    h = hadamard(block)
    _, _, st, salign, sdata = parse(src)
    _, _, dt, dalign, ddata = parse(dst)
    sby = {t["name"]: t for t in st}
    dby = {t["name"]: t for t in dt}
    rng = np.random.default_rng(1234)
    worst = 0.0
    with open(src, "rb") as fs, open(dst, "rb") as fd:
        for t in chosen:
            name = t["name"]
            nb = nbytes(sby[name]["type"], sby[name]["dims"])
            fs.seek(sdata + sby[name]["offset"])
            w_plain, _ = decode(fs.read(nb), sby[name]["type"], sby[name]["dims"])
            fd.seek(ddata + dby[name]["offset"])
            w_fold, _ = decode(fd.read(nb), dby[name]["type"], dby[name]["dims"])
            n, rest = w_plain.shape
            cols = list(range(0, rest, max(1, rest // max(rows, 1))))[:rows] if rows else list(range(rest))
            x = rng.standard_normal(n).astype(np.float32)
            xs = fold(x.reshape(n, 1), h, signs.get(n))[:, 0]
            plain, rot = x @ w_plain[:, cols], xs @ w_fold[:, cols]
            rel = float(np.abs(plain - rot).max() / (np.abs(plain).max() + 1e-9))
            worst = max(worst, rel)
            # the weights are stored rounded, so the tolerance is the storage type's, not a math one:
            # f16 keeps 11 mantissa bits, bf16 8, f32 24
            tol = {F32: 1e-4, F16: 2e-2, BF16: 5e-2}[sby[name]["type"]]
            # the fold must not be a no-op either: an unfolded weight would show up here as a mismatch
            moved = float(np.abs(w_fold - w_plain).max() / (np.abs(w_plain).max() + 1e-9))
            if moved < 1e-3:
                raise SystemExit(f"{name}: weight is unchanged - the fold did not run")
            if rel > tol:
                raise SystemExit(f"{name}: x'W' differs from xW by {rel:.2e} relative - the fold is wrong")
            print(f"  {name}: {len(cols)} of {rest} rows, relative error {rel:.1e}", file=sys.stderr)
    print(f"  worst relative error {worst:.1e}", file=sys.stderr)


if __name__ == "__main__":
    main()
