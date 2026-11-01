#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# The MTP head in our "pb16i" form: codes x per-(32 K, column) magnitudes, no per-group bias.
#
# Input is the container `flm-head-import` wrote (Q4_K cells copied unchanged from FastFlowLM's head), so the
# weights are decoded with the decoder already verified bit-for-bit against FastFlowLM's own
# (amd-oss/npu-re docs/FACTS-flm-q4nx-container-and-mtp-head.md §5). pb16i has no additive term, while the
# source blocks do (w = q4*S + M), so this conversion is lossy by construction: the residual is measured per
# tensor and written to the report rather than smoothed over.
#
#   tools/flm_head_pb16i.py <container dir from flm-head-import> <out dir>
#
# Writes <tensor>.pb16i + <tensor>.ref beside a manifest.json with the measured error, using the image layout
# and the image writer that are already in this repository (tools/q4nx_pb16i.py, npu/forward/include/q4nx_pb16i.h).
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q4nx_pb16i as P  # pack / side_table / w_bytes: the image contract, unchanged

CELL, ROWS, COLS = 4736, 32, 256


def bf16(u16):
    return ((u16.astype(np.uint32) << 16).view(np.float32))


def decode_cell(cell):
    """One 4,736-byte cell -> float32 [32, 256]: w = q4 * (scale * S) + min * M (FACTS §4.1, verified)."""
    sc = cell[0:256].reshape(8, 32).astype(np.float32)
    mn = cell[256:512].reshape(8, 32).astype(np.float32)
    nib = cell[512:4608]
    d = bf16(np.frombuffer(cell[4608:4672].tobytes(), dtype=np.uint16))
    dmin = bf16(np.frombuffer(cell[4672:4736].tobytes(), dtype=np.uint16))
    n = np.arange(32)
    k = np.arange(256)
    byte = nib[k[:, None] * 16 + n[None, :] // 2]
    q4 = np.where(n[None, :] % 2 == 0, byte & 0x0F, byte >> 4).astype(np.float32)
    g = k // 32
    return q4 * (sc[g] * d[None, :]) + mn[g] * dmin[None, :]        # [k, n]


def f16_bits(x):
    return np.asarray(x, dtype=np.float16).view(np.uint16)


def main():
    if len(sys.argv) < 3:
        print(__doc__ or "usage: flm_head_pb16i.py <container dir> <out dir>")
        return 2
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    with open(src / "model.q4nx", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
    man = []
    for name, t in hdr.items():
        if name == "__metadata__" or t["dtype"] != "I8":
            continue
        nrb, ncb, cell_bytes = t["shape"]
        assert cell_bytes == CELL, (name, cell_bytes)
        K, N = ncb * 256, nrb * 32                      # pb16i: K is the reduction dim (our columns), N the out
        lo, _ = t["data_offsets"]
        with open(src / "model.q4nx", "rb") as f:
            f.seek(base + lo)
            cells = np.frombuffer(f.read(nrb * ncb * CELL), dtype=np.uint8).reshape(nrb * ncb, CELL)
        w = np.empty((K, N), dtype=np.float32)          # [in, out]
        for r in range(nrb):
            for c in range(ncb):
                w[c * 256:(c + 1) * 256, r * 32:(r + 1) * 32] = decode_cell(cells[r * ncb + c])
        if not P.fits(K, N):
            print(f"  {name}: K={K} N={N} is not a pb16i geometry, skipped")
            continue
        # one magnitude per 128-K block per column: d = max|w| / 8 keeps every code inside [-8, 7]
        blocks = K // 128
        d = np.abs(w.reshape(blocks, 128, N)).max(axis=1) / 8.0
        d = np.where(d > 0, d, 1.0)
        # pb16i stores one int16 per (32 K, column) rescaled into that column's exponent window, so a column
        # whose 128-K blocks differ by more than ~2^13 in magnitude cannot be represented: raise the small
        # blocks to a floor (coarser codes) instead of losing the tensor. The cost is in the reported residual.
        floor = d.max(axis=0) / 2.0 ** 4
        d = np.maximum(d, floor[None, :]).astype(np.float16)
        d32 = d.astype(np.float32)
        codes = np.rint(w / np.repeat(d32, 128, axis=0)).clip(-8, 7).astype(np.int8)
        img, e_ref, mode = P.pack(codes, f16_bits(d), K, N)
        stem = name.replace("model.mtp.", "").replace(".weight", "")
        (out / f"{stem}.pb16i").write_bytes(img.tobytes())
        (out / f"{stem}.ref").write_bytes(P.side_table(e_ref, mode))
        # the residual this form costs: the additive M*min term cannot be carried at all
        approx = codes.astype(np.float32) * np.repeat(d32, 128, axis=0)
        resid = np.abs(approx - w)
        rel = float(resid.sum() / (np.abs(w).sum() or 1.0))
        man.append({"tensor": name, "image": f"{stem}.pb16i", "ref": f"{stem}.ref",
                    "K": K, "N": N, "mode": int(mode), "bytes": int(img.size),
                    "max_abs_residual": float(resid.max()), "rel_l1_residual": rel,
                    "mean_abs_w": float(np.abs(w).mean())})
        print(f"  {name:52s} {K}x{N} -> {img.size:>10,} B  mode {mode}  "
              f"max |resid| {resid.max():.3e}  rel-L1 {rel:.4e}")
        del w, codes, img
    (out / "manifest.json").write_text(json.dumps(
        {"source": str(src), "form": "pb16i", "cell": "codes x ms, no per-group bias",
         "note": "the source blocks are asymmetric (w = q4*S + M); pb16i carries no bias, so the measured "
                 "residual here is the additive term plus 4-bit rounding, by construction",
         "tensors": man}, indent=1) + "\n")
    print(f"\n{len(man)} images -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
