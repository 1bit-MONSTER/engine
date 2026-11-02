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
"""The t2 chunk kind: 2-bit ternary codes + one fp16 scale per 128-K block per row, 2.125 b/w.

Layout (docs/npu.md, "The t2 chunk kind"; npu/q4nx.h declares it for the decoder):

    A tensor of shape [N, K] (N rows = output channels, K columns = the reduction dim) is a grid of
    32-row x 256-column tiles, tiles row-major over the grid (rt outer, kt inner), each tile
    2176 bytes:

      [0, 2048)    2-bit codes: byte r*64 + c//4, code c%4 at bits 2*(c%4);
                   value 0 -> -1, 1 -> 0, 2 -> +1, 3 reserved (no converter emits it)
      [2048, 2176) IEEE fp16 scales: half at 2048 + r*4 + 2*g, g = c//128
                                                  w[r][c] = code * scale

The scale is fp16, not bf16: every source format already stores it as fp16 (PTQ1_0/TQ1_0/TQ2_0's `d`,
and pb16i's int16 `ms * 2^e_ref`), and a bf16 re-encode is not exact -- measured on the 431 ternary
Bonsai-2 images, 87.5 % of the scales are not bf16-representable (worst |rel| 3.9e-3) while 100 % are
fp16-exact. Same two bytes, so 2176 B and 2.125 b/w do not change.

Converters here: pb16i -> t2 (the engine's integer-exact image, lossless), and the GGUF kinds
PTQ1_0 / TQ1_0 -> t2 (lossless: both are ternary with an fp16 scale) and TQ2_0 -> t2 (lossy, refused
unless asked: TQ2_0's code 3 means +2, which a ternary chunk cannot hold).
"""
import struct

import numpy as np

ROWS, COLS = 32, 256            # a tile is 32 rows x 256 columns
GROUP = 128                     # one scale per 128 columns (K) per row
CODE_BYTES, SCALE_BYTES = 2048, 128
TILE = CODE_BYTES + SCALE_BYTES  # 2176
RESERVED = 3


def nbytes(K, N):
    """Bytes of a t2 tensor, or 0 if the geometry is not whole tiles."""
    if K <= 0 or N <= 0 or K % COLS or N % ROWS:
        return 0
    return (K // COLS) * (N // ROWS) * TILE


def code_offset(r, c):
    """Byte and bit shift of element (r, c) inside a tile."""
    return r * 64 + c // 4, 2 * (c % 4)


def scale_offset(r, c):
    """Byte offset of the fp16 scale that covers (r, c) inside a tile."""
    return CODE_BYTES + r * 4 + 2 * (c // GROUP)


# ── pack / unpack ────────────────────────────────────────────────────────────────────────────────


def pack(codes, scales):
    """codes int8 [N, K] in {-1, 0, 1}, scales fp16 [N, K//128] -> t2 bytes."""
    codes = np.asarray(codes, dtype=np.int8)
    N, K = codes.shape
    if N % ROWS or K % COLS:
        raise ValueError(f"t2 needs N % {ROWS} == 0 and K % {COLS} == 0, got {N}x{K}")
    if codes.min() < -1 or codes.max() > 1:
        raise ValueError("t2 codes are ternary: {-1, 0, +1}")
    scales = np.asarray(scales, dtype=np.float16)
    if scales.shape != (N, K // GROUP):
        raise ValueError(f"scales must be [N, K//128] = {(N, K // GROUP)}, got {scales.shape}")
    rt, kt = N // ROWS, K // COLS
    c2 = (codes.reshape(rt, ROWS, kt, COLS).astype(np.uint8) + 1).astype(np.uint8)   # 0,1,2
    # byte r*64 + c//4 with code c%4 at bits 2*(c%4): pack along K in groups of 4
    quad = c2.reshape(rt, ROWS, kt, COLS // 4, 4)
    packed = (quad[..., 0] | (quad[..., 1] << 2) | (quad[..., 2] << 4) | (quad[..., 3] << 6)).astype(np.uint8)
    # (rt, r, kt, 64) -> per tile (r, kt, 64) -> (rt, kt, r, 64)
    codes_out = packed.transpose(0, 2, 1, 3).reshape(rt * kt, ROWS * 64)
    sc = scales.reshape(rt, ROWS, kt, COLS // GROUP).transpose(0, 2, 1, 3).reshape(rt * kt, ROWS * (COLS // GROUP))
    sc_bits = sc.view(np.uint16) if sc.dtype == np.float16 else sc.astype(np.float16).view(np.uint16)
    out = np.empty((rt * kt, TILE), dtype=np.uint8)
    out[:, :CODE_BYTES] = codes_out
    out[:, CODE_BYTES:] = sc_bits.astype("<u2").view(np.uint8).reshape(rt * kt, SCALE_BYTES)
    return out.reshape(-1).tobytes()


def unpack(raw, K, N):
    """t2 bytes -> float32 [N, K] (the same arithmetic as npu/q4nx.cpp::dequant_tile)."""
    if len(raw) != nbytes(K, N):
        raise ValueError(f"a {N}x{K} t2 tensor is {nbytes(K, N)} B, got {len(raw)}")
    rt, kt = N // ROWS, K // COLS
    t = np.frombuffer(raw, dtype=np.uint8).reshape(rt * kt, TILE)
    packed = t[:, :CODE_BYTES].reshape(rt * kt, ROWS, 64)              # r, c//4
    quad = np.empty((rt * kt, ROWS, 64, 4), np.int8)
    for s in range(4):
        bits = (packed >> (2 * s)) & 3
        # the reserved pattern (3) decodes as 0, never as 2 * scale -- same rule as q4nx.cpp
        quad[..., s] = np.where(bits == RESERVED, 0, bits.astype(np.int8) - 1).astype(np.int8)
    codes = quad.reshape(rt * kt, ROWS, 64 * 4).transpose(0, 1, 2)     # (tile, r, c) in tile order
    sc = t[:, CODE_BYTES:].copy().view("<u2").reshape(rt * kt, ROWS, COLS // GROUP).astype(np.uint16)
    scale = sc.view(np.float16).astype(np.float32)
    codes3 = codes.reshape(rt, kt, ROWS, COLS).transpose(0, 2, 1, 3)
    scale3 = scale.reshape(rt, kt, ROWS, COLS // GROUP).transpose(0, 2, 1, 3)
    w = codes3.astype(np.float32) * np.repeat(scale3, GROUP, axis=3)
    return w.reshape(N, K)


def reserved_count(raw, K, N):
    """How many of the tensor's codes carry the reserved pattern (must be 0)."""
    t = np.frombuffer(raw, dtype=np.uint8).reshape(-1, TILE)
    p = t[:, :CODE_BYTES]
    n = 0
    for s in range(4):
        n += int((((p >> (2 * s)) & 3) == RESERVED).sum())
    return n


# ── pb16i -> t2 ──────────────────────────────────────────────────────────────────────────────────


def from_pb16i(image, ref, K, N):
    """A ternary pb16i image (+ its `<stem>.ref`) as t2 bytes. Lossless: see docs.

    pb16i stores element (k, j) of a 256-K x 32-N tile at code index
    ((k//16)*2 + nb)*256 + (k%16)*16 + cc and its magnitude at 2*((nb*8 + k//32)*16 + cc), with
    one int16 e_ref per column in the .ref side table. A t2 tile is the same 256 K x 32 N block,
    so the conversion is a permutation inside the tile plus (ms * 2^e_ref) -> fp16, which is exact
    when the four per-32 magnitudes of a 128-K block agree and the result is fp16-representable --
    both asserted here rather than assumed.
    """
    KC, NW, PB_TILE, PB_CODE = 256, 32, 4608, 4096
    raw = np.frombuffer(image, dtype=np.uint8)
    tiles = raw.size // PB_TILE
    if tiles * PB_TILE != raw.size or tiles != (K // KC) * (N // NW):
        raise ValueError(f"pb16i image is {raw.size} B, not {tiles} tiles of {PB_TILE}")
    ref = np.frombuffer(ref, dtype=np.uint8)
    if ref.size not in (2 * N, 2 * N + 1):
        raise ValueError(f"pb16i side table is {ref.size} B, not 2*{N} (+1 mode byte)")
    eref = ref[: 2 * N].view("<i2").astype(np.int64)
    t = raw.reshape(tiles, PB_TILE)

    inter = np.empty((tiles, PB_CODE * 2), np.int8)
    lo, hi = t[:, :PB_CODE] & 0x0F, t[:, :PB_CODE] >> 4
    inter[:, 0::2] = np.where(lo >= 8, lo.astype(np.int16) - 16, lo)
    inter[:, 1::2] = np.where(hi >= 8, hi.astype(np.int16) - 16, hi)
    # (tile, kk16, nb, kk%16, cc) -> (tile, kk, jn) with kk = kk16*16 + kk%16, jn = nb*16 + cc
    codes = inter.reshape(tiles, 16, 2, 16, 16).transpose(0, 1, 3, 2, 4).reshape(tiles, KC, NW)
    if codes.min() < -1 or codes.max() > 1:
        raise ValueError("pb16i image is not ternary: a t2 chunk cannot hold it")

    ms = t[:, PB_CODE:].copy().view("<i2").reshape(tiles, 2, 8, NW // 2)
    # a 128-K block is four consecutive 32-K groups; one fp16 scale per block needs them equal
    ms4 = ms.reshape(tiles, 2, 2, 4, NW // 2)
    if not (ms4 == ms4[:, :, :, 0:1, :]).all():
        raise ValueError("pb16i magnitudes differ inside a 128-K block: not representable in t2")

    # each pb16i tile is one container (32-N block, 256-K block); recover both indices
    cw = N // 32                                   # columns per core
    ti = np.arange(tiles)
    r = ti % 4
    q = ti // 4
    nt = q % (cw // NW)
    q2 = q // (cw // NW)
    ki = q2 % (K // KC)
    c = q2 // (K // KC)
    j0 = c * (N // 8) + r * cw + nt * NW            # first of the tile's 32 columns
    if np.any(j0 % 32):
        raise ValueError("a pb16i tile is not aligned to a 32-column t2 block")
    # scales: [nb][blk][cc] -> fp16(ms * 2^e_ref[column])
    col = j0[:, None, None, None] + np.arange(2)[None, :, None, None] * 16 + np.arange(NW // 2)[None, None, None, :]
    d = ms4[:, :, :, 0, :].astype(np.float64) * np.exp2(eref[col])
    half = d.astype(np.float16)
    if not np.array_equal(half.astype(np.float64), d):
        bad = int((half.astype(np.float64) != d).sum())
        raise ValueError(f"{bad} of {d.size} scales are not fp16-exact: t2 cannot hold this image exactly")

    n_blocks = (N // 32) * (K // 256)
    t2 = np.zeros((n_blocks, TILE), dtype=np.uint8)
    tblock = (j0 // 32) * (K // 256) + ki
    quad = (codes.astype(np.uint8) + 1).reshape(tiles, KC // 4, 4, NW)
    packed = (quad[:, :, 0] | (quad[:, :, 1] << 2) | (quad[:, :, 2] << 4) | (quad[:, :, 3] << 6)).astype(np.uint8)
    t2[tblock, :CODE_BYTES] = packed.transpose(0, 2, 1).reshape(tiles, ROWS * 64)
    # scales: [nb][blk][cc] -> tile order [row = nb*16 + cc][blk]
    sc = half.transpose(0, 1, 3, 2).reshape(tiles, ROWS * 2)
    sc_bits = np.ascontiguousarray(sc, dtype=np.float16).view("<u2")   # the BITS: astype would round to 0
    t2[tblock, CODE_BYTES:] = sc_bits.view(np.uint8).reshape(tiles, SCALE_BYTES)
    return t2.reshape(-1).tobytes()


# ── GGUF -> t2 ──────────────────────────────────────────────────────────────────────────────────

_POW3 = (1, 3, 9, 27, 81)


def _trits_base3(b, n_trits):
    """The first n_trits base-3 digits of each byte, the way ggml reads them:
    ((byte * 3^n mod 256) * 3) >> 8, minus 1. Shape (n_trits,) + b.shape."""
    b = np.asarray(b, np.uint8)
    out = np.empty((n_trits,) + b.shape, np.int8)
    for n in range(n_trits):
        q = ((b.astype(np.uint16) * _POW3[n]) & 0xFF).astype(np.uint16)
        out[n] = ((q * 3) >> 8).astype(np.int8) - 1
    return out


def from_ptq1_0(rows, K, N):
    """PrismML PTQ1_0 (28 B / 128 values: qs[24], qh[2], fp16 d) -> t2. Lossless.

    The trit codec is q4nx_pb16i.read_ptq1_0's, not restated here: PTQ1_0 does not lay its 128
    trits out as one base-3 run, it packs them in staged chunks (32, then 16, then 8 wide), and a
    second copy of that rule is a second chance to get it wrong.
    """
    import q4nx_pb16i as PB
    codes, dbits = PB.read_ptq1_0(bytes(rows), K, N)          # codes [K][N] int8, d bits [K/128][N]
    d = dbits.T.copy().view(np.float16)                       # [N][K/128]
    return pack(codes.T.copy().astype(np.int8), d)


def from_tq1_0(rows, K, N):
    """ggml TQ1_0 (QK_K=256: qs[48], qh[4], fp16 d = 54 B) -> t2. Lossless (ternary, one d per 256 K)."""
    QK, SZ = 256, 54
    rows = np.asarray(rows, dtype=np.uint8)
    if rows.size != N * (K // QK) * SZ:
        raise ValueError(f"TQ1_0 payload is {rows.size} B, not {N}x{K // QK}x{SZ}")
    b = rows.reshape(N, K // QK, SZ)
    tr = np.empty((N, K // QK, QK), dtype=np.int8)
    # qs[0:32]: 5 trits x 32 bytes = values n*32 + m          (ggml walks n outer, m inner)
    t = _trits_base3(b[:, :, 0:32].reshape(-1, 32).T, 5)        # (5, 32, blocks)
    tr[:, :, 0:160] = t.transpose(2, 0, 1).reshape(N, K // QK, 160)
    # qs[32:48]: 5 trits x 16 bytes = values 160 + n*16 + m
    t = _trits_base3(b[:, :, 32:48].reshape(-1, 16).T, 5)
    tr[:, :, 160:240] = t.transpose(2, 0, 1).reshape(N, K // QK, 80)
    # qh: 4 trits x 4 bytes = 16 values, 240 + n*4 + j   (QK_K/64 = 4 bytes, which is why the
    # block is 54 B and not 58: 48 + 4 + 2)
    t = _trits_base3(b[:, :, 48:52].reshape(-1, 4).T, 4)
    tr[:, :, 240:256] = t.transpose(2, 0, 1).reshape(N, K // QK, 16)
    d = np.frombuffer(b[:, :, 52:54].tobytes(), dtype="<f2").reshape(N, K // QK)
    # two 128-K t2 blocks per 256-K TQ1_0 block share the one d
    return pack(tr.reshape(N, K), np.repeat(d.astype(np.float16), 2, axis=1))


def from_tq2_0(rows, K, N, allow_nonternary=False):
    """ggml TQ2_0 (QK_K=256: qs[64], fp16 d) -> t2.

    TQ2_0 is NOT ternary: its code 3 means +2. Converting it is refused unless allow_nonternary,
    in which case every +2 is clamped to +1 and the caller is told how many weights changed.
    """
    QK, SZ = 256, 66
    rows = np.asarray(rows, dtype=np.uint8)
    if rows.size != N * (K // QK) * SZ:
        raise ValueError(f"TQ2_0 payload is {rows.size} B, not {N}x{K // QK}x{SZ}")
    b = rows.reshape(N, K // QK, SZ)
    tr = np.empty((N, K // QK, QK), dtype=np.int8)
    for g in range(2):                                    # two 32-byte groups per block
        for l in range(4):                                # four 2-bit slots per byte
            seg = ((b[:, :, g * 32:(g + 1) * 32] >> (2 * l)) & 3).astype(np.int8) - 1
            tr[:, :, g * QK // 2 + l * 32:g * QK // 2 + l * 32 + 32] = seg
    n_plus2 = int((tr == 2).sum())
    if n_plus2 and not allow_nonternary:
        raise ValueError(f"TQ2_0 has {n_plus2} +2 codes ({(tr == 2).mean() * 100:.2f} % of the tensor): "
                         f"not ternary, so not t2; pass allow_nonternary=True to clamp them to +1")
    tr = np.clip(tr, -1, 1)
    d = np.frombuffer(b[:, :, 64:66].tobytes(), dtype="<f2").reshape(N, K // QK)
    return pack(tr.reshape(N, K), np.repeat(d.astype(np.float16), 2, axis=1)), n_plus2


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    stem = sys.argv[1]
    out = sys.argv[2]
    img = open(stem + ".pb16i", "rb").read()
    ref = open(stem + ".ref", "rb").read()
    N = (len(ref) - 1) // 2
    K = (len(img) // 4608) * 32 * 256 // N
    t2 = from_pb16i(img, ref, K, N)
    open(out, "wb").write(t2)
    print(f"{stem}: {N}x{K} pb16i {len(img)} B -> t2 {len(t2)} B "
          f"({len(t2) * 8 / (N * K):.4f} bits/weight)")
