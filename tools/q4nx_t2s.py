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
"""q4nx_t2s.py -- the 2-bit ternary weight stream ("t2s") the XDNA decode kernel reads.

A ternary model is 2.125 bits per weight; the pb16i image streams 4.5, because it carries int4 nibbles
and one int16 multiplier per 32-K group.  t2s is the same weights and the same integer contract at
2.125 bits/weight, with the codes expanded back to int4 nibbles **inside the AIE core** between the DMA
and the `mmul`:

    2,176 B per k=256 x n=32 tile   (pb16i: 4,608 B)       27B decode: 6.11 GB/token instead of 12.93
    [0, 2048)    2-bit codes, 4 per byte, n-major blocks: block (nb, kb) at (nb*16 + kb)*64; 0 -> -1,
                 1 -> 0, 2 -> +1, 3 reserved.  Two byte orders exist inside a block:
                   v1  byte r*4 + j holds the 4 codes for k = kb*16 + r, n = nb*16 + 4j + {0,1,2,3}
                   v2  byte b holds, at bits 2*lane:
                         lane0 -> (k = kb*16 + b/8,     n = nb*16 + 2*(b%8))
                         lane1 -> (k = kb*16 + b/8,     n = nb*16 + 2*(b%8) + 1)
                         lane2 -> (k = kb*16 + 8 + b/8, n = nb*16 + 2*(b%8))
                         lane3 -> (k = kb*16 + 8 + b/8, n = nb*16 + 2*(b%8) + 1)
    [2048, 2176) int16 multipliers, [nb 2][gb 2][16 c]: one per 128-K block per column, i.e. pb16i's own
                 multiplier at the granularity the model actually stores (see "why this is exact")

Why v2 is the one to ship.  The mmul's operand byte for element `e = 16k + c` is `e/2`, so a 64-byte vector of
codes has to land on the operand's even bytes (`out[2i]`) and the next 64 on its odd bytes (`out[2i+1]`): a
spread, not a copy.  With v1's order the kernel has to interleave two vectors (`aie::interleave_zip`), which is
a lane-crossing shuffle and was measured at 11.1-12.6 GB/s against 30-32 GB/s for the plain two-store path on
this box.  v2's order makes the plain path produce exactly `out[2i]`/`out[2i+1]`: the operand byte `b` needs
element `2b`, and `2b = 16*(b/8) + 2*(b%8)` is precisely the (k, n) v2 puts there; `64+b` needs `128 + 2b`,
i.e. k + 8 rows down, which is what lanes 2 and 3 carry.  So v2 + the plain expansion is byte-exact (measured on
the array 2026-11-02: 3/3 seeds, 0 mismatches, `amdxdna` counter 0) with no shuffle in the core at all.

Why this is exact rather than a requantisation. The engine's pb16i image is built from a model whose scale
is one fp16 `d` per 128 K per column: `q4nx_pb16i.pack` derives one `ms` per 128-K block and then repeats it
four times to reach the 32-K granularity of the image.  This module therefore drops only duplicated bytes --
and it *checks* that they are duplicates (`from_pb16i_image` refuses with a named error otherwise), so a
container built from a scale of a different granularity cannot silently lose a multiplier.

Why the block order is n-major. The kernel's expansion has to produce the mmul's B operand, whose element
`e = r*16 + c` of a block lives in byte `e/2` with even `e` in the low nibble.  With four *consecutive-n*
codes in one wire byte, the expansion is a byte-local 1->2 map (`expand()` below), no lane shuffling and no
table:

    out[2i] = f(c0) | f(c1) << 4,   out[2i+1] = f(c2) | f(c3) << 4,   f(v) = v - 1 mod 16 = {15,0,1}

and the same order makes one `nb` (all 16 k-blocks, 1,024 B) a single contiguous run for the core's DMA.

The gate is byte-exact, not statistical: `expand()` must reproduce the pb16i image's int4 code bytes block
for block and the multipliers must be identical, which `tests/q4nx_t2s_test.py` checks on every tile of a
frozen fixture.  Nothing here runs on the device; the kernel's own copy of the expansion is
`npu-kernels`, private.

The stream is also derivable from the container alone: `from_t2()` takes a `q4nx_t2` chunk and returns
the same bytes `from_pb16i_image()` returns for the image that chunk was made from, so a model shipped
in the t2 form needs no pb16i sidecar to feed the kernel.  The two forms are the same weights in
different orders (t2 is k-major per 32-row band, this is n-major per 16x16 block) and are never
interchangeable, which is why the crossing is one function and not a byte copy.
"""
import numpy as np

import q4nx_t2

from q4nx_pb16i import CODE_BYTES as PB_CODE_BYTES, KC, NW, ROWS as PB_ROWS, TILE as PB_TILE, pack, w_bytes as pb_w_bytes

ROWS, COLS = 16, 16                       # one mmul B block: k=16 x n=16
KB, NB = KC // ROWS, NW // COLS           # 16 k-blocks x 2 n-blocks per tile
BLOCK = ROWS * COLS // 4                  # 64 B of 2-bit codes per block
CODE_BYTES = KC * NW // 4                 # 2048
MS_BYTES = (KC // 128) * NW * 2           # 128: [nb 2][gb 2][16 c] int16
TILE = CODE_BYTES + MS_BYTES              # 2176
RESERVED = 3


def fits(K, N):
    return K % KC == 0 and N % NW == 0


def w_bytes(K, N):
    return (K // KC) * (N // NW) * TILE


def bits_per_weight():
    return 8.0 * TILE / (KC * NW)


def n_tiles(K, N):
    return (K // KC) * (N // NW)


class NotRepresentable(ValueError):
    """The image's multipliers are not shared inside a 128-K block, so t2s cannot hold them."""


def _split(img):
    """pb16i image [tiles][4608] -> (codes [tiles][kb][nb][16 r][16 c] int8, ms [tiles][nb][8 g][16 c] int16)."""
    img = np.asarray(img, dtype=np.uint8)
    if img.ndim == 1:
        img = img.reshape(-1, PB_TILE)
    if img.shape[1] != PB_TILE:
        raise ValueError(f"a pb16i image tile is {PB_TILE} B, got {img.shape[1]}")
    nib = img[:, :PB_CODE_BYTES]
    lo = (nib & 0x0F).astype(np.int8)
    hi = (nib >> 4).astype(np.int8)
    c = np.empty((img.shape[0], PB_CODE_BYTES * 2), dtype=np.int8)
    c[:, 0::2] = np.where(lo >= 8, lo - 16, lo)
    c[:, 1::2] = np.where(hi >= 8, hi - 16, hi)
    codes = c.reshape(-1, KB, NB, ROWS, COLS)                      # e = ((kb*2 + nb)*16 + r)*16 + cc
    ms = np.ascontiguousarray(img[:, PB_CODE_BYTES:]).view("<i2").reshape(-1, NB, KC // 32, COLS)
    return codes, ms


def from_pb16i_image(img, K, N):
    """A pb16i image (the container's integer-exact weights) -> the t2s stream bytes."""
    if not fits(K, N):
        raise ValueError(f"K={K} N={N} is not a phase-B geometry (K%256, N%32)")
    codes, ms = _split(img)
    want = n_tiles(K, N)
    if codes.shape[0] != want:
        raise ValueError(f"image has {codes.shape[0]} tiles, {K}x{N} needs {want}")
    if np.abs(codes).max() > 1:
        raise ValueError("the image carries a code outside {-1,0,+1}: not a ternary weight")
    # 4 x 32-K groups share one 128-K block: their multipliers must be the same value, or the
    # compaction would drop a real multiplier.  Refuse rather than round.
    ms4 = ms.reshape(-1, NB, 2, 4, 16)
    if not np.array_equal(ms4, np.repeat(ms4[:, :, :, :1, :], 4, axis=3)):
        bad = int(np.count_nonzero(ms4 != np.repeat(ms4[:, :, :, :1, :], 4, axis=3)))
        raise NotRepresentable(
            f"{bad} of the image's 32-K multipliers are not shared across their 128-K block, so the "
            f"t2s stream (which carries one multiplier per 128 K) cannot hold this tensor exactly")
    v = (codes + 1).astype(np.uint8)                               # 0,1,2
    q = v.reshape(-1, KB, NB, ROWS, COLS // 4, 4)                  # four consecutive n per wire byte
    packed = (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)).astype(np.uint8)
    area = packed.transpose(0, 2, 1, 3, 4).reshape(-1, CODE_BYTES)  # block (nb, kb) at (nb*16 + kb)*64
    ms_area = np.ascontiguousarray(ms4[:, :, :, 0, :]).astype("<i2").reshape(-1, MS_BYTES // 2)
    ms_area = ms_area.view(np.uint8).reshape(-1, MS_BYTES)
    return np.concatenate([area, ms_area], axis=1)


def from_t2(chunk, K, N):
    """A container's t2 chunk -> the t2s stream bytes, without the pb16i image.

    Re-derives the image with the packer's own code (so the tile order and the int16 multipliers are
    the packer's, not a second implementation of them) and then streams it; the result is byte-
    identical to `from_pb16i_image()` on the image the chunk came from.  pb16i geometry applies on
    top of t2's, i.e. N must be a multiple of 1,024 as well as 32.
    """
    codes, bits = q4nx_t2.unpack_bits(chunk, K, N)          # [N][K] and [N][K//128], fp16 bits
    img, _eref, _mode = pack(codes.T.copy(), bits.T.copy(), K, N)
    return from_pb16i_image(img.reshape(-1), K, N)


def from_pb16i_image_v2(img, K, N):
    """The same weights in wire format v2: the plain two-store expansion is exact on it (see the module docstring).

    Only the 64 bytes inside each (nb, kb) block change, so this reuses `from_pb16i_image` for every check and for
    the ms area; the block address and the tile order are identical, which is what keeps the DMA descriptor the same.
    """
    v1 = from_pb16i_image(img, K, N)                     # geometry, ternary and multiplier checks all live there
    codes, _ms = _split(np.asarray(img, dtype=np.uint8))
    v = (codes + 1).astype(np.uint8)                     # [tile][kb][nb][16 r][16 c] -> 0,1,2
    b = np.arange(ROWS * 4)                              # the 64 wire bytes of one block
    k_lo, n0 = b // 8, 2 * (b % 8)
    wire = ((v[:, :, :, k_lo, n0] << 0) | (v[:, :, :, k_lo, n0 + 1] << 2) |
            (v[:, :, :, k_lo + 8, n0] << 4) | (v[:, :, :, k_lo + 8, n0 + 1] << 6)).astype(np.uint8)
    out = np.array(v1, copy=True)
    out[:, :CODE_BYTES] = wire.transpose(0, 2, 1, 3).reshape(-1, CODE_BYTES)
    return out


def stream(codes, dbits, K, N):
    """(codes int8 [K][N] in {-1,0,1}, d fp16 bits [K/128][N]) -> (stream bytes, e_ref int16[N], mode)."""
    img, e_ref, mode = pack(codes, dbits, K, N)                     # the pb16i image is the reference
    return from_pb16i_image(img, K, N), e_ref, mode


def expand(stream_bytes, K, N):
    """The kernel's expansion, in numpy: t2s code area -> pb16i's int4 code bytes, [tiles][4096].

    Same byte order as the pb16i image's code area *after* transposing the 128-byte blocks from t2s's
    n-major order to pb16i's k-major order; `tests/q4nx_t2s_test.py` is what pins that down.
    """
    t = _tiles(stream_bytes, K, N)
    q = np.empty((t.shape[0], CODE_BYTES, 4), dtype=np.uint8)
    for c in range(4):
        q[:, :, c] = (t[:, :CODE_BYTES] >> (2 * c)) & 3
    if (q == RESERVED).any():
        raise ValueError(f"{int((q == RESERVED).sum())} reserved codes in the stream; the converter never "
                         f"writes one, so this stream is corrupt")
    nib = (q - 1) & 0x0F
    out = np.empty((t.shape[0], CODE_BYTES * 2), dtype=np.uint8)
    out[:, 0::2] = nib[:, :, 0] | (nib[:, :, 1] << 4)
    out[:, 1::2] = nib[:, :, 2] | (nib[:, :, 3] << 4)
    return out.reshape(-1, NB, KB, 2 * BLOCK).transpose(0, 2, 1, 3).reshape(-1, PB_CODE_BYTES)


def decode_v2(stream_bytes, K, N):
    """The inverse of `from_pb16i_image_v2`: -> (codes [tiles][kb][nb][16 r][16 c] int8, ms [tiles][nb][2][16 c])."""
    t = _tiles(stream_bytes, K, N)
    blocks = np.ascontiguousarray(t[:, :CODE_BYTES]).reshape(-1, NB, KB, BLOCK)   # [tile][nb][kb][64]
    q = np.empty(blocks.shape + (4,), dtype=np.uint8)
    for lane in range(4):
        q[..., lane] = (blocks >> (2 * lane)) & 3
    if (q == RESERVED).any():
        raise ValueError("reserved code in the v2 stream")
    v = (q.astype(np.int8) - 1).transpose(0, 2, 1, 3, 4)   # [tile][kb][nb][64][lane], the layout `codes` uses
    b = np.arange(BLOCK)
    k_lo, n0 = b // 8, 2 * (b % 8)
    codes = np.empty((blocks.shape[0], KB, NB, ROWS, COLS), dtype=np.int8)
    codes[:, :, :, k_lo, n0] = v[..., 0]
    codes[:, :, :, k_lo, n0 + 1] = v[..., 1]
    codes[:, :, :, k_lo + 8, n0] = v[..., 2]
    codes[:, :, :, k_lo + 8, n0 + 1] = v[..., 3]
    ms = np.ascontiguousarray(t[:, CODE_BYTES:]).view("<i2").reshape(-1, NB, 2, COLS)
    return codes, ms


def decode(stream_bytes, K, N):
    """-> (codes [tiles][kb][nb][16 r][16 c] int8, ms [tiles][nb][2 gb][16 c] int16)."""
    t = _tiles(stream_bytes, K, N)
    q = np.empty((t.shape[0], CODE_BYTES, 4), dtype=np.uint8)
    for c in range(4):
        q[:, :, c] = (t[:, :CODE_BYTES] >> (2 * c)) & 3
    if (q == RESERVED).any():
        raise ValueError("reserved code in the stream")
    v = (q.astype(np.int8) - 1).reshape(-1, NB, KB, ROWS, 4, 4)     # (nb, kb, r, j, c): n = 4j + c
    codes = v.transpose(0, 2, 1, 3, 4, 5).reshape(-1, KB, NB, ROWS, COLS)
    ms = np.ascontiguousarray(t[:, CODE_BYTES:]).view("<i2").reshape(-1, NB, 2, COLS)
    return codes, ms


def _tiles(stream_bytes, K, N):
    b = np.frombuffer(stream_bytes, dtype=np.uint8) if not isinstance(stream_bytes, np.ndarray) else stream_bytes
    if b.size != w_bytes(K, N):
        raise ValueError(f"a {K}x{N} t2s stream is {w_bytes(K, N)} B, got {b.size}")
    return b.reshape(-1, TILE)


def size_report(K, N):
    """(t2s bytes, pb16i bytes, ratio) for a tensor of this geometry -- what the stream saves."""
    return w_bytes(K, N), pb_w_bytes(K, N), w_bytes(K, N) / pb_w_bytes(K, N)
