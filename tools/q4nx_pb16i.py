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
"""q4nx_pb16i.py -- build the "pb16i" weight images a pb16i container reads.

    tools/q4nx_pb16i.py pack   IN.gguf OUTDIR [--layers 0,1,2] [--check]
    tools/q4nx_pb16i.py verify IN.gguf OUTDIR

A ternary model (PrismML PTQ1_0, ggml type 143) stores every weight as code * d, code in
{-1,0,+1} with one fp16 d per 128 K rows per output column.  A pb16i image streams the
trits packed two per byte plus one int16 per (32 K rows, column) that carries d rescaled
into that column's window:

    w[k][j] = code[k][j] * ms[k/32][j] * 2^e_ref[j]

The device sums the integer product exactly (int32, or two int32 planes when the
converter's worst-case |S| bound reaches 2^31) and the host applies the only rounding in
the chain, C[i][j] = float32(S[i][j] * 2^e_ref[j]).  The image format, the 4,608-byte tile
and the side table are the contract in npu/forward/include/q4nx_pb16i.h; the engine loads
`<stem>.pb16i` + `<stem>.ref` from the directory a container names in `q4nx_pb16i_dir`
(env Q4NX_PB16I_DIR wins).

Tensor names follow the GGUF: `blk.<l>.<name>.weight` becomes `blk_<l>_<name>_weight`.
A `*.attn_q.weight` is emitted as its two halves (`_h0`, `_h1`): the fused full-attention
QKV has no launch that fits L1 at 14336 columns, so the engine runs q as two 6144-wide
launches plus k and v.  Every image is verified against the GGUF's own scales unless
--no-check is given.
"""
import json
import os
import struct
import sys

import numpy as np

# PTQ1_0 (ggml type 143, PrismML): 28 bytes per 128 weights -- qs[24] @0 (5 trits/byte,
# 120 values), qh[2] @24 (4 trits/byte, 8 values), fp16 d @26.  Reference:
# llama-fmt/ggml/src/ggml-prism.h block_ptq1_0 + dequantize_row_ptq1_0.
PTQ1_0 = 143
BLOCK_BYTES, QK, POW3 = 28, 128, (1, 3, 9, 27, 81, 243)
QS_N, QH_N, D_OFF = 24, 2, 26      # qs[24] @0, qh[2] @24, fp16 d @26

# The image geometry (q4nx_pb16i.h): 32 cores = 8 columns x 4 rows, 256-K x 32-N tiles.
COLS, ROWS, KC, NW = 8, 4, 256, 32
CODE_BYTES, TILE = KC * NW // 2, 4608


def fits(K, N):
    return K > 0 and N > 0 and K % KC == 0 and N % (COLS * ROWS * NW) == 0


def w_bytes(K, N):
    return (K // KC) * (N // NW) * TILE


def index(i, j, K, N):
    """Byte of the element's int4 code (and whether it is the high nibble) and of its ms."""
    cw = N // (COLS * ROWS)
    c = j // (N // COLS)
    jr = j % (N // COLS)
    r = jr // cw
    jn = jr % cw
    nt, nb, cc = jn // NW, (jn % NW) // 16, jn % 16
    kb, rr = (i % KC) // 16, (i % KC) % 16
    tile = (((c * (K // KC) + i // KC) * (cw // NW) + nt) * ROWS + r)
    e = ((kb * 2 + nb) * 16 + rr) * 16 + cc
    return (tile * TILE + e // 2, bool(e & 1),
            tile * TILE + CODE_BYTES + 2 * ((nb * 8 + ((i % KC) // 32)) * 16 + cc))


def decompose(dbits):
    """fp16 d bits [blocks][N] -> (m int64 signed, e int32), per the PrismML spec."""
    h = np.asarray(dbits, dtype=np.uint16)
    sign = np.where((h >> 15) & 1, -1, 1).astype(np.int64)
    exp = ((h >> 10) & 0x1F).astype(np.int32)
    man = (h & 0x3FF).astype(np.int64)
    m = np.where(exp == 0, man, 1024 + man).astype(np.int64) * sign
    e = np.where(exp == 0, -24, exp - 25).astype(np.int32)
    return m, e


def side_table(e_ref, mode):
    return e_ref.astype('<i2').tobytes() + bytes([mode])


def pack(codes, dbits, K, N):
    """(codes int8 [K][N], d fp16 bits [K/128][N]) -> (image uint8 [tiles][4608], e_ref, mode)."""
    if not fits(K, N):
        raise ValueError(f"K={K} N={N} is not a pb16i geometry (K%256, N%1024)")
    cw, nt_n, kt = N // (COLS * ROWS), N // (COLS * ROWS * NW), K // KC
    # d is one value per 128-K block; the image carries one per 32-K group, so expand
    m, e = decompose(np.repeat(np.asarray(dbits), 4, axis=0))     # [K/32][N] each
    blocks = K // 128
    e_col = e.reshape(blocks, 4, N)
    e_ref = e_col.min(axis=(0, 1)).astype(np.int16)
    sh = (e_col - e_ref[None, None, :]).astype(np.int64)
    if sh.max() > 20:
        raise ValueError("the column's exponent spread is too wide for the int16 image")
    ms = m.reshape(blocks, 4, N)[:, :1, :] * (np.int64(1) << sh[:, :1, :])
    if np.abs(ms).max() > 32767:
        raise ValueError(f"|m*2^(e-e_ref)| = {int(np.abs(ms).max())} does not fit int16")
    # The converter's own worst case: |S| <= sum_b 4*32*127*|ms|
    bound = (4 * 32 * 127) * np.abs(ms).astype(np.int64).sum(axis=0)
    mode = 0 if bound.max() < 2 ** 31 else 1
    out = np.zeros((COLS, kt, nt_n, ROWS, TILE), dtype=np.uint8)
    msb = ms.reshape(blocks, N)
    for c in range(COLS):
        for ki in range(kt):
            for nt in range(nt_n):
                for r in range(ROWS):
                    col0 = c * (N // COLS) + r * cw + nt * NW
                    blk = codes[ki * KC:(ki + 1) * KC, col0:col0 + NW]
                    cb = blk.reshape(16, 16, 2, 16).transpose(0, 2, 1, 3)
                    bits = (cb.reshape(-1) & 0x0F).astype(np.uint8)
                    out[c, ki, nt, r, :CODE_BYTES] = bits[0::2] | (bits[1::2] << 4)
                    a = msb[2 * ki, col0:col0 + NW].astype('<i2').view(np.uint8).reshape(2, 16, 2)
                    b = msb[2 * ki + 1, col0:col0 + NW].astype('<i2').view(np.uint8).reshape(2, 16, 2)
                    m8 = np.zeros((2, 8, 16, 2), dtype=np.uint8)
                    m8[:, 0:4, :, :] = a[:, None, :, :]
                    m8[:, 4:8, :, :] = b[:, None, :, :]
                    out[c, ki, nt, r, CODE_BYTES:] = m8.reshape(-1)
    return out.reshape(-1, TILE), e_ref, mode


def read_ptq1_0(raw, K, N):
    """The tensor's bytes -> (codes int8 [K][N], d bits uint16 [K/128][N]).

    The block is [N][K/128][28] -- one row per output column, blocked along K -- and the
    128 trits of a block are NOT a plain base-3 number: ggml_ptq1_0_trits packs them in
    staged chunks (32, then 16, then 8 wide, five trits per byte), which is why this walks
    the stages instead of dividing the byte by 3**i."""
    nb = K // QK
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(N, nb, BLOCK_BYTES)
    codes = np.empty((N, nb, QK), dtype=np.int8)
    bits = np.empty((nb, N), dtype=np.uint16)
    for bi in range(nb):
        b = blk[:, bi, :]
        qs, qh = b[:, :QS_N], b[:, QS_N:QS_N + QH_N]
        bits[bi] = b[:, D_OFF:D_OFF + 2].copy().view('<u2')[:, 0]
        vals = np.empty((N, QK), dtype=np.int8)
        pos = j = 0
        for c in (32, 16, 8):
            while j + c <= QS_N:
                for n in range(5):
                    vals[:, pos:pos + c] = (((qs[:, j:j + c].astype(np.uint16) * POW3[n])
                                             & 0xFF) * 3 >> 8).astype(np.int16) - 1
                    pos += c
                j += c
        for n in range(4):
            for m in range(QH_N):
                vals[:, pos] = (((qh[:, m].astype(np.uint16) * POW3[n]) & 0xFF) * 3
                                >> 8).astype(np.int16) - 1
                pos += 1
        assert pos == QK, (pos, QK)
        codes[:, bi, :] = vals
    return codes.reshape(N, K).T.copy(), bits


def gguf_tensors(path):
    """Yield (name, dims, type, raw bytes) for every tensor of a GGUF file."""
    f = open(path, 'rb')
    magic, ver, n_tensors, n_kv = struct.unpack('<IIQQ', f.read(24))
    if magic != 0x46554747:
        raise SystemExit(f"{path}: not a GGUF file")
    fmt = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: 'B', 8: 'B', 9: 'B',
           10: 'Q', 11: 'q', 12: 'd'}

    def rstr():
        (n,) = struct.unpack('<Q', f.read(8))
        return f.read(n).decode('utf-8', 'replace')

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            (at,) = struct.unpack('<I', f.read(4))
            (n,) = struct.unpack('<Q', f.read(8))
            return [rval(at) for _ in range(n)]
        size = struct.calcsize('<' + fmt[t])
        return struct.unpack('<' + fmt[t], f.read(size))[0]

    for _ in range(n_kv):
        key = rstr()
        (t,) = struct.unpack('<I', f.read(4))
        rval(t)
    meta = []
    for _ in range(n_tensors):
        name = rstr()
        (nd,) = struct.unpack('<I', f.read(4))
        dims = struct.unpack('<' + 'Q' * nd, f.read(8 * nd))
        (tt,) = struct.unpack('<I', f.read(4))
        (off,) = struct.unpack('<Q', f.read(8))
        meta.append((name, tuple(int(x) for x in dims), tt, off))
    # the tensor data section starts at the next multiple of the file's alignment
    # (ggml's general.alignment, 32 unless a file says otherwise), not at the byte after
    # the last tensor info -- reading from there shifts every tensor by a few bytes
    align = 32
    base = (f.tell() + align - 1) // align * align
    out = []
    for name, dims, tt, off in meta:
        n = int(np.prod(dims))
        nb = n // QK * BLOCK_BYTES if tt == PTQ1_0 else 0
        f.seek(base + off)
        out.append((name, dims, tt, f.read(nb) if nb else b''))
    f.close()
    return out


def jobs(name, dims, tt):
    """(stem, K, N, column slice) the tensor produces."""
    if tt != PTQ1_0 or len(dims) != 2:
        return []
    K, N = dims
    base = name[:-len('.weight')].replace('.', '_') + '_weight' if name.endswith('.weight') \
        else name.replace('.', '_')
    if name.endswith('.attn_q.weight') and N % 2 == 0:
        # the fused full-attention q is two launches in the engine; split it here
        return [(base.replace('_attn_q_weight', '_attn_q_h0_weight'), K, N // 2, slice(0, N // 2)),
                (base.replace('_attn_q_weight', '_attn_q_h1_weight'), K, N // 2, slice(N // 2, N))]
    return [(base, K, N, slice(0, N))]


def verify_image(img, codes, dbits, K, N, e_ref):
    """Every ms slot == m*2^(e-e_ref) for its (block, column), and every code == the trit."""
    m, e = decompose(np.repeat(np.asarray(dbits), 4, axis=0))    # [K/32][N] each
    msb = img.reshape(-1, TILE)[:, CODE_BYTES:].view('<i2').reshape(-1, 2, 8, 16)
    cw, nt_n, kt = N // (COLS * ROWS), N // (COLS * ROWS * NW), K // KC
    bad = 0
    for c in range(COLS):
        for ki in range(kt):
            for nt in (0, nt_n - 1):
                for r in (0, ROWS - 1):
                    tile = ((c * kt + ki) * nt_n + nt) * ROWS + r
                    col0 = c * (N // COLS) + r * cw + nt * NW
                    for nb in range(2):
                        for g in range(8):
                            cols = col0 + nb * 16 + np.arange(16)
                            row = ki * 8 + g
                            want = m[row][cols] << (e[row][cols] - e_ref[cols])
                            if not np.array_equal(msb[tile, nb, g].astype(np.int64), want):
                                bad += 1
                    blk = codes[ki * KC:(ki + 1) * KC, col0:col0 + NW]
                    want_bits = blk.reshape(16, 16, 2, 16).transpose(0, 2, 1, 3).reshape(-1) & 0x0F
                    got = img.reshape(-1, TILE)[tile, :CODE_BYTES]
                    back = np.empty(want_bits.size, dtype=np.int8)
                    back[0::2] = got & 0x0F
                    back[1::2] = got >> 4
                    if not np.array_equal(back, want_bits):
                        bad += 1
    return bad


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    cmd, src, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    only = None
    for i, a in enumerate(sys.argv[4:]):
        if a == '--layers':
            only = {int(x) for x in sys.argv[4 + i + 1].split(',')}
    os.makedirs(outdir, exist_ok=True)
    manifest = {}
    if os.path.exists(os.path.join(outdir, 'manifest.json')):
        manifest = json.load(open(os.path.join(outdir, 'manifest.json')))
    n_done = 0
    for name, dims, tt, raw in gguf_tensors(src):
        for stem, K, N, cols in jobs(name, dims, tt):
            if only is not None:
                layer = int(name.split('.')[1]) if name.startswith('blk.') else -1
                if layer not in only:
                    continue
            if not fits(K, N):
                print(f"  skip {stem}: K={K} N={N} is not a pb16i geometry")
                continue
            codes, d = read_ptq1_0(raw, int(dims[0]), int(dims[1]))
            codes, d = codes[:, cols], d[:, cols]
            if cmd == 'pack':
                img, e_ref, mode = pack(codes, d, K, N)
                img.tofile(os.path.join(outdir, stem + '.pb16i'))
                open(os.path.join(outdir, stem + '.ref'), 'wb').write(side_table(e_ref, mode))
                bad = verify_image(img, codes, d, K, N, e_ref)
                if bad:
                    raise SystemExit(f"{stem}: {bad} mismatches against the GGUF's own scales")
            else:
                img = np.fromfile(os.path.join(outdir, stem + '.pb16i'), dtype=np.uint8)
                ref = open(os.path.join(outdir, stem + '.ref'), 'rb').read()
                e_ref = np.frombuffer(ref[:2 * N], dtype='<i2')
                bad = verify_image(img, codes, d, K, N, e_ref)
            manifest[stem] = {'K': K, 'N': N, 'tiles': int(w_bytes(K, N) // TILE)}
            n_done += 1
            print(f"  {stem} K={K} N={N} {'ok' if not bad else f'{bad} MISMATCH'}", flush=True)
    json.dump(manifest, open(os.path.join(outdir, 'manifest.json'), 'w'), indent=1)
    print(f"{cmd}: {n_done} image(s) in {outdir}")


if __name__ == '__main__':
    main()
