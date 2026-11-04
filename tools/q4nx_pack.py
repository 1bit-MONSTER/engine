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
"""The Q4NX q4_1 tile packers, shared by the converters (needs numpy only).

`tools/gguf_to_q4nx.py` and `tools/safetensors_to_q4nx.py` write the same container and must
therefore pack identically; these functions live here so they cannot drift, and so a converter
that has no GGUF input does not need `gguf-py` installed to import them.

Layout (npu/q4nx.h): a 2-D projection becomes q4_1 tiles of 32 rows x 256 columns, each tile
5120 bytes -- bf16 scale and zero per row and 32-column group, then 4-bit codes. A tensor a
caller names on the command line can instead be stored as Q8 tiles (8704 bytes: the same bf16
scale per row and 32-column group, then int8 codes at `column*32 + row`, the layout npu/q4nx.cpp
and npu/forward/src/q4nx_dequant.c decode), which is what keeps a tensor that was never 4-bit
(a Q6_K/Q8_0 source) at 8 bits instead of crushing it to 16 levels.
"""
import numpy as np

ROWS, COLS, GROUP, TILE = 32, 256, 32, 5120
Q8 = 8704  # Chunk::Q8: 256 bf16 scales (512 B) + 32 x 256 int8 codes


def bf16_bits(x):
    """f32 -> bf16 bits (nearest even), as uint16."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return rounded


def from_bf16(bits):
    """bf16 bits (uint16) -> f32."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f16(raw):
    """f16 bits -> f32.  ascontiguousarray is not decoration: a non-contiguous view would be
    reinterpreted as float16 at the wrong stride and read as garbage."""
    return np.ascontiguousarray(raw).view(np.float16).astype(np.float32)


def quantize_q4_1(w):
    """f32 [N, K] -> q4_1 per 32-column group (min/max), scale and zero already bf16."""
    N, K = w.shape
    g = w.reshape(N, K // GROUP, GROUP)
    lo, hi = g.min(-1), g.max(-1)
    zero = from_bf16(bf16_bits(lo))
    scale = from_bf16(bf16_bits((hi - lo) / 15.0))
    safe = np.where(scale > 0, scale, 1.0)
    codes = np.clip(np.rint((g - zero[..., None]) / safe[..., None]), 0, 15).astype(np.uint8)
    return codes.reshape(N, K), scale, zero


def tiles(codes, scale, zero):
    """q4_1 parts -> [tiles, 5120] bytes, tiles row-major over the 32 x 256 grid."""
    N, K = codes.shape
    RT, KT = N // ROWS, K // COLS
    # scale/zero index g*32 + r within a tile
    def plane(v):
        return bf16_bits(v).reshape(RT, ROWS, KT, 8).transpose(0, 2, 3, 1).reshape(RT * KT, 256)
    # codes: region h*8+g (h = r/16), byte 8*j + (r%16)/2, even row in the low nibble
    c = codes.astype(np.uint8).reshape(RT, 2, 8, 2, KT, 8, 32)  # rt, h, r2, parity, kt, g, j
    packed = c[:, :, :, 0] | (c[:, :, :, 1] << 4)  # rt, h, r2, kt, g, j
    packed = packed.transpose(0, 3, 1, 4, 5, 2).reshape(RT * KT, 4096)  # rt, kt, h, g, j, r2
    return np.concatenate([plane(scale).view(np.uint8), plane(zero).view(np.uint8), packed], axis=1)


def quantize_q8(w):
    """f32 [N, K] -> symmetric int8 per 32-column group: codes int8, scale bf16."""
    N, K = w.shape
    g = w.reshape(N, K // GROUP, GROUP)
    scale = from_bf16(bf16_bits(np.abs(g).max(-1) / 127.0))
    safe = np.where(scale > 0, scale, 1.0)
    codes = np.clip(np.rint(g / safe[..., None]), -127, 127).astype(np.int8)
    return codes.reshape(N, K), scale


def tiles_q8(codes, scale):
    """Q8 parts -> [tiles, 8704]: bf16 scales (index g*32 + r), then int8 codes at c*32 + r."""
    N, K = codes.shape
    RT, KT = N // ROWS, K // COLS
    sc = bf16_bits(scale).reshape(RT, ROWS, KT, 8)
    sc = sc.transpose(0, 2, 3, 1).reshape(RT * KT, 256).view(np.uint8)
    c = codes.reshape(RT, ROWS, KT, COLS).transpose(0, 2, 3, 1).reshape(RT * KT, COLS * ROWS)
    return np.concatenate([sc, c.view(np.uint8)], axis=1)


def untile_q8(raw, N, K):
    """[tiles, 8704] Q8 -> f32 [N, K] (the decoder of npu/q4nx.cpp)."""
    RT, KT = N // ROWS, K // COLS
    t = raw.reshape(RT, KT, Q8)
    sc = from_bf16(t[..., :512].copy().view(np.uint16)).reshape(RT, KT, 8, ROWS)
    sc = sc.transpose(0, 3, 1, 2)  # rt, r, kt, g
    c = t[..., 512:].copy().view(np.int8).reshape(RT, KT, COLS, ROWS).transpose(0, 3, 1, 2)
    c = c.reshape(RT, ROWS, KT, COLS // GROUP, GROUP)  # column c = group*32 + inner
    return (c.astype(np.float32) * sc[..., None]).reshape(N, K)


def untile(raw, N, K):
    """[tiles, 5120] q4_1 -> f32 [N, K] (the decoder of npu/q4nx.cpp)."""
    RT, KT = N // ROWS, K // COLS
    t = raw.reshape(RT, KT, TILE)
    sc = from_bf16(t[..., :512].copy().view(np.uint16)).reshape(RT, KT, 8, 32)
    zr = from_bf16(t[..., 512:1024].copy().view(np.uint16)).reshape(RT, KT, 8, 32)
    p = t[..., 1024:].reshape(RT, KT, 2, 8, 32, 8)  # rt, kt, h, g, j, r2
    c = np.stack([p & 0xF, p >> 4], axis=-1)  # ..., r2, parity
    c = c.transpose(0, 2, 5, 6, 1, 3, 4).reshape(RT, 32, KT, 8, 32)  # rt, r, kt, g, j
    s = sc.transpose(0, 3, 1, 2)[..., None]  # rt, r, kt, g, 1
    z = zr.transpose(0, 3, 1, 2)[..., None]
    return (c * s + z).reshape(N, K)
