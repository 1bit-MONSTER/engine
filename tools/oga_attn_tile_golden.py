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
"""Regenerate tests/golden/oga_attn_tile{,_q4_1}.bin + .json from a Ryzen AI OGA NPU pack.

One 34,816-byte tile out of the pack's decode `dd_metastate_Llm_Token_Token_*.fconst` -- the bytes the
vendor writer produced -- together with the q4_1 container tiles the importer derives from it. The engine
test decodes the shipped tile, re-encodes it (which has to give the same bytes back) and compares its
container output with the recorded tiles, so the importer's tile layout is checked against shipped pack
bytes in CI, where no vendor stack exists.

Fixtures are exempt from the copyright check (tests/golden/*/*.bin), like the tiling golden beside them.

usage: tools/oga_attn_tile_golden.py <pack dir> [--check]
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

TILE_INDEX = 2          # layer 0, tile 2 = q_proj rows 0..63 (its bias subvolumes are not empty)
ATTN_OFF = 1856
SUBV, K, ROWS = 2176, 896, 64
COLS = 256              # q4_1 chunk width, the grid the container pads K to
FILES = ("oga_attn_tile.bin", "oga_attn_tile_q4_1.bin", "oga_attn_tile.json")


def read_tile(pack: Path):
    fc = pack / "dd_metastate_Llm_Token_Token_modellayers.0input_layernormMLADFRMSNORM.fconst"
    with fc.open("rb") as f:
        f.seek(ATTN_OFF + TILE_INDEX * (2 * (K // 128) + 2) * SUBV)
        return f.read((2 * (K // 128) + 2) * SUBV)


def decode(tile: bytes):
    """The layout the importer implements, written independently here so the fixture is not its own proof."""
    sv = np.frombuffer(tile, dtype=np.uint8).reshape(16, SUBV)
    kbs = K // 128
    code = np.zeros((ROWS, K), dtype=np.uint8)
    scale = np.zeros((ROWS, kbs), dtype=np.float32)
    zp = np.zeros((ROWS, kbs), dtype=np.uint8)
    bias = np.zeros(ROWS, dtype=np.uint16)
    kk = (np.arange(128) * 4)[:, None]
    for h in (0, 1):
        r0 = h * 32
        bias[r0:r0 + 32] = np.frombuffer(sv[h][:64].tobytes(), dtype=np.uint16)
        for kb in range(kbs):
            s = sv[2 + 2 * kb + h]
            b = s[kk + (np.arange(16) % 4)[None, :] + (np.arange(16) // 4)[None, :] * 512]
            c = np.empty((128, 32), dtype=np.uint8)
            c[:, 0::2] = b & 0x0F
            c[:, 1::2] = b >> 4
            code[r0:r0 + 32, kb * 128:(kb + 1) * 128] = c.T
            z = s[2048:2064]
            # the zero-point nibbles pair with rows even/odd, exactly as the codes do -- a correct zero
            # point sits where its 128 codes cluster, and only this pairing satisfies that on the shipped
            # tiles (engine#849). Pairing them in halves puts another row's zero point in half the rows.
            zp[r0:r0 + 32, kb] = np.stack([z & 0x0F, z >> 4], axis=-1).reshape(32)
            scale[r0:r0 + 32, kb] = (np.frombuffer(s[2112:2176].tobytes(), dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    return code, scale, zp, bias


def bf16_bits(x):
    """round-to-nearest-even f32 -> bf16, as the container's writer does"""
    u = int(np.frombuffer(np.float32(x).tobytes(), dtype=np.uint32)[0])
    return np.uint16(((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF)


def tiles_q4_1(code, scale, zp):
    """Q4_1 container chunks: [1,024 B header: 8x32 bf16 scale then 8x32 bf16 zero][4,096 B codes]."""
    kp = ((K + COLS - 1) // COLS) * COLS
    rt, kt = ROWS // 32, kp // COLS
    out = np.zeros((rt * kt, 5120), dtype=np.uint8)
    for r in range(rt):
        for c in range(kt):
            t = out[r * kt + c]
            sc16 = np.frombuffer(t[:1024].tobytes(), dtype=np.uint16).copy()
            for g in range(8):
                for rr in range(32):
                    row, group = r * 32 + rr, c * 8 + g     # group = 32-column group
                    if group * 32 >= K:
                        continue
                    kb = group // 4                          # its K block of 128
                    sc16[g * 32 + rr] = bf16_bits(scale[row, kb])                  # scale
                    sc16[256 + g * 32 + rr] = bf16_bits(-np.float32(zp[row, kb]) * scale[row, kb])  # zero
            t[:1024] = np.frombuffer(sc16.astype(np.uint16).tobytes(), dtype=np.uint8)
            for h in range(2):
                for g in range(8):
                    for j in range(32):
                        col = c * COLS + g * 32 + j
                        for r2 in range(8):
                            row = r * 32 + h * 16 + r2 * 2
                            if col >= K:
                                continue
                            t[1024 + h * 2048 + g * 256 + j * 8 + r2] = (
                                int(code[row, col]) | (int(code[row + 1, col]) << 4))
    return out.tobytes()


def main(argv):
    # --help is a request, not a path: without this it arrives as argv[1] and the tool
    # opens a file literally named "--help" (FileNotFoundError) instead of saying anything.
    if "-h" in argv or "--help" in argv:
        print(__doc__ or "usage: see the header of this file")
        return 0
    positional = [a for a in argv[1:] if not a.startswith("-")]
    if not positional:
        # The usage line is `<pack dir> [--check]`, so a bare `--check` has no pack dir. Before this
        # it became argv[1] and the tool opened a directory literally named "--check" and crashed
        # with FileNotFoundError -- a documented invocation shape reported as a traceback.
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    pack, check = Path(positional[0]), "--check" in argv
    tile = read_tile(pack)
    code, scale, zp, bias = decode(tile)
    q4 = tiles_q4_1(code, scale, zp)
    meta = {
        "source": {"pack": pack.name, "fconst": "dd_metastate_Llm_Token_Token_*", "offset": ATTN_OFF + TILE_INDEX * len(tile),
                   "tile_index": TILE_INDEX, "projection": "layers.0.self_attn.q_proj rows 0..63"},
        "layout": "amd-oss/npu-re docs/FACTS-oga-pack-weight-layout.md §7.13/§7.15",
        "tile_bytes": len(tile), "k": K, "rows": ROWS,
        "bias_bf16": [int(v) for v in bias],
        "note": "q4_1 tiles are the container's; the tile itself is the vendor writer's bytes, unmodified",
    }
    out = Path(__file__).resolve().parent.parent / "tests/golden"
    if check:
        ok = ((out / FILES[0]).read_bytes() == tile and (out / FILES[1]).read_bytes() == q4
              and json.loads((out / FILES[2]).read_text())["bias_bf16"] == meta["bias_bf16"])
        print("golden matches the pack" if ok else "golden DIFFERS from the pack")
        return 0 if ok else 1
    (out / FILES[0]).write_bytes(tile)
    (out / FILES[1]).write_bytes(q4)
    (out / FILES[2]).write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {', '.join(FILES)} to {out} ({len(tile)} + {len(q4)} bytes, {ROWS}x{K})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
