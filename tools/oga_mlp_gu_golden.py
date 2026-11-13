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
"""Regenerate tests/golden/oga_mlp_gu_{group,q4_1}.bin + .json from a Ryzen AI OGA NPU pack.

One 68,608-byte gate|up unit out of the pack's decode `dd_metastate_Llm_Token_Token_*.fconst` -- the bytes
the vendor writer produced -- plus the q4_1 container tiles the importer derives from the two projections
it holds. The engine test decodes the shipped unit, re-encodes it (which has to give the same bytes back),
splits it into gate and up and compares both against the recorded container tiles, so the MLP layout is
checked against shipped pack bytes in CI, where no vendor stack exists.

The unit is 128 columns of 4,288-byte subvolumes; the fused tensor interleaves gate and up in 32-column
chunks, so the split is gate = rows [0:32] + [64:96], up = rows [32:64] + [96:128] of the decoded unit.

The tiling (q4_1 chunks) is the same code the attention fixture uses: a 64-row projection of this pack.

usage: tools/oga_mlp_gu_golden.py <pack dir> [--check]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oga_attn_tile_golden import tiles_q4_1  # noqa: E402  (the same 64-row tiling)

MLP_OFF = 1_117_824     # layer 0: [eps][input norm][attention payload][eps][post norm] then the MLP section
GROUP_INDEX = 0         # layer 0, gate|up unit 0 = gate and up rows 0..63
K, SUBV, SUBV_ROWS = 896, 4288, 64   # 64 columns per subvolume at the fused width
UNIT_ROWS = 128
FILES = ("oga_mlp_gu_group.bin", "oga_mlp_gu_q4_1.bin", "oga_mlp_gu.json")


def read_group(pack: Path):
    fc = pack / "dd_metastate_Llm_Token_Token_modellayers.0input_layernormMLADFRMSNORM.fconst"
    n = (2 * (K // 128) + 2) * SUBV
    with fc.open("rb") as f:
        f.seek(MLP_OFF + GROUP_INDEX * n)
        return f.read(n)


def decode(tile: bytes):
    """The layout the importer implements, written independently here so the fixture is not its own proof."""
    sv = np.frombuffer(tile, dtype=np.uint8).reshape(-1, SUBV)
    kbs = K // 128
    code = np.zeros((UNIT_ROWS, K), dtype=np.uint8)
    scale = np.zeros((UNIT_ROWS, kbs), dtype=np.float32)
    zp = np.zeros((UNIT_ROWS, kbs), dtype=np.uint8)
    bias = np.zeros(UNIT_ROWS, dtype=np.uint16)
    kk = (np.arange(128) * 4)[:, None]
    pairs = np.arange(SUBV_ROWS // 2)
    for h in (0, 1):
        r0 = h * SUBV_ROWS
        bias[r0:r0 + SUBV_ROWS] = np.frombuffer(sv[h][:2 * SUBV_ROWS].tobytes(), dtype=np.uint16)
        for kb in range(kbs):
            s = sv[2 + 2 * kb + h]
            b = s[kk + (pairs % 4)[None, :] + (pairs // 4)[None, :] * 512]
            c = np.empty((128, SUBV_ROWS), dtype=np.uint8)
            c[:, 0::2] = b & 0x0F
            c[:, 1::2] = b >> 4
            code[r0:r0 + SUBV_ROWS, kb * 128:(kb + 1) * 128] = c.T
            z = s[64 * SUBV_ROWS:64 * SUBV_ROWS + SUBV_ROWS // 2]
            # the zero-point nibbles pair with rows even/odd, exactly as the codes do above -- a correct zero
            # point sits where its 128 codes cluster (engine#849; halves scored -0.02, even/odd +0.998)
            zp[r0:r0 + SUBV_ROWS, kb] = np.stack([z & 0x0F, z >> 4], axis=-1).reshape(SUBV_ROWS)
            so = 64 * SUBV_ROWS + 64
            scale[r0:r0 + SUBV_ROWS, kb] = (
                np.frombuffer(s[so:so + 2 * SUBV_ROWS].tobytes(), dtype=np.uint16).astype(np.uint32) << 16
            ).view(np.float32)
    return code, scale, zp, bias


def half(code, scale, zp, up: bool):
    """One of the two projections out of the fused unit, as 64 rows."""
    rows = np.concatenate([np.arange(0, 32), np.arange(64, 96)]) + (32 if up else 0)
    return code[rows], scale[rows], zp[rows]


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
    group = read_group(pack)
    code, scale, zp, bias = decode(group)
    gate = tiles_q4_1(*half(code, scale, zp, False))
    up = tiles_q4_1(*half(code, scale, zp, True))
    q4 = gate + up
    gate_tiles = len(gate) // 5120
    meta = {
        "source": {"pack": pack.name, "fconst": "dd_metastate_Llm_Token_Token_*",
                   "offset": MLP_OFF + GROUP_INDEX * len(group), "group_index": GROUP_INDEX,
                   "projections": "layers.0.mlp.gate_proj rows 0..63, then mlp.up_proj rows 0..63"},
        "layout": "amd-oss/npu-re docs/FACTS-oga-pack-weight-layout.md §7.17",
        "group_bytes": len(group), "k": K, "rows": UNIT_ROWS,
        "subvolume_bytes": SUBV, "subvolume_columns": SUBV_ROWS,
        "split": {"gate": "rows 0..31 and 64..95", "up": "rows 32..63 and 96..127"},
        "bias_bytes_nonzero": int(sum(1 for v in bias if v)),
        "note": "q4_1 tiles are the container's, gate first then up; the unit itself is the vendor writer's "
                "bytes, unmodified",
    }
    out = Path(__file__).resolve().parent.parent / "tests/golden"
    if check:
        ok = ((out / FILES[0]).read_bytes() == group and (out / FILES[1]).read_bytes() == q4
              and json.loads((out / FILES[2]).read_text())["split"] == meta["split"])
        print("golden matches the pack" if ok else "golden DIFFERS from the pack")
        return 0 if ok else 1
    (out / FILES[0]).write_bytes(group)
    (out / FILES[1]).write_bytes(q4)
    (out / FILES[2]).write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {', '.join(FILES)} to {out} ({len(group)} + {len(q4)} bytes, "
          f"{UNIT_ROWS}x{K} unit, {2 * gate_tiles} x 5120 B of container tiles)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
