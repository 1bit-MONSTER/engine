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
"""tools/q4nx_diff.py: reading a container whose weight is padded to a wider tile grid (engine#1097).

A converter may store a weight on a WIDER grid than the model's own dimensions -- the FastFlowLM repack
pads intermediate_size up to its 512-column grid, so an 8960-wide MLP arrives as 9216 and the container
holds 1728 tiles where the model implies 1680.  Before this, `q4nx_diff.py decode` reported
"cannot reshape array of size ... into shape ..." and counted the tensor as skipped, so the tool an
operator reaches for could not check the very tensors that were broken.

usage: tests/q4nx_diff_grid_test.py        (needs numpy; no model, no container, no device)

1. The grid rule: data_size pins the container's grid, it must dominate the model's, it may not add more
   than Q4NX_MAX_EXCESS tiles, and two different grids with the same least excess are a refusal (that
   tie is the transposed-decode case).
2. Reading at the container's grid is *exact*: a padded container decodes to the same values as the tight
   one on the model's slice, and the padded bytes cannot be read at the tight grid at all.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import q4nx_diff as D  # noqa: E402

FAIL = []


def check(ok, what):
    if not ok:
        FAIL.append(what)
        print("FAIL", what)


def tile_bytes(rng, ntiles):
    """`ntiles` well-formed 5120-B tiles: plausible bf16 scales/zero-points (random bytes would be
    inf/NaN as bf16, and NaN != NaN would fail every comparison for the wrong reason), random codes."""
    t = np.zeros((ntiles, D.TILE), dtype=np.uint8)
    scales = np.array([0x3F80, 0x3F00, 0x3E80, 0x3F40], dtype=np.uint16)   # 1.0, 0.5, 0.25, 0.75
    planes = (t[:, :D.PLANE_SCALE + D.PLANE_MIN].view(np.uint16)
              .reshape(ntiles, (D.PLANE_SCALE + D.PLANE_MIN) // 2))
    planes[:] = scales[rng.integers(0, len(scales), size=planes.shape)]
    t[:, D.PLANE_SCALE + D.PLANE_MIN:] = rng.integers(0, 256, size=(ntiles, D.PLANE_CODES), dtype=np.uint8)
    return t.reshape(-1)


def scatter(tight, ntr, ntc, ntr_p, ntc_p):
    """Place a tight [ntr*ntc, TILE] tile set into a wider [ntr_p, ntc_p] grid, zero-filling the pad."""
    out = np.zeros((ntr_p * ntc_p, D.TILE), dtype=np.uint8)
    t = np.asarray(tight).reshape(ntr, ntc, D.TILE)
    o = out.reshape(ntr_p, ntc_p, D.TILE)
    o[:ntr, :ntc] = t          # padding is appended, so the model's rows/cols are the leading ones
    return out.reshape(-1)


def test_grid_rule():
    TILE = D.TILE
    # The two shapes #1097 measured: IM padded on the columns axis (down_proj) and on the rows axis (gate/up).
    check(D.packed_grid(1728 * TILE, 1536, 8960) == (1536, 9216), "down_proj 1536x8960 in 1728 tiles -> 1536x9216")
    check(D.packed_grid(1728 * TILE, 8960, 1536) == (9216, 1536), "gate/up 8960x1536 in 1728 tiles -> 9216x1536")
    # A container that is already on the model's grid is not padded, and the caller must read it as before.
    check(D.packed_grid((1536 // 32) * (8960 // 256) * TILE, 1536, 8960) is None, "an on-grid container is None")
    # A tie is a refusal: 32x32 model = 1x1 tiles, 2 tiles fit both (1,2) and (2,1) with excess 1.
    check(D.packed_grid(2 * TILE, 32, 32) is None, "two grids with the same least excess is a refusal")
    # Smaller than the model: 2x2 tiles expected, one tile present.
    check(D.packed_grid(1 * TILE, 64, 64) is None, "a grid smaller than the model's is refused")
    # Not a whole number of tiles.
    check(D.packed_grid(1728 * TILE + 8, 1536, 8960) is None, "a size that is not whole tiles is refused")
    # More excess than the rule trusts: a 1-tile model in a 66-tile container.
    check(D.packed_grid(66 * TILE, 32, 256) is None, "an excess past Q4NX_MAX_EXCESS is refused")
    # Off the 32x256 grid altogether.
    check(D.packed_grid(1728 * TILE, 1530, 8960) is None, "an off-grid model geometry is refused")


def test_decode_matches_tight():
    rng = np.random.default_rng(1097)
    ntr, ntc = 4, 3                     # a 128x768 model weight
    rows, cols = ntr * D.ROWS, ntc * D.COLS
    tight = tile_bytes(rng, ntr * ntc)
    # Padded on the columns axis by one tile, exactly the down_proj shape of engine#1097.
    padded = scatter(tight, ntr, ntc, ntr, ntc + 1)
    check(D.packed_grid(int(padded.size), rows, cols) == (rows, cols + D.COLS),
          "the padded grid is recovered from data_size alone")
    want = D.decode_ours(tight, rows, cols)
    got = D.decode_ours(padded, rows, cols + D.COLS)[:rows, :cols]
    check(np.array_equal(got, want), "a columns-padded container decodes to the tight values on the model's slice")
    # Padded on the rows axis by two tiles: the gate/up shape.
    padded_r = scatter(tight, ntr, ntc, ntr + 2, ntc)
    got_r = D.decode_ours(padded_r, rows + 2 * D.ROWS, cols)[:rows, :cols]
    check(np.array_equal(got_r, want), "a rows-padded container decodes to the tight values on the model's slice")
    # And the failure this replaces: the padded bytes cannot be read on the tight grid.
    try:
        D.decode_ours(padded, rows, cols)
        check(False, "reading a padded container at the tight grid must fail, not silently reshape")
    except ValueError:
        pass


def test_padding_is_zero_filled():
    """The pad columns must not leak into the model's slice: they are zeros, and only [:rows, :cols] is read."""
    rng = np.random.default_rng(1107)
    ntr, ntc = 2, 2
    rows, cols = ntr * D.ROWS, ntc * D.COLS
    tight = tile_bytes(rng, ntr * ntc)
    padded = scatter(tight, ntr, ntc, ntr, ntc + 1)
    full = D.decode_ours(padded, rows, cols + D.COLS)
    check(float(np.abs(full[:, cols:]).max()) == 0.0, "the padded columns decode to zero, so the slice is the weight")


def main():
    test_grid_rule()
    test_decode_matches_tight()
    test_padding_is_zero_filled()
    if FAIL:
        for f in FAIL[:20]:
            print("FAIL", f)
        print(f"q4nx_diff_grid_test: {len(FAIL)} failure(s)")
        return 1
    print("q4nx_diff_grid_test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
