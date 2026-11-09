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
"""`tools/q4nx_tail_head.py`'s geometry and its envelope decoder, on data the tree already ships.

The tool exists so that engine#1693's measurement can be re-run (it was a scratch script). Two halves of it are
checkable without a pack, and both are checked here:

  * the **geometry** — env, tile bytes, tile count and the int4 ratio — against numbers derived by hand from
    `npu/pack_import.cpp`'s constants, so a drift in the tool's copy of them fails instead of silently measuring
    different bytes;
  * the **envelope decoder** — against `tests/golden/oga_attn_tile.{bin,json}`, a vendor-written tile of exactly the
    head's geometry (K 896, rows 64, 34,816 B). Its bias subvolumes decode to the golden's 64 raw bf16 codes, and its
    code address map is a bijection over the 2,048 code bytes — the two properties that can be pinned without a pack.

What is deliberately NOT asserted: any correlation number. On the pack engine#1693 names, the decode correlates far
from 1 although its geometry is exact, which is the issue's open question — pinning today's value would pin a bug.

Usage: tests/q4nx_tail_head_test.py   (no arguments)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import q4nx_tail_head as T  # noqa: E402

GOLDEN = os.path.join(HERE, "golden", "oga_attn_tile")

failures = 0


def check(ok, what):
    global failures
    if not ok:
        failures += 1
        print("FAIL %s" % what)
    else:
        print("ok   %s" % what)


def main():
    # ── geometry, by hand from npu/pack_import.cpp: kNormOff 64, kTileRows 64, kKBlock 128 ──
    check(T.subv_bytes(64) == 2176, "subv_bytes(64) is 2176 B (64*32 + 64 zeros + 64 scales)")
    check(T.attn_tile_bytes(896) == 34816, "attn_tile_bytes(896) is 34816 B = (2*7 + 2) subvolumes")
    check(T.attn_payload_off(896) == 1856, "attn_payload_off(896) is 1856 B = 64 + 2*896")
    check(T.attn_tile_bytes(896) % 128 == 0 and T.attn_tile_bytes(896) % 64 == 0,
          "a tile is a whole number of 128-byte rows and 64-byte halves")
    # the identity that makes engine#1693's premise checkable: the tail's surplus over a bare int4 matrix is exactly
    # the per-tile envelope, so a region whose whole-region ratio differs from its per-tile ratio is not this grid
    V, H = 151936, 896
    tiles = V // T.K_TILE_ROWS
    whole = (tiles * T.attn_tile_bytes(H)) / (V * H / 2.0)
    per_tile = T.attn_tile_bytes(H) / (T.K_TILE_ROWS * H / 2.0)
    check(abs(whole - per_tile) < 1e-12, "whole-tail int4 ratio %.6f == per-tile ratio %.6f" % (whole, per_tile))

    # ── the head's offset rule: the last grid, not the region after env (engine#1693's 0.885 was an env-based read)
    check(T.head_offset(291655296, 2374, 34816) == 209002112, "head_offset is fconst_bytes - ntiles*tile")
    check(T.head_offset(291655296, 2374, 34816) + 2374 * 34816 == 291655296, "the head grid ends exactly at EOF")
    check(T.head_offset(291655296, 2374, 34816) != T.attn_payload_off(896),
          "the head's offset is not attn_payload_off(H) -- the mistake engine#1693 records")

    # ── the envelope decoder, against the vendor tile ──
    with open(GOLDEN + ".json") as f:
        meta = json.load(f)
    blob = open(GOLDEN + ".bin", "rb").read()
    check(len(blob) == T.attn_tile_bytes(meta["k"]) == meta["tile_bytes"],
          "the golden tile is exactly attn_tile_bytes(%d) = %d B" % (meta["k"], T.attn_tile_bytes(meta["k"])))
    _, bias = T.decode_tile(blob, meta["k"], meta["rows"])
    # the golden's bias_bf16 list is the raw 16-bit codes (48245, 15569, ...), not the decoded floats
    want = [int(round(x)) for x in meta["bias_bf16"]]
    check(bias.tolist() == want, "all %d bias codes match the golden's bias_bf16 list" % len(want))

    rows, k, kbs = meta["rows"], meta["k"], meta["k"] // T.K_KBLOCK
    subv = T.subv_bytes(rows)
    seen = {}
    for h in range(2):
        for kb in range(kbs):
            s = (2 + 2 * kb + h) * subv
            for cp in range((rows // 2) // 2):
                base = (cp % 4) + (cp // 4) * 512
                for kk in range(T.K_KBLOCK):
                    seen[s + base + kk * 4] = seen.get(s + base + kk * 4, 0) + 1
    code_bytes = 64 * (rows // 2) * (2 * kbs)
    check(set(seen.values()) == {1}, "every code address is written exactly once (no collisions)")
    check(len(seen) == code_bytes, "the code map covers all %d code bytes (%d)" % (code_bytes, len(seen)))

    print("all q4nx_tail_head checks passed" if not failures else "%d check(s) failed" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
