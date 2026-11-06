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
"""`check_q4nx_dir.py --against`: a container whose q4_1 TILE ORDER is foreign is refused (#1110).

The failure this pins has no metadata signature: the two converter checkouts write the same chunk
kind, the same byte count and the same tensor names, and only the element order inside each
5120-byte tile differs.  So the fixture cannot be built by shuffling a real container's bytes -- it
has to come from a known weight, encoded both ways and read back with the decoders the checker uses.

Everything here is synthetic and tiny:

  * one F32 weight (32 x 256) in a one-tensor GGUF;
  * the same weight encoded as a single q4_1 tile in the engine's order, and again in the foreign
    (`flm`, unsigned q) order;
  * the checker must accept the first, and refuse the second naming the tensor -- and the second's
    failure must be the ORDER, so the same bytes under a foreign name are also checked.

The encoder below is the inverse of the decoders in `tools/q4nx_diff.py` (self-checked here by
round-tripping every tile through them), which is the only way a test can make a tile that is
"correct except for the order" without a converter.

usage: tests/check_q4nx_tile_order_test.py    (needs numpy + gguf-py; no model, no device)
"""
import json
import os
import struct
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "tools"))
import check_q4nx_dir as C       # noqa: E402
import q4nx_diff as D            # noqa: E402

FAIL = []
ROWS, COLS, GROUP, TILE = 32, 256, 32, 5120
NAME = "blk.0.attn_q.weight"                      # gguf name -> model.layers.0.self_attn.q_proj.weight


def check(ok, what, why=""):
    if ok:
        print("ok  ", what)
        return
    FAIL.append(what)
    print("FAIL", what, ("  -- " + why) if why else "")


def quantize(w):
    """32x256 floats -> (scale, min, q) per row and 32-wide group, the q4_1 affine of both orders."""
    s = np.zeros((ROWS, COLS // GROUP), np.float32)
    m = np.zeros((ROWS, COLS // GROUP), np.float32)
    q = np.zeros((ROWS, COLS), np.uint8)
    for r in range(ROWS):
        for g in range(COLS // GROUP):
            blk = w[r, g * GROUP:(g + 1) * GROUP]
            lo, hi = float(blk.min()), float(blk.max())
            sc = (hi - lo) / 15.0 if hi > lo else 1e-8
            s[r, g], m[r, g] = sc, lo
            q[r, g * GROUP:(g + 1) * GROUP] = np.clip(np.rint((blk - lo) / sc), 0, 15).astype(np.uint8)
    return s, m, q


def bf16_plane(x):
    """32x8 floats ([row][group]) -> 512 bf16 bytes, group-major [g*32+r] -- the order both the
    engine's reader and q4nx_diff's decoders use (the two tile orders differ in the CODE plane only).
    """
    return (np.asarray(x, np.float32).T.copy().reshape(-1).view(np.uint32) >> 16
            ).astype(np.uint16).view(np.uint8)


def encode_tile(w, order):
    """One 5120-byte q4_1 tile holding `w`, written in `order` ('ours' or 'flm')."""
    s, m, q = quantize(w)
    tile = np.zeros(TILE, np.uint8)
    tile[0:512] = bf16_plane(s)                    # [r*8+g]
    tile[512:1024] = bf16_plane(m)
    codes = tile[1024:].reshape(COLS * ROWS // 2)
    if order == "ours":
        # npu/q4nx.h: lane = r//16, position lane*2048 + col*8 + (r%16)//2, nibble (r%16)%2
        for r in range(ROWS):
            for c in range(COLS):
                codes[(r // 16) * 2048 + c * 8 + (r % 16) // 2] |= q[r, c] << (4 if (r % 16) % 2 else 0)
    elif order == "flm":
        # the foreign order: adjacent row pair per column, even row in the low nibble, unsigned q
        for c in range(COLS):
            for rp in range(ROWS // 2):
                codes[c * 16 + rp] = q[2 * rp, c] | (q[2 * rp + 1, c] << 4)
    else:
        raise ValueError(order)
    return tile


def tiny_gguf(path, w):
    """Two layer-0 weights holding the same values: q_proj and k_proj, so a tile can be renamed
    between two tensors that DO have ground truth (the order must not depend on the name)."""
    from gguf import GGUFWriter
    wr = GGUFWriter(path, "qwen3")
    wr.add_tensor(NAME, w.astype(np.float32))
    wr.add_tensor("blk.0.attn_k.weight", w.astype(np.float32))
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file()
    wr.close()


def container(dir_path, tile, tensor="model.layers.0.self_attn.q_proj.weight"):
    """A one-tensor container holding `tile`, named `tensor` (the order must not depend on the name)."""
    os.makedirs(dir_path, exist_ok=True)
    hdr = {tensor: {"dtype": "I8", "shape": [1, TILE], "data_offsets": [0, len(tile)]}}
    raw = json.dumps(hdr).encode()
    with open(os.path.join(dir_path, "model.q4nx"), "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(tile)


# --- the fixture's own soundness: each encoder must round-trip through its own decoder -------------
rng = np.random.default_rng(3)
W = rng.standard_normal((ROWS, COLS)).astype(np.float32) * 0.02
for order, decode in (("ours", lambda t: D.decode_ours(t, ROWS, COLS)),
                      ("flm", lambda t: D.decode_flm(t, ROWS, COLS, signed=False))):
    tile = encode_tile(W, order)
    back = decode(np.frombuffer(tile, dtype=np.uint8))
    err = float(np.abs(back - W).max())
    check(err < 2e-2, f"the {order} encoder round-trips through its own decoder (max|err| {err:.2e})",
          f"max|err| {err:.3e} -- the test's own fixture is wrong, fix it before trusting the rows below")
    if order == "flm":          # and the engine's decoder must NOT reproduce it: that is the bug
        wrong = float(np.abs(D.decode_ours(np.frombuffer(tile, np.uint8), ROWS, COLS) - W).max())
        check(wrong > 10 * err, f"read as 'ours' the same bytes are wrong (max|err| {wrong:.2e})",
              f"only {wrong:.3e}")

with tempfile.TemporaryDirectory() as td:
    gguf = os.path.join(td, "m.gguf")
    tiny_gguf(gguf, W)

    good = os.path.join(td, "ours")
    container(good, encode_tile(W, "ours"))
    errors, rows = C.check_q4_1_layout(os.path.join(good, "model.q4nx"), gguf)
    check(not errors, "a container in the engine's order passes --against",
          f"{len(errors)} error(s): {errors[:1]}")
    check(len(rows) == 1 and rows[0]["ours"][0] >= 0.99,
          "and the row says so with a correlation, not a verdict word",
          f"rows={rows}")

    foreign = os.path.join(td, "flm")
    container(foreign, encode_tile(W, "flm"))
    errors, rows = C.check_q4_1_layout(os.path.join(foreign, "model.q4nx"), gguf)
    check(len(errors) == 1 and "NOT in the order" in errors[0],
          "a container in the foreign order is REFUSED (engine#1110)", f"errors={errors}")
    check(len(errors) == 1 and "model.layers.0.self_attn.q_proj.weight" in errors[0],
          "and the refusal names the tensor", f"errors={errors}")
    check(len(rows) == 1 and rows[0]["flm-unsigned"][0] > 0.99 and rows[0]["ours"][0] < 0.5,
          "with the foreign reading identified, so the message can say what the bytes look like",
          f"rows={rows}")

    # the same refusal has to survive a rename: the order is a property of the bytes, not the name.
    # k_proj is a real layer-0 tensor, so it is compared against its own GGUF weight -- which is W.
    renamed = os.path.join(td, "flm-renamed")
    container(renamed, encode_tile(W, "flm"), "model.layers.0.self_attn.k_proj.weight")
    errors, _ = C.check_q4_1_layout(os.path.join(renamed, "model.q4nx"), gguf)
    check(len(errors) == 1 and "NOT in the order" in errors[0],
          "a foreign order under the OTHER tensor's name is still refused", f"errors={errors}")

    # a container that shares no tensor with the GGUF cannot be silently accepted
    odd = os.path.join(td, "odd")
    container(odd, encode_tile(W, "ours"), "model.layers.9.self_attn.q_proj.weight")
    errors, rows = C.check_q4_1_layout(os.path.join(odd, "model.q4nx"), gguf)
    check(len(errors) == 1 and "nothing was verified" in errors[0],
          "a container with no tensor in common is 'nothing was verified', not a pass",
          f"errors={errors} rows={rows}")

print(f"check_q4nx_tile_order_test: {len(FAIL)} failure(s)")
sys.exit(1 if FAIL else 0)
