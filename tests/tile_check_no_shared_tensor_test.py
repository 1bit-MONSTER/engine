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
"""`repack_gguf.py --tile-check`: an end-to-end pair that shares no tensor (engine#1110 review).

The review asked for a committed fixture that goes through the real entry point instead of a
verdict dict built by hand.  Both files here are synthetic and tiny: a one-tensor GGUF, and a Q4NX
container whose only tensor is named for a DIFFERENT layer, so `q4nx_diff`'s name mapping finds
nothing to compare.  That pair is the case that used to read as a match: two layouts, each having
compared zero tensors, tie at worst=0.0 and the tie was broken alphabetically.

Then the same 5120 bytes under the name the GGUF DOES map to: the check must stop saying "nothing
was compared" (it refuses those tiles), which is the control proving the fixture exercises the
comparison and not just the empty path.

usage: tests/tile_check_no_shared_tensor_test.py    (needs numpy + gguf-py; no model, no device)
"""
import json
import os
import struct
import sys
import tempfile
import warnings

import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)   # the noise tile decodes to NaN on purpose

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "scripts"))
import repack_gguf as R  # noqa: E402

FAIL = []


def check(ok, what, why=""):
    if ok:
        print("ok  ", what)
        return
    FAIL.append(what)
    print("FAIL", what, ("  -- " + why) if why else "")


def tiny_gguf(path):
    """One F32 weight named for layer 0. gguf stores an array's axes reversed, so the (32, 256) array
    arrives as ne = (256, 32) -- cols 256, rows 32, i.e. a whole tile in q4nx_diff's terms."""
    from gguf import GGUFWriter
    w = GGUFWriter(path, "qwen3")
    w.add_tensor("blk.0.attn_q.weight", np.random.default_rng(7).standard_normal((32, 256)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def tiny_container(dir_path, name_bytes):
    """A container whose only tensor is `name_bytes`, one 5120-B Q4_1 tile of noise."""
    os.makedirs(dir_path, exist_ok=True)
    rng = np.random.default_rng(11)
    tile = rng.integers(0, 256, size=5120, dtype=np.uint8).tobytes()
    hdr = {name_bytes: {"dtype": "I8", "shape": [32, 256], "data_offsets": [0, len(tile)]}}
    raw = json.dumps(hdr).encode()
    with open(os.path.join(dir_path, "model.q4nx"), "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(tile)


with tempfile.TemporaryDirectory() as td:
    gguf = os.path.join(td, "m.gguf")
    tiny_gguf(gguf)

    # names do not intersect: blk.0.attn_q.weight maps to model.layers.0.self_attn.q_proj.weight
    disjoint = os.path.join(td, "disjoint")
    tiny_container(disjoint, "model.layers.5.self_attn.q_proj.weight")
    rc, line = R.tile_order_check(gguf, disjoint)
    check(rc == 2 and line == "nothing was compared",
          "a container that shares no tensor with the GGUF is 'nothing was compared' (rc=2)",
          f"got rc={rc} {line!r}")
    check(R.tile_check_exit(rc, line) == 2, "--tile-check exits 2 for it, not 0", f"got {R.tile_check_exit(rc, line)}")

    # control: the same bytes under the name the GGUF maps to DOES get compared, and is refused
    # (the tile is noise, so it cannot match the F32 weight) -- proving the fixture reaches the
    # comparison path rather than only the empty one
    mapped = os.path.join(td, "mapped")
    tiny_container(mapped, "model.layers.0.self_attn.q_proj.weight")
    with np.errstate(invalid="ignore"):     # the noise tile decodes to NaN; the check must refuse it
        rc2, line2 = R.tile_order_check(gguf, mapped)
    check(rc2 == 1 and "matches NO layout" in line2,
          "the same tile under the mapped name is compared and refused (rc=1)", f"got rc={rc2} {line2!r}")
    check(R.tile_check_exit(rc2, line2) == 1, "--tile-check exits 1 for it", f"got {R.tile_check_exit(rc2, line2)}")

print(f"tile_check_no_shared_tensor_test: {len(FAIL)} failure(s)")
sys.exit(1 if FAIL else 0)
