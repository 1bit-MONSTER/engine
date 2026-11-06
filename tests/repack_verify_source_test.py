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
"""scripts/repack_gguf.py --verify-source: the container is checked against the GGUF it came from.

engine#1109: a converter checkout of the wrong VERSION packs the same values permuted inside the 5120-B
chunk, so the container decodes as noise at HTTP 200 with nothing in the log -- and neither the
converter's path nor its commit can tell.  Only the source GGUF can, which is what the repack's
tile-order measurement already does against the source (not just against layouts, #1169): decode the
container under each layout and compare every tensor with the GGUF's own dequantized weight.  This test
pins the parts that are the operator's contract rather than the comparison itself:

1. `verify_source` / `--verify-source`: a container that holds the source's values matches the engine-1
   order (rc 0) and the verdict names the source tensor and the correlation; a container holding other
   values matches nothing (rc 1) and reports a correlation near zero; the same values in the FLM order
   match their source but are refused by the repack policy (exit 1, "not this engine's order").
2. `--skip-verify` stamps the directory `tile_order=unverified` (never engine-1), and `--check-stamp`
   answers "cannot tell" for it -- serve then warns instead of passing silently.
3. A repack that verifies stamps what it measured, and a container the repack refused is STALE on the
   next cache hit instead of `stamp: current` (the refused tiles stay on disk, so before engine#1109 a
   later serve could reuse the container the guard had just rejected).

The fixtures are synthetic and tiny (one-tensor GGUF, one-tile containers, a fake converter that copies
a prepared container), so this runs in CI with numpy + gguf-py: no model, no device, no real converter.

usage: tests/repack_verify_source_test.py
"""
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPACK = os.path.join(ROOT, "scripts", "repack_gguf.py")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import repack_gguf as R  # noqa: E402
import q4nx_diff as D  # noqa: E402

FAIL = []

GGUF_TENSOR = "blk.0.attn_q.weight"          # q4nx_diff maps this to the container name below
CONTAINER_NAME = "model.layers.0.self_attn.q_proj.weight"


def check(ok, what, why=""):
    if ok:
        print("ok  ", what)
        return
    FAIL.append(what)
    print("FAIL", what, ("  -- " + why) if why else "")


def tiny_gguf(path, w):
    """One F32 weight. gguf stores an array's axes reversed, so (32, 256) arrives as ne=(256, 32)."""
    from gguf import GGUFWriter
    writer = GGUFWriter(path, "qwen3")
    writer.add_tensor(GGUF_TENSOR, np.asarray(w, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def pack_ours(w):
    """One 5120-B q4_1 tile in the engine-1 order holding w exactly: scale 1.0, zero 0, codes 0..15.

    The inverse of q4nx_diff.decode_ours for a single 32x256 tile, written out here so the fixture does
    not depend on a packer under test.
    """
    code = np.asarray(w).astype(np.uint8)
    rows, cols = code.shape
    assert (rows, cols) == (D.ROWS, D.COLS), (rows, cols)
    plane = np.zeros((2, 8, D.GROUP, 8), dtype=np.uint8)          # [row-half, group, col-in-group, pair]
    for r in range(rows):
        half, rem = divmod(r, 16)
        pair, parity = divmod(rem, 2)
        for c in range(cols):
            group, in_group = divmod(c, D.GROUP)
            if parity:
                plane[half, group, in_group, pair] |= code[r, c] << 4
            else:
                plane[half, group, in_group, pair] |= code[r, c]
    scale = np.full((8, rows), 0x3F80, dtype=np.uint16)            # bf16 1.0
    zero = np.zeros((8, rows), dtype=np.uint16)
    return scale.tobytes() + zero.tobytes() + plane.tobytes()


def pack_flm(w):
    """One 5120-B q4_1 tile in the FLM order holding w exactly, for values in -8..7 (signed nibbles)."""
    code = np.asarray(w).astype(np.uint8)
    rows, cols = code.shape
    assert (rows, cols) == (D.ROWS, D.COLS), (rows, cols)
    plane = np.zeros((cols, rows // 2), dtype=np.uint8)            # [col, row-pair]
    for r in range(rows):
        pair, parity = divmod(r, 2)
        for c in range(cols):
            if parity:
                plane[c, pair] |= (int(code[r, c]) & 0x0F) << 4
            else:
                plane[c, pair] |= int(code[r, c]) & 0x0F
    scale = np.full((8, rows), 0x3F80, dtype=np.uint16)
    zero = np.zeros((8, rows), dtype=np.uint16)
    return scale.tobytes() + zero.tobytes() + plane.tobytes()


def write_container(dir_path, tile, name=CONTAINER_NAME):
    os.makedirs(dir_path, exist_ok=True)
    header = {name: {"dtype": "I8", "shape": [D.ROWS, D.COLS], "data_offsets": [0, len(tile)]}}
    raw = json.dumps(header).encode()
    with open(os.path.join(dir_path, "model.q4nx"), "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(tile)


def fake_converter(root, payload_q4nx):
    """A checkout shape the repack accepts: convert.py copies a prepared container and its tokenizer."""
    os.makedirs(root, exist_ok=True)
    shutil.copy(payload_q4nx, os.path.join(root, "payload.q4nx"))
    with open(os.path.join(root, "convert.py"), "w", encoding="utf-8") as f:
        f.write("import os, shutil, sys\n"
                "out = sys.argv[2]\n"
                "here = os.path.dirname(os.path.abspath(__file__))\n"
                "shutil.copy(os.path.join(here, 'payload.q4nx'), os.path.join(out, 'model.q4nx'))\n"
                "open(os.path.join(out, 'tokenizer.json'), 'w').write('{}')\n")
    return root


def run(args, env=None):
    e = dict(os.environ)
    e.pop("ONEBIT_Q4NX_CONVERTER", None)
    if env:
        e.update(env)
    return subprocess.run([sys.executable, REPACK] + args, capture_output=True, text=True, env=e)


def stamp_of(d):
    with open(os.path.join(d, "repack-stamp.txt"), encoding="utf-8") as f:
        return f.read()


def main():
    rng = np.random.default_rng(1109)
    # Uniform 0..15: exactly representable in an ours q4_1 tile at scale 1.0 / zero 0.
    w = rng.integers(0, 16, size=(D.ROWS, D.COLS)).astype(np.float32)
    w_bad = rng.integers(0, 16, size=(D.ROWS, D.COLS)).astype(np.float32)
    w_flm = rng.integers(0, 8, size=(D.ROWS, D.COLS)).astype(np.float32)

    with tempfile.TemporaryDirectory(prefix="repack-verify-") as td:
        gguf = os.path.join(td, "tiny.gguf")
        tiny_gguf(gguf, w)
        good = os.path.join(td, "good")
        write_container(good, pack_ours(w))
        bad = os.path.join(td, "bad")
        write_container(bad, pack_ours(w_bad))

        # 1. The comparison: a container of the source's values matches the engine-1 order, and the
        # verdict names the source tensor and the correlation the issue was measured with.
        rc, line = R.verify_source(gguf, good)
        check(rc == 0 and line.startswith("matches layout ours"), "a container of the source's values matches layout ours", line)
        check("corr +1.0000" in line and GGUF_TENSOR in line,
              "the verdict names the source tensor and its correlation", line)
        check(R.tile_check_exit(rc, line) == 0, "--verify-source exits 0 for it", str(R.tile_check_exit(rc, line)))
        check(R.tile_order_check(gguf, good) == (rc, line), "the old --tile-check name is the same function")

        # A container of OTHER values (what a wrong converter checkout produces) matches nothing, and
        # reports a correlation near zero instead of the honest +1: the measured failure is corr +0.0325.
        rc_bad, line_bad = R.verify_source(gguf, bad)
        m = re.search(r"corr ([+-][0-9.]+)", line_bad)
        corr = float(m.group(1)) if m else None
        check(rc_bad == 1 and "matches NO layout" in line_bad, "a container of other values matches NO layout", line_bad)
        check(corr is not None and abs(corr) < 0.5, "and reports a correlation near zero, not a bare error", line_bad)
        check(GGUF_TENSOR in line_bad, "and names the source tensor it failed on", line_bad)
        check(R.tile_check_exit(rc_bad, line_bad) == 1, "--verify-source exits 1 for it", str(R.tile_check_exit(rc_bad, line_bad)))

        # The same values in the FLM order DO match their source -- the repack policy still refuses
        # them, because this engine decodes the engine-1 order.
        gguf_flm = os.path.join(td, "tiny-flm.gguf")
        tiny_gguf(gguf_flm, w_flm)
        other = os.path.join(td, "other")
        write_container(other, pack_flm(w_flm))
        rc_o, line_o = R.verify_source(gguf_flm, other)
        check(rc_o == 0 and line_o.startswith("matches layout flm"), "an FLM-order container matches its source", line_o)
        check(R.tile_check_exit(rc_o, line_o) == 1, "but the repack policy refuses an order this engine does not decode")

        # 2. The command an operator runs, on an existing directory.
        r = run(["--verify-source", good, gguf])
        check(r.returncode == 0 and "matches its source" in r.stdout and "corr +1.0000" in r.stdout,
              "--verify-source exits 0 and prints the verdict for a good container", r.stdout + r.stderr)
        r = run(["--verify-source", bad, gguf])
        check(r.returncode == 1 and "verify-source: FAIL" in r.stdout and GGUF_TENSOR in r.stdout,
              "--verify-source exits 1 and names the tensor for a bad container", r.stdout + r.stderr)

        # A container that shares no tensor with the GGUF is not evidence either way: the command says
        # it cannot verify (exit 2) instead of passing or failing.
        disjoint = os.path.join(td, "disjoint")
        write_container(disjoint, pack_ours(w), name="model.layers.5.self_attn.q_proj.weight")
        r = run(["--verify-source", disjoint, gguf])
        check(r.returncode == 2 and "WARNING: cannot verify" in r.stdout,
              "--verify-source exits 2 and warns when nothing is comparable", r.stdout + r.stderr)

        # 3. --skip-verify: the escape for tight loops records "unverified", never engine-1, and the
        # cache-hit gate answers "cannot tell" so serve warns about it instead of passing silently.
        conv_good = fake_converter(os.path.join(td, "conv-good"), os.path.join(good, "model.q4nx"))
        out_skip = os.path.join(td, "out-skip")
        r = run([gguf, out_skip, conv_good, "--skip-verify"])
        check(r.returncode == 0, "a repack with --skip-verify succeeds", r.stdout + r.stderr)
        s = stamp_of(out_skip)
        check("tile_order=unverified" in s and "skipped (--skip-verify)" in s,
              "--skip-verify stamps the directory unverified, not engine-1", s)
        r = run(["--check-stamp", out_skip], env={"ONEBIT_Q4NX_CONVERTER": conv_good})
        check(r.returncode == 2 and "UNVERIFIED" in r.stdout,
              "a skipped container is 'cannot tell' (serve warns), not 'current'", r.stdout + r.stderr)

        # 4. A repack that DOES verify stamps what it measured, and its cache hit is current.
        out_good = os.path.join(td, "out-good")
        r = run([gguf, out_good, conv_good])
        check(r.returncode == 0 and "the container is verified against" in r.stdout,
              "a repack verifies the container it just wrote and says so", r.stdout + r.stderr)
        s = stamp_of(out_good)
        check("tile_order=engine-1" in s and "corr +1.0000" in s,
              "and stamps tile_order=engine-1 with the measured verdict", s)
        r = run(["--check-stamp", out_good], env={"ONEBIT_Q4NX_CONVERTER": conv_good})
        check(r.returncode == 0 and "stamp: current" in r.stdout, "the verified directory is current", r.stdout + r.stderr)

        # 5. A directory the repack refused must not be reused: the refusal leaves model.q4nx on disk,
        # so the next cache hit is stale (serve discards and re-repacks, and the refusal repeats loudly)
        # rather than "current".  The failure names the converter commit, as the issue asks.
        conv_bad = fake_converter(os.path.join(td, "conv-bad"), os.path.join(bad, "model.q4nx"))
        out_bad = os.path.join(td, "out-bad")
        r = run([gguf, out_bad, conv_bad])
        check(r.returncode == 1 and "verify-source FAIL" in r.stdout and GGUF_TENSOR in r.stdout,
              "a repack of a bad container fails loudly and names the tensor", r.stdout + r.stderr)
        check("repack: converter:" in r.stdout and "(unknown)" in r.stdout,
              "and names the converter checkout and commit", r.stdout + r.stderr)
        s = stamp_of(out_bad)
        check("refused=tile order" in s and "tile_order=unknown" in s,
              "the refused directory is stamped with the measurement", s)
        r = run(["--check-stamp", out_bad], env={"ONEBIT_Q4NX_CONVERTER": conv_bad})
        check(r.returncode == 1 and "refused" in r.stdout,
              "a cache hit on a refused directory is stale, not current", r.stdout + r.stderr)

        # 6. A stamp from before the measurement is rebuilt rather than reused on trust: the rebuild is
        # what turns it into a verified directory.
        old = os.path.join(td, "out-old")
        os.makedirs(old)
        with open(os.path.join(old, "repack-stamp.txt"), "w", encoding="utf-8") as f:
            f.write(f"converter_path={conv_good}\nconverter_commit=(unknown)\nq4nx_default_tensor_type=Q4_1\n")
        r = run(["--check-stamp", old], env={"ONEBIT_Q4NX_CONVERTER": conv_good})
        check(r.returncode == 1 and "no measured tile_order" in r.stdout,
              "a pre-measurement stamp is stale (rebuilt), not current", r.stdout + r.stderr)

    print(f"repack_verify_source_test: {len(FAIL)} failure(s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
