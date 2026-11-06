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
"""`tools/onebp_import.py` on a synthetic package: the five behaviours a real 1BP file cannot show cheaply.

1. the package's tile is in the .1bp intra-tile layout (planes row-major, codes adjacent-pair) and the
   container must hold it permuted into THIS engine's layout -- decoded here with the engine's own
   reader (`tools/q4nx_diff.decode_ours`).  A verbatim copy decodes as noise: measured on
   ZR1-1.5B-1BP, correlation 0.003 with the model; after the permutation, 0.995-0.997.
2. a v1 package's index claims the header's Q4NX quant for EVERY entry, while its norms and biases are
   raw f32 -- the importer has to classify by rank and size, and `--verify` has to use that same
   classification.  (Verify read the raw index quant when this test was written and reported 141 of 337
   "differing" tensors on the ZR1-1.5B container it had just written itself.)
3. the check is not vacuous: one flipped byte in a tile tensor is found and named.
4. a name with no Q4NX mapping (Zaya1's `cca_conv_grp`) is refused by name and no container is written,
   which is the whole reason the packer for those models is blocked.
5. a package whose quant is not Q4NX tiles is refused instead of decoded as noise.

usage: tests/onebp_import_test.py     (needs numpy; no gguf, no model, no device)
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, os.pardir, "tools", "onebp_import.py")

FAIL = []
# The header layout, spelled out here instead of imported from the tool on purpose: a field written in
# the wrong order has to fail this test, so the two must not share one declaration.
HEADER_FMT = "<" + "I" * 5 + "i" * 8 + "I" * 24          # 148 of the 256 bytes; model_tag sits at 192
MAGIC, Q4NX, I8 = 0x00504231, 0, 1
DIMS = dict(hidden_size=256, num_layers=1, num_attention_heads=8, num_kv_heads=2, head_dim=32,
            intermediate_size=512, vocab_size=64, max_seq_len=512, tile_rows=32, tile_cols=256,
            group_size=32)


def check(ok, what, why=""):
    if ok:
        print("ok  ", what)
        return
    FAIL.append(what)
    print("FAIL", what, ("  -- " + why) if why else "")


def header(version=1, quant=Q4NX, tensor_count=0, **over):
    """The 37 header fields in the format's order, then zero padding to 256 B."""
    d = dict(DIMS)
    d.update(over)
    ints = [d[k] for k in ("hidden_size", "num_layers", "num_attention_heads", "num_kv_heads", "head_dim",
                           "intermediate_size", "vocab_size", "max_seq_len")]
    vals = [MAGIC, version, 0, quant, 0] + ints + [d[k] for k in ("tile_rows", "tile_cols", "group_size")]
    vals += [0] * 6                                    # fields 17-22: has_q_norm .. eos_token_id
    vals += [tensor_count]                             # field 23
    vals += [0] * 14                                   # fields 24-37: num_experts .. n_rot_full
    return struct.pack(HEADER_FMT, *vals) + b"\0" * (256 - struct.calcsize(HEADER_FMT))


def onebp_tile(w):
    """Encode a (32, 256) weight as a .1bp Q4NX tile: planes row-major [r*8+g], codes adjacent-pair
    (even column = low nibble).  This is the layout a real package holds -- NOT this engine's."""
    rows, cols = w.shape
    g = w.reshape(rows, 8, 32)
    mn, mx = g.min(axis=-1), g.max(axis=-1)
    sc = np.where(mx == mn, 1.0, (mx - mn) / 15.0)
    q = np.clip(np.rint((g - mn[..., None]) / sc[..., None]), 0, 15).astype(np.uint8).reshape(rows, cols)
    planes = np.concatenate([bf16_bits(sc).reshape(-1), bf16_bits(mn + 8.0 * sc * 0).reshape(-1)])
    codes = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)          # [rows, cols/2]
    return planes.tobytes() + codes.tobytes()


def decode_engine_tile(tile_bytes, rows=32, cols=256):
    """Decode a tile the way this ENGINE does (tools/q4nx_diff.decode_ours), to prove the import put it
    in the engine's layout.  Imported here, not re-implemented: the engine's own reader is the judge."""
    sys.path.insert(0, os.path.join(HERE, os.pardir, "tools"))
    import q4nx_diff
    return q4nx_diff.decode_ours(np.frombuffer(tile_bytes, dtype=np.uint8), rows, cols)


def bf16_bits(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def write_package(path, tensors, quant=Q4NX):
    """tensors: [(name, ndim, dims, bytes)] -> a v1 package (no per-entry quant)."""
    h = header(1, quant, tensor_count=len(tensors))
    index, data, off = b"", b"", 0
    for name, ndim, dims, blob in tensors:
        nm = name.encode()                              # the length excludes the NUL terminator
        index += struct.pack("<I", len(nm)) + nm + b"\0" + struct.pack("<I", ndim)
        index += struct.pack("<%dI" % ndim, *dims) + struct.pack("<QQ", off, len(blob))
        data += blob
        off += len(blob)
    with open(path, "wb") as f:
        f.write(h + index + data)


def run(args):
    p = subprocess.run([sys.executable, TOOL] + args, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def snapshot(dir_path):
    """A --config directory whose dims agree with the package, plus a tokenizer stub."""
    with open(os.path.join(dir_path, "config.json"), "w") as f:
        json.dump({"hidden_size": DIMS["hidden_size"], "num_hidden_layers": DIMS["num_layers"],
                   "num_attention_heads": DIMS["num_attention_heads"],
                   "num_key_value_heads": DIMS["num_kv_heads"],
                   "intermediate_size": DIMS["intermediate_size"],
                   "vocab_size": DIMS["vocab_size"]}, f)
    with open(os.path.join(dir_path, "tokenizer.json"), "w") as f:
        f.write('{"model":{}}\n')


def main():
    rng = np.random.default_rng(5)
    w_q = rng.standard_normal((32, 256)) * 0.05           # the weight the tile must come back as
    w_o = rng.standard_normal((32, 256)) * 0.05
    w_e = rng.standard_normal((32, 256)) * 0.05
    tile_q, tile_o, tile_e = (onebp_tile(w) for w in (w_q, w_o, w_e))
    norm = np.linspace(-1.0, 1.0, 256, dtype=np.float32).tobytes()      # 1-D f32: NOT tiles
    good = [("token_embd.weight", 2, (32, 256), tile_e),
            ("output.weight", 2, (32, 256), tile_o),
            ("blk.0.attn_q.weight", 2, (32, 256), tile_q),
            ("blk.0.attn_norm.weight", 1, (256,), norm)]
    with tempfile.TemporaryDirectory() as tmp:
        pkg = os.path.join(tmp, "m.1bp")
        cfg, out = os.path.join(tmp, "cfg"), os.path.join(tmp, "out")
        os.makedirs(cfg)
        snapshot(cfg)
        write_package(pkg, good)

        rc, log = run([pkg, out, "--config", cfg])
        check(rc == 0, "a v1 package with a Q4NX tile tensor and an f32 norm imports", log.strip()[-300:])
        check(os.path.exists(os.path.join(out, "model.q4nx")), "the container was written")
        with open(os.path.join(out, "model.q4nx"), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            table = json.loads(f.read(n))
        check(table["model.layers.0.self_attn.q_proj.weight"]["dtype"] == "I8",
              "the tile tensor stays I8 tiles (permuted, not re-quantized)")
        check(table["model.layers.0.input_layernorm.weight"]["dtype"] == "BF16",
              "the f32 norm becomes bf16, and is NOT read as tiles")

        # the import must have PERMUTED the tile into this engine's layout: decode it with the engine's
        # own reader and compare against the weight the package encoded.  A verbatim copy (the first
        # version of this tool) decodes as noise: measured on ZR1-1.5B-1BP, correlation 0.003.
        lo = table["model.layers.0.self_attn.q_proj.weight"]["data_offsets"][0]
        with open(os.path.join(out, "model.q4nx"), "rb") as f:
            f.seek(8 + n + lo)
            got = decode_engine_tile(f.read(5120))
        # the tile round-trips the quantized weight, not w_q itself (q4_1 is lossy by construction)
        err = np.abs(got - w_q).max()
        check(err < 0.02, "the container's tile decodes to the package's weight through the engine's reader",
              "max|err| %.3e" % err)

        rc, log = run([pkg, out, "--verify"])
        check(rc == 0 and "all identical" in log,
              "--verify passes on the container it just wrote (a v1 index quant must not decide the kind)", log.strip()[-300:])

        # not vacuous: one flipped byte in the tile payload
        path = os.path.join(out, "model.q4nx")
        lo = table["model.layers.0.self_attn.q_proj.weight"]["data_offsets"][0]
        with open(path, "r+b") as f:
            f.seek(8 + n + lo + 3000)
            b = f.read(1)
            f.seek(8 + n + lo + 3000)
            f.write(bytes([b[0] ^ 0xFF]))
        rc, log = run([pkg, out, "--verify"])
        check(rc == 1 and "1 differ" in log and "q_proj" in log,
              "one flipped tile byte is found and named", log.strip()[-300:])

        # a name with no Q4NX mapping: refused by name, nothing written
        bad = good + [("blk.0.cca_conv_grp.weight", 2, (32, 256), rng.integers(0, 256, size=5120, dtype=np.uint8).tobytes())]
        pkg2 = os.path.join(tmp, "z.1bp")
        out2 = os.path.join(tmp, "out2")
        write_package(pkg2, bad)
        rc, log = run([pkg2, out2, "--config", cfg])
        check(rc != 0 and "no Q4NX name for blk.0.cca_conv_grp.weight" in log,
              "a tensor with no Q4NX mapping is refused by name", log.strip()[-300:])
        check(not os.path.exists(os.path.join(out2, "model.q4nx")),
              "and no container is left behind for it")

        # a package that is not Q4NX tiles at all
        pkg3 = os.path.join(tmp, "i.1bp")
        write_package(pkg3, good, quant=I8)
        rc, log = run([pkg3, os.path.join(tmp, "out3"), "--config", cfg])
        check(rc != 0 and "only Q4NX" in log, "a non-Q4NX package is refused, not decoded", log.strip()[-300:])

    print("onebp_import_test: %d failure(s)" % len(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
