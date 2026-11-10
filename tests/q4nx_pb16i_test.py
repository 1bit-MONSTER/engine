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
"""tools/q4nx_pb16i.py: the pb16i image contract, checked without a model or a device.

usage: tests/q4nx_pb16i_test.py        (needs numpy)

1. Every geometry the 27B uses produces exactly K*N/2 + (K/32)*N*2 bytes = tiles*4,608,
   and a geometry that is not whole tiles is refused.
2. The packer puts element (i,j) where npu/forward/include/q4nx_pb16i.h says it is
   (restated here from the header's own rule: tile = (c, ki, nt, r), code at
   ((kb*2+nb)*16+rr)*16+cc / 2, ms at 4096 + 2*((nb*8+g)*16+cc)), and the mapping is a
   bijection: each code byte is claimed twice, each ms entry 32 times, per tile.
3. A synthetic ternary tensor round-trips: codes and ms read back out of the image equal
   what went in, for several (K,N).
4. The side table is e_ref int16[N] little-endian followed by one mode byte, and the
   int32/int64 split follows the converter's own worst-case bound.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import q4nx_pb16i as Q  # noqa: E402

FAIL = []
SKIPPED = []   # checks that could not run here; a pass line must name them


def check(ok, what, K=0, N=0):
    if not ok:
        FAIL.append(f"{what} (K={K} N={N})")


def header_index(i, j, K, N):
    """Restated from q4nx_pb16i.h::index(), independently of the tool's own copy."""
    cw = N // (Q.COLS * Q.ROWS)
    c = j // (N // Q.COLS)
    jr = j % (N // Q.COLS)
    r = jr // cw
    jn = jr % cw
    nt, nb, cc = jn // Q.NW, (jn % Q.NW) // 16, jn % 16
    kb, rr = (i % Q.KC) // 16, (i % Q.KC) % 16
    tile = (((c * (K // Q.KC) + i // Q.KC) * (cw // Q.NW) + nt) * Q.ROWS + r)
    e = ((kb * 2 + nb) * 16 + rr) * 16 + cc
    return tile * Q.TILE + e // 2, bool(e & 1), tile * Q.TILE + Q.CODE_BYTES + \
        2 * ((nb * 8 + ((i % Q.KC) // 32)) * 16 + cc)


GEOMS = [(5120, 17408), (17408, 5120), (5120, 10240), (6144, 5120), (5120, 6144),
         (5120, 1024), (256, 1024)]


def test_geometry():
    for K, N in GEOMS:
        check(Q.fits(K, N), "fits() refused a whole-tile geometry", K, N)
        check(Q.w_bytes(K, N) == K * N // 2 + (K // 32) * N * 2, "size != codes + ms", K, N)
        check(Q.w_bytes(K, N) % Q.TILE == 0, "size is not whole tiles", K, N)
        check(Q.w_bytes(K, N) // Q.TILE == 8 * (K // 256) * ((N // 32) // 32) * 4,
              "tile count", K, N)
        # the tool's index and the header's reading of the same element must agree
        for i in range(0, K, max(1, K // 7)):
            for j in range(0, N, max(1, N // 7)):
                a, b = Q.index(i, j, K, N), header_index(i, j, K, N)
                check(a == b, "tool index != header index", K, N)
    for K, N in [(255, 1024), (256, 1000), (0, 1024), (256, 0)]:
        check(not Q.fits(K, N), "a non-whole-tile geometry must be refused", K, N)


def test_bijection():
    for K, N in [(5120, 1024), (256, 1024), (5120, 6144)]:
        code_hits = np.zeros(Q.CODE_BYTES, dtype=np.int32)
        ms_hits = np.zeros((Q.TILE - Q.CODE_BYTES) // 2, dtype=np.int32)   # 256 int16 entries
        cw, nt_n, kt = N // (Q.COLS * Q.ROWS), N // (Q.COLS * Q.ROWS * Q.NW), K // Q.KC
        for c in (0, Q.COLS - 1):
            for ki in (0, kt - 1):
                for nt in (0, nt_n - 1):
                    for r in (0, Q.ROWS - 1):
                        code_hits[:] = 0
                        ms_hits[:] = 0
                        col0 = c * (N // Q.COLS) + r * cw + nt * Q.NW
                        base = (((c * kt + ki) * nt_n + nt) * Q.ROWS + r) * Q.TILE
                        for kk in range(Q.KC):
                            for nn in range(Q.NW):
                                cb, hi, mb = Q.index(ki * Q.KC + kk, col0 + nn, K, N)
                                code_hits[cb - base] += 1
                                ms_hits[(mb - base - Q.CODE_BYTES) // 2] += 1
                        check((code_hits == 2).all(), "a code byte is not claimed twice", K, N)
                        check((ms_hits == 32).all(), "an ms entry is not claimed 32 times", K, N)


def test_roundtrip():
    rng = np.random.default_rng(7)
    for K, N in [(5120, 1024), (256, 1024), (5120, 6144), (6144, 5120)]:
        codes = rng.integers(-1, 2, size=(K, N), dtype=np.int8)
        e = rng.integers(-18, -15, size=(K // 128, N))
        m = rng.integers(1100, 1600, size=(K // 128, N))
        bits = np.ldexp(m.astype(np.float64), e).astype(np.float16).view(np.uint16)
        img, e_ref, mode = Q.pack(codes, bits, K, N)
        check(img.shape == (Q.w_bytes(K, N) // Q.TILE, Q.TILE), "image shape", K, N)
        bad = Q.verify_image(img, codes, bits, K, N, e_ref)
        check(bad == 0, f"{bad} image slots disagree with the source scales", K, N)
        # w reconstructed from the image == code * m * 2^e
        got = np.empty((K, N), dtype=np.float64)
        for i in range(0, K, max(1, K // 11)):
            for j in range(0, N, max(1, N // 11)):
                cb, hi, mb = Q.index(i, j, K, N)
                byte = int(img.reshape(-1)[cb])
                trit = (byte >> 4) if hi else (byte & 0x0F)
                trit = trit - 16 if trit >= 8 else trit
                ms = int(np.frombuffer(img.reshape(-1)[mb:mb + 2].tobytes(), dtype='<i2')[0])
                got[i, j] = np.ldexp(float(trit * ms), int(e_ref[j]))
        # float64 is exact here: |trit*ms| <= 32767 and the scale is a power of two
        per_group = np.repeat(m.astype(np.float64) * np.ldexp(1.0, e.astype(np.int64)), 128,
                              axis=0)                      # [K][N] the real weight scale
        want = codes.astype(np.float64) * per_group
        for i in range(0, K, max(1, K // 11)):
            for j in range(0, N, max(1, N // 11)):
                check(got[i, j] == want[i, j], "w reconstructed from the image != code*scale", K, N)
        del want
        # the side table
        ref = Q.side_table(e_ref, mode)
        check(len(ref) == 2 * N + 1, "side table size", K, N)
        check(ref[-1] == mode and mode in (0, 1), "mode byte", K, N)
        back = np.frombuffer(ref[:-1], dtype='<i2')
        check(np.array_equal(back, e_ref), "e_ref round-trip", K, N)


def test_ptq1_0():
    """The reader and packer against a real ternary GGUF, end to end.

    There is no synthetic encoder here on purpose: PTQ1_0's 128 trits per block are not a
    plain base-3 number (ggml stores them in staged 32/16/8 chunks, five trits per byte),
    so a hand-built block would test my own inverse, not ggml's layout.  Instead: pack a
    real tensor and compare the image BYTE FOR BYTE against a set whose device output was
    already gated bit-exact (a set this packer wrote).  Skipped when the model or that set is
    not named on this machine -- and the skip is then reported IN THE PASS LINE, because this is
    the file's only check of the reader and a bare "ok" would read as if it had run."""
    gguf = os.environ.get("Q4NX_PB16I_GGUF",
                          "/home/bcloud/models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
    # No default reference set on purpose.  The set this test used to name
    # (`bonsai2-pb16[I]-sidecars`, and the name was misspelled so the test skipped on every run anyway) was built
    # 2026-10-31 21:40, before this packer existed (`tools/q4nx_pb16i.py` was created in 88715dea, 2026-11-01 04:51,
    # together with this test), and it comes from the container's own converter (`config.json` ->
    # `q4nx_source.converter = qwen35_to_q4nx.py`).  Measured 2026-11-08 on bonsai2-27b: `pack` differs from all 8
    # images of layer 3 byte for byte at identical sizes, and `verify` reports MISMATCH on 14/14 tensors of layers
    # 6-7 -- i.e. that set is NOT this packer's output, so comparing against it asserted nothing.  Byte equality is
    # only meaningful against a set THIS packer wrote, so it must be named explicitly (and can then be regenerated
    # and device-gated on purpose) rather than inherited from a directory name that happens to exist.
    ref = os.environ.get("Q4NX_PB16I_REF", "")
    if not os.path.exists(gguf) or not ref or not os.path.isdir(ref):
        # This is the ONLY test in the file that reads the reader, and it used to report its absence as part of the
        # suite's "ok" line.  A CI run on a machine without the model then read as "the reader was checked", which is
        # how the byte-for-byte claim below went unverified (measured 2026-11-08: `q4nx_pb16i_test: ok` while `pack`
        # differed from every image in the reference set).  Say SKIPPED in the pass line, and let
        # Q4NX_PB16I_REQUIRE=1 turn the absence into a failure for a machine that is supposed to have the model.
        why = []
        if not os.path.exists(gguf):
            why.append(f"the ternary GGUF ({gguf})")
        if not ref or not os.path.isdir(ref):
            why.append("Q4NX_PB16I_REF, a reference set THIS packer wrote (none by default, see above)")
        SKIPPED.append("reader test needs " + " and ".join(why))
        print("  (reader test SKIPPED: needs " + " and ".join(why) + ")")
        if os.environ.get("Q4NX_PB16I_REQUIRE"):
            check(False, "Q4NX_PB16I_REQUIRE is set, so the reader test may not skip", 0, 0)
        return
    want = {}
    for name, dims, tt, raw in Q.gguf_tensors(gguf):
        if name in ("blk.0.attn_gate.weight", "blk.3.attn_k.weight"):
            want[name] = (dims, raw)
    for name, (dims, raw) in want.items():
        K, N = dims
        codes, bits = Q.read_ptq1_0(raw, K, N)
        check(codes.shape == (K, N), "reader shape", K, N)
        check(int(np.abs(codes).max()) <= 1, "reader returned a non-ternary code", K, N)
        img, e_ref, mode = Q.pack(codes, bits, K, N)
        stem = name[:-len(".weight")].replace(".", "_") + "_weight"
        for suffix, got in ((".pb16i", img.reshape(-1).tobytes()),
                            (".ref", Q.side_table(e_ref, mode))):
            path = os.path.join(ref, stem + suffix)
            if not os.path.exists(path):
                print(f"  (no reference {stem}{suffix}; skipped)")
                continue
            with open(path, "rb") as f:
                check(got == f.read(), f"{stem}{suffix} != the gated reference image", K, N)


def _tiny_gguf(path, tensors):
    """Write a GGUF with `tensors` = [(name, dims, ggml_type, payload bytes)] and no metadata.

    Hand-rolled because the fixture needs control of a QUANTIZED payload (a q4_0 block), which the repo's
    GGUFWriter helper does not offer -- and because a test that pulls in the writer code it is testing is not
    a test. Layout: header, tensor infos, then the data section aligned to 32 (ggml's default), which is the
    same alignment the reader assumes."""
    import struct
    body = b""
    infos = b""
    for name, dims, tt, payload in tensors:
        nb = name.encode()
        infos += struct.pack("<Q", len(nb)) + nb
        infos += struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<I", tt) + struct.pack("<Q", len(body))
        body += payload
        body += b"\0" * (-len(body) % 32)
    head = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), 0) + infos
    with open(path, "wb") as f:
        f.write(head)
        f.write(b"\0" * (-(len(head)) % 32))
        f.write(body)


def test_q4_0_source_reader():
    """engine#1605: a sidecar can be checked against a q4_0 source only if the reader can read one.

    `gguf_tensors` used to size every non-PTQ1_0 tensor as 0 bytes and yield b"", so a caller could not tell
    "this tensor is empty" from "this reader cannot size it" -- the wall the pb16i sidecar check hit. The three
    arms below are: a q4_0 tensor is sized and decodes to the values it was built from, an unknown type is still
    skipped quietly by default, and strict=True refuses it by name instead.
    """
    import tempfile
    K, N = 64, 4
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 16, size=(N, K // 32, 32), dtype=np.uint8)
    d = (rng.standard_normal((N, K // 32)) * 0.02).astype(np.float16)
    payload = b""
    for col in range(N):
        for b in range(K // 32):
            qs = codes[col, b]
            packed = (qs[0:16] | (qs[16:32] << 4)).astype(np.uint8).tobytes()
            payload += d[col, b].tobytes() + packed
    with tempfile.TemporaryDirectory() as td:
        g = os.path.join(td, "t.gguf")
        _tiny_gguf(g, [("blk.0.ffn_gate.weight", (K, N), Q.Q4_0, payload),
                       ("blk.0.norm.weight", (K,), 0, np.zeros(K, np.float32).tobytes())])
        got = {n: (dims, tt, raw) for n, dims, tt, raw in Q.gguf_tensors(g)}
        dims, tt, raw = got["blk.0.ffn_gate.weight"]
        check(tt == Q.Q4_0 and len(raw) == N * (K // 32) * Q.Q4_0_BLOCK,
              "a q4_0 tensor is sized from its dims, not skipped", K, N)
        check(got["blk.0.norm.weight"][2] == b"", "an unsized type is still skipped by default (no behaviour change)", K, N)
        try:
            list(Q.gguf_tensors(g, strict=True))
            check(False, "strict refuses an unsized type by name", K, N)
        except SystemExit as e:
            check("blk.0.norm.weight" in str(e) and "cannot size" in str(e),
                  "strict refuses an unsized type by name", K, N)
        # engine#1605 follow-up: one tensor by name. `strict=True` refuses the whole file at the first unsized
        # tensor, so a caller that wants a single q4_0 tensor out of a real model has no way to ask for it; these
        # three arms pin the two answers it must keep apart -- absent, and present-but-unsized.
        t_dims, t_tt, t_raw = Q.tensor(g, "blk.0.ffn_gate.weight")
        check(t_dims == (K, N) and t_tt == Q.Q4_0 and t_raw == raw,
              "tensor(name) returns the one tensor asked for, bytes intact", K, N)
        try:
            Q.tensor(g, "blk.0.norm.weight")
            check(False, "tensor() refuses an unsized type by name", K, N)
        except SystemExit as e:
            check("blk.0.norm.weight" in str(e) and "cannot size" in str(e),
                  "tensor() refuses an unsized type by name", K, N)
        try:
            Q.tensor(g, "blk.0.nope.weight")
            check(False, "tensor() refuses an absent tensor by name", K, N)
        except SystemExit as e:
            check("no tensor named blk.0.nope.weight" in str(e),
                  "tensor() refuses an absent tensor by name", K, N)
        vals, dgot = Q.read_q4_0(raw, K, N)
        check(vals.shape == (K, N) and dgot.shape == (K // 32, N), "q4_0 decode shapes", K, N)
        want = np.empty((N, K // 32, 32), np.float32)
        want[:] = (codes.astype(np.float32) - 8.0) * d.astype(np.float32)[:, :, None]
        err = float(np.abs(vals.T.reshape(N, K // 32, 32) - want).max())
        check(err == 0.0, f"q4_0 decodes to the values it was built from (max|err| {err:.3e})", K, N)


def main():
    test_geometry()
    test_bijection()
    test_roundtrip()
    test_ptq1_0()
    test_q4_0_source_reader()
    if FAIL:
        for f in FAIL[:20]:
            print("FAIL", f)
        print(f"q4nx_pb16i_test: {len(FAIL)} failure(s)")
        return 1
    # A green line must say what was NOT checked: this suite's only reader test needs a model that CI does not carry,
    # and the rest of the file drives synthetic data against the header's rule.
    if SKIPPED:
        print(f"q4nx_pb16i_test: ok ({len(SKIPPED)} SKIPPED - {SKIPPED[0]})")
        return 0
    print("q4nx_pb16i_test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
