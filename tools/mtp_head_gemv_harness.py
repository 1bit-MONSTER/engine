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
"""mtp_head_gemv_harness.py -- freeze the harness the arena runner needs to score ONE MTP head matrix.

The head's matrices are not arena tasks: they are the MTP layer's own weights, and the 08:00 window measures
them on the sealed M=16 pb16i ELFs (down -> D, gate/up -> G/U, o -> O, q + the output gate -> QKVFAQ twice,
k/v -> QKVFAKV).  For the runner to say anything more than "it ran", it needs a frozen fixture:

    harness/geometry.txt        K, N, M=16
    harness/seed1_A.bin         16 rows x K int8 activations (the device consumes int8 A directly)
    harness/seed1_expected.bin  16 x N int32, the EXACT integer the device must produce:
                                C[m,n] = sum_g ( sum_{k in g} a[m,k] * code[k,n] ) * ms[g,n]
                                (codes are the source's q4-8, ms the emitted int16 magnitudes)
    harness/seed1_float.bin     16 x N float32, the float GEMV of the SAME activations against the exact source
                                weights -- so the run reports both "the device is byte-exact for the integer
                                part" and "integer part + host add term == the float oracle".
    LAYOUT                      W_BYTES=<image bytes>  (the runner streams the image as an opaque BO)
    HOST_ADD.npy                the (K/32, N) float32 table the host applies as sum_g add[g,n] * A_g[m]

Everything is derived from ~/models/npu/bonsai2-27b-mtp-head/{q4_k,pb16i-exact}, so the fixture cannot drift
from the artifact under test without this script being re-run.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import mtp_head_ref as ref  # noqa: E402
import mtp_head_pb16i_exact as exact_conv  # noqa: E402  (the converter's own packer: one tile order, one truth)

_plan = exact_conv.exact_plan_from_cells
_pack = exact_conv.pack_tiles

G = 32
M_ROWS = 16

# matrix -> the sealed M=16 ELF that has its geometry
SEALED = {
    "down": ("D_K17408_N5120", "full_i8_D_K17408_N5120_pb16i_m16.elf"),
    "gate": ("G_K5120_N17408", "full_i8_G_K5120_N17408_pb16i_m16.elf"),
    "up": ("U_K5120_N17408", "full_i8_U_K5120_N17408_pb16i_m16.elf"),
    "o": ("O_K6144_N5120", "full_i8_O_K6144_N5120_pb16i_m16.elf"),
    "k": ("QKVFAKV_K5120_N1024", "full_i8_QKVFAKV_K5120_N1024_pb16i_m16.elf"),
    "v": ("QKVFAKV_K5120_N1024", "full_i8_QKVFAKV_K5120_N1024_pb16i_m16.elf"),
}


def unpack_tiles(img, K, N):
    """Image -> ([K][N] int8 codes, [K/32][N] int16 magnitudes), the sealed pb16i order."""
    PB_KT, PB_C, PB_R, PB_W = 256, 8, 4, 32
    PB16_TILE = PB_KT * PB_W // 2 + (PB_KT // 32) * PB_W * 2
    KT, NT = K // PB_KT, N // (PB_C * PB_R * PB_W)
    W = np.frombuffer(img, dtype=np.uint8).reshape(-1, PB16_TILE)
    nib = W[:, :4096]
    lo = (nib & 0x0F).astype(np.int8)
    hi = (nib >> 4).astype(np.int8)
    c = np.empty((W.shape[0], 8192), dtype=np.int8)
    c[:, 0::2] = np.where(lo >= 8, lo - 16, lo)
    c[:, 1::2] = np.where(hi >= 8, hi - 16, hi)
    codes = c.reshape(PB_C, KT, NT, PB_R, 16, 2, 16, 16).transpose(1, 4, 6, 0, 3, 2, 5, 7).reshape(K, N)
    m = np.ascontiguousarray(W[:, 4096:]).view("<i2").reshape(PB_C, KT, NT, PB_R, 2, 8, 16)
    ms = m.transpose(1, 5, 0, 3, 2, 4, 6).reshape(K // G, N)
    return codes, ms


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=os.path.expanduser("~/models/npu/bonsai2-27b-mtp-head"))
    ap.add_argument("--out", default=os.path.expanduser("~/wt/npu-arena/head-tasks"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)

    exact = os.path.join(a.head, "pb16i-exact")
    man = json.load(open(os.path.join(exact, "manifest.json")))
    rng = np.random.default_rng(a.seed)
    made = []
    for e in man["tensors"]:
        name = e["tensor"]
        if name not in SEALED:
            continue          # fc (K=10240) has no sealed ELF: it runs on the host
        elf_tag, elf_file = SEALED[name]
        K, N = e["K"], e["N"]
        img = np.fromfile(os.path.join(exact, e["image"]), dtype=np.uint8)
        codes, ms = unpack_tiles(img, K, N)                       # [K][N], [K/32][N]
        eref = np.fromfile(os.path.join(exact, e["eref"]), dtype="<i2")
        add = np.fromfile(os.path.join(exact, e["add"]), dtype="<f4").reshape(K // G, N)

        d = os.path.join(a.out, name)
        os.makedirs(os.path.join(d, "harness"), exist_ok=True)
        with open(os.path.join(d, "harness", "geometry.txt"), "w") as f:
            f.write(f"K {K}\nN {N}\nM {M_ROWS}\n")

        # activations: int8, which is what the device consumes; keep them away from the int8 rails so the
        # integer partials stay inside int32 (|C| <= 8 * 32 * 127 * |ms| summed over K/32 groups)
        A = rng.integers(-100, 101, size=(M_ROWS, K), dtype=np.int16).astype(np.int8)
        A.tofile(os.path.join(d, "harness", f"seed{a.seed}_A.bin"))

        # the exact integer the device must produce: per 32-K group, (A_g . codes) * ms
        C = np.zeros((M_ROWS, N), dtype=np.int64)
        for g in range(K // G):
            ag = A[:, g * G:(g + 1) * G].astype(np.int64)              # [M][32]
            P = ag @ codes[g * G:(g + 1) * G, :].astype(np.int64)      # [M][N]  (int32 on the device)
            C += P * ms[g, :].astype(np.int64)[None, :]
        over = int(np.abs(C).max())
        if over >= 2 ** 31:
            raise SystemExit(f"{name}: integer partial would overflow int32 (max {over}); refusing")
        C.astype("<i4").tofile(os.path.join(d, "harness", f"seed{a.seed}_expected.bin"))

        # the float oracle for the same rows: exact source weights + the host add term
        y = (C.astype(np.float64) * np.exp2(eref.astype(np.float64))[None, :])
        for g in range(K // G):
            y += A[:, g * G:(g + 1) * G].astype(np.float64).sum(axis=1)[:, None] * add[g, :].astype(np.float64)[None, :]
        y.astype("<f4").tofile(os.path.join(d, "harness", f"seed{a.seed}_expected_float.bin"))
        np.save(os.path.join(d, "HOST_ADD.npy"), add.astype(np.float32))
        eref.astype("<i2").tofile(os.path.join(d, "EREF"))
        with open(os.path.join(d, "LAYOUT"), "w") as f:
            f.write(f"W_BYTES={img.nbytes}\n")
        with open(os.path.join(d, "ELF"), "w") as f:
            f.write(f"{elf_tag}\n{elf_file}\n")
        with open(os.path.join(d, "IMAGE"), "w") as f:
            f.write(os.path.join(exact, e["image"]) + "\n")
        made.append((name, elf_tag, K, N, img.nbytes, over))
        print(f"{name:5s} {K:6d}x{N:<6d} -> {elf_tag:20s} image {img.nbytes/1e6:6.2f} MB, "
              f"|C|max {over/1e6:8.2f} M, float-oracle rows written")
    # q is 12,288 columns but the sealed QKVFAQ ELF is 6,144 wide: the head's q tensor is exactly two halves
    # (q | output gate), and the source cell grid splits cleanly on the row-block axis, so each half is its own
    # image in the sealed order -- two launches, one per ELF.
    src = os.path.join(a.head, "q4_k", "model.q4nx")
    blob = open(src, "rb").read()
    import struct
    hlen = struct.unpack("<Q", blob[:8])[0]
    cfg = json.load(open(os.path.join(a.head, "q4_k", "config.json")))
    tq = next(t for t in cfg["tensors"] if t["name"] == "model.mtp.layers.0.self_attn.q_proj.weight")
    off = 8 + hlen + sum(x["bytes"] for x in cfg["tensors"][:cfg["tensors"].index(tq)])
    grid = [int(x) for x in tq["shape"]]          # [N/32][K/256] = [384][20]
    cells = np.frombuffer(blob[off:off + tq["bytes"]], dtype=np.uint8).reshape(grid[0], grid[1], 4736)
    env = json.load(open(os.path.join(exact, "manifest.json")))
    for half, sl in (("q_lo", slice(0, grid[0] // 2)), ("q_hi", slice(grid[0] // 2, grid[0]))):
        K, N = grid[1] * 256, (sl.stop - sl.start) * 32
        codes, ms, e_ref, add, w_exact, zeroed = _plan(cells[sl], [sl.stop - sl.start, grid[1]], N, K)
        img = _pack(codes, ms)
        d = os.path.join(a.out, half)
        os.makedirs(os.path.join(d, "harness"), exist_ok=True)
        with open(os.path.join(d, "harness", "geometry.txt"), "w") as f:
            f.write(f"K {K}\nN {N}\nM {M_ROWS}\n")
        A = rng.integers(-100, 101, size=(M_ROWS, K), dtype=np.int16).astype(np.int8)
        A.tofile(os.path.join(d, "harness", f"seed{a.seed}_A.bin"))
        C = np.zeros((M_ROWS, N), dtype=np.int64)
        for g in range(K // G):
            C += (A[:, g * G:(g + 1) * G].astype(np.int64) @ codes[g * G:(g + 1) * G, :].astype(np.int64)) \
                 * ms[g, :].astype(np.int64)[None, :]
        C.astype("<i4").tofile(os.path.join(d, "harness", f"seed{a.seed}_expected.bin"))
        y = C.astype(np.float64) * np.exp2(e_ref.astype(np.float64))[None, :]
        for g in range(K // G):
            y += A[:, g * G:(g + 1) * G].astype(np.float64).sum(axis=1)[:, None] * add[g, :].astype(np.float64)[None, :]
        y.astype("<f4").tofile(os.path.join(d, "harness", f"seed{a.seed}_expected_float.bin"))
        np.save(os.path.join(d, "HOST_ADD.npy"), add.astype(np.float32))
        e_ref.astype("<i2").tofile(os.path.join(d, "EREF"))
        img_path = os.path.join(d, "image.pb16i")
        img.astype(np.uint8).tofile(img_path)
        with open(os.path.join(d, "LAYOUT"), "w") as f:
            f.write(f"W_BYTES={img.nbytes}\n")
        with open(os.path.join(d, "ELF"), "w") as f:
            f.write("QKVFAQ_K5120_N6144\nfull_i8_QKVFAQ_K5120_N6144_pb16i_m16.elf\n")
        with open(os.path.join(d, "IMAGE"), "w") as f:
            f.write(img_path + "\n")
        made.append((half, "QKVFAQ_K5120_N6144", K, N, img.nbytes, int(np.abs(C).max())))
        print(f"{half:5s} {K:6d}x{N:<6d} -> QKVFAQ_K5120_N6144 image {img.nbytes/1e6:6.2f} MB, "
              f"|C|max {int(np.abs(C).max())/1e6:8.2f} M")

    print(f"\n{made and len(made)} head-matrix task(s) under {a.out} "
          f"(fc excluded: K=10240 has no sealed ELF and runs on the host)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
