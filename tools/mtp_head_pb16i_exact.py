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
"""mtp_head_pb16i_exact.py -- emit the MTP head as pb16i images that do NOT throw the additive min away.

The source head's cells are  w(n,k) = q4 * S[g,n] + M[g,n]  with an ADDITIVE per-(32-K group, column) min.
A pb16i tile has no bias term, so a plain conversion drops M and costs 11-18 % of the weight L1 per tensor
(that is what ~/models/npu/bonsai2-27b-mtp-head/pb16i/ reports).  It does not have to: the min is constant
per (group, column), so with A_g = sum_{k in g} a[k]

    y[n] = sum_g ms_g[n] * P_g[n]   (the device: int4 codes, int16 magnitudes, int32 partials)
         + sum_g (M_g[n] + 8*S_g[n]) * A_g    (the host, eight numbers per GEMV, shared by all columns)

and the +8 folds the unsigned -> signed code shift (code = q4 - 8, exactly representable in int4).  This tool
writes, per tensor:

    <tensor>.pb16i   the image in the sealed tile order (layout_w4.py pack-pb16i: [c][ki][nt][r], 4,608 B/tile)
    <tensor>.add     float32 [K/32][N]  the host table   M_g(n) + 8*S_g(n)
    <tensor>.eref    int16   [N]        the per-column exponent (weights = code * ms * 2^eref for the int part)
    manifest.json    geometry, per-tensor error vs the exact float weights, and how to use the table

Self-check: every tensor is decoded back from its emitted image, the full-matrix GEMV is compared against the
exact float weights (int part + host table), and the reference head step (tools/mtp_head_ref.py) is run twice --
once on the source cells and once on the emitted images -- to show the two agree to the int16 mantissa.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import mtp_head_ref as ref  # noqa: E402  (the oracle: cell decode, math, geometry)

PB_KT, PB_C, PB_R, PB_W = 256, 8, 4, 32      # the sealed pb16i tile geometry
PB16_TILE = PB_KT * PB_W // 2 + (PB_KT // 32) * PB_W * 2      # 4608
G = 32


def pack_tiles(codes, ms):
    """[K][N] int8 codes + [K/32][N] int16 magnitudes -> (tiles x 4608) bytes, layout_w4.py pack-pb16i order.

    The index transforms are the ones layout_w4.py uses (copied here so this tool has no dependency on the
    arena tree): codes [ki][kb][rr][c][r][nt][nb][cc] -> [c][ki][nt][r][kb][nb][rr][cc], magnitudes likewise.
    """
    K, N = codes.shape
    KT, NT = K // PB_KT, N // (PB_C * PB_R * PB_W)
    c8 = codes.reshape(KT, 16, 16, PB_C, PB_R, NT, 2, 16).transpose(3, 0, 5, 4, 1, 6, 2, 7)
    c8 = np.ascontiguousarray(c8).reshape(-1, 8192).astype(np.uint8) & 0x0F
    nib = (c8[:, 0::2] | (c8[:, 1::2] << 4)).astype(np.uint8)                  # [tiles][4096]
    m32 = ms                                                                   # already per 32-K group
    m7 = m32.reshape(KT, 8, PB_C, PB_R, NT, 2, 16).transpose(2, 0, 4, 3, 5, 1, 6)
    m7 = np.ascontiguousarray(m7).astype("<i2").reshape(-1, 256).view(np.uint8)  # [tiles][512]
    return np.concatenate([nib, m7], axis=1)


def unpack_tiles(img, K, N):
    """The inverse: image bytes -> ([K][N] int8 codes, [K/32][N] int16 magnitudes)."""
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


def exact_plan_from_cells(cells, grid, n_out, k_tot):
    """Decode a tensor's source cells -> (codes [K][N] int8, ms [K/32][N] int16, e_ref [N] int16,
    add [K/32][N] float32, w_exact [N][K] float32).

    The kernel takes ONE exponent per column, so the split is: S[g,n] = m[g,n] * 2^e[g,n] with |m| < 2^15,
    e_ref[n] = min_g e[g,n], and ms[g,n] = m[g,n] << (e[g,n] - e_ref[n]); it refuses if that does not fit
    int16 rather than silently wrapping (the same guard layout_w4.py's pb16i_params has).
    """
    codes = np.empty((k_tot, n_out), dtype=np.int8)               # [K][N]
    S_all = np.empty((k_tot // G, n_out), dtype=np.float64)       # per (32-K group, column) scale
    M_all = np.empty((k_tot // G, n_out), dtype=np.float64)
    w_exact = np.empty((n_out, k_tot), dtype=np.float32)          # [N][K]
    for rb in range(grid[0]):
        for cb in range(grid[1]):
            p = ref.mode0_parts(cells[rb, cb].tobytes())
            S, M, q4 = p["S"], p["M"], p["q4"]
            kas, nas = slice(cb * PB_KT, (cb + 1) * PB_KT), slice(rb * 32, (rb + 1) * 32)
            codes[kas, nas] = (q4.astype(np.int16) - 8).astype(np.int8)   # q4 is [k][n] already
            S_all[cb * 8:(cb + 1) * 8, nas] = S
            M_all[cb * 8:(cb + 1) * 8, nas] = M
            w_exact[nas, kas] = (q4 * S[np.arange(256) // G, :] + M[np.arange(256) // G, :]).T
    # One exponent per column, chosen so the LARGEST group's mantissa fits int16 (|m| <= 32767).  int16 has
    # 15 bits of dynamic range, so a column whose groups span more than that cannot hold every group: the
    # smallest ones round to 0 and their weights vanish.  That is measured and reported, not hidden -- for
    # these tensors those groups are already ~2^-15 of the column's largest, so the cost is in the noise of
    # the 4-bit grid itself (see rel_l1_error_gemv and the zeroed_groups count in the manifest).
    e_ref = (np.floor(np.log2(np.abs(S_all).max(axis=0) + 1e-30)) - 14).astype(np.int64)
    ms_f = S_all / np.exp2(e_ref)[None, :]
    ms = np.rint(ms_f).astype(np.int64)
    zeroed = int(np.count_nonzero(ms == 0) - np.count_nonzero(S_all == 0))
    if np.abs(ms).max() > 32767:
        raise SystemExit(f"|ms| up to {int(np.abs(ms).max())} does not fit int16; refusing")
    add = (M_all + 8.0 * S_all).astype(np.float32)
    return codes, ms.astype(np.int16), e_ref.astype(np.int16), add, w_exact, zeroed


class Pb16Matrix:
    """W @ a exactly as the device plus host compute it, so the head step can be checked end to end."""

    def __init__(self, codes, ms, e_ref, add):
        self.codes, self.ms, self.e_ref, self.add = codes, ms, e_ref, add
        self.N, self.K = codes.shape[1], codes.shape[0]

    def __matmul__(self, a):
        a = np.asarray(a, dtype=np.float32)
        y = np.zeros(self.N, dtype=np.float64)
        for g in range(self.K // G):
            ag = a[g * G:(g + 1) * G].astype(np.float64)
            y += (self.codes[g * G:(g + 1) * G, :].astype(np.float64).T @ ag) \
                * (self.ms[g, :].astype(np.float64) * np.exp2(self.e_ref.astype(np.float64)))
            y += ag.sum() * self.add[g, :].astype(np.float64)
        return y.astype(np.float32)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=os.path.expanduser("~/models/npu/bonsai2-27b-mtp-head/q4_k"))
    ap.add_argument("--out", default=os.path.expanduser("~/models/npu/bonsai2-27b-mtp-head/pb16i-exact"))
    ap.add_argument("--skip-head-step", action="store_true")
    a = ap.parse_args(argv)

    W, cfg = ref.load_q4k_head(a.head)          # exact float weights, per tensor (this is the reference)
    os.makedirs(a.out, exist_ok=True)
    man = {"source": a.head, "form": "pb16i-exact",
           "note": "source codes kept; the ADDITIVE per-group min travels as a host table "
                   "(add = M + 8*S), so the image is exact to the int16 mantissa instead of losing 11-18 % L1",
           "geometry": {"tile_bytes": PB16_TILE, "tile": "256 K x 32 N, [c][ki][nt][r] order "
                                                               "(layout_w4.py pack-pb16i)",
                        "group": G, "codes": "signed int4 (q4 - 8)", "magnitudes": "int16 per (32-K group, column)"},
           "tensors": []}
    rng = np.random.default_rng(0)
    probe = rng.standard_normal(1024).astype(np.float32)

    for name in ("fc", "q", "k", "v", "o", "gate", "up", "down"):
        n_out, k_tot = W[name].shape
        # the source cells for this tensor, straight out of q4_k
        t = next(x for x in cfg["tensors"] if x["name"] == {"fc": "model.mtp.fc.weight",
                                                            "q": "model.mtp.layers.0.self_attn.q_proj.weight",
                                                            "k": "model.mtp.layers.0.self_attn.k_proj.weight",
                                                            "v": "model.mtp.layers.0.self_attn.v_proj.weight",
                                                            "o": "model.mtp.layers.0.self_attn.o_proj.weight",
                                                            "gate": "model.mtp.layers.0.mlp.gate_proj.weight",
                                                            "up": "model.mtp.layers.0.mlp.up_proj.weight",
                                                            "down": "model.mtp.layers.0.mlp.down_proj.weight"}[name])
        idx = cfg["tensors"].index(t)
        blob = open(os.path.join(a.head, "model.q4nx"), "rb").read()
        import struct
        hlen = struct.unpack("<Q", blob[:8])[0]
        off = 8 + hlen + sum(x["bytes"] for x in cfg["tensors"][:idx])
        raw = blob[off:off + t["bytes"]]
        grid = [int(x) for x in t["shape"]]
        cells = np.frombuffer(raw, dtype=np.uint8).reshape(grid[0], grid[1], ref.Q4K_CELL)

        codes, ms, e_ref, add, w_exact, zeroed = exact_plan_from_cells(cells, grid, n_out, k_tot)
        img = pack_tiles(codes, ms)
        # self-check 1: the image round-trips to the same codes/magnitudes
        rc, rm = unpack_tiles(img, k_tot, n_out)
        rt_bad = int(np.count_nonzero(rc != codes) + np.count_nonzero(rm != ms))
        # self-check 2: full-matrix GEMV, device form (int) + host table, against the exact float weights
        a1 = rng.standard_normal(k_tot).astype(np.float32)
        y_int = Pb16Matrix(codes, ms, e_ref, add) @ a1
        y_true = w_exact.astype(np.float64) @ a1.astype(np.float64)
        rel = float(np.abs(y_int - y_true).sum() / (np.abs(y_true).sum() + 1e-30))

        base = os.path.join(a.out, name)
        img.astype(np.uint8).tofile(base + ".pb16i")
        add.astype("<f4").tofile(base + ".add")
        e_ref.astype("<i2").tofile(base + ".eref")
        man["tensors"].append({"tensor": name, "K": int(k_tot), "N": int(n_out),
                               "tiles": int(img.shape[0]), "image": os.path.basename(base) + ".pb16i",
                               "add": os.path.basename(base) + ".add",
                               "eref": os.path.basename(base) + ".eref",
                               "image_bytes": int(img.nbytes), "add_bytes": int(add.nbytes),
                               "roundtrip_mismatches": rt_bad,
                               "rel_l1_error_gemv": rel,
                               "e_ref_min": int(e_ref.min()), "e_ref_max": int(e_ref.max()),
                               "ms_absmax": int(np.abs(ms).max()), "add_absmax": float(np.abs(add).max()),
                               "zeroed_groups": zeroed})
        print(f"{name:5s} {k_tot:6d}x{n_out:<6d} tiles={img.shape[0]:6d} image={img.nbytes/1e6:7.2f} MB "
              f"add={add.nbytes/1e6:5.2f} MB  roundtrip={'OK' if rt_bad == 0 else f'{rt_bad} BAD'}  "
              f"rel-L1(gemv)={rel:.2e} zeroed={zeroed}")

    # self-check 3: the whole head step, source cells vs emitted images (should differ by the int16 mantissa only)
    if not a.skip_head_step:
        rng2 = np.random.default_rng(1)
        hid = rng2.standard_normal(ref.GEOM["H"]).astype(np.float32)
        emb = rng2.standard_normal(ref.GEOM["H"]).astype(np.float32)
        _, h_src, _ = ref.head_step(hid, emb, W, pos=0)
        Wimg = {}
        for e in man["tensors"]:
            img = np.fromfile(os.path.join(a.out, e["image"]), dtype=np.uint8)
            codes, ms = unpack_tiles(img, e["K"], e["N"])
            add = np.fromfile(os.path.join(a.out, e["add"]), dtype="<f4").reshape(e["K"] // G, e["N"])
            eref = np.fromfile(os.path.join(a.out, e["add"].replace(".add", ".eref")), dtype="<i2")
            Wimg[e["tensor"]] = Pb16Matrix(codes, ms, eref, add)
        for k in ("pre_fc_emb", "pre_fc_hid", "ln_in", "ln_post", "norm", "q_norm", "k_norm"):
            Wimg[k] = W[k]
        _, h_img, _ = ref.head_step(hid, emb, Wimg, pos=0)
        cos = float(np.dot(h_src, h_img) / (np.linalg.norm(h_src) * np.linalg.norm(h_img) + 1e-30))
        relh = float(np.abs(h_src - h_img).sum() / (np.abs(h_src).sum() + 1e-30))
        man["head_step_check"] = {"hidden_cosine": cos, "hidden_rel_l1": relh}
        print(f"head step: source vs emitted images -> hidden cosine={cos:.8f} rel-L1={relh:.2e}")

    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    worst = max(e["rel_l1_error_gemv"] for e in man["tensors"])
    print(f"\nemitted {len(man['tensors'])} pb16i-exact images + host tables into {a.out}; "
          f"worst per-tensor rel-L1(GEMV) = {worst:.2e}")
    return 0 if worst < 5e-3 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
