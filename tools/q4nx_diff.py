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
"""Compare two Q4NX containers, and decode either tile layout against a GGUF.

A Q4NX `model.q4nx` is a safetensors-style container: u64 header length, a JSON
table of {tensor: {dtype, shape, data_offsets}} (+ an optional `__metadata__`),
then the data.  Two converters can produce the same container *format* and the
same tensor geometry and still disagree in the payload, so this tool separates
three questions that a byte comparison alone cannot:

  compare  which tensors are byte-identical, and for the quantized ones whether it
           is the scale plane, the min plane or the codes that differ.  A
           converter that keeps Q4_0 as "scale d, min -8d, unsigned codes" and one
           that keeps it as "scale d, min 0, signed codes" are both exact against
           the GGUF and share not one min byte -- the planes say which.
  decode   decode a container under a named layout and compare against the GGUF
           ground truth, so "the files differ" becomes "each is exact under its
           own decoder, and neither reads the other's".

Layouts (both are 32x256 tiles of 5120 B: 512 scale, 512 min, 4096 codes):

  ours  codes are [row-half(2)][group(8)][col-in-group(32)][row-pair(8)], the
        16-row "parallel" split the NPU kernel walks (npu/q4nx.cpp); nibbles are
        unsigned and the Q4_0 zero-point lives in the min plane as -8d.
  flm   codes are plain column major [col(256)][row-pair(16)]; the official
        gguf->Q4NX converter keeps Q4_0 as min 0 with *signed* nibbles (q-8).

Which layout is validated against what (2026-11-05, ZR1-1.5B Q4_K_M, no device; two tensors,
`ours`/`flm(signed)`/`flm(unsigned)`):

  ours  both converters we can run agree with it -- the FLM_Q4NX_Converter in third_party/
        (npu-rt-753-pr) decodes at requant scale (max|err| 2.96e-02 / 2.31e-02) and so does
        our own pack.  This is the layout to gate on: it is what npu/forward reads.
  flm   NOT validated against any container here.  amd-oss/FLM_Q4NX_Converter @d1d5232
        ("update converter for q4_k") writes something that matches neither layout
        consistently: 5.48e+00 under `ours`, 3.46e+00 under `flm(signed)`, and its Q6_K-sourced
        tensors reach requant scale only under `flm(unsigned)` (2.31e-02) while its Q4_K-sourced
        ones match none of the three.  So `--layout flm` is the decoder for the *pre-q4_k*
        converter, and a BAD row from it means "this file is not that layout", not "this file is
        wrong" (engine#1110).  Deliberately not "fixed" blind: nothing on this box could tell a
        fix from a guess.

usage:
  tools/q4nx_diff.py compare A/model.q4nx B/model.q4nx [--summary]
  tools/q4nx_diff.py decode  --gguf model.gguf --q4nx DIR --layout ours|flm [-v]

Needs numpy and gguf (gguf-py).
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

ROWS, COLS, GROUP, TILE = 32, 256, 32, 5120
PLANE_SCALE, PLANE_MIN, PLANE_CODES = 512, 512, 4096


def load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    return hdr, 8 + n, np.memmap(path, dtype=np.uint8, mode="r")


def tensor_bytes(hdr, base, mm, name):
    lo, hi = hdr[name]["data_offsets"]
    return np.asarray(mm[base + lo:base + hi])


def bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def tiles_of(raw, rows, cols):
    """Split a tile tensor into (ntiles, 5120); `rows`/`cols` are the weight's."""
    if rows % ROWS or cols % COLS:
        raise ValueError("not a whole tile grid: %dx%d" % (rows, cols))
    return np.asarray(raw).reshape(-1, TILE)


def decode_ours(t, rows, cols):
    RT, KT = rows // ROWS, cols // COLS
    t = t.reshape(RT, KT, TILE)
    sc = bf16(t[..., :PLANE_SCALE].copy().view(np.uint16)).reshape(RT, KT, 8, ROWS)
    zr = bf16(t[..., PLANE_SCALE:PLANE_SCALE + PLANE_MIN].copy().view(np.uint16)
              ).reshape(RT, KT, 8, ROWS)
    p = t[..., PLANE_SCALE + PLANE_MIN:].reshape(RT, KT, 2, 8, COLS // 8, 8)
    c = np.stack([p & 0x0F, p >> 4], axis=-1)                  # [..., r2, parity]
    c = c.transpose(0, 2, 5, 6, 1, 3, 4).reshape(RT, ROWS, KT, 8, COLS // 8)
    s = sc.transpose(0, 3, 1, 2)[..., None]
    z = zr.transpose(0, 3, 1, 2)[..., None]
    return (c * s + z).reshape(rows, cols)


def decode_flm(t, rows, cols, signed=True):
    RT, KT = rows // ROWS, cols // COLS
    t = t.reshape(RT, KT, TILE)
    sc = bf16(t[..., :PLANE_SCALE].copy().view(np.uint16)).reshape(RT, KT, 8, ROWS)
    mn = bf16(t[..., PLANE_SCALE:PLANE_SCALE + PLANE_MIN].copy().view(np.uint16)
              ).reshape(RT, KT, 8, ROWS)
    p = t[..., PLANE_SCALE + PLANE_MIN:].reshape(RT, KT, COLS, ROWS // 2)
    lo = (p & 0x0F).astype(np.int8)
    hi = ((p >> 4) & 0x0F).astype(np.int8)
    if signed:                      # the official Q4_0 path stores q-8 as a signed nibble
        lo = np.where(lo > 7, lo - 16, lo)
        hi = np.where(hi > 7, hi - 16, hi)
    c = np.stack([lo, hi], axis=-1).reshape(RT, KT, COLS, ROWS)
    g = np.arange(COLS) // GROUP
    return (sc[:, :, g, :] * c + mn[:, :, g, :]).transpose(0, 3, 1, 2).reshape(rows, cols)


DECODERS = {"ours": decode_ours, "flm": decode_flm}

# A converter may pad a weight to a WIDER tile grid than the model's own dimensions (the FastFlowLM
# repack pads intermediate_size up to its 512-column grid: IM 8960 becomes 9216).  The container then
# holds more 5120-B tiles than the model's geometry implies, and a decoder that reshapes on the model's
# grid cannot read it at all -- which is why every padded MLP tensor used to come back "cannot reshape".
# npu/forward derives the grid the same way (engine#1097/#1107, `q4nx_packed_grid()`); this is that rule
# for the checker.  It is deliberately the same logic in two places: the tool has to be able to check a
# container without the engine, and tests/q4nx_diff_grid_test.py pins the same geometries and the same
# tie refusal as npu/forward's ctest `npu_padded_grid`, so the two cannot drift apart silently.
Q4NX_MAX_EXCESS = 64      # tiles: the fastest converter pads one axis by one step (IM 8960 -> +1 tile)


def packed_grid(nbytes, rows, cols):
    """(rows_packed, cols_packed) for a container padded wider than the model, or None.

    Only the 5120-B q4_1 tile is considered: the decoders here unpack that layout and nothing else, so a
    Q4K/Q8 chunk is "not this tool's container" rather than a padded grid (npu/forward's rule tries all
    three chunk kinds because the forward decodes all three)."""
    if rows % ROWS or cols % COLS or nbytes % TILE:
        return None
    ntr, ntc = rows // ROWS, cols // COLS
    np_ = nbytes // TILE
    best = None          # (excess, rows_p, cols_p)
    ties = 0
    d = 1
    while d * d <= np_:
        if np_ % d == 0:
            for rpt, cpt in ((d, np_ // d), (np_ // d, d)):
                if rpt < ntr or cpt < ntc:
                    continue
                excess = (rpt - ntr) + (cpt - ntc)
                rp, cp = rpt * ROWS, cpt * COLS
                if best is None or excess < best[0]:
                    best, ties = (excess, rp, cp), 0
                elif excess == best[0] and (rp, cp) != (best[1], best[2]):
                    ties += 1
        d += 1
    if best is None or ties:
        return None      # no grid fits, or the answer would be a guess
    if best[0] > Q4NX_MAX_EXCESS or (best[1], best[2]) == (rows, cols):
        return None      # not a padding this tool will trust, or nothing was padded
    return best[1], best[2]


def compare_cmd(args):
    ha, basea, ma = load(args.a)
    hb, baseb, mb = load(args.b)
    print("A header %d B, data at %d | B header %d B, data at %d"
          % (basea - 8, basea, baseb - 8, baseb))
    meta = [(ha.pop("__metadata__", None), "A"), (hb.pop("__metadata__", None), "B")]
    for m, tag in meta:
        print("  __metadata__ %s: %s" % (tag, "absent" if m is None else json.dumps(m)[:120]))
    ka, kb = set(ha), set(hb)
    print("tensors: A=%d B=%d; only A=%d only B=%d" % (len(ha), len(hb), len(ka - kb), len(kb - ka)))
    for k in sorted(ka - kb):
        print("   only A:", k)
    for k in sorted(kb - ka):
        print("   only B:", k)
    common = sorted(ka & kb)
    dt = [(k, ha[k]["dtype"], hb[k]["dtype"]) for k in common if ha[k]["dtype"] != hb[k]["dtype"]]
    sh = [(k, ha[k]["shape"], hb[k]["shape"]) for k in common if ha[k]["shape"] != hb[k]["shape"]]
    print("dtype diffs: %d | shape diffs: %d" % (len(dt), len(sh)))
    for x in dt[:8]:
        print("   ", x)
    for x in sh[:8]:
        print("   ", x)

    same, diff, planes = [], [], {}
    for k in common:
        da, db = ha[k], hb[k]
        if da.get("shape") != db.get("shape"):
            diff.append(k)
            continue
        a = tensor_bytes(ha, basea, ma, k)
        b = tensor_bytes(hb, baseb, mb, k)
        if a.tobytes() == b.tobytes():
            same.append(k)
            continue
        diff.append(k)
        if da["dtype"] == "I8" and a.size and a.size % TILE == 0 and b.size == a.size:
            ta = a.reshape(-1, TILE)
            tb = b.reshape(-1, TILE)
            for lo, hi, label in ((0, PLANE_SCALE, "scale"),
                                  (PLANE_SCALE, PLANE_SCALE + PLANE_MIN, "min"),
                                  (PLANE_SCALE + PLANE_MIN, TILE, "codes")):
                eq = int((ta[:, lo:hi] == tb[:, lo:hi]).all(axis=1).sum())
                tot = ta.shape[0]
                acc = planes.setdefault(label, [0, 0])
                acc[0] += eq
                acc[1] += tot
    print("\nper-tensor bytes: identical=%d differing=%d" % (len(same), len(diff)))
    for label in ("scale", "min", "codes"):
        if label in planes:
            eq, tot = planes[label]
            print("  plane %-5s identical tiles: %d/%d" % (label, eq, tot))
    if not args.summary:
        for k in diff[:20]:
            print("   differs:", k)
        if len(diff) > 20:
            print("   ... %d more" % (len(diff) - 20))
    return 0


# The worst per-element error a CORRECT container can show: a q4_1 container packed from a Q4_K_M GGUF
# is a requantization, so exact equality is impossible and "no error anywhere" would call every honest
# container BAD. Measured on a Qwen3-0.6B pack (Q4_K_M -> q4_1): worst max|err| 4.1e-2. A container whose
# tiles belong to different weights lands an order of magnitude above it (measured 1.1-2.0).
LAYOUT_MATCH_MAX_ERR = 0.1
LAYOUT_MATCH_RATIO = 3.0     # ... and it must beat the other layout by this much


def verdict(results):
    """(rc, line) for {layout: (exact, inexact, worst_max_err)}.

    A container is laid out ONE way, and a caller that guesses wrong gets a confident "BAD" for a good
    container. So name the layout that matched, and when none did, name the closest and say so instead
    of implying the container is broken.
    """
    # A layout that compared NO tensor is not evidence of anything: `_decode_pass` returns
    # (0, 0, 0.0) when every tensor was skipped, and worst=0.0 with rival=0.0 satisfies
    # "rival >= worst * RATIO", so such a layout used to be reported as a MATCH on whichever one
    # sorted first -- measured: a 0.6B container against a 4B GGUF came back
    # `matches layout flm (worst max|err| 0.000e+00, ...)` with 0 tensors compared (engine#1110
    # review).  Drop the layouts with no evidence before choosing, and if none is left say so:
    # every caller already has a branch for rc=2.
    results = {k: v for k, v in results.items() if v[0] + v[1] > 0}
    if not results:
        return 2, "nothing was compared"
    order = sorted(results, key=lambda k: results[k][2])
    best = order[0]
    worst = results[best][2]
    rival = results[order[1]][2] if len(order) > 1 else None
    exact = results[best][0]
    took = (exact > 0 and worst == 0.0) or (
        worst <= LAYOUT_MATCH_MAX_ERR and (rival is None or rival >= worst * LAYOUT_MATCH_RATIO))
    if took:
        other = "" if rival is None else ", next layout %s worst max|err| %.3e" % (order[1], rival)
        how = "%d tensor(s) exactly" % exact if worst == 0.0 and exact else "worst max|err| %.3e" % worst
        return 0, "matches layout %s (%s%s)" % (best, how, other)
    return 1, ("matches NO layout: closest is %s with worst max|err| %.3e (> %.3e, or not %.0fx better "
               "than %s) - the container's tiles are not these weights"
               % (best, worst, LAYOUT_MATCH_MAX_ERR, LAYOUT_MATCH_RATIO, order[1] if rival is not None else "-"))


def _decode_pass(args, layout, quiet=False):
    """One layout over every comparable tensor. Returns (exact, inexact, worst_max_err)."""
    import gguf
    from gguf.quants import dequantize

    hdr, base, mm = load(os.path.join(args.q4nx, "model.q4nx"))
    dec = DECODERS[layout]
    r = gguf.GGUFReader(args.gguf)
    want = [t for t in r.tensors if t.name.endswith(".weight") or t.name in ("output.weight",)]
    if args.tensor:
        want = [t for t in want if args.tensor in t.name]
    if not want:
        print("no tensors matched", file=sys.stderr)
        return 0, 0, 0.0
    exact = inexact = skipped = padded_n = 0
    worst = (0.0, "")
    # our converter's names, so a GGUF tensor can be looked up in the container
    lay = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
           "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm",
           "attn_k_norm": "self_attn.k_norm", "ffn_gate": "mlp.gate_proj",
           "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj", "attn_norm": "input_layernorm",
           "ffn_norm": "post_attention_layernorm", "token_embd": "model.embed_tokens",
           "output_norm": "model.norm", "output": "lm_head"}

    def q4nx_name(n):
        head, _, field = n.rpartition(".")
        if head.startswith("blk."):
            idx = head.split(".")[1]
            return "model.layers.%s.%s.weight" % (idx, lay[head.split(".", 2)[2]])
        return lay[head] + "." + field

    for t in want:
        if len(t.shape) < 2:            # 1-D norms: nothing tiled to decode
            skipped += 1
            continue
        name = q4nx_name(t.name)
        if name not in hdr:
            skipped += 1
            continue
        cols, rows = int(t.shape[0]), int(t.shape[1])
        d = hdr[name]
        if d["dtype"] != "I8":
            skipped += 1
            continue
        raw = tensor_bytes(hdr, base, mm, name)
        grid = packed_grid(int(raw.size), rows, cols)
        rows_p, cols_p = grid if grid else (rows, cols)
        try:
            got = dec(raw, rows_p, cols_p).astype(np.float64)
        except ValueError as e:
            if not quiet:
                print("skip %-46s %s" % (name, e))
            skipped += 1
            continue
        padded = 0
        if grid:                      # the container's tiles are wider; only the model's slice is the weight
            got = got[:rows, :cols]
            padded = 1
            padded_n += 1
        ref = dequantize(t.data, t.tensor_type).astype(np.float64).reshape(rows, cols)
        err = np.abs(got - ref)
        # A tile whose scale/zero planes decode to inf or NaN (a corrupt container) must read as a
        # MISMATCH, never as a 0.000e+00 error.  Note which value leaked: NaN is *truthy*, so
        # `err.any()` already counted the tensor inexact -- the error was that `mx` was NaN and
        # `nan > worst[0]` is False, so `worst` stayed 0.0 and verdict()'s `rival >= worst * RATIO`
        # tie named the layout a match.  Measured on random tile bytes
        # (tests/tile_check_no_shared_tensor_test.py): exact=0 inexact=1 worst=0.000e+00, "match".
        finite = np.isfinite(err)
        mx = float(err[finite].max()) if finite.any() else 0.0
        if not finite.all():
            mx = float("inf")
        ok = bool(finite.all()) and not err.any()
        exact += ok
        inexact += (not ok)
        if mx > worst[0]:
            worst = (mx, t.name)
        if not quiet and (args.verbose or not ok or padded):
            print("%s %-46s %-5s max|err| %.3e  rms %.3e%s"
                  % ("ok  " if ok else "BAD ", t.name, t.tensor_type.name, mx,
                     float(np.sqrt((err ** 2).mean())),
                     "  [padded %dx%d read as %dx%d]" % (rows, cols, rows_p, cols_p) if padded else ""))
    if not quiet:
        print("layout=%s: exact=%d inexact=%d skipped=%d padded=%d   worst max|err| %.3e (%s)"
              % (layout, exact, inexact, skipped, padded_n, worst[0], worst[1] or "-"))
    return exact, inexact, worst[0]


def decode_cmd(args):
    layouts = sorted(DECODERS) if args.layout == "both" else [args.layout]
    results = {}
    for layout in layouts:
        results[layout] = _decode_pass(args, layout, quiet=(len(layouts) > 1 and layout != layouts[-1]))
    rc, line = verdict(results)
    print(line)
    return rc


def self_test():
    """The verdict logic, with the cases measured on real containers plus the 0-tensor tie."""
    bad = []
    cases = [
        # what a correct container looks like: one layout exact (a round trip), the other noise
        ({"ours": (12, 0, 0.0), "flm": (0, 12, 1.6)}, 0, "ours", "an exact round trip picks its own layout"),
        # a correct requantized container: 4.1e-2 is the honest Q4_K_M -> q4_1 error, 1.6 is the other
        ({"ours": (0, 12, 4.1e-2), "flm": (0, 12, 1.6)}, 0, "ours", "a lossy but correct pack still matches"),
        ({"ours": (0, 12, 4.1e-2)}, 0, "ours", "a single layout under the tolerance matches"),
        # a wrong converter: both layouts far above the tolerance, so the container is named as wrong
        ({"ours": (0, 12, 2.0), "flm": (0, 12, 1.1)}, 1, "NO", "tiles from other weights match nothing"),
        # ambiguous: close to the tolerance and not clearly better than its rival
        ({"ours": (0, 12, 0.09), "flm": (0, 12, 0.12)}, 1, "NO", "not clearly better than the rival"),
        ({}, 2, "nothing", "nothing compared"),
        # both layouts skipped every tensor: a 0-vs-0 tie is not a match (engine#1110 review)
        ({"ours": (0, 0, 0.0), "flm": (0, 0, 0.0)}, 2, "nothing", "two layouts that compared no tensor"),
        # one layout compared tensors and the other did not: the one with evidence decides
        ({"ours": (0, 0, 0.0), "flm": (0, 12, 1.6)}, 1, "NO", "a layout with no evidence cannot win"),
    ]
    for results, want_rc, want_word, what in cases:
        rc, line = verdict(results)
        if rc != want_rc or want_word not in line:
            bad.append("%s: got rc=%d %r" % (what, rc, line))
    if bad:
        print("self-test FAILED:")
        for b in bad:
            print("  " + b)
        return 1
    print("self-test passed: the verdict names the matching layout, accepts a lossy-but-correct pack, "
          "refuses tiles from other weights, and treats a layout that compared nothing as no evidence")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true", help="the layout verdict, on synthetic results")
    sub = ap.add_subparsers(dest="cmd", required=False)
    c = sub.add_parser("compare", help="compare two model.q4nx containers")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--summary", action="store_true", help="counts only")
    c.set_defaults(func=compare_cmd)
    d = sub.add_parser("decode", help="decode a container and check it against its GGUF")
    d.add_argument("--gguf", required=True)
    d.add_argument("--q4nx", required=True, help="the model directory")
    d.add_argument("--layout", required=True, choices=sorted(DECODERS) + ["both"],
                   help="which tile layout the container uses; 'both' tries each and names the one that "
                        "matches (a wrong guess reports a good container as BAD, which is how the "
                        "converter mismatch in engine#1110 hid)")
    d.add_argument("--tensor", help="only tensors whose GGUF name contains this")
    d.add_argument("-v", "--verbose", action="store_true")
    d.set_defaults(func=decode_cmd)
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
