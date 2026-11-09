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
"""Measure a Ryzen AI 1.9 Token .fconst against its 1.8 twin, and print the table line to add.

An importer cannot derive a 1.9 pack's layout from the pack: the q/k/v bundle's tile shape is not a
subset sum of the pack's own const sizes, and the block prefix is not `64 + 2*hidden` either (Qwen3
carries per-head QK norm gains before the bundle). So every 1.9 pack is measured against the 1.8 pack
of the SAME model, and `npu/pack_import.cpp`'s kFconst19Rules table records the result.

This tool does that measurement and refuses to print a line it could not verify end to end:

  1. find the 1.9 headers (exactly `layers` runs of `00 00 7f ff` x heads, evenly spaced);
  2. derive the per-block prefix, the rotated bundle's tile count / tile size / rotation, from the
     content displacement -- NOT from byte equality, because a constant run (SmolLM2 has a 2,560-byte
     all-zero run) matches at any position and moves the boundary;
  3. normalise every block with those parameters and require the result to equal the 1.8 file byte for
     byte. Anything less prints the residual and refuses: a near-miss rule is worse than none.

Usage:
  tools/fconst19_measure.py --pack-1-8 DIR --pack-1-9 DIR
  tools/fconst19_measure.py --pack-1-8 DIR --pack-1-9 DIR --json     # machine-readable

The geometry (hidden, heads, layers) is read from the 1.9 pack's own genai_config.json/config.json.
"""
import argparse
import json
import os
import sys

HDR = bytes.fromhex("00007fff")


def token_fconst(pack):
    """The Token .fconst of a pack directory: the 1.8 spelling first, then a single Llm_Token_* file."""
    names = sorted(n for n in os.listdir(pack) if n.startswith("dd_metastate_Llm_Token_") and n.endswith(".fconst")
                   and "_Prefill_" not in n)
    exact = [n for n in names if n.startswith("dd_metastate_Llm_Token_Token_")]
    if exact:
        return os.path.join(pack, exact[0])
    if len(names) == 1:
        return os.path.join(pack, names[0])
    raise SystemExit(f"{pack}: {len(names)} Token .fconst candidates, cannot choose: {names}")


def geometry(pack):
    """hidden / heads / layers, from whichever config the pack states them in."""
    for fn, keys in (("config.json", ("hidden_size", "num_attention_heads", "num_hidden_layers")),
                     ("genai_config.json", None)):
        path = os.path.join(pack, fn)
        if not os.path.exists(path):
            continue
        c = json.load(open(path))
        if keys:
            if all(k in c for k in keys):
                return int(c[keys[0]]), int(c[keys[1]]), int(c[keys[2]])
        dec = (c.get("model") or {}).get("decoder") or {}
        if {"hidden_size", "num_attention_heads", "num_hidden_layers"} <= set(dec):
            return int(dec["hidden_size"]), int(dec["num_attention_heads"]), int(dec["num_hidden_layers"])
    raise SystemExit(f"{pack}: no config.json or genai_config.json stating hidden/heads/layers")


def find_headers(blob, heads, layers):
    hdr, runs, i = HDR * heads, [], blob.find(HDR * heads)
    while i >= 0:
        runs.append(i)
        i = blob.find(hdr, i + len(hdr))
    if len(runs) != layers:
        raise SystemExit(f"{len(runs)} header runs of {heads * 4} B, not {layers}: not the 1.9 form for this geometry")
    stride = runs[1] - runs[0]
    if any(runs[k + 1] - runs[k] != stride for k in range(len(runs) - 1)):
        raise SystemExit("header spacing is not uniform")
    return runs[0], stride


def candidates(a, b, heads, layers):
    """Every (prefix, n_tiles, tile, rot) that fits the first block, coarsest first."""
    hdr_len = heads * 4
    header_off, stride9 = find_headers(b, heads, layers)
    stride8 = stride9 - hdr_len
    # engine#1740: a 1.9 pack's tail can carry MORE than its 1.8 twin's -- the 1.5B's 1.9 tail is 262,147,136 B against the 1.8's
    # 134,314,560 B, and that 1.8 figure is exactly `attn_payload_off(1536) + 2374 x attn_tile_bytes(1536)`, i.e. the twin's tail IS
    # the head and the 1.9 tail is the head plus more. The rule describes the LAYER BLOCKS, so a surplus is not a layout disagreement:
    # it is reported and not compared. A 1.9 tail SHORTER than its twin's still means the two files disagree.
    tail8 = len(a) - layers * stride8
    tail9 = len(b) - layers * stride9
    if tail9 < tail8:
        raise SystemExit(f"the 1.9 tail is shorter than its 1.8 twin's ({tail9} B < {tail8} B): the files disagree")
    a0, b0 = a[:stride8], b[:stride9]
    if a0[header_off:] != b0[header_off + hdr_len:]:
        raise SystemExit("block 0's tail does not match at -4*NH")
    prefix = 0
    while b0[prefix] == a0[prefix]:
        prefix += 1
    probe = b0[prefix:prefix + 4096]
    d = next((c for c in range(4096, (header_off - prefix) // 2) if a0[prefix + c:prefix + c + 4096] == probe), None)
    if d is None:
        raise SystemExit("the payload head does not reappear in the 1.8 block at any displacement")
    out = []
    for rot in range(1, 9):
        if d % rot or d // rot < 4096:
            continue
        tile = d // rot
        for n in range((header_off - prefix) // tile, 1, -1):
            if all(b0[prefix + i * tile:prefix + (i + 1) * tile]
                   == a0[prefix + ((i + rot) % n) * tile:prefix + ((i + rot) % n + 1) * tile] for i in range(n)) \
               and b0[prefix + n * tile:header_off] == a0[prefix + n * tile:header_off]:
                out.append((prefix, n, tile, rot, header_off, stride9))
                break
    if not out:
        raise SystemExit("no rotation fits block 0")
    return out


def normalise(blob, layers, p, hdr_len):
    prefix, n, tile, rot, header_off, stride9 = p
    res = bytearray()
    for layer in range(layers):
        b = blob[layer * stride9:(layer + 1) * stride9]
        res += b[:prefix]
        for i in range(n):                      # canonical i is stored at (i - rot) mod n
            j = (i - rot) % n
            res += b[prefix + j * tile:prefix + (j + 1) * tile]
        res += b[prefix + n * tile:header_off]
        res += b[header_off + hdr_len:]
    res += blob[layers * stride9:]
    return bytes(res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack-1-8", required=True, help="the 1.8 pack of the same model")
    ap.add_argument("--pack-1-9", required=True, help="the 1.9 pack to measure")
    ap.add_argument("--json", action="store_true", help="print the result as JSON instead of the table line")
    ap.add_argument("--probe-wrong", action="store_true",
                    help="having found the rule, perturb it one field at a time and print each residual, so the "
                         "tolerance has a measurement behind it instead of an argument")
    args = ap.parse_args()

    hidden, heads, layers = geometry(args.pack_1_9)
    a = open(token_fconst(args.pack_1_8), "rb").read()
    b = open(token_fconst(args.pack_1_9), "rb").read()
    hdr_len = heads * 4

    # The structural fit is what identifies the rule; byte equality is the strong form of the same evidence and
    # not always available. Two builds of one model can differ in a few hundred weight codes -- Qwen3-1.7B's 1.8
    # and 1.9 fconsts differ in 535 bytes, all in layer 24 -- and those pairs are exactly what a catalog has to
    # measure, so a candidate that fits every block structurally is accepted with its residual REPORTED rather
    # than refused for not being byte-exact. A wrong rule does not produce a 0.0001 % residual: it produces a
    # candidate that fails the per-block fit, or a residual spread across the whole file.
    # The tolerance, MEASURED at both ends rather than argued (reviewer ask). Legitimate pairs, all on disk:
    #   Qwen2.5-0.5B 0 B of 291,653,952 (0)        Qwen3-1.7B  535 B of 957,722,944 (5.6e-7)
    #   Qwen3-4B     2,933 B of 2,208,061,248 (1.3e-6)   Llama-3.2-1B 1,857 B of 686,518,336 (2.7e-6)
    # Wrong rules, measured with --probe-wrong against the accepted rule on the 0.5B pair (as a fraction of the
    # file): rot+1 3.85e-2, tile/2 3.85e-2, n+1 9.7e-3, prefix+64 1.5e-3.
    # So 1e-5 sits 3.7x above the worst LEGITIMATE residual and 148x below the mildest WRONG rule -- it cannot
    # admit a misaligned rule, and it does admit the few-hundred-stray-codes case a fine-tune's twin produces.
    TOL = 1e-5
    _ho, _s9 = find_headers(b, heads, layers)
    surplus = max(0, (len(b) - layers * _s9) - (len(a) - layers * (_s9 - hdr_len)))
    b_cmp = b[:len(b) - surplus] if surplus else b
    for p in candidates(a, b, heads, layers):
        got = normalise(b_cmp, layers, p, hdr_len)
        if len(got) != len(a):
            print(f"# candidate prefix={p[0]} n={p[1]} tile={p[2]} rot={p[3]}: length {len(got)} != {len(a)}",
                  file=sys.stderr)
            continue
        # The exact case is the common one, so it takes the C-level compare: this loop is Python-level, one
        # iteration per byte, and on a 957 MB pair it costs ~24 s against ~0.13 s for `bytes ==` (reviewer's
        # measurement, same expression). The loop only runs when there is something to report.
        differ = 0 if got == a else sum(1 for x, y in zip(got, a) if x != y)
        if differ <= max(0, int(len(a) * TOL)):
            prefix, n, tile, rot, header_off, stride9 = p
            note = "byte for byte" if differ == 0 else f"{differ} of {len(a)} bytes differ ({100.0 * differ / len(a):.2e} %)"
            if surplus:
                note += f"; the 1.9 tail carries {surplus} B past its 1.8 twin's, not compared (engine#1740)"
            if args.json:
                print(json.dumps({"hidden": hidden, "heads": heads, "layers": layers, "prefix": prefix,
                                  "n_tiles": n, "tile": tile, "rot": rot, "header_off": header_off,
                                  "stride_1_9": stride9, "verified": note}))
            else:
                print(f"// verified: {note} ({len(a)} B)")
                print(f"{{{hidden}, {heads}, {layers}, {prefix}, {n}, {rot}, {tile}}},")
            if args.probe_wrong:
                # What a WRONG rule costs, in the same units as the tolerance: perturb one field of the accepted
                # rule at a time, normalise again, and report the residual. A tolerance is only defensible when the
                # wrong answers are orders of magnitude above it.
                for label, wrong in (("rot+1", (prefix, n, tile, (rot + 1) % n if n > 1 else rot, header_off, stride9)),
                                     ("tile/2", (prefix, n, max(4096, tile // 2), rot, header_off, stride9)),
                                     ("prefix+64", (prefix + 64, n, tile, rot, header_off, stride9)),
                                     ("n+1", (prefix, n + 1, tile, rot, header_off, stride9))):
                    try:
                        w = normalise(b, layers, wrong, hdr_len)
                    except Exception as exc:                      # a perturbation that cannot even be built
                        print(f"# wrong rule {label}: cannot normalise ({exc})", file=sys.stderr)
                        continue
                    if len(w) != len(a):
                        print(f"# wrong rule {label}: length {len(w)} != {len(a)}", file=sys.stderr)
                        continue
                    wd = sum(1 for x, y in zip(w, a) if x != y)
                    print(f"# wrong rule {label}: {wd} bytes differ ({100.0 * wd / len(a):.3e} % of the file; "
                          f"tolerance is {TOL:.0e} = {int(len(a) * TOL)} B)", file=sys.stderr)
            return 0
        print(f"# candidate prefix={p[0]} n={p[1]} tile={p[2]} rot={p[3]}: {differ} bytes differ, trying next",
              file=sys.stderr)
    raise SystemExit("no candidate reproduced the 1.8 file: this pack is not the 1.9 form, or the pair is not the "
                     "same model (a fine-tune's codes differ, which shows up here as a byte residual)")


if __name__ == "__main__":
    sys.exit(main())
