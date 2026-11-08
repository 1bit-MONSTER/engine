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
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
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
    if len(a) - layers * stride8 != len(b) - layers * stride9:
        raise SystemExit("the two files' tails differ in length")
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
    args = ap.parse_args()

    hidden, heads, layers = geometry(args.pack_1_9)
    a = open(token_fconst(args.pack_1_8), "rb").read()
    b = open(token_fconst(args.pack_1_9), "rb").read()
    hdr_len = heads * 4

    for p in candidates(a, b, heads, layers):
        got = normalise(b, layers, p, hdr_len)
        if got == a:
            prefix, n, tile, rot, header_off, stride9 = p
            if args.json:
                print(json.dumps({"hidden": hidden, "heads": heads, "layers": layers, "prefix": prefix,
                                  "n_tiles": n, "tile": tile, "rot": rot, "header_off": header_off,
                                  "stride_1_9": stride9, "verified": "whole file, byte for byte"}))
            else:
                print(f"// verified: the whole file, byte for byte ({len(a)} B)")
                print(f"{{{hidden}, {heads}, {layers}, {prefix}, {n}, {rot}, {tile}}},")
            return 0
        differ = sum(1 for x, y in zip(got, a) if x != y)
        print(f"# candidate prefix={prefix} n={n} tile={tile} rot={rot}: {differ} bytes differ, trying next",
              file=sys.stderr)
    raise SystemExit("no candidate reproduced the 1.8 file: this pack is not the 1.9 form, or the pair is not the "
                     "same model (a fine-tune's codes differ, which shows up here as a byte residual)")


if __name__ == "__main__":
    sys.exit(main())
