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
"""Report a FastFlowLM NPU2 model source's container without loading a device (engine#1786, item F1).

The NPU2 model sources on the Hub are a safetensors container named `model.q4nx`: 8-byte little-endian
header length, then that many bytes of JSON mapping tensor name -> {dtype, shape, data_offsets}, then
the payload. The tensor NAMES are the model's own HF names, so a report from this script says exactly
what the importer has to accept. No device, no model weights are read: only the header.

  tools/flm/q4nx_container_report.py <dir-or-model.q4nx> [...]
"""
import json
import os
import struct
import sys
from collections import Counter


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 64 << 20:
            raise ValueError(f"{path}: implausible header length {n}")
        hdr = json.loads(f.read(n))
    meta = hdr.pop("__metadata__", {}) or {}
    return hdr, meta, n


def report(path):
    hdr, meta, n = read_header(path)
    dtypes = Counter(v.get("dtype", "?") for v in hdr.values())
    payload = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in hdr.values())
    layers = {k.split(".layers.")[1].split(".")[0] for k in hdr if ".layers." in k}
    print(f"{os.path.basename(path)}: header {n} B, {len(hdr)} tensors, payload {payload/1e9:.3f} GB")
    print(f"  dtypes {dict(dtypes)}")
    print(f"  layers seen: {len(layers)}  metadata keys: {sorted(meta)}")
    if meta:
        for k, v in list(meta.items())[:8]:
            print(f"    {k} = {v}")
    # The first layer's tensors, because they are the set the importer must map per layer.
    for k in sorted(hdr):
        if ".layers.0." in k:
            v = hdr[k]
            print(f"    {k:52} {v['dtype']:5} {v['shape']}")
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    rc = 0
    for arg in argv[1:]:
        path = os.path.join(arg, "model.q4nx") if os.path.isdir(arg) else arg
        if not os.path.isfile(path):
            print(f"{arg}: no model.q4nx", file=sys.stderr)
            rc = 1
            continue
        rc |= report(path)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
