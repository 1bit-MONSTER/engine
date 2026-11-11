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
"""hf_tensor.py -- read ONE tensor out of a safetensors shard, over HTTP range requests.

Written because I got a safetensors offset wrong four times in one night and each time concluded something
false about a pack. The rule that bit me, and the only reason this file exists:

    a safetensors `data_offsets` entry is relative to the START OF THE DATA SECTION, i.e. `8 + header_len`
    bytes into the file -- not to byte 8.

Fetch `bytes=8+off` and you get a slice of whatever tensor is 11,656 B (this shard's header) away from the one
you asked for, which still *looks* like a tensor: it has a plausible size, it is finite, and a correlation
against it is a correlation against noise. Two of those cost me a retraction on a merged PR.

So this tool does the arithmetic once, prints the tensor's own statistics so a wrong read is visible rather
than plausible (a norm gain is ~1.0, not +-0.03), and can write the raw bytes for a comparison script.

usage:
  tools/hf_tensor.py --repo tencent/Hunyuan-7B-Instruct \
      --tensor model.layers.0.self_attn.query_layernorm.weight [--out q.bin] [--json]
  tools/hf_tensor.py --self-test

Exit: 0 ok, 1 the tensor or shard was not found, 2 cannot tell (no network / not a safetensors repo).
"""

import argparse
import json
import pathlib
import struct
import sys
import urllib.request

UA = {"User-Agent": "1bit-aba/hf_tensor"}  # HF rejects the default python urllib agent on some endpoints
DTYPE_TABLE = {
    "F64": ("<d", 8), "F32": ("<f", 4), "F16": ("<e", 2), "BF16": (None, 2),
    "I64": ("<q", 8), "I32": ("<i", 4), "I16": ("<h", 2), "I8": ("<b", 1),
    "U8": ("<B", 1), "BOOL": ("<?", 1),
}


def http_range(url, start, end_inclusive):
    """Inclusive byte range, as safetensors' data_offsets are written [start, end)."""
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=%d-%d" % (start, end_inclusive)})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def api_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def data_section_start(first8):
    """The byte the offsets are measured from: 8 + header_len. The whole point of this file."""
    (hlen,) = struct.unpack("<Q", first8)
    return 8 + hlen


def values(raw, dtype):
    """Decode a tensor's bytes to floats, for the statistics only (the raw bytes are what a caller compares)."""
    if dtype == "BF16":
        out = []
        for i in range(0, len(raw) - 1, 2):
            (u,) = struct.unpack("<H", raw[i:i + 2])
            (v,) = struct.unpack("<f", struct.pack("<I", u << 16))
            out.append(v)
        return out
    fmt, size = DTYPE_TABLE.get(dtype, (None, None))
    if fmt is None:
        return None
    n = len(raw) // size
    return list(struct.unpack("<%d%s" % (n, fmt[1]), raw[: n * size]))


def read_tensor(repo, tensor, revision="main", base="https://huggingface.co"):
    """(dtype, shape, raw_bytes, shard) for one tensor, reading only its header and its own bytes."""
    idx = api_json("%s/%s/resolve/%s/model.safetensors.index.json" % (base, repo, revision))
    shard = idx["weight_map"].get(tensor)
    if shard is None:
        raise KeyError("%s is not in %s's index (%d tensors)" % (tensor, repo, len(idx["weight_map"])))
    url = "%s/%s/resolve/%s/%s" % (base, repo, revision, shard)
    first8 = http_range(url, 0, 7)
    if len(first8) != 8:
        raise ValueError("%s: a range request returned %d bytes, not 8" % (url, len(first8)))
    dstart = data_section_start(first8)
    header = json.loads(http_range(url, 8, dstart - 1).decode("utf-8", "replace").rstrip("\0 "))
    meta = header.get(tensor)
    if meta is None:
        raise KeyError("%s is not in %s's header" % (tensor, shard))
    lo, hi = meta["data_offsets"]
    if hi < lo:
        raise ValueError("%s: data_offsets %r are inverted" % (tensor, meta["data_offsets"]))
    # the rule this file exists for
    raw = http_range(url, dstart + lo, dstart + hi - 1)
    return meta.get("dtype", ""), meta.get("shape", []), raw, shard


def describe(dtype, shape, raw):
    v = values(raw, dtype)
    if not v:
        return {"dtype": dtype, "shape": shape, "bytes": len(raw), "decoded": False}
    fin = [x for x in v if x == x and abs(x) != float("inf")]
    return {
        "dtype": dtype,
        "shape": shape,
        "bytes": len(raw),
        "decoded": True,
        "n": len(v),
        "mean": sum(fin) / len(fin) if fin else None,
        "min": min(fin) if fin else None,
        "max": max(fin) if fin else None,
        "nonfinite": len(v) - len(fin),
    }


def self_test():
    """The offset rule on a synthetic shard, and the statistics on a synthetic norm gain."""
    import tempfile

    failures = 0

    def check(ok, what):
        nonlocal failures
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            failures += 1

    d = pathlib.Path(tempfile.mkdtemp())
    # two tensors, the FIRST one deliberately long so the second's offset is past it (the 11,656-byte trap)
    first = bytes(range(256)) * 45           # 11,520 B
    second = struct.pack("<128H", *[struct.unpack("<H", struct.pack("<e", 1.5))[0]] * 128)  # F16 [128] of 1.5
    hdr = json.dumps({
        "a.first": {"dtype": "U8", "shape": [len(first)], "data_offsets": [0, len(first)]},
        "b.second": {"dtype": "F16", "shape": [128], "data_offsets": [len(first), len(first) + len(second)]},
    }).encode()
    shard = d / "model.safetensors"
    shard.write_bytes(struct.pack("<Q", len(hdr)) + hdr + first + second)

    blob = shard.read_bytes()
    start = data_section_start(blob[:8])
    check(start == 8 + len(hdr), "the data section starts at 8 + header_len, not at 8")
    (hlen,) = struct.unpack("<Q", blob[:8])
    header = json.loads(blob[8:start].decode())
    lo, hi = header["b.second"]["data_offsets"]
    right = blob[start + lo:start + hi]
    wrong = blob[8 + lo:8 + hi]
    check(right == second, "the offset applied from the data section returns the tensor")
    check(wrong != second, "... and the same offset applied from byte 8 does NOT (the mistake this file prevents)")
    stats = describe("F16", [128], right)
    check(abs(stats["mean"] - 1.5) < 1e-6 and stats["min"] == stats["max"] == 1.5,
          "statistics make a norm gain visible (mean 1.5, not a plausible-looking slice of another tensor)")
    if failures:
        print("self-test: %d failure(s)" % failures)
    else:
        print("self-test: ok")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--repo")
    ap.add_argument("--tensor")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", help="write the raw tensor bytes here (what a comparison script consumes)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.repo or not a.tensor:
        ap.print_help()
        return 2
    try:
        dtype, shape, raw, shard = read_tensor(a.repo, a.tensor, a.revision)
    except KeyError as e:
        print("hf_tensor: %s" % e, file=sys.stderr)
        return 1
    except (ValueError, OSError) as e:
        print("hf_tensor: cannot tell: %s" % e, file=sys.stderr)
        return 2
    stats = describe(dtype, shape, raw)
    stats["tensor"] = a.tensor
    stats["shard"] = shard
    if a.out:
        pathlib.Path(a.out).write_bytes(raw)
        stats["wrote"] = a.out
    if a.json:
        print(json.dumps(stats))
    else:
        print("%s  %s %s  %d B  (%s)" % (a.tensor, dtype, shape, len(raw), shard))
        if stats["decoded"]:
            print("  mean %+.6f  min %+.6f  max %+.6f  nonfinite %d" %
                  (stats["mean"], stats["min"], stats["max"], stats["nonfinite"]))
            print("  a norm gain is ~1.0 and a weight matrix is ~0: check these before trusting a correlation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
