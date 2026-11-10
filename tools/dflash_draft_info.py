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
"""dflash_draft_info.py -- what a DFlash draft package declares, checked against its own config (engine#1780).

A DFlash draft is a two-file model source: `config.json` (model_type "dflash": block size, mask token, the target
layers it taps, the draft's own geometry) and a Q4NX container with the draft's weights. Before a forward for it is
written, three questions have to be answered from the bytes, not from the names:

  1. is every projection in a form our reader already decodes?  The container labels them `I8 [tiles, 5120]`, which
     reads as a row x column int8 matrix and is not one: it is the ordinary Q4NX q4_1 tile grid (32 rows x 256
     columns per 5120-byte tile: 512 B of bf16 scales, 512 B of bf16 minimums, 4096 B of 4-bit codes), so the tile
     count must equal (N/32) * (K/256) for the N x K the config gives that role.
  2. do the container's own scalars agree with config.json?  (block size, taps, rope theta, eps, window, scale.)
  3. are the tile headers sane?  Every scale finite and positive, every minimum finite. A wrong layout fails this at
     once; the right one also shows min/scale near -7.5, the q4_1 signature of a symmetric 16-level group.

    tools/dflash_draft_info.py <package dir>            # PASS/FAIL lines, exit 1 on any FAIL
    tools/dflash_draft_info.py <package dir> --json     # the census as JSON
    tools/dflash_draft_info.py --selftest               # a synthetic package and its negative controls; no model

Standard library only, and it reads tile headers rather than whole tensors, so the 0.27 GB package takes seconds.
It decodes nothing into a model and runs nothing: it is the gate before a loader, not a loader.
"""
import argparse
import json
import math
import os
import struct
import sys
import tempfile

TILE_ROWS, TILE_COLS, TILE_BYTES, HDR_BYTES = 32, 256, 5120, 1024
ROLES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
REQUIRED = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim", "block_size", "target_layer_ids", "draft_weight")


def bf16(u16):
    """A bf16 value: the top 16 bits of an IEEE float32."""
    return struct.unpack("<f", struct.pack("<I", u16 << 16))[0]


def role_shape(role, c):
    """(N, K) = (out, in) of a projection, from the config alone."""
    h, im = c["hidden_size"], c["intermediate_size"]
    q = c["num_attention_heads"] * c["head_dim"]
    kv = c["num_key_value_heads"] * c["head_dim"]
    return {"self_attn.q_proj": (q, h), "self_attn.k_proj": (kv, h), "self_attn.v_proj": (kv, h),
            "self_attn.o_proj": (h, q), "mlp.gate_proj": (im, h), "mlp.up_proj": (im, h),
            "mlp.down_proj": (h, im)}[role]


def tiles_for(n, k):
    """Tile count of an N x K projection, or None when it does not fill whole tiles."""
    if n % TILE_ROWS or k % TILE_COLS:
        return None
    return (n // TILE_ROWS) * (k // TILE_COLS)


def read_header(path):
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("shorter than its 8-byte header length")
        (n,) = struct.unpack("<Q", raw)
        size = os.path.getsize(path)
        if n <= 0 or n > size - 8:
            raise ValueError(f"header length {n} does not fit a {size}-byte file")
        return json.loads(f.read(n)), 8 + n, size


def tile_header_stats(f, base, lo, ntiles):
    """Scale/minimum sanity over every tile header of one tensor. Reads 1024 of each 5120 bytes."""
    bad_scale = bad_min = 0
    ratios = []
    for t in range(ntiles):
        f.seek(base + lo + t * TILE_BYTES)
        h = f.read(HDR_BYTES)
        if len(h) != HDR_BYTES:
            return {"short_read_at_tile": t}
        vals = struct.unpack("<512H", h)
        for i in range(256):
            s, m = bf16(vals[i]), bf16(vals[256 + i])
            if not math.isfinite(s) or s <= 0.0:
                bad_scale += 1
            elif not math.isfinite(m):
                bad_min += 1
            elif t % 16 == 0 and i % 16 == 0:        # a sample is enough for the median
                ratios.append(m / s)
    ratios.sort()
    return {"groups": ntiles * 256, "bad_scale": bad_scale, "bad_min": bad_min,
            "min_over_scale_median": round(ratios[len(ratios) // 2], 3) if ratios else None}


def inspect(pkg):
    """Returns (checks, census). `checks` is a list of (ok, text); the census is JSON-serialisable."""
    checks, census = [], {"package": pkg}

    def check(ok, text):
        checks.append((bool(ok), text))
        return bool(ok)

    cfg_path = os.path.join(pkg, "config.json")
    if not check(os.path.isfile(cfg_path), f"config.json is present in {pkg}"):
        return checks, census
    c = json.load(open(cfg_path))
    if not check(c.get("model_type") == "dflash", f"config model_type is dflash (got {c.get('model_type')!r})"):
        return checks, census
    missing = [k for k in REQUIRED if k not in c]
    if not check(not missing, f"config carries the geometry and the interface keys (missing: {missing or 'none'})"):
        return checks, census
    census["config"] = {k: c[k] for k in REQUIRED + ("mask_token_id", "sliding_window", "causal") if k in c}
    wpath = os.path.join(pkg, c["draft_weight"])
    if not check(os.path.isfile(wpath), f"the draft weight {c['draft_weight']} is present"):
        return checks, census
    try:
        hdr, base, size = read_header(wpath)
    except (ValueError, json.JSONDecodeError) as e:
        check(False, f"the weight file parses as a Q4NX table: {e}")
        return checks, census
    names = [k for k in hdr if k != "__metadata__"]
    check(True, f"the weight file parses as a Q4NX table: {len(names)} tensors, {base}-byte header, {size} bytes")
    end = max((hdr[k]["data_offsets"][1] for k in names), default=0)
    check(base + end <= size, f"every tensor lies inside the file (payload ends at {base + end} of {size})")

    def span(name):
        lo, hi = hdr[name]["data_offsets"]
        return lo, hi - lo

    # 1. the projections: an ordinary q4_1 tile grid, sized by the config
    L = c["num_hidden_layers"]
    proj, wrong, absent = [], [], []
    with open(wpath, "rb") as f:
        for layer in range(L):
            for role in ROLES:
                name = f"model.layers.{layer}.{role}.weight"
                if name not in hdr:
                    absent.append(name)
                    continue
                n, k = role_shape(role, c)
                want = tiles_for(n, k)
                lo, nbytes = span(name)
                row = {"name": name, "rows": n, "cols": k, "tiles_expected": want,
                       "bytes": nbytes, "declared": [hdr[name]["dtype"], hdr[name]["shape"]]}
                if want is None or nbytes != want * TILE_BYTES:
                    wrong.append(f"{name}: {nbytes} B, expected {want} tiles x {TILE_BYTES}")
                else:
                    row.update(tile_header_stats(f, base, lo, want))
                proj.append(row)
    check(not absent, f"all {L} x {len(ROLES)} projections are in the container (absent: {absent[:3] or 'none'})")
    check(not wrong and proj,
          f"each projection is a whole q4_1 tile grid of the config's N x K: bytes == (N/32)(K/256) x 5120 "
          f"({len(proj) - len(wrong)} of {len(proj)}{'; ' + wrong[0] if wrong else ''})")
    bad = [r["name"] for r in proj if r.get("bad_scale") or r.get("bad_min") or "short_read_at_tile" in r]
    groups = sum(r.get("groups", 0) for r in proj)
    check(not bad and groups, f"every tile header is sane: scale finite and positive, minimum finite, over {groups} "
                              f"32-weight groups (first bad: {bad[0] if bad else 'none'})")
    ratios = sorted(r["min_over_scale_median"] for r in proj if r.get("min_over_scale_median") is not None)
    if ratios:
        check(-8.5 <= ratios[0] and ratios[-1] <= -6.5,
              f"minimum/scale sits at the q4_1 signature of a symmetric 16-level group: median per tensor "
              f"{ratios[0]:.3f} .. {ratios[-1]:.3f} (expected near -7.5)")
    census["projections"] = proj

    # 2. the container's own scalars against config.json
    def scalars(name, fmt):
        if name not in hdr:
            return None
        lo, nbytes = span(name)
        with open(wpath, "rb") as f:
            f.seek(base + lo)
            return list(struct.unpack("<" + fmt * (nbytes // 4), f.read(nbytes)))

    pairs = (("dflash.block_size", "i", "block_size", None), ("dflash.target_layer_ids", "i", "target_layer_ids", None),
             ("dflash.sliding_window", "i", "sliding_window", None), ("dflash.rope_theta", "f", "rope_theta", 1e-6),
             ("dflash.rms_eps", "f", "rms_norm_eps", 1e-6), ("dflash.embedding_scale", "f", "embedding_scale", 1e-6))
    stated = {}
    for tname, fmt, key, rel in pairs:
        got = scalars(tname, fmt)
        if got is None or key not in c:
            continue
        want = c[key] if isinstance(c[key], list) else [c[key]]
        same = len(got) == len(want) and all(
            (g == w) if rel is None else abs(g - w) <= rel * max(1.0, abs(w)) for g, w in zip(got, want))
        stated[tname] = got
        check(same, f"{tname} in the container equals config.json's {key} ({got} vs {want})")
    census["container_scalars"] = stated

    # 3. the feature projection and the norms: 2-byte tensors with the sizes the config implies
    h, taps = c["hidden_size"], len(c["target_layer_ids"])
    if "model.fc.weight" in hdr:
        _, nbytes = span("model.fc.weight")
        check(nbytes == h * taps * h * 2,
              f"model.fc.weight is hidden x (taps x hidden) at 2 bytes: {nbytes} B == {h} x {taps * h} x 2 "
              f"(declared {hdr['model.fc.weight']['dtype']} {hdr['model.fc.weight']['shape']})")
    else:
        check(False, "model.fc.weight (the tap-fusing projection) is in the container")
    norms = [k for k in names if k.endswith("norm.weight")]
    off = [k for k in norms if span(k)[1] not in (2 * h, 2 * c["head_dim"])]
    check(norms and not off, f"{len(norms)} norm weights are 2 bytes x hidden or x head_dim (off-size: {off[:2] or 'none'})")
    known = {f"model.layers.{layer}.{r}.weight" for layer in range(L) for r in ROLES} | set(norms) | \
            {"model.fc.weight"} | {p[0] for p in pairs}
    extra = sorted(set(names) - known)
    check(not extra, f"no tensor is left unexplained (unexplained: {extra[:4] or 'none'})")
    census["tensors"] = len(names)
    census["unexplained"] = extra
    return checks, census


# ---- self-test: a synthetic package small enough to write in milliseconds ----------------------------------------

def _write_package(d, mutate=None):
    c = {"model_type": "dflash", "architectures": ["DFlashDraftModel"], "draft_weight": "dflash_weight.q4nx",
         "block_size": 8, "mask_token_id": 4, "target_layer_ids": [1, 3], "num_hidden_layers": 1,
         "hidden_size": 256, "intermediate_size": 512, "num_attention_heads": 2, "num_key_value_heads": 1,
         "head_dim": 128, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "sliding_window": 1024,
         "embedding_scale": 16.0, "causal": False}
    tensors = {}

    def tile():
        s = struct.pack("<H", struct.unpack("<I", struct.pack("<f", 0.125))[0] >> 16) * 256
        m = struct.pack("<H", struct.unpack("<I", struct.pack("<f", -0.9375))[0] >> 16) * 256
        return s + m + bytes(4096)

    for role in ROLES:
        n, k = role_shape(role, c)
        nt = tiles_for(n, k)
        tensors[f"model.layers.0.{role}.weight"] = ("I8", [nt, TILE_BYTES], tile() * nt)
    for nm, w in (("model.norm.weight", 256), ("model.hidden_norm.weight", 256),
                  ("model.layers.0.input_layernorm.weight", 256), ("model.layers.0.post_attention_layernorm.weight", 256),
                  ("model.layers.0.self_attn.q_norm.weight", 128), ("model.layers.0.self_attn.k_norm.weight", 128)):
        tensors[nm] = ("BF16", [w], struct.pack("<H", 0x3F80) * w)
    tensors["model.fc.weight"] = ("BF16", [256 // 32, 512, 32], bytes(256 * 512 * 2))
    tensors["dflash.block_size"] = ("I32", [1], struct.pack("<i", 8))
    tensors["dflash.target_layer_ids"] = ("I32", [2], struct.pack("<2i", 1, 3))
    tensors["dflash.sliding_window"] = ("I32", [1], struct.pack("<i", 1024))
    tensors["dflash.rope_theta"] = ("F32", [1], struct.pack("<f", 1000000.0))
    tensors["dflash.rms_eps"] = ("F32", [1], struct.pack("<f", 1e-06))
    tensors["dflash.embedding_scale"] = ("F32", [1], struct.pack("<f", 16.0))
    if mutate:
        mutate(c, tensors)
    hdr, blob, off = {}, bytearray(), 0
    for name, (dtype, shape, data) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        blob += data
        off += len(data)
    hj = json.dumps(hdr).encode()
    with open(os.path.join(d, "dflash_weight.q4nx"), "wb") as f:
        f.write(struct.pack("<Q", len(hj)) + hj + bytes(blob))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(c, f)


def selftest():
    fails = 0

    def expect(cond, what):
        nonlocal fails
        print(f"{'PASS' if cond else 'FAIL'}  {what}")
        fails += 0 if cond else 1

    def run(mutate=None):
        with tempfile.TemporaryDirectory() as d:
            _write_package(d, mutate)
            checks, census = inspect(d)
            return [t for ok, t in checks if not ok], checks, census

    expect(abs(bf16(0x3F80) - 1.0) < 1e-9 and abs(bf16(0xBF80) + 1.0) < 1e-9, "bf16: 0x3F80 is 1.0 and 0xBF80 is -1.0")
    expect(tiles_for(4096, 2560) == 1280 and tiles_for(1024, 2560) == 320 and tiles_for(5120, 2560) == 1600
           and tiles_for(2560, 5120) == 1600,
           "tile arithmetic: 4096x2560 is 1280 tiles, 1024x2560 is 320, 5120x2560 and 2560x5120 are both 1600")
    expect(tiles_for(1280, 2560) == 400, "a 1280x2560 reading of the same 1280-tile tensor would be 400 tiles, not 1280")
    expect(tiles_for(100, 2560) is None, "a shape that does not fill whole tiles has no tile count")
    bad, checks, census = run()
    expect(not bad and len(checks) >= 14, f"a well-formed synthetic package passes every check ({len(checks)} checks, failing: {bad})")
    expect(len(census.get("projections", [])) == 7 and all(r["min_over_scale_median"] == -7.5 for r in census["projections"]),
           "its census lists the 7 projections with minimum/scale -7.5")

    def short_q(c, t):
        dt, sh, data = t["model.layers.0.self_attn.q_proj.weight"]
        t["model.layers.0.self_attn.q_proj.weight"] = (dt, sh, data[:-TILE_BYTES])
    bad, _, _ = run(short_q)
    expect(any("whole q4_1 tile grid" in b for b in bad), "a projection one tile short of its config shape FAILS by name")

    def head_dim_40(c, t):
        c["head_dim"] = 40
    bad, _, _ = run(head_dim_40)
    expect(any("whole q4_1 tile grid" in b for b in bad),
           "a config whose head_dim disagrees with the stored tiles FAILS (the geometry is checked, not assumed)")

    def nan_scale(c, t):
        dt, sh, data = t["model.layers.0.mlp.up_proj.weight"]
        t["model.layers.0.mlp.up_proj.weight"] = (dt, sh, struct.pack("<H", 0x7FC0) + data[2:])
    bad, _, _ = run(nan_scale)
    expect(any("every tile header is sane" in b for b in bad), "one NaN scale in one tile header FAILS")

    def f16_matrix(c, t):                       # plain f16 weights where a tile grid is declared
        dt, sh, data = t["model.layers.0.self_attn.k_proj.weight"]
        t["model.layers.0.self_attn.k_proj.weight"] = (dt, sh, struct.pack("<e", -0.02) * (len(data) // 2))
    bad, _, _ = run(f16_matrix)
    expect(any("every tile header is sane" in b or "q4_1 signature" in b for b in bad),
           "a plain half-float matrix of the same byte count does not pass as a tile grid")

    def block_mismatch(c, t):
        c["block_size"] = 16
    bad, _, _ = run(block_mismatch)
    expect(any("dflash.block_size" in b for b in bad), "a block size that differs between container and config FAILS")

    def stray(c, t):
        t["model.selector.weight"] = ("BF16", [4], bytes(8))
    bad, _, _ = run(stray)
    expect(any("unexplained" in b for b in bad), "a tensor this tool does not know (a second draft family) FAILS as unexplained")

    def not_dflash(c, t):
        c["model_type"] = "qwen3"
    bad, _, _ = run(not_dflash)
    expect(any("model_type is dflash" in b for b in bad), "a config that is not a dflash draft is refused at the first check")
    return fails


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("package", nargs="?", help="directory with config.json and the draft weight")
    ap.add_argument("--json", action="store_true", help="print the census as JSON instead of PASS/FAIL lines")
    ap.add_argument("--selftest", action="store_true", help="a synthetic package and its negative controls")
    a = ap.parse_args(argv)
    if a.selftest:
        return 1 if selftest() else 0
    if not a.package or not os.path.isdir(a.package):
        print("usage: dflash_draft_info.py <package dir> [--json] | --selftest", file=sys.stderr)
        return 2
    checks, census = inspect(a.package)
    failed = [t for ok, t in checks if not ok]
    if a.json:
        census["checks"] = [{"ok": ok, "what": t} for ok, t in checks]
        print(json.dumps(census, indent=1))
    else:
        for ok, t in checks:
            print(f"{'PASS' if ok else 'FAIL'}  {t}")
        print(f"{len(checks) - len(failed)} of {len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
