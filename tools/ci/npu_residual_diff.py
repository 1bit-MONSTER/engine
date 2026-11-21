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
"""npu_residual_diff.py -- compare two arms' per-layer residual dumps (engine#3196).

`NPU_INFER_DUMP_DIR=<dir>` writes `<dir>/layer_<ll>.f32` per layer: the residual stream after that layer,
M x H little-endian f32, row-major (docs/npu.md), so a bit-exactness gate can ask WHERE two arms part instead of
only whether their logits match byte for byte.

That distinction is the whole point. On a container whose `lm_head` the device does not claim, the device falls
through to an f32 host matvec while the integer arm takes its own int8 head ladder -- so a logits comparison is
deciding between two HEADS, not between two bodies, and it says nothing about the kernel family the gate exists
for (engine#3196: device vs f32 max|d| 6.58 with the same argmax, device vs i8 max|d| 16.35 with the argmax
flipped, and the i8 logit vector ~0.3x the others). The residuals below are taken BEFORE the head, so they
answer the body question on its own.

usage: npu_residual_diff.py DIR_A DIR_B [--band LO:HI] [--threshold REL] [--self-test]
One line for the renderer (the last non-empty line is the only one a runner keeps):
  - the band summary and the verdict, naming the worst layer and the two arms.
Exit: 0 every common layer is within the threshold, 1 one is not (named), 2 cannot tell -- no common layer, an
empty dump, or a file whose length is not a whole number of floats (a partial write says nothing either way).
"""
import argparse
import math
import pathlib
import struct
import sys
import tempfile

F32 = struct.Struct('<f')
# The flat band engine#3196 measured over: layers 4..43 keep the same shape across the families in the corpus,
# while the first layers carry load-time quantisation and the last ones sit next to the paged/contiguous KV tail.
DEFAULT_BAND = (4, 43)
DEFAULT_THRESHOLD = 2e-2


def read_dump(path):
    raw = pathlib.Path(path).read_bytes()
    if not raw or len(raw) % F32.size:
        return None, f'{path.name} is {len(raw)} B, not a whole number of f32 values'
    n = len(raw) // F32.size
    return struct.unpack(f'<{n}f', raw), None


def layers(directory):
    """({layer index: path}, reason) for one arm's dump directory."""
    d = pathlib.Path(directory)
    if not d.is_dir():
        return {}, f'{directory} is not a directory'
    got = {}
    for p in d.glob('layer_*.f32'):
        try:
            got[int(p.stem.split('_', 1)[1])] = p
        except (IndexError, ValueError):
            continue
    return got, (None if got else f'{directory} has no layer_*.f32')


def relative(a, b):
    """L2(a-b) / L2(b): the arms are compared in the reference's own scale, so a band summary is a ratio."""
    if len(a) != len(b):
        return None, f'length {len(a)} vs {len(b)}'
    num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    den = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / den, None


def compare(dir_a, dir_b, band=DEFAULT_BAND, threshold=DEFAULT_THRESHOLD):
    """(verdict dict, reason). Verdict: {'common', 'band': [(layer, rel)], 'worst': (layer, rel), 'over': [...]}."""
    a, why_a = layers(dir_a)
    b, why_b = layers(dir_b)
    if not a or not b:
        return None, why_a or why_b
    common = sorted(set(a) & set(b))
    if not common:
        return None, f'{dir_a} and {dir_b} have no layer in common'
    band_rows, skipped = [], []
    for l in common:
        va, err_a = read_dump(a[l])
        vb, err_b = read_dump(b[l])
        if err_a or err_b:
            skipped.append((l, err_a or err_b))
            continue
        rel, err = relative(va, vb)
        if err:
            skipped.append((l, f'layer {l:02d}: {err}'))
            continue
        if band[0] <= l <= band[1]:
            band_rows.append((l, rel))
    if not band_rows:
        return None, f'no layer in the {band[0]}..{band[1]} band had two comparable dumps'
    worst = max(band_rows, key=lambda r: r[1])
    over = [r for r in band_rows if r[1] > threshold]
    return {'dir_a': str(dir_a), 'dir_b': str(dir_b), 'common': len(common), 'band': band_rows,
            'worst': worst, 'over': over, 'skipped': skipped, 'threshold': threshold, 'band_range': band}, None


def report(v):
    mean = sum(r for _, r in v['band']) / len(v['band'])
    lo, hi = v['band_range']
    print(f'residuals: {v["common"]} layer(s) in common; band {lo}..{hi}: mean rel {mean:.2e}, '
          f'worst layer {v["worst"][0]:02d} at {v["worst"][1]:.2e}')
    if v['skipped']:
        print(f'residuals: {len(v["skipped"])} layer(s) not compared ({v["skipped"][0][1]})')
    if v['over']:
        named = ', '.join(f'{l:02d}={r:.2e}' for l, r in v['over'][:6])
        print(f'residuals: FAIL {len(v["over"])} layer(s) over {v["threshold"]:.0e} between '
              f'{v["dir_a"]} and {v["dir_b"]}: {named}')
        return 1
    print(f'residuals: every layer {lo}..{hi} is within {v["threshold"]:.0e} between '
          f'{v["dir_a"]} and {v["dir_b"]} -- the two bodies agree up to layer {hi}, so a logits '
          f'difference is not theirs')
    return 0


def self_test():
    ok = [0, 0]

    def chk(cond, what):
        if cond:
            ok[0] += 1
        else:
            ok[1] += 1
            print(f'FAIL {what}')

    def dump(d, layer, values):
        p = pathlib.Path(d) / f'layer_{layer:02d}.f32'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(struct.pack(f'<{len(values)}f', *values))

    with tempfile.TemporaryDirectory() as tmp:
        a, b, c = (pathlib.Path(tmp) / n for n in ('a', 'b', 'c'))
        base = [1.0, -2.0, 0.5, 3.0]
        for l in range(0, 8):
            dump(a, l, base)
            dump(b, l, base)                       # identical → rel 0
            dump(c, l, [x * 1.1 for x in base])    # 10 % scale → rel ~0.1
        v, why = compare(a, b)
        chk(v is not None and not v['over'], 'identical dumps are within the threshold')
        v2, _ = compare(a, c)
        chk(v2 is not None and v2['over'], 'a 10 % scale difference is over the threshold, and named')
        # a layer outside the band does not decide the verdict
        dump(c, 40, [x * 5 for x in base])
        v3, _ = compare(a, c)
        chk(v3 is not None and all(l <= 7 for l, _ in v3['over']), 'layers outside the band are not in the band verdict')
        # cannot-tell cases
        chk(compare(a, pathlib.Path(tmp) / 'nope')[0] is None, 'a missing directory is cannot-tell')
        (pathlib.Path(tmp) / 'short').mkdir()
        (pathlib.Path(tmp) / 'short' / 'layer_05.f32').write_bytes(b'\x00\x00\x00')
        chk(compare(a, pathlib.Path(tmp) / 'short')[0] is None, 'a partial f32 file is cannot-tell, not a pass')
        v4, _ = compare(pathlib.Path(tmp) / 'nope', pathlib.Path(tmp) / 'nope2')
        chk(v4 is None, 'two missing directories are cannot-tell')
    # the band default is the one engine#3196 measured over
    chk(DEFAULT_BAND == (4, 43), 'the default band is layers 4..43')
    print(f'npu_residual_diff self-test: {ok[0]} ok, {ok[1]} failed')
    return 0 if ok[1] == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('dir_a', nargs='?')
    ap.add_argument('dir_b', nargs='?')
    ap.add_argument('--band', default=f'{DEFAULT_BAND[0]}:{DEFAULT_BAND[1]}',
                    help='layers that decide the verdict (default %(default)s)')
    ap.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                    help='relative L2 above which a layer fails (default %(default)s)')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.dir_a or not a.dir_b:
        ap.error('two dump directories are required (or --self-test)')
    try:
        lo, hi = (int(x) for x in a.band.split(':', 1))
    except ValueError:
        print(f'npu_residual_diff: --band wants LO:HI, got {a.band!r}', file=sys.stderr)
        return 2
    v, why = compare(a.dir_a, a.dir_b, (lo, hi), a.threshold)
    if v is None:
        print(f'residuals: {why} — cannot tell what the bodies did, and that is not a pass', file=sys.stderr)
        return 2
    return report(v)


if __name__ == '__main__':
    sys.exit(main())
