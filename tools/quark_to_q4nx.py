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
"""Pack an AMD Quark weight-only int4 model into a Q4NX directory, with Quark's own codes (docs/npu.md).

usage: tools/quark_to_q4nx.py PARTS.safetensors HF_DIR OUT_DIR [--allow-uint4] [--tokenizer DIR]

PARTS is Quark's frozen model written as one safetensors file: for every quantized linear
`<base>.weight` (the dequantized value, BF16 or F32), `<base>._weight_quantizer.scale` [N, K/G] and
`<base>._weight_quantizer.zero_point` [N, K/G] (integers, stored as F32), plus every other tensor the
model has (AWQ rescales the norms and the previous linear of each scaling pair, so those values come from
PARTS too).  HF_DIR is the BF16 checkpoint it was made from: names, order, config and tokenizer.  The
container is written by tools/safetensors_to_q4nx.py's own packer, so it differs from an RTN pack only
where Quark changed something.

The map from Quark to q4_1 (w = code * scale + zero, per output row and 32-column group, bf16):
  * int4 (symmetric, q in [-8, 7], zp = 0): code = q + 8, scale = s, zero = -8 s.  EXACT: -8 s is a
    power-of-two multiple of a bf16 scale, so bf16 holds it.  Measured on Qwen3-1.7B AWQ int4_wo_32:
    0.0 relative difference to Quark's codes.
  * uint4 (asymmetric, q and zp in [0, 15]): code = q, zero = -zp s, which bf16 ROUNDS.  The rounding is a
    per-group constant offset (up to ~0.05 step) that a float reader of the pack sees coherently; on
    AWQ's down_proj it took a weight-only KLD from 0.08 to 0.99 (engine QUARK-EVAL, 2026-11-15).  The NPU's
    int8 path absorbs most of it, but a Q4NX file is not NPU-only, so uint4 is refused unless
    --allow-uint4 is given.
  G must be a multiple of 32 (each Quark group then covers G/32 q4_1 groups).  A tensor Quark left
  unquantized (lm_head, which Quark excludes) is RTN'd exactly as the baseline pack would.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import safetensors_to_q4nx as s2q  # noqa: E402
from q4nx_pack import bf16_bits, from_bf16  # noqa: E402

SCALE, ZERO = "._weight_quantizer.scale", "._weight_quantizer.zero_point"


class QuarkParts:
    """The `source` hook of safetensors_to_q4nx.pack: Quark's values and codes, by tensor name."""

    def __init__(self, path, allow_uint4=False):
        self.path = path
        self.header, self.off = s2q.read_header(path)
        self.allow_uint4 = allow_uint4
        self.stats = {"quark_linears": 0, "quark_values": 0, "int4": 0, "uint4": 0,
                      "max_off_grid": 0.0, "max_zero_round_steps": 0.0}

    def has(self, name):
        return name in self.header and name != "__metadata__"

    def get(self, name):
        return s2q.read_tensor(self.path, self.header, self.off, name)

    def value(self, name, w):
        if not self.has(name):
            return w
        v = self.get(name)
        if v.shape != w.shape:
            raise SystemExit(f"{self.path}: {name} has shape {list(v.shape)}, the checkpoint {list(w.shape)}")
        self.stats["quark_values"] += 1
        return v

    def q4_1(self, name, w):
        if not name.endswith(".weight"):
            return None
        base = name[:-len(".weight")]
        if not (self.has(base + SCALE) and self.has(base + ZERO)):
            return None
        N, K = w.shape
        s = self.get(base + SCALE).reshape(N, -1)
        zp = self.get(base + ZERO).reshape(N, -1)
        G = K // s.shape[1]
        if G * s.shape[1] != K or G % 32:
            raise SystemExit(f"{base}: group {G} does not tile into 32-column q4_1 groups (K {K})")
        # w here is Quark's own dequantized value (value() ran first): bf16((q - zp) * s).  bf16 rounding puts
        # it a few hundredths of a step off the grid; a wrong (q, zp, s) convention lands ~0.5 off everywhere.
        sg = np.repeat(s, G, axis=1)
        zg = np.repeat(zp, G, axis=1)
        qf = w / np.where(sg > 0, sg, 1.0) + zg
        q = np.rint(qf)
        off = float(np.abs(qf - q).max())
        self.stats["max_off_grid"] = max(self.stats["max_off_grid"], off)
        if off > 0.25:
            raise SystemExit(f"{base}: the weight is not on Quark's grid (max {off:.3g} step off)")
        signed = q.min() < 0
        qmin = -8.0 if signed else 0.0
        if q.min() < qmin or q.max() > qmin + 15:
            raise SystemExit(f"{base}: codes {q.min():.0f}..{q.max():.0f} are not a 4-bit range")
        if not signed and not self.allow_uint4:
            raise SystemExit(f"{base}: asymmetric uint4 does not map exactly into q4_1 (the bf16 zero rounds; see "
                             "this file's docstring). Quantize with int4_wo_32/int4_wo_128, or pass --allow-uint4 "
                             "for an NPU-only pack.")
        rep = G // 32
        scale = np.repeat(s, rep, axis=1)
        zero = np.repeat((qmin - zp) * s, rep, axis=1)
        zb = from_bf16(bf16_bits(zero))
        step = np.where(scale > 0, scale, 1.0)
        self.stats["max_zero_round_steps"] = max(self.stats["max_zero_round_steps"],
                                                 float((np.abs(zb - zero) / step).max()))
        self.stats["quark_linears"] += 1
        self.stats["uint4" if not signed else "int4"] += 1
        return (q - qmin).astype(np.uint8), scale.astype(np.float32), zero.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("parts")
    p.add_argument("hf")
    p.add_argument("out")
    p.add_argument("--tokenizer")
    p.add_argument("--allow-uint4", action="store_true")
    a = p.parse_args()
    src = QuarkParts(a.parts, a.allow_uint4)
    args = argparse.Namespace(hf=a.hf, out=a.out, tokenizer=a.tokenizer, dtype="auto", arch="auto",
                              weight_layout="rm")
    s2q.pack(args, source=src)
    print(json.dumps(src.stats))
    if src.stats["quark_linears"] == 0:
        raise SystemExit(f"{a.parts}: no Quark-quantized linear found (no {SCALE} tensors); the pack would be RTN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
