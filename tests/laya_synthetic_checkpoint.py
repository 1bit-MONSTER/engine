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
"""laya_synthetic_checkpoint.py <dir>: the scoring head's real tensor names at d=32, so the Laya rows can run.

engine#3161: `laya_route_e2e` / `laya_models_e2e` / `laya_eval_e2e` / `laya_route_serve` / `laya_checkpoint` are
registered only when `ONEBIT_LAYA_ROUTE_MODEL` names a checkpoint, and nothing has ever set it -- so the `--laya-models`
rail (which changed tonight with engine#3130's `--port 0` read-back) has been covered only by a row nobody executes.

The rows do not need a *good* scorer, they need a *loadable* one: `tests/laya_models_e2e.sh` pins
`"min_confidence": 1.01` so every request lands on its default class while Laya is still asked. A synthetic checkpoint
is therefore sufficient, and this is the same shape `tests/laya_finetune_test.py` already writes for its own fixture.

Written with `struct` and `json` only -- no numpy, no torch -- because CMake runs this at CONFIGURE time and a build box
must not need a scientific stack to register a test. The values come from a fixed seed, so two configures produce the
same bytes.
"""
import argparse
import json
import os
import random
import struct
import sys

D, F_, DH = 32, 64, 32          # the dims tests/laya_finetune_test.py uses (d 32, 16-head layout of laya/scorer.cpp)


def f32(values):
    return struct.pack("<%df" % len(values), *values)


def randn(rng, shape, scale=1.0):
    n = 1
    for s in shape:
        n *= s
    return f32([scale * rng.gauss(0.0, 1.0) for _ in range(n)])


def ones(n, v=1.0):
    return f32([v] * n)


def write_safetensors(path, tensors):
    """Safetensors: <u64 header length><JSON header, 8-byte padded><raw F32 blobs in header order>."""
    header, blobs, off = {}, [], 0
    for name, (shape, raw) in tensors.items():
        header[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    os.replace(tmp, path)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", help="checkpoint directory to create")
    ap.add_argument("--seed", type=int, default=1234, help="fixed seed, so two configures agree byte for byte")
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    t = {}
    t["encoder.final_norm.weight"] = ((D,), randn(rng, (D,)))
    t["type_emb.weight"] = ((3, D), randn(rng, (3, D), 0.02))
    for i in range(2):
        p = "head.layers.%d." % i
        t[p + "norm1.weight"], t[p + "norm1.bias"] = ((D,), ones(D)), ((D,), ones(D, 0.0))
        t[p + "self_attn.in_proj_weight"] = ((3 * D, D), randn(rng, (3 * D, D), 0.05))
        t[p + "self_attn.in_proj_bias"] = ((3 * D,), ones(3 * D, 0.0))
        t[p + "self_attn.out_proj.weight"] = ((D, D), randn(rng, (D, D), 0.05))
        t[p + "self_attn.out_proj.bias"] = ((D,), ones(D, 0.0))
        t[p + "norm2.weight"], t[p + "norm2.bias"] = ((D,), ones(D)), ((D,), ones(D, 0.0))
        t[p + "linear1.weight"] = ((F_, D), randn(rng, (F_, D), 0.05))
        t[p + "linear1.bias"] = ((F_,), ones(F_, 0.0))
        t[p + "linear2.weight"] = ((D, F_), randn(rng, (D, F_), 0.05))
        t[p + "linear2.bias"] = ((D,), ones(D, 0.0))
    t["scorer.0.weight"], t["scorer.0.bias"] = ((D,), ones(D)), ((D,), ones(D, 0.0))
    t["scorer.1.weight"] = ((DH, D), randn(rng, (DH, D), 0.1))
    t["scorer.1.bias"] = ((DH,), ones(DH, 0.0))
    t["scorer.3.weight"] = ((1, DH), randn(rng, (1, DH), 0.1))
    t["scorer.3.bias"] = ((1,), ones(1, 0.0))
    t["temperature"] = ((3,), ones(3))

    os.makedirs(a.dir, exist_ok=True)
    write_safetensors(os.path.join(a.dir, "model.safetensors"), t)
    with open(os.path.join(a.dir, "rl_agent_config.json"), "w") as f:
        json.dump({"temperature_by_options": {}}, f)
    print("laya_synthetic_checkpoint: %d tensors at d=%d -> %s" % (len(t), D, a.dir))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
