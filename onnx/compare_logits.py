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

"""Compare two logits dumps step by step: KL divergence and top-1 agreement.

    python3 onnx/compare_logits.py --forward DIR --ort DIR [--steps N]

The forward route (`1bit npu-run --dump-logits DIR`) writes `decode_logits_idx<step>.bin`, and
`1bit ort-run --dump-logits DIR` writes `logits_<step>.bin`; both are raw float32 logits of the
vocabulary, one file per generated token, in generation order. KL is taken as
KL(forward || ort) in nats over the softmax of the raw logits, which is what "the same model"
means when two different kernels compute it: a small number says the next-token distribution
moved only as much as the numerics did, and top-1 agreement says the tokens came out the same.
"""

import argparse
import glob
import os
import re
import sys


def load_dir(path):
    """(step, file) pairs in generation order, for either naming convention."""
    out = []
    for f in glob.glob(os.path.join(path, "*.bin")):
        m = re.search(r"(?:decode_logits_idx|logits_)(-?\d+)\.bin$", f)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", required=True, help="dir from `1bit npu-run --dump-logits`")
    ap.add_argument("--ort", required=True, help="dir from `1bit ort-run --dump-logits`")
    ap.add_argument("--steps", type=int, default=0, help="limit to the first N common steps")
    args = ap.parse_args()

    import numpy as np

    fwd = {s: f for s, f in load_dir(args.forward) if s >= 0}
    ort = {s: f for s, f in load_dir(args.ort) if s >= 0}
    steps = sorted(set(fwd) & set(ort))
    if args.steps:
        steps = steps[: args.steps]
    if not steps:
        print(f"no common steps: {len(fwd)} in {args.forward}, {len(ort)} in {args.ort}")
        return 1

    print(f"steps: {len(steps)}  (forward {len(fwd)} files, ort {len(ort)} files)")
    print(f"{'step':>4}  {'KL(forward||ort)':>16}  {'top1':>4}  {'max|dlogit|':>11}  {'sum|dlogit|':>12}")
    kls, agree = [], 0
    for s in steps:
        a = np.fromfile(fwd[s], dtype=np.float32)
        b = np.fromfile(ort[s], dtype=np.float32)
        if a.shape != b.shape:
            print(f"step {s}: shape mismatch {a.shape} vs {b.shape}")
            return 1
        la = a - a.max()
        lb = b - b.max()
        pa = np.exp(la)
        pa /= pa.sum()
        pb = np.exp(lb)
        pb /= pb.sum()
        kl = float(np.sum(pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))))
        top = int(a.argmax()) == int(b.argmax())
        agree += top
        kls.append(kl)
        print(
            f"{s:4d}  {kl:16.3e}  {'yes' if top else 'NO':>4}  "
            f"{float(np.abs(a - b).max()):11.3e}  {float(np.abs(a - b).sum()):12.3e}"
        )
    print(
        f"\nmean KL {sum(kls) / len(kls):.3e} nats, max KL {max(kls):.3e} nats, "
        f"top-1 agreement {agree}/{len(steps)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
