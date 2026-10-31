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
"""Dynamic int8 quantisation for the CPU graph: the only lever left on that path.

`BENCHMARK-PARITY.md` round 3 measured prefill at ~760 GFLOP/s on eight cores, which is what fp32
cores do, and decode re-reading the 2.4 GB weight file per token. Both are precision problems, not
graph problems: int8 weights are 4x smaller to stream and int8 matmul runs ~2-4x the fp32 rate on the
same cores (VNNI). Nothing else in this directory changes speed on CPU.

Activations stay float and are quantised per-call (dynamic), so no calibration data is needed and the
KV cache keeps its fp32 contract with `ort_run --cached`.

**Measured on `ryzen`, Qwen3-0.6B, 8 threads, 128-token fp32 graph as the reference** — dynamic int8 is
not shippable and this file is kept as the evidence, not as a recommended step:

| | fp32 | int8 dynamic |
|---|---|---|
| prefill 512 | 1141 tok/s | **1628** tok/s (+43%) |
| prefill 2048 | 567 tok/s | **860** tok/s (+51%) |
| decode 512 | 12.98 tok/s | 12.28 (−5%) |
| decode 2048 | 6.67 tok/s | 6.36 (−5%) |
| peak RSS | 6.60 GB | **4.68 GB** |
| tokens for `15 23 40` | `13 16 13 16` | `13 220 16 24` |

Accuracy is the blocker: mean KL **0.28 nats** over four steps, max|dlogit| 3.47, top-1 agreement 1/4 —
the greedy answer changes at the second token. Variants measured, all worse: quantising attention's
activation-by-activation matmuls too (KL 1.72), per-tensor scales (KL 0.36). The tied head is already
left in float by `MatMulConstBOnly` (its weight reaches the MatMul through a `Transpose`, so it is not
a constant input) — excluding it explicitly changes nothing, byte for byte.

So the error is entirely the 28 layers' **uint8 activations**, which is what dynamic quantisation does
by design. The accurate route is static QDQ with calibration (real scales per activation tensor, which
needs prompts to calibrate on) or int4 `MatMulNBits`, which is what the NPU side already runs. Prefill
is where int8 pays (+43-51%) because it is the FLOP-bound stage; decode does not move (−5%) because it
is latency/bandwidth-bound on the same 0.6 GB either way.

    python3 quantize_cpu.py --model ~/wt/onnx-artifacts/qwen3-0.6b-cached.onnx \\
                           --out ~/wt/onnx-artifacts/qwen3-0.6b-cached-int8.onnx
"""
import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="fp32 cached graph from export_qwen3.py --cached")
    ap.add_argument("--out", required=True, help="where to write the int8 graph")
    ap.add_argument(
        "--weight-type",
        default="qint8",
        choices=["qint8", "quint8"],
        help="qint8 for MatMul weights (the default here); quint8 is onnxruntime's own default",
    )
    ap.add_argument(
        "--keep-head-float",
        action="store_true",
        help="leave the vocab-sized projection (tied embedding / lm head) in float — the usual largest "
        "error term in small models, because logits are a 1024-wide dot product over the quantised table",
    )
    ap.add_argument(
        "--no-per-channel",
        action="store_true",
        help="disable per-channel weight scales (wider error for a smaller file; on by default)",
    )
    ap.add_argument(
        "--all-matmuls",
        action="store_true",
        help="also quantise matmuls whose weight is not a constant, e.g. attention's QK^T and score.V "
        "— measured to destroy this model (KL 1.72 nats, top-1 0/4), off by default",
    )
    a = ap.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic

    if not os.path.exists(a.model):
        print(f"missing model: {a.model}", file=sys.stderr)
        return 2
    exclude = []
    if a.keep_head_float:
        import onnx

        m = onnx.load(a.model, load_external_data=False)
        dims = {i.name: list(i.dims) for i in m.graph.initializer}  # TensorProto.dims is a plain int list
        vocab = max((max(d) for d in dims.values() if d), default=0)
        consumers = {}
        for n in m.graph.node:
            for i in n.input:
                consumers.setdefault(i, []).append(n)
        # The head weight is usually feeding a Transpose that feeds the MatMul, not the MatMul directly.
        for w, d in dims.items():
            if vocab not in d:
                continue
            frontier = [n for n in consumers.get(w, []) if n.op_type not in ("Gather",)]
            for _ in range(2):
                found = [n.name for n in frontier if n.op_type in ("MatMul", "Gemm")]
                exclude += found
                frontier = [c for n in frontier for o in n.output for c in consumers.get(o, [])]
        exclude = sorted(set(exclude))
        print(f"keeping {len(exclude)} vocab-sized projection(s) float (vocab={vocab})")

    wt = QuantType.QInt8 if a.weight_type == "qint8" else QuantType.QUInt8
    # MatMulConstBOnly keeps the projection weights quantised and leaves attention's activation-by-
    # activation matmuls (QK^T, scores.V) in float, which is the one thing that must not be touched.
    quantize_dynamic(
        a.model,
        a.out,
        weight_type=wt,
        per_channel=not a.no_per_channel,
        extra_options={"MatMulConstBOnly": not a.all_matmuls},
        nodes_to_exclude=exclude,
        # 2.4 GB fp32 -> well under the 2 GB protobuf limit after int8, so no sidecar is needed
        use_external_data_format=False,
    )
    mb = os.path.getsize(a.out) / 1e6
    print(
        f"wrote {a.out} ({mb:.0f} MB, {a.weight_type}, per_channel={not a.no_per_channel}, "
        f"const_weight_matmuls_only={not a.all_matmuls})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
