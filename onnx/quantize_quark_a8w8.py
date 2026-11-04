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

"""Quantise an ONNX model to A8W8 QDQ with AMD's Quark ONNX flow.

This is the export the plugin EP's fused path consumes: every projection becomes

    x_q --DequantizeLinear--> x_dq \
                                    MatMul --> y --QuantizeLinear--> y_q
    w_q --DequantizeLinear--> w_dq /

with int8 activations, int8 weights, float32 scales and (with Quark's A8W8 config) zero points of 0.
`docs/onnx-pack.md` describes what the EP does with it; `~/briefs/TQDQ-STATE.md` holds the measured
numbers for the run this script produced.

Quark is not on PyPI as a supported wheel for every platform: use the source tree
(https://github.com/amd/Quark, the `release/0.13` branch) on PYTHONPATH. Its ONNX path needs
python >= 3.11,< 3.14, onnx <= 1.22, onnxruntime, torch (CPU is enough) and the requirements in its
own requirements.txt. Quark generates `quark/version.py` at install time, so a plain source checkout
needs that file created by hand (`__version__ = '0.13'`).

Calibration data is a `.npz` with `input_ids` and `attention_mask` (int64, [samples, seqlen]); the
model's own tokenizer and any local text are enough, no dataset download:

    python3 -c "import numpy as np,glob \\
        from transformers import AutoTokenizer as T; tok=T.from_pretrained('<hf dir>') \\
        ids=tok('\\n'.join(open(p).read() for p in glob.glob('docs/*.md')), \\
                return_tensors='np')['input_ids'][0] \\
        x=np.stack([ids[i*512:(i+1)*512] for i in range(32)]).astype(np.int64) \\
        np.savez('calib.npz', input_ids=x, attention_mask=np.ones_like(x))"

    python3 onnx/quantize_quark_a8w8.py --model qwen3-0.6b.onnx --out qwen3-0.6b-a8w8.onnx \\
        --calib calib.npz

The second command also prints what the plugin EP's matcher will be able to claim: the number of
`DequantizeLinear -> MatMul -> QuantizeLinear` groups whose weight is a constant int8 initializer
(the projections) versus the matmuls whose other operand is an activation (attention's QK^T, which
no int8 x int8 kernel can take and which stay on the CPU EP).
"""

import argparse
import copy
import os
import sys

import numpy as np
import onnx


class NpzCalibrationDataReader:
    """Quark's calibration interface: one dict of graph inputs per call, None when done."""

    def __init__(self, npz_path: str, input_names: list) -> None:
        data = np.load(npz_path)
        self.inputs = [data[name] for name in input_names]
        self.count = self.inputs[0].shape[0]
        self.index = 0
        self.input_names = input_names

    def get_next(self):
        if self.index >= self.count:
            return None
        row = {name: arr[self.index : self.index + 1] for name, arr in zip(self.input_names, self.inputs)}
        self.index += 1
        return row

    def __len__(self) -> int:
        return self.count


def shared_concat_outputs(model) -> list:
    """Concat outputs read by more than one node, with the reader count.

    Quark's A8W8 refinement runs `align_quantize_info(..., align_concat=True)` (the config default) and
    that step asserts every Concat output has exactly one reader
    (`quark/onnx/utils/model_utils.py:_find_node_output_qdq`:
    `assert len(self.in_name_to_nodes[tensor_name]) == 1`). An eager cache-less transformer export
    shares two of them across the whole stack: the rotary frequency concat feeds `Cos` and `Sin`, and
    the additive attention mask feeds one `Expand` per attention block -- 56 readers on Qwen3-0.6B.
    Any such graph dies in refinement with a bare `AssertionError` and no node named, before a single
    weight is written.

    Splitting the tensor with an `Identity` does not survive: `onnxruntime.quantization` removes
    Identity nodes and renames the producer's output, so the share reappears (measured -- the
    assertion then names the Identity's own output, with all 56 readers). The supported escape is the
    step's own knob, `extra_options['AlignConcat']`; this function is what decides when to use it.
    """
    readers = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp:
                readers[inp] = readers.get(inp, 0) + 1
    return [
        (out, readers[out])
        for node in model.graph.node
        if node.op_type == "Concat"
        for out in node.output
        if readers.get(out, 0) > 1
    ]


def count_qdq_groups(path: str) -> int:
    """MatMuls whose two operands are DequantizeLinear outputs and whose weight side is a constant.

    A group the plugin EP can fuse needs exactly that: both operands dequantised (so both are int8
    behind a DQ), and a weight that is an initializer (an activation-times-activation matmul cannot
    be folded into an int8 x int8 kernel).
    """
    model = onnx.load(path, load_external_data=False)
    producer = {}
    for node in model.graph.node:
        for out in node.output:
            producer[out] = node
    initializers = {i.name for i in model.graph.initializer}

    def dq_source(value):
        """The int8 value a DequantizeLinear dequantises, or None."""
        node = producer.get(value)
        if node is None or node.op_type != "DequantizeLinear":
            return None
        return node.input[0]

    groups = 0
    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) != 2:
            continue
        x_q = dq_source(node.input[0])
        w_q = dq_source(node.input[1])
        if x_q is None or w_q is None:
            continue
        if w_q in initializers and x_q not in initializers:
            groups += 1
    return groups


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="float ONNX model to quantise")
    ap.add_argument("--out", required=True, help="A8W8 QDQ model to write")
    ap.add_argument("--calib", required=True, help=".npz with input_ids and attention_mask")
    ap.add_argument("--config", default="A8W8", help="Quark ONNX config name (default A8W8)")
    ap.add_argument("--seq-input", default="input_ids", help="first graph input's name")
    ap.add_argument("--op-types", default="MatMul", help="comma separated ops to quantise")
    ap.add_argument(
        "--per-tensor-weights",
        action="store_true",
        help="one weight scale for the whole tensor. The default is per-channel (one scale per output "
        "column), which is what the fused epilogue folds as s_w[n] and what a transformer needs: with a "
        "single int8 scale per weight matrix the 0.6B model answered gibberish",
    )
    ap.add_argument(
        "--matmul-only",
        action="store_true",
        help="exclude every node that is not a MatMul from being quantised as an op. Measured on a 0.6B "
        "export: the 196 projection MatMuls get an activation Q/DQ, the 57 MatMuls without a constant "
        "weight (attention's QK^T and AV, the rotary one) get none, and Quark still puts a Q/DQ pair on "
        "the tensors those projections consume -- after 84 Mul and 28 Reshape nodes, all of which feed a "
        "MatMul. That is the int8 activation side, not a non-MatMul op being quantised; without this flag "
        "Quark also quantises the masking/softmax path, which with a small calibration set is what wrecks "
        "the model",
    )
    args = ap.parse_args()

    try:
        from quark.onnx import Config, ModelQuantizer
        from quark.onnx.quantization.config import get_default_config
    except ImportError as exc:  # pragma: no cover - environment problem, not a bug here
        print(f"quantize_quark_a8w8: FAIL cannot import Quark: {exc}", file=sys.stderr)
        print("  put the Quark source tree on PYTHONPATH (see this file's docstring)", file=sys.stderr)
        return 1

    model = onnx.load(args.model, load_external_data=False)
    input_names = [i.name for i in model.graph.input]
    # The calibration file carries the graph's real inputs; anything else is not something a
    # calibration reader can invent.
    missing = [n for n in input_names if n not in np.load(args.calib).files]
    if missing:
        print(f"quantize_quark_a8w8: FAIL calibration is missing {missing}", file=sys.stderr)
        return 1

    dataloader = NpzCalibrationDataReader(args.calib, input_names)
    quant_config = copy.deepcopy(get_default_config(args.config))
    quant_config.use_external_data_format = True  # 0.6B of int8 is past the 2 GB proto limit soon
    quant_config.include_cle = False
    quant_config.op_types_to_quantize = args.op_types.split(",")
    quant_config.per_channel = not args.per_tensor_weights
    if args.matmul_only:
        # Projection matmuls are the ones with a constant weight: x @ W. Attention's QK^T and AV have
        # an activation on both sides, and quantising them drags the rotary/mask/softmax path into
        # int8, which is a different (and much less forgiving) experiment.
        initializers = {i.name for i in model.graph.initializer}
        excluded = [
            n.name
            for n in model.graph.node
            if not (n.op_type == "MatMul" and len(n.input) == 2 and n.input[1] in initializers)
        ]
        quant_config.nodes_to_exclude = excluded
        print(f"quantize_quark_a8w8: excluding {len(excluded)} nodes, quantising "
              f"{len(model.graph.node) - len(excluded)} projection matmuls")
    # Quark's align_concat step asserts one reader per Concat output, which an eager cache-less export
    # violates by construction (see shared_concat_outputs). The step only rewrites the quantisation
    # parameters of Concats that carry Q/DQ on both sides, so it is a no-op on a graph where no Concat is
    # quantised -- which is every graph this flag combination exports -- and on any other graph skipping
    # it costs DPU alignment, never numerics.
    shared = shared_concat_outputs(model)
    if shared:
        for tensor, count in shared:
            print(f"quantize_quark_a8w8: Concat output {tensor} has {count} readers, so align_concat would "
                  f"assert -> AlignConcat off for this graph")
        quant_config.extra_options["AlignConcat"] = False
        excluded_names = set(quant_config.nodes_to_exclude or [])
        quantised_concats = [n.name for n in model.graph.node
                             if n.op_type == "Concat" and n.name not in excluded_names]
        print(f"  Concat nodes that would be quantised anyway: {len(quantised_concats)}")

    config = Config(global_quant_config=quant_config)
    print(f"quantize_quark_a8w8: config {args.config}")
    print(f"  activation={quant_config.activation_type} weight={quant_config.weight_type} "
          f"calibrate={quant_config.calibrate_method} op_types={quant_config.op_types_to_quantize}")
    print(f"  calibration: {len(dataloader)} samples, {dataloader.inputs[0].shape[1]} tokens each")

    outdir = os.path.dirname(args.out)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    ModelQuantizer(config).quantize_model(args.model, args.out, dataloader)

    # What the EP will find in the result. The count is the *prediction*; running it through
    # `ort-run --ep libonebit_ep.so` is the proof (the EP logs `qdq_groups=` per graph).
    print(f"quantize_quark_a8w8: wrote {args.out}")
    print(f"  fuseable DQ->MatMul->Q groups (constant int8 weight): {count_qdq_groups(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
