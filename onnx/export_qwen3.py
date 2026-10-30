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

"""Export a Qwen3 causal LM to a plain ONNX graph that ONNX Runtime can run.

    python3 onnx/export_qwen3.py --model-dir ~/models/Qwen3-0.6B-hf --out ~/models/qwen3-0.6b.onnx

The point is a graph ONNXRuntime itself schedules: no custom-domain operators, no fused
attention, no KV cache — MatMul/Add/Softmax/LayerNormalization/Gather/Slice/Concat and nothing
else, every matmul a plain float32 MatMul over an initializer so the plugin EP can take it.

Interface (what onnx/ort_run.cpp drives):
    input_ids      int64   [1, seq]
    attention_mask int64   [1, seq]
    logits         float32 [1, seq, vocab]

One decoding step is a full re-run over the growing sequence, because this graph keeps no cache.
That is deliberate: it is the honest shape for comparing a scheduler against the forward route,
and a cached graph would be a different (and much bigger) piece of work. It also means tokens/s
here is not comparable to a KV-cached server and is reported as this graph's rate.

`--check` prints the operator census and every distinct MatMul (K, N) in the graph, which is what
the state file and the Ryzen AI pack note are written from.
"""

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3 -> plain ONNX")
    ap.add_argument("--model-dir", required=True, help="HF directory with config.json + weights")
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--seq-len", type=int, default=8, help="trace length (axis is dynamic)")
    # torch's torchscript exporter tops out at opset 20; asking for 21 only warns and exports 20.
    ap.add_argument("--opset", type=int, default=20)
    ap.add_argument("--check", action="store_true", help="census the exported graph")
    ap.add_argument(
        "--keep-spill", action="store_true", help="leave torch's per-tensor weight files in place"
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    # eager attention: sdpa traces into ops that are not plain MatMul and would hide the matmuls
    # from the EP's support query.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    class Logits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits

    wrapper = Logits(model).eval()
    ids = torch.ones((1, args.seq_len), dtype=torch.long)
    mask = torch.ones((1, args.seq_len), dtype=torch.long)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (ids, mask),
            args.out,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {1: "seq"},
                "attention_mask": {1: "seq"},
                "logits": {1: "seq"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
            # A 0.6B fp32 model is 3 GB of weights; protobuf caps a single file at 2 GB, and the
            # per-tensor spill files torch writes otherwise have no stable name to ship.
            external_data=True,
        )
    # torch spills every large initializer into its own file next to the model (one per tensor,
    # named after it). Repack into a single sidecar: one file to copy, one file ORT loads, and the
    # per-tensor names never leak into the artifact.
    import onnx

    proto = onnx.load(args.out, load_external_data=False)
    spills = {
        e.value
        for t in proto.graph.initializer
        if t.data_location == onnx.TensorProto.EXTERNAL
        for e in t.external_data
        if e.key == "location"
    }
    sidecar = os.path.basename(args.out) + ".data"
    if spills or not args.keep_spill:
        model = onnx.load(args.out)
        onnx.save_model(
            model,
            args.out,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=sidecar,
            size_threshold=1024,
            convert_attribute=False,
        )
        for name in spills:
            path = os.path.join(os.path.dirname(os.path.abspath(args.out)), name)
            if os.path.isfile(path) and name != sidecar:
                os.unlink(path)

    size = sum(
        os.path.getsize(p)
        for p in (args.out, args.out + ".data", args.out + "_data")
        if os.path.exists(p)
    )
    print(f"wrote {args.out} ({size / 1e9:.2f} GB)")

    if not args.check:
        return 0

    from collections import Counter

    graph = onnx.load(args.out, load_external_data=False).graph
    inits = {i.name: i for i in graph.initializer}
    census = Counter(n.op_type for n in graph.node)
    print("\noperators:")
    for op, n in census.most_common():
        print(f"  {n:6d}  {op}")

    shapes = Counter()
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        w = inits.get(node.input[1])
        if w is None or len(w.dims) != 2:
            shapes[("?", "?")] += 1
            continue
        shapes[(int(w.dims[0]), int(w.dims[1]))] += 1
    print("\nMatMul (K, N) x count:")
    for (k, n), count in sorted(shapes.items(), key=lambda kv: -kv[1]):
        print(f"  x{count:<4d} K={k} N={n}")
    doms = {n.domain for n in graph.node if n.domain}
    print(f"\ndomains: {sorted(doms) if doms else ['(ai.onnx only)']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
