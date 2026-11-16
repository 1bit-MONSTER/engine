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
#
# fold_initializer_transpose: store a weight in the orientation the matmul wants, so the Transpose node goes away
# (engine#2366).
#
# A packed whisper export reaches the lm head through `Transpose(val_452) -> MatMul`. The transpose is of a
# 768x51865 initializer, 159 MB, and it runs on every window: `[audio-node] 785 Transpose 768x51865 = 101.258 ms`
# beside `786 MatMul 1x128x51865 = 193.072 ms`, 26 executions per transcription, ~2.6 s of a 3.5 s wall clock.
# Nothing is computed by it -- the weight does not change between windows -- so the fix is to store it already
# transposed and drop the node (engine#2366: proven twice that a runtime cache in audio_exec cannot help; the
# cross-execution cache copies the 159 MB per pass and regressed the encoder to 10.36 s).
#
# `--check` exits 1 when an initializer-fed Transpose survives, so a pack build can refuse the fold-less file
# instead of silently paying the 2.6 s again.

import sys

import numpy as np
import onnx
from onnx import numpy_helper


def initializer_transposes(model):
    """Transpose nodes whose input is a graph initializer: the only ones a weight rewrite can remove."""
    inits = {i.name for i in model.graph.initializer}
    return [n for n in model.graph.node if n.op_type == "Transpose" and n.input[0] in inits]


def fold(model):
    """Rewrite every initializer-fed Transpose in place. Returns [(src, dst, shape)] for what it folded."""
    folded = []
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    by_perm = {}
    folded_nodes = []
    for node in initializer_transposes(model):
        src = node.input[0]
        perm = [a.ints for a in node.attribute if a.name == "perm"]
        if not perm:
            raise SystemExit(f"{src}: Transpose without a perm attribute cannot be folded")
        key = (src, tuple(perm[0]))
        dst = by_perm.get(key)
        # A second reference to the same weight in the SAME orientation folds once and reuses the name. A different
        # orientation needs an initializer of its own: one name per weight, keyed by the perm, used to give both
        # consumers the first perm's tensor while the graph declared the second one's shape (engine#2779) -- and
        # `--check` could not see it, because no Transpose is left. The key is the PERM, not the shape: on a square
        # weight the two orientations have the same shape, so a shape test would still collapse them, and that is
        # the case that validates and computes wrong values with no symptom.
        if dst is None:
            taken = any(k[0] == src for k in by_perm)
            dst = f"{src}_t" if not taken else f"{src}_t_" + "".join(str(p) for p in perm[0])
            want = np.transpose(numpy_helper.to_array(inits[src]), perm[0])
            new = numpy_helper.from_array(np.ascontiguousarray(want), dst)
            graph.initializer.append(new)
            inits[dst] = new
            by_perm[key] = dst
            folded.append((src, dst, list(want.shape)))
        for consumer in graph.node:
            for i, name in enumerate(consumer.input):
                if name == node.output[0]:
                    consumer.input[i] = dst
        folded_nodes.append(node)
    if folded_nodes:
        keep = [n for n in graph.node if not any(n is f for f in folded_nodes)]
        del graph.node[:]
        graph.node.extend(keep)
        # An initializer the fold replaced and nothing else reads would otherwise be packed into the container.
        used = {name for n in graph.node for name in n.input}
        used |= {o.name for o in graph.output}
        keepi = [i for i in graph.initializer if i.name in used or i.name not in {f[0] for f in folded}]
        if len(keepi) != len(graph.initializer):
            del graph.initializer[:]
            graph.initializer.extend(keepi)
    return folded


def main(argv):
    if len(argv) < 2:
        print("usage: fold_initializer_transpose.py [--check] <model.onnx> [out.onnx]", file=sys.stderr)
        return 2
    check = argv[1] == "--check"
    args = argv[2:] if check else argv[1:]
    path = args[0]
    model = onnx.load(path)
    before = len(model.graph.node)
    if check:
        bad = initializer_transposes(model)
        for n in bad:
            print(f"{n.name or n.output[0]}: Transpose of initializer {n.input[0]} is still in the graph")
        print(f"{path}: nodes {before}, initializer-fed Transposes {len(bad)}")
        return 1 if bad else 0
    folded = fold(model)
    out = args[1] if len(args) > 1 else path
    onnx.save(model, out)
    for src, dst, shape in folded:
        print(f"folded {src} -> {dst} {shape}")
    print(f"{out}: nodes {before} -> {len(model.graph.node)}, folded {len(folded)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
