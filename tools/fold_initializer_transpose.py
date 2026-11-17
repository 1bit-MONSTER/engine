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

import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper


def initializer_transposes(model):
    """Transpose nodes whose input is a graph initializer: the only ones a weight rewrite can remove."""
    inits = {i.name for i in model.graph.initializer}
    return [n for n in model.graph.node if n.op_type == "Transpose" and n.input[0] in inits]


def folded_name(src, perm, want, inits, taken):
    """The initializer a fold of `src` under `perm` is stored as, created if it is not there yet.

    engine#2779: one weight folded under TWO different perms must not collapse onto one tensor. The first fold of a
    source keeps the plain `<src>_t` name (so every existing pack and every printed name is unchanged), and each further
    DISTINCT perm gets the perm in its name, which makes the collision impossible rather than unlikely. A repeated
    identical perm still reuses the one tensor, which is the case the old comment was written for.

    A name is only reused when it already holds EXACTLY this perm's values: an unrelated `<src>_t` in the source graph
    (or one of a different shape) is a collision too, and rewiring consumers onto it is the same silent corruption in a
    different disguise, so a free name is picked instead of trusting the name.
    """
    candidates = [f"{src}_t", f"{src}_t_" + "_".join(str(d) for d in perm)]
    i = 2
    while len(candidates) < 8:
        candidates.append(f"{src}_t_{i}")
        i += 1
    for name in candidates:
        existing = inits.get(name)
        if existing is None:
            return name
        have = numpy_helper.to_array(existing)
        if have.shape == want.shape and np.array_equal(have, want):
            return name
    raise SystemExit(f"{src}: no free initializer name for perm {perm} near {src}_t (engine#2779)")


def provenance_mismatches(model):
    """Folded tensors whose shape is not `source dims` permuted by the recorded perm. Returns one line each.

    Only initializers the fold produced (their `doc_string` starts `folded:`) are checked, so an unpacked graph answers
    with an empty list rather than guessing. A missing source is reported too: the provenance says where the tensor came
    from, and a source that is no longer in the graph means the pack cannot be verified, which is not the same as fine.
    """
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    bad = []
    for t in model.graph.initializer:
        if not t.doc_string.startswith("folded:"):
            continue
        _, src, perm_s = t.doc_string.split(":")
        perm = [int(d) for d in perm_s.split(",")]
        if src not in inits:
            bad.append(f"{t.name}: folded from {src}, which is not in this graph: cannot verify (not a pass)")
            continue
        want = list(np.transpose(inits[src], perm).shape)
        have = list(inits[t.name].shape)
        if want != have:
            bad.append(f"{t.name}: folded from {src} perm {perm_s} should be {want}, is {have}")
    return bad


def fold(model):
    """Rewrite every initializer-fed Transpose in place. Returns [(src, dst, shape)] for what it folded."""
    folded = []
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    # (src, perm) -> the tensor that perm was folded into, and every name this run has already produced.
    by_perm = {}
    taken = set(inits)
    for node in initializer_transposes(model):
        src = node.input[0]
        perm = [a.ints for a in node.attribute if a.name == "perm"]
        if not perm:
            raise SystemExit(f"{src}: Transpose without a perm attribute cannot be folded")
        array = numpy_helper.to_array(inits[src])
        want = np.transpose(array, perm[0])
        key = (src, tuple(perm[0]))
        if key in by_perm:
            dst = by_perm[key]
        else:
            dst = folded_name(src, perm[0], want, inits, taken)
            if dst not in inits:
                new = numpy_helper.from_array(np.ascontiguousarray(want), dst)
                # engine#2779 review: after folding, the perm is gone from the graph, so `--check` could only prove the
                # NODE was removed -- never that the weight is the right orientation. The provenance makes the check
                # able to fail: `--check` re-derives the permuted shape from the recorded perm and the source dims.
                new.doc_string = f"folded:{src}:{','.join(str(d) for d in perm[0])}"
                graph.initializer.append(new)
                inits[dst] = new
                taken.add(dst)
            by_perm[key] = dst
        for consumer in graph.node:
            for i, name in enumerate(consumer.input):
                if name == node.output[0]:
                    consumer.input[i] = dst
        folded.append((src, dst, list(want.shape)))
    if folded:
        gone = {f[1] for f in folded}
        keep = [n for n in graph.node if not (n.op_type == "Transpose" and n.input[0] in inits
                                              and f"{n.input[0]}_t" in gone)]
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
    # Whether the SOURCE kept its weights external has to be read before `onnx.load()` pulls them into memory: after
    # that the initializers carry no `data_location`, so the answer is gone (engine#2769 review).
    was_external = any(i.data_location == onnx.TensorProto.EXTERNAL
                       for i in onnx.load(path, load_external_data=False).graph.initializer)
    model = onnx.load(path)
    before = len(model.graph.node)
    if check:
        bad = initializer_transposes(model)
        for n in bad:
            print(f"{n.name or n.output[0]}: Transpose of initializer {n.input[0]} is still in the graph")
        # A folded tensor records its source and perm, so the check can assert the ORIENTATION, not just the absence of
        # the node (engine#2779 review). A fold with no provenance beside a `<src>_t` name is reported, not ignored.
        wrong = provenance_mismatches(model)
        for line in wrong:
            print(line)
        inits = {i.name: i for i in model.graph.initializer}
        folded_marks = [i.name for i in model.graph.initializer if i.doc_string.startswith("folded:")]
        print(f"{path}: nodes {before}, initializer-fed Transposes {len(bad)}, "
              f"folded with provenance {len(folded_marks)}, orientation mismatches {len(wrong)}")
        return 1 if bad or wrong else 0
    folded = fold(model)
    out = args[1] if len(args) > 1 else path
    # engine#2769 review: the pack's layout is the export's, and `onnx.save()` inlines every external tensor by default
    # -- a 159 MB weight lands inside the graph and the `.onnx.data` beside it becomes dead weight. If the source graph
    # kept its weights external, keep them external, one sidecar next to the output (same shape as the input's).
    if was_external:
        onnx.save(model, out, save_as_external_data=True, all_tensors_to_one_file=True,
                  location=os.path.basename(out) + ".data", size_threshold=1024)
    else:
        onnx.save(model, out)
    for src, dst, shape in folded:
        print(f"folded {src} -> {dst} {shape}")
    print(f"{out}: nodes {before} -> {len(model.graph.node)}, folded {len(folded)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
