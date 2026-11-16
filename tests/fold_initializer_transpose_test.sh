#!/usr/bin/env bash
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
# fold_initializer_transpose_test: the fold must remove the node, rewire the consumer, and refuse a graph that
# still has one (engine#2366).
#
# The case that matters is the whisper head: the lm head is weight-tied to `token_embedding.weight`, so the same
# 159 MB initializer feeds both an embedding gather (untransposed) and `Transpose -> MatMul`. The fold must add
# the transposed copy for the matmul and KEEP the original for the gather.

set -uo pipefail

# The tool needs `onnx`; the CI image does not install it, so this is a SKIP there rather than a red build
# (the same 77 the C++ device tests use). Everything the fold does is still checked wherever onnx is present.
python3 -c "import onnx, numpy" 2>/dev/null || { echo "SKIP: python onnx module not installed"; exit 77; }

TOOL=${1:-$(dirname "$0")/../tools/fold_initializer_transpose.py}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/foldtest.XXXXXX") || exit 1
trap 'rm -rf "$tmp"' EXIT
fail=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: want '$3' got '$2'"; fail=$((fail+1)); fi; }

python3 - "$tmp/m.onnx" <<'PY' || { echo "FAIL could not build the fixture"; exit 1; }
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

w = numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(4, 3), "w")     # 4x3, embedded and transposed
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
g = helper.make_graph(
    [helper.make_node("Transpose", ["w"], ["wt"], perm=[1, 0]),
     helper.make_node("MatMul", ["x", "wt"], ["y"])],
    "g", [x], [y], [w])
# The gather path is what makes the original initializer survive the fold.
g.output.append(helper.make_tensor_value_info("e", TensorProto.FLOAT, None))
g.node.append(helper.make_node("Gather", ["w"], ["e"], axis=0))
onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), sys.argv[1])
PY

rc=0; python3 "$TOOL" --check "$tmp/m.onnx" >/dev/null 2>&1 || rc=$?
check "check refuses a graph with an initializer-fed Transpose" "$rc" 1

python3 "$TOOL" "$tmp/m.onnx" "$tmp/out.onnx" >"$tmp/log" 2>&1 || { cat "$tmp/log"; fail=$((fail+1)); }
check "fold reports one folded weight" "$(grep -c '^folded ' "$tmp/log")" 1

read -r nodes wt_used w_kept shape <<EOF
$(python3 - "$tmp/out.onnx" <<'PY'
import sys
import onnx
m = onnx.load(sys.argv[1])
inits = {i.name: i for i in m.graph.initializer}
mm = [n for n in m.graph.node if n.op_type == "MatMul"][0]
print(len(m.graph.node), mm.input[1], "w" in inits,
      "x".join(str(d) for d in inits.get(mm.input[1]).dims) if mm.input[1] in inits else "-")
PY
)
EOF
check "the Transpose node is gone" "$nodes" 2      # Transpose + MatMul + Gather -> MatMul + Gather
check "the matmul reads the transposed initializer" "$wt_used" "w_t"
check "the gather's untransposed copy is kept" "$w_kept" "True"
check "the folded weight has the matmul orientation" "$shape" "3x4"

rc=0; python3 "$TOOL" --check "$tmp/out.onnx" >/dev/null 2>&1 || rc=$?
check "check accepts the folded graph" "$rc" 0

# engine#2779: one initializer folded two ways. Naming the folded weight after the SOURCE alone gave both consumers the
# first perm's array. The graph keeps no Transpose, so --check cannot see it, and a square weight would pack and compute
# wrong values with no symptom at all. The two consumers must end up on two initializers of their own shapes.
python3 - "$tmp/two.onnx" <<'FIXTURE'
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

w = numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), "w")      # 2x3
g = helper.make_graph(
    [helper.make_node("Transpose", ["w"], ["a"], perm=[1, 0]),                     # 3x2
     helper.make_node("Transpose", ["w"], ["b"], perm=[0, 1]),                     # 2x3
     helper.make_node("Identity", ["a"], ["out_a"]),
     helper.make_node("Identity", ["b"], ["out_b"])],
    "two", [], [helper.make_tensor_value_info("out_a", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("out_b", TensorProto.FLOAT, None)], [w])
onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), sys.argv[1])
FIXTURE

python3 "$TOOL" "$tmp/two.onnx" "$tmp/two_out.onnx" >"$tmp/two.log" 2>&1 || { cat "$tmp/two.log"; fail=$((fail+1)); }
check "a two-perm decoder folds two weights" "$(grep -c '^folded ' "$tmp/two.log")" 2

python3 - "$tmp/two_out.onnx" >"$tmp/two.txt" <<'PROBE'
import sys
import numpy as np
import onnx
from onnx import numpy_helper
m = onnx.load(sys.argv[1])
inits = {i.name: list(numpy_helper.to_array(i).shape) for i in m.graph.initializer}
ids = [n for n in m.graph.node if n.op_type == "Identity"]
a, b = inits.get(ids[0].input[0], []), inits.get(ids[1].input[0], [])
print("x".join(str(d) for d in a), "x".join(str(d) for d in b), str(ids[0].input[0] != ids[1].input[0]))
PROBE
read -r a_shape b_shape distinct <"$tmp/two.txt"
check "the [1,0] consumer reads a 3x2 weight" "$a_shape" "3x2"
check "the [0,1] consumer reads a 2x3 weight" "$b_shape" "2x3"
check "the two consumers read different initializers" "$distinct" "True"

rc=0; python3 "$TOOL" --check "$tmp/two_out.onnx" >/dev/null 2>&1 || rc=$?
check "check accepts the two-perm folded graph" "$rc" 0

echo "---"
if [ "$fail" -ne 0 ]; then echo "$fail failed"; exit 1; fi
echo "fold_initializer_transpose: all checks ok"
