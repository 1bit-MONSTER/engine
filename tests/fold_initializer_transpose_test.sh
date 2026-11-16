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

# engine#2779: one weight feeding TWO Transposes with different perms used to fold to a single `{src}_t`, so both
# consumers read the first perm's tensor while the graph declared the second one's shape. `--check` cannot see it
# (no Transpose is left), so the pack built green and was invalid.
python3 - "$tmp/two.onnx" "$tmp/square.onnx" <<'PY' || { echo "FAIL could not build the two-perm fixtures"; exit 1; }
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

def two_perm(path, shape, perms):
    w = numpy_helper.from_array(np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape), "w")
    nodes = [helper.make_node("Transpose", ["w"], [f"t{i}"], perm=p) for i, p in enumerate(perms)]
    nodes += [helper.make_node("Identity", ["t0"], ["y0"]), helper.make_node("Identity", ["t1"], ["y1"])]
    outs = [helper.make_tensor_value_info("y0", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("y1", TensorProto.FLOAT, None)]
    onnx.save(helper.make_model(helper.make_graph(nodes, "g", [], outs, [w]),
                                opset_imports=[helper.make_opsetid("", 17)]), path)

# Two different orientations of a NON-square weight: (2,3,4) -> (2,4,3) and -> (4,3,2).
two_perm(sys.argv[1], (2, 3, 4), [[0, 2, 1], [2, 1, 0]])
# ... and of a SQUARE one: (3,3,3) -> (3,3,3) both ways, so a shape-based discriminator cannot tell them apart.
two_perm(sys.argv[2], (3, 3, 3), [[1, 0, 2], [0, 2, 1]])
PY

for m in two square; do
  python3 "$TOOL" "$tmp/$m.onnx" "$tmp/$m.out.onnx" >"$tmp/$m.log" 2>&1 || { cat "$tmp/$m.log"; fail=$((fail+1)); }
  check "$m: two different perms fold to two weights" "$(grep -c '^folded ' "$tmp/$m.log")" 2
  read -r same shapes values <<EOF
$(python3 - "$tmp/$m.out.onnx" <<'PY'
import sys
import numpy as np
import onnx
from onnx import numpy_helper
after = onnx.load(sys.argv[1])
inits = {i.name: numpy_helper.to_array(i) for i in after.graph.initializer}
reads = [n.input[0] for n in after.graph.node if n.op_type == "Identity"]
w0, w1 = inits[reads[0]], inits[reads[1]]
print(int(reads[0] == reads[1]),
      "x".join(str(d) for d in w0.shape) + "/" + "x".join(str(d) for d in w1.shape),
      int(np.array_equal(w0, w1)))
PY
)
EOF
  check "$m: the two consumers read DIFFERENT initializers" "$same" "0"
  check "$m: and the two tensors differ" "$values" "0"
  case "$m" in
    two)    check "two-perm shapes are the two orientations" "$shapes" "2x4x3/4x3x2" ;;
    square) check "square weight: same shape, still two distinct tensors" "$shapes" "3x3x3/3x3x3" ;;
  esac
done

echo "---"
if [ "$fail" -ne 0 ]; then echo "$fail failed"; exit 1; fi
echo "fold_initializer_transpose: all checks ok"
