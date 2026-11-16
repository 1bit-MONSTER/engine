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

# engine#2779: ONE weight folded under TWO DIFFERENT perms. The fold used to name the result after the source only, so
# the second perm reused the first one's tensor and rewired its consumers to the wrong orientation while --check passed.
python3 - "$tmp/two.onnx" <<'PY2' || { echo "FAIL could not build the two-perm fixture"; exit 1; }
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
w = numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(2, 3, 2), "w")     # 2x3x2: two perms differ in shape
g = helper.make_graph(
    [helper.make_node("Transpose", ["w"], ["a"], perm=[1, 0, 2]),
     helper.make_node("Transpose", ["w"], ["b"], perm=[0, 2, 1]),
     helper.make_node("Identity", ["a"], ["oa"]),
     helper.make_node("Identity", ["b"], ["ob"])],
    "g",
    [],
    [helper.make_tensor_value_info("oa", TensorProto.FLOAT, None),
     helper.make_tensor_value_info("ob", TensorProto.FLOAT, None)],
    [w])
onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), sys.argv[1])
PY2

python3 "$TOOL" "$tmp/two.onnx" "$tmp/two-out.onnx" >"$tmp/two.log" 2>&1 || { cat "$tmp/two.log"; fail=$((fail+1)); }
check "two perms fold into two weights" "$(grep -c '^folded ' "$tmp/two.log")" 2
read -r distinct d0 d1 ok_a ok_b <<EOF
$(python3 - "$tmp/two-out.onnx" <<'PY2'
import sys
import numpy as np
import onnx
from onnx import numpy_helper
m = onnx.load(sys.argv[1])
inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
ident = {n.output[0]: n.input[0] for n in m.graph.node if n.op_type == "Identity"}
a, b = ident["oa"], ident["ob"]
w = np.arange(12, dtype=np.float32).reshape(2, 3, 2)     # the source is dropped when nothing else reads it
print(a != b,
      "x".join(str(d) for d in inits[a].shape), "x".join(str(d) for d in inits[b].shape),
      np.array_equal(inits[a], np.transpose(w, (1, 0, 2))),
      np.array_equal(inits[b], np.transpose(w, (0, 2, 1))))
PY2
)
EOF
check "each perm gets its own name" "$distinct" "True"
check "perm [1,0,2] tensor has shape" "$d0" "3x2x2"
check "perm [0,2,1] tensor has shape" "$d1" "2x2x3"
check "consumer a reads perm [1,0,2]'s values" "$ok_a" "True"
check "consumer b reads perm [0,2,1]'s values" "$ok_b" "True"

# ...and the case with NO symptom at all: a square weight, where both perms have the same shape and only the values
# differ. This is the arm that would pass silently on a shape-only check.
python3 - "$tmp/sq.onnx" <<'PY2' || { echo "FAIL could not build the square fixture"; exit 1; }
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
w = numpy_helper.from_array(np.arange(4, dtype=np.float32).reshape(2, 2), "w")        # [[0,1],[2,3]]
g = helper.make_graph(
    [helper.make_node("Transpose", ["w"], ["a"], perm=[0, 1]),
     helper.make_node("Transpose", ["w"], ["b"], perm=[1, 0]),
     helper.make_node("Identity", ["a"], ["oa"]),
     helper.make_node("Identity", ["b"], ["ob"])],
    "g",
    [],
    [helper.make_tensor_value_info("oa", TensorProto.FLOAT, None),
     helper.make_tensor_value_info("ob", TensorProto.FLOAT, None)],
    [w])
onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), sys.argv[1])
PY2

python3 "$TOOL" "$tmp/sq.onnx" "$tmp/sq-out.onnx" >"$tmp/sq.log" 2>&1 || { cat "$tmp/sq.log"; fail=$((fail+1)); }
same_shape_dir=$(python3 - "$tmp/sq-out.onnx" <<'PY2'
import sys
import numpy as np
import onnx
from onnx import numpy_helper
m = onnx.load(sys.argv[1])
inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
ident = {n.output[0]: n.input[0] for n in m.graph.node if n.op_type == "Identity"}
a, b = ident["oa"], ident["ob"]
print("different" if not np.array_equal(inits[a], inits[b]) else "same")
PY2
)
check "square weight: the two consumers still read different values" "$same_shape_dir" "different"

# ...and the collision in the other direction: the SOURCE graph already carries an initializer called `w_t` that has
# nothing to do with the fold. Reusing it because the name matches would be the same silent corruption on a graph that
# never had two perms at all.
python3 - "$tmp/taken.onnx" <<'PY2' || { echo "FAIL could not build the taken-name fixture"; exit 1; }
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
w   = numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), "w")
# an unrelated tensor that happens to be called w_t, with the RIGHT shape and the WRONG values
w_t = numpy_helper.from_array(np.full((3, 2), 7.0, dtype=np.float32), "w_t")
g = helper.make_graph(
    [helper.make_node("Transpose", ["w"], ["a"], perm=[1, 0]),
     helper.make_node("Identity", ["a"], ["oa"]),
     helper.make_node("Identity", ["w_t"], ["keep"])],
    "g",
    [],
    [helper.make_tensor_value_info("oa", TensorProto.FLOAT, None),
     helper.make_tensor_value_info("keep", TensorProto.FLOAT, None)],
    [w, w_t])
onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), sys.argv[1])
PY2

python3 "$TOOL" "$tmp/taken.onnx" "$tmp/taken-out.onnx" >"$tmp/taken.log" 2>&1 || { cat "$tmp/taken.log"; fail=$((fail+1)); }
taken_verdict=$(python3 - "$tmp/taken-out.onnx" <<'PY2'
import sys
import numpy as np
import onnx
from onnx import numpy_helper
m = onnx.load(sys.argv[1])
inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
ident = {n.output[0]: n.input[0] for n in m.graph.node if n.op_type == "Identity"}
read = ident.get("oa")
w = np.arange(6, dtype=np.float32).reshape(2, 3)
print("right" if read and np.array_equal(inits[read], np.transpose(w, (1, 0))) else f"WRONG({read})")
PY2
)
check "an unrelated w_t in the source graph is not reused" "$taken_verdict" "right"

echo "---"
if [ "$fail" -ne 0 ]; then echo "$fail failed"; exit 1; fi
echo "fold_initializer_transpose: all checks ok"
