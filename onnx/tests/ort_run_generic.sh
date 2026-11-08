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
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# ort_run_generic.sh BINARY PYTHON — `ort-run --input/--output/--dump-all` on a graph it has never
# seen (docs/onnx.md, "Any graph"). The model is a two-node MatMul+Relu written here with python
# `onnx` (no ONNX Runtime python needed), so the check runs anywhere the C++ runner builds.
set -uo pipefail
bin=${1:?usage: ort_run_generic.sh <ort_run> <python3>}
py=${2:?usage: ort_run_generic.sh <ort_run> <python3>}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

"$py" - "$scratch" <<'PY'
import sys, numpy as np, onnx
from onnx import TensorProto, helper
d = sys.argv[1]
w = np.arange(16, dtype=np.float32).reshape(4, 4) / 16.0
g = helper.make_graph(
    [helper.make_node("MatMul", ["x", "W"], ["mm"]), helper.make_node("Relu", ["mm"], ["y"])],
    "generic_io", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 4])],
    [helper.make_tensor("W", TensorProto.FLOAT, [4, 4], w.ravel().tolist())])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)])
# python `onnx` 1.23 stamps IR version 14 by default and ONNX Runtime 1.23.2 refuses anything above
# 11 ("Unsupported model IR version: 14, max supported IR version: 11"), so the fixture is pinned to
# an IR the runtime takes. Anyone exporting with a newer onnx hits the same wall.
m.ir_version = 10
onnx.save(m, f"{d}/m.onnx")
np.save(f"{d}/w.npy", w)
np.array(np.random.default_rng(5).standard_normal((1, 4, 4)), dtype="<f4").tofile(f"{d}/x.bin")
PY
check "the fixture model and its input exist" '[ -s "$scratch/m.onnx" ] && [ -s "$scratch/x.bin" ]'

# 1. named input -> named output
"$bin" --model "$scratch/m.onnx" --input "x=$scratch/x.bin:float32:1x4x4" --output "y=$scratch/y.bin" \
  >"$scratch/run1.log" 2>&1
check "a graph it has never seen runs (rc 0)" '[ $? -eq 0 ]'
check "it printed the input it read" 'grep -q "in      : x float32" "$scratch/run1.log"'
check "and the output it wrote" 'grep -q "out     : y float32" "$scratch/run1.log"'

# 2. the numbers are the graph's numbers (numpy, not the runner's own arithmetic)
"$py" - "$scratch" <<'PY'
import sys, numpy as np
d = sys.argv[1]
x = np.fromfile(f"{d}/x.bin", dtype="<f4").reshape(1, 4, 4)
w = np.load(f"{d}/w.npy")
want = np.maximum(x @ w, 0.0)
got = np.fromfile(f"{d}/y.bin", dtype="<f4").reshape(1, 4, 4)
sys.exit(0 if np.allclose(got, want, rtol=1e-6, atol=1e-6) else 1)
PY
check "the output equals a reference computed independently (numpy)" '[ $? -eq 0 ]'

# 3. --dump-all writes every output with a sidecar that says what it is
"$bin" --model "$scratch/m.onnx" --input "x=$scratch/x.bin:float32:1x4x4" --dump-all "$scratch/dump" \
  >"$scratch/run2.log" 2>&1
check "--dump-all writes the tensor" '[ -s "$scratch/dump/y.bin" ]'
check "and a sidecar naming dtype, shape and path" 'grep -q "y float32x1x4x4" "$scratch/dump/outputs.txt"'

# 4. a file whose size disagrees with the declared shape is refused by name, not truncated
head -c 32 "$scratch/x.bin" >"$scratch/short.bin"
"$bin" --model "$scratch/m.onnx" --input "x=$scratch/short.bin:float32:1x4x4" --output "y=$scratch/no.bin" \
  >"$scratch/run3.log" 2>&1
rc=$?
check "a short input file is refused (non-zero rc $rc)" '[ $rc -ne 0 ]'
check "the refusal names the file and the size it needed" \
  'grep -q "holds 32 B, float32 of 16 elements needs 64 B" "$scratch/run3.log"'
check "and nothing was written" '[ ! -e "$scratch/no.bin" ]'

# 5. a dtype this runner does not carry is refused by name too
"$bin" --model "$scratch/m.onnx" --input "x=$scratch/x.bin:bfloat16:1x4x4" >"$scratch/run4.log" 2>&1
check "an unsupported dtype is refused by name" 'grep -q "not one this runner carries" "$scratch/run4.log"'

# 6. ONNX's own integer ops (engine #1436): the pinned runtime has ConvInteger for uint8 x uint8 only, and
#    DynamicQuantizeLinear always feeds uint8, so an int8-weight export cannot create a session. The refusal
#    must say it is the runtime's coverage and point at the other exports; the uint8 x uint8 graph still runs.
mkdir -p "$scratch/int8"
cp "$scratch/m.onnx" "$scratch/int8/model.onnx"
"$py" - "$scratch" <<'PY'
import sys, numpy as np, onnx
from onnx import TensorProto as T, helper
d = sys.argv[1]
w = (np.arange(9).reshape(1, 1, 3, 3) % 4)
for wt, wn, name in ((T.INT8, np.int8, "model_int8.onnx"), (T.UINT8, np.uint8, "conv_u8u8.onnx")):
    g = helper.make_graph([helper.make_node("ConvInteger", ["x", "W"], ["y"])], "conv_integer",
                          [helper.make_tensor_value_info("x", T.UINT8, [1, 1, 4, 4])],
                          [helper.make_tensor_value_info("y", T.INT32, [1, 1, 2, 2])],
                          [helper.make_tensor("W", wt, [1, 1, 3, 3], w.astype(wn).ravel().tolist())])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.save(m, f"{d}/int8/{name}")
np.save(f"{d}/int8/w.npy", w)
np.arange(16, dtype=np.uint8).tofile(f"{d}/int8/x.bin")
PY
"$bin" --model "$scratch/int8/model_int8.onnx" --input "x=$scratch/int8/x.bin:uint8:1x1x4x4" >"$scratch/run5.log" 2>&1
rc=$?
check "a uint8 x int8 ConvInteger graph cannot create a session (non-zero rc $rc)" '[ $rc -ne 0 ]'
check "the refusal says it is the runtime's kernel coverage" \
  'grep -q "has no CPU kernel for ConvInteger at this graph.s input types: a runtime coverage gap" "$scratch/run5.log"'
check "and what the pin does carry" 'grep -q "ConvInteger runs only uint8 x uint8" "$scratch/run5.log"'
check "and the other exports next to it" 'grep -q "other exports in .*: conv_u8u8.onnx, model.onnx" "$scratch/run5.log"'
"$bin" --model "$scratch/int8/conv_u8u8.onnx" --input "x=$scratch/int8/x.bin:uint8:1x1x4x4" \
  --output "y=$scratch/int8/y.bin" >"$scratch/run6.log" 2>&1
check "the uint8 x uint8 ConvInteger graph runs (rc 0)" '[ $? -eq 0 ]'
"$py" - "$scratch" <<'PY'
import sys, numpy as np
d = sys.argv[1] + "/int8"
x = np.fromfile(f"{d}/x.bin", dtype=np.uint8).reshape(4, 4).astype(np.int64)
w = np.load(f"{d}/w.npy").reshape(3, 3)
want = np.array([[(x[i:i + 3, j:j + 3] * w).sum() for j in range(2)] for i in range(2)])
got = np.fromfile(f"{d}/y.bin", dtype="<i4").reshape(2, 2)
sys.exit(0 if (got == want).all() else 1)
PY
check "and equals a numpy convolution" '[ $? -eq 0 ]'

echo "ort_run_generic: $fail failure(s)"
exit $fail
