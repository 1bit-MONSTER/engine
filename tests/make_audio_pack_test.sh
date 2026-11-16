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
# make_audio_pack_test: the pack on disk is the folded one, and a pack that cannot be folded is refused
# (engine#2366).
#
# The arms that matter are the second and the third: a script that folds and then writes whatever it has is
# indistinguishable from this one until something re-reads the file it wrote, and a script that falls back to
# the unfurled decoder when `onnx` is missing turns a hard error into a silent 26x transpose per transcription.

set -uo pipefail

SRC=$(cd "$(dirname "$0")/.." && pwd)
SCRIPT=$SRC/tools/make-audio-pack.sh
python3 -c "import onnx, numpy" 2>/dev/null || { echo "SKIP: python onnx module not installed"; exit 77; }

tmp=$(mktemp -d "${TMPDIR:-/tmp}/packtest.XXXXXX") || exit 1
trap 'rm -rf "$tmp"' EXIT
fail=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: want '$3' got '$2'"; fail=$((fail+1)); fi; }

# An encoder with no initializer-fed Transpose, and a decoder with one feeding a MatMul -- the shape the script
# is supposed to remove. `tiny` is the smallest graph `onnx` will still accept.
python3 - "$tmp/enc.onnx" "$tmp/dec.onnx" <<'PY' || { echo "FAIL could not build the fixture"; exit 1; }
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

def save(path, nodes, inits, ins, outs):
    g = helper.make_graph(nodes, "g", ins, outs, inits)
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), path)

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
w = numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(4, 3), "w")
save(sys.argv[1], [helper.make_node("Identity", ["x"], ["y2"])], [], [x],
     [helper.make_tensor_value_info("y2", TensorProto.FLOAT, [1, 3])])
save(sys.argv[2], [helper.make_node("Transpose", ["w"], ["wt"], perm=[1, 0]),
                   helper.make_node("MatMul", ["x", "wt"], ["y"])], [w], [x], [y])
PY
echo "sidecar" > "$tmp/dec.onnx.data"

out=$tmp/pack
"$SCRIPT" --out "$out" "$tmp/enc.onnx" "$tmp/dec.onnx" "$tmp/dec.onnx.data" >"$tmp/log" 2>&1
check "the pack is written" "$?" 0
check "the encoder is copied" "$([ -f "$out/encoder_model.onnx" ] && echo yes)" yes
check "the sidecar is copied beside the graph" "$([ -f "$out/dec.onnx.data" ] && echo yes)" yes

read -r left nodes <<EOF
$(python3 - "$out/decoder_model.onnx" <<'PY'
import sys
import onnx
m = onnx.load(sys.argv[1])
inits = {i.name for i in m.graph.initializer}
print(sum(1 for n in m.graph.node if n.op_type == "Transpose" and n.input[0] in inits),
      len(m.graph.node))
PY
)
EOF
check "the pack's decoder has no initializer-fed Transpose" "$left" 0
check "and the node was actually removed" "$nodes" 1      # Transpose + MatMul -> MatMul

# The refusal arm: with `onnx` unimportable the script must fail rather than write the unfurled decoder. The
# stub is a module that raises, so the check is the same one CI would hit with no onnx installed at all.
mkdir -p "$tmp/stub" && echo 'raise ImportError("no onnx here")' > "$tmp/stub/onnx.py"
rc=0; PYTHONPATH=$tmp/stub "$SCRIPT" --out "$tmp/refused" "$tmp/enc.onnx" "$tmp/dec.onnx" >"$tmp/log2" 2>&1 || rc=$?
check "a pack that cannot be folded is refused" "$rc" 1
check "nothing was written" "$([ -e "$tmp/refused" ] && echo written || echo absent)" absent
check "and the error names the fold command" "$(grep -c 'fold_initializer_transpose.py' "$tmp/log2")" 1

echo "---"
if [ "$fail" -ne 0 ]; then echo "$fail failed"; exit 1; fi
echo "make_audio_pack: all checks ok"
