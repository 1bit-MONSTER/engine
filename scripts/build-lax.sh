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
# build-lax.sh <prefix>
#
# Builds the Qwen3.6-35B-A3B whole-layer MoE decode (docs/npu-lax.md) from the open
# kernels pinned in third_party/OpenFlowLM-Next, into <prefix>:
#   <prefix>/kernels/lax_l, lax_a     the merged whole-layer design, linear-attention and
#                                     full-attention control texts (one xclbin)
#   <prefix>/kernels/ln, lm_head_q8   the final RMSNorm and the q8 lm_head
#   <prefix>/src/open_kernels         the pinned tree, with its XRT harness built in
#                                     harness/build/run_kernel and the ln / lm_head builds
#                                     where its tools look for them (--designs default)
# The tree is copied to <prefix>/src first: the kernel generator writes into it, and
# the submodule stays clean. Run the tools from <prefix>/src/open_kernels (docs/npu-lax.md).
#
# Needs the IRON toolchain the designs are written for (mlir-aie 1.4.2 + Peano, in a
# Python venv: IRON_VENV) and the AIE tools that provide aiebu-asm (AIETOOLS, the
# aietools directory of a Vitis install); xclbinutil on PATH; XRT headers in
# /opt/xilinx/xrt (or XILINX_XRT). Four kernel builds, a few minutes each.
set -euo pipefail
prefix=${1:?usage: build-lax.sh <prefix>}
: "${IRON_VENV:?set IRON_VENV to the mlir-aie 1.4.2 + Peano venv}"
: "${AIETOOLS:?set AIETOOLS to a Vitis aietools directory (aiebu-asm)}"
root=$(cd "$(dirname "$0")/.." && pwd)
sub=$root/third_party/OpenFlowLM-Next
[ -f "$sub/open_kernels/build_design.py" ] ||
    { echo "third_party/OpenFlowLM-Next is empty: git submodule update --init third_party/OpenFlowLM-Next"; exit 1; }
mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd)

commit=$(git -C "$sub" rev-parse --short=12 HEAD)
rm -rf "$prefix/src"
mkdir -p "$prefix/src"
git -C "$sub" archive HEAD open_kernels | tar -x -C "$prefix/src"
ok=$prefix/src/open_kernels

# shellcheck disable=SC1091
source "$IRON_VENV/bin/activate"
export PATH="$AIETOOLS/bin:$VIRTUAL_ENV/bin:/usr/bin:$PATH"
unset PYTHONPATH
spec=recipes/specs/qwen36-35b-a3b.json
cd "$ok"

# The final norm and the lm_head (the recipe's own build sets). This also regenerates
# the whole-layer designs' kernel sources (designs/layer_x/gen_kernels.py) for the spec.
# --force: <prefix>/src is fresh on every run, so its build cache key alone proves nothing.
python export_qwen36_kernels.py --spec "$spec" --only ln,lm_head_q8 --out "$prefix/kernels" --force

# The merged whole-layer design, both control texts from the same source. The three
# ONDV flags are the on-device routing fixes (docs/npu-lax.md, "What made it correct").
export OPEN_KERNELS_SPEC=$spec MOE_ONDEVICE_ROUTE=1 ONDV_EMIT_SHARED=1 ONDV_PKTDONE_ACQ=1
LAX_KIND=0 python build_design.py designs/layer_x/lax.py "$prefix/kernels/lax_l"
LAX_KIND=1 python build_design.py designs/layer_x/lax.py "$prefix/kernels/lax_a"
deactivate

# The tools find the ln / lm_head builds under designs/<build_dir> (the recipe's manifest),
# which is where the export builds them; check rather than assume.
python3 - "$prefix/kernels" "$ok/designs" <<'PY'
import json, os, sys
kernels, designs = sys.argv[1], sys.argv[2]
builds = json.load(open(os.path.join(kernels, "manifest.json")))["builds"]
for name in ("ln", "lm_head_q8"):
    xclbin = os.path.join(designs, builds[name]["build_dir"], "final.xclbin")
    if not os.path.isfile(xclbin):
        sys.exit(f"{xclbin} missing after the export")
PY

cmake -S harness -B harness/build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build harness/build -j "$(nproc)" > /dev/null

echo "lax ($commit): kernels in $prefix/kernels, tools in $ok (harness/build/run_kernel)"
md5sum "$prefix"/kernels/lax_l/insts.bin "$prefix"/kernels/lax_a/insts.bin
