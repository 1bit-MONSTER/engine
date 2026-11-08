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
# npu_prefill_tgate.sh -- build the prefill GEMM kernel at several T and gate each build numerically.
#
# Why this exists: the prefill design's whole weight stream is refetched once per T-tile, so
# "is a bigger T faster per row?" is the question behind engine#1421, and answering it needs three
# things that were each rediscovered by hand: the toolchain that actually builds our designs, a
# fixture generator, and the device gate.  This is those three in one place, with every version
# pinned in the output so a number can never be quoted without the toolchain that produced it.
#
#   tools/npu_prefill_tgate.sh              # T = 256 512 1024, shape qkv (N=2560 K=2560)
#   T_LIST="256 1024" SHAPE=qkv tools/npu_prefill_tgate.sh
#
# Requires (all on this box, none of them fetched by this script):
#   * aiecc + Peano: ~/amd-oss/iron-venv  (aiecc 20a9c2fb2dc, llvm-aie 773413fb18d0 inside its
#     site-packages).  NOTE the *other* venv, ~/amd-oss/iron-upstream-venv (aiecc d53582d3e0f), does
#     NOT build these designs: it rejects `--alloc-scheme=basic-sequential`, so the pin is load-bearing.
#   * our kernel designs: $OFNK (default ~/OpenFlowLM-Next/open_kernels)
#   * the harness: built from $OFNK/harness with -DXRT_INCLUDE_DIR=/usr/include
#     -DXRT_LIB_DIR=/usr/lib/x86_64-linux-gnu (XRT is in /usr on this box)
#   * the device: run under `npu-hold -l <label> -t 10 -- <this script>`; it never takes :18097.
#
# The gate is the design's own: make_test.py writes a fp64 reference from the same q4_1 pool bytes the
# kernel streams, run_kernel issues the N-buffer run, compare.py reports per-token cosine.  A build that
# "drains nothing" shows up as a cosine near zero rather than as a crash, which is exactly the failure
# this gate caught twice before (a T>256 fifo overfeed, and a later all-zero drain).
set -u
OFNK=${OFNK:-$HOME/OpenFlowLM-Next/open_kernels}
VENV=${VENV:-$HOME/amd-oss/iron-venv}
PEANO=${PEANO:-$VENV/lib/python3.14/site-packages/llvm-aie}
SHAPE=${SHAPE:-qkv}
T_LIST=${T_LIST:-"256 512 1024"}
WORK=${WORK:-$HOME/wt/npu-tgate-$(date +%Y%m%dT%H%M%S)}
HARNESS=${HARNESS:-$HOME/wt/harness-build/run_kernel}

export PEANO_INSTALL_DIR=$PEANO
export PATH=$VENV/bin:$PATH
export LD_LIBRARY_PATH=$PEANO/lib:${LD_LIBRARY_PATH:-}

die() { echo "npu_prefill_tgate: $*" >&2; exit 1; }
[ -x "$VENV/bin/aiecc" ] || die "no aiecc at $VENV/bin/aiecc"
[ -d "$OFNK/designs/gemm_q4_prefill" ] || die "no designs at $OFNK (set OFNK)"
[ -x "$HARNESS" ] || die "no run_kernel at $HARNESS (build $OFNK/harness with XRT_INCLUDE_DIR=/usr/include)"

mkdir -p "$WORK"
{
    echo "== npu_prefill_tgate $(date -Is)  shape=$SHAPE  T=$T_LIST"
    echo "   aiecc:  $("$VENV/bin/aiecc" --version 2>&1 | sed -n 's/.*git SHA: *//p' | head -1)"
    echo "   peano:  $("$PEANO/bin/clang++" --version | head -1)"
    echo "   work:   $WORK"
} | tee "$WORK/run.log"

cd "$OFNK/designs/gemm_q4_prefill" || die "no design dir"
rc=0
for T in $T_LIST; do
    out="$WORK/build_t$T"
    echo "--- build T=$T" | tee -a "$WORK/run.log"
    GQP_N=${GQP_N:-2560} GQP_K=${GQP_K:-2560} GQP_T=$T \
        timeout 1800 python "$OFNK/build_design.py" gemm_q4_prefill.py "$out" >>"$WORK/run.log" 2>&1 \
        || { echo "    BUILD FAILED (see run.log)" | tee -a "$WORK/run.log"; rc=1; continue; }
    # fixtures and the run description, at this T
    python make_test.py --shape "$SHAPE" --tokens "$T" >>"$WORK/run.log" 2>&1 || { rc=1; continue; }
    tag="${SHAPE}_t$T"
    # run_kernel resolves every `buf` path against the CFG's own directory, and compare.py reads the
    # dumped y_<tag>.bin from the design directory - so the cfg is written next to the fixtures and the
    # run happens there.  (Learned the hard way: a cfg in a scratch dir cannot find w_<shape>.bin.)
    sed -e "s|build_${tag}/final.xclbin|$out/final.xclbin|" \
        -e "s|build_${tag}/insts.bin|$out/insts.bin|" "run_$tag.cfg" > "gate_$tag.cfg"
    echo "--- run $tag" | tee -a "$WORK/run.log"
    timeout 600 "$HARNESS" "gate_$tag.cfg" >>"$WORK/run.log" 2>&1 || { rm -f "gate_$tag.cfg"; rc=1; continue; }
    echo "--- compare $tag" | tee -a "$WORK/run.log"
    python compare.py "$tag" 2>&1 | tee -a "$WORK/run.log" | grep -E "per-token cosine|rel_fro|absmax" || rc=1
    rm -f "gate_$tag.cfg"
done
echo "== done (rc=$rc)  full log: $WORK/run.log"
exit $rc
