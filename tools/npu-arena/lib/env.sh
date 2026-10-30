# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# lib/env.sh -- sourced by every arena build/run.  Resolves the toolchain from the
# environment (or from a local TOOLCHAIN.lock, which is deliberately NOT in the
# repo: it pins absolute paths on one box, see TOOLCHAIN.lock.example).  No
# network, no fallbacks: a missing path is an error.

ARENA=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENGINE_TREE=${ENGINE_TREE:-$(cd "$ARENA/../.." && pwd)}

# A local lock (if present) fills any variable the environment did not set.
LOCK=$ARENA/TOOLCHAIN.lock
if [ -f "$LOCK" ]; then
    while IFS='=' read -r k v; do
        case "$k" in ''|\#*) continue ;; esac
        v=$(printf '%s' "$v" | sed 's/^"//; s/"$//')
        if [ -z "$(eval "printf '%s' \"\${$k:-}\"")" ]; then export "$k=$v"; fi
    done < "$LOCK"
fi

MLIR_AIE=${MLIR_AIE:-$HOME/mlir-aie}
MLIR_AIE_INSTALL=${MLIR_AIE_INSTALL:-$MLIR_AIE/install_tmp}
PEANO=${PEANO:-$MLIR_AIE/.venv/lib/python3.14/site-packages/llvm-aie}
AIE_API_HEADERS=${AIE_API_HEADERS:-$HOME/Xilinx/2026.1/Vitis/aietools/include}
AIE_KERNELS_INC=${AIE_KERNELS_INC:-$MLIR_AIE/.venv/lib/python3.14/site-packages/mlir_aie/include/aie_kernels/aie2p}
GENERATORS=${GENERATORS:-}
ENGINE_BUILD=${ENGINE_BUILD:-$ENGINE_TREE/build}
NPU_RT_LIB=${NPU_RT_LIB:-$ENGINE_BUILD/libonebit_npu_rt.a}
HOST_CXX=${HOST_CXX:-g++}
GPU_HOLD=${GPU_HOLD:-$HOME/.local/bin/gpu-hold}
DEVICE_LOCK=${DEVICE_LOCK:-/tmp/1bit-npu-device.lock}

arena_require() {
    local bad=0 p
    for p in "$@"; do
        if [ ! -e "$p" ]; then echo "arena: MISSING pinned path: $p" >&2; bad=1; fi
    done
    return $bad
}

# The exact AIE2P compile environment (same as generators/elf_emit.sh expects).
arena_aie_env() {
    export PATH="$MLIR_AIE_INSTALL/bin:$PEANO/bin:$PATH"
    export PYTHONPATH="$MLIR_AIE_INSTALL/python:$MLIR_AIE/.venv/lib/python3.14/site-packages"
    export LD_LIBRARY_PATH="$MLIR_AIE_INSTALL/python/aie/_mlir_libs"
}

# Toolchain identity, printed into every score line so a score can be trusted.
arena_toolchain_id() {
    local aie peano
    aie=$(git -C "$MLIR_AIE" rev-parse --short=12 HEAD 2>/dev/null || echo "unknown")
    peano=$("$PEANO/bin/clang" --version 2>/dev/null | head -1 | grep -o 'c9c5ecb\|[0-9a-f]\{40\}' | head -1)
    echo "mlir-aie=$aie peano=${peano:-unknown}"
}
