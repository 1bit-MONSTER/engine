#!/bin/bash
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
# make_probe.sh <PHASE> [<PHASE> ...] -- build the DMA-ceiling probe submissions for
# a phase: the phase's own tap/tile geometry with the matrix unit replaced by
# mm_dmabw.cc (no arithmetic), in two core configurations:
#
#   nm16   M=16, m=16, -r 1  -> 8 cores   (the shipped decode twin's shape)
#   r4     M=64, m=16, -r 4  -> 32 cores  (same B taps, B broadcast down 4 rows;
#                                          M=64/m=16 = 4 M-tiles is what lets the
#                                          unmodified generator place 4 rows)
#
# The B layout is the blocked one (_b8), because that is what the shipping
# container uses now; each probe is scored with the arena runner --no-check, so
# its GB/s is the attainable ceiling for that phase's bytes.
set -u
ARENA=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$ARENA/lib/env.sh"
# The probe microkernel is AIE kernel source and lives in the private kernel tree
# (npu-kernels: npu/probe/mm_dmabw.cc), not in this repository.
PROBE_SRC=${PROBE_SRC:-${NPU_KERNELS:-$HOME/wt/npu-kernels}/npu/probe/mm_dmabw.cc}
arena_require "$PROBE_SRC" || exit 1
for T in "$@"; do
    K=$(sed -n 's/^K=//p' "$ARENA/tasks/$T/kernel/GEOMETRY" 2>/dev/null)
    N=$(sed -n 's/^N=//p' "$ARENA/tasks/$T/kernel/GEOMETRY" 2>/dev/null)
    [ -n "$K" ] && [ -n "$N" ] || { echo "make_probe: no GEOMETRY for $T"; continue; }
    for variant in nm16 r4; do
        D=$ARENA/probe/$T-$variant
        mkdir -p "$D"
        cp -n "$ARENA/tasks/$T/kernel"/{n1_core_i8_v27.py,elf_emit.sh,silu_quant.h,zero.cc} "$D/" 2>/dev/null
        cp "$ARENA/submissions/$T/cand-blocked-b/n1_blocked.py" "$D/" 2>/dev/null
        if [ "$variant" = nm16 ]; then MM=16; MR=1; DM=16; else MM=64; MR=4; DM=64; fi
        cat > "$D/build.sh" <<EOF
#!/bin/bash
# DMA-ceiling probe: $T, $variant (microkernel = mm_dmabw.cc, no arithmetic).
set -u
ARENA="\${ARENA:-$ARENA}"
. "\$ARENA/lib/env.sh"
TASK=$T; K=$K; N=$N
M=$MM; MR=$MR; DIM_M=$DM
mkdir -p out
work=\$(mktemp -d /tmp/arena-probe.\$\$.XXXXXX); trap 'rm -rf "\$work"' EXIT
arena_aie_env
set -e
"\$PEANO/bin/clang++" --target=aie2p-none-unknown-elf --std=c++20 -O2 \\
    -DDIM_M=\$DIM_M -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY \\
    -isystem "\$PEANO/include/c++/v1" -I "\$AIE_API_HEADERS" -I "\$AIE_KERNELS_INC" \\
    -c "$PROBE_SRC" -o "\$work/mm_32x64x128.o"
python3 n1_blocked.py -M \$M -K \$K -N \$N -m 16 -k 64 -n 128 -c 8 -r \$MR -b 5 --blocked-b \\
    > "\$work/d.mlir"
. ./elf_emit.sh
emit_full_elf "\$MLIR_AIE_INSTALL/bin/aiecc" "\$PEANO" "\$MLIR_AIE_INSTALL" \\
    "\$work/d.mlir" "\$PWD/out/\$TASK.elf"
[ -s "\$PWD/out/\$TASK.elf" ] || { echo "probe build failed"; exit 1; }
echo "built out/\$TASK.elf \$(stat -c%s "\$PWD/out/\$TASK.elf") B ($variant)"
EOF
        chmod +x "$D/build.sh"
        printf 'W_BYTES=%d\n' $((K * N)) > "$D/LAYOUT"
        cp "$ARENA/submissions/$T/cand-blocked-b/layout.sh" "$D/layout.sh"
        chmod +x "$D/layout.sh"
        echo "probe $T/$variant ready (K=$K N=$N M=$MM rows=$MR)"
    done
done
