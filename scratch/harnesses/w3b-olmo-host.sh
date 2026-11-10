#!/usr/bin/env bash
# W3b gate, HOST HALF (device-free, so it costs no npu-hold time): info -> import --weight-layout b8.
# Engine binary: ~/wt/zha-main/build-npu/1bit, built 2026-11-09 22:16 from tree HEAD 51b6b547c
# (npu-pack: measure the 1.9 Token fconst rule ... (#1767)) -- recorded because the board row must name the engine commit.
set -uo pipefail
B=$HOME/wt/zha-main/build-npu/1bit
PACK=$HOME/scratch-packs/olmo1b-2048x6144
OUT=$HOME/scratch-packs/olmo1b-b8-container
K=$HOME/ogaload/oga13/kernels-2048-6144-b8

echo "=== engine: $("$B" --version 2>&1 | head -1)  ($B)"
echo "=== kernel set: $K ($(ls "$K"/*.elf 2>/dev/null | wc -l) ELFs)"
echo
echo "=== 1. npu-pack info (writes nothing) ==="
"$B" npu-pack info "$PACK" 2>&1 | head -30
echo
echo "=== 2. npu-pack import --weight-layout b8 ==="
rm -rf "$OUT"
"$B" npu-pack import "$PACK" -o "$OUT" --weight-layout b8 2>&1 | tail -25
rc=$?
echo
echo "=== result ==="
echo "import rc=$rc"
[ -d "$OUT" ] && { echo "container files:"; ls -la "$OUT" | head -12; echo "size: $(du -sh "$OUT" | cut -f1)"; }
