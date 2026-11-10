#!/usr/bin/env bash
# W3b: build my two families with the generalised generator (private repo, commit ce6d4b4a).
#  2048x6144 (OLMo-1B):           H 2048, IM 8192,  16 heads / 16 kv, hd 128 -> QKV 2048x6144
#  2048x2560 (Qwen2.5-3B hybrid): H 2048, IM 11008, 16 heads /  2 kv, hd 128 -> QKV 2048x2560
# Geometry read from each pack's own ONNX (SSMLP/GQO/MatMulNBits attributes), not from a table.
# Header note (measured, not assumed): the compiler is Peano, but the microkernel's aie_api/aie.hpp -> aie_adf.hpp chain
# includes <adf.h>, which ships ONLY in aietools (mlir_aie's include has aie.hpp but not adf.h; checked with find over ~ and /usr).
# So the Vitis/aietools INCLUDE PATH is required here, and every device-proven _b8 set was built that way. VITIS_INC is left at the
# generator's default; see STATE for the compliance question this raises against plan section 2.
set -uo pipefail
G=$HOME/wt/nk-arena/scripts/emit-i8-family.sh
P=$(~/iron-prefix/venv/bin/python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
build() { # name hidden intermediate heads kv hd
  local out=$HOME/ogaload/oga13/kernels-$1-b8
  echo "### $1 -> $out"
  $HOME/lb/thermal-run.sh "$G" --hidden "$2" --intermediate "$3" --heads "$4" --kv-heads "$5" --head-dim "$6" --out "$out" 2>&1 \
    | grep -E 'geometry:|^(QKV|O |GU|D )|out:|FAIL'
}
build 2048-6144 2048 8192  16 16 128
build 2048-2560 2048 11008 16  2 128
echo "=== resulting sets ==="
for d in 2048-6144 2048-2560; do echo "-- $d: $(ls $HOME/ogaload/oga13/kernels-$d-b8/*.elf 2>/dev/null | wc -l) ELF(s)"; ls $HOME/ogaload/oga13/kernels-$d-b8/ 2>/dev/null; done
