#!/usr/bin/env bash
# Classify a libggml-hrx.so against every candidate signal for "the private HIP kernels are compiled in" (engine#1443).
#
# Scratch harness. The point of the table is that a gate needs BOTH sides: a library that must pass and one that must
# fail. Two of my earlier proposals were never held against a fail case, and one of them would have weakened a release
# check silently -- so this prints every candidate for every library side by side.
#
# usage: signal_probe.sh <label=path> ...
set -uo pipefail

elfcount() { LC_ALL=C grep -aob $'\x7fELF' "$1" | wc -l; }
gfxcount() { strings -a "$1" 2>/dev/null | grep -cE 'gfx11|gfx12'; }
addonsym() { nm -D --defined-only "$1" 2>/dev/null | grep -c ggml_hrx_hip_addon_register; }
triple()   { strings -a "$1" 2>/dev/null | grep -c amdgcn-amd-amdhsa; }

printf '%-26s %5s %10s %6s %7s %6s %10s\n' label bytes embeddedELF gfx_strs addon_sym amdgcn verdict
for spec in "$@"; do
    label=${spec%%=*}; f=${spec#*=}
    case "$f" in "~/"*) f="$HOME/${f#"~/"}" ;; esac   # ~ is not expanded after = in a spec
    if [ ! -f "$f" ]; then printf '%-26s  MISSING: %s\n' "$label" "$f"; continue; fi
    e=$(elfcount "$f"); g=$(gfxcount "$f"); a=$(addonsym "$f"); t=$(triple "$f")
    # the current check's own rule
    if [ "$t" -ne 0 ]; then v="passes today"; else v="REFUSED today"; fi
    printf '%-26s %5s %10s %6s %7s %6s %10s\n' "$label" "$(stat -c%s "$f")" "$e" "$g" "$a" "$t" "$v"
done
cat <<'NOTE'

Read the table as: a usable gate must pass every row that HAS the kernels and refuse every row that does not.
'embeddedELF' passes the three kernel-bearing libraries (56, 3, 51) — but until a kernel-less one is measured,
it is unproven whether it refuses anything, which is why it is not proposed as a replacement yet.
NOTE
