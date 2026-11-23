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
# weekly_package_checks_test.sh - three rows of scripts/weekly.sh package_checks (engine#3287), run on miniature
# engine tarballs. The v2026.48 cut found that each of them failed every release package, the board-tested c35
# payload included, so a release could only be packaged by hand-waiving 22 FAIL lines:
#   1. the NPU set count demanded "the 0.6B, 1.7B and 4B sets"; now: at least one signed set, and every set is a
#      geometry config/npu-catalogue-sets.json names (the catalogue filter decides which models are offered)
#   2. "every ELF is a host program" rejected the signed-manifest sets' AIE kernels (<set>/<id>/layer1.elf), which the
#      allowlist admits by design; an AIE ELF anywhere else must still fail
#   3. the packed 1bit's strings were matched against private file names as substrings, so a private basename
#      matched inside a longer PUBLIC path that ends with it; a whole private name must still fail
# Each case asserts on the row's own line, not on the overall verdict: the miniature tarball fails other rows (no
# HRX library, no helpers) by construction. No compiler, device or network; git and python3 only.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
for t in git python3 tar strings od; do command -v "$t" > /dev/null || { echo "weekly_package_checks_test: skip (no $t)"; exit 77; }; done
w=$(mktemp -d); trap 'rm -rf "$w"' EXIT
pass=0 failn=0
ok() { echo "ok   $*"; pass=$((pass + 1)); }
no() { echo "FAIL $*"; failn=$((failn + 1)); }
has() { grep -qF -- "$2" "$1"; }

# the private pins package_checks reads names from: a checkout per pin, tracking files under the config's path=
for pin in npu-runtime npu-kernels gpu-kernels; do
    p=$(sed -n 's/^path=//p' "$root/config/$pin.config")
    g=$w/W/$pin; mkdir -p "$g/$p"
    case $pin in npu-runtime) printf 'x\n' > "$g/$p/bar.h" ;; npu-kernels) printf 'x\n' > "$g/$p/zq_secret.cpp" ;; *) printf 'x\n' > "$g/$p/k.hip" ;; esac
    git -C "$g" init -q && git -C "$g" add -A && git -C "$g" -c user.name=t -c user.email=t@t commit -q -m t
done
# package_checks unpacks with --use-compress-program=unzstd; the miniature tarballs are plain tar and the shim is cat,
# so the test needs no zstd (the CI image has none)
mkdir -p "$w/bin"; printf '#!/bin/sh\nexec cat\n' > "$w/bin/unzstd"; chmod +x "$w/bin/unzstd"
elf() {  # elf <file> <e_machine>: a 64-byte ELF64 header, enough for the magic and machine checks
    python3 - "$1" "$2" <<'PY'
import struct, sys
h = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9) + struct.pack("<HHI", 2, int(sys.argv[2]), 1) + bytes(64 - 24)
open(sys.argv[1], "wb").write(h)
PY
}
mkset() {  # mkset <top> <geometry>: a signed-manifest (kmanifest2-shaped) set with one AIE kernel
    local d=$1/npu/kernels/$2; mkdir -p "$d/38bf"
    printf 'npumanifest 2\n' > "$d/manifest"; printf 'sig\n' > "$d/manifest.sig"; elf "$d/38bf/layer1.elf" 1
}
run() {  # run <name>: package_checks on $w/<name>.tar, output in $w/<name>.out
    (cd "$w" && tar -cf "$1.tar" "$1")
    PATH=$w/bin:$PATH W=$w/W SRC=$root BUILD=$w/nobuild WEEKLY_NPU_KERNELS=1 bash -c '
        set -uo pipefail
        . "$SRC/scripts/pack-elf.sh"; . "$SRC/scripts/pack-check.sh"
        eval "$(sed -n "/^private_tracked_names() {/,/^}/p; /^package_checks() {/,/^}/p; /^links_ok() {/,/^}/p" "$SRC/scripts/weekly.sh")"
        package_checks "$1"' _ "$w/$1.tar" > "$w/$1.out" 2>&1
}

# A: what v2026.48 ships - two catalogue sets with AIE kernels, a 1bit whose only private-looking string is public
a=$w/A; mkdir -p "$a"; mkset "$a" 1024x3072x16x8; mkset "$a" 2560x9728x32x8
printf 'a public message naming a public file (npu/foo_bar.h:Thing has no hook)\n' > "$a/1bit"
run A
! has "$w/A.out" "want the 0.6B" && has "$w/A.out" "ok   NPU kernel sets: 2 (1024x3072x16x8 2560x9728x32x8)" &&
    ok "two catalogue sets are what the row asks for (no '0.6B, 1.7B and 4B' demand)" || { no "set count row on A"; cat "$w/A.out"; }
! grep -q "layer1.elf is an ELF for machine" "$w/A.out" && ok "the signed sets' AIE kernels are not called non-host ELFs" ||
    { no "AIE kernels at <set>/<id>/layer1.elf still FAIL the host-ELF row"; grep "ELF for machine" "$w/A.out"; }
! has "$w/A.out" "names a private source file" && ok "a public path ending in a private basename (npu/foo_bar.h vs bar.h) does not match" ||
    { no "the public npu/foo_bar.h string still matches bar.h"; grep "private source" "$w/A.out"; }

# B: the same rows must still catch the real thing
b=$w/B; mkdir -p "$b/bin"; mkset "$b" 1024x3072x16x8; mkset "$b" 4096x1x1x1
elf "$b/bin/stray.so" 1
printf 'npu/zq_secret.cpp:41: assertion failed\n' > "$b/1bit"
run B
has "$w/B.out" "FAIL NPU kernel set 4096x1x1x1 is not in config/npu-catalogue-sets.json" &&
    ok "a set no catalogue entry can load is refused" || { no "an unknown set geometry passed"; cat "$w/B.out"; }
grep -q "bin/stray.so is an ELF for machine 1" "$w/B.out" && ! grep -q "layer1.elf is an ELF for machine" "$w/B.out" &&
    ok "an AIE ELF outside a set still fails the host-ELF row, the set kernels do not" || { no "host-ELF row on B"; grep "ELF" "$w/B.out"; }
has "$w/B.out" "FAIL the packed 1bit names a private source file (1 strings)" &&
    ok "a whole private file name (zq_secret.cpp) in the 1bit still fails" || { no "zq_secret.cpp was not caught"; grep -i "private" "$w/B.out"; }

# C: no set at all, with WEEKLY_NPU_KERNELS=1
c=$w/C; mkdir -p "$c"; printf 'nothing\n' > "$c/1bit"
run C
has "$w/C.out" "FAIL WEEKLY_NPU_KERNELS=1 but C.tar carries no signed NPU kernel set" &&
    ok "a package with no set fails when the NPU kernels are on" || { no "a set-less package passed the set row"; cat "$w/C.out"; }

echo "weekly_package_checks_test: $pass passed, $failn failed"
[ "$failn" = 0 ]
