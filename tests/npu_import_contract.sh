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
# engine#812: the XDNA import class advertises device execution on an imported dma-buf, and this is
# the test that ties the advertisement to what the running driver does with it.
#
# The two halves are separate on purpose:
#   - the advertisement is a property of the pinned UMD (third_party/hrx-system, libamdf): memory
#     profile ordinal 1 is the import class and it carries AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API
#     (libamdf/src/xdna/umd/drm/memory_profile.c). It is read from that source, so a pin that narrows
#     the contract is noticed here rather than in a caller's pipeline plan;
#   - what executes is measured on the device with libamdf's own CTS: the imported-backing cases
#     (`ImportedBacking/*`, GPU export -> NPU import -> NPU executes on the imported range) and the
#     registered-pages control (`ResidentGpuXdnaTest.RegisteredCausalRoundTrip`), which has to pass in
#     every world -- it is what proves an imported-backing failure is the import, not the array.
#
# The pairing is the contract, and it is keyed on the loaded module, because that is what decides it:
# a module carrying third_party/patches/amdxdna/0006+0007 (the #812 pair: don't invalidate an imported
# dma-buf BO at mmap time, don't re-invalidate it from the notifier) executes on an imported range; a
# module without them refuses every submission of the process after the first mmap of an import
# (amdxdna's population step is client-wide and cannot HMM-refault the exporter's pages: dmesg says
# `aie2_cmd_submit: Populate ranges failed, ret -14`). The fingerprint table in that directory's
# README says which loaded module is which, so that is what this reads.
#
#   ONEBIT_IMPORT_CONTRACT=advertised  the acceptance form: fail whenever the advertisement above is
#                                      not delivered by the running module. That is red on a driver
#                                      without the pair -- which is engine#812 -- and it is the gate
#                                      that goes green when one is loaded. Default: table only.
#
# Needs the array and a GPU (the CTS exports the dma-buf from the iGPU). Run it under npu-hold; it
# costs ~1.2 s of device time. Skips (77) without /dev/accel/accel0 or without a libamdf CTS build,
# so public CI neither runs nor fails it:
#   ONEBIT_LIBAMDF_CTS_DIR=<build>/libamdf ctest -R npu_import_contract
#
# usage: tests/npu_import_contract.sh <repo root>   [env: ONEBIT_LIBAMDF_CTS_DIR, AMDF_GPU_TARGET]
set -uo pipefail

root=${1:?usage: npu_import_contract.sh <repo root>}
cts=${ONEBIT_LIBAMDF_CTS_DIR:-}
gfx=${AMDF_GPU_TARGET:-gfx1151}
strict=${ONEBIT_IMPORT_CONTRACT:-table}

[ -e /dev/accel/accel0 ] || { echo "skip: no /dev/accel/accel0 (this box has no XDNA NPU)"; exit 77; }
if [ -z "$cts" ] || [ ! -x "$cts/libamdf_cts_interop_gpu_xdna_recipes_execution_dynamic_bin" ]; then
    echo "skip: no libamdf CTS build here; set ONEBIT_LIBAMDF_CTS_DIR to one (the -S <hrx-system>"
    echo "      -DAMDF_BUILD=ON -DLOOM_TARGET_XDNA=ON build's libamdf/ directory)"
    exit 77
fi

fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; [ -n "${3:-}" ] && echo "     $3"; fail=1; fi; }

loaded=$(cat /sys/module/amdxdna/srcversion 2>/dev/null)
[ -n "$loaded" ] || { echo "skip: amdxdna is not loaded"; exit 77; }

# The advertisement, read from the pinned UMD at the branch that describes the import class.
profile=$root/third_party/hrx-system/libamdf/src/xdna/umd/drm/memory_profile.c
[ -f "$profile" ] || { echo "skip: $profile is not checked out (submodule)"; exit 77; }
advertised=no
if awk '/memory_profile_ordinal == 1/{f=1} f&&/FOREIGN_API/{print "yes"; exit}' "$profile" | grep -q yes; then
    advertised=yes
fi
check "the pinned libamdf advertises FOREIGN_API on the import class" "[ '$advertised' = yes ]" \
      "no FOREIGN_API in $profile's ordinal-1 branch: the contract below is the narrowed one"

# Which loaded modules carry the pair: the README's fingerprint table, row by row.
table=$root/third_party/patches/amdxdna/README.md
carriers=$(grep -oE '^\|[^|]*0006[^|]*0007[^|]*\|[^|]*`[0-9A-F]{23}`' "$table" | grep -oE '[0-9A-F]{23}')
known=$(grep -oE '^\|[^|]*\| *`[0-9A-F]{23}`' "$table" | grep -oE '[0-9A-F]{23}')
carrier=no
grep -qx "$loaded" <<<"$carriers" && carrier=yes
if ! grep -qx "$loaded" <<<"$known"; then
    msg="loaded amdxdna $loaded is not in $table's fingerprint table, so nobody has measured what it does with an import"
    if [ "$strict" = advertised ]; then echo "FAIL $msg"; fail=1; else echo "skip: $msg"; exit 77; fi
fi
echo "== amdxdna $loaded: #812 pair $carrier =="

bind=$cts/libamdf_cts_interop_gpu_xdna_linux_memory_dynamic_bin
execu=$cts/libamdf_cts_interop_gpu_xdna_recipes_execution_dynamic_bin
log=$(mktemp -d)
trap 'rm -rf "$log"' EXIT

timeout 300 "$bind" --amdf_library="$cts/libamdf.so" --amdf_gpu_target="$gfx" > "$log/bind.txt" 2>&1
check "binding: host access on an imported range (7 cases)" \
      "grep -q '\[  PASSED  \]' '$log/bind.txt' && ! grep -q '\[  FAILED  \]' '$log/bind.txt'" \
      "$(grep -E '\[  FAILED  \]|Failure' "$log/bind.txt" | head -3)"

timeout 600 "$execu" --amdf_library="$cts/libamdf.so" --amdf_gpu_target="$gfx" \
    --gtest_filter='ImportedBacking/*:ResidentGpuXdnaTest.*' > "$log/exec.txt" 2>&1

check "control: the array executes (registered caller pages)" \
      "grep -q '\[       OK \] ResidentGpuXdnaTest.RegisteredCausalRoundTrip' '$log/exec.txt'" \
      "$(grep -A2 'RUN.*RegisteredCausalRoundTrip' "$log/exec.txt" | head -3)"
imported=$(grep -cE '^\[ RUN      \] ImportedBacking/' "$log/exec.txt")
imported_ok=$(grep -cE '^\[       OK \] ImportedBacking/' "$log/exec.txt")
check "the run covered the imported-backing cases" "[ '$imported' -ge 8 ]" "only $imported arm(s) ran"

if [ "$carrier" = yes ] && [ "$advertised" = yes ]; then
    # The pair is loaded: the advertisement must be true, arm by arm.
    check "execution on an imported dma-buf ($imported arms, the advertised capability)" \
          "[ '$imported_ok' = '$imported' ]" \
          "$(grep -E '^\[  FAILED  \] ImportedBacking/' "$log/exec.txt" | head -3)"
    verdict="delivers what it advertises"
elif [ "$carrier" = yes ]; then
    # Loaded and no longer advertised (the pin narrowed it): either result is honest, so only record it.
    echo "note: $imported_ok of $imported imported arms pass and the pin no longer advertises execution"
    verdict="executes; not advertised (the pin narrowed the contract)"
elif [ "$strict" = advertised ] && [ "$advertised" = yes ]; then
    # The acceptance form: this is engine#812 itself -- a capability advertised and not delivered.
    echo "FAIL the pinned libamdf advertises FOREIGN_API dma-buf execution and amdxdna $loaded"
    echo "     refuses it: $((imported - imported_ok)) of $imported imported arms failed"
    grep -m1 -A3 'kernel_queue_create' "$log/exec.txt" | sed 's/^/     /'
    fail=1
    verdict="advertises execution it does not deliver (#812)"
else
    # Documented state of a driver without the pair: imported execution is refused, and it must be
    # refused at the same place every time -- a different failure would be a different bug.
    check "execution on an imported dma-buf is refused (module has no 0006/0007)" \
          "[ '$imported_ok' = 0 ]" "$imported_ok of $imported arms passed on a module without the pair"
    check "the refusal is the import's, at kernel_queue_create" \
          "grep -q 'kernel_queue_create' '$log/exec.txt'" \
          "$(grep -E '^\[  PASSED  \]|Failure' "$log/exec.txt" | head -3)"
    verdict="refuses it; the load of 0006/0007 is the owner's call (#812)"
fi

echo "== verdict: amdxdna $loaded, #812 pair $carrier: the import class advertises FOREIGN_API device"
echo "   execution and this module ${verdict} =="
exit $((fail))
