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
#
# npu_window11_compare.sh -- the device-free half of the gate row: what `npu-window11.sh --compare` must accept
# and must refuse, on hand-written logs.  No device, no model, no build products.
#
# The folded row (engine#876) rests on two extra assertions, and both are tested here because a gate that gets
# them wrong is worse than no gate: a folded log may only be compared against another folded log at the same
# tap layer, and while taps are present they have to be identical -- text equality alone is what let a wrong
# activation rotation through (#808), because a rotation defect keeps the text well-formed.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GW="$HERE/../scripts/npu-window11.sh"
[ -x "$GW" ] || { echo "no gate script at $GW"; exit 1; }
T=$(mktemp -d "${TMPDIR:-/tmp}/w11cmp.XXXXXX")
trap 'rm -rf "$T"' EXIT

# A log is: optional folded header, the three transcripts, the prof rows, then the taps.
mklog() {   # mklog <path> <folded:yes|no> <tap-hash> <text#> <ms>
    local p=$1 folded=$2 th=$3 text=$4 ms=$5
    : > "$p"
    [ "$folded" = yes ] && echo "folded layer=0 taps=2" >> "$p"
    for i in 1 2 3; do
        echo "text $i sha256=${text}$i chars=11 text=hello world" >> "$p"
    done
    echo "prof dec G M=16 ms=$ms launches=64" >> "$p"
    echo "prof dec O M=16 ms=0.500 launches=64" >> "$p"
    if [ "$folded" = yes ]; then
        echo "tap [pb16i-trace] G raw[0..3]=1 2 3 4 rot[0..3]=${th} 2 3 4 as=0.1 planes=1 A[0..7]=1 2 3 4 5 6 7 8 Cm[0..7]=9 10 11 12 13 14 15 16 out[0..7]=0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 eref[0..3]=1 2 3 4" >> "$p"
        echo "tap [qkv-dump] l=0 m=0 q0=1 q1=2 k0=3 v0=4 gate0=5" >> "$p"
    fi
}

run() {   # run <expect: pass|fail> <description> <A> <B>
    local want=$1 desc=$2 a=$3 b=$4 out rc
    # --tol 10 explicitly: since engine#834/#880 the script refuses a comparison with no measured tolerance, so
    # a case that leaves it out would "fail" (exit 2) for the wrong reason and the accept cases would never run.
    out=$("$GW" --compare "$a" "$b" --tol 10 2>&1); rc=$?
    if { [ "$want" = pass ] && [ $rc -eq 0 ]; } || { [ "$want" = fail ] && [ $rc -ne 0 ]; }; then
        echo "ok   $desc"
    else
        echo "FAIL $desc (expected $want, exit $rc)"; echo "$out" | sed 's/^/     /'; return 1
    fi
}

fails=0
mklog "$T/f1" yes A 0.700 1.100
mklog "$T/f2" yes A 0.700 1.100
mklog "$T/f3" yes B 0.700 1.100        # same text, one rotated activation differs
mklog "$T/f4" yes A 0.700 1.900        # same everything, ms moved 73 % (must NOT fail the folded row)
mklog "$T/f5" no  A 0.700 1.100        # not a folded run
mklog "$T/f6" no  A 0.700 1.100
mklog "$T/f7" no  A 0.700 1.900        # plain row, ms outside tolerance -> must fail
: > "$T/f8"; head -5 "$T/f1" > "$T/f8"  # folded header and transcripts, taps truncated away

run pass "two identical folded runs"                      "$T/f1" "$T/f2" || fails=$((fails+1))
run fail "one rotated activation differs"                 "$T/f1" "$T/f3" || fails=$((fails+1))
run pass "folded run, same taps, ms moved 73 %"           "$T/f1" "$T/f4" || fails=$((fails+1))
run fail "folded compared with a non-folded run"          "$T/f1" "$T/f5" || fails=$((fails+1))
run fail "folded run whose taps are missing"              "$T/f1" "$T/f8" || fails=$((fails+1))
run pass "two identical plain runs"                       "$T/f5" "$T/f6" || fails=$((fails+1))
run fail "plain run, ms outside tolerance still fails"    "$T/f5" "$T/f7" || fails=$((fails+1))

# A folded row's ms is reported but not asserted (f4 above) -- unless the caller supplies the same-binary control
# pair engine#834 asks for, in which case the tolerance is a measurement and it bites on the folded row too.
mklog "$T/c1" yes A 0.700 1.100
mklog "$T/c2" yes A 0.700 1.150       # same binary twice: worst row 4.5 % -> tolerance 5 %
out=$("$GW" --compare "$T/f1" "$T/f4" --control "$T/c1" "$T/c2" 2>&1); rc=$?
if [ $rc -ne 0 ] && echo "$out" | grep -q "outside tolerance"; then
    echo "ok   a measured control scores the folded row's ms (73 % move fails)"
else
    echo "FAIL a measured control did not score the folded row (exit $rc)"; echo "$out" | sed 's/^/     /'; fails=$((fails+1))
fi

# The comparison itself has to name the difference, not just exit non-zero.
"$GW" --compare "$T/f1" "$T/f3" --tol 10 2>&1 | grep -q "tap 0 differs" || { echo "FAIL the mismatch is not named"; fails=$((fails+1)); }
"$GW" --compare "$T/f1" "$T/f3" --tol 10 2>&1 | grep -q "A .*rot\[0..3\]=A" || { echo "FAIL the differing line is not shown"; fails=$((fails+1)); }

# The plumbing, not just the comparison: --folded has to reach the serve's environment, or the row silently compares
# two tap-free runs and passes on text alone -- the exact failure mode it exists to prevent.  A stub binary records
# its environment and exits, so this needs no device and no model.
STUB="$T/stub"
cat > "$STUB" <<'SH'
#!/bin/bash
env | grep -E '^NPU_(PB16I_TRACE|QKV_DUMP)=' | sort > "$STUB_ENV"
SH
chmod +x "$STUB"
: > "$T/fake.q4nx"; mkdir -p "$T/kernels"
STUB_ENV="$T/env.folded" ONEBIT_BIN="$STUB" timeout 30 "$GW" "$T/fake.q4nx" "$T/kernels" \
    --folded --tap-layer 3 --label stub --out "$T/stub.log" --port 1 >/dev/null 2>&1
if grep -q '^NPU_PB16I_TRACE=3$' "$T/env.folded" 2>/dev/null && \
   grep -q '^NPU_QKV_DUMP=3$' "$T/env.folded" 2>/dev/null; then
    echo "ok   --folded reaches the serve's environment (layer 3)"
else
    echo "FAIL --folded did not set the taps on the serve"; cat "$T/env.folded" 2>/dev/null | sed 's/^/     /'; fails=$((fails+1))
fi
STUB_ENV="$T/env.plain" ONEBIT_BIN="$STUB" timeout 30 "$GW" "$T/fake.q4nx" "$T/kernels" \
    --label stub --out "$T/stub2.log" --port 1 >/dev/null 2>&1
if [ -s "$T/env.plain" ]; then
    echo "FAIL a plain run set the taps ($(tr '\n' ' ' < "$T/env.plain"))"; fails=$((fails+1))
else
    echo "ok   a plain run leaves the taps off"
fi

[ "$fails" = 0 ] && echo "npu_window11_compare: PASS (10 cases)" && exit 0
echo "npu_window11_compare: $fails case(s) FAILED"; exit 1
