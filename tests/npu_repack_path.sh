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
# Where `1bit serve -m x.gguf --device npu` finds the GGUF -> Q4NX repack script (the repack runs
# before anything touches the device, so this needs no NPU):
#   - a package keeps the script beside 1bit (scripts/package-linux.sh): that copy is used,
#   - ONEBIT_Q4NX_REPACK still overrides it,
#   - with neither, the error names the path it looked for and the override that fixes it.
# A release package shipped without the script died at the first GGUF with "repack script not
# found: ../src/scripts/repack_gguf.py", a build-machine path (docs/npu.md, "From a GGUF").
# A build without the NPU lane (public CI) has no repack code at all and refuses a .gguf on
# --device npu first, so the test skips there (77, SKIP_RETURN_CODE) instead of failing.
#
# usage: tests/npu_repack_path.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: npu_repack_path.sh path/to/1bit}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; [ -n "${3:-}" ] && echo "     $3"; fail=1; fi; }

python3 - "$scratch" <<'PY'
import struct, sys
def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
kv = s("general.architecture") + struct.pack("<I", 8) + s("qwen3")
open(sys.argv[1] + "/m.gguf", "wb").write(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + kv)
PY

# a stand-in for the repack: record how it was called and fail, so serve stops there
cat > "$scratch/stub.py" <<'PY'
import os, sys
open(os.environ["REPACK_MARKER"], "w").write(" ".join(sys.argv[1:]))
sys.exit(1)
PY

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')

# a copy of 1bit on its own, the way a package installs it (the binary and the libraries beside it)
install() {  # install <dir>
    mkdir -p "$1"
    cp "$bin" "$1/1bit"
    cp "$(dirname "$bin")"/*.so* "$1/" 2>/dev/null
}

# serve_npu <dir with 1bit> <marker> [extra env...]: one --device npu attempt, output in $scratch/out.log
serve_npu() {
    local dir=$1 marker=$2; shift 2
    # from a directory with no ../src beside it: a build tree's baked fallback is relative, and a
    # caller's working directory must not decide whether the script is found
    local cwd=$scratch/cwd; mkdir -p "$cwd"
    ( cd "$cwd" && env REPACK_MARKER="$marker" ONEBIT_Q4NX_CACHE="$scratch/cache" "$@" \
        timeout 60 "$dir/1bit" serve -m "$scratch/m.gguf" --device npu --port "$port" > "$scratch/out.log" 2>&1 )
    echo $?
}

install "$scratch/plain"

# 0. the probe, with no script anywhere: classify the build before asserting anything about it
rc=$(serve_npu "$scratch/plain" "$scratch/none.marker")
cp "$scratch/out.log" "$scratch/probe.log"
if grep -qE "cannot run a \.gguf|has no NPU lane" "$scratch/probe.log"; then
    echo "skip: this build has no NPU lane, so it has no GGUF repack path: $(head -1 "$scratch/probe.log")"
    exit 77
fi

# 1. the packaged layout: the copy beside 1bit is the one that runs
install "$scratch/pkg"
mkdir -p "$scratch/pkg/scripts"
cp "$scratch/stub.py" "$scratch/pkg/scripts/repack_gguf.py"
rc=$(serve_npu "$scratch/pkg" "$scratch/pkg.marker")
check "a package's scripts/repack_gguf.py beside 1bit is the one that runs" \
      '[ -f "$scratch/pkg.marker" ] && grep -q "m.gguf" "$scratch/pkg.marker"'
check "  (the run stops at the stand-in: exit 1, GGUF repack failed)" \
      '[ "$rc" != 0 ] && grep -q "GGUF repack failed" "$scratch/out.log"'

# 2. ONEBIT_Q4NX_REPACK overrides it, packaged or not
rc=$(serve_npu "$scratch/plain" "$scratch/env.marker" ONEBIT_Q4NX_REPACK="$scratch/stub.py")
check "ONEBIT_Q4NX_REPACK overrides the path" '[ -f "$scratch/env.marker" ]'

# 3. neither (the probe): the error says which path it looked for, and what to set
check "with no script anywhere, serve says so (exit $rc)" \
      '[ -f "$scratch/probe.log" ] && grep -q "Q4NX repack script not found" "$scratch/probe.log" && grep -q "set ONEBIT_Q4NX_REPACK" "$scratch/probe.log"' \
      "it said: $(head -2 "$scratch/probe.log" 2>/dev/null | tr '\n' ' ')"
check "  and it did not claim a build-machine path exists" '[ ! -f "$scratch/none.marker" ]'

[ $fail = 0 ] && echo "npu_repack_path: PASS"
exit $fail
