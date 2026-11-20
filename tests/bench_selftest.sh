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
# tools/bench.py against `1bit serve` with tests/fake_backend.py behind it, without a GPU: two
# configs over two interleaved rounds, a table with the baseline and a delta, the JSON record,
# and no serve left running afterwards.
#
# usage: tests/bench_selftest.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: bench_selftest.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

# engine#2858: a bench table is only comparable to another if both name the build they came from. The JSON must carry a
# provenance stanza, and an unavailable value must be the explicit string "unknown" rather than an empty field a reader would
# take for a value -- the same rule `recorded_at` learned the hard way. No server needed: this asserts the helper directly.
python3 - "$here/../tools" <<'INNER'
import sys, subprocess
sys.path.insert(0, sys.argv[1])
import bench
p = bench.provenance()
real = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
ok = p["engine_rev"] == real and p["engine_dirty"] in ("yes", "no")
print("ok   bench.py names the build it measured" if ok else
      "FAIL bench.py provenance: engine_rev %r vs git %r, dirty %r" % (p["engine_rev"], real, p["engine_dirty"]))
q = bench.provenance("/nonexistent/binary-xyz", repo="/tmp")
ok2 = q["binary_sha256"] == "unknown" and q["engine_rev"] == "unknown"
print("ok   an unavailable provenance value is the explicit string unknown" if ok2 else
      "FAIL provenance of a missing binary/absent repo should be 'unknown', got %r" % (q,))
sys.exit(0 if (ok and ok2) else 1)
INNER

python3 - "$scratch/plain.gguf" <<'PY'
import struct, sys
def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
kv = s("general.architecture") + struct.pack("<I", 8) + s("llama")
open(sys.argv[1], "wb").write(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + kv)
PY

out=$(python3 "$here/../tools/bench.py" -m "$scratch/plain.gguf" --onebit "$bin" --rounds 2 --prompt-reps 2 \
      --decode-reps 1 --decode-tokens 8 --prompt-chars 2000 --log-dir "$scratch/logs" --json "$scratch/out.json" \
      --config "a=--device cpu --llama-server $here/fake_backend.py" \
      --config "b=--device cpu --llama-server $here/fake_backend.py --no-recipes" 2>"$scratch/err")
echo "$out"
check "a row per config" '[[ "$out" == *"| a | "* ]] && [[ "$out" == *"| b | "* ]]'
check "the second is measured against the first" '[[ "$out" == *"(+0.0%)"* ]]'
check "the JSON holds 2 rounds x 2 prompt runs per config" \
    '[ "$(python3 -c "import json; r=json.load(open(\"$scratch/out.json\")); print(len(r[\"results\"][\"b\"][\"prompt\"]))")" = 4 ]'
check "no serve left running" '! pgrep -f "serve -m $scratch/plain.gguf" >/dev/null'

if [ $fail -ne 0 ]; then cat "$scratch/err"; for f in "$scratch"/logs/*.log; do echo "--- $f"; cat "$f"; done; echo FAIL; exit 1; fi
echo PASS
