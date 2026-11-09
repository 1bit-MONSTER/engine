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
# The port read-backs in tests/serve_port.sh (engine #1485): two servers, two log lines, one job -- hand a test the
# port that was actually bound instead of one it picked and released.
#
# No server, no device, no model: the parsers are fed the exact lines the two servers print (app/serve.cpp and
# tools/server/server-http.cpp), and the failing wait is driven with a process that is already gone. This is the
# test that would have caught the end-anchored pattern #1523/#1525 shipped, which matched one of serve's three
# ready-line shapes and returned empty for the other two.
set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=tests/serve_port.sh
. "$root/tests/serve_port.sh"

fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=$(( fail + 1 )); fi; }
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

# 1bit serve: three ready-line shapes, all of which must yield the port (the #1523/#1525 finding).
printf '1bit serve: ready on http://127.0.0.1:41235\n' > "$scratch/serve-plain"
printf '1bit serve: ready on http://127.0.0.1:41236 (laya routes each request; 3 models)\n' > "$scratch/serve-laya"
printf '1bit serve: ready on http://127.0.0.1:41237 (laya-models: model auto is routed)\n' > "$scratch/serve-layamodels"
check "serve: the plain ready line yields its port" \
      '[ "$(serve_port_from_log "$scratch/serve-plain")" = 41235 ]'
check "serve: a suffixed ready line yields the same field" \
      '[ "$(serve_port_from_log "$scratch/serve-laya")" = 41236 ]'
check "serve: the laya-models suffix does not hide the port" \
      '[ "$(serve_port_from_log "$scratch/serve-layamodels")" = 41237 ]'

# --port 0 with an IPv6 --host: the port is the field after the URL's LAST colon.
printf '1bit serve: ready on http://::1:41238\n' > "$scratch/serve-ipv6"
check "serve: an IPv6 host keeps the port (last colon, not first)" \
      '[ "$(serve_port_from_log "$scratch/serve-ipv6")" = 41238 ]'

# llama-server (the vendored llama.cpp) in its own words: SRV_INF prefixes the line with "srv  <func>: ".
printf 'srv  load_model: loaded the model in 1.2 s\nsrv  http_bind: listening on http://127.0.0.1:41239\n' \
    > "$scratch/llama"
check "llama-server: the listening line yields its port" \
      '[ "$(llama_port_from_log "$scratch/llama")" = 41239 ]'

# ...and it must keep working when the line carries a suffix, since that is the shape a future change adds.
printf 'srv  http_bind: listening on http://127.0.0.1:41241 (tls off)\n' > "$scratch/llama-suffix"
check "llama-server: a suffix after the address does not hide the port" \
      '[ "$(llama_port_from_log "$scratch/llama-suffix")" = 41241 ]'

# A log without the line yields nothing -- the failure this must not paper over with a default.
printf 'srv  load_model: loading\n' > "$scratch/notyet"
check "neither parser invents a port before the server says" \
      '[ -z "$(llama_port_from_log "$scratch/notyet")" ] && [ -z "$(serve_port_from_log "$scratch/notyet")" ]'

# The wait stops when the server is gone instead of sleeping out the whole budget, and reports failure.
false &
dead=$!
wait "$dead" 2>/dev/null
start=$(date +%s)
if llama_port_wait "$scratch/notyet" "$dead" 600; then
    check "wait: a dead server is a failure, not a port" 'false'
else
    elapsed=$(( $(date +%s) - start ))
    check "wait: a dead server is a failure, not a port" '[ "$elapsed" -lt 5 ]'
fi

# The wait returns the port as soon as the line appears, without waiting for the budget to run out.
( sleep 0.3; printf 'srv  http_bind: listening on http://127.0.0.1:41242\n' >> "$scratch/late" ) &
appender=$!
sleep 30 &
alive=$!
start=$(date +%s)
got=$(llama_port_wait "$scratch/late" "$alive" 600)
elapsed=$(( $(date +%s) - start ))
kill "$alive" "$appender" 2>/dev/null; wait 2>/dev/null
check "wait: the port is returned as soon as it is printed (${elapsed}s, got ${got:-none})" \
      '[ "$got" = 41242 ] && [ "$elapsed" -lt 5 ]'

if [ "$fail" = 0 ]; then
    echo "serve_port_test: all checks passed"
else
    echo "serve_port_test: $fail check(s) failed"
fi
exit "$fail"
