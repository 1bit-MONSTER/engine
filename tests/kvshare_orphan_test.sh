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
# The handoff driver must not be able to orphan its server. tests/kvshare_fd_handoff.py runs an HRX llama-server in
# its own session, so the `finally` that reaps it covers a clean exit, Ctrl-C and SIGTERM -- but not SIGKILL (OOM,
# a CI timeout, `kill -9`). An orphaned llama-server held 51 GiB of GTT for 14 h on this box on 2026-11-08 and
# knocked out another service, so the child is spawned with PR_SET_PDEATHSIG. This pins that: SIGKILL the parent
# and the child must go with it.
#
# No GPU and no model: the child is a sleep, spawned through the driver's own `die_with_parent`.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

python3 - "$here" > "$scratch/child.pid" <<'PY' &
import subprocess, sys, time
sys.path.insert(0, sys.argv[1])
import kvshare_fd_handoff as k
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                         preexec_fn=k.die_with_parent, start_new_session=True)
print(child.pid, flush=True)
time.sleep(120)
PY
parent=$!
for _ in $(seq 1 50); do [ -s "$scratch/child.pid" ] && break; sleep 0.1; done
child=$(cat "$scratch/child.pid" 2>/dev/null || true)
check "the driver spawned a child in its own session (pid ${child:-none})" '[ -n "$child" ] && kill -0 "$child" 2>/dev/null'

kill -9 "$parent" 2>/dev/null
wait "$parent" 2>/dev/null
gone=0
for _ in $(seq 1 40); do kill -0 "$child" 2>/dev/null || { gone=1; break; }; sleep 0.05; done
check "SIGKILL of the driver takes the child with it (PR_SET_PDEATHSIG)" '[ "$gone" = 1 ]'

# and the negative control: a child spawned *without* it survives, which is what the flag is protecting against
python3 - "$here" > "$scratch/leak.pid" <<'PY' &
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(child.pid, flush=True)
time.sleep(120)
PY
parent=$!
for _ in $(seq 1 50); do [ -s "$scratch/leak.pid" ] && break; sleep 0.1; done
leak=$(cat "$scratch/leak.pid" 2>/dev/null || true)
kill -9 "$parent" 2>/dev/null; wait "$parent" 2>/dev/null; sleep 0.3
check "without the flag the child is orphaned (the case this test exists for)" '[ -n "$leak" ] && kill -0 "$leak" 2>/dev/null'
kill -9 "$leak" 2>/dev/null

if [ $fail -ne 0 ]; then echo FAIL; exit 1; fi
echo PASS
