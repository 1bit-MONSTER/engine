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
# The shared-KV route (ONEBIT_KV_SHARE=1, docs/hybrid.md) without a model, a GPU or a dma-buf: the fake
# backend speaks the two endpoints the real llama-server has (fork 1bit-MONSTER/llama.cpp#164: ?action=share
# sends [uint32 length][JSON] plus the region fds over a unix socket and returns a share_id; ?action=adopt
# takes the tokens back and refuses a wrong share_id), and fake_kvshare_provider.py plays the provider: it
# fails unless it was handed one readable fd per region, which is only true if the engine received the fds
# over the socket and passed them on through exec.
#
# What this pins down that the GPU gate cannot (that one needs an HRX box): the listen -> share -> fds ->
# provider -> adopt order, the inherited descriptors, that adopt carries the share_id, that the socket is
# cleaned up, and the fallback when a provider fails.
#
# usage: tests/hybrid_kvshare_route.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: hybrid_kvshare_route.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
touch "$scratch/tiny.gguf"
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
pid=
cleanup() { [ -n "$pid" ] && { kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; }; rm -rf "$scratch"; }
trap cleanup EXIT

start() {   # start <log> [env...]
    local log=$1; shift
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    env FAKE_KVSHARE_MARKER="$scratch/provider.json" FAKE_SLOT_MARKER="$scratch/slot.json" "$@" "$bin" serve -m "$scratch/tiny.gguf" --device hybrid --port "$port" --alias hybrid-model \
        --llama-server "$here/fake_backend.py" --no-mem-check >"$log" 2>&1 &
    pid=$!
    api="http://127.0.0.1:$port"
    local code=000
    for _ in $(seq 1 100); do code=$(curl -s -o /dev/null -w '%{http_code}' "$api/health"); [ "$code" = 200 ] && break; sleep 0.1; done
    # A server that never came up because it could not bind is a port collision, not a defect in
    # what this test checks: name the port and stop, before an empty body turns into a JSON error.
    if grep -qiE 'EADDRINUSE|address already in use|cannot listen on' "$log"; then
        echo "FAIL: hybrid_kvshare_route: could not bind a port $port ($(grep -m1 -oiE 'EADDRINUSE|address already in use|cannot listen on [^ ]*' "$log" | head -1))"
        exit 1
    fi
    # this serve's OWN slot directory, from its child's argv: /tmp/1bit-hybrid-* is shared with every other serve
    # on the box, so globbing it reports another test's in-flight socket as this route leaking one (see the same
    # fix and the measurement in tests/hybrid_route.sh)
    slotdir=$(for c in $(pgrep -P "$pid" 2>/dev/null); do
        tr '\0' '\n' <"/proc/$c/cmdline" 2>/dev/null | grep -m1 '^/tmp/1bit-hybrid-'
    done | head -1)
    [ "$code" = 200 ]
}
stop() { kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; pid=; }
ask() { curl -s "$api/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model": "hybrid-model", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'; }

check "serve starts with the shared route on" \
    'start "$scratch/a.log" ONEBIT_KV_SHARE=1 ONEBIT_HYBRID_PREFILL="$here/fake_kvshare_provider.py"'
reply=$(ask)
check "the reply comes through" '[[ "$reply" == *Paris* ]]'
check "the request was pinned to slot 0" '[[ "$reply" == *"\"fake_id_slot\":0"* ]]'
sleep 0.3
# the share time is logged with %.3f ms, so a sub-millisecond match would assert the box's load, not the path
check "the handoff ran (share + adopt, not the file route)" 'grep -q "hybrid: .* share [0-9.]* ms, first byte" "$scratch/a.log"'
check "no fallback happened" '! grep -q "the backend prefills this request itself" "$scratch/a.log"'
check "the provider got one readable fd per region" \
    'python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d[\"fds\"]==1 and d[\"readable\"]==[16] and d[\"n_ctx\"]==4096 else 1)" "$scratch/provider.json" 2>/dev/null'
# engine#941: the decoder is left the prompt's last token and adopt says so. Request shape only: the backend
# side of `leave_one_token` is fork 1bit-MONSTER/llama.cpp#189, which the pin does not carry yet
check "adopt asks the decoder to leave one token" \
    'python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d[\"action\"]==\"adopt\" and d[\"body\"].get(\"leave_one_token\") is True else 1)" "$scratch/slot.json" 2>/dev/null'
check "the socket is gone from the slot directory" '[ -n "$slotdir" ] && ! ls "$slotdir"/kvshare-*.sock 2>/dev/null | grep -q .'
stop

# the adopt must carry the share_id: the fake backend refuses a wrong or missing one, so a provider that runs
# but whose id does not come back leaves the request to the backend -- that is the fallback this checks
check "serve starts with a provider that refuses the protocol" \
    'start "$scratch/b.log" ONEBIT_KV_SHARE=1 ONEBIT_HYBRID_PREFILL=/bin/false'
reply=$(ask)
check "the reply still comes through" '[[ "$reply" == *Paris* ]]'
check "the fallback is named once" 'grep -q "the backend prefills this request itself" "$scratch/b.log"'
check "no share/adopt line was reported" '! grep -q "hybrid: .* share [0-9.]* ms, first byte" "$scratch/b.log"'
stop

# a provider that finishes its write into the region *after* its deadline: the deadline starts a teardown, it is
# not a verdict on the work. The provider takes the SIGTERM, writes the region it was handed two seconds later and
# exits 0 -- the handoff must still be adopted, or the request pays for the provider's prefill and then for the
# decoder to re-process the whole prompt (engine#941).
cat > "$scratch/late_provider.py" <<'PY'
#!/usr/bin/env python3
import json, os, signal, sys, time

args = sys.argv[1:]
opt = {args[i]: args[i + 1] for i in range(0, len(args) - 1, 2)}
ids = [int(line) for line in open(opt["--tokens"]) if line.strip()]
header = json.load(open(opt["--layout"]))
fds = [int(x) for x in opt["--fds"].split(",") if x]


def finish():
    regions = header.get("regions") or []
    readable = []
    for fd, region in zip(fds, regions):
        data = os.pread(fd, min(16, int(region.get("size", 1))), int(region.get("offset", 0)))
        if not data:
            sys.exit(1)
        readable.append(len(data))
    with open(os.environ["FAKE_KVSHARE_MARKER"], "w") as f:
        json.dump({"fds": len(fds), "readable": readable, "n_ctx": (header.get("params") or {}).get("n_ctx")}, f)
    print(json.dumps({"n_tokens": len(ids), "prefill_ms": 2000.0, "fds": len(fds)}), flush=True)
    sys.exit(0)


def term(*_a):
    time.sleep(2.0)                     # the write into the decoder's region, completed inside the grace window
    finish()


signal.signal(signal.SIGTERM, term)
time.sleep(30)
finish()
PY
chmod +x "$scratch/late_provider.py"
rm -f "$scratch/provider.json"
check "serve starts with a provider that finishes after its deadline" \
    'start "$scratch/c.log" ONEBIT_KV_SHARE=1 ONEBIT_HYBRID_PREFILL="$scratch/late_provider.py" ONEBIT_HYBRID_PREFILL_TIMEOUT_S=1'
reply=$(ask)
check "a late provider's cells are adopted, not discarded" '[[ "$reply" == *"\"fake_id_slot\":0"* ]]'
check "no fallback for work that landed" '! grep -q "the backend prefills this request itself" "$scratch/c.log"'
check "the share+adopt line was reported" 'grep -q "hybrid: .* share [0-9.]* ms, first byte" "$scratch/c.log"'
check "the late provider could still read its region fd" \
    'python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d[\"fds\"]==1 and d[\"readable\"]==[16] else 1)" "$scratch/provider.json" 2>/dev/null'
stop

# a decoder that cannot share its cache: the real server refuses a params or model mismatch, and refuses a
# hybrid cache on a pin without the recurrent-state arm. ONEBIT_KV_SHARE=1 must not turn that into "no
# provider at all" -- the state-file route the decoder does support is used instead, because a handoff by
# file still beats the decoder prefilling the prompt itself. share_ms separates the two routes in the log
# line (0.000 = the file route).
check "serve starts against a decoder that cannot share" \
    'start "$scratch/d.log" ONEBIT_KV_SHARE=1 FAKE_KVSHARE_REFUSE=1 ONEBIT_HYBRID_PREFILL="$here/fake_prefill_provider.py" FAKE_PREFILL_COPY=1'
reply=$(ask)
check "the reply comes through on the state-file route" '[[ "$reply" == *Paris* ]]'
check "the request was pinned to slot 0" '[[ "$reply" == *"\"fake_id_slot\":0"* ]]'
check "the fallback to the state file is named" 'grep -q "cannot share its KV" "$scratch/d.log"'
check "the provider's state file was restored (share 0.000 ms, the file route)" 'grep -q "hybrid: .* share 0\.000 ms, first byte" "$scratch/d.log"'
check "the decoder was not left to prefill the prompt itself" '! grep -q "the backend prefills this request itself" "$scratch/d.log"'
stop

# a provider that exits 0 but says it prefilled fewer tokens than it was handed: the region's last row was never
# written, so the cells must NOT be adopted -- the decoder prefills the prompt itself instead. The file route has
# carried this cross-check since #941; the shared route adopted on the cell count alone.
check "serve starts with a provider that claims a short prefill" \
    'start "$scratch/f.log" ONEBIT_KV_SHARE=1 FAKE_KVSHARE_SHORT=1 ONEBIT_HYBRID_PREFILL="$here/fake_kvshare_provider.py"'
reply=$(ask)
check "the reply still comes through" '[[ "$reply" == *Paris* ]]'
check "a short claim is named" 'grep -q "provider prefilled [0-9]* of [0-9]* tokens into the shared region" "$scratch/f.log"'
check "no cells were adopted from a short region" '! grep -q "share [0-9.]* ms, first byte" "$scratch/f.log"'
check "the decoder prefills the prompt itself instead" 'grep -q "the backend prefills this request itself" "$scratch/f.log"'
stop

# the other side of the same boundary: the export worked and the provider wrote into the region, but the cells
# cannot be adopted (the real refusal when another sequence holds them). That is not a handshake failure, so the
# engine's designed behaviour is the plain fallback -- it must NOT re-run the prompt through the state-file route.
rm -f "$scratch/provider.json"
check "serve starts with a decoder that refuses the adopt" \
    'start "$scratch/e.log" ONEBIT_KV_SHARE=1 FAKE_KVSHARE_ADOPT_REFUSE=1 ONEBIT_HYBRID_PREFILL="$here/fake_kvshare_provider.py"'
reply=$(ask)
check "the reply comes through when the adopt is refused" '[[ "$reply" == *Paris* ]]'
check "the provider did run and wrote into the region" \
    'python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d[\"fds\"]==1 else 1)" "$scratch/provider.json" 2>/dev/null'
check "the refused adopt is named" 'grep -q "adopt answered 400" "$scratch/e.log"'
check "an adopt refusal does not fall back to the state file" '! grep -q "cannot share its KV" "$scratch/e.log"'
stop

if [ $fail -ne 0 ]; then echo "--- log (tail)"; tail -30 "$scratch/a.log"; echo FAIL; exit 1; fi
echo PASS
