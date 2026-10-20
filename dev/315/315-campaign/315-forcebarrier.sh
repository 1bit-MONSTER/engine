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
# engine#315: the last live mechanism family, tested directly - intra-submission packet ordering.
#
# The reasoning this rests on:
#   * the driver does NOT infer memory hazards; it trusts recorded flags
#     (`aql_block_processor.c:126` "Do not infer memory hazards here by walking command operands"), and
#     `..._packet_has_barrier` (:103-116) sets the AQL barrier bit only for EXECUTION_BARRIER packets, FINAL
#     packets, and packet 0 of a submission with waits or an inline acquire;
#   * the queue profile of GLM's decode shows **92 of 612 dispatches carry no barrier**
#     (`evidence/315-queue-profile.md`), i.e. 15% of the program may overlap a neighbour;
#   * every lever tested so far operates at *submission* granularity - `GRAPH_SYNC` after each replay
#     (`315gsync2`: control 1 fault in 4, fix 0 in 2), the host read path (`scacquire` + covered timepoint),
#     and coherence (`315accept`: 9 clean + 1 fault in 10). None of them orders two packets *inside* one
#     submission, which is exactly where this family lives;
#   * `HRX_COMMAND_BUFFER_MODE=pm4`, the other lever on the packet mechanism, is unusable: all four of its arms
#     failed the FIRST request with HTTP 500 (`315pm4`), so it cannot compare fault rates.
#
# So: force every payload packet to be barrier-bearing, env-gated and off by default, and run the paired A/B.
# Both cells run the SAME build (so the only difference is the env var), with the same inducer.
#
# Readout, pre-committed:
#   fault rate falls in the forced-barrier cell   -> intra-submission overlap IS the carrier; the mechanism is
#       "a missing recorded execution dependency in the generated command stream", and the next step is to name
#       the specific producer/consumer pair by enumerating the unbarriered dispatches on the layer-0 path
#   fault rate unchanged                          -> packet ordering inside a submission is not the carrier either,
#       and the ordering family is closed completely (submission-level AND packet-level)
#
# Diagnostic, not acceptance: no registered criterion changes and no arm from it is eligible.
#
# Usage: 315-forcebarrier.sh [pairs]
set -uo pipefail
PAIRS=${1:-4}
TAG=315fbar
SRC=$HOME/wt/llama-315copy
HRXSRC=$HOME/wt/hrx-forcebarrier
B=$HOME/wt/315fbar-build
LOG=$HOME/wt/315fbar-build.log

exec 9>"$HOME/.cache/lax-decode/box.lock"
if flock -w "${FBAR_LOCK_WAIT:-10800}" 9; then
  echo "fbar: box lock held $(date -Is)"
else
  echo "fbar: lock unobtainable - aborting rather than building concurrently"
  exit 3
fi

if [ ! -f "$HRXSRC/runtime/src/iree/hal/drivers/amdgpu/aql_block_processor.c" ]; then
  echo "fbar: copying the runtime tree to $HRXSRC $(date -Is)"
  rm -rf "$HRXSRC"
  cp -a "$HOME/wt/pin-fork-hrx/third_party/hrx-system" "$HRXSRC" || { echo "fbar: copy failed"; exit 1; }
fi

python3 - "$HRXSRC/runtime/src/iree/hal/drivers/amdgpu/aql_block_processor.c" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text()
if "HRX_FORCE_BARRIERS" in t:
    print("fbar: patch already present"); raise SystemExit(0)
anchor = """static bool iree_hal_amdgpu_aql_block_processor_packet_has_barrier(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    uint32_t packet_index,
    iree_hal_amdgpu_aql_block_processor_packet_flags_t packet_flags) {"""
add = anchor + """
  // engine#315 diagnostic: HRX_FORCE_BARRIERS=1 makes every payload packet barrier-bearing, so a fault that
  // needs two unbarriered packets to overlap disappears when it is set. Off unless the variable is set to a
  // non-empty value other than "0"; nothing changes for any other run.
  {
    const char* hrx_force_barriers = getenv("HRX_FORCE_BARRIERS");
    if (hrx_force_barriers != NULL && hrx_force_barriers[0] != '\\0' && hrx_force_barriers[0] != '0') {
      return true;
    }
  }"""
assert t.count(anchor) == 1, "anchor not unique"
t = t.replace(anchor, add)
if "#include <stdlib.h>" not in t:
    i = t.find("#include")
    t = t[:i] + "#include <stdlib.h>\n" + t[i:]
p.write_text(t); print("fbar: patched", p)
PY
grep -c HRX_FORCE_BARRIERS "$HRXSRC/runtime/src/iree/hal/drivers/amdgpu/aql_block_processor.c" || { echo "fbar: patch missing"; exit 1; }

echo "fbar: configuring $B $(date -Is)"
rm -rf "$B"
cmake -S "$SRC" -B "$B" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm-therock/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-therock/bin/amdclang++ \
  -DGGML_HIP=OFF -DGGML_HRX=ON -DGGML_VULKAN=OFF \
  -DHRX_SOURCE_DIR="$HRXSRC" \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DBUILD_TESTING=OFF \
  > "$LOG" 2>&1 || { echo "fbar: cmake failed"; tail -20 "$LOG"; exit 1; }
echo "fbar: building -j 6 $(date -Is)"
cmake --build "$B" -j 6 >> "$LOG" 2>&1 || { echo "fbar: build failed"; tail -30 "$LOG"; exit 1; }
"$B/bin/llama-server" --version || exit 1

# Fix-cell launcher: same binary, one env var.
W=$HOME/wt/315fbar-on-bin
rm -rf "$W"; mkdir -p "$W"
for f in "$B"/bin/*; do b=$(basename "$f"); if [ "$b" != "llama-server" ]; then ln -sf "$f" "$W/$b"; fi; done
printf '%s\n' "#!/bin/sh" "# engine#315: forced-barrier cell - same binary as the control, one env var." "export HRX_FORCE_BARRIERS=1" "exec $B/bin/llama-server \"\$@\"" > "$W/llama-server"
chmod +x "$W/llama-server"

echo "fbar: running the paired A/B $(date -Is)"
sleep 2
CBIN="$B/bin" FBIN="$W" NOLOCK=1 \
REGIME_LO=${REGIME_LO:-80} REGIME_HI=${REGIME_HI:-92} ARM_TIMEOUT=${ARM_TIMEOUT:-1800} \
  bash "$HOME/wt/315-attrib.sh" "$PAIRS" "$TAG" 1 \
  "GGML_HRX_TENSOR_FP=tail:16 GGML_HRX_TENSOR_FP_NANONLY=1"

c=$HOME/wt/$TAG-console.log
echo "=== fbar readout $(date -Is) ==="
grep -hE "^  pair" "$c" || true
echo "control(no env): OK=$(grep -c 'arm c: OK' "$c" || true) FAULT=$(grep -c 'arm c: \(FAULT\|INVALID\)' "$c" || true) OOR=$(grep -c 'arm c: OOR' "$c" || true)"
echo "fix(FRB=1):      OK=$(grep -c 'arm l: OK' "$c" || true) FAULT=$(grep -c 'arm l: \(FAULT\|INVALID\)' "$c" || true) OOR=$(grep -c 'arm l: OOR' "$c" || true)"
echo "A fall in the fix cell names intra-submission overlap as the carrier; no change closes the ordering family."
