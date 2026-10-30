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

# engine#612: the one entry point for the demo. Each row is a script of its own, because each needs a
# different build and a different device; this drives them and says what each row is.
#
#   tools/showcase-612.sh rows            what rows exist, what each proves, and how to build for it
#   tools/showcase-612.sh npu-streams     row 1: two opted-in NPU streams (--npu-streams 2) vs solo
#   tools/showcase-612.sh hrx-batched N.. row 2: 1bit serve --parallel N aggregate, HRX
#   tools/showcase-612.sh hybrid          row 3: what the hybrid handoff costs, and the verdict
#
# A row's script is only on the branch that measured it until the pull request merges: the message
# says which one when a script is missing. Everything a row prints carries its pin, its repeat count
# and its timestamp -- that is the point of the demo, not the numbers themselves.
set -uo pipefail

dir=$(dirname "$0")
rows() {
    cat <<'TXT'
row            script                              build                                    device
npu-streams    tools/showcase-612-npu-streams.sh   -DONEBIT_NPU=ON (needs the NPU runtime)   accel0 (NPU)
hrx-batched    tools/showcase-612-hrx.sh           -DONEBIT_HRX=ON (TheRock)                HRX0 (iGPU)
hybrid         docs/hybrid.md "Status"             -DONEBIT_HRX=ON .. --prefill-provider    HRX0 + a prefill provider

The NPU row needs /tmp/1bit-npu-device.lock free: it waits for it and never takes it from a running
job. GPU rows belong under ~/.local/bin/gpu-hold -l <label> -t <minutes> -- (the fleet's fair queue).
TXT
}

case "${1:-rows}" in
    rows) rows ;;
    npu-streams)
        script=$dir/showcase-612-npu-streams.sh
        [ -x "$script" ] || { echo "no $script: it is in pull request #640 (engine#612 item 1) until that merges"; exit 1; }
        shift
        exec "$script" "$@"
        ;;
    hrx-batched)
        script=$dir/showcase-612-hrx.sh
        [ -x "$script" ] || { echo "no $script: it is in pull request #643 (engine#612 item 2) until that merges"; exit 1; }
        shift
        [ $# -ge 1 ] || { echo "usage: $0 hrx-batched <parallel> [parallel ...]   e.g. 2 4 8"; exit 1; }
        exec "$script" "$@"
        ;;
    hybrid)
        echo "The hybrid row has no script of its own: at this pin there is no NPU prefill provider to"
        echo "measure, only the handoff, and that was measured once by hand. The numbers and the reason"
        echo "the row is 'not at this pin' are in docs/hybrid.md, section Status."
        ;;
    *) echo "unknown row '$1': try '$0 rows'"; exit 1 ;;
esac
