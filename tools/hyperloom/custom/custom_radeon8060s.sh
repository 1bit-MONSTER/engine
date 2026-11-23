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
# Hyperloom 1.2.0 custom-workload entrypoint: the engine's HRX route (llama-bench + `1bit serve` + the KLD /
# greedy-text gate in benchmark.py), measured on the Strix Halo box against the engine payload installed there.
#
# Hyperloom runs on the orchestration host and resolves this file by board name (`--gpu-type radeon8060s`). One round =
# one `box-hold gpu` ticket over ssh (../box_ticket.sh): the levers are validated here, benchmark.py + box_bench.sh are
# shipped to the box, box_bench.sh checks RAM and GTT room and runs benchmark.py, and the report benchmark.py wrote on
# the box becomes $RESULT_DIR/$RESULT_FILENAME.json here.
#
# MODEL (Hyperloom's --model, or HL_BOX_MODEL when --model is a local stand-in) is a path ON THE BOX. The gate's references live on the box too (HL_STATE, default
# $HL_BOX_DIR/state/hrx-<model file name>), because the KLD base logits are recorded and compared where they are made.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_TAG=hl-hrx
BOXPATH='^/[A-Za-z0-9._/+-]+$'   # an absolute path on the box, in characters that cannot break the ssh command
declare -A LEVERS=(
    [HL_B]='^[0-9]{2,5}$'
    [HL_UB]='^[0-9]{2,5}$'
    [HL_FA]='^(on|off|auto)$'
    [HL_CTK]='^(f16|bf16|q8_0|q4_0|q4_1|q5_0|q5_1|iq4_nl)$'
    [HL_CTV]='^(f16|bf16|q8_0|q4_0|q4_1|q5_0|q5_1|iq4_nl)$'
    [HL_MTP_MAX]='^[0-9]$'
    [HL_MTP_HEAD]=$BOXPATH
    [HL_DFLASH]='^[01]$'
    [HL_DFLASH_GGUF]=$BOXPATH
    [HL_PARALLEL]='^[1-8]$'
    [HL_CTX]='^[0-9]{3,7}$'
    [HL_LONGCTX]='^[0-9]{3,7}$'
    [HL_LONGTOK]='^[0-9]{1,5}$'
    [HL_REPS]='^[1-5]$'
    [HL_TOKENS]='^[0-9]{1,4}$'
    [GGML_HRX_Q8_PREFILL_RELAX]='^[01]$'
    [GGML_HRX_GRAPH_PROGRAM_CACHE]='^[0-9]{1,6}$'
    [ONEBIT_HRX_DECODE_SPLIT]='^[0-9]{1,3}$'
    # pins, not levers: where the box keeps the build, the weights, the wiki text and the references; the job's RAM need
    [MODEL]=$BOXPATH
    [HL_BOX_MODEL]=$BOXPATH
    [HL_PACKAGE]=$BOXPATH
    [HL_WIKI]=$BOXPATH
    [HL_STATE]=$BOXPATH
    [HL_RAM_NEED_GB]='^[0-9]{1,3}$'
    [HL_GTT_NEED_GIB]='^[0-9]{1,3}$'
)
# HL_BOX_MODEL (a pin) names the GGUF on the box when Hyperloom's --model is a local stand-in: 1.2.0 resolves --model
# on the orchestration host (precision, serving path), where the weights are not.
[ -n "${HL_BOX_MODEL:-}" ] && export MODEL=$HL_BOX_MODEL
. "$HERE/../box_ticket.sh"
: "${MODEL:?Hyperloom sets MODEL (--model): a GGUF path on the box}"
export HL_WIKI=${HL_WIKI:-$HL_BOX_DIR/wiki.test.raw}
export HL_STATE=${HL_STATE:-$HL_BOX_DIR/state/hrx-$(basename "$MODEL" .gguf)}

hl_levers
hl_ship box_bench.sh benchmark.py
hl_ticket bash "$RDIR/box_bench.sh" "$ROUT"
hl_fetch
if [ -f "$RESULT_DIR/box/inferencex_result.json" ]; then
    cp "$RESULT_DIR/box/inferencex_result.json" "$RESULT_DIR/$RESULT_FILENAME.json"
else
    hl_fail_report "unmeasured: $(cat "$RESULT_DIR/box/status")"
fi
python3 - "$RESULT_DIR/$RESULT_FILENAME.json" "$RESULT_DIR/box/status" "$LEVERS_JSON" "$TICKET" <<'EOF' || exit 1
import json, sys
path, status, levers, ticket = sys.argv[1:]
rep = json.load(open(path))
bs = rep.setdefault("bench_summary", {})
bs["box_status"] = open(status).read().strip()
bs["levers_requested"] = json.loads(levers)
bs["ticket"] = ticket
try:
    bs["ram"] = open(status.rsplit("/", 1)[0] + "/ram.txt").read().strip().splitlines()
    bs["ram_low_gib"] = int(open(status.rsplit("/", 1)[0] + "/ram.low").read().strip())
except (OSError, ValueError):
    pass
json.dump(rep, open(path, "w"), indent=1)
print("[hl-hrx] output_throughput=%s gate=%s (%s)" % (rep.get("output_throughput"),
      "PASS" if rep.get("quality_gate", {}).get("passed") else "FAIL", rep.get("quality_gate", {}).get("reason")))
EOF
exit 0
