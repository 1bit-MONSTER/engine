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
# demo-seq-probe.sh - the take's own model sequence, N times, as a gate.
#
# Cycle 30 item 4, and it exists because of the c29 take: on that stick ONE NPU client
# (Qwen3-0.6B-NPU-1bit) answered the game's quick turn at 22:16:47 and then died at 22:17:41 on its NEXT
# request - the picture page's prompt, `max_tokens 96`, thinking off - with
#   `npu runtime: invalid call (code 614): state 8, run 0 of 28`
# on the very first dispatch of that request, followed by an `aie2_set_cmd_timeout` TDR (power-cycle, one
# context skipped). The image page then fell back to the 4B, so the take shipped a picture whose prompt the
# NPU did not write. Neither a long context nor two NPU clients was involved; what the two turns have in
# common is a model switch in between (max_models.llm=1, so loading the 4B evicts the NPU).
#
# So this probe does exactly that sequence, in the demo pages' own request shapes (copied from
# ryzen:~/demo/game.py narrate() and ~/demo/img.py expand(), which is what the take ran):
#   1. the game's quick turn on the NPU       (system + NPU_SUFFIX, max_tokens 80)
#   2. a turn on the big model                (evicts the NPU's backend: the switch)
#   3. the picture page's prompt on the NPU    (max_tokens 96, thinking off)  <- the c29 failure
# A round PASSES only if all three answered 200 AND step 3's answer came from the NPU model - a fallback
# answer is a FAIL here even though the page deliberately falls back, because the gate is "the NPU wrote it".
#
# Usage: demo-seq-probe.sh [rounds] [--url BASE] [--npu MODEL] [--big MODEL] [--dmesg FILE] [--state TEXT]
#   --url     Lemonade API base       (default $LEMONADE, then http://1bit-os.local:13305/api/v1)
#   --npu     the NPU model id        (default $NPU_MODEL, then Qwen3-0.6B-NPU-1bit)
#   --big     the evicting model id   (default $BIG_MODEL, then Qwen3-4B-Instruct-2507-1bit)
#   --dmesg   kernel log to count TDRs in (default: dmesg(1) if readable; "-" skips TDR counting)
#   --state   the game STATE text     (default: a short boiler-room line)
# Run it ON the appliance (dmesg readable) or from the LAN with --url; exit 0 when every round passed,
# 1 when any did not, 2 on usage/setup error.
set -uo pipefail

rounds=10
url=${LEMONADE:-http://1bit-os.local:13305/api/v1}
npu=${NPU_MODEL:-Qwen3-0.6B-NPU-1bit}
big=${BIG_MODEL:-Qwen3-4B-Instruct-2507-1bit}
dmesg_file=''
state='boiler pressure 6.2 bar, feedwater 104 C, drum level low'
while [ $# -gt 0 ]; do
    case $1 in
    --url) url=${2:?}; shift 2 ;;
    --npu) npu=${2:?}; shift 2 ;;
    --big) big=${2:?}; shift 2 ;;
    --dmesg) dmesg_file=${2:?}; shift 2 ;;
    --state) state=${2:?}; shift 2 ;;
    -h | --help) sed -n '28,48p' "$0"; exit 0 ;;
    *[!0-9]*) echo "demo-seq-probe: unknown argument: $1" >&2; exit 2 ;;
    *) rounds=$1; shift ;;
    esac
done
command -v curl > /dev/null 2>&1 || { echo "demo-seq-probe: curl is required" >&2; exit 2; }
command -v python3 > /dev/null 2>&1 || { echo "demo-seq-probe: python3 is required (JSON)" >&2; exit 2; }

# The two prompts are the demo's, byte-for-byte in the parts that matter (they decide prompt length, which
# is one of the variables the c29 failure could have come from): game.py's SYSTEM + its NPU_SUFFIX, and
# img.py's one-shot SYSTEM with its "IDEA: {idea}\nPROMPT:" tail removed, exactly as expand() sends it.
game_system="You narrate a boiler room night shift game. Address the player as 'you', present tense, one short paragraph of at most 45 words. Never write 'STATE', never repeat the action back, never use lists, headings or code. Use only numbers that appear in STATE; never invent or recompute one, and if STATE gives a conversion, apply it exactly. Stay concrete: what the player sees, hears and smells as the action lands."
img_system="You write Stable Diffusion prompts. Rewrite the idea with more visual detail. Answer with the prompt only, one line, comma separated phrases, no preamble, no quotes.
IDEA: a lighthouse at dawn
PROMPT: lone stone lighthouse at dawn, cold sea mist, warm lantern glow, wet granite, cinematic wide shot, soft rim light, muted teal sky"
idea='a power plant boiler room at night, steam and gauges, glowing control panel'

# tdr_count: how many times the kernel says the NPU array stopped dispatching.
# A FILE is treated as the whole window (after-the-fact analysis of a captured log, so its matches are the
# answer); live dmesg is treated as a running log, so the probe reports the delta across the rounds.
tdr_count() {
    if [ -n "$dmesg_file" ] && [ "$dmesg_file" != - ]; then
        grep -c 'aie2_set_cmd_timeout\|power-cycling NPU' "$dmesg_file" 2> /dev/null
    elif [ -n "$dmesg_file" ]; then
        echo 0
    elif dmesg > /dev/null 2>&1; then
        dmesg 2> /dev/null | grep -c 'aie2_set_cmd_timeout\|power-cycling NPU'
    else
        echo -1   # unreadable: reported as n/a, never as 0
    fi
}

# post <path> <json>: the HTTP status and body, one tab apart; status 000 when nothing answered.
post() {
    curl -sS -m 900 -o /tmp/dsq.body -w '%{http_code}' -H 'Content-Type: application/json' \
        -X POST --data-binary "$2" "$url$1" 2> /tmp/dsq.err | tr -d '\n'
    printf '\t'
    tr -d '\n' < /tmp/dsq.body | cut -c1-400
}

# field <json-path-expression> <python expr over d>: pulls one value out of the last body, '' when absent.
field() {
    python3 - "$1" 2> /dev/null <<'PY'
import json, sys
try:
    d = json.load(open("/tmp/dsq.body"))
except Exception:
    print(""); raise SystemExit
print(eval(sys.argv[1], {"d": d, "json": json}))
PY
}

# answer_check: is there text in the reply, or did the budget go somewhere the client cannot see?
# Cycle 30, from pi-release's hardware rehearsal of their own demo-sequence row: a thinking model can spend the WHOLE
# budget in `reasoning_content` and return an empty `content` - a 200 with nothing in it. Their row read that as "no
# answer text" and their fix names the state; this probe had the same hole one level deeper (it judged a turn on the
# HTTP status and the answering model, never on the text), so it would have passed a blank beat. Echoes one token:
#   ok | blank-thinking-only (budget spent, no visible text) | blank (nothing at all)
answer_check() {
    python3 - 2> /dev/null <<'PY2'
import json
try:
    d = json.load(open("/tmp/dsq.body"))
except Exception:
    print("blank"); raise SystemExit
ch = (d.get("choices") or [{}])[0]
msg = ch.get("message") or {}
text = (msg.get("content") or "").strip()
if text:
    print("ok"); raise SystemExit
used = ((d.get("usage") or {}).get("completion_tokens") or 0) + (len(msg.get("reasoning_content") or "") and 1 or 0)
print("blank-thinking-only" if used else "blank")
PY2
}

say() { printf '%s\n' "$*"; }
say "demo-seq-probe: $rounds round(s) of <game NPU turn | big-model turn | picture prompt on the NPU>"
say "  url=$url npu=$npu big=$big"
tdr0=$(tdr_count)
if [ -n "$dmesg_file" ] && [ "$dmesg_file" != - ]; then
    say "  TDRs in $dmesg_file: $tdr0 (a captured log is the whole window)"
else
    say "  TDRs before: $([ "$tdr0" -lt 0 ] && echo 'n/a (dmesg unreadable, pass --dmesg FILE)' || echo "$tdr0")"
fi
pass=0 fails=0 c614=0 fallbacks=0
for i in $(seq 1 "$rounds"); do
    bad=''
    # 1. the game's quick turn on the NPU
    st_body=$(post /chat/completions "$(python3 - "$npu" "$game_system" "$state" <<'PY'
import json, sys
model, system, state = sys.argv[1], sys.argv[2], sys.argv[3]
print(json.dumps({"model": model,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": "STATE: %s\nACTION: open the feedwater valve" % state}],
                  "max_tokens": 80, "temperature": 0.4,
                  "chat_template_kwargs": {"enable_thinking": False}}))
PY
)")
    st=${st_body%%$'\t'*} game_model=$(field 'd.get("model","")')
    case $st_body in *614*) c614=$((c614 + 1)) ;; esac
    [ "$st" = 200 ] && [ -n "$game_model" ] || bad="$bad game-http=$st"
    # the game beat's own text, not just its status: a 200 with an empty content is the take's "blank beat" one layer down.
    # Judged only on a 200 - a non-200 has no content to judge, and naming it both would report one fault twice.
    if [ "$st" = 200 ]; then gv=$(answer_check); [ "$gv" = ok ] || bad="$bad game-$gv"; fi
    # 2. a turn on the big model - the model switch that evicts the NPU (max_models.llm=1)
    bg_body=$(post /chat/completions "$(python3 - "$big" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": "Reply with the word ready."}],
                  "max_tokens": 8, "temperature": 0.0}))
PY
)")
    bg=${bg_body%%$'\t'*}
    case $bg_body in *614*) c614=$((c614 + 1)) ;; esac
    [ "$bg" = 200 ] || bad="$bad big-http=$bg"
    # 3. the picture page's prompt, on the NPU, first dispatch after the switch: the c29 failure
    ic_body=$(post /chat/completions "$(python3 - "$npu" "$img_system" "$idea" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1],
                  "messages": [{"role": "system", "content": sys.argv[2]},
                               {"role": "user", "content":
                                "Rewrite this idea as a detailed image prompt, with lighting, materials "
                                "and mood: %s" % sys.argv[3]}],
                  "max_tokens": 96, "temperature": 0.4,
                  "chat_template_kwargs": {"enable_thinking": False}}))
PY
)")
    ic=${ic_body%%$'\t'*} ic_model=$(field 'd.get("model","")')
    ic_text=$(field '(d.get("choices") or [{}])[0].get("message",{}).get("content","")[:60]')
    case $ic_body in *614*) c614=$((c614 + 1)) ;; esac
    [ "$ic" = 200 ] || bad="$bad picture-http=$ic"
    # the gate's whole point: the NPU answered step 3, not the fallback and not the model it was switched to.
    # Only a 200 is judged this way - a 500 is already counted above, and calling it a fallback too would
    # report one fault twice (measured: a 614-only arm read "2 x 614, 2 fallback answers" before this).
    if [ "$ic" = 200 ] && [ "$ic_model" != "$npu" ]; then
        bad="$bad picture-answered-by=${ic_model:-none}"
        fallbacks=$((fallbacks + 1))
    fi
    # and the prompt's own text: `THINKING-ONLY` is a distinct failure from "no answer at all", because the client sees a
    # clean 200 in both cases and the picture page would then render the fallback's prompt without saying so
    if [ "$ic" = 200 ]; then iv=$(answer_check); [ "$iv" = ok ] || bad="$bad picture-$iv"; fi
    if [ -z "$bad" ]; then
        pass=$((pass + 1))
        say "  round $i/$(printf '%s' "$rounds"): PASS  game=$game_model picture=$ic_model  pic-answer='$ic_text'"
    else
        fails=$((fails + 1))
        say "  round $i/$(printf '%s' "$rounds"): FAIL $bad"
        case $ic_body in *614*) say "    body: ${ic_body#*$'\t'}" ;; *) : ;; esac
    fi
done

tdr1=$(tdr_count)
if [ -n "$dmesg_file" ] && [ "$dmesg_file" != - ]; then tdrs=$tdr1; else tdrs=$([ "$tdr0" -lt 0 ] && echo -1 || echo $((tdr1 - tdr0))); fi
say "demo-seq-probe: $pass/$rounds round(s) passed, $fails failed, $c614 x 614, $fallbacks fallback answer(s), $([ "$tdrs" -lt 0 ] && echo 'TDRs n/a' || echo "$tdrs TDR(s)")"
if [ "$fails" = 0 ] && [ "$c614" = 0 ] && [ "$fallbacks" = 0 ] && [ "$tdrs" = 0 ]; then
    say "RESULT: PASS - the NPU answered the picture prompt $pass/$rounds time(s), 0 x 614, 0 TDR"
    exit 0
fi
say "RESULT: FAIL - see the round lines above (a fallback answer is a failure here: the gate asks who wrote the prompt)"
exit 1
