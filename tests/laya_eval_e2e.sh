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
# laya_eval_e2e.sh: `1bit route --laya-models FILE --eval` and `--dump-encoder` on the real scorer
# (docs/laya-models.md, "Measuring and fine-tuning the router"). Four labelled states, one of them past the
# table's size gate. Checks: the JSON report's counts, the per-case file, the text report; the dump's
# files agree with each other (rows x d x 4 bytes, one line per sequence, the gated case skipped, labels
# moved with --rotate); refusals by name.
set -uo pipefail

bin=${1:?usage: laya_eval_e2e.sh path/to/1bit path/to/laya-model-dir}
laya_model=${2:?usage: laya_eval_e2e.sh path/to/1bit path/to/laya-model-dir}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
fail=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi; }

cat > "$scratch/table.json" <<'JSON'
{"question": {"instructions": "What does the game agent need to answer this step?",
  "classes": {"reflex": "one quick action from what is in view right now",
              "plan": "planning: a crafting tree, a build plan, choosing the next goals"}},
 "default": "plan", "size_gate": {"chars": 300, "class": "plan"},
 "classes": {"reflex": {"model": "small.gguf", "device": "npu"}, "plan": {"model": "big.gguf", "device": "hrx"}}}
JSON
long=$(printf 'Plan the whole base layout. %.0s' $(seq 1 20))
cat > "$scratch/cases.jsonl" <<JSONL
{"state": "A zombie is 2 blocks to the left. Next action?", "label": "reflex", "src": "a"}
{"state": "The log is right in front of you. Next action?", "label": "reflex", "src": "b"}
{"state": "You have 3 iron ore and 6 planks. Which tools should be crafted first, and in what order?", "label": "plan", "src": "c"}
{"state": "$long", "label": "plan", "src": "d"}
{"state": "no outcome yet: left out of an eval"}
JSONL

R=("$bin" route --laya-models "$scratch/table.json" --laya-model "$laya_model")
"${R[@]}" --eval "$scratch/cases.jsonl" --json --per-case "$scratch/per.jsonl" > "$scratch/eval.json" 2> "$scratch/eval.err"
check "eval --json exits 0" "[ $? -eq 0 ]"
check "eval report counts" "python3 -c \"
import json; r = json.load(open('$scratch/eval.json'))
assert r['n'] == 4 and r['gated'] == 1, r
assert sum(map(sum, r['confusion']['matrix'])) == 4
assert 0 <= r['accuracy'] <= 1 and r['correct'] == round(r['accuracy'] * 4)
assert set(r['classes']) == {'reflex', 'plan'} and r['classes']['reflex']['support'] == 2
for k in ('ece', 'log_loss', 'mean_confidence', 'decision_ms_p50', 'decision_ms_p90'): assert k in r
\""
check "per-case file: one line per labelled case, p sums to 1" "python3 -c \"
import json; L = [json.loads(x) for x in open('$scratch/per.jsonl')]
assert [c['src'] for c in L] == ['a', 'b', 'c', 'd']
assert all(abs(sum(c['probabilities'].values()) - 1) < 1e-4 for c in L)
assert L[3]['gated'] and L[3]['chosen'] == 'plan' and not L[0]['gated']
\""
"${R[@]}" --eval "$scratch/cases.jsonl" > "$scratch/eval.txt" 2>&1
check "text report" "grep -q '^accuracy [0-9]/4' '$scratch/eval.txt' && grep -q 'confusion' '$scratch/eval.txt'"

"${R[@]}" --dump-encoder "$scratch/cases.jsonl" --out "$scratch/dump" --rotate > "$scratch/dump.out" 2>&1
check "dump exits 0" "[ $? -eq 0 ]"
check "dump files agree" "python3 -c \"
import json, os; d = '$scratch/dump'; info = json.load(open(d + '/dump.json'))
C = [json.loads(x) for x in open(d + '/cases.jsonl')]
assert info['skipped_size_gated'] == 1 and info['cases'] == len(C) == 8, (info, len(C))
assert os.path.getsize(d + '/h.f32') == info['rows'] * info['d'] * 4
assert sum(c['len'] for c in C) == info['rows'] and C[1]['offset'] == C[0]['len']
assert all(len(c['markers']) == 2 and max(c['markers']) < c['len'] for c in C)
a0, a1 = C[0], C[1]
assert a0['src'] == a1['src'] == 'a' and a0['options'] == ['reflex', 'plan'] and a1['options'] == ['plan', 'reflex']
assert a0['label'] == 0 and a1['label'] == 1
assert [c['label'] for c in C if c['src'] == '$scratch/cases.jsonl:5'] == [-1, -1]
\""

"${R[@]}" --eval "$scratch/missing.jsonl" > /dev/null 2> "$scratch/e1"
check "refuses a missing case file" "grep -q 'cannot read' '$scratch/e1'"
echo '{"state": "x", "label": "chat"}' > "$scratch/bad.jsonl"
"${R[@]}" --eval "$scratch/bad.jsonl" > /dev/null 2> "$scratch/e2"
check "refuses a label that is not a class" "grep -q 'not one of the table' '$scratch/e2'"
"${R[@]}" --eval "$scratch/cases.jsonl" --dump-encoder "$scratch/cases.jsonl" > /dev/null 2> "$scratch/e3"
check "refuses --eval with --dump-encoder" "grep -q 'exactly one of' '$scratch/e3'"
"${R[@]}" --dump-encoder "$scratch/cases.jsonl" > /dev/null 2> "$scratch/e4"
check "refuses a dump without --out" "grep -q -- '--out DIR' '$scratch/e4'"
"$bin" route --eval "$scratch/cases.jsonl" > /dev/null 2> "$scratch/e5"
check "refuses --eval without --laya-models" "grep -q 'need --laya-models' '$scratch/e5'"

[ $fail = 0 ] && echo PASS || echo FAIL
exit $fail
