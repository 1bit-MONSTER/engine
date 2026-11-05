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
# `1bit laya-first` (docs/laya-first.md §4): the routing table that ships, and the
# `collection.router` policy it desugars into. The table is the product surface; the policy is the
# mechanism Lemonade already understands, so this test pins the desugaring rather than the wording:
#   - the shipped table validates (--check exit 0) and names four classes,
#   - the emitted policy is byte-identical to tests/laya_first_policy.json (so a change to the
#     table or the desugar is a deliberate diff, not drift),
#   - one rule per class, `route_to` that class's model, every candidate declared, default_model
#     among them,
#   - every class rule carries min_score 0.0 - Laya's confidence for `code` is about 0.44, so the
#     fork's default band of 0.5 would silently drop exactly the decisions this table routes,
#   - the classifier entry declares the four class labels (which is what makes a typo in a rule's
#     label a parse error in the fork),
#   - a table with a device typo, or missing a class, is refused with a message naming the problem,
#   - --laya-first FILE replaces the compiled-in table, and --policy FILE writes what stdout shows,
#   - when the Lemonade fork's route_policy.schema.json is reachable, the emitted policy validates
#     against it (the policy is for Lemonade; its schema is the contract).
#
# usage: tests/laya_first.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: laya_first.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
# For a verdict already computed (a python block's exit status) rather than a shell command.
check_rc() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; fail=1; fi; }

check "the shipped table validates (--check)" '"$bin" laya-first --check >/dev/null 2>&1'
check "--check names all four classes" '"$bin" laya-first --check 2>&1 | grep -q "4 classes"'

"$bin" laya-first >"$scratch/policy.json" 2>"$scratch/err"
check "the emitted policy is byte-identical to tests/laya_first_policy.json" 'diff -q "$scratch/policy.json" "$here/laya_first_policy.json" >/dev/null'

"$bin" laya-first --policy "$scratch/written.json" >/dev/null 2>&1
check "--policy FILE writes what stdout shows" 'diff -q "$scratch/written.json" "$scratch/policy.json" >/dev/null'

python3 - "$scratch/policy.json" <<'PY'
import json, sys
policy = json.load(open(sys.argv[1]))
routing = policy["routing"]
fail = 0


def check(ok, label):
    global fail
    print(("ok   " if ok else "FAIL ") + label)
    if not ok:
        fail = 1


classes = ["short", "prose", "code", "long_doc"]
models = {c["match"]["label"]: c["route_to"] for c in routing["rules"]}
check(policy["recipe"] == "collection.router" and policy["version"] == "1",
      "the policy is a version-1 collection.router")
check(sorted(models) == sorted(classes), "one rule per class (%s)" % ", ".join(sorted(models)))
check(len(routing["rules"]) == 4, "no extra rules: %d" % len(routing["rules"]))
check(all(r["match"]["classifier"] == "laya" for r in routing["rules"]),
      "every rule matches the laya classifier")
check(all(0.0 < r["match"]["min_score"] < 0.05 for r in routing["rules"]),
      "every class rule has a low but NON-ZERO floor (0.0 would make the first rule match every class, "
      "since a label Laya did not choose scores exactly 0.0)")
check(all(r["route_to"] in routing["candidates"] for r in routing["rules"]),
      "every route_to is a candidate")
check(routing["default_model"] in routing["candidates"], "default_model is a candidate")
check(sorted(policy["components"]) == sorted(routing["candidates"]),
      "components and candidates agree")
classifier = routing["classifiers"][0]
check(classifier["type"] == "laya" and sorted(classifier["labels"]) == sorted(classes),
      "the classifier declares the four class labels")
check("model" not in classifier and "prompt" not in classifier,
      "the classifier takes no model or prompt")
sys.exit(fail)
PY
check_rc $? "the desugared policy is shaped as documented"

# A broken table is refused, and the message says what is wrong.
python3 - "$scratch" "$here/../config/laya-first.json" <<'PY'
import json, sys
table = json.load(open(sys.argv[2]))
bad_device = json.loads(json.dumps(table))
bad_device["classes"]["code"]["device"] = "vulkan"
json.dump(bad_device, open(sys.argv[1] + "/bad-device.json", "w"))
missing = json.loads(json.dumps(table))
del missing["classes"]["long_doc"]
json.dump(missing, open(sys.argv[1] + "/missing-class.json", "w"))
PY
msg=$("$bin" laya-first --laya-first "$scratch/bad-device.json" 2>&1 >/dev/null)
check "a device typo is refused, naming it" '[[ "$msg" == *"vulkan"* ]]'
msg=$("$bin" laya-first --laya-first "$scratch/missing-class.json" 2>&1 >/dev/null)
check "a missing class is refused, naming it" '[[ "$msg" == *"long_doc"* ]]'
check "a refused table exits non-zero" '! "$bin" laya-first --laya-first "$scratch/bad-device.json" >/dev/null 2>&1'

# The policy is for Lemonade, so Lemonade's own schema is the contract. Guarded: only runs where a
# fork checkout (or $LEMONADE_ROUTE_POLICY_SCHEMA) is available, never silently skipped in CI logs.
schema=${LEMONADE_ROUTE_POLICY_SCHEMA:-}
[ -z "$schema" ] && schema=$HOME/wt/fork-laya/src/cpp/resources/schemas/route_policy.schema.json
if [ -f "$schema" ] && python3 -c 'import jsonschema' 2>/dev/null; then
    python3 - "$schema" "$scratch/policy.json" <<'PY'
import json, sys
import jsonschema
schema = json.load(open(sys.argv[1]))
policy = json.load(open(sys.argv[2]))
errors = list(jsonschema.Draft202012Validator(schema).iter_errors(policy))
for e in errors[:5]:
    print("   ", list(e.path), e.message[:160])
print(("ok   the policy validates against Lemonade's route_policy.schema.json" if not errors
       else "FAIL the policy does not validate against Lemonade's schema (%d errors)" % len(errors)))
sys.exit(1 if errors else 0)
PY
    check_rc $? "Lemonade's schema accepts the emitted policy"
else
    echo "ok   Lemonade's schema check skipped (set LEMONADE_ROUTE_POLICY_SCHEMA to enable)"
fi

if [ "$fail" = 0 ]; then echo "PASS"; else echo "see the FAIL lines above"; fi
exit "$fail"
