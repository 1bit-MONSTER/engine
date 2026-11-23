#!/bin/sh
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
# engine#3437: 1bit-npu-idle-evict turns Lemonade's idle eviction on for every NPU model (auto_evict true,
# evict_idle_timeout 600) in recipe_options.json, under Lemonade's canonical ids, and leaves GPU models and anything a
# person set alone. Fake catalogue and config in a temporary directory: no Lemonade, no device, no root.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
h=$here/os/v3/puppy/embed/usr/local/bin/1bit-npu-idle-evict
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
q() { python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps($2))" "$1"; }
cat > "$t/cat.json" << 'J'
{"Qwen3-0.6B-NPU-1bit": {"checkpoint": "FastFlowLM/Qwen3-0.6B-NPU2", "recipe": "onebit"},
 "Qwen3-4B-NPU-1bit":   {"checkpoint": "FastFlowLM/Qwen3-4B-NPU2", "recipe": "onebit"},
 "Odd-Name-1bit":       {"checkpoint": "FastFlowLM/Odd", "recipe": "onebit"},
 "Qwen3-4B-Instruct-2507-1bit": {"checkpoint": "unsloth/Qwen3-4B-Instruct-2507-GGUF:x.gguf", "recipe": "onebit"},
 "Llama-3.1-8B-NPU":    {"checkpoint": "amd/Llama-3.1-8B-onnx-ryzenai-npu", "recipe": "ryzenai-llm"}}
J
mkdir -p "$t/cfg"
printf '%s\n' '{"Big-1bit": {"recipe": "onebit", "recipe_options": {"onebit_backend": "npu"}}, "Coder": {"recipe": "llamacpp"}}' > "$t/cfg/user_models.json"
printf '%s\n' '{"builtin.Qwen3-4B-NPU-1bit": {"evict_idle_timeout": 1200}, "user.Coder": {"ctx_size": 0}}' > "$t/cfg/recipe_options.json"

sh "$h" "$t/cat.json" "$t/cfg" > "$t/out"; check $? "exit 0"
r=$t/cfg/recipe_options.json
[ "$(q "$r" 'd["builtin.Qwen3-0.6B-NPU-1bit"]')" = '{"auto_evict": true, "evict_idle_timeout": 600}' ]
check $? "an onebit NPU catalogue model gets auto_evict + a 600 s idle timeout, under builtin.<name>"
[ "$(q "$r" 'd["builtin.Odd-Name-1bit"]["auto_evict"]')" = true ] && [ "$(q "$r" 'd["user.Big-1bit"]["auto_evict"]')" = true ]
check $? "an NPU model is known by its FastFlowLM/ checkpoint, and a user model by onebit_backend npu"
[ "$(q "$r" 'd["builtin.Qwen3-4B-NPU-1bit"]')" = '{"evict_idle_timeout": 1200, "auto_evict": true}' ]
check $? "a timeout a person set (1200 s) is kept; only the missing key is added"
[ "$(q "$r" '[k for k in d if "Instruct" in k or "Llama" in k or k == "user.Coder" and d[k] != {"ctx_size": 0}]')" = '[]' ]
check $? "GPU models, the Windows-only ryzenai-llm NPU entry and another user model's options are left alone"
cp "$r" "$t/once"; sh "$h" "$t/cat.json" "$t/cfg" > "$t/out2"
cmp -s "$r" "$t/once" && grep -q "(already set)" "$t/out2"; check $? "a second run changes nothing"

# an old bare key (Lemonade migrates it to builtin.<name>) is the one extended, not duplicated
rm -rf "${t:?}/c2"; mkdir -p "$t/c2"; printf '%s\n' '{"Qwen3-0.6B-NPU-1bit": {"ctx_size": 4096}}' > "$t/c2/recipe_options.json"
sh "$h" "$t/cat.json" "$t/c2" > /dev/null
[ "$(q "$t/c2/recipe_options.json" '[d["Qwen3-0.6B-NPU-1bit"].get("auto_evict"), "builtin.Qwen3-0.6B-NPU-1bit" in d]')" = '[true, false]' ]
check $? "a bare key not yet migrated is extended in place, not duplicated"

# a recipe_options.json that is not an object is left as it is (exit 1); no catalogue is not an error
rm -rf "${t:?}/c3"; mkdir -p "$t/c3"; echo '[1, 2]' > "$t/c3/recipe_options.json"
sh "$h" "$t/cat.json" "$t/c3" 2> /dev/null; [ $? = 1 ] && [ "$(cat "$t/c3/recipe_options.json")" = '[1, 2]' ]
check $? "a recipe_options.json that is not an object is left untouched (exit 1)"
sh "$h" "$t/none.json" "$t/c4"; check $? "no catalogue (no payload yet) is not an error"

echo "npu_idle_evict_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
