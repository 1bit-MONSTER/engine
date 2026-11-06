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
# The chat-template line and pin (app/template_pin.h, docs/serve.md "Chat template"). A GGUF's
# tokenizer.chat_template is a Jinja2 program that runs on every request, so:
#   - a load names the template's sha256 in one line, whether or not it is pinned,
#   - a model with no template and no pin says nothing (the route tests count lines),
#   - a pin (--template-pin FILE, default config/template-pins.json built in) that does not match
#     warns and still serves; --template-pin-strict (or ONEBIT_TEMPLATE_PIN_STRICT=1) refuses the load,
#   - a Q4NX / NPU model directory is covered the same way, from its tokenizer_config.json (or
#     chat_template.jinja): the same line, the same pin, keyed by the directory's name,
#   - a pin matches the model id (--alias, Lemonade's catalog entry) or the file's own name,
#   - "none" pins a model that must carry no template,
#   - a pin file that is not a pin file, or a value that is not a sha256, is refused by name,
#   - --no-template-pin logs the sha and compares nothing.
#
# usage: tests/template_pin.sh path/to/1bit
set -uo pipefail

bin=${1:?usage: template_pin.sh path/to/1bit}
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

tpl='{% for m in messages %}{{ "<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n" }}{% endfor %}'
sha=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest())' "$tpl")
other='{% for m in messages %}{{ m["role"] }}: {{ m["content"] }}\n{% endfor %}'
other_sha=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest())' "$other")

# Fixtures: header-only GGUFs. "tpl" carries the template above, "bare" carries none (the header-only
# fixtures the other route tests use), "zaya1" is named after a model the built-in pins know.
python3 - "$scratch" "$tpl" <<'PY'
import struct, sys
def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
def u32(k, v): return s(k) + struct.pack("<I", 4) + struct.pack("<I", v)
def st(k, v): return s(k) + struct.pack("<I", 8) + s(v)
base = [st("general.architecture", "qwen3"), u32("qwen3.block_count", 28), u32("qwen3.context_length", 40960),
        u32("qwen3.embedding_length", 1024), u32("qwen3.attention.head_count", 16),
        u32("qwen3.attention.head_count_kv", 8)]
for name, kv in (("tpl", [st("tokenizer.chat_template", sys.argv[2])]), ("bare", []),
                 ("zaya1-8b-q4_k_m", [st("tokenizer.chat_template", "{% for m in messages %}{{ m }}{% endfor %}")]),
                 ("qwen3-4b-instruct-2507-q4_k_m",
                  [st("tokenizer.chat_template", "{% for m in messages %}{{ m }}{% endfor %}")])):
    all_kv = base + kv
    with open(f"{sys.argv[1]}/{name}.gguf", "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(all_kv)) + b"".join(all_kv))
PY

pins() { printf '{"pins": {%s}}\n' "$1" >"$scratch/pins.json"; }

# every run: the plan, never a backend, stderr in $scratch/err
run() { local model=$1; shift; "$bin" serve -m "$scratch/$model.gguf" --device cpu \
    --llama-server "$here/fake_backend.py" --no-mem-check --port 43190 --print-backend "$@" \
    >"$scratch/out" 2>"$scratch/err"; }

pins ""
run tpl --template-pin "$scratch/pins.json"; rc=$?
check "an unpinned model still names its template (exit $rc)" \
    '[ "$rc" = 0 ] && grep -qx "1bit serve: template sha256='"$sha"' (tpl)" "$scratch/err"'

run bare --template-pin "$scratch/pins.json"; rc=$?
check "a model with no template and no pin says nothing (exit $rc)" '[ "$rc" = 0 ] && [ ! -s "$scratch/err" ]'

pins "\"tpl\": \"$sha\""
run tpl --template-pin "$scratch/pins.json"; rc=$?
check "a matching pin is silent about the match (exit $rc)" \
    '[ "$rc" = 0 ] && grep -q "template sha256=$sha" "$scratch/err" && ! grep -q warning "$scratch/err"'

pins "\"TPL\": {\"sha256\": \"$other_sha\"}"
run tpl --template-pin "$scratch/pins.json"; rc=$?
check "a pin that does not match warns and serves (exit $rc)" \
    '[ "$rc" = 0 ] && grep -q "warning: template pin: tpl (tpl) expects $other_sha, the model carries $sha" "$scratch/err"'

run tpl --template-pin "$scratch/pins.json" --template-pin-strict; rc=$?
check "--template-pin-strict refuses it (exit $rc)" \
    '[ "$rc" = 1 ] && grep -q "expects $other_sha, the model carries $sha in tokenizer.chat_template" "$scratch/err" &&
     grep -q "refused, --template-pin-strict" "$scratch/err" && ! grep -q "^{" "$scratch/out"'

export ONEBIT_TEMPLATE_PIN_STRICT=1
run tpl --template-pin "$scratch/pins.json"; rc=$?
unset ONEBIT_TEMPLATE_PIN_STRICT
check "ONEBIT_TEMPLATE_PIN_STRICT=1 is the same (exit $rc)" \
    '[ "$rc" = 1 ] && grep -q "refused, --template-pin-strict" "$scratch/err"'

pins "\"Qwen3-4B-1bit\": \"$sha\""
run tpl --alias Qwen3-4B-1bit --template-pin "$scratch/pins.json"; rc=$?
check "a pin matches the catalog id from --alias (exit $rc)" \
    '[ "$rc" = 0 ] && grep -q "template sha256=$sha (Qwen3-4B-1bit)" "$scratch/err" && ! grep -q warning "$scratch/err"'

pins "\"bare\": \"none\""
run bare --template-pin "$scratch/pins.json"; rc=$?
check "a \"none\" pin: no template, no warning (exit $rc)" \
    '[ "$rc" = 0 ] && grep -qx "1bit serve: template sha256=none (bare)" "$scratch/err" && ! grep -q warning "$scratch/err"'

pins "\"bare\": \"$sha\""
run bare --template-pin "$scratch/pins.json"; rc=$?
check "a template the pin expects but the file lacks warns (exit $rc)" \
    '[ "$rc" = 0 ] && grep -q "warning: template pin: bare (bare) expects $sha, the model carries none" "$scratch/err"'

# The built-in pins (config/template-pins.json): a file named after a pinned model with a template that
# is not the pinned one. This is the check that the built-in table is compiled in and used by default.
run zaya1-8b-q4_k_m; rc=$?
check "the built-in pins compare without --template-pin (exit $rc)" \
    '[ "$rc" = 0 ] && grep -q "template sha256=" "$scratch/err" &&
     grep -q "warning: template pin: zaya1-8b-q4_k_m (zaya1-8b-q4_k_m.gguf) expects b55ecfbbbcbb238a011465bfa139366db18790103f9205060cf2855a3f51b30f" "$scratch/err"'

# A row the appliance ships, so the stick's own chat model is covered by the built-in table rather than
# only by the demo/scoreboard names (config/template-pins.json records unsloth/Qwen3-4B-Instruct-2507).
run qwen3-4b-instruct-2507-q4_k_m; rc=$?
check "the appliance's staged chat model is a built-in pin (exit $rc)" \
    '[ "$rc" = 0 ] && grep -q "warning: template pin: qwen3-4b-instruct-2507-q4_k_m (qwen3-4b-instruct-2507-q4_k_m.gguf) expects c979e0e71a3e21b8f208e6ab120d5cb29327885f29d2a8b18fda67a723798e18" "$scratch/err"'

# A model directory (a Q4NX / NPU container): the template is tokenizer_config.json's "chat_template",
# or chat_template.jinja beside it. --print-backend on a directory exits 5 (the NPU runs in serve's own
# process), which is after the line, so the line is what these cases check.
mkdir -p "$scratch/contj" "$scratch/cont" "$scratch/contempty"
python3 - "$scratch" "$tpl" <<'PY2'
import json, sys
sc, tpl = sys.argv[1], sys.argv[2]
json.dump({"chat_template": tpl, "tokenizer_class": "Qwen2Tokenizer"},
          open(f"{sc}/cont/tokenizer_config.json", "w"))
open(f"{sc}/contj/chat_template.jinja", "w").write(tpl)
json.dump({"tokenizer_class": "Qwen2Tokenizer"}, open(f"{sc}/contempty/tokenizer_config.json", "w"))
PY2
run_dir() { local d=$1; shift; "$bin" serve -m "$scratch/$d" --device cpu --llama-server "$here/fake_backend.py" \
    --no-mem-check --port 43191 --print-backend "$@" >"$scratch/out" 2>"$scratch/err"; }
run_dir cont; rc=$?
check "a model directory names its tokenizer_config.json template (exit $rc)" \
    '[ "$rc" = 5 ] && grep -qx "1bit serve: template sha256='"$sha"' (cont)" "$scratch/err"'
run_dir contj; rc=$?
check "a chat_template.jinja directory names its template (exit $rc)" \
    '[ "$rc" = 5 ] && grep -qx "1bit serve: template sha256='"$sha"' (contj)" "$scratch/err"'
run_dir contempty; rc=$?
check "a directory with no template and no pin says nothing about a template (exit $rc)" \
    '[ "$rc" = 5 ] && ! grep -q "template sha256=" "$scratch/err" && grep -q "runs in serve.s own process" "$scratch/err"'
pins "\"cont\": \"$other_sha\""
run_dir cont --template-pin "$scratch/pins.json"; rc=$?
check "a directory's pin mismatch warns (exit $rc)" \
    '[ "$rc" = 5 ] && grep -q "warning: template pin: cont (cont) expects $other_sha, the model carries $sha" "$scratch/err"'
run_dir cont --template-pin "$scratch/pins.json" --template-pin-strict; rc=$?
check "a directory's pin refuses the load when strict (exit $rc)" \
    '[ "$rc" = 1 ] && grep -q "refused, --template-pin-strict" "$scratch/err"'
pins "\"contj\": \"none\""
run_dir contj --template-pin "$scratch/pins.json"; rc=$?
check "a directory pinned \"none\" with a template warns (exit $rc)" \
    '[ "$rc" = 5 ] && grep -q "warning: template pin: contj (contj) expects none, the model carries $sha" "$scratch/err"'

run zaya1-8b-q4_k_m --no-template-pin; rc=$?
check "--no-template-pin names the sha and compares nothing (exit $rc)" \
    '[ "$rc" = 0 ] && grep -q "template sha256=" "$scratch/err" && ! grep -q warning "$scratch/err"'

pins "\"tpl\": \"$sha\""
sed 's/'$sha'/zzzz/' "$scratch/pins.json" >"$scratch/bad.json"
run tpl --template-pin "$scratch/bad.json"; rc=$?
check "a value that is not a sha256 is refused by name (exit $rc)" \
    '[ "$rc" = 1 ] && grep -q "pins.tpl: \"zzzz\" is not a sha256" "$scratch/err"'

printf '{"rows": []}\n' >"$scratch/nopins.json"
run tpl --template-pin "$scratch/nopins.json"; rc=$?
check "a file that is not a pin file is refused (exit $rc)" \
    '[ "$rc" = 1 ] && grep -q "expected an object with a \"pins\" object" "$scratch/err"'

run tpl --template-pin "$scratch/missing.json"; rc=$?
check "an unreadable --template-pin is refused (exit $rc)" \
    '[ "$rc" = 1 ] && grep -q "cannot read $scratch/missing.json" "$scratch/err"'

exit $fail
