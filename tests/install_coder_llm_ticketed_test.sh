#!/bin/bash
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
# tests/install_coder_llm_ticketed_test.sh — scripts/install-coder-llm-ticketed.sh on a fake unit directory and a stub
# systemctl (no real unit is touched):
#   1 --diff prints the change and modifies nothing;
#   2 --status says NOT ticketed (exit 1) before the install;
#   3 --install backs up the drop-in, keeps the server command byte for byte behind the wrapper, installs the wrapper,
#     reloads and restarts (the stub records both), and --status then says ticketed (exit 0);
#   4 a second --install changes nothing (idempotent);
#   5 CODER_LLM_UNIT=1bit-canonical (engine#2579): the drop-in gets that unit's own label, demand file and URL (from the
#     --port of its serve line), the pr-agent poll off, the command byte for byte; the restart is of that unit; a serve
#     line with no --port is refused rather than guessed.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
inst=${1:-$root/scripts/install-coder-llm-ticketed.sh}
[ -x "$inst" ] || { echo "not executable: $inst"; exit 2; }
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
fail=0
mkdir -p "$scratch/unit" "$scratch/bin"
serve='/opt/x/1bit serve -m /models/m.gguf --device hrx --host 127.0.0.1 --port 8090 --ctx-size 131072 --alias qwen3-coder --parallel 4'
cat > "$scratch/unit/zz.conf" <<EOF
[Service]
Environment=HSA_USE_SVM=0
ExecStart=
ExecStart=$serve
EOF
cp "$scratch/unit/zz.conf" "$scratch/orig.conf"
# the stub: "show -p ExecStart" answers from the drop-in on disk (what a daemon-reload would load); everything is logged
cat > "$scratch/systemctl" <<EOF
#!/bin/bash
echo "\$*" >> "$scratch/calls"
if [ "\$1" = show ]; then grep '^ExecStart=.' "$scratch/unit/zz.conf" | tail -1 | sed 's/^ExecStart=//'; fi
EOF
chmod +x "$scratch/systemctl"
envs=(CODER_LLM_UNIT_DIR="$scratch/unit" CODER_LLM_DROPIN=zz.conf CODER_LLM_BIN_DIR="$scratch/bin"
      CODER_LLM_SYSTEMCTL="$scratch/systemctl" GPU_TOKEN_FILE="$scratch/none")

out=$(env "${envs[@]}" "$inst" --diff)
cmp -s "$scratch/unit/zz.conf" "$scratch/orig.conf" || { echo "FAIL 1: --diff modified the drop-in"; fail=1; }
grep -q "^+ExecStart=$scratch/bin/coder-llm-ticketed -- $serve\$" <<< "$out" || { echo "FAIL 1: --diff does not show the wrapped command"; fail=1; }
[ -e "$scratch/bin/coder-llm-ticketed" ] && { echo "FAIL 1: --diff installed the wrapper"; fail=1; }

env "${envs[@]}" "$inst" --status > "$scratch/s1" 2>&1; rc=$?
[ "$rc" = 1 ] && grep -q "STATUS: NOT ticketed" "$scratch/s1" || { echo "FAIL 2: --status before install: rc $rc, $(tail -1 "$scratch/s1")"; fail=1; }

env "${envs[@]}" "$inst" --install > "$scratch/i1" 2>&1; rc=$?
[ "$rc" = 0 ] || { echo "FAIL 3: --install rc $rc: $(tail -3 "$scratch/i1")"; fail=1; }
bak=$(ls "$scratch/unit"/zz.conf.bak-* 2> /dev/null | head -1)
[ -n "$bak" ] && cmp -s "$bak" "$scratch/orig.conf" || { echo "FAIL 3: no faithful backup ($bak)"; fail=1; }
[ "$(grep -c '^ExecStart=.' "$scratch/unit/zz.conf")" = 1 ] || { echo "FAIL 3: not exactly one command line"; fail=1; }
grep -qxF "ExecStart=$scratch/bin/coder-llm-ticketed -- $serve" "$scratch/unit/zz.conf" || { echo "FAIL 3: the server command did not survive byte for byte"; fail=1; }
grep -qx 'ExecStart=' "$scratch/unit/zz.conf" || { echo "FAIL 3: the ExecStart= reset line is gone"; fail=1; }
grep -qx 'Environment=HSA_USE_SVM=0' "$scratch/unit/zz.conf" || { echo "FAIL 3: an existing line was lost"; fail=1; }
cmp -s "$scratch/bin/coder-llm-ticketed" "$root/scripts/coder-llm-ticketed" || { echo "FAIL 3: the wrapper was not installed"; fail=1; }
grep -qx 'daemon-reload' "$scratch/calls" && grep -qx 'restart coder-llm' "$scratch/calls" || { echo "FAIL 3: no daemon-reload + restart ($(tr '\n' ';' < "$scratch/calls"))"; fail=1; }
grep -q "STATUS: ticketed" "$scratch/i1" || { echo "FAIL 3: --status after install: $(tail -1 "$scratch/i1")"; fail=1; }

cp "$scratch/unit/zz.conf" "$scratch/after.conf"
env "${envs[@]}" "$inst" --install > "$scratch/i2" 2>&1; rc=$?
[ "$rc" = 0 ] && cmp -s "$scratch/unit/zz.conf" "$scratch/after.conf" && grep -q "already ticketed" "$scratch/i2" \
    || { echo "FAIL 4: a second --install was not a no-op (rc $rc)"; fail=1; }
[ "$(ls "$scratch/unit"/zz.conf.bak-* | wc -l)" = 1 ] || { echo "FAIL 4: a second backup was written"; fail=1; }

# ---- 5: another resident model, 1bit-canonical -------------------------------------------------------------------
mkdir -p "$scratch/cunit"; : > "$scratch/calls"
cserve='/opt/x/1bit serve -m /models/q.gguf --device hrx --host 127.0.0.1 --port 8095 --ctx-size 8192 --alias qwen3-4b'
printf '[Service]\nEnvironment=HSA_USERPTR_FOR_PAGED_MEM=0\nExecStart=\nExecStart=%s\n' "$cserve" > "$scratch/cunit/zz.conf"
sed "s|$scratch/unit/zz.conf|$scratch/cunit/zz.conf|" "$scratch/systemctl" > "$scratch/csystemctl"; chmod +x "$scratch/csystemctl"
cenvs=(CODER_LLM_UNIT=1bit-canonical CODER_LLM_UNIT_DIR="$scratch/cunit" CODER_LLM_DROPIN=zz.conf CODER_LLM_BIN_DIR="$scratch/bin"
       CODER_LLM_SYSTEMCTL="$scratch/csystemctl" GPU_TOKEN_FILE="$scratch/none")
env "${cenvs[@]}" "$inst" --install > "$scratch/c1" 2>&1; rc=$?
[ "$rc" = 0 ] || { echo "FAIL 5: --install of 1bit-canonical rc $rc: $(tail -2 "$scratch/c1")"; fail=1; }
for want in "Environment=CODER_LLM_LABEL=1bit-canonical" "Environment=CODER_LLM_DEMAND=$HOME/.cache/1bit-canonical/demand" \
            "Environment=CODER_LLM_URL=http://127.0.0.1:8095" "Environment=CODER_LLM_GH_POLL=0" "Environment=CODER_LLM_DEMAND_TTL_MIN=5" \
            "ExecStart=$scratch/bin/coder-llm-ticketed -- $cserve"; do
    grep -qxF "$want" "$scratch/cunit/zz.conf" || { echo "FAIL 5: missing '$want'"; fail=1; }
done
grep -q 'CODER_LLM_GH_POLL=1' "$scratch/cunit/zz.conf" && { echo "FAIL 5: the pr-agent poll is on for 1bit-canonical"; fail=1; }
grep -qx 'restart 1bit-canonical' "$scratch/calls" || { echo "FAIL 5: 1bit-canonical was not restarted ($(tr '\n' ';' < "$scratch/calls"))"; fail=1; }
grep -qx 'restart coder-llm' "$scratch/calls" && { echo "FAIL 5: coder-llm was restarted for a 1bit-canonical install"; fail=1; }
printf '[Service]\nExecStart=\nExecStart=/opt/x/1bit serve -m /models/q.gguf\n' > "$scratch/cunit/zz.conf"
env "${cenvs[@]}" "$inst" --diff > "$scratch/c2" 2>&1; rc=$?
[ "$rc" = 2 ] && grep -q 'no --port' "$scratch/c2" || { echo "FAIL 5: a serve line with no --port was not refused (rc $rc)"; fail=1; }

[ "$fail" = 0 ] && echo "install_coder_llm_ticketed: ok (--diff changes nothing; status before = NOT; install keeps the command byte for byte, backs up, reloads+restarts; idempotent; another unit gets its own label, demand and URL, and no --port is refused)"
exit "$fail"
