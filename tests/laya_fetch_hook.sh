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
# tests/laya_fetch_hook.sh <1bit>
#
# engine#3350 part 1: `1bit route --serve` with no Laya checkpoint and ONEBIT_LAYA_FETCH=<fetcher> runs `<fetcher>
# fetch <dir>` before it loads, and the fetcher's stdout does NOT reach route --serve's stdout (the one-JSON-line
# protocol lemond reads). The fake fetcher records its arguments, prints noise on stdout, and puts no real checkpoint
# in place, so the scorer then fails to load: that exit is expected here, the hook is what is checked. Without
# ONEBIT_LAYA_FETCH nothing is run. No checkpoint, no network, no device.
set -u
bin=${1:?usage: laya_fetch_hook.sh <1bit>}
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }

cat > "$t/fetcher" << EOF
#!/bin/sh
echo "\$*" > "$t/args"
echo "NOISE ON STDOUT"
exit 0
EOF
chmod 755 "$t/fetcher"

echo '{"state":"hi"}' | ONEBIT_LAYA_FETCH=$t/fetcher timeout 60 "$bin" route --serve --laya-model "$t/laya" > "$t/out" 2> "$t/err"
[ "$(cat "$t/args" 2> /dev/null)" = "fetch $t/laya" ]; check $? "the fetcher is run as \`<fetcher> fetch <dir>\` when the checkpoint is missing"
! grep -q NOISE "$t/out"; check $? "the fetcher's stdout does not reach route --serve's stdout"
grep -q "NOISE ON STDOUT" "$t/err"; check $? "it goes to stderr (lemond's log) instead"
grep -q "pulling it with $t/fetcher" "$t/err"; check $? "route --serve says it is pulling the checkpoint"

rm -f "$t/args"
echo '{"state":"hi"}' | timeout 60 "$bin" route --serve --laya-model "$t/laya2" > /dev/null 2>&1
[ ! -e "$t/args" ]; check $? "without ONEBIT_LAYA_FETCH no fetcher runs"

mkdir -p "$t/laya3/typed-decisions"; : > "$t/laya3/typed-decisions/model.safetensors"
echo '{"state":"hi"}' | ONEBIT_LAYA_FETCH=$t/fetcher timeout 60 "$bin" route --serve --laya-model "$t/laya3" > /dev/null 2>&1
[ ! -e "$t/args" ]; check $? "a checkpoint that is already there is not fetched again"

echo "laya_fetch_hook: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
