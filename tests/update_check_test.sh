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
# OS updater U1: `1bit-update --check [--channel stable|test]` reads GitHub's release list (here canned JSON through
# ONEBIT_UPDATE_API_JSON, so no network) and says what is running, what the channel offers and whether it is newer.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
u=$here/os/v3/puppy/embed/usr/local/bin/1bit-update
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
esc=$(printf '\033')
cat > "$t/rel.json" << J
[{"tag_name":"v2026.49-rc2","draft":true,"prerelease":true,"published_at":"2026-11-25T00:00:00Z","assets":[]},
 {"tag_name":"v2026.49-rc1","draft":false,"prerelease":true,"published_at":"2026-11-24T00:00:00Z","body":"Inkling test build \\u001b[31mRED",
  "assets":[{"name":"1bit-os-v2026.49-rc1.img.zst","size":1468006400,"browser_download_url":"https://x/1bit-os-v2026.49-rc1.img.zst"},
            {"name":"SHA256SUMS","browser_download_url":"https://x/SHA256SUMS"},{"name":"SHA256SUMS.sig","browser_download_url":"https://x/SHA256SUMS.sig"}]},
 {"tag_name":"v2026.48","draft":false,"prerelease":false,"published_at":"2026-11-23T00:00:00Z","body":"| a | b |\n1bit OS v2026.48",
  "assets":[{"name":"1bit-os-v2026.48.img.zst","size":1358100365,"browser_download_url":"https://x/1bit-os-v2026.48.img.zst"},{"name":"SHA256SUMS","browser_download_url":"https://x/SHA256SUMS"}]},
 {"tag_name":"v2026.47","draft":false,"prerelease":false,"published_at":"2026-11-22T00:00:00Z","assets":[]}]
J
run() {  # run <running tag> [args...]
    printf 'payload=lemonade-onebit-%s\n' "$1" > "$t/id"; shift
    ONEBIT_UPDATE_IDENTITY=$t/id ONEBIT_UPDATE_CONF=$t/update.conf ONEBIT_UPDATE_API_JSON=$t/rel.json sh "$u" --check "$@" 2>&1
}
o=$(run v2026.47); echo "$o" | grep -q '^stable: v2026.48 (2026-11-23)$' && echo "$o" | grep -q '^an update is available'
check $? "stable: the newest full release (v2026.48), newer than v2026.47"
echo "$o" | grep -q 'carries no SHA256SUMS.sig, so --latest will refuse it'
check $? "a release without SHA256SUMS.sig is said to be refused by --latest"
o=$(run v2026.48 --channel test); echo "$o" | grep -q '^test: v2026.49-rc1 (2026-11-24, pre-release)$' && echo "$o" | grep -q '1400 MiB'
check $? "test: the newest non-draft (v2026.49-rc1; the draft rc2 is skipped), with its image size"
! echo "$o" | grep -q "$esc" && echo "$o" | grep -q 'Inkling test build \[31mRED'
check $? "a terminal escape in the remote notes is stripped (printable ASCII only)"
o=$(run v2026.48); echo "$o" | grep -q '^up to date$' && ! echo "$o" | grep -q '| a | b |'
check $? "running the newest stable: up to date; a markdown table row is not a note"
echo 'channel=test' > "$t/update.conf"; o=$(run v2026.49-rc1); echo "$o" | grep -q '^test: v2026.49-rc1' && echo "$o" | grep -q '^up to date$'
check $? "the channel comes from update.conf; the same rc is up to date"
o=$(run v2026.49 --channel test); echo "$o" | grep -q '^up to date$'
check $? "a final release sorts after its own release candidates (v2026.49 > v2026.49-rc1)"
rm -f "$t/update.conf"
printf '%s\n' '{"message":"API rate limit exceeded for 1.2.3.4.","documentation_url":"https://docs.github.com"}' > "$t/rl.json"
o=$(printf 'payload=lemonade-onebit-v2026.48\n' > "$t/id"; ONEBIT_UPDATE_IDENTITY=$t/id ONEBIT_UPDATE_API_JSON=$t/rl.json sh "$u" --check 2>&1); rc=$?
[ $rc = 0 ] && echo "$o" | grep -q 'no answer from the stable channel: API rate limit exceeded'
check $? "a rate-limit answer is said plainly, exit 0"
o=$(printf 'payload=lemonade-onebit-v2026.48\n' > "$t/id"; ONEBIT_UPDATE_IDENTITY=$t/id ONEBIT_UPDATE_API=http://127.0.0.1:9/x sh "$u" --check 2>&1); rc=$?
[ $rc = 0 ] && echo "$o" | grep -q 'could not reach GitHub'
check $? "a non-HTTPS release URL is refused (curl --proto =https), said, exit 0"
sh "$u" --check --channel nightly > /dev/null 2>&1; [ $? = 1 ]; check $? "an unknown --channel is a usage error (exit 1)"

echo "update_check_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
