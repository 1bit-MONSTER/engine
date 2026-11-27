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
# tests/fetch_private_ssh_test.sh -- scripts/fetch-private.sh fetches over a read-only deploy-key ssh alias when the box
# has one (orchestrator 2026-11-27), and over https with gh credentials otherwise. Hermetic: the "remote" is a local bare
# repository; a stub ssh answers `ssh -G` like a config with (or without) `Host github-<repo>`, and serves git-upload-pack
# from the bare repo; https is redirected to the same repo with url.insteadOf. Real fetches, no network, no keys.
#   1. alias mapped -> fetched over ssh, checkout at the pin
#   2. no alias -> https, ssh never carries a fetch
#   3. alias mapped but the ssh fetch fails (key not accepted) -> falls back to https and still succeeds
#   4. FETCH_PRIVATE_SSH_HOST=none -> ssh is not even asked
#   5. both fail -> exit 1, and the message names the deploy key
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
fp=$root/scripts/fetch-private.sh
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
fails=0; ok() { echo "ok   $*"; }; bad() { echo "FAIL $*"; fails=$((fails + 1)); }
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t GIT_CONFIG_NOSYSTEM=1 HOME=$tmp/home
mkdir -p "$HOME" "$tmp/bin"
git init -q "$tmp/src" && mkdir -p "$tmp/src/addons/hip" && echo k > "$tmp/src/addons/hip/kernel.hip" && echo x > "$tmp/src/other" \
  && git -C "$tmp/src" add -A && git -C "$tmp/src" commit -q -m c && sha=$(git -C "$tmp/src" rev-parse HEAD) \
  && git clone -q --bare "$tmp/src" "$tmp/bare.git" \
  && git -C "$tmp/bare.git" config uploadpack.allowAnySHA1InWant true && git -C "$tmp/bare.git" config uploadpack.allowFilter true \
  || { echo "FAIL: cannot build the fixture repo"; exit 1; }
printf 'repo=1bit-MONSTER/gpu-kernels\nsha=%s\npath=addons/hip\nrequire=kernel.hip\n' "$sha" > "$tmp/pin.config"
cat > "$tmp/bin/ssh" <<'STUB'
#!/bin/bash
if [ "$1" = -G ]; then echo "ssh -G $2" >> "$STUBLOG"
  if [ -n "${STUB_ALIAS:-}" ] && [ "$2" = "$STUB_ALIAS" ]; then echo "hostname github.com"; else echo "hostname $2"; fi; exit 0; fi
echo "ssh fetch ${*: -1}" >> "$STUBLOG"
[ -n "${STUB_SSH_FAIL:-}" ] && { echo "Permission denied (publickey)." >&2; exit 255; }
eval "set -- ${*: -1}"; exec "$1" "$STUBBARE"
STUB
chmod +x "$tmp/bin/ssh"
export STUBBARE=$tmp/bare.git PATH="$tmp/bin:$PATH"
https_ok=(env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0="url.file://$tmp/bare.git.insteadOf" GIT_CONFIG_VALUE_0="https://github.com/1bit-MONSTER/gpu-kernels.git")
https_bad=(env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0="url.file://$tmp/nowhere.git.insteadOf" GIT_CONFIG_VALUE_0="https://github.com/1bit-MONSTER/gpu-kernels.git")
run() { local n=$1; shift; : > "$tmp/log$n"; STUBLOG=$tmp/log$n "$@" bash "$fp" "$tmp/pin.config" "$tmp/dest$n" > "$tmp/out$n" 2> "$tmp/err$n"; }

run 1 "${https_bad[@]}" STUB_ALIAS=github-gpu-kernels; rc=$?
[ "$rc" = 0 ] && [ "$(cat "$tmp/out1")" = "$tmp/dest1/addons/hip" ] && [ "$(git -C "$tmp/dest1" rev-parse HEAD)" = "$sha" ] \
  && grep -q "ssh fetch git-upload-pack '1bit-MONSTER/gpu-kernels.git'" "$tmp/log1" && grep -q "over ssh alias github-gpu-kernels" "$tmp/err1" \
  && ok "alias mapped: fetched over ssh (https unreachable), at the pin" || bad "ssh path: rc $rc $(cat "$tmp/err1") $(cat "$tmp/log1")"

run 2 "${https_ok[@]}"; rc=$?
[ "$rc" = 0 ] && [ "$(git -C "$tmp/dest2" rev-parse HEAD)" = "$sha" ] && ! grep -q "ssh fetch" "$tmp/log2" \
  && grep -q "ssh -G github-gpu-kernels" "$tmp/log2" && ok "no alias: https, ssh carried no fetch" || bad "https path: rc $rc $(cat "$tmp/err2")"

run 3 "${https_ok[@]}" STUB_ALIAS=github-gpu-kernels STUB_SSH_FAIL=1; rc=$?
[ "$rc" = 0 ] && [ "$(git -C "$tmp/dest3" rev-parse HEAD)" = "$sha" ] && grep -q "ssh fetch" "$tmp/log3" && grep -q "trying https" "$tmp/err3" \
  && ok "ssh fetch refused: fell back to https and succeeded" || bad "fallback: rc $rc $(cat "$tmp/err3")"

run 4 "${https_ok[@]}" STUB_ALIAS=github-gpu-kernels FETCH_PRIVATE_SSH_HOST=none; rc=$?
[ "$rc" = 0 ] && [ ! -s "$tmp/log4" ] && ok "FETCH_PRIVATE_SSH_HOST=none: ssh not asked" || bad "none: rc $rc log $(cat "$tmp/log4")"

run 5 "${https_bad[@]}" STUB_ALIAS=github-gpu-kernels STUB_SSH_FAIL=1; rc=$?
[ "$rc" = 1 ] && grep -q "deploy key on ssh alias github-gpu-kernels" "$tmp/err5" && ok "both fail: exit 1, the message names the deploy key" || bad "both fail: rc $rc $(cat "$tmp/err5")"

[ "$fails" = 0 ] && { echo "fetch_private_ssh_test: pass"; exit 0; }
echo "fetch_private_ssh_test: $fails failure(s)"; exit 1
