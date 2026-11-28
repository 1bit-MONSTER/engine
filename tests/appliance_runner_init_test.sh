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
# tests/appliance_runner_init_test.sh -- scripts/appliance/50github-runner (the strixhalo-bench runner's boot script),
# with stub sudo/pgrep and a stub run.sh, no runner:
#   1. started from a boot environment with HOME=/, the runner gets HOME=/home/bcloud (2026-11-28: HOME=/ failed every
#      hw-gate job in "Prepare the appliance runner")
#   2. the already-running check is by account: the pr-agent runner's listener (user ghrunner) does not stop it, even when
#      it runs by a relative path
#   3. this runner's own listener running (user bcloud): not started twice, by absolute path or as ./bin/Runner.Listener
#      after a self-update re-exec (review cl-flm, #3609)
set -uo pipefail
S=${1:?usage: appliance_runner_init_test.sh <50github-runner>}
T=$(mktemp -d) || exit 1
trap 'rm -rf "${T:?}"' EXIT
fail=0; ok() { echo "ok   $*"; }; bad() { echo "FAIL $*"; fail=1; }
mkdir -p "$T/bin" "$T/runner/bin"
# sudo: drop "-H -u <user>" and run the rest; pgrep: match the pattern against $T/procs (one command line per line)
printf '#!/bin/sh\nwhile [ $# -gt 0 ]; do case $1 in -H) shift ;; -u) shift 2 ;; *) break ;; esac; done\nexec "$@"\n' > "$T/bin/sudo"
# pgrep: -u <user> -f <ERE> against $T/procs ("<user> <command line>" per line)
printf '#!/bin/sh\nu=""\nwhile [ $# -gt 1 ]; do case $1 in -u) u=$2; shift 2 ;; -f) shift ;; *) break ;; esac; done\nwhile read -r pu pc; do [ -z "$u" ] || [ "$pu" = "$u" ] || continue; printf "%%s\\n" "$pc" | grep -qE -- "$1" && exit 0; done < %q/procs\nexit 1\n' "$T" > "$T/bin/pgrep"
printf '#!/bin/sh\necho "HOME=$HOME" > %q/started\n' "$T" > "$T/runner/run.sh"
chmod +x "$T/bin/sudo" "$T/bin/pgrep" "$T/runner/run.sh"
run() { rm -f "$T/started"; env -i HOME=/ PATH="$T/bin:/usr/bin:/bin" GITHUB_RUNNER_DIR="$T/runner" sh "$S" start; for _i in 1 2 3 4 5 6 7 8 9 10; do [ -e "$T/started" ] && break; sleep 0.1; done; }

: > "$T/procs"; run
[ "$(cat "$T/started" 2> /dev/null)" = "HOME=/home/bcloud" ] && ok "boot environment HOME=/ -> the runner starts with HOME=/home/bcloud" || bad "HOME: $(cat "$T/started" 2> /dev/null || echo 'not started')"

printf 'ghrunner /var/lib/ghrunner/actions-runner-pr-agent/bin/Runner.Listener run\nghrunner ./bin/Runner.Listener run\n' > "$T/procs"; run
[ -e "$T/started" ] && ok "the pr-agent runner's listener (absolute or relative) does not stop this runner from starting" || bad "pr-agent listener blocked the start"

echo "bcloud /home/bcloud/actions-runner/bin/Runner.Listener run" > "$T/procs"; run
[ ! -e "$T/started" ] && ok "this runner's own listener running (absolute path): not started twice" || bad "started twice (absolute)"

echo "bcloud ./bin/Runner.Listener run" > "$T/procs"; run
[ ! -e "$T/started" ] && ok "this runner's own listener after a self-update re-exec (./bin/Runner.Listener): not started twice" || bad "started twice (relative re-exec)"

[ "$fail" = 0 ] && echo "appliance_runner_init: PASS" || echo "appliance_runner_init: FAIL"
exit "$fail"
