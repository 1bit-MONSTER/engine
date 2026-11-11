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
# ci-apt-install.sh - install the CI build's apt dependencies, retrying the mirror on a time budget.
#
# engine#2210. ci.yml already retried this for engine#1814, and the shape was right: each attempt re-reads the
# index, so a later attempt can reach a different mirror. What it did not have was enough TIME. On 2026-11-11
# 14:53Z four consecutive attempts failed and took `main` plus eleven open PRs red at once (#2199-#2205, #2183,
# #2187, #2189, #2207) -- four attempts spanned ~90 s of back-off, and a mirror pool returning 404s on debs its
# own fresh index lists does not repair itself inside 90 s.
#
# So: the back-off doubles (15, 30, 60, 120, 240, then 300 s steps to a ~23 min deadline), each apt call carries
# Acquire::Retries so a per-connection 404 or timeout is retried inside the attempt instead of ending it, and the
# deadline keeps the step bounded so it fails rather than sleeping into the job timeout.
#
# usage: scripts/ci-apt-install.sh [--deadline-seconds N]
# exit: 0 installed, 1 the mirror never served the packages
set -uo pipefail

deadline_seconds=1500
while [ $# -gt 0 ]; do
    case "$1" in
        --deadline-seconds) deadline_seconds=${2:-1500}; shift 2 ;;
        -h|--help) sed -n '17,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "ci-apt-install: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

apt_opts="-o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30"
packages="ca-certificates git python3 python3-venv curl jq file procps make gcc-16 g++-16 cmake ninja-build
          libcurl4-openssl-dev libzstd-dev libssl-dev libdrm-dev libpcre2-dev"

attempt=0; delay=15; started=$(date +%s); deadline=$((started + deadline_seconds))
while :; do
    attempt=$((attempt + 1))
    # shellcheck disable=SC2086  # both option and package lists are intentional word splits
    if apt-get $apt_opts update &&
       DEBIAN_FRONTEND=noninteractive apt-get $apt_opts install -y --no-install-recommends $packages; then
        printf 'ci-apt-install: installed on attempt %d after %ss\n' "$attempt" "$(( $(date +%s) - started ))"
        exit 0
    fi
    now=$(date +%s)
    if [ $((now + delay)) -gt "$deadline" ]; then break; fi
    printf 'ci-apt-install: attempt %d failed after %ss (mirror 404 is a known transient, engine#1814/engine#2210); retrying in %ss\n' \
        "$attempt" "$((now - started))" "$delay"
    sleep "$delay"
    delay=$(( delay * 2 )); if [ "$delay" -gt 300 ]; then delay=300; fi
done
printf '::error::apt-get update+install failed after %d attempt(s) and %ss -- if the 404s name debs the index lists, it is the mirror (engine#1814), not this tree; re-run the job\n' \
    "$attempt" "$(( $(date +%s) - started ))"
exit 1
