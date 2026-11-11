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
# ci-apt-install.sh - make sure the CI build's apt dependencies are present, without depending on the network.
#
# engine#2210. On 2026-11-11 the build step failed four times in a row and took `main` plus eleven open PRs red at
# once (#2199-#2205, #2183, #2187, #2189, #2207). The step reported it as "404s on debs the index lists" and
# engine#1814's retry was written for that reading, quite reasonably -- but the cause turned out to be the
# **build container's DNS** (engine#2211, fixed by #2224 giving the container its own resolvers). An apt that
# cannot resolve the archive fails the same way a mirror 404 does, and the error text names the mirror.
#
# That is the lesson this script is built around, and it is not "retry harder". The runners are self-hosted and
# the whole toolchain is already installed, so the common case fetches nothing -- the step was making a network
# round trip to fetch packages that were present, and that round trip is the only reason a DNS or mirror fault
# could fail a build at all. Checking first means the normal path touches no network, so neither fault can.
#
# So: check every package with dpkg, install only what is missing, and if apt has to run, give it a time budget
# (doubling back-off to a deadline) instead of a fixed attempt count -- because when something upstream is wrong,
# time is the only lever a retry has.
#
# usage: scripts/ci-apt-install.sh [--deadline-seconds N]
# exit: 0 all packages present (fetching nothing, or after a successful install), 1 the mirror never served them
set -uo pipefail

deadline_seconds=1500
while [ $# -gt 0 ]; do
    case "$1" in
        --deadline-seconds) deadline_seconds=${2:-1500}; shift 2 ;;
        -h|--help) sed -n '17,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "ci-apt-install: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

packages="ca-certificates git python3 python3-venv curl jq file procps make
          gcc-16 g++-16 cmake ninja-build
          libcurl4-openssl-dev libzstd-dev libssl-dev libdrm-dev libpcre2-dev"

# ── 1. what the build needs, and what is already here ───────────────────────────────────────────────────────────
# dpkg rather than `command -v`: a package can be installed without its binary on PATH (a -dev package has no
# binary at all), and this list is mostly the latter.
missing=""
total=0
for pkg in $packages; do
    total=$((total + 1))
    case "$(dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null)" in
        *"install ok installed"*) ;;
        *) missing="$missing $pkg" ;;
    esac
done

if [ -z "$missing" ]; then
    printf 'ci-apt-install: all %d package(s) already installed; nothing to fetch\n' "$total"
    exit 0
fi
printf 'ci-apt-install: %d of %d package(s) missing:%s\n' \
    "$(printf '%s\n' $missing | wc -l)" "$total" "$missing"

# ── 2. install only those, on a time budget ─────────────────────────────────────────────────────────────────────
apt_opts="-o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30"
attempt=0; delay=15; started=$(date +%s); deadline=$((started + deadline_seconds))
while :; do
    attempt=$((attempt + 1))
    # shellcheck disable=SC2086  # option, package and missing lists are intentional word splits
    if apt-get $apt_opts update &&
       DEBIAN_FRONTEND=noninteractive apt-get $apt_opts install -y --no-install-recommends $missing; then
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
