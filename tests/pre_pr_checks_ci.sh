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
# tests/pre_pr_checks_ci.sh — run the pre-PR checks from CI, where the checkout has no history (engine#3131).
#
# `actions/checkout` at its default depth fetches one ref, so `origin/main` does not resolve and the four rows that
# compare against a base answer cannot-tell: the incident-catching half of the suite is then not enforced by the
# `pre_pr_checks_tree` ctest row, only reported. `git fetch --unshallow origin` costs 4.9 s on this repository
# (measured 2026-11-21), so the row asks for the history first and keeps `--base-optional` for the case where the
# fetch cannot happen (no network, no such remote): eight rows enforced beats twelve rows red for an environment
# the PR did not choose.
#
# usage: tests/pre_pr_checks_ci.sh [--plan] [--runner PATH] [--base REF]
#   --plan   print what it would run and exit 0 (the arms use this; nothing is fetched and no check runs)
set -uo pipefail

plan=0
here=$(cd "$(dirname "$0")/.." && pwd)
runner=$here/scripts/pre-pr-checks.sh
base=origin/main
while [ $# -gt 0 ]; do
    case "$1" in
        --plan) plan=1; shift ;;
        --runner) runner=${2:?--runner needs a path}; shift 2 ;;
        --base) base=${2:?--base needs a ref}; shift 2 ;;
        -h|--help) sed -n '18,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "pre_pr_checks_ci: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

say() { printf 'pre_pr_checks_ci: %s\n' "$*" >&2; }

fallback=0
shallow=$(git rev-parse --is-shallow-repository 2>/dev/null || echo unknown)
if [ "$shallow" = true ]; then
    say "the checkout is shallow, so the four --base rows would answer cannot-tell: fetching history"
    if [ "$plan" = 1 ]; then
        say "(plan) would run: git fetch --unshallow --quiet origin"
    elif timeout 300 git fetch --unshallow --quiet origin 2>/dev/null; then
        say "unshallowed"
    else
        say "could not unshallow: the --base rows will report cannot-tell, counted and named"
        fallback=1
    fi
fi
if ! git rev-parse --verify -q "$base^{commit}" >/dev/null; then
    if [ "$plan" = 1 ]; then
        say "(plan) $base does not resolve; would try: git fetch --quiet origin ${base#origin/}"
        fallback=1
    elif timeout 120 git fetch --quiet origin "${base#origin/}" 2>/dev/null && git rev-parse --verify -q "$base^{commit}" >/dev/null; then
        say "fetched $base"
    else
        say "$base does not resolve here: the --base rows will report cannot-tell, counted and named"
        fallback=1
    fi
fi

argv=(--base "$base" --tree-only)
[ "$fallback" = 1 ] && argv+=(--base-optional)
if [ "$plan" = 1 ]; then
    printf 'argv: %s\n' "${argv[*]}"
    exit 0
fi
exec bash "$runner" "${argv[@]}"
