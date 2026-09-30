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
# census-watch.sh — daily HF new-model watcher entry point
#
# Wraps tools/census_watch.py so the daily watch is one command, usable from the systemd
# units (scripts/1bit-census-watch.{service,timer}) or by hand. The automation is reproducible from the repo
# instead of living ad-hoc on one box.
#
# What it does: polls the newest HF text-generation models, reads each one's architecture
# class from the config inline in the listing, and compares it against the generated
# registry (registry/architectures.json) — the same read tools/census.py's coverage()
# makes. Any new class the registry does not map is the alert: that is the one thing that
# silently moves the mapped share the census reports.
#
# Exit codes (same contract as tools/census_watch.py):
#   0  no uncovered classes among the newest models
#   1  an uncovered class arrived (the alert)
#   2  the refresh step refused to touch the checkout
#   *  runtime failure (no network, unreadable registry, ...)
#
# Usage:
#   scripts/census-watch.sh [--limit N]      # N newest models, default 120
#
# Every run is appended to $CENSUS_WATCH_LOG_DIR (default ~/.1bit/logs) AND echoed to
# stdout, so systemd/journald/CI all see the same output. Watcher state (the model ids
# already classified) lives at $CENSUS_WATCH_STATE, outside the checkout.
#
# CENSUS_WATCH_REFRESH=1 fast-forwards the checkout to origin/main before each run. It
# refuses to touch a checkout that is on a branch or has local changes, so pointing this
# at a working lane can never wipe it. It is meant for the ~/census-main worktree, which
# is a `git worktree add --detach ~/census-main origin/main`.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${CENSUS_WATCH_LOG_DIR:-${HOME}/.1bit/logs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG_DIR}/census-watch-${STAMP}.log"

mkdir -p "${LOG_DIR}"

if [ "${CENSUS_WATCH_REFRESH:-0}" = "1" ]; then
    branch="$(git -C "${ROOT}" rev-parse --abbrev-ref HEAD)"
    if [ "${branch}" != "HEAD" ]; then
        echo "census-watch: refusing to refresh: ${ROOT} is on branch ${branch}" >&2
        exit 2
    fi
    if ! git -C "${ROOT}" diff --quiet || ! git -C "${ROOT}" diff --cached --quiet; then
        echo "census-watch: refusing to refresh: ${ROOT} has local changes" >&2
        exit 2
    fi
    git -C "${ROOT}" fetch -q origin main
    git -C "${ROOT}" reset -q --hard origin/main
    echo "census-watch: refreshed ${ROOT} to origin/main ($(git -C "${ROOT}" rev-parse --short HEAD))"
fi

echo "== census-watch $(date -u +%FT%TZ) ==" | tee "${LOG}"

set +e
python3 "${ROOT}/tools/census_watch.py" "$@" 2>&1 | tee -a "${LOG}"
RC=${PIPESTATUS[0]}
set -e

if [ "${RC}" -ne 0 ]; then
    echo "census-watch: EXIT ${RC} (an uncovered class needs a registry mapping;" \
         "full log: ${LOG})" | tee -a "${LOG}"
fi

exit "${RC}"
