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
# install_accel_sampler.sh: deploy tools/diag/accel_sampler.sh together with the provenance of what was deployed
# (engine#2795).
#
# The unit runs a copy in ~/.cache, placed by hand, so the running instrument and the repository drifted to 207 vs 253
# lines with nothing noticing -- and every summary it wrote was labelled with a timestamp but no version. Two things fix
# that: the copy's sha256 and commit are written beside it as `deployed.provenance`, and the script prints both (with
# DRIFTED when they disagree) in every summary it writes.
#
# usage: install_accel_sampler.sh [--repo <checkout>] [--to <path>] [--selftest]
#   --repo      whose tools/diag/accel_sampler.sh to install          (default: this script's repository)
#   --to        where to write it   (default: $ACCEL_SAMPLER_DIR/accel-sampler.sh, i.e. ~/.cache/accel-sampler/…)
#   --selftest  after installing, prove the deployed copy reports this provenance (and that a modified copy does not)
# It never restarts the unit: that is the owner's step, and this script says so rather than doing it.
set -u

repo=$(cd "$(dirname "$0")/../.." && pwd -P)
to=${ACCEL_SAMPLER_DIR:-$HOME/.cache/accel-sampler}/accel-sampler.sh
selftest=0
while [ $# -gt 0 ]; do
    case $1 in
        --repo) repo=$2; shift 2 ;;
        --to) to=$2; shift 2 ;;
        --selftest) selftest=1; shift ;;
        -h|--help) sed -n '18,32p' "$0"; exit 0 ;;
        *) echo "install_accel_sampler: unknown option $1 (try --help)" >&2; exit 2 ;;
    esac
done

src=$repo/tools/diag/accel_sampler.sh
[ -f "$src" ] || { echo "install_accel_sampler: no $src" >&2; exit 2; }
mkdir -p "$(dirname "$to")" || exit 2

commit=$(git -C "$repo" rev-parse --short=9 HEAD 2> /dev/null || echo unknown)
dirty=""
git -C "$repo" diff --quiet -- tools/diag/accel_sampler.sh 2> /dev/null || dirty="-dirty"
sha=$(sha256sum -- "$src" | cut -c1-12)

# keep what was there, so a bad install is one `cp` away from undone
[ -f "$to" ] && cp -p "$to" "$to.pre-$(date -u +%Y%m%dT%H%M%SZ)" 2> /dev/null
install -m 755 "$src" "$to" || { echo "install_accel_sampler: install failed" >&2; exit 2; }
cat > "$to.provenance" <<EOF
sha=$sha
commit=${commit}${dirty}
source=$src
installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
installed_by=$(id -un)@$(hostname -s 2> /dev/null || echo unknown)
EOF
echo "installed $to"
echo "  sha=$sha commit=${commit}${dirty}"
echo "  provenance: $to.provenance"
echo "  the unit is NOT restarted by this script: 'systemctl --user restart accel-sampler' is the owner's step"

if [ "$selftest" = 1 ]; then
    ACCEL_SAMPLER_DIR=$(dirname "$to") ACCEL_SAMPLER_META="$to.provenance" \
        bash "$to" summary > /dev/null 2>&1
    sum=$(dirname "$to")/summary.txt
    if grep -q "script sha256 ${sha};" "$sum" 2> /dev/null; then
        echo "  selftest: the deployed copy reports sha $sha in its summary"
    else
        echo "  selftest FAIL: the summary does not name the deployed sha $sha ($sum)" >&2
        exit 1
    fi
fi
