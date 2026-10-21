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
# install-lemonade-ppa.sh [prefix]: run as root from an unpacked engine package
# (1bit-<version>-linux-x86_64/). Puts the package in <prefix> (default /opt/1bit) and points the
# Lemonade PPA's lemond at it (docs/lemonade.md, "Ubuntu 26.04: the PPA's lemond"): the systemd
# drop-in lemonade/1bit-hrx.conf sets LEMONADE_HRX_HRX_BIN to <prefix>/lemonade-hrx, so Lemonade's
# HRX recipe (llamacpp-hrx) runs the engine's llama-server. lemond itself is the PPA's, unpatched:
#   add-apt-repository ppa:lemonade-team/stable && apt install lemonade-server
# NPU model directories stay with `<prefix>/1bit serve`.
set -euo pipefail

src=$(dirname "$(readlink -f "$0")")
prefix=${1:-/opt/1bit}
dropin_dir=${LEMOND_DROPIN_DIR:-/etc/systemd/system/lemond.service.d}

[ "$(id -u)" = 0 ] || { echo "install-lemonade-ppa.sh: run as root (sudo)" >&2; exit 1; }
for f in 1bit lemonade-hrx hrx/llama/bin/llama-server lemonade/1bit-hrx.conf; do
    [ -e "$src/$f" ] || { echo "install-lemonade-ppa.sh: $src/$f missing: run it from an unpacked engine package" >&2; exit 1; }
done
case $prefix in /?*) ;; *) echo "install-lemonade-ppa.sh: the prefix is an absolute path below /" >&2; exit 1 ;; esac
if [ -e "$prefix" ] && [ ! -f "$prefix/1bit" ]; then
    echo "install-lemonade-ppa.sh: $prefix exists and is not an engine package; not replacing it" >&2; exit 1
fi
case $prefix in /home/* | /root/*) echo "install-lemonade-ppa.sh: lemond runs with ProtectHome=yes; pick a prefix outside /home" >&2; exit 1 ;; esac
command -v lemond >/dev/null || echo "install-lemonade-ppa.sh: no lemond yet: add-apt-repository ppa:lemonade-team/stable && apt install lemonade-server" >&2

if [ "$(readlink -f "$src")" != "$(readlink -f "$prefix" 2>/dev/null || echo "$prefix")" ]; then
    rm -rf "$prefix"
    mkdir -p "$(dirname "$prefix")"
    cp -a "$src" "$prefix"
fi
chmod 0755 "$prefix" "$prefix/lemonade-hrx" "$prefix/1bit"

mkdir -p "$dropin_dir"
sed "s|/opt/1bit/lemonade-hrx|$prefix/lemonade-hrx|" "$prefix/lemonade/1bit-hrx.conf" >"$dropin_dir/1bit-hrx.conf"
echo "installed $prefix and $dropin_dir/1bit-hrx.conf (LEMONADE_HRX_HRX_BIN=$prefix/lemonade-hrx)"

# /dev/kfd and /dev/dri/renderD* for HRX0: the PPA's postinst does this when the group exists
if getent group render >/dev/null && getent passwd lemonade >/dev/null && ! id -nG lemonade | grep -qw render; then
    usermod -a -G render lemonade && echo "added the lemonade user to the render group"
fi

if [ -d /run/systemd/system ]; then
    systemctl daemon-reload
    if systemctl is-active --quiet lemond; then systemctl restart lemond && echo "restarted lemond"; fi
fi
[ -d /opt/rocm-therock ] || echo "note: HRX needs TheRock in /opt/rocm-therock (docs/hrx.md)" >&2
