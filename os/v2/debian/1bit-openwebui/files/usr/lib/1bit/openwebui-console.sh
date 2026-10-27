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
# The console line for the optional front end (engine#594), written where the banner it belongs beside
# already is: /etc/issue.d/*.issue, which agetty prints after /etc/issue in version-sort order
# (util-linux >= 2.36; Ubuntu 26.04 ships 2.41, and debian/1bit-appliance ships /etc/issue itself).
#
# The line is the port answering or it says so, exactly as the menu's own 13305 probe reads: 8080
# answers -> (up), it does not -> (starting). Gone with the package (postrm), so a box without it never
# says anything about OpenWebUI.
set -u
unit=/usr/lib/systemd/system/1bit-openwebui.service
dir=${ISSUE_DIR:-/etc/issue.d}   # ISSUE_DIR is for the check that runs this off a box; /etc/issue.d on one
out=$dir/50-1bit-openwebui.issue
[ -e "$unit" ] || { rm -f "$out"; exit 0; }

ip=$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
[ -n "$ip" ] || ip="<this-box>"
if wget -q -O /dev/null -T 2 http://127.0.0.1:8080/ 2>/dev/null; then st=up; else st=starting; fi

mkdir -p "$dir"
umask 022
printf '  Chat UI ............... http://%s:8080/   (%s)\n' "$ip" "$st" > "$out"
