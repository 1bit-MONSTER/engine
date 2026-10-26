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
# First boot of an installed 1bit OS: own SSH host keys, and the marker release-check reads for the
# mode row. On a live boot /etc/1bit/live exists instead and this is skipped by the unit's condition.
set -u
mkdir -p /etc/1bit
if [ ! -s /etc/ssh/ssh_host_ed25519_key ]; then
    ssh-keygen -q -t ed25519 -N '' -f /etc/ssh/ssh_host_ed25519_key
fi
# the live image's public key is not this machine's identity
: > /etc/1bit/installed
if [ -s /etc/1bit/live ]; then rm -f /etc/1bit/live; fi
echo "1bit: first boot on this machine (SSH host key generated, /etc/1bit/installed written)"
