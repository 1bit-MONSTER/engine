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
# First-boot state Lemonade cannot start without: the API key. Idempotent, so it runs as the service's
# ExecStartPre on every boot and the key never changes underneath the owner.
set -u
env=/etc/1bit/lemonade.env
[ -s "$env" ] && exit 0
mkdir -p /etc/1bit
key=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
umask 077
{
    echo "# 1bit OS: Lemonade's environment. The API key is what release-check's lemonade row needs."
    echo "LEMONADE_API_KEY=$key"
    echo "LEMONADE_ADMIN_API_KEY=$key"
} > "$env"
chmod 600 "$env"
echo "1bit: wrote $env with a new LEMONADE_API_KEY"
