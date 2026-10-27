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
# OpenWebUI, the optional second front end (engine#594), as the upstream container on this box's own
# Lemonade API. Started by systemd, and it may fail: nothing on this box depends on it, so an absent
# runtime, a missing key or a pull with no network is this unit's failure and nothing else's.
#
# The API key is Lemonade's, never a second one: it is read out of the environment file this box's
# lemond itself reads (/etc/1bit/lemonade.env, written by 1bit-appliance's ensure-env.sh), and out of
# a stick's /data/lemonade/lemond.conf where one was prepared elsewhere.
set -u

img=${ONEBIT_OPENWEBUI_IMAGE:-ghcr.io/open-webui/open-webui:main}
data=/data/openwebui
envf=/etc/1bit/openwebui.env

# podman first: it needs no daemon, so there is no second unit to be up before this one works
rt=
for c in podman docker; do
    command -v "$c" >/dev/null 2>&1 && { rt=$c; break; }
done
[ -n "$rt" ] || {
    echo "1bit-openwebui: no container runtime: apt install podman (or docker.io), then systemctl start 1bit-openwebui" >&2
    exit 1
}

key=
for f in /etc/1bit/lemonade.env /data/lemonade/lemond.conf; do
    [ -s "$f" ] || continue
    key=$(sed -n 's/^[[:space:]]*\(export[[:space:]]*\)\{0,1\}LEMONADE_API_KEY=//p' "$f" 2>/dev/null | tail -n 1)
    [ -n "$key" ] && break
done
[ -n "$key" ] || {
    echo "1bit-openwebui: no LEMONADE_API_KEY in /etc/1bit/lemonade.env (is 1bit-lemonade installed and started?)" >&2
    exit 1
}

mkdir -p "$data" /etc/1bit
# the container's data directory is the owner's accounts and chats: not world-readable, and created here
# because on a box that has just installed the package nothing else has made it yet (tmpfiles.d does the
# boot after that, with the same mode)
chmod 700 "$data"
# The container's environment is generated on every start, so rotating the key in Lemonade's own file
# is the whole of the key rotation. WEBUI_SECRET_KEY signs login sessions and must not change under the
# owner, so it is generated once and read back from this file after that.
secret=$(sed -n 's/^WEBUI_SECRET_KEY=//p' "$envf" 2>/dev/null | tail -n 1)
[ -n "$secret" ] || secret=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
umask 077
{
    echo "# 1bit OS: OpenWebUI's environment. Generated at every start from Lemonade's key."
    echo "OPENAI_API_BASE_URL=http://127.0.0.1:13305/api/v1"
    echo "OPENAI_API_KEY=$key"
    echo "WEBUI_SECRET_KEY=$secret"
    echo "WEBUI_PORT=8080"
    # the box's only backend is Lemonade: there is no Ollama endpoint to look for
    echo "ENABLE_OLLAMA_API=false"
} > "$envf"
chmod 600 "$envf"

# --network host: OPENAI_API_BASE_URL above is the host's loopback, and 8080 is the port the console
# and the menu send people to. On a bridge both would need the host's address instead, and 13305
# published by a Lemonade unit that does not do that.
$rt image inspect "$img" >/dev/null 2>&1 || $rt pull "$img" || {
    echo "1bit-openwebui: could not pull $img (no network?); it is tried again on the next start" >&2
    exit 1
}
exec $rt run --rm --name 1bit-openwebui --network host \
    -v "$data":/app/backend/data \
    --env-file "$envf" "$img"
