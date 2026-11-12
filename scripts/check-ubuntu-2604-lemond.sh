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
# check-ubuntu-2604-lemond.sh <1bit-*-linux-x86_64.tar.zst> <model.gguf> [--gpu]
#
# The install path of docs/lemonade.md ("Ubuntu 26.04: the PPA's lemond"), end to end in a clean
# ubuntu:26.04 container: the Lemonade PPA's lemond, the engine package through
# install-lemonade-ppa.sh, the model registered with Lemonade's HRX recipe (llamacpp-hrx), a load
# and a chat completion through lemond. It passes when lemond started lemonade-hrx (not a download
# of its own HRX build) and the llama-server it ran had GGML_HRX_STRICT=1.
#
# Without --gpu (any x86-64 box, no GPU): the llama-server is a stand-in that records its argv and
# environment and runs Ubuntu's own CPU llama.cpp. With --gpu (Strix Halo): /dev/kfd, /dev/dri and
# /opt/rocm-therock come in from the host and the engine's HRX llama-server runs on HRX0.
set -euo pipefail

# usage(): the header comment as prose, anchored on the comment block rather than a line range -- two of these ranges had
# drifted past the end of their header and printed code to whoever asked for help (engine#2129).
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

[ $# -ge 2 ] || { usage; exit 2; }
pkg=$(readlink -f "$1") model=$(readlink -f "$2") gpu=${3:-}
name=1bit-u2604-check-$$
runtime=${CONTAINER_RUNTIME:-docker}
opts=(-d --name "$name" -v "$pkg:/tmp/engine.tar.zst:ro" -v "$(dirname "$model"):/models:ro")
if [ "$gpu" = --gpu ]; then
    render_gid=$(getent group render | cut -d: -f3)
    video_gid=$(getent group video | cut -d: -f3)
    opts+=(--device /dev/kfd --device /dev/dri --group-add "$render_gid"
           --group-add "$video_gid" -v /opt/rocm-therock:/opt/rocm-therock:ro
           -e RENDER_GID="$render_gid" -e VIDEO_GID="$video_gid")
fi
trap '$runtime rm -f "$name" >/dev/null 2>&1' EXIT
$runtime run "${opts[@]}" ubuntu:26.04 sleep infinity >/dev/null

$runtime exec -e GPU="$gpu" -e GGUF="$(basename "$model")" "$name" bash -euo pipefail -c '
export DEBIAN_FRONTEND=noninteractive
step() { echo "== $*"; }
step "the Lemonade PPA and its lemond"
apt-get update -qq && apt-get install -y -qq software-properties-common zstd curl python3 >/dev/null
add-apt-repository -y ppa:lemonade-team/stable >/dev/null 2>&1
apt-get install -y -qq lemonade-server >/dev/null
echo "lemond $(dpkg-query -W -f="\${Version}" lemonade-server)"

step "the engine package, through install-lemonade-ppa.sh"
mkdir -p /tmp/engine && tar -C /tmp/engine -I zstd -xf /tmp/engine.tar.zst
/tmp/engine/*/install-lemonade-ppa.sh
grep -v "^#" /etc/systemd/system/lemond.service.d/1bit-hrx.conf
# no systemd in the container: lemond runs with the drop-in Environment= lines, as the lemonade user
drop_env=$(sed -n "s/^Environment=//p" /etc/systemd/system/lemond.service.d/1bit-hrx.conf)

extra_env=()
if [ "$GPU" != --gpu ]; then
    step "CPU stand-in for the HRX llama-server (no gfx1151 here)"
    apt-get install -y -qq llama.cpp >/dev/null
    mkdir -p /opt/standin
    cat >/opt/standin/llama-server <<"EOF"
#!/bin/bash
{ echo "ARGV: $*"; env | grep -E "^(GGML_|HSA_|HRX_|IREE_)" | sort; } >>/tmp/backend.log
a=(); skip=0; for x in "$@"; do if [ $skip = 1 ]; then skip=0; continue; fi; [ "$x" = --device ] && { skip=1; continue; }; a+=("$x"); done
exec /usr/bin/llama-server "${a[@]}" -ngl 0
EOF
    chmod 755 /opt/standin/llama-server
    extra_env=(ONEBIT_HRX_LLAMA_SERVER=/opt/standin/llama-server)
fi

step "the model, registered with Lemonade'"'"'s HRX recipe (llamacpp-hrx)"
C=/var/lib/lemonade/.config/lemonade
H=/var/lib/lemonade/.cache/huggingface/hub/models--local--check-GGUF
snap=$H/snapshots/0000000000000000000000000000000000000000
mkdir -p "$C" "$snap" "$H/refs"
printf %s 0000000000000000000000000000000000000000 >"$H/refs/main"
ln -sf "/models/$GGUF" "$snap/$GGUF"
# without a gfx1100/gfx1151 Lemonade hides HRX models; on Strix Halo the filter lets it through
[ "$GPU" = --gpu ] && echo "{}" >"$C/config.json" || echo "{\"disable_model_filtering\": true}" >"$C/config.json"
echo "{\"Check-HRX\": {\"checkpoint\": \"local/check-GGUF:$GGUF\", \"recipe\": \"llamacpp-hrx\", \"labels\": [\"chat\"]}}" >"$C/user_models.json"
chown -R lemonade:lemonade /var/lib/lemonade

step "lemond"
cd /var/lib/lemonade
# runuser gives the user only the groups of the container image /etc/group, so the
# render GID that --group-add put on the main container process is gone by the time lemond runs:
# /dev/kfd is root:render, HSA then cannot open it, libggml-hrx registers no device and
# llama-server refuses --device HRX0 with "invalid device: HRX0" (measured 2026-10-31 on Strix
# Halo). setpriv keeps the host render and video GIDs, so the --gpu path reaches HRX0.
launcher=(runuser -u lemonade --)
if [ "$GPU" = --gpu ]; then
    # --init-groups and --groups are mutually exclusive in setpriv (it exits 1 on both), and the
    # only supplementary groups this process needs are the two that open the GPU nodes.
    launcher=(setpriv --reuid lemonade --regid "$(id -g lemonade)" --groups "$RENDER_GID,$VIDEO_GID" --)
fi
"${launcher[@]}" env HOME=/var/lib/lemonade $drop_env "${extra_env[@]}" /usr/bin/lemond >/tmp/lemond.log 2>&1 &
for _ in $(seq 60); do curl -sf localhost:13305/api/v1/health >/dev/null && break; sleep 1; done
curl -s localhost:13305/api/v1/models | python3 -c "import json,sys; [print(\"listed:\", m[\"id\"], m[\"recipe\"]) for m in json.load(sys.stdin)[\"data\"]]"
curl -sf -m 600 localhost:13305/api/v1/load -H "Content-Type: application/json" -d "{\"model_name\": \"Check-HRX\", \"ctx_size\": 4096}"; echo
curl -sf -m 600 localhost:13305/api/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\": \"Check-HRX\", \"max_tokens\": 32, \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+3? /no_think\"}]}" |
    python3 -c "import json,sys; print(\"reply:\", json.load(sys.stdin)[\"choices\"][0][\"message\"][\"content\"].strip())"

step "what lemond started"
# several llama-servers can be up (Lemonade plus a backend): take the first pid. Not `| head -1`,
# which under pipefail would kill pgrep and abort the container script. (This block is a single-quoted
# string, so no apostrophes and no $'...' in here: sed -n 1p reads to the end, head does not.)
pid=$(pgrep -x llama-server | sed -n 1p)
# the argv is where the device is named: 1bit serve logs the memory it checked but not the device,
# so grepping lemond.log for HRX0 (as this script did until 2026-10-31) fails on a working run
cmdline=$(tr "\0" " " </proc/$pid/cmdline)
printf "%s\n" "$cmdline"
runuser -u lemonade -- cat /proc/$pid/environ | tr "\0" "\n" | grep -E "^(GGML_HRX|HSA_|HRX_|IREE_)" | sort
grep -E "1bit serve|HRX|hrx" /tmp/lemond.log | tail -n 20
fail=0
# NOT `... | grep -qx`: grep leaving on the match kills tr (141) and pipefail then turns a variable
# that IS set into "FAIL: no GGML_HRX_STRICT=1". Capture the environment, then match it in the shell
# (a here-string is a temp file, so there is no pipe left for grep to kill).
env2=$(runuser -u lemonade -- cat /proc/$pid/environ | tr "\0" "\n")
grep -qx GGML_HRX_STRICT=1 <<<"$env2" || { echo "FAIL: no GGML_HRX_STRICT=1"; fail=1; }
# hrx.hrx_bin set to a path skips Lemonade'"'"'s install of its own HRX build (cache/bin/llamacpp-hrx)
[ ! -e /var/lib/lemonade/.cache/lemonade/bin/llamacpp-hrx ] || { echo "FAIL: lemond installed its own HRX build"; fail=1; }
# --gpu means the engine HRX llama-server on the GPU, i.e. the backend was started with --device HRX0
[ "$GPU" != --gpu ] || grep -q -- "--device HRX0" <<<"$cmdline" || { echo "FAIL: the backend was not started on HRX0"; fail=1; }
[ $fail = 0 ] && echo "PASS" || exit 1
'
