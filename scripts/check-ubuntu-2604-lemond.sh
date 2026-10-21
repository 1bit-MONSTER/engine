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

[ $# -ge 2 ] || { sed -n '16,27p' "$0"; exit 2; }
pkg=$(readlink -f "$1") model=$(readlink -f "$2") gpu=${3:-}
name=1bit-u2604-check-$$
runtime=${CONTAINER_RUNTIME:-docker}
opts=(-d --name "$name" -v "$pkg:/tmp/engine.tar.zst:ro" -v "$(dirname "$model"):/models:ro")
if [ "$gpu" = --gpu ]; then
    opts+=(--device /dev/kfd --device /dev/dri --group-add "$(getent group render | cut -d: -f3)"
           --group-add "$(getent group video | cut -d: -f3)" -v /opt/rocm-therock:/opt/rocm-therock:ro)
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
runuser -u lemonade -- env HOME=/var/lib/lemonade $drop_env "${extra_env[@]}" /usr/bin/lemond >/tmp/lemond.log 2>&1 &
for _ in $(seq 60); do curl -sf localhost:13305/api/v1/health >/dev/null && break; sleep 1; done
curl -s localhost:13305/api/v1/models | python3 -c "import json,sys; [print(\"listed:\", m[\"id\"], m[\"recipe\"]) for m in json.load(sys.stdin)[\"data\"]]"
curl -sf -m 600 localhost:13305/api/v1/load -H "Content-Type: application/json" -d "{\"model_name\": \"Check-HRX\", \"ctx_size\": 4096}"; echo
curl -sf -m 600 localhost:13305/api/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\": \"Check-HRX\", \"max_tokens\": 32, \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+3? /no_think\"}]}" |
    python3 -c "import json,sys; print(\"reply:\", json.load(sys.stdin)[\"choices\"][0][\"message\"][\"content\"].strip())"

step "what lemond started"
pid=$(pgrep -x llama-server | head -1)
tr "\0" " " </proc/$pid/cmdline; echo
runuser -u lemonade -- cat /proc/$pid/environ | tr "\0" "\n" | grep -E "^(GGML_HRX|HSA_|HRX_|IREE_)" | sort
grep -E "1bit serve|HRX|hrx" /tmp/lemond.log | tail -n 20
fail=0
runuser -u lemonade -- cat /proc/$pid/environ | tr "\0" "\n" | grep -qx GGML_HRX_STRICT=1 || { echo "FAIL: no GGML_HRX_STRICT=1"; fail=1; }
# hrx.hrx_bin set to a path skips Lemonade'"'"'s install of its own HRX build (cache/bin/llamacpp-hrx)
[ ! -e /var/lib/lemonade/.cache/lemonade/bin/llamacpp-hrx ] || { echo "FAIL: lemond installed its own HRX build"; fail=1; }
[ "$GPU" != --gpu ] || grep -q "HRX0" /tmp/lemond.log || { echo "FAIL: no HRX0 in the backend log"; fail=1; }
[ $fail = 0 ] && echo "PASS" || exit 1
'
