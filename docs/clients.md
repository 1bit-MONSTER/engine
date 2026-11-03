<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Connect a client

1bit OS runs [Lemonade Server](https://github.com/lemonade-sdk/lemonade), and Lemonade is what a client
talks to. On the appliance it listens on **`0.0.0.0:13305`** — every interface, not just loopback — with
an OpenAI-compatible API under **`/api/v1`** (and `/v1`), a WebSocket on **9000**, and its web app at the
root. Every request needs the API key as a Bearer token:

```
Authorization: Bearer <key>
```

The key is the only gate. Anything on the same network that has it can use the box.

| client | how it connects |
|---|---|
| [`curl`](#2-curl) | the URL plus the header above |
| [the Lemonade web app](#3-the-lemonade-web-app-in-a-browser) | `http://<box>:13305` in a browser |
| [Lemonade Mobile](#4-lemonade-mobile-on-a-phone) | the server URL and the key, in the app's settings |
| [OpenAI SDKs and OpenAI-compatible apps](#5-openai-sdks-and-openai-compatible-apps) | base URL `http://<box>:13305/api/v1` + the key |
| [the `lemonade` CLI](#6-the-lemonade-cli-from-another-machine) | `--host`, `--port`, `--api-key` |

## 1. Find the box, and its key

**The address.** On the box's console or over SSH:

```sh
hostname -I                 # e.g. 192.168.1.42
hostname                    # 1bit-os - the name the image sets in /etc/hostname
```

**The name is `1bit-os`, not an address.** Puppy only invents a per-machine `puppypc<digits>` while
`/etc/hostname` still holds its stock word, and the shipped image sets `1bit-os` instead (measured: on the
cycle-13 image it still booted as `puppypc19108`, which is why older notes name it that way). Read the
address off the box rather than guessing `1bit.local` — mDNS is not claimed. If you cannot get to a
console, the router's client list shows the box
(the MAC of the wired or wireless interface), and the box announces the server on the LAN itself —
`lemond` logs `Net Broadcast` on each RFC1918 interface, and `lemonade scan` (run anywhere on the same
network with the CLI installed) lists the beacons it hears.

**The key.** It is one line in Lemonade's environment file on the `1BIT-DATA` partition:

```sh
sudo sed -n 's/^LEMONADE_API_KEY=//p' /data/lemonade/lemond.conf
```

`/etc/1bit/lemond.conf` is a symlink to that same file, so either path reads the same key. It is
`0600 root`, minted by `prepare-stick --mint` when the stick is written, and written on first boot if the
file is empty. There is no `lemonade` CLI command that prints it — the file is the interface.

## 2. `curl`

List the models, with the key:

```sh
BOX=192.168.1.42
KEY=$(ssh root@$BOX "sed -n 's/^LEMONADE_API_KEY=//p' /data/lemonade/lemond.conf")   # or copy it by hand

curl -s -H "Authorization: Bearer $KEY" http://$BOX:13305/api/v1/models
```

One chat completion, and the one-line check that it worked:

```sh
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-0.6B-NPU-1bit","messages":[{"role":"user","content":"Say OK."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  http://$BOX:13305/api/v1/chat/completions
```

`http=200` and a `choices[0].message.content` in the body is the whole check. Without the key the same
request answers **401**; with a wrong key, also 401.

The `chat_template_kwargs` bit is for the thinking models in the catalogue: without it Qwen3 answers in
`reasoning_content` and `content` stays empty, which is correct but reads like a failure in a one-line
test. Drop it if you want the reasoning, and read `reasoning_content` instead.

**Which model id.** `GET /api/v1/models` is the list to use — it is Lemonade's catalogue plus whatever is
on the stick's `1BIT-DATA:models/`, registered by the boot. On the shipped appliance that includes the
HRX rows (`Qwen3-0.6B-1bit`, `Qwen3-4B-1bit`, `Qwen3.8-27B-1bit`, `gpt-oss-20b-mxfp4-1bit`, …) and the
NPU rows (`Qwen3-0.6B-NPU-1bit`, `Qwen3-1.7B-NPU-1bit`, `Qwen3-4B-NPU-1bit`,
`Qwen3.6-35B-A3B-NPU-1bit`). The engine's own model list is in `/etc/1bit/onebit-models.txt` on the box.
Those rows are **filtered by the hardware**: on a machine with neither the GPU nor the NPU (a VM, say)
the onebit rows do not appear, and `GET /api/v1/models?show_all=true` lists the whole catalogue instead.
`Qwen3-0.6B-NPU-1bit` is the one the board boots test with: "1BIT OK." at 93.2 / 95.3 tok/s through
Lemonade (2026-11-02 board run).

## 3. The Lemonade web app in a browser

Open `http://<box>:13305` from any machine on the LAN. Lemonade's web app is served there — chat, the
model catalogue, downloads, settings and logs. The page itself loads without a key (it is static), and
every request it makes needs one, so the first time it asks for the API key; paste the key from
[1](#1-find-the-box-and-its-key).

## 4. Lemonade Mobile on a phone

Lemonade's phone client is **Lemonade Mobile**, by Geramy Loveless, and it is the front door to the same
server: no cloud account and no Lemonade service in the path, just the box you configured.

| | |
|---|---|
| iPhone / iPad | [App Store 6757372210](https://apps.apple.com/app/id6757372210), "Lemonade Mobile" (Lemonade Chat), iOS/iPadOS 13 or later |
| Android | [Lemonade Mobile Omni](https://play.google.com/store/apps/details?id=com.lemonade.mobile.chat.ai) on Google Play (`com.lemonade.mobile.chat.ai`), the same app family from the same developer |

In the app: **Settings → Manage servers → add a server**, then

- **URL**: `http://<box>:13305` (the scheme and the port are part of it; no trailing `/api/v1`)
- **API key**: the key from [1](#1-find-the-box-and-its-key)

The app manages several servers, so a second box (or the same model on another machine) is a second
entry. The App Store listing calls the API key optional — that is for a Lemonade running with no key. **On
1bit OS it is required**: without it the app gets 401 from every request.

## 5. OpenAI SDKs and OpenAI-compatible apps

Any OpenAI client works; the two values are the base URL and the key. Python, against the `openai`
package:

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.1.42:13305/api/v1", api_key="<the key>")

print([m.id for m in client.models.list()])
r = client.chat.completions.create(model="Qwen3-0.6B-NPU-1bit",
                                   messages=[{"role": "user", "content": "Say OK."}],
                                   max_tokens=16, temperature=0,
                                   extra_body={"chat_template_kwargs": {"enable_thinking": False}})
print(r.choices[0].message.content)
```

Without `extra_body` a thinking model puts its answer in `reasoning_content` and leaves `content` empty,
the same as the `curl` example above.

The same two values fill in any app that speaks the OpenAI API and lets you set a base URL — Open WebUI,
Continue, a chat front-end, an agent framework. Give it `http://<box>:13305/api/v1` (or `…/v1`; Lemonade
answers both) and the key. Lemonade's own docs keep a list of clients it is tested with at
[lemonade-server.ai](https://lemonade-server.ai).

## 6. The `lemonade` CLI from another machine

Where Lemonade's CLI is installed (it ships with Lemonade Server), point it at the box:

```sh
lemonade --host 192.168.1.42 --port 13305 --api-key "$KEY" status
lemonade --host 192.168.1.42 --port 13305 --api-key "$KEY" list
lemonade --host 192.168.1.42 --port 13305 --api-key "$KEY" load Qwen3-0.6B-NPU-1bit
```

The same three values have environment variables, which is how a script sets them once:
`LEMONADE_HOST`, `LEMONADE_PORT`, `LEMONADE_API_KEY`. `lemonade scan` finds servers on the LAN that
announce themselves (the beacon `lemond` sends — the same one the mobile app can use).

On the box itself the CLI needs no `--host`: `lemonade status` uses `127.0.0.1:13305`, and it still needs
`--api-key` (or `LEMONADE_API_KEY` in the environment) because the server requires it on every route.

## 7. Security: the key is the whole gate

| | |
|---|---|
| the gate | `LEMONADE_API_KEY` on every `/api/*`, `/v0/*` and `/v1/*` request, as a Bearer token. No key, no service |
| network | the appliance opens **13305** (Lemonade) and **22** (SSH) in its firewall, and allows the RFC1918 ranges through `/etc/hosts.allow`; everything else inbound is dropped. It is a LAN service |
| never | forward 13305 from your router to the internet. If you need it from outside, use a VPN, or an HTTPS reverse proxy in front of it — Lemonade's own [FAQ](https://lemonade-server.ai/docs/guide/faq/) says the same |
| SSH | keys only: `/data/ssh/authorized_keys` on the stick is copied to `/root/.ssh/authorized_keys` at boot. `prepare-stick --ssh-pubkey` writes it |
| rotate the key | write the new one and restart the service: `sudo sh -c 'printf "LEMONADE_API_KEY=%s\n" "$(head -c 24 /dev/urandom \| od -An -tx1 \| tr -d " \\n")" > /data/lemonade/lemond.conf' && sudo /etc/init.d/10lemonade restart`. Every client's key has to be updated with it. Re-running `prepare-stick --mint` on a reflashed stick does the same thing offline |

## Sources

- the App Store listing, [apps.apple.com/app/id6757372210](https://apps.apple.com/app/id6757372210) — name,
  developer, iOS requirement, "Settings → Manage servers"
- [Lemonade Mobile Omni on Google Play](https://play.google.com/store/apps/details?id=com.lemonade.mobile.chat.ai)
- Lemonade's [FAQ](https://lemonade-server.ai/docs/guide/faq/) (remote access) and
  [CLI reference](https://lemonade-server.ai/docs/guide/cli/) (`--host`, `--port`, `--api-key`,
  `--discovery`, and their `LEMONADE_*` variables)

## What was verified for this page

Against the cycle-13 appliance image booted in QEMU (`sha256 dcc619ff…`, kernel
`7.3.0-rc4-1bit-perfopt+`, Secure Boot on):

| claim | measured |
|---|---|
| Lemonade listens on every interface | `netstat`: `lemond` on `0.0.0.0:13305`, WebSocket `9000`, `sshd` on `0.0.0.0:22` |
| `/api/v1` and `/v1` are the API | `GET /api/v1/models` and `GET /v1/models` both answer |
| the key is required, and only the key | with a key minted: no key and a wrong key → **401**, the right key → **200** (both paths) |
| the key's file | `readlink -f /etc/1bit/lemond.conf` → `/data/lemonade/lemond.conf`, `0600 root`; a key written there plus `/etc/init.d/10lemonade restart` takes effect on the next request |
| the web app | `GET http://<box>:13305/` → 200, `text/html`, `<title>Lemonade App</title>`, with and without a key (it is static; its own API calls are what 401) |
| the `curl` chat example | `POST /api/v1/chat/completions` with `chat_template_kwargs.enable_thinking=false`: **200**, `content: "Sure! 😊"`, `finish_reason: stop` |
| the OpenAI SDK example | run from another machine against the forwarded port with `base_url=…/api/v1` and `api_key`: `models.list()` → `['rlk-cpu']`, `chat.completions.create()` → 200 with `content: 'Sure! 😊'` once `enable_thinking` is off |
| the CLI flags | `lemonade --help` on the box: `--host HOST`, `--port PORT`, `--api-key KEY`, `--discovery/--no-discovery`, `LEMONADE_HOST/PORT/API_KEY`; `lemonade --host 10.0.2.2 --port 21305 --api-key <key> status` answered |
| the discovery beacon | `lemonade scan` on the box: `Discovered: puppypc19045 at http://127.0.0.1:13305/api/v1/`; `lemond` logs `Net Broadcast` on each RFC1918 interface |
| the firewall | `iptables -S INPUT`: `-P INPUT DROP` with `ACCEPT` for `--dport 22` and `--dport 13305`; `/etc/hosts.allow` carries the RFC1918 `1bit-lan` line |
| Lemonade Mobile | **not run on a phone here** — verified as: the two store listings exist with the names above, the app's own "Getting started" text gives the same URL + API-key flow, and the HTTP API it talks to is the one measured above |

The QEMU VM has no GPU and no NPU, so its `GET /api/v1/models` lists none of the appliance's onebit rows
(they are filtered by hardware) and a onebit model cannot be loaded there. The chat examples above were
therefore exercised on a GGUF registered through the same `user_models.json` mechanism with the `llamacpp`
recipe — the answer came from llama.cpp on the VM's CPU at 1.7-2.3 tok/s, not from the engine. What the
board run adds, and this page does not re-measure, is the onebit route itself: `Qwen3-0.6B-NPU-1bit`
through Lemonade on the Ryzen AI MAX+ 395.

Not verified on this page: a phone against a real box on a real LAN (the app needs a device), and mDNS —
avahi-daemon runs on the box, but this page does not claim `<name>.local` resolves, so read the address
with `hostname -I` or the router instead.
