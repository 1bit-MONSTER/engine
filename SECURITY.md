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
# Security

## Reporting a vulnerability

Please report security problems privately, through GitHub's
[private vulnerability reporting](https://github.com/1bit-MONSTER/engine/security/advisories/new)
(the **Security** tab, then **Report a vulnerability**). Do not open a public issue or discussion
for them.

Useful to include: what is affected (`1bit serve`, a build script, a workflow, a package), how to
reproduce it, and what an attacker gains. We reply on the advisory, and credit reporters in it
unless they ask us not to.

## Scope

- **The engine:** `1bit` and `1bit serve`, and the scripts and workflows in this repository.
- **Our packages:** the weekly release packages (the Linux tarball, the Windows zip, 1bit OS).
- **Out of scope here:** the pinned upstreams (llama.cpp, HRX, ZINC, Lemonade, the XDNA driver and
  the others in `third_party/`). Report those to their own projects, though we want to know if a
  pin we ship is affected.

`1bit serve` listens on `127.0.0.1` unless it is told otherwise. It has no authentication of its
own; put it behind something that does before exposing it to a network.

## Supported versions

The latest weekly release and `main`.
