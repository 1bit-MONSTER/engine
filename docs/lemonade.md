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
# Embedded Lemonade

`1bit lemonade` runs [Lemonade](https://github.com/lemonade-sdk/lemonade)'s
server core inside the `1bit` process. That includes every Lemonade backend,
its router and its model catalog. The engine's own `onebit` backend is
compiled in alongside them. Everything after `lemonade` goes to Lemonade's own CLI:

```bash
1bit lemonade --port 13305                  # the same options as lemond
1bit lemonade --help
```

## Vendored tree

`third_party/lemonade` is Lemonade v11.9.0, vendored complete, plus the local
deltas listed in its `UPSTREAM.md`:

- the `onebit` backend and `GET /v1/registry`
- the `hrx-b66` pin and the HRX model-registry annotations
- a small CMake patch that makes it embeddable

It is copied from 1bit-MONSTER `main` (`third_party/lemonade`, last changed in
`256e68dd7`). Files in `third_party/` keep their own license (Apache-2.0) and
are exempt from this repository's copyright notice.

## What step 1 covers

- The server core builds into `1bit` (about 13 MB) and starts.
- `/api/v1/health` reports `ok`.
- The catalog lists every recipe, including `llamacpp-hrx`.
- `/api/v1/system-info` answers.
- `tests/smoke_lemonade.sh` checks all of this with scratch cache and config
  dirs, in CI and in `ctest`.

On Strix Halo the catalog has 197 models across 16 recipes, and Lemonade
reports the NPU as present.

## Not yet

- **The `onebit` backend lists no models.** Native artifacts are registered
  when the NPU engine lands (step 3).
- **Lemonade's HRX recipe downloads AMD's HRX build.** Step 2 points it at this
  repository's HRX + Vulkan build instead.
- **The web UI is not built.** That needs Node.js; without it, Lemonade serves
  its static status page at `/`.
