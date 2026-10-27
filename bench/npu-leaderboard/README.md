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
# NPU leaderboard records

One JSON file per measured run. `tools/npu_leaderboard.py` refuses to build the
leaderboard if a record cannot say how it was measured — that is the point of the
directory, so keep the fields honest rather than convenient.

```json
{
  "id": "27b-npudx-20261027",          // unique; also the file name
  "host": "strixhalo",                  // rows are never compared across hosts
  "model": "Qwen3.8-27B-NPU2",
  "route": "npu",                       // npu | hrx | hybrid
  "quant": "Q4NX (int8 GEMM)",
  "metric": { "name": "decode", "value": 63.1, "unit": "s/token" },
  "settings": { "device": "accel0", "threads": 4, "pin": "0-3", "tokens": 8 },
  "instrument": "wall clock / tokens, 505 s for 8 greedy tokens",
  "repeats": 1,
  "timestamp": "2026-10-27T14:40:00Z",
  "source": "~/lb/587-npudx-smoke.sh run log on strixhalo",
  "notes": "free-form; say what would make this number wrong"
}
```

Rules the generator enforces:

- `id`, `host`, `model`, `route`, `quant`, `metric`, `settings`, `instrument`,
  `repeats`, `timestamp`, `source` are all required;
- `metric.unit` decides the sort direction: `s/token`, `s`, `ms`, `ms/token` rank
  low-to-high, everything else (`tok/s`, ...) ranks high-to-low;
- `repeats` must be a positive integer — a run reported as "stable" without a
  count is not a measurement;
- `instrument` and `source` cannot be empty strings.

Adding a row is adding a file. Then:

```sh
tools/npu_leaderboard.py          # rewrite docs/npu-leaderboard.md
tools/npu_leaderboard.py --check  # validate only
```

Do not edit `docs/npu-leaderboard.md` by hand; it says so at the top.
