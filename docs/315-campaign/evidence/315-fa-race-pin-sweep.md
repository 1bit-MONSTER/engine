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
<!-- Copyright 2026 bong-water-water-bong
     SPDX-License-Identifier: Apache-2.0

     engine#315 evidence: the standalone FA-race harness pointed at the pin's build, 2026-10-09.
     Harness: ~/.cache/kvnan/fa-race.cpp (the earlier investigation's own debug harness, header says
     "not for commit", so the path is cited rather than the file copied). Driver: tools/fa-sweep.sh. -->

# FA-race sweep against the pin (`~/wt/315fp-build`, e44c9d01a)

`fa-race <T> <cells> <visible> <reps>` repeats one prefill `FLASH_ATTN_EXT` on HRX0 with identical inputs and
counts runs whose output differs from run 0, grouped by `(query tile, head, 16-channel block)`.

```
T        cells    visible  reps   result
1        2048     810      250    checksum=-7.353318 bad_runs=0
2        2048     810      250    checksum=-7.709488 bad_runs=0
8        2048     810      250    checksum=35.336398 bad_runs=0
16       2048     810      250    checksum=41.561140 bad_runs=0
190      2048     810      250    checksum=146.350555 bad_runs=0
1        4864     4798     250    checksum=-2.715973 bad_runs=0
16       4864     4798     250    checksum=14.680580 bad_runs=0
190      4864     4798     250    checksum=1.190266 bad_runs=0
192      4864     4798     250    checksum=-5.256695 bad_runs=0
SWEEP-DONE
```

2250 reps, nine shapes, zero bad runs. `T=190 cells=4864 visible=4798` reproduces the October checksum
(`1.190266`) exactly, so the same path is exercised - and where the October logs recorded `bad_runs` 2-11 per
250 on the same shape, the pin records 0.

For contrast, the October logs: `fa-race1.log` `bad_runs=5` at that shape with lines like
`run 14: 256 differ (112 nonfinite): [qtile 0 head 9 ch 96..111: 256]`; `h336.log` an A/B of
`bad_runs=0 0 0 0` (h336 libraries) against `4 11 4 2` (control) at the same checksum; `e2e.log` 24/24 clean
model-level runs. Cached-kernel hashes in `bis1-4.log` (`md5sum jit190*/*.bin`) plus the instruction profiles
from `tools/fa-variant-profile.py` show the `b*/c*/e*/s*` tags are byte-identical kernels, so those logs'
differing `bad_runs` are not a code bisection.
