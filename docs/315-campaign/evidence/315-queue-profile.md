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

     engine#315 evidence: the engine's own queue/dispatch profile of GLM's decode, which was already on the box
     (2026-10-05) and is quoted here because it constrains two live leads. Not a new run: every number below is
     a verbatim line from ~/wt/hrxsys-prof-run.log, and the script that produced it is quoted so it can be
     reproduced. -->

# The queue profile of GLM's decode: one block, 92 unbarriered dispatches of 612

## The artefact, verbatim

```text
def-GLM-4.7-Flash-Q4_K_M:   cb 3 execs 23 dispatches 612 barriers 519 no-barrier-dispatches 92 blocks 1 | span 15348 us busy 14288 us idle 1060 us; barrier gaps 1062 us over 519 (median 2.04 us); block-boundary gaps 0 us over 0; between token programs 571 us, token period 15942 us
b256k-GLM-4.7-Flash-Q4_K_M: cb 3 execs 23 dispatches 612 barriers 519 no-barrier-dispatches 92 blocks 1 | span 15337 us busy 14274 us idle 1063 us; barrier gaps 1061 us over 519 (median 2.04 us); block-boundary gaps 0 us over 0; between token programs 567 us, token period 15899 us
b512k-GLM-4.7-Flash-Q4_K_M: cb 3 execs 23 dispatches 612 barriers 519 no-barrier-dispatches 92 blocks 1 | span 15348 us busy 14290 us idle 1058 us; barrier gaps 1059 us over 519 (median 2.00 us); block-boundary gaps 0 us over 0; between token programs 571 us, token period 15915 us
```

Produced by `~/wt/hrxsys-prof-run.sh` (2026-10-05), which for each configuration runs `~/wt/hrxsys-prof.sh`:
`llama-server -m GLM-4.7-Flash-Q4_K_M.gguf -dev HRX0 -ngl 99 -fa on -c 2048 -np 1`, one completion of
`n_predict: 24`, `temperature: 0`, under `HRX_PROFILE_MODE=dispatch HRX_PROFILE_FILE=<dir>/run.prof`, analysed by
`~/wt/hrx-disasm-test2/.../iree-profile`. The whole thing holds the box lock (`exec 9>…/box.lock; flock 9`).

```sh
ssh strixhalo 'head -3 ~/wt/hrxsys-prof-run.log'
ssh strixhalo 'bash ~/wt/hrxsys-prof-run.sh'      # ~4 min per configuration, holds the box lock
```

## What it constrains

**1. GLM's decode program is ONE block, not two.** `blocks 1` at the default, at
`HRX_AQL_BLOCK_SIZE=262144` and at `524288`, with `block-boundary gaps 0 us over 0`. The lead that has been
carried as live - "GLM is the only model whose decode splits into two 128 KB AQL command blocks (919 + 760
dispatches) with 1,399 barriers per token" - is **not reproduced** in this decode: 612 dispatches, 519 barriers,
one block, zero block-boundary gaps. Either the two-block figure comes from a different workload (it is quoted
for the detector's mixed-request arms, while this is one 24-token greedy completion), or it is wrong. Until the
same profile is taken of a detector arm, the block-boundary candidate has no demonstrated reachability for this
model, and `HRX_AQL_BLOCK_SIZE` is not a lever that reaches it: the knob changed nothing in the profile *and*
nothing in throughput (`~/wt/hrxsys-split.log`: pp/tg ratios 0.9965-1.0073 across the same three settings).

**2. 92 of 612 dispatches carry no barrier - 15 % of the decode program.** Those are the dispatches that may
overlap a neighbour, and they are the reason the read-path candidate is testable at all: with
`busy 14288 us of a 15348 us span` (idle 1060 us) the queue is essentially back-to-back, so a consumer that is
not ordered after its producer has a real window. A host read of a host-visible mapping that is not itself
ordered after those writes reads memory the GPU may still be writing, or has not written at all.

**3. The analyser is half-broken and the record should say so.** `rt-gaps.py` crashed with
`ValueError: max() iterable argument is empty` (`~/wt/rt-gaps.py:72`) and the `auto` configuration's line never
reached the log, so the file has three of the four configurations it intended. The three that are there are
complete lines, and the numbers agree across configurations.

## Relation to the two candidate families

* The **AQL block-split** lead loses its reachability premise (point 1); mechanism-doc section 32 had already
  retracted the asymmetry it was built on, and this is the second, independent reason to stop treating it as the
  leading candidate.
* The **read-path / ordering** family gains its quantitative setting (point 2): the window exists because 15 % of
  the program is unbarriered and the queue is saturated, not because of a block transition. This is the family
  the queued paired `GRAPH_SYNC` experiment tests (`315gsync2`).
