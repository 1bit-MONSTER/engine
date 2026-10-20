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

## Appendix: a harness observation found while the row probe ran (2026-10-10 11:05)

At `gtt_start = 98 GiB` for every arm, the row probe's pair-4 arm was running while **pair 1's arm server was
still alive**: `315row-arm-1l.scope` active, `llama-server … --port 19486`, up 15 minutes, after that arm had
already been scored `OK` in the console. Card GTT read **113.5 GiB** against the fleet baseline + composition of
~80 GiB - i.e. one leaked 17 GiB model server on top. Stopping the scope released 13 GiB (`113G -> 100G`) and left
only the three neighbours and the running arm.

```sh
ssh strixhalo 'systemctl --user list-units --state=running "315row*" --no-legend'
#  315row-arm-1l.scope   [systemd-run] … --port 19486 …      <- scored OK minutes earlier, still resident
#  315row-arm-4l.scope   [systemd-run] … --port 19554 …      <- the arm actually running
ssh strixhalo 'systemctl --user stop 315row-arm-1l.scope'; echo $(( $(cat /sys/class/drm/card*/device/mem_info_gtt_used)/1073741824 ))G
# 100G
```

The teardown in `315-paired.sh` is `systemctl --user stop $TAG-arm-$pair$cell` followed by a loop of ~15 s waiting
for the port to disappear; a *clean* arm's server can take longer than that to retire, and the runner then proceeds
to the next arm with the previous one's ~17 GiB still resident. Two consequences worth carrying forward, neither of
which invalidates the arms already recorded (each is labelled by its own `gtt_start`, and the gate is that label):

* later arms in a long session start from a higher baseline than the composition intended, which is how a session
  can drift out of the `[95, 110]` gate or into the detector's allocation-failure `INVALID` class even though its
  composition was correct;
* on a box with 120 GiB of GTT and a co-tenant, a 113 GiB reading is one leak away from the OOM that killed a
  co-tenant service earlier the same day (see the row-probe notes), so a leaked arm is worth stopping rather than
  tolerating - but only *between* arms, never during one.

This is the same family as the "silent arm loss" trap already recorded in `KNOWN-ISSUES.md` (an abnormal arm death
leaving an orphan holding the flock), with a second trigger: a *normal* arm whose teardown outlives the 15 s window.

## Appendix 2: `HRX_COMMAND_BUFFER_MODE=pm4` is unusable on this workload (2026-10-10)

The other lever on the packet mechanism is the command-buffer mode: `aql` (the default) versus `pm4`
(`logical_device.c:244-253`, and PM4/AUTO additionally provision upload queue-control storage for *reusable*
dispatch command buffers with dynamic binding-table fixups). The paired A/B was run - control = the default AQL
path, fix cell = the same binary behind a launcher exporting `HRX_COMMAND_BUFFER_MODE=pm4`, both cells carrying
the same inducer, arms alternating, `REGIME_LO=80`:

```text
pair 1 arm c: OK   nan=0 req=66 gtt_start=98G      pair 1 arm l: FAULT nan=0 req=1 gtt_start=97G
pair 2 arm c: OK   nan=0 req=66 gtt_start=98G      pair 2 arm l: FAULT nan=0 req=1 gtt_start=97G
pair 3 arm c: FAULT nan=1 req=19 gtt_start=98G     pair 3 arm l: FAULT nan=0 req=1 gtt_start=97G
pair 4 arm c: OK   nan=0 req=66 gtt_start=98G      pair 4 arm l: FAULT nan=0 req=1 gtt_start=97G
control: OK=3 FAULT=1 OOR=0        fix(pm4): OK=0 FAULT=4 OOR=0
```

**Every PM4 arm stopped at `req=1` with `nan=0`**, and the detector's own record for one of them is
`{"requests": 1, "ok": 0, "http_failures": 1, "first_failure": "HTTP 500"}` - i.e. the *first* request failed.
That is not a fault-rate comparison: the PM4 cell never reached the workload, so the four "FAULT" verdicts are
the harness's short-request rule, not Face A or Face B. The honest reading is **inconclusive for the mechanism**
and a separate diagnostic finding: **`HRX_COMMAND_BUFFER_MODE=pm4` fails the first request with HTTP 500 on this
stack, at the pin**, which is worth its own report and is out of #315's scope.

The consequence for this issue is that the packet-level lever has to be the barrier bit itself (see
`315-forcebarrier.sh`), not the command-buffer mode.
