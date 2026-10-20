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
<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Why no forward-only fix is available: the engine has no way to detect or repair a stale restore

Recorded as the answer to criterion 2, after six cause families were closed and the trigger was
characterised (ambient ~5% vs forced eviction ~95%). This is not a restatement of "the cause is unknown";
it is a specific structural argument about what a fix would have to do.

## The trigger's physical shape

KFD **evicts and restores the arm's own queue buffers while the arm is live** — measured per-arm
(`Q_EVICT`/`Q_RESTORE` attributed to the arm's own pid, `315-eviction-hits-the-arm.md`) — and a consumer
then reads bytes that differ from what the producer wrote. The forcer (`~/wt/evictonly`) provokes this by
`mprotect` churn on a `hipHostRegister`'d range; the operands are the arm's own buffers, not the forcer's.

## What the engine actually validates

Searched the pin for any content validation on the path from restore to consumption:

    command-program-executor.cpp  `validate_program_constant_access` (:705) and
                                  `verify_command_program` (:1585)
                                  -> both validate the PROGRAM's structure/legality, never memory contents
    common/sampling.cpp           `std::isnan(logits[i])` (:598, :615)
                                  -> the only CONTENT check anywhere, and it is Face-A only

So on the whole replay → restore → consume path there is:

- **no post-sync revalidation** of restored memory;
- **no checksum, generation counter, or epoch** that would let a consumer know a buffer was evicted and
  restored (a `generation` field exists on allocations, but it tracks allocation identity, not restore
  fidelity, and nothing compares it across a sync);
- **no re-read** after the sync that the eviction window would have disturbed.

## Why that forecloses a forward-only engine change

For the engine to *repair* this, it would have to distinguish "the bytes I read are what the device wrote"
from "the bytes I read are a pre-eviction stale copy". Nothing available to the engine carries that
information:

- the failure is **silent** — Face B produces well-formed bytes that merely differ, and Face A's NaN is
  caught only by the existing guard, which aborts rather than repairs;
- the window is on the **device/runtime side** (queue eviction and restore), which the goal's boundaries
  place out of scope ("AMD driver/kernel changes … record as a concrete proposal with evidence … but do not
  attempt them");
- re-reading or re-running after every sync would be a **performance-destroying** change, and the record
  already shows the sync-everywhere lever is null (`GGML_HRX_DIAGNOSTIC_GRAPH_SYNC`, 0/4 vs 1/4, Fisher
  p = 1.00), so it would not even work.

## What a fix would have to be, stated so it can be judged

1. **Out of scope**: the runtime guarantees restore fidelity, or KFD's eviction/restore is made transparent
   to in-flight consumers. That is the driver/runtime lane the boundaries exclude — and the record's own
   prior runtime proposal is the right place for it.
2. **In scope but a workaround, not a repair**: the NaN-scrub already present (`GGML_HRX_NAN_CONTINUE`,
   NaN → `-INFINITY`). It addresses Face A only, masks rather than repairs, and the issue's own rule forbids
   recording it as a limitation until the mechanism is named.
3. **Neither**: a forward-only change that "fixes" this by altering ordering or coherence has been tried
   four times over (coherence attribute, GRAPH_SYNC, forced AQL barriers, graph reorder) and every one is
   falsified on the record.

So criterion 2 is **not achievable within this goal's scope on present evidence**, and the honest
disposition is that the issue closes with the trigger named and replicated, six cause families closed, and
the fix question assigned to the out-of-scope runtime lane. That is a conclusion about the *fix*, reached
after the cause search, not a substitute for it.

---

# Face B already has a detector — which closes the last "detect and recover" avenue

The one remaining shape a forward-only change could have taken was *detect the corruption and refuse or
recover*, on the theory that the engine was serving bad data silently. Read at the pin, that avenue is
already closed for both faces.

## Face A — detection exists

`common/sampling.cpp:598-620`: the sampler scans the logits, and on NaN logs
`HRX returned NaN logits: count=… of …, contiguous=…, continuing=no` and **aborts** unless
`GGML_HRX_NAN_CONTINUE` is set. So Face A is detected and fails loudly, by construction.

## Face B — detection also exists, at the parse site

`common/chat.cpp:3290-3291`:

    LOG_WRN("%s: unparsed %s output: %s\n", __func__, common_chat_format_name(params.format), ...);
    LOG_DBG("%s: full %s output triggering error:\n=== BEGIN ===\n%s\n=== END ===\n", ...);
    throw std::runtime_error(std::string("The model produced output that does not match the expected ")
                             + common_chat_format_name(params.format) + " format");

So the engine **already logs the offending text** and **throws** — the HTTP 500 the issue records *is* the
detector firing. There is no silent-corruption gap to close: the failure is not being swallowed.

## Why that forecloses the detect-and-recover avenue

A forward-only change of the form "notice the corruption and do something better" has nothing to add:

- Face A already aborts (and the scrub option already exists, and is a workaround);
- Face B already logs and throws, with the full text in the debug log;
- neither detector can **distinguish** "the model genuinely emitted malformed text" from "the model emitted
  correct text into a buffer that a restore left stale". The information is not present at either site —
  which is the same conclusion as the section above, reached from the detection end rather than the data
  end.

So the two arguments agree: the engine can see *that* something is wrong (both faces) but cannot see
*why*, and the "why" is a device/runtime restore-fidelity question that the goal's boundaries place out of
scope.

## Net

Criterion 2 has no achievable forward-only form within scope:

    repair the stale bytes            -> needs runtime restore fidelity   (out of scope)
    detect and refuse                 -> ALREADY DONE for both faces      (nothing to add)
    re-read / re-run after sync       -> performance-destroying, and null on the record (GRAPH_SYNC p=1.00)
    reorder / re-fence the read       -> tried four times, all falsified
