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
<!-- engine#315: what the 142 recorded arms say about the two faces, read after the fact from the arm
     logs on strixhalo. This is the landscape any mechanism has to fit, and it is where the structural
     constraints below come from. Parsed with /tmp/faceb2.py and /tmp/faceb3.py over
     ~/wt/rt-det/*/{detector.json,server.log,run.log}. -->

# The two faces across the 142 recorded arms

## Method

For every `~/wt/rt-det/*/` on strixhalo that has a `detector.json`:

- `nan` = count of the guard's `HRX returned NaN logits` lines (Face A)
- `fb` = count of `Content-only` 500 lines (Face B)
- `idn` = `identical_distinct` = `len(set(outputs))` over 8 identical greedy requests sent with
  `cache_prompt=false` and `temperature=0`, so **1 = deterministic** and >1 = nondeterminism
- `gtt` = the arm's `gtt_start` from `run.log`, banded at the pre-registered gate

142 of the 509 arm directories have a `detector.json` (the rest predate it).

## The two faces never co-occur

| class | arms |
|---|---|
| Face A (NaN logits) | 23 |
| Face B (format 500) | 7 |
| **both** | **0** |
| fully clean (>=66 requests, every signal zero, `idn <= 1`) | 108 |

No arm in the record has produced both faces. That is the strongest structural constraint available on any
mechanism that claims to unify them: at the observable level the two are mutually exclusive in 142 draws.

## Face A travels with finite nondeterminism; Face B does not

`identical_distinct` distribution: **1 in 135 arms, 2 in 3, 3 in 2, 4 in 2**. Every divergent arm (7/7) is a
Face-A arm, and every Face-A arm with `idn > 1` also has `nan=1`. So inside one arm the same 8 identical
greedy requests both produced NaN logits in some runs and produced *different* finite outputs in others:
the event is nondeterministic corruption, not a fixed wrong answer. No Face-B arm is divergent (0/7) - and
their `idn=1` is measured over the identical requests that completed before they aborted (`req` 13-47 means
6-7 of the 8 identical requests had run), so Face B's generation was *consistent* up to the failure.

## Face B shows no dependence on the gate

| gtt band | all arms | Face A | Face B | clean >=66 |
|---|---|---|---|---|
| 95-110 | 113 | 22 | 7 | 84 |
| <95 | 11 | 1 | 0 | 8 |
| >110 | 18 | 0 | 0 | 16 |

Face B: 7/113 in-band versus 0/29 out-of-band (Fisher p ~ 0.3) - no evidence it tracks the gate. Face A:
22/113 in-band versus 1/29 out, and that single out-of-band event is not usable:

**Caveat on `315nan-p20l` (gtt=44).** It faulted *after* I SIGKILLed the arm server of pair 19 in the same
session. A server dying mid-run can leave the following arm's HRX state inconsistent, so this event is
recorded as ambiguous rather than as evidence that Face A occurs outside the regime.

## Face B is the server's own chat-template parse

"Content-only" is the *parser name*, not a description of the model: `common/chat.cpp:812` returns that
string, and `tests/test-chat-peg-parser.cpp:180` has a "Content-only parser" case. So the 500 means the text
the model generated failed the parse the engine configured - a *text*-level failure. The 7 Face-B arms have
zero guard lines, but **none of them had the logits fingerprint enabled**, so "their logits were NaN-free" is
not established by these logs; that is what `315-faceb.sh` is for.

## The run that produced most of these numbers

`315nan` completed its 20-pair cap after the 8400 s wrapper timeout killed only the wrapper: final line
`1 clean fix-cell arms, 17 faulted-or-invalid in the fix cell, out of 21 arms run`. Section 41 cleared its
instruments of fabricating NaN by source inspection, so those are computed-NaN faults, but their *rate*
(16/17) remains instrument-inflated against the recorded 5.4%.

## Tooling failure to record

The attribution session (`bd4c969ee`) **took the box lock at 09:37:25 and then died** with
`315-attrib.sh: line 70: unexpected EOF while looking for matching '` - because an edited `315-attrib.sh` was
scp'd over the file that session was executing, and bash re-read it at stale byte offsets. No attribution
data exists. The failure was mine, not the box's, and the same trap must not be repeated while
`315-chain.sh` runs: new probes go in new filenames.

## What actually precedes the 500: the parser rejecting the model's own degenerate output

The line immediately before every Face-B 500 is the parser, and what it is rejecting is the model's own text:

```
1.47.161.803 W common_chat_peg_parse: unparsed Content-only output:  a..........* it on ${... and: it itns it that'~><es x case a a~ it
1.27.713.034 W common_chat_peg_parse: unparsed Content-only output:  a. ? $\ (/C ( ( ( / ( ( (?) ( ( a a *_x
1.08.109.609 W common_chat_peg_parse: unparsed Content-only output:  .,。_./  ?, exactly,。，*。 **) 一*-S、 " "*--p— '--    -*( 
3.29.726.901 W common_chat_peg_parse: unparsed Content-only output:  a of in一个文件 a某( as<<* ? ( probably** ? of ? AND < as   <
1.13.054.990 W common_chat_peg_parse: unparsed Content-only output:  We A ( the).
```

Two flavours, both degenerate: symbol/script soup that makes `grep` treat the log as a binary file (invalid bytes), and a 12-character
fragment. Five of the seven Face-B arms show one of these. `n_tokens = 120` and `truncated = 0` on the same line as the
`p1c` case, and that arm's offender is inside the identical block (request 13 of 16), so the *same* prompt produced
`identical_distinct=1` over the requests that succeeded and then this fragment on the seventh. The offender's output
therefore differed from its siblings - which confirms the event is anomalous, not a property of the prompt, and rules out
"the model legitimately failed a strict format on a hard prompt" as the explanation for that arm.

**What this makes Face B.** The guard that produces Face A is unconditional and silent when `nan_count == 0`, and no
Face-B arm logged a guard line - so those requests' logits were NaN-free. Degenerate sampling with NaN-free logits is what a
*finite* corruption of the logits or of the hidden state feeding them looks like: symbol soup, or a premature stop. So both
faces are the same class of event, separated by what the corrupted memory happens to contain: NaNs (Face A, caught by the
guard) or stale finite values (Face B, which reaches the sampler and fails later, at the parser).

**What is not yet measured.** No Face-B arm had `GGML_HRX_LOGITS_FP` enabled, so the logits of an offending request were
never fingerprinted. `315-faceb.sh` fixes that, and it can test the strong form of this claim directly: with the fingerprint
on, the eight identical greedy requests must log the *same* logits hash, so a corrupted sample shows up as one hash differing
from its seven siblings. That is the discriminating measurement for the Face-B half, and it is the reason the probe sets
`GGML_HRX_LOGITS_FP=1` as well as `GGML_HRX_NAN_CONTINUE=1`.

## Face B measured directly: a faulting request with no NaN anywhere in the pipeline

The chain-probe session (2026-10-10, tag `315chain`, in-regime at `gtt_start=98G`) faulted on its very first pair
in both cells, and unlike every earlier Face-B arm this one ran with a 60-name tensor fingerprint that includes
`norm`, `result_norm` and `result_output`:

| arm | detector.json | parser line | fingerprint |
|---|---|---|---|
| `315chain-p1l` | `requests 64, ok 63, http_failures 1, utf8 0, parse 0, identical_distinct 1, first_failure "HTTP 500"` | `unparsed Content-only output:  the}:}:}甚;` | **0 lines** |
| `315chain-p1c` | `requests 24, ok 23, http_failures 1, ...` | `unparsed Content-only output:  a gaps gaps gaps gaps gaps gaps gaps gaps gaps gaps gaps.` | **0 lines** |

Zero fingerprint lines with `GGML_HRX_TENSOR_FP_NANONLY=1` and a list spanning the residual stream, both attention
branches, the whole FFN/MoE chain and the output norm means **there was no NaN in any named node of the faulting
request, including the logits**. That is the measurement section 44 listed as missing: Face B is a *finite*
corruption, and the two text flavours show it plainly - one garbled (`the}:}:}甚;`), one a repetition loop
(`gaps gaps gaps ...`), which is what near-degenerate logits produce under greedy sampling.

Consequence for technique: the NaN-only fingerprint is blind to this half. Localising *finite* corruption needs
hashes, not NaN counts - `GGML_HRX_LOGITS_FP` prints a hash per sample, and for the 8 identical greedy requests the
hashes must repeat, so a corrupted sample shows as one hash differing from its seven siblings.

## The coherence candidate's own campaign, re-read

`315accept-console.log` (2026-10-09) is the session that tested the *patch* rather than the knob:
`fix_bin = ~/wt/315fix-build/bin` (unconditional `HRX_MEMORY_TYPE_HOST_COHERENT`), `control_bin = ~/wt/315instr3-build/bin`.
The driver records the allocation type per arm, and the cells separate cleanly - fix arms show `hv52=0 hv56=1`
(host-visible *coherent*), control arms `hv52=1 hv56=0` - so the patch was demonstrably live in the cell being
scored. Result: **9 clean and 1 fault (Face-B class: `nan=0, req=33`) out of 10 fix-cell arms**, against the recorded
2/37 for the shipped build. Making the allocation coherent did not remove the fault.

So the *coherence bit* is not the mechanism. What that leaves inside the same family is the timing of the sync that a
non-coherent host read depends on, which is what `315-graphsync.sh` tests: `GGML_HRX_DIAGNOSTIC_GRAPH_SYNC` calls
`hrx_stream_synchronize` after every graph replay (`command-program-executor.cpp:2034`), so every later read -
including the host's read of the logits - must follow the device's writes.

## A gap in the record: no stock baseline has ever been measured

Every session's `control_bin` on disk is `~/wt/315instr3-build`, whose tree carries **233 insertions** of its own
(`command-program-executor.{cpp,h}` and `host-memory.cpp`) - i.e. the control cell has always been a *candidate*, not
a baseline. `~/wt/315fix-build` is the same family with the coherence gate removed (240 insertions). A true
stock-pin control does not exist on the box; building one from a clean worktree at the pin is a prerequisite for the
task-5 control cell and for any "unchanged relative to baseline" claim in task-6.
