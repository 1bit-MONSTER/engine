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

     engine#315 evidence: the engine's own CI steps, run locally at the current pin on 2026-10-09 so
     task-6 has a baseline and so the gates that will judge the eventual fix are known to work in this
     environment. Stored as Markdown because the repository's CI requires a copyright notice on every
     file. -->

# engine CI baseline at pin `e44c9d01a`

Run on `ryzen` in the worktree `~/wt315-prereg` (`1bit/315-preregistration`), 2026-10-09, at
`third_party/llama.cpp = e44c9d01a4d5110cecaf45472a717f46cd1abbf9`. Every step below is a step of
`.github/workflows/ci.yml` or a `ctest` test that `ci.yml`'s final `ctest` runs, executed directly
rather than through the workflow.

| step | command | result |
|---|---|---|
| copyright | `python3 tools/copyright.py --check` | **exit 0** (it failed before `b777bdc`: three `evidence/*.txt` files had no comment style and the copied probe script had no notice) |
| pin gate | `python3 tools/check_pins.py 6d5e470` | **exit 0** - `third_party/llama.cpp: 26330cc20 -> e44c9d01a is ahead` (plus hrx-system, xdna-driver, laya) |
| pin gate, longer span | `python3 tools/check_pins.py beec5e6` | **exit 0** - `third_party/llama.cpp: cf4dfd801 -> e44c9d01a is ahead` |
| gguf -> Q4NX | `python3 tests/gguf_to_q4nx_test.py` | PASS |
| ternary -> Q4_0 | `python3 tests/ternary_to_q4_0_test.py` | `PASS: PTQ1_0 -> Q4_0 exact, other tensors and metadata kept` |
| MTP draft vocab | `python3 tests/mtp_draft_vocab_test.py` | `OK` |
| registry gaps (ctest `registry_gaps`) | `python3 tools/registry_build.py --check-gaps` | exit 0 - `none of the 15 reviewed significant classes is mapped without a reason` |
| registry gaps fire (ctest `registry_gaps_fire`) | `python3 tests/registry_gaps_test.py` | PASS (2 checks) |
| build (`onebit` + the 14 test targets, gcc-15) + `ctest` | not run | nothing to regress yet: there is no fix revision to compare against, and this is the CPU-only half of the workflow |

## What the pin gate does and does not cover

`tools/check_pins.py` compares **submodule gitlinks** between a base commit and `HEAD` and asks GitHub's
compare API whether the new commit is ahead of the old one. It is the machine enforcement of the
objective's "must not be a revert" clause, and it works here (rows 2-3).

**Gap found and CLOSED (2026-10-13).** As originally recorded, `check_pins.py` covered only the
submodule gitlinks, not the hrx revision declared in `registry/architectures.json`
(`"sources": { "llama.cpp (hrx)": "e44c9d01a4d5..." }`). `tests/registry_gaps_test.py` reads that file
but checks class coverage, not the pin, and nothing tied the registry value to the gitlink.

Demonstrated rather than asserted: setting the registry's declared revision to all-zeros passed
`check_pins.py`, `registry_build.py --check-gaps` **and** `tests/registry_gaps_test.py` — every gate
green on an inconsistent tree.

`tools/check_pins.py` now reads the registry and requires it to agree with the gitlink at
`third_party/llama.cpp`, failing with an explicit message when they diverge. Verified both directions:
exit 0 on the consistent tree (`"llama.cpp (hrx)" agrees with the gitlink (e44c9d01a)`) and exit 1 on
the half-move, with the message naming both revisions and pointing at the file that must be moved too.
So a half-move can no longer pass CI, and task-7's "move both by hand" is now enforced by the gate.

## The pin lineage, and why the fix cannot be "move the pin"

`git log -- third_party/llama.cpp`:

| commit | move | title |
|---|---|---|
| `4aa4ac2` | e57beb9721af | HRX bump: HIP plumbing hook, HRX fixes, Flash-Next |
| `11d6e2f` | 522dab4789ce | long-prompt fix; hrx-system moves to our fork (Loom NaN backport) |
| `c18b319` | f2099e9b7 | AMD's core with our ggml-hrx |
| `beec5e6` | f2099e9b7 -> cf4dfd801 | **Avoid subgroup votes in HRX decode-split flash attention** (#342) |
| `6d5e470` | cf4dfd801 -> 26330cc20 | **pin llama.cpp to the shared-memory FA tile visibility (engine#314/315)** (#347) |
| `805d30f` | 26330cc20 -> e44c9d01a | AMD's tested pair |

`6d5e470` is already a #315-motivated pin move, aimed at exactly the flash-attention visibility that
sections 26-27 of `315-mechanism.md` now falsify as the locus. So "bump the pin to the next AMD pair"
is not a fix for this issue - it has been tried and the fault survived it (section 22 finding 2 records
that the coherence and ordering levers failed; section 26 kills the FA masked-row lever outright).
The eventual change has to be a new fork commit that is a **descendant of `e44c9d01a`** and carries
whatever the mechanism work ends up naming.

## Closing the recorded gap: the registry pin and the gitlink agree today

The baseline above recorded a gap - "nothing ties `registry/architectures.json`'s hrx pin to the gitlink" -
which reads as a possible divergence. Checked 2026-10-10:

```
$ git ls-tree HEAD third_party/llama.cpp
160000 commit e44c9d01a4d5110cecaf45472a717f46cd1abbf9	third_party/llama.cpp
$ grep -n 'llama.cpp (hrx)' registry/architectures.json
4:  "llama.cpp (hrx)": "e44c9d01a4d5110cecaf45472a717f46cd1abbf9"
```

They are the same commit. So the gap is one of *enforcement* - no check compares the two, and `check_pins.py`
enforces only the gitlink - not of current divergence. A forward-only fix therefore edits **two** files by hand
(the gitlink and `registry/architectures.json` line 4) and both must stay equal; a CI check that compares them
would be a reasonable addition but is not part of this change. `KNOWN-ISSUES.md` lines 81-84 hold the other
mechanical constraint (why `f2099e9b7` cannot be adopted: it is a strict ancestor of the shipped `26330cc20`,
so pinning it is a revert).

---

# Criterion-4 control re-run: queued (2026-10-13 ~16:07)

The objective's fourth criterion needs the **dense-model control and the MUL_MAT_ID op-test unchanged
relative to baseline**. The baseline for both is the file above (2026-10-09, pin `e44c9d01a`), and the two
things it records that a re-run must match are:

- the 5 HRX buffer tests, all `[PASS]`, and the 9 runnable tests overall;
- the **MUL_MAT_ID `nmse` values**, e.g. `mxfp4 k=2880 tokens=1 rows=1 … nmse=1.56e-05 OK` — a numeric
  fingerprint, not just a pass/fail, so a drift in the reduce or the id resolution would show.

## Status

A re-run is **queued** (`~/wt/315crit4.sh`, tag `315crit4`), waiting for the peer agent's replication to
release the GPU. It re-runs the buffer tests and the op-test from `~/wt/v315-build` (present, with 8
`test-hrx*` binaries) under `systemd-run --user --scope -p MemoryMax=4G`, exactly as the baseline did, and
extracts the `MUL_MAT_ID` lines into `~/wt/315crit4/mul_mat_id.txt` for a direct diff against the table
above.

## Why it is queued rather than run now

The tests need HRX0, and the box is currently held by the peer's alignment replication (with the box lock
taken). Running them concurrently would perturb that measurement and risk its GTT headroom — the same
reasoning that deferred my earlier probes. Nothing about criterion 4 is blocked in principle; it is
sequenced behind a live run.

## What a result would and would not establish

- **Would**: whether the shipped pin's own regression controls are unchanged *today*, which is criterion 4's
  claim, and it would do so with the baseline's own numbers rather than a pass/fail.
- **Would not**: anything about the #315 defect. No fix has landed, so there is nothing for these controls
  to regress *against* — which is why criterion 4 cannot be *met* until criterion 2 produces something to
  compare. This run establishes that the controls are stable on the current pin, i.e. that the baseline the
  future fix must match is still the baseline.

---

# Update: the control re-run now requires a sustained clear window

The first `315crit4.sh` waited on the **first** 30-second poll showing no other driver, which is unsafe here:
another agent launches campaigns back-to-back, so a brief gap between two of its runs would let this control
start and collide with its next launch — corrupting both measurements.

Replaced by `315crit4b.sh`, which requires a **sustained 120-second clear window** (four consecutive 30-second
polls with no other driver) before starting, and still takes the box lock. Launched 16:46:23 and waiting.

Nothing about the criterion-4 content changed — same baseline, same commands, same `MUL_MAT_ID` extraction —
only the gate on when it is safe to begin.

---

# Update: the control's first attempt matched no tests — my binary names were wrong

The first corrected run (17:05) produced a header and then "done" with **no test output at all**. Diagnosed
rather than rerun blindly: my script looked for

    test-hrx-backend-buffer  test-hrx-host-buffer  test-hrx-host-transfer
    test-hrx-host-staging    test-hrx-host-weight-cache

and **none of those binaries exist**. The real contents of `~/wt/v315-build/bin` are:

    test-hrx-attention-sink   test-hrx-buffer           test-hrx-decode-stride    test-hrx-fa-masked-v
    test-hrx-hadamard         test-hrx-moe-split        test-hrx-mul-mat-id-k32   test-hrx-mxfp4

So the baseline's **"5 HRX buffer test cases"** (`backend_buffer`, `host_buffer`, `host_transfer`,
`host_staging`, `host_weight_cache`) are **cases inside `test-hrx-buffer`**, not separate binaries — I had
misread the baseline's output as a list of executables.

Note also that the objective's named control, the **MUL_MAT_ID op-test**, is present as
`test-hrx-mul-mat-id-k32`, and `test-hrx-mxfp4` carries the `MUL_MAT_ID mxfp4 … nmse=` lines the baseline
table records.

Corrected as `315crit4c.sh`, which runs the real binaries and greps for the `MUL_MAT_ID`/`mxfp4` lines to
diff against the baseline table.

This is the fourth harness/parse defect of the session (a line-based counter on newline-free NDJSON; a
`pgrep` pattern matching its own caller; `md5sum` over streaming responses; now guessed binary names). Each
was caught before its output was cited, but the pattern is consistent enough to state plainly: **read the
baseline and list the directory, rather than reconstructing either from memory.**
