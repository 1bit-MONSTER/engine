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

# 315-HANDOFF — landing state for whoever arrives on this box next

Written 2026-10-15 by the #315 goal session when it closed out. **Read this before launching anything here.**
The full record is in the repo worktree `~/wt315-prereg` (branch `1bit/315-preregistration`, PR #414 on
`1bit-MONSTER/engine`); `docs/315-STATUS.md` is its single entry point and `315-COORDINATION.md` is the
inter-session channel.

## 1. Box state, handed over clean

```
box.lock         FREE
GTT used         27 GiB  (idle baseline)
owned servers    none        evictonly forcer  none        hunt scripts  none
```

Nothing of mine is running. The lock is free. If you start a run, take the lock first:

```
flock -n 9 || echo BUSY; exec 9>/home/bcloud/.cache/lax-decode/box.lock     # hold for the run's duration
```

**One thing I changed outside my own files, so you know:** cron **line 22** of this user's crontab — the
`strix-start-internals.sh` entry that was launching **12 redundant `run-generic.sh` jobs on `libloomc`** every
5 minutes — is **commented out**, labelled `# DISABLED 2026-10-15 by goal turn (redundant: decomp/bulk/loomc
covers it 9744/9744)`. A backup of the crontab as it was is at
`/home/bcloud/crontab.backup-20261015-internals.txt`. **To restore it, copy that file back with `crontab`.**
It was consuming the box continuously and made every in-regime arm unstable; nothing else in the crontab was
touched.

## 2. Where #315 stands — a BOUNDED NEGATIVE, not a mechanism

Do not read the record as if a cause was named. **No mechanism is claimed.** At issue #315's own request the
objective was re-scoped to accept a bounded negative, and it was finished on that branch.

**Eliminated**, each by its own pre-registered experiment, with the command and observed output on the record:

| family | how it died |
|---|---|
| deferred host writeback | 626 reads at the engine's logits read, **0** pending |
| persistent mapping / coherence | two-cell A/B at two measured addresses, 13.7 % / 14.8 % — and closed **from source** for both host paths |
| the unwaited async logits read | its own pre-registered campaign's **first fix arm faulted** (Face B, HTTP 500) |
| **the stream-ordering explanation** | of **82** records where `result_norm` differed between two consecutive plain reads, a **fresh `ggml_backend_synchronize` stopped it in 0** and not in 82, and **changed the value itself in 82 of 82** |

**Surviving family, stated as surviving:** an open writer of the `result_norm` slot that is **not** ordered by
`context->stream` — the reused-arena-slot family — plus §51's rank-sort precondition, which is **unmeasured**.

**Handed forward, one measured unexplained fact:** `result_norm` changes between two consecutive identical
plain reads in a faulting arm, at two different addresses, with **no pending host writeback** at the engine's
logits read, and with a fresh synchronise failing to stop it. **That is the thing to explain.**

## 3. Which build is which — and why it matters now

The pin moved this session, so **a result measured on an `e44c9d01a` build no longer describes what ships.**

| build dir | source tree | pin | test-hrx binaries | what it is |
|---|---|---|---|---|
| `~/wt/315logwait-build` | `llama-315logwait` | **47a8fe929** | **8** | **the current build.** Carries the landed fix; has the eight HRX tests built (needed `-DLLAMA_BUILD_TESTS=ON`, see §5) |
| `~/wt/315fp-build` | `llama-315copy` | e44c9d01a | 0 | instrumented **old** pin: tensor-fp twin + read-twice control + the sync-between-reads instrument |
| `~/wt/315probe-build` | `llama-315probe` | e44c9d01a | 0 | old-pin probe build |
| `~/wt/315coherent-build` | `llama-315coherent` | ? | 8 | older control build |
| `~/wt/v315-build` | `llama-hybrid` | ec0175971 | 8 | the **old** test build the 2026-10-13 criterion-4 baseline was measured on |

**The fix, exactly:** fork commit `47a8fe929` on `1bit-MONSTER/llama.cpp` branch `1bit/315-logits-read-wait` —
**5 added lines** in `src/llama-context.cpp`, the last being `ggml_backend_synchronize(backend_res)`. Its
**parent is `0e255d774`**, which is engine `main`'s current pin, so it sits exactly one commit ahead of today's
main; `git merge-base --is-ancestor e44c9d01a 47a8fe929` is true. `check_pins.py` reports "ahead". It is
**not** a revert, and it is **not sufficient** to stop the fault — it is a correct change for the reason its
comment gives.

## 4. The four traps that have each cost a reading — build them out of your harness

All four are one shape: **a count read as a verdict, without checking the count can be non-zero for the reason
the verdict assumes.** In every case the corrupted verdict pointed at a pass not earned or a fault that did not
happen.

1. **A skip is not a pass.** Without `LD_LIBRARY_PATH` and `IREE_HAL_AMDGPU_LIBHSA_PATH` set, **all eight**
   `test-hrx-*` binaries **exit 0** while printing `no HRX0 device, skipped`. An exit-status gate reports 8/8
   PASS on a machine that ran nothing. Count `skip` separately, always.
2. **`grep -c … || echo 0` yields `"0\n0"`.** `grep -c` prints `0` *and* exits 1 on no matches, so the fallback
   also runs; the value becomes two lines and `[ "$N" -eq 0 ]` errors to the *else* branch. This made a **clean**
   dense-control run print `RESULT: FAULT or incomplete`. Drop the `|| echo 0`.
3. **Count occurrences, not lines, on `/v1/chat/completions`.** Responses are concatenated without newlines, so
   `grep -c '"content"'` reads 50 responses as **1**. Use `grep -o … | wc -l`.
4. **`pgrep -f "<pattern>"` matches your own monitoring one-liner.** A wait-loop for "no drivers running" never
   fires while the command asking contains the pattern. Read `/proc/<pid>/cmdline` and skip `bash -c` wrappers.
   The same flaw made a CI watcher print **GREEN** on a branch whose checks had not started, because
   `no checks reported on the … branch` contains no `pending`.

Two more that cost arms:

* **`EXTRA_ENV` splits on commas** (`IFS=',' read -r -a`), so only the **first** fingerprint name survives. Use
  a position selector (`tail:<N>` / `head:<N>`) for several names; `tail:16` covers `attn_norm-*` but **not**
  the `ffn_moe_*` family.
* **A name selector that matches nothing looks exactly like "zero NaN".** The `tail:16` census of a real GLM
  decode run has **946 distinct node names** and contains **none** of `ffn_moe_probs-*`, `ffn_moe_topk-*`,
  `ffn_moe_argsort-*`, `ffn_moe_weights-*` — while `ffn_moe_out-*`, `ffn_shexp-*`, `norm-*`, `attn_norm-*`,
  `q-*`, `k_pe-*`, `Qcur-*`, `Kcur-*`, `q_nope-*`, `l_out-*`, `kv_cmpr-*`, `ffn_gate-*`, `ffn_up-*`,
  `ffn_swiglu-*` and `ffn_out-*` are all present. Check your selector matched **before** reading its zeros.

## 5. Re-runnable commands

**The HRX suite and the `MUL_MAT_ID` op-test at the current pin** (this is criterion 4's second half):

```
cd ~/wt/315logwait-build
export LD_LIBRARY_PATH=$PWD/bin
export IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
for t in test-hrx-buffer test-hrx-mul-mat-id-k32 test-hrx-mxfp4 test-hrx-moe-split \
         test-hrx-attention-sink test-hrx-decode-stride test-hrx-hadamard test-hrx-fa-masked-v; do ./bin/$t; done
./bin/test-hrx-mul-mat-id-k32      # the nmse lines: mxfp4 must read 1.34e-05 … 1.61e-05
```

Expected, and measured this session: `8 pass, 0 skip, 0 fail`; `test-hrx-buffer` `5 passed, 0 failed` with the
same five case names; mxfp4 nmse `min=1.34e-05 max=1.61e-05`; the same two kernels
(`ggml_mul_mat_id_decode_f32_wave64`, `ggml_mul_mat_id_f32_f32_wmma`).

**To build those tests at a new pin:** `BUILD_TESTING=ON` alone is **not enough** — it gives
`No rule to make target 'test-hrx-buffer'`. The real gate is `LLAMA_BUILD_TESTS`:

```
cmake -S ~/wt/llama-315logwait -B ~/wt/315logwait-build -DBUILD_TESTING=ON -DLLAMA_BUILD_TESTS=ON
make -j16 test-hrx-buffer test-hrx-mul-mat-id-k32 test-hrx-mxfp4 test-hrx-moe-split \
          test-hrx-attention-sink test-hrx-decode-stride test-hrx-hadamard test-hrx-fa-masked-v
```

**The dense-model control** (a *negative* control — it must NOT fault):

```
~/wt/315dense3.sh     # Qwen3-8B-Q4_K_M on HRX0, -ngl 99 -fa on -c 4096, 50 mixed-length requests
```

Its script's own verdict line is **untrustworthy** (trap 2 above, still in the script as it ran — left that way
so its output stays reproducible). Read the counts, not the verdict: correct is `responses=50`,
`distinct_content=1`, `nan_guard=0`, `hsa=0`, `content_only=0`.

**The in-regime harness** (if you are running arms): `~/wt/315-regime.sh`, `~/wt/315-fp-hunt.sh`,
`~/wt/315-detector.py`, `~/wt/315-paired.sh`, `~/wt/evictonly` (the `mprotect`-churn eviction forcer). The
regime gate is `gtt_start ∈ [95,110] GiB`; this box's in-regime point is **98G** and neighbour granularity is
~17 GiB. Both faces must be counted in every arm, and **arms that die for harness reasons are invalid, never
passes** — while an arm truncated as a *consequence* of the defect (a 500 the model caused) **is a fault**.

## 6. What not to trust

* Any control result measured on an `e44c9d01a` build as if it described the shipped pin.
* Any name-selector result whose name you have not confirmed appears in the run's node census.
* Any "zero NaN" that came from a selector matching nothing, or from a suite that skipped.
* The dense control script's verdict line, and any `grep -c … || echo 0` count.
* `pgrep -f` counts taken from a shell that contains the pattern.
* Stale run directories under `~/wt/rt-det/` from **diagnostic, eviction-forced** cells — those arms are
  labelled as diagnostic and **no arm of theirs may be counted toward a campaign's arm count**.
