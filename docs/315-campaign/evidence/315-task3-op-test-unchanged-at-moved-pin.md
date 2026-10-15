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

# task 3, op-test half: the HRX suite and MUL_MAT_ID are UNCHANGED at the moved pin `47a8fe929`

The record's criterion-4 controls were measured at the **old** pin `e44c9d01a`
(`315-crit4-control-result.md`, 2026-10-13). The pin has since moved (`95238dc`, and the registry moved with it
in `88d3b71a5`), so the controls had to be re-measured **at the moved pin** — that is the whole point of a
regression criterion, and re-running them against the old pin would have proved nothing.

## Build, and that it did not disturb the harness

`~/wt/315logwait-build`, source `~/wt/llama-315logwait` at `47a8fe929`, configured with
`-DBUILD_TESTING=ON -DLLAMA_BUILD_TESTS=ON`. `LLAMA_BUILD_TESTS` is the real gate — `BUILD_TESTING=ON` alone
produced `make: *** No rule to make target 'test-hrx-buffer'`, which is recorded because the first two attempts
failed that way.

Because this reconfigures the same build directory the harness uses, the reconfiguration was verified not to
change the harness rather than assumed:

```
BEFORE  3cbb1c74c2877d54...  bin/llama-server
        0e7d1ca116735a60...  bin/libggml-hrx.so.0
        ac303d259bb572d9...  bin/libllama.so.0
gate now: 13 test-hrx targets visible
present=8 of 8
HARNESS UNCHANGED: hashes identical
```

## The trap this run walked into first: a SKIP is not a PASS

Run without the HRX runtime environment, **all eight binaries exit 0** and every one of them is vacuous:

```
test-hrx-mul-mat-id-k32: no HRX0 device, skipped
test-hrx-buffer:         test skipped: no device available for HRX buffer test (5 device case(s))
```

A script keying on exit status would have reported **8/8 PASS** on a machine that ran nothing. That is the same
shape of error as the recorded line-counting defect, and it is why the numbers below come from a run with

    LD_LIBRARY_PATH=<build>/bin
    IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1

set, and why the harness counts `skip` separately from `pass`.

## Result: unchanged

| measure | recorded baseline | at `47a8fe929` | verdict |
|---|---|---|---|
| HRX suite | 8 of 8 pass | **8 pass, 0 skip, 0 fail** | unchanged |
| `test-hrx-buffer` cases | `5 passed, 0 failed, 5 total` | **`5 passed, 0 failed`** | unchanged |
| its case names | `backend_buffer`, `host_buffer`, `host_transfer`, `host_staging`, `host_weight_cache` | **the same five** (`sort -u`) | unchanged |
| `MUL_MAT_ID` mxfp4 nmse | 1.34e-05 … 1.61e-05 | **min=1.34e-05 max=1.61e-05** | unchanged, exact |
| kernels selected | `ggml_mul_mat_id_decode_f32_wave64`, `ggml_mul_mat_id_f32_f32_wmma` | **the same two** | unchanged |

Verbatim opener from the op-test at the new pin:

```
mxfp4 k= 2880 tokens= 1  loom_libs:ggml_mul_mat_id_decode_f32_wave64      nmse=1.57e-05  OK
mxfp4 k= 2880 tokens= 7  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.34e-05  OK
mxfp4 k= 1152 tokens= 7  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.61e-05  OK
```

The `nmse` figures matter more than the pass/fail: they are a numeric fingerprint of the reduce and
id-resolution path, so an exact match means the op genuinely did not drift, not merely that it avoided failing.

## Re-runnable

```
ssh strixhalo
cd ~/wt/315logwait-build
export LD_LIBRARY_PATH=$PWD/bin
export IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
for t in test-hrx-buffer test-hrx-mul-mat-id-k32 test-hrx-mxfp4 test-hrx-moe-split \
         test-hrx-attention-sink test-hrx-decode-stride test-hrx-hadamard test-hrx-fa-masked-v; do ./bin/$t; done
./bin/test-hrx-mul-mat-id-k32            # the nmse lines
```

## Bounds

* This half is **unchanged**, which is the criterion's claim, and it is a claim about the controls only. It does
  not speak to the fault.
* `test-hrx-ops` and `test-hrx-loom-jit` remain absent from the build, consistently with the recorded note that
  they do not compile at this revision.
* The dense-model half is run separately and reported in its own file.

---

## Criterion 4 closed: engine CI green, and both controls unchanged

Recorded at `ae9da10f9`, when `gh pr checks 414` had settled with **7 of 7 checks passing**:

```
CodeQL             pass
build              pass
pr_agent           pass
Analyze (actions)  pass
Analyze (c-cpp)    pass
Analyze (python)   pass
Analyze (rust)     pass
```

CI did not pass on arrival. It passed only after the two defects in
`315-what-the-merge-and-the-gates-found.md` were fixed — the half-move (registry vs gitlink) and the 46 files
missing the full copyright notice. No gate was loosened to obtain this: the pin check still refuses
half-moves and the copyright check still requires the full notice.

| criterion-4 half | claim | where measured |
|---|---|---|
| engine CI | **green**, 7 of 7 | PR #414 (`gh pr checks 414`) |
| `MUL_MAT_ID` op-test + HRX suite | **unchanged** at `47a8fe929` | `315-task3-op-test-unchanged-at-moved-pin.md` |
| dense-model control | **clean**, unchanged at `47a8fe929` | `315-task3-dense-control-clean-at-moved-pin.md` |

The dense half was **not run** when the record last reported this criterion (`315-crit4-control-result.md`
said the objective "names it alongside the op-test" and that it "should be named explicitly before it is
claimed either way"); it is named, run and measured here, at the moved pin rather than the old one.

### A note on how the green below was checked — a fourth instance of the same class

The first watcher written to confirm this green reported:

```
===== CI on PR #414 — final revision =====
no checks reported on the '1bit/315-preregistration' branch |
--- counts ---
      1
GREEN: no failing check
```

**That is a vacuous pass and it was not used.** The loop broke out of its wait as soon as
`gh pr checks 414` printed no `pending` line — and the string `no checks reported on the … branch` contains no
`pending`, so a branch whose checks had not started yet satisfied "nothing is pending" and the watcher declared
green before a single check existed. The API confirmed CI was in fact in progress at that moment:

```
pr_agent: in_progress -        build: in_progress -
Analyze (python): in_progress -         Analyze (c-cpp): in_progress -
Analyze (actions): in_progress -        Analyze (rust): in_progress -
```

It is the **fourth instance this turn** of the one class — the op-test suite's eight silent `skipped` exits
worth 8/8 PASS, the dense-control script's `"0\n0"` worth a false FAULT, and a response counter read as a
verdict. Here the count of pending checks was read as a verdict on CI. The corrected watcher requires that at
least six checks **exist** and that none are pending before it will say anything, and reports `INCONCLUSIVE`
rather than green otherwise.

The pattern is worth stating once, plainly, because it has now cost this investigation four readings in a
single turn: **a count read as a verdict, without first checking that the count can be non-zero for the reason
the verdict assumes.** In all four cases the corrupted verdict pointed the same way — at a pass that was not
earned, or a fault that did not happen.
