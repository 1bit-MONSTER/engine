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
# HRX hardware gate

`ci.yml` runs on a GitHub runner with no GPU, so it cannot see an HRX kernel or pin that computes
the wrong thing. Eight HRX correctness bugs were found by hand within ten days: #308, #309, #310, #313, #318, #319, #369
and #370. The hardware gate checks every HRX pin bump, and every PR that asks for it, on the Strix Halo box before merge.

## When it runs

`.github/workflows/hw-gate.yml`, job `hw_gate`, on the self-hosted runner `strixhalo-engine`:

- **A labelled PR.** Add the label `hw-gate` to a PR from this repository. Adding the label starts
  the job, and every later push runs it again.
- **Every PR that bump-hrx opens** (branch `bump-hrx/*`).
- **By hand.** Actions -> hw-gate -> Run workflow, with a ref (a branch, tag, SHA or `pull/N/head`).
  The run posts the commit status `hw-gate` on the commit it checked, so a PR shows the result.

Both runners this page names - `strixhalo-engine` here and `strixhalo-bench` for the bench jobs - are installed and
supervised as one recipe: a restart policy, an OOM guard and a dedicated user, so a build that dies cannot take the box's
other jobs with it. That install, and the 2026-11-08 outage that the missing restart policy turned into four hours, are in
[ci-runners.md](ci-runners.md).

A pin bump raised **by hand** must therefore do one of the two automated things: name its branch `bump-hrx/<sha>`
(the `bump-hrx` workflow's own convention) or add the `hw-gate` label. Neither is a formality - without one the job
does not run and the PR simply reads `hw_gate: skipping`, which looks like a pass in a list of ticks (engine#672: a
hand-raised llama.cpp bump on a `1bit/bump-hrx-<sha>` branch went to merge with no gate status until the label was
added; the gate then ran normally).

PRs from forks never run there: the runner is the owner's box.

## The HRX build check

The gate runs only when asked, and `ci.yml` builds the engine without `ONEBIT_HRX`. So until engine#1077 no job
compiled `ggml/src/ggml-hrx` at PR time, and every pin bump merged that week read `hw_gate: skipping`.
`.github/workflows/hrx-build.yml`, job `hrx_build`, closes that gap without the hardware: it configures the engine with
`-DONEBIT_HRX=ON` and builds `onebit`. That builds `llama_hrx` (the pinned llama.cpp with `GGML_HRX=ON`, and the pinned
hrx-system) through `cmake/hrx.cmake`, then links `onebit` with its HRX wiring.

- **When.** On every same-repo PR that changes an input of that build: the `third_party/llama.cpp` or
  `third_party/hrx-system` gitlink, `CMakeLists.txt`, `cmake/hrx.cmake`, `config/hrx-toolchain.config`,
  `config/hrx-system-hold.config`, `app/serve.cpp` (the HRX wiring) or the workflow itself. It also runs by hand
  (Actions -> hrx-build -> Run workflow). It needs no label.
- **Configure only, for a test registration.** A PR whose only change among those inputs is a ctest registration
  added to `CMakeLists.txt` or to a `cmake/tests-*.cmake` family file it includes (`add_test(...)`,
  `set_tests_properties(...)`, `set_property(TEST ...)`, with comments and blank lines) still configures with `-DONEBIT_HRX=ON` but skips the compile. Such a line creates no target and sets no
  flag. It is still evaluated at configure time, including inside `if(ONEBIT_HRX)` blocks that `ci.yml` never
  configures, which is why the configure stays. Files outside the inputs above (the new test's own script, docs) do not
  count: on their own they do not run this job. `tools/ci/hrx_build_scope.py` decides, and it builds whenever it is
  unsure: a line that is not a registration, a registration inside `foreach`/`function`/`macro`, a pure removal, a
  `CMakeLists.txt` entry with no changed lines (a mode change or rename), or a crash. Every changed line in **every**
  registration file is read, so a target change in a family file next to an `add_test` in `CMakeLists.txt` builds.
- **Where.** On the box's bench runner (`strixhalo-bench`, `[self-hosted, strixhalo, bench]`), with TheRock at
  `/opt/rocm-therock`, under `~/lb/thermal-run.sh`. That is not the gate's pr-agent runner, which a cold build would hold
  while every PR-Agent review queued behind it. It takes no GPU token and no NPU lock. Fork PRs never run it. It costs
  no GitHub Actions minutes. The token is read-only, and the build tree persists in `~/.cache/hrx-build/bench` on the
  box, so a later run rebuilds only what changed.
- **Memory.** The box is shared, and on Strix Halo GTT is host RAM. Before it compiles, the job waits for 24 GB of
  `MemAvailable` (`scripts/mem-wait.sh`, polled once a minute for up to 15 minutes). If memory doesn't free up, the job
  fails with the reading, and a re-run is the fix. The compile itself runs under `scripts/mem-guard.sh --term 12`,
  which stops the whole `thermal-run` scope if `MemAvailable` falls below 12 GB. On 2026-11-08 both self-hosted
  runners were OOM-killed, the second time 76 s after a restart handed this job a compile (engine#1504).
- **What it is not.** It is a compile and link check of the public plumbing only. It does not build the private HIP
  add-on (the gate builds that) and does not dispatch anything. Green means the pinned pair builds into the engine. It
  does not replace the `hw-gate` label on a pin bump.

> **From 20:01 to 21:52 on 2026-11-09 this job failed on every branch that ran it, and a red badge was not a signal** (engine #1733).
> Three errors sat in `third_party/llama.cpp`'s own HRX backend — `ggml-hrx.cpp:517,610,854`: a `void` return tested as
> `bool`, and two declarations that differ only in return type — on every branch measured (five of them), and #1681's pin bump
> failed the same way at 9m15s while its `build` was green. **The window closed when the pin moved**, and the pins are what to
> check first if it returns: every failing run in it pinned `llama.cpp` **`8f124f906`**, every passing run after it pinned
> **`60ae4c935`** (the fork fix is llama.cpp#260 → `4e570337`, taken by engine#1736):
>
> ```
> 20:01–21:52  ten failures   ->  llama.cpp 8f124f906      (void at :383, bool at :517: the TU cannot compile)
> 21:41, 21:42  success       ->  llama.cpp 60ae4c935       (first green: the pin that carries the fix)
> 22:04 onward  all success   ->  llama.cpp 60ae4c935       (the branches that had failed come back green)
> 00:21         success       ->  llama.cpp 831629fe9
> 02:23         failure       ->  exit 75 from "Wait for memory (or refuse)", NOT a compile failure
> ```
>
> So the red **was** a property of the pinned tree, and it was not mysterious: `8f124f906` declares `device_free_total` `void` at
> `:383` (engine #252) and `bool` at `:517` (engine #240), which gcc rejects as a redeclaration before it ever reaches `:610`. A
> red `hrx_build` names its own cause in the log, and the pin is one command away (`git ls-tree <sha> third_party/llama.cpp`).
> Read a red badge as meaningful again.

## What it checks

`scripts/hw-gate.sh` builds this checkout's HRX pair (`third_party/llama.cpp` and
`third_party/hrx-system`) with the configuration of `cmake/hrx.cmake`, plus `test-backend-ops`
and `llama-perplexity`, with the private HIP kernels compiled in as the release builds them: the
private add-on at the commit the release config pins, fetched by
the private-fetch helper with the runner box's `gh` login and passed as
`GGML_HRX_HIP_ADDON_DIR`. A llama.cpp pin and an add-on pin that do not compile together fail
the build step (2026-10-20: an add-on pin used a kernel-side type the pinned
llama.cpp did not have yet), and so does a build whose `libggml-hrx.so` lacks the add-on. A box
that cannot read the add-on stops with exit 2; `HW_GATE_PRIVATE=0` gates the public plumbing
only. On a failed build the job log shows the compiler's error lines without their source
excerpts, since the log is public and the add-on's sources are not. It then runs six checks:

| check | what | catches |
|---|---|---|
| ops | one `test-backend-ops -b HRX0 -j 4` run over MUL_MAT, MUL_MAT_ID, FLASH_ATTN_EXT, GATED_DELTA_NET, SSM_CONV, GET_ROWS, ROPE, RMS_NORM, SOFT_MAX | a kernel that disagrees with the CPU (#310, #313, #370, the IQ ports) |
| fa-decode-q8 | the add-on's own long-context decode-attention check against a double-precision reference, quantised K with both cache value types and two head groupings, six runs in one hold | a mixed cache-type bug `test-backend-ops` cannot see: it has no `FLASH_ATTN_EXT` case with K and V of different types |
| kat | `tools/kat.py check` on five models (below), against `tests/golden/hw-gate/kat-<model>.json` | wrong answers, empty replies, NaN logprobs, nondeterminism, cached-prefix and multi-ubatch bugs, drift from the release pin (#308, #309, #315, #318, #319, #369) |
| kld | `llama-perplexity --kl-divergence` on wiki.test.raw, 512 tokens x 8 chunks, against base logits stored at the release pin | small numeric drift that greedy answers do not show |
| needle | `tests/hrx_long_prompt.py`: the 320-record vault-code needle (7.1K tokens on Qwen3, 7.4K on ZAYA1) on Qwen3-0.6B and ZAYA1-8B | #319, #369 |
| placement | every graph the server scheduled for each model (load, warmup, the kat prompts, the needle) is all on HRX, read from ggml-hrx's placement report | a pin that leaves a node on the CPU, which `1bit serve` now refuses to load by default ([serve.md](serve.md#off-hrx-placement-strict-by-default)) |

The models (`MODEL_TABLE` in the script, files under `~/models`):

| name | file | covers | kat repeats | needle |
|---|---|---|---|---|
| qwen3-0.6b | Qwen3-0.6B-Q4_K_M.gguf | small dense K-quant | 3 | yes |
| zaya1-8b | zaya1-8b-Q4_K_M.gguf | ZAYA1 (16 experts, 1 active per token), K-quant | 2 | yes |
| qwen3.8-27b | qwen38/UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf | gated delta net + attention hybrid, Unsloth dynamic mix | 2 | no |
| qwen3-coder-30b-a3b | Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf | qwen3moe: 128 experts, 8 active per token (MUL_MAT_ID with top-8 routing) | 2 | no |
| qwen3-4b-iq4xs | Qwen3-4B-IQ4_XS.gguf | an IQ quant (IQ4_XS matmul and GET_ROWS) | 2 | no |
| qwen3-0.6b-kq8 | Qwen3-0.6B-Q4_K_M.gguf, served with `-fa on -ctk q8_0` | a q8_0 K cache with an f16 V cache | 2 | yes |
| qwen3-0.6b-kvq8 | Qwen3-0.6B-Q4_K_M.gguf, served with `-fa on -ctk q8_0 -ctv q8_0` | q8_0 K and V caches | 2 | yes |

The last two rows give `llama-server` extra arguments (the fifth column of `MODEL_TABLE`). Their kat
needle (about 4.7K tokens) and the 7.1K-token needle decode past 1,024 KV cells, where the add-on's
split decode kernel (`fa_decode_q8_d128`) takes over from its prompt kernel; the short kat cases
decode through the prompt kernel with one query. Both bugs of gpu-kernels #11 showed only on that
path: a launch parameter registered on the wrong kernel (every long decode failed with "compiled
ABI does not match manifest", HTTP 500) and an f16 V read that covered dims 0..31 only. The rows
run kat, the needle and placement but not kld, since a KLD base is stored per model file and
recorded with the default f16 cache.

kat repeats every case and fails on any of:
- a missing expected answer;
- an empty reply or a NaN logprob;
- two different texts for the same request;
- text that leaves the golden before token 8;
- a logprob more than 0.5 from the golden.

KLD thresholds live per model in `tests/golden/hw-gate/manifest.json` (`limits`). The defaults are:
- mean KLD at most 0.01;
- KLD p99 at most 0.2;
- the same top token at least 95% of the time.

A rerun at the recording pin gives 0 on every one.

The placement check reuses engine #390's report. The kat server runs with `GGML_HRX_REPORT=2`
(one line per graph) and `-lv 4` (llama-server drops the library's INFO lines below it), and with
`GGML_HRX_STRICT=0`, so a node off HRX is counted rather than aborting the server and losing the
kat. `tools/hw_gate_manifest.py placement` then reads the server log the way serve does:

```
placement_observer: placement: 512 nodes, 0 off HRX (none), 1 graph splits
```

Any graph with a node off HRX fails the check, with the worst graph's counts and op types
(`20 of 354 nodes off HRX (MUL_MAT x 20), 5 graph splits; 1 of 41 graphs had nodes off HRX`). A
log with no report at all fails too: the check saw nothing. With `--phases placement` alone the
server answers one short chat request instead of the kat, so a prompt graph and decode graphs are
scheduled. The judge's verdicts are in `tests/hw_gate_test.py` (ctest `hw_gate_judges`).

One `test-backend-ops` case has a history of intermittency (KNOWN-ISSUES.md, engine#370, **closed as not
reproducing**: `MUL_MAT(type_a=f32,type_b=f32,m=1,n=1,k=2048)`, about 1 run in 40 when it was seen). When that is the only
failing case, the case runs alone three more times and the check passes if all three are clean;
the summary says so.

## Box etiquette

The gate shares the box with other work. It takes the GPU nine times: once for all the ops, once
for the fa-decode-q8 runs, and once per `MODEL_TABLE` row (the server for kat and the needle, then
llama-perplexity for the KLD, in one hold).
Before the first of them it takes the fleet's GPU token, `~/GPU-TOKEN`: the file long GPU jobs
(Hyperloom, agents' benchmark holds) write their name into while they own the GPU. The gate waits
while another job holds it, up to `HW_GATE_TOKEN_WAIT` seconds (default 3600), logging the holder;
then it writes `hw-gate <run> pid <pid> <time>` into it and removes it on exit, only if the content
is still its own. A wait that times out fails the run with exit 2 and names the holder. Two gate
runs on 2026-10-20 (#481) died with "server failed (rc 137)" while Hyperloom held the token and
loaded models of its own. Hyperloom's own token_take waits while a `scripts/hw-gate.sh` process
exists, so the two take turns. `HW_GATE_TOKEN=none` skips the token for a caller that already
holds it.

Each of those steps:
- **No NPU lock.** A GPU step takes no `~/npu.lock` at all (engine#682): the NPU and the GPU run at the same
  time on this box by measurement, not by assumption — engine#673 measured a GPU-busy NPU stream keeping
  ~87 % of its bandwidth (7.40 GB/s against 8.2–8.5 idle; 10.5–10.9 ms → 12.0 ms per launch; 160/160
  byte-exact, zero resets). The lock is what the **NPU phase** takes, below, because an NPU dispatch must
  not overlap another NPU dispatch.
- **Heat.** It waits for Tctl to drop to 75 C, then runs under `~/lb/thermal-run.sh`, which freezes it at 78 C and kills it at 93 C.
- **Memory.** It runs under `scripts/mem-guard.sh` (kills the step below 12 GB MemAvailable). Before a step starts, `tools/mem_budget.py` estimates it. A step that would take GTT to 95 GiB or more (the engine#315 fault zone), or past MemAvailable, is refused. The gate then waits for memory to free up and retries every minute, up to ten times.
- **The PR-Agent model.** coder-llm (Vulkan, :8090) loses its device when long HRX dispatches starve it. The gate runs on `strixhalo-engine`, the runner that also runs the pr-agent jobs, and a runner takes one job at a time, so no review is generating during the gate. Before each step the gate also waits for coder-llm's slots to be idle. Afterwards it reports any new `ring ... timeout` in the kernel log.

The build runs under thermal-run too, and resumes after a thermal kill. Build trees and ccache
persist under `~/.cache/hw-gate`, so a PR that does not move the pins rebuilds in seconds.

## Goldens: where they live

- **kat goldens are in git:** `tests/golden/hw-gate/kat-*.json`. They are the text and per-token logprobs of every case, a few KB per model.
- **KLD base logits are not in git.** They are 0.4-1.6 GB per model (`llama-perplexity` stores
  the whole vocabulary for every scored token), past GitHub's file limit, and only the gate box
  reads them. They live on that box under `~/.cache/hw-gate/kld/<model>.kld`.
  `tests/golden/hw-gate/manifest.json` pins each one: its sha256, the model file fingerprint
  (size + sha256 of the first 64 MiB), the text's sha256, and the pins it was recorded at. The gate
  refuses a base, model or text that does not match. A private repository would add a token and a
  multi-GB download to every run for no gain while the gate has one box.

Record them again only at a release pin, after the gate passes there:

```sh
scripts/hw-gate.sh --phases build,kat,kld --record   # writes kat-*.json, the .kld files and manifest.json
```

and commit the changed `tests/golden/hw-gate/` files in the same PR as the pin they belong to.

## Running it by hand

```sh
scripts/hw-gate.sh                                     # everything, on this checkout
scripts/hw-gate.sh --phases ops --models qwen3-0.6b    # a subset
scripts/hw-gate.sh --bin /path/to/llama.cpp/build/bin  # check binaries built elsewhere (no build)
```

The settings are environment variables at the top of the script: `HW_GATE_STORE`, `HW_GATE_MODELS`,
`HW_GATE_LOCK`, `HW_GATE_OPS`, `HW_GATE_MEM_FLOOR` and the others. Logs and `summary.md` go to
`$HW_GATE_OUT` (default `~/.cache/hw-gate/runs/<time>`). Exit code: 0 pass, 1 a check failed, 2
the box could not run the gate (lock timeout, a model or base file missing, memory refused).

## Measured

Strix Halo, 2026-10-12, at the release pin (main d2a56a4, llama.cpp 0e255d77, hrx-system
98d05d94), on a box shared with about ten other GPU jobs:

| step | holds the GPU lock |
|---|---|
| build, incremental (no pin change) | 6 s, no lock |
| build, cold (llama.cpp + hrx-system, 4 cores under thermal-run) | about 20 min, no lock |
| ops: 611/611 cases pass | 269 s |
| Qwen3-0.6B: kat (3 repeats) + needle | 67 s |
| ZAYA1-8B: kat (2 repeats) | 172 s |
| Qwen3.8-27B: kat (2 repeats) | 213 s |
| KLD: 0.6B / 8B / 27B, 4,096 tokens each | 6 / 24 / 29 s |

That is about 13 minutes of GPU time in all.

Strix Halo, 2026-10-20, performance power mode, main 55929b5 (llama.cpp 00f2a68d, hrx-system
82804197) plus this coverage change, a quiet box (3 s waiting in all):

| step | holds the GPU lock |
|---|---|
| ops: 611/611 cases pass | 20 s |
| Qwen3-0.6B: kat + needle + KLD | 20 s |
| ZAYA1-8B: kat + needle + KLD (the needle: 7,421-token prompt at 1,207 tok/s, about 8 s) | 70 s |
| Qwen3.8-27B: kat + KLD | 184 s |
| Qwen3-Coder-30B-A3B (MoE): kat + KLD | 38 s |
| Qwen3-4B IQ4_XS: kat + KLD | 31 s |

363 s of GPU time in all; the MoE model, the IQ model and the ZAYA1 needle add about 77 s of it.
Waiting for `~/npu.lock` used to cost more than the gate
itself: flock is not first-come-first-served, and with about ten jobs queueing, each acquisition
waited 10 to 60 minutes, twice cutting a candidate's gate short while an NPU serve held the lock.
Since engine#682 a GPU step takes no NPU lock, so that wait is gone; the summary keeps reporting
waiting (heat, memory) and the step's own time.

What the goldens record at the release pin:
- Qwen3-0.6B misses one short needle (`varied-720w`: "25"). The check holds that case to the golden's text instead of the answer.
- On `varied-1620w`, the cached path answers differently from a fresh request at a near-tie: top-2 margin 0.19, a WARN.
- ZAYA1-8B and Qwen3.8-27B answer every case.

The MoE and IQ rows (`qwen3-coder-30b-a3b`, `qwen3-4b-iq4xs`) were recorded on 2026-10-20 at
llama.cpp 00f2a68d / hrx-system 82804197 (`manifest.json` `recorded_at`). Both answer every kat case.
Their HRX base logits against a CPU run of the same file (`--device none`) give:

| model | mean KLD | KLD p99 | same top token |
|---|---|---|---|
| Qwen3-Coder-30B-A3B Q4_K_M | 0.0054 | 0.043 | 96.7% |
| Qwen3-4B IQ4_XS | 0.0054 | 0.057 | 96.4% |
| Qwen3-0.6B Q4_K_M (the release-pin base, for comparison) | 0.0080 | 0.039 | 95.8% |

That is the usual GPU-vs-CPU rounding gap of a 4-bit file, not a wrong kernel.

## Not covered yet

- **Lock fairness.** `~/npu.lock` is a plain flock, so a job that re-queues quickly can starve another
  NPU job. GPU steps are no longer behind it (engine#682); the NPU phase and any NPU serve still are.
- **Not exercised:** `1bit serve` itself (the gate talks to llama-server with serve's HRX flags),
  `--parallel`, `--mtp` / `--dflash`, vision, and contexts past 8K.
- **One copy of the KLD bases** on the box. A spare copy of the release-pin set is on ryzen
  `~/backups/hw-gate-kld-0e255d77`; the MoE and IQ bases are in `~/backups/hw-gate-kld-00f2a68d`.
- **An intermittent bug** that fires about once in 40 runs (#370's class) can pass a single gate
  run. kat repeats every case and the ops step re-runs a known flake, but only repeated runs
  catch such a bug reliably.
